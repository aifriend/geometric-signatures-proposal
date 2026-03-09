"""Independent, composable pipeline stages.

Each stage is a pure function that takes and returns data objects.
The ``runner.py`` orchestrator handles disk I/O between stages,
enabling restarts and partial execution.

Stages can also be called directly in notebooks or scripts.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np

from ..analysis.base import AnalysisResult
from ..analysis.preprocess import preprocess_for_analysis
from ..config import ExperimentConfig
from ..motifs import MotifSwitches, build_single_ablation_variants
from ..population import NeuralPopulationData
from ..statistics.aggregation import (
    AggregatedResult,
    aggregate_across_seeds,
    compare_variants,
)
from ..statistics.permutation import PermutationTestResult

logger = logging.getLogger(__name__)


def stage_generate_variants(
    config: ExperimentConfig,
) -> dict[str, MotifSwitches]:
    """Generate ablation variant configurations.

    Produces the complete model plus one ablation per motif.

    Args:
        config: Experiment configuration (motifs define the complete model).

    Returns:
        Mapping of variant name → MotifSwitches. Always includes
        "complete" plus one "ablate_<motif>" per motif.
    """
    variants = build_single_ablation_variants(config.motifs)
    logger.info("Generated %d variants: %s", len(variants), list(variants.keys()))
    return variants


def stage_preprocess(
    data: NeuralPopulationData,
    method: str | None = None,
    n_components: int | None = None,
    normalize: str = "zscore",
    trial_average: bool | None = None,
) -> NeuralPopulationData:
    """Preprocess neural data for a specific analysis method.

    Thin wrapper around ``preprocess_for_analysis`` with method-specific
    defaults.

    Args:
        data: Raw neural population data.
        method: Analysis method name for default selection (e.g., "rsa").
        n_components: PCA components (None = skip or use method default).
        normalize: Normalization method ("zscore" or "none").
        trial_average: Whether to average trials by condition.

    Returns:
        Preprocessed NeuralPopulationData.
    """
    return preprocess_for_analysis(
        data,
        method=method,
        n_components=n_components,
        normalize=normalize,
        trial_average=trial_average,
    )


def stage_analyze(
    data: NeuralPopulationData,
    methods: Sequence[str],
    preprocess: bool = True,
) -> dict[str, AnalysisResult]:
    """Run selected analysis methods on neural data.

    Args:
        data: Neural population data (preprocessed or raw).
        methods: Analysis method names to run.
        preprocess: Whether to apply method-specific preprocessing.

    Returns:
        Mapping of method name → AnalysisResult.
    """
    from ..analysis import run_analysis

    results = run_analysis(data, methods, preprocess=preprocess)
    logger.info(
        "Analysis complete: %s (scalars per method: %s)",
        list(results.keys()),
        {k: len(v.scalars) for k, v in results.items()},
    )
    return results


def stage_aggregate(
    results_by_seed: dict[str, list[AnalysisResult]],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> dict[str, AggregatedResult]:
    """Aggregate analysis results across seeds.

    Args:
        results_by_seed: Mapping of method name → list of AnalysisResult
            (one per seed, same method).
        confidence: Bootstrap CI confidence level.
        n_bootstrap: Bootstrap resamples.
        rng: Random generator for reproducibility.

    Returns:
        Mapping of method name → AggregatedResult.
    """
    if rng is None:
        rng = np.random.default_rng()

    aggregated: dict[str, AggregatedResult] = {}
    for method_name, seed_results in results_by_seed.items():
        if len(seed_results) == 0:
            logger.warning("No results for method %s, skipping.", method_name)
            continue

        agg = aggregate_across_seeds(
            seed_results,
            confidence=confidence,
            n_bootstrap=n_bootstrap,
            rng=rng,
        )
        aggregated[method_name] = agg
        logger.info(
            "Aggregated %s: %d seeds, %d metrics",
            method_name, agg.n_seeds, len(agg.scalar_means),
        )

    return aggregated


def stage_compare(
    variant_results: dict[str, dict[str, list[AnalysisResult]]],
    reference_variant: str = "complete",
    metric_keys: Sequence[str] | None = None,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, dict[str, PermutationTestResult]]:
    """Statistical comparison of variants against a reference.

    Compares each non-reference variant against the reference variant
    for every specified metric using permutation tests.

    Args:
        variant_results: Mapping of variant name → method name → list of
            AnalysisResult (one per seed).
        reference_variant: Variant to compare against (default "complete").
        metric_keys: Specific metrics to compare. If None, compares all
            scalar metrics present in the reference variant.
        n_permutations: Permutations per test.
        rng: Random generator for reproducibility.

    Returns:
        Mapping of "variant_vs_reference.method.metric" → PermutationTestResult.
    """
    if rng is None:
        rng = np.random.default_rng()

    if reference_variant not in variant_results:
        raise ValueError(
            f"Reference variant '{reference_variant}' not in results. "
            f"Available: {list(variant_results.keys())}"
        )

    ref_methods = variant_results[reference_variant]
    comparisons: dict[str, dict[str, PermutationTestResult]] = {}

    for variant_name, methods in variant_results.items():
        if variant_name == reference_variant:
            continue

        comparisons[variant_name] = {}

        for method_name, seed_results in methods.items():
            if method_name not in ref_methods:
                logger.warning(
                    "Method %s not in reference variant, skipping.", method_name
                )
                continue

            ref_results = ref_methods[method_name]

            # Determine metrics to compare
            keys = metric_keys
            if keys is None:
                # Use all scalar keys from reference
                all_keys: set[str] = set()
                for r in ref_results:
                    all_keys.update(r.scalars.keys())
                keys = sorted(all_keys)

            for metric_key in keys:
                # Check metric exists in both
                ref_has = all(metric_key in r.scalars for r in ref_results)
                var_has = all(metric_key in r.scalars for r in seed_results)
                if not (ref_has and var_has):
                    continue

                comparison_key = f"{method_name}.{metric_key}"
                try:
                    result = compare_variants(
                        ref_results,
                        seed_results,
                        metric_key,
                        n_permutations=n_permutations,
                        rng=rng,
                    )
                    comparisons[variant_name][comparison_key] = result
                except Exception as e:
                    logger.warning(
                        "Comparison failed: %s vs %s on %s.%s: %s",
                        reference_variant, variant_name, method_name,
                        metric_key, e,
                    )

    n_total = sum(len(v) for v in comparisons.values())
    logger.info(
        "Compared %d variants against '%s': %d total tests",
        len(comparisons), reference_variant, n_total,
    )
    return comparisons
