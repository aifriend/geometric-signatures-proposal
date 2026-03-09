"""Multi-seed result aggregation.

Combines analysis results from multiple seeds into summary statistics
with bootstrap confidence intervals. This is the bridge between
individual-seed analysis (Step 6) and variant comparison.

Typical flow:
1. Train N seeds of variant "complete" and "ablate_attractor"
2. Run analysis on each seed → N AnalysisResults per variant
3. ``aggregate_across_seeds()`` → AggregatedResult per variant
4. ``compare_variants()`` → PermutationTestResult per metric
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..analysis.base import AnalysisResult
from .bootstrap import BootstrapCI, bootstrap_confidence_interval
from .permutation import PermutationTestResult, permutation_test

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatedResult:
    """Aggregated analysis across multiple seeds.

    Attributes:
        variant: Variant name (e.g. "complete", "ablate_attractor").
        method: Analysis method name (e.g. "population_geometry").
        scalar_means: Mean of each scalar metric across seeds.
        scalar_sems: Standard error of the mean for each scalar.
        scalar_cis: Bootstrap 95% CI for each scalar metric.
        n_seeds: Number of seeds aggregated.
        seed_values: Raw per-seed values for each scalar metric.
    """

    variant: str
    method: str
    scalar_means: dict[str, float]
    scalar_sems: dict[str, float]
    scalar_cis: dict[str, BootstrapCI]
    n_seeds: int
    seed_values: dict[str, np.ndarray]


def aggregate_across_seeds(
    results: Sequence[AnalysisResult],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    rng: np.random.Generator | None = None,
) -> AggregatedResult:
    """Aggregate analysis results from multiple seeds.

    Takes a sequence of AnalysisResult objects (one per seed, same method
    and variant) and computes mean ± SEM with bootstrap CIs for every
    scalar metric.

    Args:
        results: Analysis results from different seeds (same method).
        confidence: CI confidence level (default 0.95).
        n_bootstrap: Bootstrap resamples for CIs (default 10000).
        rng: Numpy random generator for reproducibility.

    Returns:
        AggregatedResult with means, SEMs, and CIs.

    Raises:
        ValueError: If results is empty or methods don't match.
    """
    if len(results) == 0:
        raise ValueError("Need at least one result to aggregate.")

    if rng is None:
        rng = np.random.default_rng()

    # Validate all results are from the same method
    methods = {r.method for r in results}
    if len(methods) > 1:
        raise ValueError(
            f"All results must be from the same method, got: {methods}"
        )

    method = results[0].method
    variant = results[0].variant

    # Collect scalar keys from all results (union)
    all_keys: set[str] = set()
    for r in results:
        all_keys.update(r.scalars.keys())

    # Build per-key arrays
    seed_values: dict[str, np.ndarray] = {}
    scalar_means: dict[str, float] = {}
    scalar_sems: dict[str, float] = {}
    scalar_cis: dict[str, BootstrapCI] = {}

    n_seeds = len(results)

    for key in sorted(all_keys):
        values = np.array([
            r.scalars[key] for r in results if key in r.scalars
        ])

        if len(values) == 0:
            continue

        seed_values[key] = values
        scalar_means[key] = float(values.mean())

        if len(values) > 1:
            scalar_sems[key] = float(values.std(ddof=1) / np.sqrt(len(values)))
            scalar_cis[key] = bootstrap_confidence_interval(
                values,
                confidence=confidence,
                n_bootstrap=n_bootstrap,
                rng=rng,
            )
        else:
            scalar_sems[key] = 0.0
            scalar_cis[key] = BootstrapCI(
                point_estimate=float(values[0]),
                ci_lower=float(values[0]),
                ci_upper=float(values[0]),
                confidence_level=confidence,
                n_bootstrap=0,
            )

    return AggregatedResult(
        variant=variant,
        method=method,
        scalar_means=scalar_means,
        scalar_sems=scalar_sems,
        scalar_cis=scalar_cis,
        n_seeds=n_seeds,
        seed_values=seed_values,
    )


def compare_variants(
    results_a: Sequence[AnalysisResult],
    results_b: Sequence[AnalysisResult],
    metric_key: str,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
) -> PermutationTestResult:
    """Statistical comparison of a specific metric between two variants.

    Extracts the given scalar metric from each seed's result and runs
    a two-sample permutation test.

    Args:
        results_a: Results from variant A (e.g., "complete", 10 seeds).
        results_b: Results from variant B (e.g., "ablate_attractor", 10 seeds).
        metric_key: Name of the scalar metric to compare.
        n_permutations: Number of permutations (default 1000).
        rng: Numpy random generator for reproducibility.

    Returns:
        PermutationTestResult for the metric comparison.

    Raises:
        ValueError: If the metric is missing from any result.
    """
    if rng is None:
        rng = np.random.default_rng()

    values_a = _extract_metric(results_a, metric_key)
    values_b = _extract_metric(results_b, metric_key)

    if len(values_a) < 2 or len(values_b) < 2:
        logger.warning(
            "Small sample sizes (%d, %d) for metric '%s' — "
            "permutation test may have low power.",
            len(values_a), len(values_b), metric_key,
        )

    return permutation_test(
        values_a,
        values_b,
        n_permutations=n_permutations,
        rng=rng,
    )


def _extract_metric(
    results: Sequence[AnalysisResult],
    metric_key: str,
) -> np.ndarray:
    """Extract a scalar metric from a sequence of results."""
    values = []
    for r in results:
        if metric_key not in r.scalars:
            raise ValueError(
                f"Metric '{metric_key}' not found in result "
                f"(method={r.method}, seed={r.seed}). "
                f"Available: {list(r.scalars.keys())}"
            )
        values.append(r.scalars[metric_key])
    return np.array(values)
