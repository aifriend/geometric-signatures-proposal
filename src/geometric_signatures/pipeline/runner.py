"""Pipeline orchestrator with disk persistence and partial execution.

Wires stages together into a full experiment pipeline:
``generate_variants → train(×N seeds) → preprocess → analyze →
aggregate → compare``

Each stage reads/writes intermediate results to disk, enabling
restarts after failures.

Usage::

    from geometric_signatures.pipeline import run_pipeline, PipelineOptions

    result = run_pipeline(
        config=config,
        output_dir=Path("runs/experiment_01"),
        options=PipelineOptions(skip_training=True),  # reuse checkpoints
    )
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..analysis.base import AnalysisResult
from ..analysis.results import load_results, save_results
from ..config import ExperimentConfig
from ..population import NeuralPopulationData
from ..statistics.aggregation import AggregatedResult
from ..statistics.permutation import PermutationTestResult

from .stages import (
    stage_aggregate,
    stage_analyze,
    stage_compare,
    stage_generate_variants,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    """Options for controlling pipeline execution.

    Attributes:
        skip_training: If True, load population data from disk instead
            of training models. Requires prior training output.
        skip_analysis: If True, load analysis results from disk.
        skip_statistics: If True, skip aggregation and comparison.
        variants: Subset of variant names to process. None = all.
        analysis_methods: Analysis methods to run. None = use config.
        device: Training device — "auto", "cpu", "cuda", "cuda:N",
            or "mps" (Apple Silicon). Resolved in trainer.
    """

    skip_training: bool = False
    skip_analysis: bool = False
    skip_statistics: bool = False
    variants: tuple[str, ...] | None = None
    analysis_methods: tuple[str, ...] | None = None
    device: str = "auto"


@dataclass(frozen=True)
class PipelineResult:
    """Result of a full pipeline run.

    Attributes:
        analysis_results: variant → method → list[AnalysisResult] (per seed).
        aggregated_results: variant → method → AggregatedResult.
        comparisons: variant → "method.metric" → PermutationTestResult.
        n_variants: Number of variants processed.
        n_seeds: Number of seeds per variant.
    """

    analysis_results: dict[str, dict[str, list[AnalysisResult]]]
    aggregated_results: dict[str, dict[str, AggregatedResult]]
    comparisons: dict[str, dict[str, PermutationTestResult]]
    n_variants: int
    n_seeds: int


def run_pipeline(
    config: ExperimentConfig,
    output_dir: Path,
    options: PipelineOptions | None = None,
) -> PipelineResult:
    """Run the full experiment pipeline.

    Stages: generate_variants → train → analyze → aggregate → compare.
    Supports partial execution via ``PipelineOptions``.

    Args:
        config: Experiment configuration.
        output_dir: Base directory for all outputs.
        options: Execution options (defaults to running everything).

    Returns:
        PipelineResult with all analysis, aggregation, and comparison data.
    """
    if options is None:
        options = PipelineOptions()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Stage 1: Generate variants ---
    all_variants = stage_generate_variants(config)

    # Filter to requested variants
    if options.variants is not None:
        all_variants = {
            k: v for k, v in all_variants.items()
            if k in options.variants
        }
        logger.info("Filtered to %d variants: %s",
                     len(all_variants), list(all_variants.keys()))

    # Determine seeds
    if config.seeds is not None:
        seeds = list(config.seeds.seed_list())
    else:
        seeds = [config.experiment.seed]

    # Determine analysis methods
    methods: tuple[str, ...]
    if options.analysis_methods is not None:
        methods = options.analysis_methods
    elif config.analysis is not None:
        methods = config.analysis.methods
    else:
        methods = ("population_geometry", "cka", "rsa")

    # --- Stage 2: Train (or load) ---
    population_data: dict[str, list[NeuralPopulationData]] = {}

    if options.skip_training:
        logger.info("Skipping training — loading population data from disk.")
        for variant_name in all_variants:
            population_data[variant_name] = _load_population_data(
                output_dir, variant_name, seeds
            )
    else:
        population_data = _run_training(
            config, all_variants, seeds, output_dir, options.device
        )

    # --- Stage 3+4: Analyze (or load) ---
    all_analysis: dict[str, dict[str, list[AnalysisResult]]] = {}

    if options.skip_analysis:
        logger.info("Skipping analysis — loading results from disk.")
        for variant_name in all_variants:
            all_analysis[variant_name] = _load_analysis_results(
                output_dir, variant_name, seeds, methods
            )
    else:
        for variant_name, pop_data_list in population_data.items():
            variant_results: dict[str, list[AnalysisResult]] = {
                m: [] for m in methods
            }

            for seed_idx, pop_data in enumerate(pop_data_list):
                seed = seeds[seed_idx] if seed_idx < len(seeds) else seed_idx

                logger.info(
                    "Analyzing variant=%s seed=%d (%d/%d)",
                    variant_name, seed, seed_idx + 1, len(pop_data_list),
                )

                results = stage_analyze(pop_data, methods)

                # Tag results with variant/seed metadata
                for method_name, result in results.items():
                    tagged = AnalysisResult(
                        method=result.method,
                        config_hash=result.config_hash,
                        seed=seed,
                        variant=variant_name,
                        arrays=result.arrays,
                        scalars=result.scalars,
                    )
                    variant_results[method_name].append(tagged)

                # Save per-seed results
                results_dir = (
                    output_dir / variant_name / f"seed_{seed}" / "analysis"
                )
                save_results(results, results_dir)

            all_analysis[variant_name] = variant_results

    # --- Stage 5+6: Aggregate + Compare ---
    aggregated: dict[str, dict[str, AggregatedResult]] = {}
    comparisons: dict[str, dict[str, PermutationTestResult]] = {}

    if not options.skip_statistics:
        rng = np.random.default_rng(config.experiment.seed)

        # Aggregate per variant
        for variant_name, method_results in all_analysis.items():
            aggregated[variant_name] = stage_aggregate(
                method_results, rng=rng
            )

        # Compare variants against reference
        n_perms = 1000
        if config.analysis is not None:
            n_perms = config.analysis.n_permutations

        comparisons = stage_compare(
            all_analysis,
            reference_variant="complete",
            n_permutations=n_perms,
            rng=rng,
        )

        # Save comparison summary
        _save_comparison_summary(comparisons, output_dir / "comparisons.json")

    return PipelineResult(
        analysis_results=all_analysis,
        aggregated_results=aggregated,
        comparisons=comparisons,
        n_variants=len(all_variants),
        n_seeds=len(seeds),
    )


def _run_training(
    config: ExperimentConfig,
    variants: dict[str, Any],
    seeds: list[int],
    output_dir: Path,
    device: str,
    progress_callback: Callable[..., None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, list[NeuralPopulationData]]:
    """Run training for all variants and seeds."""
    try:
        from ..training.trainer import train_single_seed
    except ImportError:
        raise ImportError(
            "Training requires torch. Install with: uv sync --extra train"
        )

    population_data: dict[str, list[NeuralPopulationData]] = {}

    for variant_name, motifs in variants.items():
        if cancel_event is not None and cancel_event.is_set():
            break

        pop_list: list[NeuralPopulationData] = []

        for seed in seeds:
            if cancel_event is not None and cancel_event.is_set():
                break

            # Skip already-completed runs (manifest + population data exist)
            seed_dir = output_dir / variant_name / f"seed_{seed}"
            manifest_path = seed_dir / "manifest.json"
            pop_path = seed_dir / "population.npz"
            if manifest_path.exists() and pop_path.exists():
                logger.info(
                    "Skipping variant=%s seed=%d (already completed)",
                    variant_name, seed,
                )
                pop = _load_single_population(pop_path)
                pop_list.append(pop)
                continue

            logger.info(
                "Training variant=%s seed=%d", variant_name, seed
            )

            # Create variant config with these motifs
            variant_config = _config_with_motifs(config, motifs)

            try:
                result = train_single_seed(
                    variant_config, seed, output_dir, device,
                    progress_callback=progress_callback,
                    cancel_event=cancel_event,
                )
                pop = result.population_data
                pop_list.append(pop)

                # Persist population data for later analysis stages
                _save_population_data(pop, result.checkpoint_path, seed)
            except Exception as e:
                logger.error(
                    "Training failed: variant=%s seed=%d: %s",
                    variant_name, seed, e,
                )
                continue

        population_data[variant_name] = pop_list

    return population_data


def _config_with_motifs(
    config: ExperimentConfig, motifs: Any
) -> ExperimentConfig:
    """Create a new config with different motif switches."""
    from dataclasses import replace
    return replace(config, motifs=motifs)


def _save_population_data(
    pop: NeuralPopulationData,
    checkpoint_path: str,
    seed: int,
) -> None:
    """Save population data alongside the checkpoint."""
    seed_dir = Path(checkpoint_path).parent
    npz_path = seed_dir / "population.npz"
    np.savez_compressed(
        str(npz_path),
        activity=pop.activity,
        trial_labels=np.array(pop.trial_labels),
        time_axis=pop.time_axis,
    )
    logger.info("Saved population data to %s", npz_path)


def _load_single_population(npz_path: Path) -> NeuralPopulationData:
    """Load a single population data file from a completed run."""
    data = np.load(str(npz_path), allow_pickle=True)
    activity = data["activity"]
    trial_labels = tuple(str(s) for s in data["trial_labels"])
    time_axis = data["time_axis"]
    n_units = activity.shape[2]
    unit_labels = tuple(f"unit_{i}" for i in range(n_units))
    return NeuralPopulationData(
        activity=activity,
        trial_labels=trial_labels,
        time_axis=time_axis,
        unit_labels=unit_labels,
        source="rnn",
        metadata={},
    )


def _resolve_variant_dir(
    output_dir: Path,
    variant_name: str,
) -> Path:
    """Find the actual variant directory on disk.

    The trainer shortens variant names (e.g. 'ablate_attractor' instead
    of 'ablate_attractor_dynamics').  If the full-name directory doesn't
    exist, search for a prefix match among existing directories.

    Mapping from full motif names to trainer short names:
        ablate_normalization_gain_modulation → ablate_normalization
        ablate_attractor_dynamics            → ablate_attractor
        ablate_selective_gating              → ablate_gating
        ablate_expansion_recoding            → ablate_expansion
    """
    direct = output_dir / variant_name
    if direct.exists():
        return direct

    # Build short-name mapping (mirrors trainer._variant_name logic)
    _SHORT_NAMES: dict[str, str] = {
        "ablate_normalization_gain_modulation": "ablate_normalization",
        "ablate_attractor_dynamics": "ablate_attractor",
        "ablate_selective_gating": "ablate_gating",
        "ablate_expansion_recoding": "ablate_expansion",
    }
    short = _SHORT_NAMES.get(variant_name)
    if short is not None:
        short_path = output_dir / short
        if short_path.exists():
            return short_path

    # Fallback: prefix match in either direction
    for d in sorted(output_dir.iterdir()):
        if not d.is_dir():
            continue
        if variant_name.startswith(d.name) or d.name.startswith(variant_name):
            return d

    return direct  # fallback to original path (will trigger warnings)


def _load_population_data(
    output_dir: Path,
    variant_name: str,
    seeds: list[int],
) -> list[NeuralPopulationData]:
    """Load population data from saved numpy files."""
    variant_dir = _resolve_variant_dir(output_dir, variant_name)
    pop_list: list[NeuralPopulationData] = []
    for seed in seeds:
        npz_path = variant_dir / f"seed_{seed}" / "population.npz"
        if npz_path.exists():
            data = np.load(str(npz_path))
            activity = data["activity"]
            n_trials = activity.shape[0]
            n_time = activity.shape[1]
            n_units = activity.shape[2]
            pop = NeuralPopulationData(
                activity=activity,
                trial_labels=tuple(
                    data["trial_labels"]) if "trial_labels" in data else tuple(
                    f"trial_{i}" for i in range(n_trials)
                ),
                time_axis=data["time_axis"] if "time_axis" in data else np.arange(
                    n_time, dtype=np.float64
                ),
                unit_labels=tuple(f"u{i}" for i in range(n_units)),
                source="rnn",
                metadata={},
            )
            pop_list.append(pop)
        else:
            logger.warning("Population data not found: %s", npz_path)
    return pop_list


def _load_analysis_results(
    output_dir: Path,
    variant_name: str,
    seeds: list[int],
    methods: Sequence[str],
) -> dict[str, list[AnalysisResult]]:
    """Load analysis results from disk."""
    variant_dir = _resolve_variant_dir(output_dir, variant_name)
    method_results: dict[str, list[AnalysisResult]] = {m: [] for m in methods}

    for seed in seeds:
        results_dir = variant_dir / f"seed_{seed}" / "analysis"
        if not results_dir.exists():
            logger.warning("Analysis results not found: %s", results_dir)
            continue

        loaded = load_results(results_dir, methods=list(methods))
        for method_name, result in loaded.items():
            method_results[method_name].append(result)

    return method_results


def _save_comparison_summary(
    comparisons: dict[str, dict[str, PermutationTestResult]],
    path: Path,
) -> None:
    """Save comparison results as a JSON summary."""
    path.parent.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {}
    for variant_name, tests in comparisons.items():
        summary[variant_name] = {}
        for key, result in tests.items():
            summary[variant_name][key] = {
                "observed_statistic": result.observed_statistic,
                "p_value": result.p_value,
                "n_permutations": result.n_permutations,
            }

    path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    logger.info("Comparison summary saved to %s", path)
