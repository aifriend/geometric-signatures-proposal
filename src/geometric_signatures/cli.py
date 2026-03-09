"""Command-line interface for geometric signature experiments.

Provides subcommands for the full research workflow::

    # Train all ablation variants with multi-seed
    uv run geometric-signatures train config/experiment.yaml --output-dir runs/

    # Analyze existing training results
    uv run geometric-signatures analyze runs/baseline/ --methods cka,rsa

    # Compare two systems (e.g., RNN vs. biological)
    uv run geometric-signatures compare runs/rnn/ runs/bio/ --output figures/

    # Query experiment catalog
    uv run geometric-signatures status runs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


def cmd_train(args: argparse.Namespace) -> int:
    """Train constrained RNNs for all ablation variants.

    Loads config, generates ablation variants, trains each with N seeds,
    and registers all runs in the experiment catalog.
    """
    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "Error: PyTorch is required for training. "
            "Install with: uv sync --extra train",
            file=sys.stderr,
        )
        return 1

    from .config import load_experiment_config
    from .logging_config import setup_logging
    from .pipeline.runner import PipelineOptions, run_pipeline

    setup_logging(level=args.log_level, log_dir=args.output_dir / "logs")

    config = load_experiment_config(args.config)
    logger.info("Loaded config: %s", config.experiment.name)

    options = PipelineOptions(
        skip_analysis=True,
        skip_statistics=True,
        variants=tuple(args.variants) if args.variants else None,
        device=args.device,
    )

    result = run_pipeline(config, args.output_dir, options=options)

    print(f"\nTraining complete.")
    print(f"  Variants: {result.n_variants}")
    print(f"  Seeds per variant: {result.n_seeds}")
    print(f"  Output directory: {args.output_dir}")

    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run analysis methods on existing training results.

    Loads population data from a previous training run and applies
    the specified analysis methods (CKA, RSA, population geometry, etc.).
    """
    from .config import load_experiment_config
    from .logging_config import setup_logging
    from .pipeline.runner import PipelineOptions, run_pipeline

    setup_logging(level=args.log_level, log_dir=args.output_dir / "logs")

    config = load_experiment_config(args.config)
    logger.info("Loaded config: %s", config.experiment.name)

    methods = tuple(args.methods.split(",")) if args.methods else None

    options = PipelineOptions(
        skip_training=True,
        skip_statistics=not args.with_stats,
        analysis_methods=methods,
        variants=tuple(args.variants) if args.variants else None,
    )

    result = run_pipeline(config, args.output_dir, options=options)

    print(f"\nAnalysis complete.")
    print(f"  Variants analyzed: {len(result.analysis_results)}")
    for variant, methods_dict in result.analysis_results.items():
        method_names = list(methods_dict.keys()) if methods_dict else []
        print(f"    {variant}: {method_names}")
    print(f"  Output directory: {args.output_dir}")

    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Compare geometric signatures across two systems.

    Takes two directories of analysis results (e.g., RNN and biological)
    and performs statistical comparison of their geometric signatures.
    """
    from .comparison.cross_system import compare_across_systems
    from .logging_config import setup_logging
    from .statistics.aggregation import AggregatedResult

    setup_logging(level=args.log_level)

    # Load aggregated results from each directory
    results_a = _load_aggregated_results(args.dir_a)
    results_b = _load_aggregated_results(args.dir_b)

    if not results_a:
        print(f"Error: No aggregated results found in {args.dir_a}", file=sys.stderr)
        return 1
    if not results_b:
        print(f"Error: No aggregated results found in {args.dir_b}", file=sys.stderr)
        return 1

    methods = tuple(args.methods.split(",")) if args.methods else None

    comparison = compare_across_systems(
        results_a,
        results_b,
        methods=methods,
        n_permutations=args.n_permutations,
        alpha=args.alpha,
        system_a_name=args.dir_a.name,
        system_b_name=args.dir_b.name,
    )

    # Print summary
    print(f"\n=== Cross-System Comparison ===")
    print(f"  System A: {comparison.system_a}")
    print(f"  System B: {comparison.system_b}")
    print(f"  Metrics compared: {len(comparison.metric_similarities)}")
    print(f"\n  Shared signatures (p >= {args.alpha}):")
    for sig in comparison.shared_signatures:
        sim = comparison.metric_similarities[sig]
        p = comparison.significance[sig].p_value
        print(f"    {sig}: similarity={sim:.3f}, p={p:.4f}")

    print(f"\n  Divergent signatures (p < {args.alpha}):")
    for sig in comparison.divergent_signatures:
        sim = comparison.metric_similarities[sig]
        p = comparison.significance[sig].p_value
        d = comparison.effect_sizes[sig]
        print(f"    {sig}: similarity={sim:.3f}, p={p:.4f}, d={d:.3f}")

    # Generate figures if output specified
    if args.output:
        try:
            from .figures.plotting import (
                fig_cross_system_comparison,
                fig_effect_size_forest,
            )
            from .figures.style import apply_style

            args.output.mkdir(parents=True, exist_ok=True)
            apply_style("paper")

            fig_cross_system_comparison(
                comparison,
                output=args.output / "cross_system_comparison.pdf",
            )
            fig_effect_size_forest(
                comparison,
                output=args.output / "effect_sizes.pdf",
            )
            print(f"\n  Figures saved to: {args.output}")
        except ImportError:
            print(
                "\n  Warning: matplotlib not installed, skipping figures. "
                "Install with: uv sync --extra figures",
                file=sys.stderr,
            )

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Query the experiment catalog for run status.

    Lists registered runs, optionally filtered by variant or status.
    """
    from .tracking import ExperimentCatalog

    catalog_path = args.catalog_dir / "experiment_catalog.db"
    if not catalog_path.exists():
        print(f"No experiment catalog found at {catalog_path}")
        print("Run training first or specify correct directory.")
        return 1

    catalog = ExperimentCatalog(catalog_path)
    records = catalog.query(
        variant=args.variant,
        status=args.filter_status,
    )

    if not records:
        print("No matching runs found.")
        return 0

    print(f"\n=== Experiment Catalog ({len(records)} runs) ===\n")

    # Group by variant
    by_variant: dict[str, list[Any]] = {}
    for rec in records:
        by_variant.setdefault(rec.variant_name, []).append(rec)

    for variant, runs in sorted(by_variant.items()):
        completed = sum(1 for r in runs if r.status == "completed")
        failed = sum(1 for r in runs if r.status == "failed")
        running = sum(1 for r in runs if r.status == "running")

        print(f"  {variant}: {len(runs)} run(s)")
        print(f"    ✓ completed: {completed}")
        if failed:
            print(f"    ✗ failed: {failed}")
        if running:
            print(f"    ◌ running: {running}")

        seeds = sorted(r.seed for r in runs)
        print(f"    seeds: {seeds}")
        print()

    return 0


def _load_aggregated_results(
    directory: Path,
) -> dict[str, Any]:
    """Load aggregated results from a directory.

    Looks for JSON/npz files in the standard pipeline output structure.

    Returns:
        Dict mapping method names to AggregatedResult objects.
    """
    from .analysis import AnalysisResult
    from .statistics.aggregation import AggregatedResult, aggregate_across_seeds

    results_by_method: dict[str, list[AnalysisResult]] = {}

    # Look for analysis result files
    for result_file in sorted(directory.rglob("*_result.json")):
        try:
            result = AnalysisResult.load(result_file)
            results_by_method.setdefault(result.method, []).append(result)
        except Exception as e:
            logger.warning("Failed to load %s: %s", result_file, e)
            continue

    # Aggregate each method's results
    aggregated: dict[str, AggregatedResult] = {}
    for method, results in results_by_method.items():
        if results:
            try:
                aggregated[method] = aggregate_across_seeds(results)
            except Exception as e:
                logger.warning("Failed to aggregate %s: %s", method, e)

    return aggregated


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        Configured argument parser with subcommands.
    """
    parser = argparse.ArgumentParser(
        prog="geometric-signatures",
        description=(
            "Geometric Signatures of Computational Motifs — "
            "research toolchain for PhD experiments"
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version="geometric-signatures 0.1.0",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ----- train -----
    train_parser = subparsers.add_parser(
        "train",
        help="Train constrained RNNs for ablation variants",
        description="Train all ablation variants with multi-seed execution.",
    )
    train_parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment YAML config file",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        dest="output_dir",
        help="Output directory for training results (default: runs/)",
    )
    train_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help=(
            "Training device: auto, cpu, cuda, cuda:N, or mps "
            "(default: auto — picks best available)"
        ),
    )
    train_parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variant names to train (default: all)",
    )
    train_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        help="Logging level (default: INFO)",
    )

    # ----- analyze -----
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run analysis methods on training results",
        description="Apply analysis pipeline to existing training output.",
    )
    analyze_parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment YAML config file",
    )
    analyze_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs"),
        dest="output_dir",
        help="Directory with training results (default: runs/)",
    )
    analyze_parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated analysis methods (default: from config)",
    )
    analyze_parser.add_argument(
        "--variants",
        nargs="+",
        default=None,
        help="Subset of variant names to analyze (default: all)",
    )
    analyze_parser.add_argument(
        "--with-stats",
        action="store_true",
        dest="with_stats",
        help="Also run statistical aggregation and comparison",
    )
    analyze_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        help="Logging level (default: INFO)",
    )

    # ----- compare -----
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare geometric signatures across two systems",
        description="Statistical comparison of geometric signatures.",
    )
    compare_parser.add_argument(
        "dir_a",
        type=Path,
        help="Directory with first system's analysis results",
    )
    compare_parser.add_argument(
        "dir_b",
        type=Path,
        help="Directory with second system's analysis results",
    )
    compare_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for comparison figures",
    )
    compare_parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods to compare (default: all common)",
    )
    compare_parser.add_argument(
        "--n-permutations",
        type=int,
        default=1000,
        dest="n_permutations",
        help="Number of permutations for testing (default: 1000)",
    )
    compare_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold (default: 0.05)",
    )
    compare_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        dest="log_level",
        help="Logging level (default: INFO)",
    )

    # ----- status -----
    status_parser = subparsers.add_parser(
        "status",
        help="Query experiment catalog",
        description="List and filter registered experiment runs.",
    )
    status_parser.add_argument(
        "catalog_dir",
        type=Path,
        help="Directory containing experiment_catalog.db",
    )
    status_parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Filter by variant name",
    )
    status_parser.add_argument(
        "--status",
        type=str,
        default=None,
        dest="filter_status",
        help="Filter by run status (completed, failed, running)",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Command-line arguments. Defaults to sys.argv[1:].

    Returns:
        Exit code (0 = success, non-zero = error).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "train": cmd_train,
        "analyze": cmd_analyze,
        "compare": cmd_compare,
        "status": cmd_status,
    }

    handler = commands.get(args.command)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


def cli_entry() -> None:
    """Entry point for ``pyproject.toml`` console script."""
    sys.exit(main())
