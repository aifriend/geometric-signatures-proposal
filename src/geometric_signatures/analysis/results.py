"""Analysis result persistence utilities.

Provides batch save/load for multiple AnalysisResult objects and
result collection helpers.

The core ``AnalysisResult`` dataclass (with save/load) lives in
``base.py`` to avoid circular imports. This module adds convenience
functions for working with collections of results.
"""

from __future__ import annotations

from pathlib import Path

from .base import AnalysisResult


def save_results(
    results: dict[str, AnalysisResult],
    output_dir: Path,
) -> dict[str, Path]:
    """Save a dict of analysis results to disk.

    Creates a sub-file per method:
        ``<output_dir>/<method_name>``

    Args:
        results: Mapping from method name to AnalysisResult.
        output_dir: Directory to save results.

    Returns:
        Mapping from method name to saved base path (without extension).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for method_name, result in results.items():
        base_path = output_dir / method_name
        result.save(base_path)
        paths[method_name] = base_path

    return paths


def load_results(
    output_dir: Path,
    methods: list[str] | None = None,
) -> dict[str, AnalysisResult]:
    """Load analysis results from disk.

    Args:
        output_dir: Directory containing saved results.
        methods: Optional list of method names to load. If None,
            loads all methods found in the directory.

    Returns:
        Mapping from method name to AnalysisResult.
    """
    output_dir = Path(output_dir)
    results: dict[str, AnalysisResult] = {}

    if methods is not None:
        # Load specific methods
        for method_name in methods:
            base_path = output_dir / method_name
            json_path = Path(str(base_path) + ".json")
            if json_path.exists():
                results[method_name] = AnalysisResult.load(base_path)
    else:
        # Discover all .json files and load corresponding results
        for json_path in output_dir.glob("*.json"):
            method_name = json_path.stem
            base_path = output_dir / method_name
            results[method_name] = AnalysisResult.load(base_path)

    return results
