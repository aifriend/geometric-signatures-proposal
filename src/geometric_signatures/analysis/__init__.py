"""Analysis pipeline: method registry and execution.

Provides a registry of analysis methods and a ``run_analysis`` function
that dispatches to registered methods on preprocessed data.

Usage::

    from geometric_signatures.analysis import run_analysis
    results = run_analysis(population_data, methods=["persistent_homology", "cka"])
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from .base import AnalysisMethod, AnalysisResult
from .geometry_method import GeometryMethod
from .preprocess import preprocess_for_analysis
from .results import load_results, save_results
from .similarity_method import CKAMethod, RSAMethod, linear_cka_numpy

if TYPE_CHECKING:
    from ..population import NeuralPopulationData

# Lazy imports for optional dependencies
_ANALYSIS_REGISTRY: dict[str, type[Any]] = {
    "cka": CKAMethod,
    "rsa": RSAMethod,
    "population_geometry": GeometryMethod,
}


def _get_marble_class() -> type[Any]:
    from .marble_method import MARBLEMethod

    return MARBLEMethod


def _get_topology_class() -> type[Any]:
    from .topology_method import TopologyMethod

    return TopologyMethod


# Full registry with lazy entries for optional deps
ANALYSIS_REGISTRY: dict[str, type[Any]] = {
    **_ANALYSIS_REGISTRY,
}


def get_analysis_method(name: str) -> Any:
    """Get an analysis method class by name.

    Args:
        name: Method name (e.g., "persistent_homology", "cka").

    Returns:
        Instantiated analysis method.

    Raises:
        ValueError: If the method name is not recognized.
        ImportError: If required dependencies are not installed.
    """
    if name in _ANALYSIS_REGISTRY:
        return _ANALYSIS_REGISTRY[name]()

    # Lazy-loaded methods with optional dependencies
    if name == "marble":
        cls = _get_marble_class()
        return cls()
    if name == "persistent_homology":
        cls = _get_topology_class()
        return cls()

    available = list(_ANALYSIS_REGISTRY.keys()) + ["marble", "persistent_homology"]
    raise ValueError(f"Unknown analysis method: {name}. Available: {available}")


def run_analysis(
    data: Any,
    methods: Sequence[str],
    preprocess: bool = True,
) -> dict[str, AnalysisResult]:
    """Run multiple analysis methods on neural population data.

    Args:
        data: NeuralPopulationData instance.
        methods: List of method names to run.
        preprocess: Whether to apply method-specific preprocessing (default True).

    Returns:
        Mapping from method name to AnalysisResult.
    """
    results: dict[str, AnalysisResult] = {}

    import logging

    logger = logging.getLogger(__name__)

    for method_name in methods:
        try:
            method = get_analysis_method(method_name)

            # Apply method-specific preprocessing
            if preprocess:
                processed = preprocess_for_analysis(data, method=method_name)
            else:
                processed = data

            result = method.compute(processed)
            results[method_name] = result
        except Exception as e:
            logger.warning(
                "Analysis method '%s' failed: %s. Skipping.", method_name, e
            )
            continue

    return results


__all__ = [
    "ANALYSIS_REGISTRY",
    "AnalysisMethod",
    "AnalysisResult",
    "CKAMethod",
    "GeometryMethod",
    "RSAMethod",
    "get_analysis_method",
    "linear_cka_numpy",
    "load_results",
    "preprocess_for_analysis",
    "run_analysis",
    "save_results",
]
