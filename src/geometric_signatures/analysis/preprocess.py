"""Preprocessing for analysis methods.

Different analysis methods need different preprocessing:
- MARBLE: No PCA (works on high-dim data directly)
- Persistent homology: PCA to ~50 dims (computational efficiency)
- RSA: Trial-averaged activity per condition
- CKA: Raw or PCA-reduced
- Population geometry: PCA to ~20 dims

This module provides a single ``preprocess_for_analysis`` function with
method-specific defaults, plus building blocks for custom preprocessing.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..population import NeuralPopulationData

# Method-specific preprocessing defaults
PREPROCESS_DEFAULTS: dict[str, dict[str, Any]] = {
    "marble": {"n_components": None, "trial_average": False, "normalize": "none"},
    "persistent_homology": {
        "n_components": 50,
        "trial_average": False,
        "normalize": "zscore",
    },
    "rsa": {"n_components": None, "trial_average": True, "normalize": "zscore"},
    "cka": {"n_components": None, "trial_average": False, "normalize": "zscore"},
    "population_geometry": {
        "n_components": 20,
        "trial_average": False,
        "normalize": "zscore",
    },
}


def pca_reduce(
    activity: np.ndarray,
    n_components: int,
) -> np.ndarray:
    """Reduce dimensionality via PCA.

    Args:
        activity: Neural activity, shape (n_trials, n_timepoints, n_units).
        n_components: Number of PCA components to keep.

    Returns:
        Reduced activity, shape (n_trials, n_timepoints, n_components).
    """
    n_trials, n_time, n_units = activity.shape

    # Reshape to (n_trials * n_time, n_units) for PCA
    flat = activity.reshape(-1, n_units)

    # Center
    mean = flat.mean(axis=0)
    centered = flat - mean

    # SVD-based PCA (no sklearn dependency)
    n_components = min(n_components, n_units, flat.shape[0])
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    reduced = U[:, :n_components] * S[:n_components]

    return np.asarray(reduced.reshape(n_trials, n_time, n_components))


def zscore_normalize(activity: np.ndarray) -> np.ndarray:
    """Z-score normalize each unit across trials and time.

    Args:
        activity: Neural activity, shape (n_trials, n_timepoints, n_units).

    Returns:
        Z-scored activity, same shape.
    """
    n_trials, n_time, n_units = activity.shape
    flat = activity.reshape(-1, n_units)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    # Avoid division by zero
    std = np.where(std < 1e-10, 1.0, std)
    normalized = (flat - mean) / std
    return np.asarray(normalized.reshape(n_trials, n_time, n_units))


def trial_average_by_condition(
    data: NeuralPopulationData,
) -> np.ndarray:
    """Average activity within each unique trial label (task/condition).

    Args:
        data: NeuralPopulationData with trial_labels.

    Returns:
        Averaged activity, shape (n_conditions, n_timepoints, n_units).
    """
    unique_labels = sorted(set(data.trial_labels))
    averages = []
    for label in unique_labels:
        mask = np.array([lbl == label for lbl in data.trial_labels])
        avg = data.activity[mask].mean(axis=0)
        averages.append(avg)
    return np.stack(averages, axis=0)


def preprocess_for_analysis(
    data: NeuralPopulationData,
    method: str | None = None,
    n_components: int | None = None,
    normalize: str | None = None,
    trial_average: bool | None = None,
) -> NeuralPopulationData:
    """Preprocess neural population data for a specific analysis method.

    Uses method-specific defaults from ``PREPROCESS_DEFAULTS`` when
    explicit arguments are not provided.

    Args:
        data: Input NeuralPopulationData.
        method: Analysis method name (e.g., "persistent_homology").
            Used to look up defaults. If None, uses explicit args only.
        n_components: PCA components. None = no PCA.
        normalize: Normalization method ("zscore" or "none").
        trial_average: Whether to average trials by condition.

    Returns:
        Preprocessed NeuralPopulationData.
    """
    # Get method-specific defaults
    defaults: dict[str, Any] = {}
    if method is not None and method in PREPROCESS_DEFAULTS:
        defaults = PREPROCESS_DEFAULTS[method]

    # Resolve parameters (explicit args override defaults)
    n_comp = n_components if n_components is not None else defaults.get("n_components")
    norm = normalize if normalize is not None else defaults.get("normalize", "none")
    avg = trial_average if trial_average is not None else defaults.get(
        "trial_average", False
    )

    activity = data.activity.copy()

    # Step 1: Normalize
    if norm == "zscore":
        activity = zscore_normalize(activity)

    # Step 2: PCA
    if n_comp is not None and n_comp < activity.shape[2]:
        activity = pca_reduce(activity, n_comp)

    # Step 3: Trial averaging
    if avg:
        averaged = trial_average_by_condition(
            NeuralPopulationData(
                activity=activity,
                trial_labels=data.trial_labels,
                time_axis=data.time_axis,
                unit_labels=tuple(f"pc_{i}" for i in range(activity.shape[2]))
                if n_comp is not None
                else data.unit_labels,
                source=data.source,
                metadata=dict(data.metadata),
                trial_metadata=data.trial_metadata,
            )
        )
        unique_labels = tuple(sorted(set(data.trial_labels)))
        unit_labels = (
            tuple(f"pc_{i}" for i in range(averaged.shape[2]))
            if n_comp is not None
            else data.unit_labels
        )
        return NeuralPopulationData(
            activity=averaged,
            trial_labels=unique_labels,
            time_axis=data.time_axis,
            unit_labels=unit_labels,
            source=data.source,
            metadata=dict(data.metadata),
            trial_metadata=None,  # metadata no longer meaningful after averaging
        )

    # Build updated unit labels if PCA was applied
    unit_labels = (
        tuple(f"pc_{i}" for i in range(activity.shape[2]))
        if n_comp is not None and n_comp < len(data.unit_labels)
        else data.unit_labels
    )

    return NeuralPopulationData(
        activity=activity,
        trial_labels=data.trial_labels,
        time_axis=data.time_axis,
        unit_labels=unit_labels,
        source=data.source,
        metadata=dict(data.metadata),
        trial_metadata=data.trial_metadata,
    )
