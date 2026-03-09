"""Allen Brain Observatory Visual Behavior Ophys data loader.

Loads two-photon calcium imaging experiments from the Allen Institute
via the AllenSDK. Requires ``allensdk`` package::

    pip install allensdk

Typical workflow::

    from geometric_signatures.data.allen import load_allen_experiment
    data = load_allen_experiment(experiment_id, cache_dir=Path('data/allen'))
    # data is a NeuralPopulationData ready for analysis
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ..population import NeuralPopulationData

logger = logging.getLogger(__name__)

try:
    from allensdk.brain_observatory.behavior.behavior_project_cache import (
        VisualBehaviorOphysProjectCache,
    )

    ALLENSDK_AVAILABLE = True
except ImportError:
    ALLENSDK_AVAILABLE = False


def _check_allensdk_available() -> None:
    """Raise ImportError if AllenSDK is not installed."""
    if not ALLENSDK_AVAILABLE:
        raise ImportError(
            "Allen data loading requires allensdk. Install with: "
            "pip install allensdk"
        )


def load_allen_experiment(
    experiment_id: int,
    cache_dir: Path,
    stimulus_name: str | None = None,
    normalize: str = "zscore",
    window: tuple[float, float] = (-0.5, 2.0),
) -> NeuralPopulationData:
    """Load an Allen Visual Behavior Ophys experiment.

    Downloads ΔF/F traces, aligns to stimulus presentations, and
    normalizes.

    Args:
        experiment_id: Allen experiment ID (integer).
        cache_dir: Local cache directory for downloaded data.
        stimulus_name: Filter by stimulus type (e.g., "natural_scenes").
            None = use all stimulus presentations.
        normalize: Normalization method ("zscore", "max", "range", "none").
        window: Time window around stimulus onset in seconds.

    Returns:
        NeuralPopulationData with source="allen".

    Raises:
        ImportError: If allensdk is not installed.
    """
    _check_allensdk_available()

    from .neural_preprocessing import normalize_population

    cache = VisualBehaviorOphysProjectCache.from_s3_cache(str(cache_dir))

    logger.info("Loading Allen experiment: %d", experiment_id)

    experiment = cache.get_behavior_ophys_experiment(experiment_id)

    # Get ΔF/F traces — already computed by Allen pipeline
    dff_traces = experiment.dff_traces  # DataFrame: rows=cells, cols=timestamps
    timestamps = experiment.ophys_timestamps

    # Get cell specimen IDs
    cell_ids = list(dff_traces.index)
    n_cells = len(cell_ids)

    # Build continuous data matrix (n_timepoints, n_cells)
    dff_matrix = np.array(dff_traces.values).T  # (n_timepoints, n_cells)

    # Get stimulus presentations
    stim_table = experiment.stimulus_presentations
    if stimulus_name is not None:
        stim_table = stim_table[
            stim_table["stimulus_name"] == stimulus_name
        ]

    if len(stim_table) == 0:
        raise ValueError(
            f"No presentations found for stimulus '{stimulus_name}'. "
            f"Available: {list(stim_table['stimulus_name'].unique())}"
        )

    # Extract onset times
    onset_times = stim_table["start_time"].values

    # Align trials
    dt = float(np.median(np.diff(timestamps)))
    trial_time = np.arange(window[0], window[1], dt)
    n_window = len(trial_time)
    n_trials = len(onset_times)

    aligned = np.zeros((n_trials, n_window, n_cells))
    for i, onset in enumerate(onset_times):
        target_times = onset + trial_time
        indices = np.searchsorted(timestamps, target_times)
        indices = np.clip(indices, 0, len(timestamps) - 1)
        aligned[i] = dff_matrix[indices]

    # Extract trial labels from stimulus type
    if "stimulus_name" in stim_table.columns:
        trial_labels = tuple(stim_table["stimulus_name"].values)
    else:
        trial_labels = tuple(f"stim_{i}" for i in range(n_trials))

    # Normalize
    if normalize != "none":
        aligned = normalize_population(aligned, method=normalize)

    unit_labels = tuple(f"cell_{int(c)}" for c in cell_ids)

    return NeuralPopulationData(
        activity=aligned,
        trial_labels=trial_labels,
        time_axis=trial_time,
        unit_labels=unit_labels,
        source="allen",
        metadata={
            "experiment_id": experiment_id,
            "stimulus_name": stimulus_name or "all",
            "n_cells": n_cells,
            "window": list(window),
        },
    )


def list_allen_experiments(
    area: str | None = None,
    cache_dir: Path | None = None,
) -> list[int]:
    """List available Allen Visual Behavior Ophys experiments.

    Args:
        area: Filter by visual area (e.g., "VISp", "VISl").
        cache_dir: Local cache directory.

    Returns:
        List of experiment IDs.
    """
    _check_allensdk_available()

    cache_path = str(cache_dir) if cache_dir else "allen_cache"
    cache = VisualBehaviorOphysProjectCache.from_s3_cache(cache_path)

    experiments = cache.get_ophys_experiment_table()

    if area is not None:
        experiments = experiments[
            experiments["targeted_structure"] == area
        ]

    return list(experiments.index)
