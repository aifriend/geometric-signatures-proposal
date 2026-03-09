"""Raw neural signal preprocessing for biological data.

Converts raw electrophysiology (spike times) and imaging (fluorescence)
signals into trial-aligned, binned population activity matrices suitable
for ``NeuralPopulationData``.
"""

from __future__ import annotations

import numpy as np


def bin_spikes(
    spike_times: np.ndarray,
    spike_clusters: np.ndarray,
    bin_size: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin spike times into a firing rate matrix.

    Args:
        spike_times: 1-D array of spike times in seconds.
        spike_clusters: 1-D array of cluster IDs (same length as spike_times).
        bin_size: Temporal bin size in seconds.
        t_start: Start time (default: min spike time).
        t_end: End time (default: max spike time).

    Returns:
        Tuple of:
        - rates: Firing rate matrix, shape (n_bins, n_clusters), in Hz.
        - bin_centers: Time of each bin center, shape (n_bins,).
        - cluster_ids: Sorted unique cluster IDs, shape (n_clusters,).
    """
    spike_times = np.asarray(spike_times)
    spike_clusters = np.asarray(spike_clusters)

    if t_start is None:
        t_start = float(spike_times.min())
    if t_end is None:
        t_end = float(spike_times.max())

    # Create bins
    bin_edges = np.arange(t_start, t_end + bin_size, bin_size)
    n_bins = len(bin_edges) - 1
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Get unique clusters
    cluster_ids = np.sort(np.unique(spike_clusters))
    cluster_to_idx = {int(c): i for i, c in enumerate(cluster_ids)}
    n_clusters = len(cluster_ids)

    # Count spikes per bin per cluster
    rates = np.zeros((n_bins, n_clusters))

    # Digitize spike times into bins
    bin_indices = np.digitize(spike_times, bin_edges) - 1
    valid = (bin_indices >= 0) & (bin_indices < n_bins)

    for spike_idx in np.where(valid)[0]:
        b = bin_indices[spike_idx]
        c = cluster_to_idx.get(int(spike_clusters[spike_idx]))
        if c is not None:
            rates[b, c] += 1.0

    # Convert counts to rates (Hz)
    rates /= bin_size

    return rates, bin_centers, cluster_ids


def compute_delta_f_over_f(
    fluorescence: np.ndarray,
    baseline_percentile: float = 10.0,
    baseline_window: int | None = None,
) -> np.ndarray:
    """Compute ΔF/F from raw fluorescence traces.

    Args:
        fluorescence: Raw fluorescence, shape (..., n_timepoints).
        baseline_percentile: Percentile for baseline estimation (default 10).
        baseline_window: If given, use only the first N timepoints for
            baseline. Otherwise use percentile across all timepoints.

    Returns:
        ΔF/F array with same shape as input.
    """
    fluorescence = np.asarray(fluorescence, dtype=float)

    if baseline_window is not None:
        baseline = np.mean(fluorescence[..., :baseline_window], axis=-1, keepdims=True)
    else:
        baseline = np.percentile(
            fluorescence, baseline_percentile, axis=-1, keepdims=True
        )

    # Avoid division by zero
    baseline = np.where(np.abs(baseline) < 1e-10, 1e-10, baseline)

    return np.asarray((fluorescence - baseline) / baseline)


def align_trials(
    neural_data: np.ndarray,
    time_axis: np.ndarray,
    event_times: np.ndarray,
    window: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract trial-aligned neural data around event times.

    Args:
        neural_data: Continuous data, shape (n_timepoints, n_units).
        time_axis: Timestamps for each row of neural_data, shape (n_timepoints,).
        event_times: Event onset times, shape (n_trials,).
        window: (pre_seconds, post_seconds) — time window relative to event.
            E.g. (-0.5, 1.5) for 0.5s before to 1.5s after each event.

    Returns:
        Tuple of:
        - aligned: Trial-aligned data, shape (n_trials, n_window_bins, n_units).
        - trial_time: Time axis relative to event, shape (n_window_bins,).
    """
    neural_data = np.asarray(neural_data)
    time_axis = np.asarray(time_axis)
    event_times = np.asarray(event_times)

    dt = float(np.median(np.diff(time_axis)))
    pre, post = window

    trial_time = np.arange(pre, post, dt)
    n_window = len(trial_time)
    n_units = neural_data.shape[1]
    n_trials = len(event_times)

    aligned = np.zeros((n_trials, n_window, n_units))

    for i, event_t in enumerate(event_times):
        target_times = event_t + trial_time
        # Find nearest indices
        indices = np.searchsorted(time_axis, target_times)
        indices = np.clip(indices, 0, len(time_axis) - 1)
        aligned[i] = neural_data[indices]

    return aligned, trial_time


def normalize_population(
    activity: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Normalize population activity.

    Args:
        activity: Activity array, shape (..., n_units).
        method: Normalization method — "zscore", "max", or "range".

    Returns:
        Normalized activity with same shape.
    """
    activity = np.asarray(activity, dtype=float)

    if method == "zscore":
        # Z-score per unit (last axis)
        mean = activity.mean(axis=tuple(range(activity.ndim - 1)), keepdims=True)
        std = activity.std(axis=tuple(range(activity.ndim - 1)), keepdims=True)
        std = np.where(std < 1e-10, 1.0, std)
        return np.asarray((activity - mean) / std)

    elif method == "max":
        # Divide by max absolute value per unit
        max_abs = np.max(np.abs(activity), axis=tuple(range(activity.ndim - 1)), keepdims=True)
        max_abs = np.where(max_abs < 1e-10, 1.0, max_abs)
        return np.asarray(activity / max_abs)

    elif method == "range":
        # Scale to [0, 1] per unit
        mins = activity.min(axis=tuple(range(activity.ndim - 1)), keepdims=True)
        maxs = activity.max(axis=tuple(range(activity.ndim - 1)), keepdims=True)
        denom = maxs - mins
        denom = np.where(denom < 1e-10, 1.0, denom)
        return np.asarray((activity - mins) / denom)

    else:
        raise ValueError(f"Unknown normalization method: {method}")
