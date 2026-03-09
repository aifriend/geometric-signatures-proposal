"""Population geometry analysis method.

Computes geometric statistics of neural population activity:
- **Participation ratio**: Effective dimensionality of the data
  (how many PCA dimensions explain the variance).
- **Effective dimensionality**: Number of PCA components needed
  for a given variance threshold.
- **Condition separability**: How well-separated different task
  conditions are in neural space (cluster quality).
- **Neural trajectory statistics**: Speed, curvature, volume.

These are model-agnostic geometry measures that complement
manifold-specific methods (MARBLE) and topological methods (PH).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import AnalysisResult

if TYPE_CHECKING:
    from ..population import NeuralPopulationData

logger = logging.getLogger(__name__)


def participation_ratio(activity: np.ndarray) -> float:
    """Compute participation ratio (effective dimensionality).

    PR = (sum λ_i)^2 / sum λ_i^2

    Where λ_i are eigenvalues of the covariance matrix.
    PR = 1 for a single-dimension subspace, PR = N for isotropic.

    Args:
        activity: Neural activity, shape (n_samples, n_features).

    Returns:
        Participation ratio.
    """
    # Center
    centered = activity - activity.mean(axis=0)
    # Covariance
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    if len(eigenvalues) == 0:
        return 0.0

    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues**2).sum()

    if sum_eig_sq < 1e-20:
        return 0.0

    return float(sum_eig**2 / sum_eig_sq)


def effective_dimensionality(
    activity: np.ndarray,
    variance_threshold: float = 0.95,
) -> int:
    """Number of PCA components for a given variance threshold.

    Args:
        activity: Neural activity, shape (n_samples, n_features).
        variance_threshold: Fraction of variance to explain (default 0.95).

    Returns:
        Number of dimensions needed.
    """
    centered = activity - activity.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 0]

    if len(eigenvalues) == 0:
        return 0

    total_var = eigenvalues.sum()
    cumulative = np.cumsum(eigenvalues) / total_var
    n_dims = int(np.searchsorted(cumulative, variance_threshold) + 1)
    return min(n_dims, len(eigenvalues))


def condition_separability(
    activity: np.ndarray,
    labels: tuple[str, ...],
) -> float:
    """Compute condition separability using between/within class scatter ratio.

    A simple measure of how well-separated different conditions are:
    separability = trace(S_between) / trace(S_within)

    Args:
        activity: Neural activity, shape (n_samples, n_features).
        labels: Condition label per sample.

    Returns:
        Separability ratio (higher = better separated).
    """
    unique_labels = sorted(set(labels))
    if len(unique_labels) < 2:
        return 0.0

    overall_mean = activity.mean(axis=0)
    n_features = activity.shape[1]

    s_between = np.zeros((n_features, n_features))
    s_within = np.zeros((n_features, n_features))

    label_array = np.array(labels)
    for label in unique_labels:
        mask = label_array == label
        group = activity[mask]
        n_group = group.shape[0]
        if n_group == 0:
            continue

        group_mean = group.mean(axis=0)
        diff = (group_mean - overall_mean).reshape(-1, 1)
        s_between += n_group * (diff @ diff.T)

        centered = group - group_mean
        s_within += centered.T @ centered

    trace_between = np.trace(s_between)
    trace_within = np.trace(s_within)

    if trace_within < 1e-10:
        return float("inf") if trace_between > 1e-10 else 0.0

    return float(trace_between / trace_within)


def trajectory_speed(
    activity: np.ndarray,
) -> np.ndarray:
    """Compute neural trajectory speed (magnitude of temporal derivative).

    Args:
        activity: Neural activity, shape (n_trials, n_timepoints, n_units).

    Returns:
        Speed per trial per timepoint, shape (n_trials, n_timepoints - 1).
    """
    diffs = np.diff(activity, axis=1)
    return np.asarray(np.linalg.norm(diffs, axis=2))


class GeometryMethod:
    """Population geometry analysis.

    Computes model-agnostic geometric statistics:
    participation ratio, effective dimensionality, condition
    separability, and trajectory dynamics.

    Attributes:
        name: Method identifier ("population_geometry").
        variance_threshold: Variance threshold for effective dimensionality.
    """

    name: str = "population_geometry"

    def __init__(self, variance_threshold: float = 0.95) -> None:
        self.variance_threshold = variance_threshold

    def compute(self, data: Any) -> AnalysisResult:
        """Compute geometric statistics on neural population data.

        Args:
            data: NeuralPopulationData instance.

        Returns:
            AnalysisResult with geometric scalar metrics and arrays.
        """
        from ..population import NeuralPopulationData

        assert isinstance(data, NeuralPopulationData)

        activity = data.activity  # (n_trials, n_timepoints, n_units)
        n_trials, n_time, n_units = activity.shape

        # Flatten for whole-dataset statistics
        flat = activity.reshape(-1, n_units)  # (n_trials * n_time, n_units)

        # Participation ratio
        pr = participation_ratio(flat)

        # Effective dimensionality
        eff_dim = effective_dimensionality(flat, self.variance_threshold)

        # Condition separability (using trial labels)
        # Repeat labels for each timepoint
        expanded_labels = tuple(
            label for label in data.trial_labels for _ in range(n_time)
        )
        separability = condition_separability(flat, expanded_labels)

        # Trajectory speed statistics
        speeds = trajectory_speed(activity)
        mean_speed = float(speeds.mean())
        std_speed = float(speeds.std())
        max_speed = float(speeds.max())

        # Per-trial trajectory length
        traj_lengths = speeds.sum(axis=1)  # (n_trials,)
        mean_traj_length = float(traj_lengths.mean())

        # Variance explained by top PCs
        centered = flat - flat.mean(axis=0)
        _, S, _ = np.linalg.svd(centered, full_matrices=False)
        var_explained = S**2 / (S**2).sum()
        top_3_var = float(var_explained[:3].sum()) if len(var_explained) >= 3 else 0.0

        return AnalysisResult(
            method=self.name,
            config_hash="",
            seed=0,
            variant="",
            arrays={
                "variance_explained": var_explained,
                "trajectory_speeds": speeds,
                "trajectory_lengths": traj_lengths,
            },
            scalars={
                "participation_ratio": pr,
                "effective_dimensionality": float(eff_dim),
                "condition_separability": separability,
                "mean_trajectory_speed": mean_speed,
                "std_trajectory_speed": std_speed,
                "max_trajectory_speed": max_speed,
                "mean_trajectory_length": mean_traj_length,
                "top_3_variance_explained": top_3_var,
            },
        )
