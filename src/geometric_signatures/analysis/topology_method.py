"""Persistent homology analysis method.

Computes topological features of neural population manifolds using
persistent homology. The primary backend is Ripser.py (lighter and
easier to install than giotto-tda).

Typical workflow:
1. Preprocess: PCA to ~50 dims (high-dim PH is expensive).
2. Compute persistence diagrams up to dimension ``max_dim``.
3. Extract topological features: Betti numbers, total persistence,
   persistence entropy, longest bar per dimension.

Requires: ``ripser`` (in ``analysis`` dependency group).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import AnalysisResult

if TYPE_CHECKING:
    from ..population import NeuralPopulationData

logger = logging.getLogger(__name__)

try:
    from ripser import ripser

    RIPSER_AVAILABLE = True
except ImportError:
    RIPSER_AVAILABLE = False


class TopologyMethod:
    """Persistent homology of neural population manifolds.

    Attributes:
        name: Method identifier ("persistent_homology").
        max_dim: Maximum homology dimension to compute (default: 2).
        max_points: Maximum number of points to subsample (for speed).
    """

    name: str = "persistent_homology"

    def __init__(
        self,
        max_dim: int = 2,
        max_points: int = 500,
    ) -> None:
        self.max_dim = max_dim
        self.max_points = max_points

    def compute(self, data: Any) -> AnalysisResult:
        """Compute persistent homology on neural population data.

        Expects preprocessed data (PCA-reduced to manageable dimensionality).

        Args:
            data: NeuralPopulationData instance (ideally PCA-preprocessed).

        Returns:
            AnalysisResult with persistence diagrams and topological features.

        Raises:
            ImportError: If ripser is not installed.
        """
        if not RIPSER_AVAILABLE:
            raise ImportError(
                "ripser not installed. Run: uv sync --extra analysis"
            )

        from ..population import NeuralPopulationData

        assert isinstance(data, NeuralPopulationData)

        activity = data.activity  # (n_trials, n_timepoints, n_units)
        n_trials, n_time, n_units = activity.shape

        # Flatten to point cloud: (n_trials * n_time, n_units)
        point_cloud = activity.reshape(-1, n_units)

        # Subsample if too many points
        if point_cloud.shape[0] > self.max_points:
            rng = np.random.default_rng(42)
            indices = rng.choice(
                point_cloud.shape[0], self.max_points, replace=False
            )
            point_cloud = point_cloud[indices]

        logger.info(
            "Computing PH on %d points in %d dims (max_dim=%d)",
            point_cloud.shape[0],
            point_cloud.shape[1],
            self.max_dim,
        )

        # Run Ripser
        result = ripser(point_cloud, maxdim=self.max_dim)
        diagrams = result["dgms"]  # list of (birth, death) arrays per dim

        # Extract features per dimension
        arrays: dict[str, np.ndarray] = {}
        scalars: dict[str, float] = {}

        for dim in range(self.max_dim + 1):
            dgm = diagrams[dim]
            key_prefix = f"H{dim}"

            # Store diagram
            arrays[f"{key_prefix}_diagram"] = dgm

            # Filter out infinite bars for statistics
            finite = dgm[np.isfinite(dgm[:, 1])] if len(dgm) > 0 else dgm

            if len(finite) > 0:
                persistence = finite[:, 1] - finite[:, 0]

                # Betti number (count of features)
                scalars[f"{key_prefix}_betti"] = float(len(finite))

                # Total persistence
                scalars[f"{key_prefix}_total_persistence"] = float(
                    persistence.sum()
                )

                # Longest bar
                scalars[f"{key_prefix}_longest_bar"] = float(persistence.max())

                # Persistence entropy
                normed = persistence / persistence.sum()
                entropy = -float((normed * np.log(normed + 1e-10)).sum())
                scalars[f"{key_prefix}_entropy"] = entropy

                # Mean persistence
                scalars[f"{key_prefix}_mean_persistence"] = float(
                    persistence.mean()
                )
            else:
                scalars[f"{key_prefix}_betti"] = 0.0
                scalars[f"{key_prefix}_total_persistence"] = 0.0
                scalars[f"{key_prefix}_longest_bar"] = 0.0
                scalars[f"{key_prefix}_entropy"] = 0.0
                scalars[f"{key_prefix}_mean_persistence"] = 0.0

        return AnalysisResult(
            method=self.name,
            config_hash="",
            seed=0,
            variant="",
            arrays=arrays,
            scalars=scalars,
        )
