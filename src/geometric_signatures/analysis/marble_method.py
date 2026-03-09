"""MARBLE manifold analysis method.

Wraps the MARBLE-nn package (Nature Methods 2024) for computing
manifold-aware embeddings of neural population dynamics.

MARBLE uses PyTorch Geometric under the hood:
- Constructs a geometric dataset from neural trajectories
- Trains a graph neural network to learn manifold structure
- Produces low-dimensional embeddings that capture geometry

**Important**: MARBLE works best on raw (high-dimensional) data.
Do NOT apply PCA before MARBLE — it handles dimensionality internally.

Requires optional dependencies: ``MARBLE-nn``, ``torch-geometric``.
Install with: ``uv sync --extra analysis-marble``
"""

from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import AnalysisResult

if TYPE_CHECKING:
    from ..population import NeuralPopulationData

logger = logging.getLogger(__name__)

try:
    import MARBLE  # noqa: N811

    MARBLE_AVAILABLE = True
except ImportError:
    MARBLE_AVAILABLE = False


class MARBLEMethod:
    """MARBLE manifold embedding analysis.

    Constructs neural trajectory graph, trains embedding network,
    and extracts manifold-aware representations.

    Attributes:
        name: Method identifier ("marble").
        order: Derivative order for velocity estimation (default: 1).
        n_neighbors: Number of neighbors for graph construction.
        embedding_dim: Target embedding dimension.
    """

    name: str = "marble"

    def __init__(
        self,
        order: int = 1,
        n_neighbors: int = 30,
        embedding_dim: int = 3,
        epochs: int = 100,
        max_points: int = 5000,
    ) -> None:
        self.order = order
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        self.epochs = epochs
        self.max_points = max_points

    def compute(self, data: Any) -> AnalysisResult:
        """Compute MARBLE embedding on neural population data.

        Args:
            data: NeuralPopulationData instance.

        Returns:
            AnalysisResult with embedding arrays and scalar metrics.

        Raises:
            ImportError: If MARBLE-nn is not installed.
        """
        if not MARBLE_AVAILABLE:
            raise ImportError(
                "MARBLE-nn not installed. "
                "Run: uv sync --extra analysis-marble"
            )

        # Import here to avoid issues if MARBLE is not available
        from ..population import NeuralPopulationData

        assert isinstance(data, NeuralPopulationData)

        activity = data.activity  # (n_trials, n_timepoints, n_units)
        n_trials, n_time, n_units = activity.shape

        logger.info(
            "Running MARBLE on %d trials × %d timepoints × %d units",
            n_trials,
            n_time,
            n_units,
        )

        # Reshape: treat each trial as a separate trajectory
        # MARBLE expects: positions (n_points, n_dims), velocities (n_points, n_dims)
        positions = activity.reshape(-1, n_units)  # (n_trials * n_time, n_units)

        # Compute velocities (finite differences along time axis)
        velocities_list = []
        for trial in range(n_trials):
            trial_data = activity[trial]  # (n_time, n_units)
            # Forward differences, pad last timestep with zeros
            vel = np.diff(trial_data, axis=0)
            vel = np.concatenate([vel, np.zeros((1, n_units))], axis=0)
            velocities_list.append(vel)
        velocities = np.concatenate(velocities_list, axis=0)

        # PCA to reduce dimensionality before graph construction
        # High-dim data (256 units) creates disconnected kNN graphs
        from sklearn.decomposition import PCA

        pca_dim = min(50, n_units)
        if n_units > pca_dim:
            pca = PCA(n_components=pca_dim)
            positions = pca.fit_transform(positions)
            velocities = pca.transform(velocities)
            logger.info(
                "PCA-reduced %d → %d dims for MARBLE graph", n_units, pca_dim
            )

        # Subsample if too many points (MARBLE graph construction is O(n^2))
        n_points = positions.shape[0]
        if n_points > self.max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(n_points, size=self.max_points, replace=False)
            idx.sort()
            positions = positions[idx]
            velocities = velocities[idx]
            logger.info(
                "Subsampled %d → %d points for MARBLE", n_points, self.max_points
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Construct MARBLE dataset
            data_marble = MARBLE.construct_dataset(
                anchor=positions,
                vector=velocities,
                k=self.n_neighbors,
            )

            # Build and train model
            params = {
                "order": self.order,
                "hidden_channels": self.embedding_dim * 2,
                "out_channels": self.embedding_dim,
                "epochs": self.epochs,
            }
            model = MARBLE.net(data_marble, params=params)
            model.fit(data_marble, outdir=tmpdir)

            # Transform to get embeddings
            data_marble = model.transform(data_marble)
            embedding = data_marble.emb.detach().cpu().numpy()

        # Reshape embedding back to (n_trials, n_time, embedding_dim)
        embedding_3d = embedding.reshape(n_trials, n_time, -1)

        # Compute scalar metrics from embedding
        # Inter-trial variance in embedding space
        trial_means = embedding_3d.mean(axis=1)  # (n_trials, emb_dim)
        inter_trial_var = float(np.var(trial_means, axis=0).sum())

        # Mean embedding norm
        mean_norm = float(np.linalg.norm(trial_means, axis=1).mean())

        return AnalysisResult(
            method=self.name,
            config_hash="",  # filled by pipeline
            seed=0,
            variant="",
            arrays={
                "embedding": embedding_3d,
                "trial_means": trial_means,
            },
            scalars={
                "inter_trial_variance": inter_trial_var,
                "mean_embedding_norm": mean_norm,
            },
        )
