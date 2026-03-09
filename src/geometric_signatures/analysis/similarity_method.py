"""Representational similarity analysis (RSA) and CKA methods.

Two complementary approaches for comparing neural representations:

- **RSA** (Kriegeskorte et al. 2008): Builds Representational Dissimilarity
  Matrices (RDMs) from condition-averaged activity and compares them.
  Uses ``rsatoolbox`` when available.

- **CKA** (Kornblith et al. 2019): Centered Kernel Alignment for
  comparing representation geometry. Pure numpy implementation (no extra deps).

RSA expects trial-averaged data (n_conditions, n_timepoints, n_units).
CKA works on raw or PCA-reduced data.
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
    import rsatoolbox

    RSATOOLBOX_AVAILABLE = True
except ImportError:
    RSATOOLBOX_AVAILABLE = False


def linear_cka_numpy(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear CKA between two representation matrices.

    Implements Kornblith et al. 2019 — zero external dependencies.

    Args:
        X: First representation, shape (n_samples, n_features_x).
        Y: Second representation, shape (n_samples, n_features_y).

    Returns:
        CKA similarity in [0, 1].
    """
    # Center
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # HSIC (Hilbert-Schmidt Independence Criterion)
    XtY = X.T @ Y
    XtX = X.T @ X
    YtY = Y.T @ Y

    hsic_xy = np.linalg.norm(XtY, "fro") ** 2
    hsic_xx = np.linalg.norm(XtX, "fro") ** 2
    hsic_yy = np.linalg.norm(YtY, "fro") ** 2

    denom = np.sqrt(hsic_xx * hsic_yy)
    if denom < 1e-10:
        return 0.0

    return float(hsic_xy / denom)


class RSAMethod:
    """Representational Similarity Analysis.

    Builds RDMs from condition-averaged activity and computes
    representational dissimilarity statistics.

    Attributes:
        name: Method identifier ("rsa").
        rdm_method: Method for computing RDMs ("euclidean" or "correlation").
    """

    name: str = "rsa"

    def __init__(self, rdm_method: str = "euclidean") -> None:
        self.rdm_method = rdm_method

    def compute(self, data: Any) -> AnalysisResult:
        """Compute RSA on neural population data.

        Expects trial-averaged data or will average internally.

        Args:
            data: NeuralPopulationData (ideally trial-averaged).

        Returns:
            AnalysisResult with RDM array and scalar dissimilarity stats.
        """
        from ..population import NeuralPopulationData

        assert isinstance(data, NeuralPopulationData)

        activity = data.activity  # (n_conditions, n_timepoints, n_units)
        n_cond = activity.shape[0]

        # Flatten time: each condition → one vector
        flat = activity.reshape(n_cond, -1)  # (n_conditions, n_time * n_units)

        if RSATOOLBOX_AVAILABLE:
            rdm = self._compute_rdm_rsatoolbox(flat, data.trial_labels)
        else:
            rdm = self._compute_rdm_numpy(flat)

        # Scalar features from RDM
        upper_tri = rdm[np.triu_indices(n_cond, k=1)]

        scalars = {
            "mean_dissimilarity": float(upper_tri.mean()) if len(upper_tri) > 0 else 0.0,
            "std_dissimilarity": float(upper_tri.std()) if len(upper_tri) > 0 else 0.0,
            "max_dissimilarity": float(upper_tri.max()) if len(upper_tri) > 0 else 0.0,
            "min_dissimilarity": float(upper_tri.min()) if len(upper_tri) > 0 else 0.0,
            "n_conditions": float(n_cond),
        }

        return AnalysisResult(
            method=self.name,
            config_hash="",
            seed=0,
            variant="",
            arrays={"rdm": rdm, "rdm_vector": upper_tri},
            scalars=scalars,
        )

    def _compute_rdm_rsatoolbox(
        self, flat: np.ndarray, labels: tuple[str, ...]
    ) -> np.ndarray:
        """Compute RDM using rsatoolbox."""
        obs_descriptors = {"condition": list(labels)}
        dataset: Any = rsatoolbox.data.Dataset(  # type: ignore[attr-defined,no-untyped-call]
            measurements=flat,
            obs_descriptors=obs_descriptors,
        )
        rdm_obj: Any = rsatoolbox.rdm.calc_rdm(  # type: ignore[attr-defined]
            dataset, method=self.rdm_method,
        )
        # Get full square matrix
        rdm_vector: Any = rdm_obj.get_vectors()
        n = flat.shape[0]
        rdm = np.zeros((n, n))
        rdm[np.triu_indices(n, k=1)] = rdm_vector.flatten()
        rdm = rdm + rdm.T
        return np.asarray(rdm)

    def _compute_rdm_numpy(self, flat: np.ndarray) -> np.ndarray:
        """Compute pairwise distance RDM with pure numpy (no scipy needed)."""
        n = flat.shape[0]
        rdm = np.zeros((n, n))

        if self.rdm_method == "correlation":
            # Correlation distance = 1 - Pearson r
            centered = flat - flat.mean(axis=1, keepdims=True)
            norms = np.linalg.norm(centered, axis=1, keepdims=True)
            norms = np.where(norms < 1e-10, 1.0, norms)
            normed = centered / norms
            corr = normed @ normed.T
            rdm = np.asarray(1.0 - corr)
        else:
            # Euclidean distance
            for i in range(n):
                for j in range(i + 1, n):
                    d = float(np.linalg.norm(flat[i] - flat[j]))
                    rdm[i, j] = d
                    rdm[j, i] = d

        return np.asarray(rdm)


class CKAMethod:
    """Centered Kernel Alignment for representation comparison.

    Uses pure numpy implementation (Kornblith et al. 2019).
    No external dependency required.

    CKA is typically used to compare representations across
    different systems/conditions rather than within a single dataset.
    When applied to a single dataset, computes self-CKA across
    different time windows or condition splits.

    Attributes:
        name: Method identifier ("cka").
    """

    name: str = "cka"

    def compute(self, data: Any) -> AnalysisResult:
        """Compute CKA-based metrics on neural population data.

        Computes:
        - Self-CKA between first and second half of trials (split-half reliability).
        - CKA between different time windows (temporal stability).

        Args:
            data: NeuralPopulationData instance.

        Returns:
            AnalysisResult with CKA similarity matrices and scalar metrics.
        """
        from ..population import NeuralPopulationData

        assert isinstance(data, NeuralPopulationData)

        activity = data.activity  # (n_trials, n_timepoints, n_units)
        n_trials, n_time, n_units = activity.shape

        # 1. Split-half CKA reliability
        half = n_trials // 2
        if half > 0:
            X1 = activity[:half].reshape(half, -1)  # (half, n_time * n_units)
            X2 = activity[half : 2 * half].reshape(half, -1)
            # Need same number of samples
            min_trials = min(X1.shape[0], X2.shape[0])
            split_half_cka = linear_cka_numpy(X1[:min_trials], X2[:min_trials])
        else:
            split_half_cka = 0.0

        # 2. Temporal CKA: compare first half of time with second half
        t_half = n_time // 2
        if t_half > 0:
            X_early = activity[:, :t_half, :].reshape(n_trials, -1)
            X_late = activity[:, t_half : 2 * t_half, :].reshape(n_trials, -1)
            temporal_cka = linear_cka_numpy(X_early, X_late)
        else:
            temporal_cka = 0.0

        # 3. Time-resolved CKA matrix (optional, if enough timepoints)
        n_windows = min(5, n_time)
        window_size = n_time // n_windows
        cka_matrix = np.zeros((n_windows, n_windows))
        for i in range(n_windows):
            Xi = activity[:, i * window_size : (i + 1) * window_size, :].reshape(
                n_trials, -1
            )
            for j in range(i, n_windows):
                Xj = activity[:, j * window_size : (j + 1) * window_size, :].reshape(
                    n_trials, -1
                )
                cka_val = linear_cka_numpy(Xi, Xj)
                cka_matrix[i, j] = cka_val
                cka_matrix[j, i] = cka_val

        return AnalysisResult(
            method=self.name,
            config_hash="",
            seed=0,
            variant="",
            arrays={
                "cka_temporal_matrix": cka_matrix,
            },
            scalars={
                "split_half_cka": split_half_cka,
                "temporal_cka": temporal_cka,
                "mean_self_cka": float(np.diag(cka_matrix).mean()),
            },
        )
