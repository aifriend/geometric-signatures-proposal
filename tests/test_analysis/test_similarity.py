"""Tests for RSA and CKA analysis methods."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis.similarity_method import (
    CKAMethod,
    RSAMethod,
    linear_cka_numpy,
)
from geometric_signatures.population import NeuralPopulationData


@pytest.fixture()
def sample_data() -> NeuralPopulationData:
    """Small synthetic data for similarity tests."""
    rng = np.random.default_rng(42)
    return NeuralPopulationData(
        activity=rng.standard_normal((8, 10, 6)),
        trial_labels=("task_a", "task_a", "task_b", "task_b",
                       "task_a", "task_a", "task_b", "task_b"),
        time_axis=np.arange(10, dtype=np.float64),
        unit_labels=tuple(f"u{i}" for i in range(6)),
        source="rnn",
        metadata={},
    )


class TestLinearCKA:
    """Tests for linear CKA implementation."""

    def test_self_similarity_is_one(self) -> None:
        X = np.random.randn(10, 5)
        cka = linear_cka_numpy(X, X)
        np.testing.assert_allclose(cka, 1.0, atol=1e-10)

    def test_range_zero_to_one(self) -> None:
        X = np.random.randn(10, 5)
        Y = np.random.randn(10, 3)
        cka = linear_cka_numpy(X, Y)
        assert 0.0 <= cka <= 1.0 + 1e-10

    def test_symmetric(self) -> None:
        X = np.random.randn(10, 5)
        Y = np.random.randn(10, 3)
        assert abs(linear_cka_numpy(X, Y) - linear_cka_numpy(Y, X)) < 1e-10

    def test_same_structure_orthogonal_dims_high_cka(self) -> None:
        """CKA is rotation-invariant — same data in different dims gives CKA ≈ 1."""
        n = 100
        X = np.zeros((n, 2))
        X[:, 0] = np.arange(n)
        Y = np.zeros((n, 2))
        Y[:, 1] = np.arange(n)
        cka = linear_cka_numpy(X, Y)
        # Same sample-sample structure → CKA ≈ 1 regardless of feature alignment
        np.testing.assert_allclose(cka, 1.0, atol=1e-10)

    def test_independent_representations_low_cka(self) -> None:
        """Independent random representations should have low CKA."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        Y = rng.standard_normal((100, 5))
        cka = linear_cka_numpy(X, Y)
        assert cka < 0.5

    def test_identical_up_to_rotation(self) -> None:
        """Rotated representations should have high CKA."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        # Random orthogonal rotation
        Q, _ = np.linalg.qr(rng.standard_normal((5, 5)))
        Y = X @ Q
        cka = linear_cka_numpy(X, Y)
        np.testing.assert_allclose(cka, 1.0, atol=1e-10)

    def test_handles_zero_matrix(self) -> None:
        X = np.zeros((10, 5))
        Y = np.random.randn(10, 5)
        cka = linear_cka_numpy(X, Y)
        assert cka == 0.0


class TestRSAMethod:
    """Tests for RSA analysis."""

    def test_compute_returns_analysis_result(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = RSAMethod()
        result = method.compute(sample_data)
        assert result.method == "rsa"

    def test_rdm_is_symmetric(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod()
        result = method.compute(sample_data)
        rdm = result.arrays["rdm"]
        np.testing.assert_allclose(rdm, rdm.T, atol=1e-10)

    def test_rdm_diagonal_is_zero(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod()
        result = method.compute(sample_data)
        rdm = result.arrays["rdm"]
        np.testing.assert_allclose(np.diag(rdm), 0.0, atol=1e-10)

    def test_rdm_shape(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod()
        result = method.compute(sample_data)
        n_cond = sample_data.activity.shape[0]
        rdm = result.arrays["rdm"]
        assert rdm.shape == (n_cond, n_cond)

    def test_scalars_present(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod()
        result = method.compute(sample_data)
        assert "mean_dissimilarity" in result.scalars
        assert "std_dissimilarity" in result.scalars
        assert "n_conditions" in result.scalars

    def test_rdm_values_nonnegative(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod(rdm_method="euclidean")
        result = method.compute(sample_data)
        rdm = result.arrays["rdm"]
        assert np.all(rdm >= -1e-10)

    def test_correlation_rdm_method(self, sample_data: NeuralPopulationData) -> None:
        method = RSAMethod(rdm_method="correlation")
        result = method.compute(sample_data)
        assert result.method == "rsa"
        assert "rdm" in result.arrays


class TestCKAMethod:
    """Tests for CKA analysis."""

    def test_compute_returns_analysis_result(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        assert result.method == "cka"

    def test_split_half_cka_range(self, sample_data: NeuralPopulationData) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        cka_val = result.scalars["split_half_cka"]
        assert 0.0 <= cka_val <= 1.0 + 1e-10

    def test_temporal_cka_present(self, sample_data: NeuralPopulationData) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        assert "temporal_cka" in result.scalars

    def test_temporal_matrix_shape(self, sample_data: NeuralPopulationData) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        matrix = result.arrays["cka_temporal_matrix"]
        assert matrix.ndim == 2
        assert matrix.shape[0] == matrix.shape[1]

    def test_temporal_matrix_symmetric(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        matrix = result.arrays["cka_temporal_matrix"]
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)

    def test_mean_self_cka_present(self, sample_data: NeuralPopulationData) -> None:
        method = CKAMethod()
        result = method.compute(sample_data)
        assert "mean_self_cka" in result.scalars
