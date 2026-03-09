"""Tests for population geometry analysis method."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis.geometry_method import (
    GeometryMethod,
    condition_separability,
    effective_dimensionality,
    participation_ratio,
    trajectory_speed,
)
from geometric_signatures.population import NeuralPopulationData


@pytest.fixture()
def sample_data() -> NeuralPopulationData:
    """Small synthetic data for geometry tests."""
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


class TestParticipationRatio:
    """Tests for participation ratio computation."""

    def test_isotropic_data(self) -> None:
        """Isotropic data should have PR ≈ n_features."""
        rng = np.random.default_rng(42)
        n_features = 5
        data = rng.standard_normal((1000, n_features))
        pr = participation_ratio(data)
        np.testing.assert_allclose(pr, n_features, atol=0.5)

    def test_one_dimensional(self) -> None:
        """Data along a single axis should have PR ≈ 1."""
        rng = np.random.default_rng(42)
        data = np.zeros((100, 5))
        data[:, 0] = rng.standard_normal(100)
        pr = participation_ratio(data)
        np.testing.assert_allclose(pr, 1.0, atol=0.1)

    def test_positive_value(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 10))
        pr = participation_ratio(data)
        assert pr > 0

    def test_zero_data(self) -> None:
        data = np.zeros((10, 5))
        pr = participation_ratio(data)
        assert pr == 0.0


class TestEffectiveDimensionality:
    """Tests for effective dimensionality."""

    def test_returns_integer(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 10))
        dim = effective_dimensionality(data)
        assert isinstance(dim, int)

    def test_bounded_by_n_features(self) -> None:
        rng = np.random.default_rng(42)
        n_features = 5
        data = rng.standard_normal((50, n_features))
        dim = effective_dimensionality(data, variance_threshold=0.99)
        assert dim <= n_features

    def test_lower_threshold_gives_fewer_dims(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal((50, 10))
        dim_high = effective_dimensionality(data, variance_threshold=0.99)
        dim_low = effective_dimensionality(data, variance_threshold=0.5)
        assert dim_low <= dim_high

    def test_one_dim_data(self) -> None:
        data = np.zeros((100, 5))
        data[:, 0] = np.arange(100)
        dim = effective_dimensionality(data, variance_threshold=0.95)
        assert dim == 1


class TestConditionSeparability:
    """Tests for condition separability."""

    def test_well_separated_conditions(self) -> None:
        """Clearly separated groups should have high separability."""
        rng = np.random.default_rng(42)
        group_a = rng.standard_normal((50, 5)) + np.array([10, 0, 0, 0, 0])
        group_b = rng.standard_normal((50, 5)) + np.array([-10, 0, 0, 0, 0])
        data = np.vstack([group_a, group_b])
        labels = tuple(["a"] * 50 + ["b"] * 50)
        sep = condition_separability(data, labels)
        assert sep > 1.0

    def test_overlapping_conditions(self) -> None:
        """Overlapping groups should have low separability."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((100, 5))
        labels = tuple(["a"] * 50 + ["b"] * 50)
        sep = condition_separability(data, labels)
        assert sep < 1.0

    def test_single_condition_returns_zero(self) -> None:
        data = np.random.randn(10, 5)
        labels = tuple(["a"] * 10)
        sep = condition_separability(data, labels)
        assert sep == 0.0


class TestTrajectorySpeed:
    """Tests for trajectory speed computation."""

    def test_output_shape(self) -> None:
        activity = np.random.randn(4, 10, 6)
        speeds = trajectory_speed(activity)
        assert speeds.shape == (4, 9)  # n_time - 1

    def test_nonnegative(self) -> None:
        activity = np.random.randn(4, 10, 6)
        speeds = trajectory_speed(activity)
        assert np.all(speeds >= 0)

    def test_stationary_has_zero_speed(self) -> None:
        activity = np.ones((4, 10, 6))
        speeds = trajectory_speed(activity)
        np.testing.assert_allclose(speeds, 0.0)


class TestGeometryMethod:
    """Tests for the full GeometryMethod."""

    def test_compute_returns_analysis_result(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = GeometryMethod()
        result = method.compute(sample_data)
        assert result.method == "population_geometry"

    def test_scalars_present(self, sample_data: NeuralPopulationData) -> None:
        method = GeometryMethod()
        result = method.compute(sample_data)
        expected_keys = {
            "participation_ratio",
            "effective_dimensionality",
            "condition_separability",
            "mean_trajectory_speed",
            "std_trajectory_speed",
            "max_trajectory_speed",
            "mean_trajectory_length",
            "top_3_variance_explained",
        }
        assert expected_keys.issubset(set(result.scalars.keys()))

    def test_arrays_present(self, sample_data: NeuralPopulationData) -> None:
        method = GeometryMethod()
        result = method.compute(sample_data)
        assert "variance_explained" in result.arrays
        assert "trajectory_speeds" in result.arrays
        assert "trajectory_lengths" in result.arrays

    def test_participation_ratio_positive(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = GeometryMethod()
        result = method.compute(sample_data)
        assert result.scalars["participation_ratio"] > 0

    def test_variance_explained_sums_to_one(
        self, sample_data: NeuralPopulationData
    ) -> None:
        method = GeometryMethod()
        result = method.compute(sample_data)
        ve = result.arrays["variance_explained"]
        np.testing.assert_allclose(ve.sum(), 1.0, atol=1e-10)
