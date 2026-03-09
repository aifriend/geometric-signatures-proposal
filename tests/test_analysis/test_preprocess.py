"""Tests for analysis preprocessing."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis.preprocess import (
    PREPROCESS_DEFAULTS,
    pca_reduce,
    preprocess_for_analysis,
    trial_average_by_condition,
    zscore_normalize,
)
from geometric_signatures.population import NeuralPopulationData


@pytest.fixture()
def sample_data() -> NeuralPopulationData:
    """Create small synthetic data for preprocessing tests."""
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


class TestPCAReduce:
    """Tests for PCA dimensionality reduction."""

    def test_reduces_dimensionality(self) -> None:
        activity = np.random.randn(4, 10, 8)
        reduced = pca_reduce(activity, n_components=3)
        assert reduced.shape == (4, 10, 3)

    def test_preserves_trial_structure(self) -> None:
        activity = np.random.randn(6, 5, 10)
        reduced = pca_reduce(activity, n_components=4)
        assert reduced.shape[0] == 6
        assert reduced.shape[1] == 5

    def test_caps_at_n_units(self) -> None:
        """n_components > n_units should be capped."""
        activity = np.random.randn(4, 10, 3)
        reduced = pca_reduce(activity, n_components=10)
        assert reduced.shape[2] <= 3

    def test_deterministic(self) -> None:
        activity = np.random.randn(4, 10, 8)
        r1 = pca_reduce(activity, n_components=3)
        r2 = pca_reduce(activity, n_components=3)
        np.testing.assert_array_equal(r1, r2)


class TestZscoreNormalize:
    """Tests for z-score normalization."""

    def test_output_shape(self) -> None:
        activity = np.random.randn(4, 10, 6)
        normalized = zscore_normalize(activity)
        assert normalized.shape == activity.shape

    def test_zero_mean(self) -> None:
        activity = np.random.randn(4, 10, 6)
        normalized = zscore_normalize(activity)
        flat = normalized.reshape(-1, 6)
        means = flat.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_unit_std(self) -> None:
        activity = np.random.randn(20, 10, 6) * 5 + 3
        normalized = zscore_normalize(activity)
        flat = normalized.reshape(-1, 6)
        stds = flat.std(axis=0)
        np.testing.assert_allclose(stds, 1.0, atol=1e-10)

    def test_handles_constant_units(self) -> None:
        """Units with zero variance should not produce NaN."""
        activity = np.ones((4, 10, 3))
        normalized = zscore_normalize(activity)
        assert not np.any(np.isnan(normalized))


class TestTrialAverageByCondition:
    """Tests for trial averaging."""

    def test_output_shape(self, sample_data: NeuralPopulationData) -> None:
        averaged = trial_average_by_condition(sample_data)
        # 2 unique labels → 2 condition averages
        assert averaged.shape[0] == 2
        assert averaged.shape[1] == sample_data.activity.shape[1]
        assert averaged.shape[2] == sample_data.activity.shape[2]

    def test_averages_correctly(self) -> None:
        activity = np.array([
            [[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]], [[7.0, 8.0]]
        ])
        data = NeuralPopulationData(
            activity=activity,
            trial_labels=("a", "a", "b", "b"),
            time_axis=np.array([0.0]),
            unit_labels=("u0", "u1"),
            source="rnn",
            metadata={},
        )
        averaged = trial_average_by_condition(data)
        # "a" average: (1+3)/2=2, (2+4)/2=3
        np.testing.assert_allclose(averaged[0, 0], [2.0, 3.0])
        # "b" average: (5+7)/2=6, (6+8)/2=7
        np.testing.assert_allclose(averaged[1, 0], [6.0, 7.0])


class TestPreprocessForAnalysis:
    """Tests for the main preprocessing function."""

    def test_no_preprocessing(self, sample_data: NeuralPopulationData) -> None:
        result = preprocess_for_analysis(sample_data, normalize="none")
        assert result.activity.shape == sample_data.activity.shape

    def test_zscore_preprocessing(self, sample_data: NeuralPopulationData) -> None:
        result = preprocess_for_analysis(
            sample_data, normalize="zscore"
        )
        flat = result.activity.reshape(-1, result.activity.shape[2])
        means = flat.mean(axis=0)
        np.testing.assert_allclose(means, 0.0, atol=1e-10)

    def test_pca_preprocessing(self, sample_data: NeuralPopulationData) -> None:
        result = preprocess_for_analysis(
            sample_data, n_components=3, normalize="none"
        )
        assert result.activity.shape[2] == 3
        assert len(result.unit_labels) == 3

    def test_trial_averaging(self, sample_data: NeuralPopulationData) -> None:
        result = preprocess_for_analysis(
            sample_data, trial_average=True, normalize="none"
        )
        # 2 unique conditions → 2 averaged trials
        assert result.activity.shape[0] == 2
        assert len(result.trial_labels) == 2
        # Trial metadata should be None after averaging
        assert result.trial_metadata is None

    def test_method_defaults_marble(self, sample_data: NeuralPopulationData) -> None:
        """MARBLE default: no PCA, no averaging."""
        result = preprocess_for_analysis(sample_data, method="marble")
        assert result.activity.shape == sample_data.activity.shape

    def test_method_defaults_persistent_homology(
        self, sample_data: NeuralPopulationData
    ) -> None:
        """PH default: PCA to 50, zscore. But our data has only 6 units."""
        result = preprocess_for_analysis(
            sample_data, method="persistent_homology"
        )
        # PCA capped at n_units=6
        assert result.activity.shape[2] == 6

    def test_method_defaults_rsa(self, sample_data: NeuralPopulationData) -> None:
        """RSA default: trial averaging."""
        result = preprocess_for_analysis(sample_data, method="rsa")
        assert result.activity.shape[0] == 2  # 2 conditions

    def test_explicit_args_override_defaults(
        self, sample_data: NeuralPopulationData
    ) -> None:
        """Explicit args take priority over method defaults."""
        result = preprocess_for_analysis(
            sample_data,
            method="rsa",
            trial_average=False,
            normalize="none",
        )
        # Should NOT average despite RSA default
        assert result.activity.shape[0] == 8

    def test_preprocess_defaults_exist_for_all_methods(self) -> None:
        """All registered methods should have preprocessing defaults."""
        expected = {"marble", "persistent_homology", "rsa", "cka", "population_geometry"}
        assert expected == set(PREPROCESS_DEFAULTS.keys())
