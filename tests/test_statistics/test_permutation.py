"""Tests for permutation testing."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.statistics.permutation import (
    PermutationTestResult,
    permutation_test,
)


class TestPermutationTest:
    """Tests for the two-sample permutation test."""

    def test_returns_result_type(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        result = permutation_test(a, b, n_permutations=100, rng=rng)
        assert isinstance(result, PermutationTestResult)

    def test_same_distribution_high_p_value(self) -> None:
        """Samples from the same distribution → non-significant."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(50)
        b = rng.standard_normal(50)
        result = permutation_test(a, b, n_permutations=500, rng=rng)
        # Should NOT reject null (same distribution)
        assert result.p_value > 0.05

    def test_different_distributions_low_p_value(self) -> None:
        """Clearly separated distributions → significant."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(30) + 5.0  # mean = 5
        b = rng.standard_normal(30) - 5.0  # mean = -5
        result = permutation_test(a, b, n_permutations=500, rng=rng)
        assert result.p_value < 0.05

    def test_p_value_range(self) -> None:
        """p-value must be in (0, 1]."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20)
        result = permutation_test(a, b, n_permutations=100, rng=rng)
        assert 0 < result.p_value <= 1.0

    def test_p_value_never_zero(self) -> None:
        """p-value should never be exactly 0 (floor at 1/(n+1))."""
        rng = np.random.default_rng(42)
        a = np.ones(10) * 100
        b = np.ones(10) * -100
        result = permutation_test(a, b, n_permutations=200, rng=rng)
        assert result.p_value > 0

    def test_null_distribution_shape(self) -> None:
        rng = np.random.default_rng(42)
        a = rng.standard_normal(10)
        b = rng.standard_normal(10)
        n_perm = 300
        result = permutation_test(a, b, n_permutations=n_perm, rng=rng)
        assert result.null_distribution.shape == (n_perm,)
        assert result.n_permutations == n_perm

    def test_deterministic_with_seed(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([6.0, 7.0, 8.0, 9.0, 10.0])

        r1 = permutation_test(a, b, n_permutations=200, rng=np.random.default_rng(99))
        r2 = permutation_test(a, b, n_permutations=200, rng=np.random.default_rng(99))

        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_custom_statistic(self) -> None:
        """Works with a custom test statistic."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20) + 3
        b = rng.standard_normal(20)

        def median_diff(x: np.ndarray, y: np.ndarray) -> float:
            return float(np.median(x) - np.median(y))

        result = permutation_test(
            a, b, statistic_fn=median_diff, n_permutations=500, rng=rng
        )
        assert result.observed_statistic > 0  # median(a) > median(b)

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            permutation_test(np.array([]), np.array([1, 2, 3]))

    def test_observed_statistic_is_mean_diff(self) -> None:
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([1.0, 2.0, 3.0])
        rng = np.random.default_rng(42)
        result = permutation_test(a, b, n_permutations=100, rng=rng)
        expected = a.mean() - b.mean()
        np.testing.assert_allclose(result.observed_statistic, expected)
