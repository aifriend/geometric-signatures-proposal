"""Tests for bootstrap confidence intervals and effect sizes."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.statistics.bootstrap import (
    BootstrapCI,
    bootstrap_confidence_interval,
    effect_size_cohens_d,
)


class TestBootstrapCI:
    """Tests for bootstrap confidence interval computation."""

    def test_returns_bootstrap_ci_type(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(50)
        result = bootstrap_confidence_interval(data, n_bootstrap=500, rng=rng)
        assert isinstance(result, BootstrapCI)

    def test_point_estimate_is_mean(self) -> None:
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rng = np.random.default_rng(42)
        result = bootstrap_confidence_interval(data, n_bootstrap=500, rng=rng)
        np.testing.assert_allclose(result.point_estimate, 3.0)

    def test_ci_contains_point_estimate(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        result = bootstrap_confidence_interval(data, n_bootstrap=1000, rng=rng)
        assert result.ci_lower <= result.point_estimate <= result.ci_upper

    def test_ci_lower_less_than_upper(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(50)
        result = bootstrap_confidence_interval(data, n_bootstrap=1000, rng=rng)
        assert result.ci_lower <= result.ci_upper

    def test_wider_ci_for_lower_confidence(self) -> None:
        """99% CI should be wider than 90% CI."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)

        ci_90 = bootstrap_confidence_interval(
            data, confidence=0.90, n_bootstrap=5000,
            rng=np.random.default_rng(42),
        )
        ci_99 = bootstrap_confidence_interval(
            data, confidence=0.99, n_bootstrap=5000,
            rng=np.random.default_rng(42),
        )

        width_90 = ci_90.ci_upper - ci_90.ci_lower
        width_99 = ci_99.ci_upper - ci_99.ci_lower
        assert width_99 > width_90

    def test_confidence_level_stored(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(20)
        result = bootstrap_confidence_interval(
            data, confidence=0.90, n_bootstrap=100, rng=rng
        )
        assert result.confidence_level == 0.90

    def test_n_bootstrap_stored(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(20)
        result = bootstrap_confidence_interval(data, n_bootstrap=777, rng=rng)
        assert result.n_bootstrap == 777

    def test_deterministic_with_seed(self) -> None:
        data = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        r1 = bootstrap_confidence_interval(
            data, n_bootstrap=500, rng=np.random.default_rng(42)
        )
        r2 = bootstrap_confidence_interval(
            data, n_bootstrap=500, rng=np.random.default_rng(42)
        )
        assert r1.ci_lower == r2.ci_lower
        assert r1.ci_upper == r2.ci_upper

    def test_custom_statistic(self) -> None:
        rng = np.random.default_rng(42)
        data = rng.standard_normal(100)
        result = bootstrap_confidence_interval(
            data,
            statistic_fn=lambda x: float(np.median(x)),
            n_bootstrap=500,
            rng=rng,
        )
        np.testing.assert_allclose(
            result.point_estimate, np.median(data), atol=1e-10
        )

    def test_empty_data_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            bootstrap_confidence_interval(np.array([]))

    def test_invalid_confidence_raises(self) -> None:
        data = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Confidence"):
            bootstrap_confidence_interval(data, confidence=0.0)
        with pytest.raises(ValueError, match="Confidence"):
            bootstrap_confidence_interval(data, confidence=1.0)

    def test_single_observation(self) -> None:
        """Single observation → degenerate CI (point = lower = upper)."""
        rng = np.random.default_rng(42)
        data = np.array([5.0])
        result = bootstrap_confidence_interval(data, n_bootstrap=100, rng=rng)
        assert result.point_estimate == 5.0
        assert result.ci_lower == 5.0
        assert result.ci_upper == 5.0


class TestEffectSizeCohensD:
    """Tests for Cohen's d effect size."""

    def test_identical_groups_zero_d(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = effect_size_cohens_d(a, a)
        np.testing.assert_allclose(d, 0.0, atol=1e-10)

    def test_positive_when_a_greater(self) -> None:
        a = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        d = effect_size_cohens_d(a, b)
        assert d > 0

    def test_negative_when_b_greater(self) -> None:
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([10.0, 11.0, 12.0, 13.0, 14.0])
        d = effect_size_cohens_d(a, b)
        assert d < 0

    def test_antisymmetric(self) -> None:
        """Cohen's d(A, B) = -d(B, A)."""
        rng = np.random.default_rng(42)
        a = rng.standard_normal(20)
        b = rng.standard_normal(20) + 2.0
        d_ab = effect_size_cohens_d(a, b)
        d_ba = effect_size_cohens_d(b, a)
        np.testing.assert_allclose(d_ab, -d_ba, atol=1e-10)

    def test_known_value(self) -> None:
        """Two groups separated by exactly 1 pooled SD → d ≈ 1."""
        # Groups with known mean difference and identical variance
        a = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        b = a + np.std(a, ddof=1)  # shift by 1 sample std
        d = effect_size_cohens_d(a, b)
        np.testing.assert_allclose(d, -1.0, atol=0.01)

    def test_too_few_observations_raises(self) -> None:
        with pytest.raises(ValueError, match=">= 2"):
            effect_size_cohens_d(np.array([1.0]), np.array([2.0, 3.0]))

    def test_constant_groups_zero_d(self) -> None:
        """Both groups constant → d = 0 (pooled std is 0)."""
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([5.0, 5.0, 5.0])
        d = effect_size_cohens_d(a, b)
        assert d == 0.0
