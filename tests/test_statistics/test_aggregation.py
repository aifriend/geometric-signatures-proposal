"""Tests for multi-seed result aggregation and variant comparison."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.analysis.base import AnalysisResult
from geometric_signatures.statistics.aggregation import (
    AggregatedResult,
    aggregate_across_seeds,
    compare_variants,
)
from geometric_signatures.statistics.bootstrap import BootstrapCI
from geometric_signatures.statistics.permutation import PermutationTestResult


def _make_result(
    method: str = "population_geometry",
    seed: int = 0,
    variant: str = "complete",
    scalars: dict[str, float] | None = None,
) -> AnalysisResult:
    """Helper to create a minimal AnalysisResult."""
    if scalars is None:
        scalars = {"participation_ratio": 3.0 + seed * 0.1}
    return AnalysisResult(
        method=method,
        config_hash="abc123",
        seed=seed,
        variant=variant,
        scalars=scalars,
    )


class TestAggregateAcrossSeeds:
    """Tests for aggregate_across_seeds."""

    def test_returns_aggregated_result(self) -> None:
        results = [_make_result(seed=i) for i in range(5)]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        assert isinstance(agg, AggregatedResult)

    def test_n_seeds(self) -> None:
        results = [_make_result(seed=i) for i in range(7)]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        assert agg.n_seeds == 7

    def test_mean_is_correct(self) -> None:
        results = [
            _make_result(seed=0, scalars={"metric": 1.0}),
            _make_result(seed=1, scalars={"metric": 2.0}),
            _make_result(seed=2, scalars={"metric": 3.0}),
        ]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        np.testing.assert_allclose(agg.scalar_means["metric"], 2.0)

    def test_sem_is_correct(self) -> None:
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        results = [
            _make_result(seed=i, scalars={"metric": v})
            for i, v in enumerate(values)
        ]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        expected_sem = np.std(values, ddof=1) / np.sqrt(len(values))
        np.testing.assert_allclose(agg.scalar_sems["metric"], expected_sem)

    def test_ci_contains_mean(self) -> None:
        results = [_make_result(seed=i) for i in range(10)]
        agg = aggregate_across_seeds(results, n_bootstrap=1000,
                                     rng=np.random.default_rng(42))
        key = "participation_ratio"
        ci = agg.scalar_cis[key]
        assert isinstance(ci, BootstrapCI)
        assert ci.ci_lower <= agg.scalar_means[key] <= ci.ci_upper

    def test_seed_values_stored(self) -> None:
        results = [
            _make_result(seed=i, scalars={"m": float(i)})
            for i in range(5)
        ]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        np.testing.assert_array_equal(
            agg.seed_values["m"], [0.0, 1.0, 2.0, 3.0, 4.0]
        )

    def test_variant_and_method_preserved(self) -> None:
        results = [
            _make_result(seed=i, variant="ablate_gating", method="cka")
            for i in range(3)
        ]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        assert agg.variant == "ablate_gating"
        assert agg.method == "cka"

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            aggregate_across_seeds([])

    def test_mixed_methods_raises(self) -> None:
        results = [
            _make_result(seed=0, method="cka"),
            _make_result(seed=1, method="rsa"),
        ]
        with pytest.raises(ValueError, match="same method"):
            aggregate_across_seeds(results)

    def test_single_seed_degeneate_ci(self) -> None:
        """Single seed → SEM=0, degenerate CI."""
        results = [_make_result(seed=0, scalars={"m": 5.0})]
        agg = aggregate_across_seeds(results, n_bootstrap=100,
                                     rng=np.random.default_rng(42))
        assert agg.scalar_sems["m"] == 0.0
        assert agg.scalar_cis["m"].ci_lower == 5.0
        assert agg.scalar_cis["m"].ci_upper == 5.0

    def test_deterministic_with_seed(self) -> None:
        results = [_make_result(seed=i) for i in range(5)]
        a1 = aggregate_across_seeds(results, n_bootstrap=500,
                                    rng=np.random.default_rng(42))
        a2 = aggregate_across_seeds(results, n_bootstrap=500,
                                    rng=np.random.default_rng(42))
        key = "participation_ratio"
        assert a1.scalar_cis[key].ci_lower == a2.scalar_cis[key].ci_lower
        assert a1.scalar_cis[key].ci_upper == a2.scalar_cis[key].ci_upper


class TestCompareVariants:
    """Tests for variant comparison via permutation test."""

    def test_returns_permutation_test_result(self) -> None:
        a = [_make_result(seed=i, variant="complete", scalars={"m": float(i)})
             for i in range(5)]
        b = [_make_result(seed=i, variant="ablate_x", scalars={"m": float(i + 10)})
             for i in range(5)]
        result = compare_variants(a, b, "m", n_permutations=100,
                                  rng=np.random.default_rng(42))
        assert isinstance(result, PermutationTestResult)

    def test_same_values_high_p(self) -> None:
        """Identical metrics → non-significant."""
        a = [_make_result(seed=i, variant="complete", scalars={"m": 5.0})
             for i in range(10)]
        b = [_make_result(seed=i, variant="ablate_x", scalars={"m": 5.0})
             for i in range(10)]
        result = compare_variants(a, b, "m", n_permutations=500,
                                  rng=np.random.default_rng(42))
        assert result.p_value > 0.05

    def test_different_values_low_p(self) -> None:
        """Clearly different metrics → significant."""
        rng = np.random.default_rng(42)
        a = [_make_result(seed=i, variant="complete",
                          scalars={"m": 10.0 + rng.standard_normal() * 0.1})
             for i in range(15)]
        b = [_make_result(seed=i, variant="ablate_x",
                          scalars={"m": 1.0 + rng.standard_normal() * 0.1})
             for i in range(15)]
        result = compare_variants(a, b, "m", n_permutations=500,
                                  rng=np.random.default_rng(42))
        assert result.p_value < 0.05

    def test_missing_metric_raises(self) -> None:
        a = [_make_result(seed=0, scalars={"m": 1.0})]
        b = [_make_result(seed=0, scalars={"other": 2.0})]
        with pytest.raises(ValueError, match="not found"):
            compare_variants(a, b, "m")

    def test_observed_statistic_sign(self) -> None:
        """When A > B, observed statistic should be positive."""
        a = [_make_result(seed=i, scalars={"m": 10.0}) for i in range(5)]
        b = [_make_result(seed=i, scalars={"m": 2.0}) for i in range(5)]
        result = compare_variants(a, b, "m", n_permutations=100,
                                  rng=np.random.default_rng(42))
        assert result.observed_statistic > 0
