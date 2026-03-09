"""Tests for cross-system comparison."""

from __future__ import annotations

import numpy as np
import pytest

from geometric_signatures.comparison.cross_system import (
    CrossSystemResult,
    compare_across_systems,
    identify_shared_signatures,
)
from geometric_signatures.statistics.aggregation import AggregatedResult
from geometric_signatures.statistics.bootstrap import BootstrapCI
from geometric_signatures.statistics.permutation import PermutationTestResult


def _make_aggregated(
    method: str,
    variant: str,
    seed_values: dict[str, list[float]],
) -> AggregatedResult:
    """Helper to create AggregatedResult from seed values."""
    scalar_means = {k: float(np.mean(v)) for k, v in seed_values.items()}
    scalar_sems = {k: float(np.std(v) / np.sqrt(len(v))) for k, v in seed_values.items()}
    scalar_cis = {
        k: BootstrapCI(
            point_estimate=scalar_means[k],
            ci_lower=scalar_means[k] - 1.96 * scalar_sems[k],
            ci_upper=scalar_means[k] + 1.96 * scalar_sems[k],
            confidence_level=0.95,
            n_bootstrap=100,
        )
        for k, v in seed_values.items()
    }
    return AggregatedResult(
        variant=variant,
        method=method,
        n_seeds=len(next(iter(seed_values.values()))),
        scalar_means=scalar_means,
        scalar_sems=scalar_sems,
        scalar_cis=scalar_cis,
        seed_values=seed_values,
    )


class TestCompareAcrossSystems:
    """Tests for compare_across_systems."""

    def test_returns_cross_system_result(self) -> None:
        """Returns CrossSystemResult type."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "rnn",
                {"split_half_cka": [0.8, 0.85, 0.82, 0.79, 0.83]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "bio",
                {"split_half_cka": [0.75, 0.78, 0.77, 0.76, 0.79]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b,
            system_a_name="rnn", system_b_name="bio",
            rng=rng,
        )

        assert isinstance(result, CrossSystemResult)

    def test_system_names_stored(self) -> None:
        """System names are preserved in result."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "rnn",
                {"metric": [1.0, 1.1, 0.9, 1.0, 1.05]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "bio",
                {"metric": [1.0, 0.95, 1.1, 1.0, 0.98]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b,
            system_a_name="my_rnn", system_b_name="my_bio",
            rng=rng,
        )

        assert result.system_a == "my_rnn"
        assert result.system_b == "my_bio"

    def test_same_values_high_p(self) -> None:
        """Same underlying values → high p-value → shared signature."""
        rng = np.random.default_rng(42)
        values = [0.8, 0.85, 0.82, 0.79, 0.83]
        results_a = {
            "cka": _make_aggregated("cka", "a", {"metric": values}),
        }
        results_b = {
            "cka": _make_aggregated("cka", "b", {"metric": values}),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        assert result.significance["cka.metric"].p_value > 0.05
        assert "cka.metric" in result.shared_signatures

    def test_different_values_low_p(self) -> None:
        """Very different values → low p-value → divergent signature."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "a",
                {"metric": [0.1, 0.11, 0.12, 0.09, 0.10]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "b",
                {"metric": [10.0, 10.1, 10.2, 9.9, 10.0]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        assert result.significance["cka.metric"].p_value < 0.05
        assert "cka.metric" in result.divergent_signatures

    def test_effect_sizes_present(self) -> None:
        """Effect sizes are computed for all compared metrics."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "a",
                {"m1": [1.0, 1.1, 0.9], "m2": [2.0, 2.1, 1.9]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "b",
                {"m1": [1.5, 1.6, 1.4], "m2": [2.0, 2.1, 1.9]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        assert "cka.m1" in result.effect_sizes
        assert "cka.m2" in result.effect_sizes

    def test_similarity_range(self) -> None:
        """Metric similarities are in [0, 1]."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "a",
                {"metric": [0.5, 0.6, 0.55, 0.52, 0.58]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "b",
                {"metric": [5.0, 5.1, 5.2, 4.9, 5.0]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        for sim in result.metric_similarities.values():
            assert 0.0 <= sim <= 1.0

    def test_multiple_methods(self) -> None:
        """Compares across multiple analysis methods."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated("cka", "a", {"m": [1.0, 1.1, 0.9]}),
            "rsa": _make_aggregated("rsa", "a", {"m": [2.0, 2.1, 1.9]}),
        }
        results_b = {
            "cka": _make_aggregated("cka", "b", {"m": [1.5, 1.6, 1.4]}),
            "rsa": _make_aggregated("rsa", "b", {"m": [2.5, 2.6, 2.4]}),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        assert "cka" in result.methods
        assert "rsa" in result.methods
        assert "cka.m" in result.significance
        assert "rsa.m" in result.significance

    def test_filter_methods(self) -> None:
        """Only compares specified methods."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated("cka", "a", {"m": [1.0, 1.1, 0.9]}),
            "rsa": _make_aggregated("rsa", "a", {"m": [2.0, 2.1, 1.9]}),
        }
        results_b = {
            "cka": _make_aggregated("cka", "b", {"m": [1.5, 1.6, 1.4]}),
            "rsa": _make_aggregated("rsa", "b", {"m": [2.5, 2.6, 2.4]}),
        }

        result = compare_across_systems(
            results_a, results_b,
            methods=["cka"],
            rng=rng,
        )

        assert "cka" in result.methods
        assert "rsa" not in result.methods
        assert "cka.m" in result.significance
        assert "rsa.m" not in result.significance

    def test_no_common_methods_raises(self) -> None:
        """Raises ValueError when no methods overlap."""
        results_a = {
            "cka": _make_aggregated("cka", "a", {"m": [1.0, 1.1, 0.9]}),
        }
        results_b = {
            "rsa": _make_aggregated("rsa", "b", {"m": [2.0, 2.1, 1.9]}),
        }

        with pytest.raises(ValueError, match="No common methods"):
            compare_across_systems(results_a, results_b)

    def test_no_comparable_metrics_raises(self) -> None:
        """Raises ValueError when no metrics overlap."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated("cka", "a", {"metric_a": [1.0, 1.1, 0.9]}),
        }
        results_b = {
            "cka": _make_aggregated("cka", "b", {"metric_b": [2.0, 2.1, 1.9]}),
        }

        with pytest.raises(ValueError, match="No comparable metrics"):
            compare_across_systems(results_a, results_b, rng=rng)

    def test_deterministic_with_seed(self) -> None:
        """Results are reproducible with same random seed."""
        results_a = {
            "cka": _make_aggregated(
                "cka", "a",
                {"metric": [0.5, 0.6, 0.55, 0.52, 0.58]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "b",
                {"metric": [0.7, 0.75, 0.72, 0.69, 0.73]},
            ),
        }

        r1 = compare_across_systems(
            results_a, results_b, rng=np.random.default_rng(42),
        )
        r2 = compare_across_systems(
            results_a, results_b, rng=np.random.default_rng(42),
        )

        assert r1.significance["cka.metric"].p_value == r2.significance["cka.metric"].p_value

    def test_shared_divergent_partition(self) -> None:
        """Shared and divergent signatures partition all metrics."""
        rng = np.random.default_rng(42)
        results_a = {
            "cka": _make_aggregated(
                "cka", "a",
                {"same": [1.0, 1.1, 0.9, 1.0, 1.05],
                 "diff": [0.1, 0.11, 0.09, 0.10, 0.12]},
            ),
        }
        results_b = {
            "cka": _make_aggregated(
                "cka", "b",
                {"same": [1.0, 0.95, 1.05, 1.02, 0.98],
                 "diff": [10.0, 10.1, 9.9, 10.0, 10.05]},
            ),
        }

        result = compare_across_systems(
            results_a, results_b, rng=rng,
        )

        all_metrics = set(result.shared_signatures) | set(result.divergent_signatures)
        assert all_metrics == set(result.significance.keys())


class TestIdentifySharedSignatures:
    """Tests for identify_shared_signatures."""

    def test_classifies_by_alpha(self) -> None:
        """Metrics with p < alpha are divergent, p >= alpha are shared."""
        significance = {
            "m1": PermutationTestResult(
                observed_statistic=1.0, p_value=0.01,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
            "m2": PermutationTestResult(
                observed_statistic=0.5, p_value=0.10,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
            "m3": PermutationTestResult(
                observed_statistic=0.1, p_value=0.50,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
        }

        shared, divergent = identify_shared_signatures(significance, alpha=0.05)

        assert "m1" in divergent
        assert "m2" in shared
        assert "m3" in shared

    def test_empty_input(self) -> None:
        """Empty significance dict returns empty tuples."""
        shared, divergent = identify_shared_signatures({})

        assert shared == ()
        assert divergent == ()

    def test_all_shared(self) -> None:
        """All metrics shared when all p-values are high."""
        significance = {
            "m1": PermutationTestResult(
                observed_statistic=0.1, p_value=0.80,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
            "m2": PermutationTestResult(
                observed_statistic=0.2, p_value=0.90,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
        }

        shared, divergent = identify_shared_signatures(significance, alpha=0.05)

        assert len(shared) == 2
        assert len(divergent) == 0

    def test_all_divergent(self) -> None:
        """All metrics divergent when all p-values are low."""
        significance = {
            "m1": PermutationTestResult(
                observed_statistic=5.0, p_value=0.001,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
            "m2": PermutationTestResult(
                observed_statistic=4.0, p_value=0.01,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
        }

        shared, divergent = identify_shared_signatures(significance, alpha=0.05)

        assert len(shared) == 0
        assert len(divergent) == 2

    def test_custom_alpha(self) -> None:
        """Custom alpha changes classification boundary."""
        significance = {
            "m1": PermutationTestResult(
                observed_statistic=1.0, p_value=0.08,
                null_distribution=np.zeros(100), n_permutations=100,
            ),
        }

        # With alpha=0.05, p=0.08 → shared
        shared_05, div_05 = identify_shared_signatures(
            significance, alpha=0.05
        )
        assert "m1" in shared_05

        # With alpha=0.10, p=0.08 → divergent
        shared_10, div_10 = identify_shared_signatures(
            significance, alpha=0.10
        )
        assert "m1" in div_10
