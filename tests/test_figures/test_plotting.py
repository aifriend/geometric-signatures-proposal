"""Tests for figure generation functions (smoke tests).

Each test verifies that a figure function produces a valid output file
without errors. We don't test visual correctness — just that the functions
run, write a file, and the file has non-zero size.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for CI

from geometric_signatures.comparison.cross_system import CrossSystemResult
from geometric_signatures.figures.plotting import (
    _p_to_stars,
    fig_ablation_heatmap,
    fig_cross_system_comparison,
    fig_effect_size_forest,
    fig_metric_comparison_bar,
    fig_persistence_summary,
)
from geometric_signatures.statistics.aggregation import AggregatedResult
from geometric_signatures.statistics.bootstrap import BootstrapCI
from geometric_signatures.statistics.permutation import PermutationTestResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_aggregated(
    variant: str,
    method: str,
    scalar_means: dict[str, float],
    n_seeds: int = 5,
) -> AggregatedResult:
    """Create an AggregatedResult with matching SEMs and CIs."""
    rng = np.random.default_rng(42)
    scalar_sems = {k: abs(v) * 0.1 for k, v in scalar_means.items()}
    scalar_cis = {
        k: BootstrapCI(
            point_estimate=v,
            ci_lower=v - abs(v) * 0.2,
            ci_upper=v + abs(v) * 0.2,
            confidence_level=0.95,
            n_bootstrap=100,
        )
        for k, v in scalar_means.items()
    }
    seed_values = {
        k: rng.normal(v, abs(v) * 0.2, n_seeds)
        for k, v in scalar_means.items()
    }
    return AggregatedResult(
        variant=variant,
        method=method,
        scalar_means=scalar_means,
        scalar_sems=scalar_sems,
        scalar_cis=scalar_cis,
        n_seeds=n_seeds,
        seed_values=seed_values,
    )


def _make_cross_system_result(
    n_metrics: int = 4,
) -> CrossSystemResult:
    """Create a CrossSystemResult with mock data."""
    rng = np.random.default_rng(42)
    metric_names = [f"metric_{i}" for i in range(n_metrics)]

    metric_similarities = {m: rng.uniform(0.3, 1.0) for m in metric_names}
    effect_sizes = {m: rng.uniform(-1.5, 1.5) for m in metric_names}
    p_values = rng.uniform(0.001, 0.2, n_metrics)

    significance = {}
    shared = []
    divergent = []
    for i, m in enumerate(metric_names):
        null_dist = rng.standard_normal(100)
        significance[m] = PermutationTestResult(
            observed_statistic=effect_sizes[m],
            p_value=float(p_values[i]),
            null_distribution=null_dist,
            n_permutations=100,
        )
        if p_values[i] >= 0.05:
            shared.append(m)
        else:
            divergent.append(m)

    return CrossSystemResult(
        system_a="rnn_complete",
        system_b="ibl_visp",
        methods=("cka", "rsa"),
        metric_similarities=metric_similarities,
        significance=significance,
        effect_sizes=effect_sizes,
        shared_signatures=tuple(shared),
        divergent_signatures=tuple(divergent),
    )


# ---------------------------------------------------------------------------
# Tests: _p_to_stars helper
# ---------------------------------------------------------------------------


class TestPToStars:
    """Tests for p-value to significance stars conversion."""

    def test_very_significant(self) -> None:
        """p < 0.001 → '***'."""
        assert _p_to_stars(0.0001) == "***"

    def test_significant(self) -> None:
        """p < 0.01 → '**'."""
        assert _p_to_stars(0.005) == "**"

    def test_marginally_significant(self) -> None:
        """p < 0.05 → '*'."""
        assert _p_to_stars(0.03) == "*"

    def test_not_significant(self) -> None:
        """p >= 0.05 → ''."""
        assert _p_to_stars(0.1) == ""
        assert _p_to_stars(0.05) == ""
        assert _p_to_stars(1.0) == ""

    def test_boundary_values(self) -> None:
        """Boundary p-values handled correctly."""
        assert _p_to_stars(0.001) == "**"  # exactly 0.001 → not < 0.001
        assert _p_to_stars(0.01) == "*"    # exactly 0.01 → not < 0.01


# ---------------------------------------------------------------------------
# Tests: fig_ablation_heatmap
# ---------------------------------------------------------------------------


class TestFigAblationHeatmap:
    """Smoke tests for ablation heatmap figure."""

    def test_creates_pdf(self, tmp_path: Path) -> None:
        """Produces a PDF file with non-zero size."""
        aggregated = {
            "complete": {
                "cka": _make_aggregated("complete", "cka", {"split_half": 0.9}),
            },
            "ablate_gating": {
                "cka": _make_aggregated("ablate_gating", "cka", {"split_half": 0.6}),
            },
        }
        output = tmp_path / "heatmap.pdf"
        fig_ablation_heatmap(
            aggregated, metrics=["cka.split_half"], output=output
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_creates_png(self, tmp_path: Path) -> None:
        """Produces a PNG file."""
        aggregated = {
            "variant_a": {
                "rsa": _make_aggregated("variant_a", "rsa", {"rdm_corr": 0.8}),
            },
        }
        output = tmp_path / "heatmap.png"
        fig_ablation_heatmap(
            aggregated, metrics=["rsa.rdm_corr"], output=output, fmt="png"
        )
        assert output.exists()
        assert output.stat().st_size > 0

    def test_multiple_variants_and_metrics(self, tmp_path: Path) -> None:
        """Works with multiple variants and metrics."""
        variants = ["complete", "ablate_a", "ablate_b", "ablate_c"]
        metrics_dict = {"m1": 0.9, "m2": 0.7}
        aggregated = {
            v: {"method": _make_aggregated(v, "method", metrics_dict)}
            for v in variants
        }
        output = tmp_path / "heatmap_multi.pdf"
        fig_ablation_heatmap(
            aggregated,
            metrics=["method.m1", "method.m2"],
            output=output,
        )
        assert output.exists()

    def test_missing_metric_handled(self, tmp_path: Path) -> None:
        """Missing metrics show empty cells without crashing."""
        aggregated = {
            "complete": {
                "cka": _make_aggregated("complete", "cka", {"metric_a": 0.9}),
            },
        }
        output = tmp_path / "heatmap_missing.pdf"
        # Ask for a metric that doesn't exist in the results
        fig_ablation_heatmap(
            aggregated, metrics=["cka.metric_a", "cka.nonexistent"], output=output
        )
        assert output.exists()

    def test_reference_variant_highlighted(self, tmp_path: Path) -> None:
        """Reference variant parameter doesn't crash."""
        aggregated = {
            "complete": {
                "cka": _make_aggregated("complete", "cka", {"m": 0.9}),
            },
            "ablated": {
                "cka": _make_aggregated("ablated", "cka", {"m": 0.5}),
            },
        }
        output = tmp_path / "heatmap_ref.pdf"
        fig_ablation_heatmap(
            aggregated,
            metrics=["cka.m"],
            output=output,
            reference_variant="complete",
        )
        assert output.exists()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        """Creates parent directories when they don't exist."""
        aggregated = {
            "v": {"m": _make_aggregated("v", "m", {"x": 1.0})},
        }
        output = tmp_path / "nested" / "dir" / "heatmap.pdf"
        fig_ablation_heatmap(aggregated, metrics=["m.x"], output=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Tests: fig_metric_comparison_bar
# ---------------------------------------------------------------------------


class TestFigMetricComparisonBar:
    """Smoke tests for metric comparison bar chart."""

    def test_creates_pdf(self, tmp_path: Path) -> None:
        """Produces a PDF file."""
        aggregated = {
            "complete": _make_aggregated("complete", "cka", {"split_half": 0.9}),
            "ablated": _make_aggregated("ablated", "cka", {"split_half": 0.5}),
        }
        output = tmp_path / "bar.pdf"
        fig_metric_comparison_bar(aggregated, metric="split_half", output=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_with_significance(self, tmp_path: Path) -> None:
        """Works with significance stars."""
        aggregated = {
            "complete": _make_aggregated("complete", "cka", {"m": 0.9}),
            "ablated": _make_aggregated("ablated", "cka", {"m": 0.5}),
        }
        output = tmp_path / "bar_sig.pdf"
        fig_metric_comparison_bar(
            aggregated,
            metric="m",
            output=output,
            significance={"complete": 0.5, "ablated": 0.001},
        )
        assert output.exists()

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title applied without errors."""
        aggregated = {
            "v1": _make_aggregated("v1", "m", {"metric": 0.8}),
        }
        output = tmp_path / "bar_title.pdf"
        fig_metric_comparison_bar(
            aggregated,
            metric="metric",
            output=output,
            title="Custom Title Here",
        )
        assert output.exists()

    def test_many_variants(self, tmp_path: Path) -> None:
        """Handles many variants without layout issues."""
        aggregated = {
            f"variant_{i}": _make_aggregated(f"variant_{i}", "m", {"x": float(i)})
            for i in range(10)
        }
        output = tmp_path / "bar_many.pdf"
        fig_metric_comparison_bar(aggregated, metric="x", output=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Tests: fig_effect_size_forest
# ---------------------------------------------------------------------------


class TestFigEffectSizeForest:
    """Smoke tests for effect size forest plot."""

    def test_creates_pdf(self, tmp_path: Path) -> None:
        """Produces a PDF file."""
        comparison = _make_cross_system_result(n_metrics=5)
        output = tmp_path / "forest.pdf"
        fig_effect_size_forest(comparison, output=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_creates_png(self, tmp_path: Path) -> None:
        """Produces a PNG file."""
        comparison = _make_cross_system_result(n_metrics=3)
        output = tmp_path / "forest.png"
        fig_effect_size_forest(comparison, output=output, fmt="png")
        assert output.exists()

    def test_many_metrics(self, tmp_path: Path) -> None:
        """Handles many metrics in the forest plot."""
        comparison = _make_cross_system_result(n_metrics=15)
        output = tmp_path / "forest_many.pdf"
        fig_effect_size_forest(comparison, output=output)
        assert output.exists()

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title parameter works."""
        comparison = _make_cross_system_result(n_metrics=3)
        output = tmp_path / "forest_titled.pdf"
        fig_effect_size_forest(
            comparison, output=output, title="My Custom Forest Plot"
        )
        assert output.exists()


# ---------------------------------------------------------------------------
# Tests: fig_persistence_summary
# ---------------------------------------------------------------------------


class TestFigPersistenceSummary:
    """Smoke tests for persistence summary bar chart."""

    def test_creates_pdf_with_persistence_metrics(self, tmp_path: Path) -> None:
        """Produces a figure when persistence-related metrics exist."""
        aggregated = {
            "complete": _make_aggregated(
                "complete", "ph", {"betti_0": 3.0, "betti_1": 1.5, "persistence_entropy": 2.1}
            ),
            "ablated": _make_aggregated(
                "ablated", "ph", {"betti_0": 2.0, "betti_1": 0.5, "persistence_entropy": 1.2}
            ),
        }
        output = tmp_path / "persistence.pdf"
        fig_persistence_summary(aggregated, output=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_fallback_to_all_metrics(self, tmp_path: Path) -> None:
        """Falls back to all scalar metrics when no persistence keywords found."""
        aggregated = {
            "v1": _make_aggregated("v1", "m", {"some_metric": 5.0, "other": 3.0}),
        }
        output = tmp_path / "persistence_fallback.pdf"
        fig_persistence_summary(aggregated, output=output)
        assert output.exists()

    def test_no_metrics_produces_placeholder(self, tmp_path: Path) -> None:
        """Produces a placeholder figure when no scalar metrics exist."""
        aggregated = {
            "v1": AggregatedResult(
                variant="v1",
                method="m",
                scalar_means={},
                scalar_sems={},
                scalar_cis={},
                n_seeds=5,
                seed_values={},
            ),
        }
        output = tmp_path / "persistence_empty.pdf"
        fig_persistence_summary(aggregated, output=output)
        assert output.exists()

    def test_single_variant(self, tmp_path: Path) -> None:
        """Works with a single variant."""
        aggregated = {
            "complete": _make_aggregated(
                "complete", "ph", {"betti_0": 3.0, "lifetime_mean": 0.5}
            ),
        }
        output = tmp_path / "persistence_single.pdf"
        fig_persistence_summary(aggregated, output=output)
        assert output.exists()


# ---------------------------------------------------------------------------
# Tests: fig_cross_system_comparison
# ---------------------------------------------------------------------------


class TestFigCrossSystemComparison:
    """Smoke tests for cross-system comparison figure."""

    def test_creates_pdf(self, tmp_path: Path) -> None:
        """Produces a PDF file."""
        comparison = _make_cross_system_result(n_metrics=5)
        output = tmp_path / "cross_system.pdf"
        fig_cross_system_comparison(comparison, output=output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_custom_title(self, tmp_path: Path) -> None:
        """Custom title overrides default system comparison title."""
        comparison = _make_cross_system_result(n_metrics=3)
        output = tmp_path / "cross_system_titled.pdf"
        fig_cross_system_comparison(
            comparison, output=output, title="RNN vs. Biology"
        )
        assert output.exists()

    def test_creates_svg(self, tmp_path: Path) -> None:
        """Produces an SVG file."""
        comparison = _make_cross_system_result(n_metrics=4)
        output = tmp_path / "cross_system.svg"
        fig_cross_system_comparison(comparison, output=output, fmt="svg")
        assert output.exists()
        assert output.stat().st_size > 0

    def test_many_metrics(self, tmp_path: Path) -> None:
        """Handles many metrics without layout issues."""
        comparison = _make_cross_system_result(n_metrics=12)
        output = tmp_path / "cross_system_many.pdf"
        fig_cross_system_comparison(comparison, output=output)
        assert output.exists()
