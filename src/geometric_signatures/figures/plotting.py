"""Typed figure generators for geometric signature analysis.

Each function takes analysis results and produces a publication-quality
figure with error bars and significance annotations from multi-seed data.

All functions follow the pattern:
    fig_<name>(data, output, fmt="pdf") → None

They write directly to disk and use the currently active matplotlib style.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from ..comparison.cross_system import CrossSystemResult
from ..statistics.aggregation import AggregatedResult


def fig_ablation_heatmap(
    aggregated: dict[str, dict[str, AggregatedResult]],
    metrics: Sequence[str],
    output: Path,
    fmt: str = "pdf",
    title: str = "Motif Ablation Effect Sizes",
    reference_variant: str = "complete",
) -> None:
    """Heatmap of metric values across variants and analysis methods.

    Rows = variants, columns = metrics. Cell color = mean value; text
    shows mean ± SEM. The reference variant row is highlighted.

    Args:
        aggregated: Nested dict: {variant: {method: AggregatedResult}}.
        metrics: Metric keys to display (e.g., "cka.split_half_cka").
        output: File path for the figure.
        fmt: Image format ("pdf", "png", "svg").
        title: Figure title.
        reference_variant: Variant to highlight as reference.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    variants = sorted(aggregated.keys())
    n_variants = len(variants)
    n_metrics = len(metrics)

    # Build data matrix
    data = np.full((n_variants, n_metrics), np.nan)
    annotations: list[list[str]] = []

    for i, variant in enumerate(variants):
        row_annot: list[str] = []
        for j, metric in enumerate(metrics):
            # Parse "method.metric_name"
            parts = metric.split(".", 1)
            if len(parts) == 2:
                method, metric_name = parts
            else:
                method, metric_name = metric, metric

            agg = aggregated.get(variant, {}).get(method)
            if agg is not None and metric_name in agg.scalar_means:
                mean = agg.scalar_means[metric_name]
                sem = agg.scalar_sems.get(metric_name, 0.0)
                data[i, j] = mean
                row_annot.append(f"{mean:.2f}\n±{sem:.2f}")
            else:
                row_annot.append("")
        annotations.append(row_annot)

    fig, ax = plt.subplots(figsize=(max(4, n_metrics * 1.5), max(3, n_variants * 0.8)))

    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r")

    # Annotate cells
    for i in range(n_variants):
        for j in range(n_metrics):
            if annotations[i][j]:
                ax.text(
                    j, i, annotations[i][j],
                    ha="center", va="center", fontsize=6,
                )

    # Axis labels
    short_metrics = [m.split(".")[-1] if "." in m else m for m in metrics]
    ax.set_xticks(range(n_metrics))
    ax.set_xticklabels(short_metrics, rotation=45, ha="right")
    ax.set_yticks(range(n_variants))
    ax.set_yticklabels(variants)

    # Highlight reference variant
    if reference_variant in variants:
        ref_idx = variants.index(reference_variant)
        ax.axhline(ref_idx - 0.5, color="black", linewidth=2)
        ax.axhline(ref_idx + 0.5, color="black", linewidth=2)

    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)

    fig.savefig(output, format=fmt)
    plt.close(fig)


def fig_metric_comparison_bar(
    aggregated: dict[str, AggregatedResult],
    metric: str,
    output: Path,
    fmt: str = "pdf",
    title: str | None = None,
    significance: dict[str, float] | None = None,
) -> None:
    """Bar chart comparing a single metric across variants.

    Bars show mean ± SEM with optional significance stars.

    Args:
        aggregated: {variant: AggregatedResult} for one method.
        metric: Metric key to plot (must be in scalar_means).
        output: File path for the figure.
        fmt: Image format.
        title: Figure title (defaults to metric name).
        significance: Optional {variant: p_value} for significance stars.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    variants = sorted(aggregated.keys())
    means = []
    sems = []

    for v in variants:
        agg = aggregated[v]
        means.append(agg.scalar_means.get(metric, 0.0))
        sems.append(agg.scalar_sems.get(metric, 0.0))

    x = np.arange(len(variants))

    fig, ax = plt.subplots()
    bars = ax.bar(x, means, yerr=sems, capsize=4, alpha=0.8, edgecolor="black")

    # Add significance stars
    if significance:
        y_max = max(m + s for m, s in zip(means, sems))
        for i, v in enumerate(variants):
            p = significance.get(v)
            if p is not None:
                stars = _p_to_stars(p)
                if stars:
                    ax.text(
                        i, y_max * 1.05, stars,
                        ha="center", va="bottom", fontsize=10,
                    )

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(title or metric)

    fig.savefig(output, format=fmt)
    plt.close(fig)


def fig_effect_size_forest(
    comparison: CrossSystemResult,
    output: Path,
    fmt: str = "pdf",
    title: str = "Effect Sizes (Cohen's d)",
) -> None:
    """Forest plot of effect sizes across metrics.

    Horizontal bars show Cohen's d for each metric, with a vertical
    line at d=0. Significant metrics are colored differently.

    Args:
        comparison: CrossSystemResult from compare_across_systems.
        output: File path for the figure.
        fmt: Image format.
        title: Figure title.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = sorted(comparison.effect_sizes.keys())
    effect_sizes = [comparison.effect_sizes[m] for m in metrics]
    p_values = [comparison.significance[m].p_value for m in metrics]

    n = len(metrics)
    y_pos = np.arange(n)

    fig, ax = plt.subplots(figsize=(6, max(3, n * 0.4)))

    colors = ["#D55E00" if p < 0.05 else "#0072B2" for p in p_values]
    ax.barh(y_pos, effect_sizes, color=colors, height=0.6, alpha=0.8)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Cohen's d")
    ax.set_title(title)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#D55E00", alpha=0.8, label="p < 0.05"),
        Patch(facecolor="#0072B2", alpha=0.8, label="p ≥ 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    fig.savefig(output, format=fmt)
    plt.close(fig)


def fig_persistence_summary(
    aggregated: dict[str, AggregatedResult],
    output: Path,
    fmt: str = "pdf",
    title: str = "Persistent Homology Summary",
) -> None:
    """Bar chart of topological summary statistics across variants.

    Plots Betti numbers or persistence entropy from persistent_homology
    analysis results, with error bars from multi-seed aggregation.

    Args:
        aggregated: {variant: AggregatedResult} with persistent_homology
            method results.
        output: File path for the figure.
        fmt: Image format.
        title: Figure title.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    variants = sorted(aggregated.keys())

    # Find all persistence-related scalar metrics
    all_metrics: set[str] = set()
    for agg in aggregated.values():
        all_metrics.update(agg.scalar_means.keys())

    persistence_metrics = sorted(
        m for m in all_metrics
        if any(
            keyword in m.lower()
            for keyword in ("betti", "persistence", "lifetime", "entropy")
        )
    )

    if not persistence_metrics:
        # Fallback: use all scalar metrics
        persistence_metrics = sorted(all_metrics)

    n_variants = len(variants)
    n_metrics = len(persistence_metrics)

    if n_metrics == 0:
        # Nothing to plot
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No persistence metrics found", transform=ax.transAxes,
                ha="center", va="center")
        fig.savefig(output, format=fmt)
        plt.close(fig)
        return

    x = np.arange(n_metrics)
    width = 0.8 / n_variants

    fig, ax = plt.subplots(figsize=(max(6, n_metrics * 2), 4))

    for i, variant in enumerate(variants):
        agg = aggregated[variant]
        means = [agg.scalar_means.get(m, 0.0) for m in persistence_metrics]
        sems = [agg.scalar_sems.get(m, 0.0) for m in persistence_metrics]
        offset = (i - n_variants / 2 + 0.5) * width
        ax.bar(
            x + offset, means, width, yerr=sems,
            label=variant, capsize=3, alpha=0.8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(persistence_metrics, rotation=45, ha="right")
    ax.set_title(title)
    ax.legend()

    fig.savefig(output, format=fmt)
    plt.close(fig)


def fig_cross_system_comparison(
    comparison: CrossSystemResult,
    output: Path,
    fmt: str = "pdf",
    title: str | None = None,
) -> None:
    """Summary figure of cross-system comparison.

    Combines similarity scores and significance in a single figure:
    bars show similarity (1 - |d|), colored by significance.

    Args:
        comparison: CrossSystemResult from compare_across_systems.
        output: File path for the figure.
        fmt: Image format.
        title: Figure title.
    """
    output.parent.mkdir(parents=True, exist_ok=True)

    metrics = sorted(comparison.metric_similarities.keys())
    similarities = [comparison.metric_similarities[m] for m in metrics]
    p_values = [comparison.significance[m].p_value for m in metrics]

    n = len(metrics)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), 4))

    colors = [
        "#009E73" if p >= 0.05 else "#D55E00"
        for p in p_values
    ]
    bars = ax.bar(x, similarities, color=colors, alpha=0.8, edgecolor="black")

    # Add significance stars above bars
    for i, (sim, p) in enumerate(zip(similarities, p_values)):
        stars = _p_to_stars(p)
        if stars:
            ax.text(i, sim + 0.02, stars, ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylabel("Similarity (1 − |d|)")
    ax.set_ylim(0, 1.15)

    default_title = f"{comparison.system_a} vs. {comparison.system_b}"
    ax.set_title(title or default_title)

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#009E73", alpha=0.8, label="Shared (p ≥ 0.05)"),
        Patch(facecolor="#D55E00", alpha=0.8, label="Divergent (p < 0.05)"),
    ]
    ax.legend(handles=legend_elements)

    fig.savefig(output, format=fmt)
    plt.close(fig)


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return ""
