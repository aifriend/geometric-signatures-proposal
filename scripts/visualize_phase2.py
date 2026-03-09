#!/usr/bin/env python3
"""Phase 2 analysis visualization — publication-quality figures."""
import json
import os
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec

# ── Config ──────────────────────────────────────────────────────────────
RUNS_DIR = "runs/phase2"
COMPARISONS_FILE = os.path.join(RUNS_DIR, "comparisons.json")
OUTPUT_DIR = "figures/phase2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Variant display names (short labels for plots)
VARIANT_LABELS = {
    "ablate_normalization_gain_modulation": "– Normalization",
    "ablate_expansion_recoding": "– Expansion",
    "ablate_attractor_dynamics": "– Attractor",
    "ablate_selective_gating": "– Gating",
    "complete": "Complete",
}

VARIANT_ORDER = [
    "ablate_normalization_gain_modulation",
    "ablate_expansion_recoding",
    "ablate_attractor_dynamics",
    "ablate_selective_gating",
]

VARIANT_COLORS = {
    "ablate_normalization_gain_modulation": "#d62728",
    "ablate_expansion_recoding": "#ff7f0e",
    "ablate_attractor_dynamics": "#f0c75e",
    "ablate_selective_gating": "#2ca02c",
    "complete": "#1f77b4",
}

METHOD_COLORS = {
    "persistent_homology": "#9467bd",
    "rsa": "#e377c2",
    "cka": "#17becf",
    "population_geometry": "#8c564b",
}

# Map long variant dir names to where analysis data lives
ANALYSIS_DIRS = {
    "complete": "complete",
    "ablate_normalization_gain_modulation": "ablate_normalization_gain_modulation",
    "ablate_attractor_dynamics": "ablate_attractor_dynamics",
    "ablate_selective_gating": "ablate_selective_gating",
    "ablate_expansion_recoding": "ablate_expansion_recoding",
}


def load_comparisons():
    with open(COMPARISONS_FILE) as f:
        return json.load(f)


def load_seed_scalars(variant_dir, method):
    """Load per-seed scalar values for a given variant and method."""
    pattern = os.path.join(RUNS_DIR, variant_dir, "seed_*", "analysis", f"{method}.json")
    files = sorted(glob.glob(pattern))
    all_scalars = []
    for fpath in files:
        with open(fpath) as f:
            data = json.load(f)
        all_scalars.append(data["scalars"])
    return all_scalars


# ══════════════════════════════════════════════════════════════════════════
# Figure 1: Significance Overview — horizontal bar chart
# ══════════════════════════════════════════════════════════════════════════
def fig1_significance_bars(comparisons):
    fig, ax = plt.subplots(figsize=(8, 4))

    variants = VARIANT_ORDER
    labels = [VARIANT_LABELS[v] for v in variants]
    total = 31

    sig_005 = []
    sig_001 = []
    for v in variants:
        tests = comparisons[v]
        s005 = sum(1 for t in tests.values() if t["p_value"] < 0.05)
        s001 = sum(1 for t in tests.values() if t["p_value"] < 0.01)
        sig_005.append(s005)
        sig_001.append(s001)

    y = np.arange(len(variants))
    colors = [VARIANT_COLORS[v] for v in variants]

    bars_all = ax.barh(y, sig_005, height=0.5, color=colors, alpha=0.4,
                       edgecolor=[c for c in colors], linewidth=1.5, label="p < 0.05")
    bars_strong = ax.barh(y, sig_001, height=0.5, color=colors, alpha=0.9,
                          edgecolor=[c for c in colors], linewidth=1.5, label="p < 0.01")

    # Annotations
    for i, (s5, s1) in enumerate(zip(sig_005, sig_001)):
        ax.text(s5 + 0.5, i, f"{s5}/31", va="center", fontsize=10, fontweight="bold",
                color=colors[i])

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Number of significant metrics", fontsize=12)
    ax.set_title("Geometric Impact of Motif Ablations", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 32)
    ax.axvline(x=total, color="gray", linestyle="--", alpha=0.3, label=f"Total ({total})")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig1_significance_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 2: P-value heatmap across all metrics × variants
# ══════════════════════════════════════════════════════════════════════════
def fig2_pvalue_heatmap(comparisons):
    variants = VARIANT_ORDER
    # Get all metrics, grouped by method
    all_metrics = sorted(comparisons[variants[0]].keys())

    # Group metrics by method for visual separation
    methods_order = ["persistent_homology", "rsa", "cka", "population_geometry"]
    grouped_metrics = []
    for method in methods_order:
        method_metrics = [m for m in all_metrics if m.startswith(method + ".")]
        grouped_metrics.extend(method_metrics)

    # Build p-value matrix
    n_metrics = len(grouped_metrics)
    n_variants = len(variants)
    pmat = np.ones((n_metrics, n_variants))

    for j, v in enumerate(variants):
        for i, m in enumerate(grouped_metrics):
            if m in comparisons[v]:
                pmat[i, j] = comparisons[v][m]["p_value"]

    # Transform: -log10(p) for better visualization
    log_pmat = -np.log10(np.clip(pmat, 1e-10, 1.0))

    fig, ax = plt.subplots(figsize=(7, 12))

    # Custom colormap: white → yellow → orange → red
    cmap = LinearSegmentedColormap.from_list("sig",
        [(0, "#f7f7f7"), (0.3, "#fee08b"), (0.6, "#fc8d59"), (1.0, "#d73027")])

    im = ax.imshow(log_pmat, aspect="auto", cmap=cmap, vmin=0, vmax=3.5)

    # Y-axis: metric names (shortened)
    short_names = []
    for m in grouped_metrics:
        parts = m.split(".", 1)
        short_names.append(parts[1] if len(parts) > 1 else m)

    ax.set_yticks(range(n_metrics))
    ax.set_yticklabels(short_names, fontsize=8)

    ax.set_xticks(range(n_variants))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in variants], fontsize=10, rotation=30, ha="right")

    # Add significance markers
    for i in range(n_metrics):
        for j in range(n_variants):
            p = pmat[i, j]
            if p < 0.001:
                ax.text(j, i, "***", ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            elif p < 0.01:
                ax.text(j, i, "**", ha="center", va="center", fontsize=7, fontweight="bold", color="white")
            elif p < 0.05:
                ax.text(j, i, "*", ha="center", va="center", fontsize=7, fontweight="bold", color="black")

    # Method group separators
    cumulative = 0
    for method in methods_order[:-1]:
        count = sum(1 for m in grouped_metrics if m.startswith(method + "."))
        cumulative += count
        ax.axhline(y=cumulative - 0.5, color="black", linewidth=0.8, alpha=0.5)

    # Method labels on the right
    cumulative = 0
    for method in methods_order:
        count = sum(1 for m in grouped_metrics if m.startswith(method + "."))
        mid = cumulative + count / 2 - 0.5
        ax.text(n_variants + 0.1, mid, method.replace("_", "\n"),
                fontsize=7, va="center", ha="left", color=METHOD_COLORS[method],
                fontweight="bold")
        cumulative += count

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.15)
    cbar.set_label("$-\\log_{10}(p)$", fontsize=11)
    # Add reference lines to colorbar
    for pval, label in [(0.05, "p=0.05"), (0.01, "p=0.01"), (0.001, "p=0.001")]:
        cbar.ax.axhline(y=-np.log10(pval), color="black", linewidth=0.8, linestyle="--")
        cbar.ax.text(1.6, -np.log10(pval), label, fontsize=7, va="center")

    ax.set_title("Statistical Significance Heatmap\n(permutation tests, 1000 permutations)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_pvalue_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 3: Key metrics box plots — per-seed distributions
# ══════════════════════════════════════════════════════════════════════════
def fig3_metric_distributions():
    """Box plots for key metrics showing per-seed distributions across variants."""
    key_metrics = {
        "persistent_homology": [
            ("H0_total_persistence", "H0 Total Persistence"),
            ("H1_betti", "H1 Betti Number"),
        ],
        "rsa": [
            ("mean_dissimilarity", "Mean RSA Dissimilarity"),
        ],
        "cka": [
            ("mean_self_cka", "Mean Self-CKA"),
        ],
        "population_geometry": [
            ("effective_dimensionality", "Effective Dimensionality"),
            ("participation_ratio", "Participation Ratio"),
            ("mean_trajectory_speed", "Mean Trajectory Speed"),
            ("condition_separability", "Condition Separability"),
        ],
    }

    all_variants = ["complete"] + VARIANT_ORDER
    n_plots = sum(len(v) for v in key_metrics.values())
    ncols = 2
    nrows = (n_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows))
    axes = axes.flatten()

    plot_idx = 0
    for method, metrics in key_metrics.items():
        # Load per-seed data for each variant
        variant_data = {}
        for v in all_variants:
            vdir = ANALYSIS_DIRS[v]
            scalars = load_seed_scalars(vdir, method)
            if scalars:
                variant_data[v] = scalars

        for metric_key, metric_label in metrics:
            ax = axes[plot_idx]
            positions = []
            data_list = []
            colors_list = []
            labels_list = []

            for i, v in enumerate(all_variants):
                if v in variant_data:
                    values = [s.get(metric_key) for s in variant_data[v]
                              if metric_key in s and s[metric_key] is not None
                              and not (isinstance(s[metric_key], float) and np.isnan(s[metric_key]))]
                    if values:
                        positions.append(i)
                        data_list.append(values)
                        colors_list.append(VARIANT_COLORS[v])
                        labels_list.append(VARIANT_LABELS[v])

            if data_list:
                bp = ax.boxplot(data_list, positions=positions, widths=0.6, patch_artist=True,
                                showfliers=True, flierprops=dict(marker="o", markersize=4))
                for patch, color in zip(bp["boxes"], colors_list):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.6)
                for median in bp["medians"]:
                    median.set_color("black")
                    median.set_linewidth(1.5)

                # Overlay individual points
                for pos, vals, color in zip(positions, data_list, colors_list):
                    jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
                    ax.scatter([pos + j for j in jitter], vals, c=color, s=20,
                              alpha=0.7, edgecolors="white", linewidths=0.5, zorder=3)

            ax.set_xticks(range(len(all_variants)))
            ax.set_xticklabels([VARIANT_LABELS[v] for v in all_variants],
                               fontsize=8, rotation=25, ha="right")
            ax.set_title(metric_label, fontsize=11, fontweight="bold")
            ax.set_ylabel("Value", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

            plot_idx += 1

    # Hide unused axes
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Per-Seed Metric Distributions Across Ablation Variants",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_metric_distributions.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 4: Method-level radar chart — fraction of significant metrics
# ══════════════════════════════════════════════════════════════════════════
def fig4_method_radar(comparisons):
    methods = ["persistent_homology", "rsa", "cka", "population_geometry"]
    method_labels = ["Persistent\nHomology", "RSA", "CKA", "Population\nGeometry"]
    variants = VARIANT_ORDER

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), subplot_kw=dict(projection="polar"))

    for ax_idx, variant in enumerate(variants):
        ax = axes[ax_idx]
        fractions = []
        for method in methods:
            method_metrics = {k: v for k, v in comparisons[variant].items()
                              if k.startswith(method + ".")}
            total = len(method_metrics)
            sig = sum(1 for t in method_metrics.values() if t["p_value"] < 0.05)
            fractions.append(sig / total if total > 0 else 0)

        # Close the radar
        angles = np.linspace(0, 2 * np.pi, len(methods), endpoint=False).tolist()
        fractions_closed = fractions + [fractions[0]]
        angles_closed = angles + [angles[0]]

        ax.fill(angles_closed, fractions_closed, alpha=0.25, color=VARIANT_COLORS[variant])
        ax.plot(angles_closed, fractions_closed, "o-", color=VARIANT_COLORS[variant],
                linewidth=2, markersize=6)

        ax.set_xticks(angles)
        ax.set_xticklabels(method_labels, fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7, color="gray")
        ax.set_title(VARIANT_LABELS[variant], fontsize=11, fontweight="bold",
                     color=VARIANT_COLORS[variant], pad=15)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Fraction of Significant Metrics by Analysis Method",
                 fontsize=14, fontweight="bold", y=1.05)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_method_radar.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 5: Effect size heatmap — standardized differences
# ══════════════════════════════════════════════════════════════════════════
def fig5_effect_sizes():
    """Compute Cohen's d from per-seed data for key metrics."""
    methods = ["persistent_homology", "rsa", "cka", "population_geometry"]
    all_variants = ["complete"] + VARIANT_ORDER

    # Load all per-seed data
    variant_method_data = {}
    for v in all_variants:
        vdir = ANALYSIS_DIRS[v]
        variant_method_data[v] = {}
        for method in methods:
            scalars = load_seed_scalars(vdir, method)
            if scalars:
                variant_method_data[v][method] = scalars

    # Get all scalar keys from complete variant
    all_metrics = []
    for method in methods:
        if method in variant_method_data["complete"]:
            keys = sorted(variant_method_data["complete"][method][0].keys())
            for k in keys:
                all_metrics.append((method, k))

    # Compute Cohen's d for each ablation vs complete
    effect_matrix = np.full((len(all_metrics), len(VARIANT_ORDER)), np.nan)

    for j, variant in enumerate(VARIANT_ORDER):
        for i, (method, key) in enumerate(all_metrics):
            # Get complete values
            if method not in variant_method_data.get("complete", {}):
                continue
            complete_vals = [s.get(key) for s in variant_method_data["complete"][method]
                             if key in s and s[key] is not None
                             and not (isinstance(s[key], float) and np.isnan(s[key]))]
            # Get ablation values
            if method not in variant_method_data.get(variant, {}):
                continue
            ablation_vals = [s.get(key) for s in variant_method_data[variant][method]
                             if key in s and s[key] is not None
                             and not (isinstance(s[key], float) and np.isnan(s[key]))]

            if len(complete_vals) >= 3 and len(ablation_vals) >= 3:
                c_arr = np.array(complete_vals, dtype=float)
                a_arr = np.array(ablation_vals, dtype=float)
                pooled_std = np.sqrt((np.var(c_arr, ddof=1) + np.var(a_arr, ddof=1)) / 2)
                if pooled_std > 1e-12:
                    cohen_d = (np.mean(a_arr) - np.mean(c_arr)) / pooled_std
                    effect_matrix[i, j] = cohen_d

    # Plot
    fig, ax = plt.subplots(figsize=(7, 12))

    # Diverging colormap centered at 0
    vmax = np.nanmax(np.abs(effect_matrix))
    vmax = min(vmax, 5)  # Cap for readability

    cmap = plt.cm.RdBu_r
    im = ax.imshow(np.clip(effect_matrix, -vmax, vmax), aspect="auto",
                   cmap=cmap, vmin=-vmax, vmax=vmax)

    # Labels
    metric_labels = [f"{m}.{k}" for m, k in all_metrics]
    short_labels = [k for _, k in all_metrics]
    ax.set_yticks(range(len(all_metrics)))
    ax.set_yticklabels(short_labels, fontsize=8)
    ax.set_xticks(range(len(VARIANT_ORDER)))
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER],
                       fontsize=10, rotation=30, ha="right")

    # Method group separators
    cumulative = 0
    for method in methods:
        count = sum(1 for m, _ in all_metrics if m == method)
        if cumulative > 0:
            ax.axhline(y=cumulative - 0.5, color="black", linewidth=0.8)
        mid = cumulative + count / 2 - 0.5
        ax.text(len(VARIANT_ORDER) + 0.1, mid, method.replace("_", "\n"),
                fontsize=7, va="center", ha="left", color=METHOD_COLORS[method],
                fontweight="bold")
        cumulative += count

    cbar = plt.colorbar(im, ax=ax, shrink=0.5, pad=0.15)
    cbar.set_label("Cohen's d (effect size)", fontsize=11)

    ax.set_title("Effect Sizes: Ablation vs Complete\n(Cohen's d, per-seed distributions)",
                 fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig5_effect_sizes.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Figure 6: Summary panel — combined overview
# ══════════════════════════════════════════════════════════════════════════
def fig6_summary_panel(comparisons):
    """Combined 2x2 summary panel for presentations."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    variants = VARIANT_ORDER
    methods = ["persistent_homology", "rsa", "cka", "population_geometry"]
    method_labels = ["Pers. Homology", "RSA", "CKA", "Pop. Geometry"]

    # ── Panel A: Significance count bars ──
    ax1 = fig.add_subplot(gs[0, 0])
    sig_counts = []
    for v in variants:
        sig_counts.append(sum(1 for t in comparisons[v].values() if t["p_value"] < 0.05))

    colors = [VARIANT_COLORS[v] for v in variants]
    labels = [VARIANT_LABELS[v] for v in variants]
    bars = ax1.bar(range(len(variants)), sig_counts, color=colors, edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, sig_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha="center", fontweight="bold", fontsize=11)
    ax1.set_xticks(range(len(variants)))
    ax1.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax1.set_ylabel("Significant metrics (p < 0.05)", fontsize=10)
    ax1.set_ylim(0, 32)
    ax1.axhline(y=31, color="gray", linestyle="--", alpha=0.3)
    ax1.set_title("A) Geometric Disruption per Ablation", fontsize=12, fontweight="bold")
    ax1.grid(axis="y", alpha=0.2)

    # ── Panel B: Method breakdown (stacked bar) ──
    ax2 = fig.add_subplot(gs[0, 1])
    bottom = np.zeros(len(variants))
    for method, mlabel in zip(methods, method_labels):
        counts = []
        for v in variants:
            method_metrics = {k: t for k, t in comparisons[v].items()
                              if k.startswith(method + ".")}
            sig = sum(1 for t in method_metrics.values() if t["p_value"] < 0.05)
            counts.append(sig)
        ax2.bar(range(len(variants)), counts, bottom=bottom, label=mlabel,
                color=METHOD_COLORS[method], alpha=0.8, edgecolor="white", linewidth=0.5)
        bottom += np.array(counts)

    ax2.set_xticks(range(len(variants)))
    ax2.set_xticklabels(labels, fontsize=9, rotation=20, ha="right")
    ax2.set_ylabel("Significant metrics (p < 0.05)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.set_title("B) Disruption by Analysis Method", fontsize=12, fontweight="bold")
    ax2.grid(axis="y", alpha=0.2)

    # ── Panel C: Min p-value per method (shows strongest signal) ──
    ax3 = fig.add_subplot(gs[1, 0])
    x = np.arange(len(methods))
    width = 0.18
    for i, v in enumerate(variants):
        min_pvals = []
        for method in methods:
            method_metrics = {k: t for k, t in comparisons[v].items()
                              if k.startswith(method + ".")}
            if method_metrics:
                min_p = min(t["p_value"] for t in method_metrics.values())
            else:
                min_p = 1.0
            min_pvals.append(-np.log10(max(min_p, 0.0005)))

        offset = (i - len(variants)/2 + 0.5) * width
        ax3.bar(x + offset, min_pvals, width, color=VARIANT_COLORS[v],
                label=VARIANT_LABELS[v], alpha=0.85, edgecolor="white")

    ax3.axhline(y=-np.log10(0.05), color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax3.text(len(methods) - 0.5, -np.log10(0.05) + 0.05, "p = 0.05", fontsize=8, color="red")
    ax3.axhline(y=-np.log10(0.01), color="darkred", linestyle=":", alpha=0.5, linewidth=1)
    ax3.text(len(methods) - 0.5, -np.log10(0.01) + 0.05, "p = 0.01", fontsize=8, color="darkred")

    ax3.set_xticks(x)
    ax3.set_xticklabels(method_labels, fontsize=9)
    ax3.set_ylabel("$-\\log_{10}(p_{min})$", fontsize=10)
    ax3.legend(fontsize=7, loc="upper left", ncol=2)
    ax3.set_title("C) Strongest Signal per Method", fontsize=12, fontweight="bold")
    ax3.grid(axis="y", alpha=0.2)

    # ── Panel D: Motif impact ranking summary ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    summary_text = (
        "Motif Impact Ranking\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    rankings = [
        ("1. Normalization", "24/31 sig.", "Most critical motif.\n"
         "   Affects ALL analysis methods.\n"
         "   SVD fails in 30% of seeds."),
        ("2. Expansion", "14/31 sig.", "Strong impact on RSA, CKA,\n"
         "   dimensionality, trajectories."),
        ("3. Attractor", " 6/31 sig.", "Moderate impact on topology\n"
         "   and trajectory dynamics."),
        ("4. Gating", " 3/31 sig.", "Minimal geometric impact.\n"
         "   Other motifs compensate."),
    ]

    y_pos = 0.85
    for rank, count, desc in rankings:
        variant_key = VARIANT_ORDER[rankings.index((rank, count, desc))]
        color = VARIANT_COLORS[variant_key]
        ax4.text(0.05, y_pos, rank, fontsize=12, fontweight="bold", color=color,
                 transform=ax4.transAxes, va="top")
        ax4.text(0.55, y_pos, count, fontsize=11, fontweight="bold",
                 transform=ax4.transAxes, va="top", color="black")
        ax4.text(0.05, y_pos - 0.06, desc, fontsize=8, color="gray",
                 transform=ax4.transAxes, va="top", family="monospace")
        y_pos -= 0.22

    ax4.set_title("D) Impact Summary", fontsize=12, fontweight="bold")

    fig.suptitle("Phase 2: Geometric Signatures of Computational Motifs — Analysis Results",
                 fontsize=15, fontweight="bold", y=0.98)

    path = os.path.join(OUTPUT_DIR, "fig6_summary_panel.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading comparisons...")
    comparisons = load_comparisons()

    print("\nGenerating figures...\n")
    fig1_significance_bars(comparisons)
    fig2_pvalue_heatmap(comparisons)
    fig3_metric_distributions()
    fig4_method_radar(comparisons)
    fig5_effect_sizes()
    fig6_summary_panel(comparisons)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
