"""Publication-quality figure generation.

Provides typed figure generators for geometric signature analysis results,
with style presets for papers, posters, and presentations.

Typical workflow::

    from geometric_signatures.figures import apply_style, fig_ablation_heatmap
    apply_style("paper")
    fig_ablation_heatmap(aggregated_results, output=Path("figures/heatmap.pdf"))

Style presets::

    apply_style("paper")         # Nature/Science formatting
    apply_style("poster")        # Large fonts, high contrast
    apply_style("presentation")  # Dark background, bold colors
"""

from .plotting import (
    fig_ablation_heatmap,
    fig_cross_system_comparison,
    fig_effect_size_forest,
    fig_metric_comparison_bar,
    fig_persistence_summary,
)
from .style import STYLE_PRESETS, apply_style

__all__ = [
    "STYLE_PRESETS",
    "apply_style",
    "fig_ablation_heatmap",
    "fig_cross_system_comparison",
    "fig_effect_size_forest",
    "fig_metric_comparison_bar",
    "fig_persistence_summary",
]
