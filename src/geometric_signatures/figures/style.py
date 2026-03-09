"""Matplotlib style presets for publication-quality figures.

Provides rcParams presets for different presentation media:
- **paper**: Nature/Science-style formatting (small, high-DPI, serif)
- **poster**: Large fonts, high contrast, bold lines
- **presentation**: Dark background, large elements, sans-serif
"""

from __future__ import annotations

from typing import Any

import matplotlib as mpl
from cycler import cycler

STYLE_PRESETS: dict[str, dict[str, Any]] = {
    "paper": {
        # Figure dimensions for single-column Nature figure
        "figure.figsize": (3.5, 2.625),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Fonts
        "font.family": "serif",
        "font.size": 7,
        "axes.titlesize": 8,
        "axes.labelsize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        # Lines and markers
        "lines.linewidth": 1.0,
        "lines.markersize": 3,
        "axes.linewidth": 0.5,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.3,
        "ytick.minor.width": 0.3,
        # Colors
        "axes.prop_cycle": cycler(
            color=[
                "#0072B2",  # Blue
                "#D55E00",  # Vermillion
                "#009E73",  # Bluish green
                "#CC79A7",  # Reddish purple
                "#F0E442",  # Yellow
                "#56B4E9",  # Sky blue
                "#E69F00",  # Orange
                "#000000",  # Black
            ]
        ),
        # Grid and spines
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Legend
        "legend.frameon": False,
        "legend.borderaxespad": 0.5,
    },
    "poster": {
        "figure.figsize": (10, 7.5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        # Fonts
        "font.family": "sans-serif",
        "font.size": 18,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        # Lines and markers
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.linewidth": 1.5,
        "xtick.major.width": 1.5,
        "ytick.major.width": 1.5,
        # Colors
        "axes.prop_cycle": cycler(
            color=[
                "#0072B2",
                "#D55E00",
                "#009E73",
                "#CC79A7",
                "#F0E442",
                "#56B4E9",
                "#E69F00",
                "#000000",
            ]
        ),
        # Grid and spines
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    },
    "presentation": {
        "figure.figsize": (12, 8),
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#2E2E2E",
        # Fonts
        "font.family": "sans-serif",
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        # Dark theme
        "figure.facecolor": "#2E2E2E",
        "axes.facecolor": "#2E2E2E",
        "axes.edgecolor": "#CCCCCC",
        "axes.labelcolor": "#FFFFFF",
        "xtick.color": "#CCCCCC",
        "ytick.color": "#CCCCCC",
        "text.color": "#FFFFFF",
        # Lines and markers
        "lines.linewidth": 2.5,
        "lines.markersize": 8,
        "axes.linewidth": 1.0,
        # Colors — brighter palette for dark background
        "axes.prop_cycle": cycler(
            color=[
                "#56B4E9",  # Sky blue
                "#E69F00",  # Orange
                "#009E73",  # Bluish green
                "#F0E442",  # Yellow
                "#CC79A7",  # Reddish purple
                "#0072B2",  # Blue
                "#D55E00",  # Vermillion
                "#FFFFFF",  # White
            ]
        ),
        # Grid and spines
        "axes.grid": True,
        "grid.color": "#444444",
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "legend.facecolor": "#2E2E2E",
        "legend.edgecolor": "#444444",
    },
}


def apply_style(preset: str = "paper") -> None:
    """Apply a style preset to matplotlib rcParams.

    Args:
        preset: Style preset name — "paper", "poster", or "presentation".

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset not in STYLE_PRESETS:
        raise ValueError(
            f"Unknown style preset: {preset!r}. "
            f"Available: {list(STYLE_PRESETS.keys())}"
        )

    # Reset to defaults first for clean slate
    mpl.rcdefaults()

    # Apply preset
    mpl.rcParams.update(STYLE_PRESETS[preset])
