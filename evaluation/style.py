"""Shared figure styling.

Colours come from a palette validated for colour-vision deficiency and contrast
against the light chart surface (worst all-pairs CVD dE 9.2, normal-vision 24.0).
The categorical order is fixed: series 1, then 2, then 3, never cycled. Past three
series we facet instead of inventing a fourth hue.
"""

import matplotlib as mpl

# Chart chrome
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical series, in fixed order.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")

# Status palette -- reserved for state, never reused as a series colour.
STATUS_WARNING = "#fab219"
STATUS_CRITICAL = "#d03b3b"
STATUS_GOOD = "#0ca30c"


def apply() -> None:
    """Recessive chrome, thin marks, generous padding."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "semibold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Solid hairline grid, one shade off the surface. Never dashed:
            # dashing reads as "projection" or "threshold" when it is just a grid.
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 6.0,
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
        }
    )
