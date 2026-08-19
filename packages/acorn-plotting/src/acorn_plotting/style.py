from __future__ import annotations

import matplotlib


ACORN_RC: dict = {
    "figure.facecolor":     "white",
    "axes.facecolor":       "white",
    "axes.edgecolor":       "#000000",
    "axes.linewidth":       0.75,
    "axes.labelsize":       11,
    "axes.titlesize":       11,
    "axes.titlepad":        8,
    "axes.labelpad":        6,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "axes.grid":            False,
    "xtick.direction":      "out",
    "ytick.direction":      "out",
    "xtick.color":          "#000000",
    "ytick.color":          "#000000",
    "xtick.labelsize":      9,
    "ytick.labelsize":      9,
    "xtick.major.size":     4.0,
    "ytick.major.size":     4.0,
    "xtick.major.width":    0.75,
    "ytick.major.width":    0.75,
    "xtick.minor.size":     2.0,
    "ytick.minor.size":     2.0,
    "xtick.minor.width":    0.5,
    "ytick.minor.width":    0.5,
    "lines.linewidth":      1.0,
    "patch.linewidth":      0.75,
    "font.family":          "sans-serif",
    "font.sans-serif":      ["Arial", "Helvetica Neue", "Helvetica", "DejaVu Sans"],
    "font.size":            10,
    "legend.fontsize":      9,
    "legend.frameon":       False,
    "legend.borderpad":     0.4,
    "savefig.dpi":          300,
    "savefig.bbox":         "tight",
}

PALETTE = [
    "#2E75B6",
    "#C00000",
    "#70AD47",
    "#ED7D31",
    "#7030A0",
    "#00B0F0",
    "#FFC000",
    "#FF0000",
]


def apply_acorn_style() -> None:
    matplotlib.rcParams.update(ACORN_RC)
