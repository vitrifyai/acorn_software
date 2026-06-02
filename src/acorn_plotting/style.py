"""Matplotlib style helpers for ACORN publication figures."""
from __future__ import annotations

import matplotlib


# Minimal rcParams override — works even if a style sheet is not installed.
ACORN_RC: dict = {
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#444444",
    "axes.linewidth":    1.0,
    "axes.labelsize":    11,
    "axes.titlesize":    12,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "lines.linewidth":   1.5,
    "patch.linewidth":   0.8,
    "legend.fontsize":   9,
    "legend.frameon":    True,
    "legend.framealpha": 0.85,
    "savefig.dpi":       150,
    "savefig.bbox":      "tight",
    "font.family":       "sans-serif",
}

PALETTE = [
    "#4878CF",  # blue
    "#D65F5F",  # coral-red
    "#6ACC65",  # green
    "#B47CC7",  # purple
    "#C4AD66",  # ochre
    "#77BEDB",  # sky
]


def apply_acorn_style() -> None:
    """Apply ACORN rcParams to the current matplotlib session."""
    matplotlib.rcParams.update(ACORN_RC)
