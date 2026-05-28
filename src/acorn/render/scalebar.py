"""Scale-bar geometry helpers — no GUI imports, works headless."""

from __future__ import annotations

import numpy as np


def nice_scalebar_nm(
    pixel_size: float,
    img_width: int,
    target_frac: float = 0.15,
) -> float:
    """
    Return a 'round' physical length (nm) that renders as ~target_frac of img_width.

    Picks the nearest value from [1, 2, 5] × 10^n that is closest to the
    target physical width.
    """
    if pixel_size <= 0:
        return 100.0
    target_nm = pixel_size * img_width * target_frac
    mag = 10 ** np.floor(np.log10(max(target_nm, 1e-9)))
    candidates = [mag * m for m in (1, 2, 5, 10)]
    return float(min(candidates, key=lambda x: abs(x - target_nm)))


def format_scale_label(length_nm: float) -> str:
    """Human-readable label for a scalebar of given nm length."""
    if length_nm >= 1000:
        val = length_nm / 1000
        return f"{val:.4g} µm"
    elif length_nm < 1:
        val = length_nm * 1000
        return f"{val:.4g} pm"
    else:
        return f"{length_nm:.4g} nm"


def draw_scalebar(
    ax,
    length_nm: float,
    pixel_size: float,
    x_frac: float = 0.03,
    y_frac: float = 0.93,
    color: str = "#FFFFFF",
    linewidth: float = 2.0,
    fontsize: int = 12,
) -> list:
    """
    Draw a calibrated scale bar onto a matplotlib Axes.

    Parameters
    ----------
    ax          : matplotlib Axes with the cryo-EM image
    length_nm   : physical length of the bar in nm
    pixel_size  : calibrated pixel size in nm/px
    x_frac      : fractional x position of the bar's left edge
    y_frac      : fractional y position of the bar's top edge
    color       : bar and label colour
    linewidth   : not used for bar thickness (bar is a Rectangle); kept for API
    fontsize    : label font size in points

    Returns
    -------
    list of matplotlib artists added (for removal on redraw)
    """
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    artists = []
    h_ax, w_ax = ax.get_ylim()[0], ax.get_xlim()[1]   # image dims in data coords
    # ax limits: xlim=(−0.5, W−0.5), ylim=(H−0.5, −0.5) → h_ax ≈ H, w_ax ≈ W
    h_img = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
    w_img = abs(ax.get_xlim()[1] - ax.get_xlim()[0])

    if pixel_size <= 0:
        return artists

    bar_px = length_nm / pixel_size
    x0 = x_frac * w_img + ax.get_xlim()[0]
    y0 = y_frac * h_img + min(ax.get_ylim())
    bar_h = max(h_img * 0.007, 3)

    rect = Rectangle((x0, y0), bar_px, bar_h, color=color, zorder=6, linewidth=0)
    ax.add_patch(rect)
    artists.append(rect)

    # Invisible Line2D with pickradius so the scale bar is easy to click.
    # Line2D.contains() uses display-pixel distance, giving a reliable hit area
    # regardless of zoom — same technique used for arrow hit testing.
    hit_line = Line2D(
        [x0, x0 + bar_px], [y0 + bar_h / 2, y0 + bar_h / 2],
        color="none", lw=0, alpha=0, zorder=5, pickradius=12,
    )
    ax.add_line(hit_line)
    artists.append(hit_line)

    label = format_scale_label(length_nm)
    txt = ax.text(
        x0 + bar_px / 2, y0 - bar_h * 1.6, label,
        ha="center", va="bottom",
        color=color,
        fontsize=fontsize,
        fontweight="bold",
        zorder=7,
    )
    artists.append(txt)
    return artists
