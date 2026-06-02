"""Figure-generation helpers for ACORN measurement data."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

_XLABEL_MAP = {
    "ecd_nm":       "Equivalent Circular Diameter (nm)",
    "feret_nm":     "Feret Diameter (nm)",
    "area_nm2":     "Area (nm²)",
    "perimeter_nm": "Perimeter (nm)",
    "circularity":  "Circularity",
    "aspect_ratio": "Aspect Ratio",
    "bbox_w_nm":    "Bounding Box Width (nm)",
    "bbox_h_nm":    "Bounding Box Height (nm)",
}

PLOT_TYPES = ["histogram", "violin", "box", "waterfall", "scatter"]


def _groups(df, metric, label_col):
    """Return list of (label, values) tuples for grouped plotting."""
    if label_col and label_col in df.columns:
        labels = sorted(df[label_col].dropna().unique().tolist())
        return [(str(lbl), df[df[label_col] == lbl][metric].dropna().values)
                for lbl in labels]
    return [("all", df[metric].dropna().values)]


# ---------------------------------------------------------------------------
# Individual plot builders
# ---------------------------------------------------------------------------

def plot_histogram(df, metric="ecd_nm", n_bins=30, label_col="label",
                   title=None, palette=None, output_path=None):
    import matplotlib.pyplot as plt
    from acorn_plotting.style import apply_acorn_style, PALETTE
    apply_acorn_style()
    pal = palette or PALETTE

    fig, ax = plt.subplots(figsize=(6, 4))
    all_vals = df[metric].dropna().values
    if len(all_vals) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _finish(fig, ax, metric, "Count", title, output_path)
        return fig

    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)
    grps = _groups(df, metric, label_col)
    for i, (lbl, vals) in enumerate(grps):
        kw = dict(bins=bins, alpha=0.7, color=pal[i % len(pal)], edgecolor="white")
        if len(grps) > 1:
            kw["label"] = lbl
        ax.hist(vals, **kw)
    if len(grps) > 1:
        ax.legend()

    n, mean, std, med = _stats(all_vals)
    _finish(fig, ax, metric, "Count",
            title or f"n={n}  mean={mean:.1f}  std={std:.1f}  median={med:.1f}",
            output_path)
    return fig


def plot_violin(df, metric="ecd_nm", label_col="label",
                title=None, palette=None, output_path=None):
    import matplotlib.pyplot as plt
    from acorn_plotting.style import apply_acorn_style, PALETTE
    apply_acorn_style()
    pal = palette or PALETTE

    grps = _groups(df, metric, label_col)
    data  = [v for _, v in grps if len(v) > 0]
    lbls  = [l for l, v in grps if len(v) > 0]

    fig, ax = plt.subplots(figsize=(max(4, len(lbls) * 1.4), 5))
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _finish(fig, ax, metric, "", title, output_path)
        return fig

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=True, showextrema=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(pal[i % len(pal)])
        pc.set_alpha(0.7)
    for part in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[part].set_color("#444444")
        parts[part].set_linewidth(1.0)

    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls)
    _finish(fig, ax, metric, _XLABEL_MAP.get(metric, metric), title, output_path,
            xlabel="", ylabel=_XLABEL_MAP.get(metric, metric))
    return fig


def plot_box(df, metric="ecd_nm", label_col="label",
             title=None, palette=None, output_path=None):
    import matplotlib.pyplot as plt
    from acorn_plotting.style import apply_acorn_style, PALETTE
    apply_acorn_style()
    pal = palette or PALETTE

    grps = _groups(df, metric, label_col)
    data = [v for _, v in grps if len(v) > 0]
    lbls = [l for l, v in grps if len(v) > 0]

    fig, ax = plt.subplots(figsize=(max(4, len(lbls) * 1.4), 5))
    if not data:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _finish(fig, ax, metric, "", title, output_path)
        return fig

    bp = ax.boxplot(data, patch_artist=True, labels=lbls,
                    medianprops=dict(color="#222222", linewidth=1.5))
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(pal[i % len(pal)])
        patch.set_alpha(0.7)

    _finish(fig, ax, metric, _XLABEL_MAP.get(metric, metric), title, output_path,
            xlabel="", ylabel=_XLABEL_MAP.get(metric, metric))
    return fig


def plot_waterfall(df, metric="ecd_nm", label_col="label", n_bins=30,
                   title=None, palette=None, output_path=None):
    """Ridge / waterfall plot — one row per label, stacked vertically."""
    import matplotlib.pyplot as plt
    from acorn_plotting.style import apply_acorn_style, PALETTE
    apply_acorn_style()
    pal = palette or PALETTE

    grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
    n_grps = len(grps)

    if n_grps == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _finish(fig, ax, metric, "", title, output_path)
        return fig

    fig, axes = plt.subplots(n_grps, 1, figsize=(6, 1.6 * n_grps + 1),
                              sharex=True)
    if n_grps == 1:
        axes = [axes]

    all_vals = df[metric].dropna().values
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    for i, (ax, (lbl, vals)) in enumerate(zip(axes, grps)):
        color = pal[i % len(pal)]
        ax.hist(vals, bins=bins, color=color, alpha=0.8, edgecolor="white")
        ax.set_ylabel(str(lbl), rotation=0, ha="right", va="center", fontsize=8)
        ax.yaxis.set_ticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_xlabel(_XLABEL_MAP.get(metric, metric))
    fig.suptitle(title or _XLABEL_MAP.get(metric, metric), fontsize=11)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
    return fig


def plot_scatter(df, x="ecd_nm", y="aspect_ratio", label_col="label",
                 title=None, palette=None, output_path=None):
    import matplotlib.pyplot as plt
    from acorn_plotting.style import apply_acorn_style, PALETTE
    apply_acorn_style()
    pal = palette or PALETTE

    fig, ax = plt.subplots(figsize=(6, 5))
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        _finish(fig, ax, x, y, title, output_path, xlabel=x, ylabel=y)
        return fig

    grps = _groups(df, x, label_col)
    for i, (lbl, _) in enumerate(grps):
        if label_col and label_col in df.columns:
            sub = df[df[label_col] == lbl]
        else:
            sub = df
        kw = dict(color=pal[i % len(pal)], alpha=0.7, s=20)
        if len(grps) > 1:
            kw["label"] = lbl
        ax.scatter(sub[x], sub[y], **kw)
    if len(grps) > 1:
        ax.legend()

    _finish(fig, ax, x, y, title or f"{y} vs {x}", output_path,
            xlabel=_XLABEL_MAP.get(x, x), ylabel=_XLABEL_MAP.get(y, y))
    return fig


# ---------------------------------------------------------------------------
# Unified dispatcher (used by both the panel and CLU)
# ---------------------------------------------------------------------------

def build_figure(df, plot_type="histogram", metric="ecd_nm", scatter_y="aspect_ratio",
                 n_bins=30, label_col="label", title=None, palette=None,
                 output_path=None):
    """
    Build and return a matplotlib Figure.

    Parameters
    ----------
    plot_type : str
        One of: histogram, violin, box, waterfall, scatter.
    metric : str
        Primary metric column (x-axis for histogram/scatter/waterfall).
    scatter_y : str
        Y-axis column when plot_type='scatter'.
    """
    kw = dict(df=df, label_col=label_col, title=title,
              palette=palette, output_path=output_path)
    if plot_type == "violin":
        return plot_violin(metric=metric, **kw)
    if plot_type == "box":
        return plot_box(metric=metric, **kw)
    if plot_type == "waterfall":
        return plot_waterfall(metric=metric, n_bins=n_bins, **kw)
    if plot_type == "scatter":
        return plot_scatter(x=metric, y=scatter_y, **kw)
    return plot_histogram(metric=metric, n_bins=n_bins, **kw)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _stats(vals):
    n = len(vals)
    return n, float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))


def _finish(fig, ax, metric, ylabel, title, output_path,
            xlabel=None, ylabel_=None):
    ax.set_xlabel(xlabel if xlabel is not None else _XLABEL_MAP.get(metric, metric))
    if ylabel_ is not None:
        ax.set_ylabel(ylabel_)
    elif ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path)
