from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

_XLABEL_MAP = {
    "ecd_nm":       "Equivalent circular diameter (nm)",
    "feret_nm":     "Feret diameter (nm)",
    "area_nm2":     "Area (nm²)",
    "perimeter_nm": "Perimeter (nm)",
    "circularity":  "Circularity",
    "aspect_ratio": "Aspect ratio",
    "bbox_w_nm":    "Bounding box width (nm)",
    "bbox_h_nm":    "Bounding box height (nm)",
}

PLOT_TYPES = ["scatter", "histogram", "box+jitter", "violin", "box", "waterfall"]


def _groups(df, metric, label_col):
    if label_col and label_col in df.columns:
        labels = sorted(df[label_col].dropna().unique().tolist())
        return [(str(lbl), df[df[label_col] == lbl][metric].dropna().values)
                for lbl in labels]
    return [("all", df[metric].dropna().values)]


def _stats(vals):
    n = len(vals)
    return n, float(np.mean(vals)), float(np.std(vals)), float(np.median(vals))


def _sig_bracket(ax, x1, x2, y, p, h=None):
    if h is None:
        ylim = ax.get_ylim()
        h = (ylim[1] - ylim[0]) * 0.025
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y],
            lw=0.75, color="#000000", clip_on=False)
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    ax.text((x1 + x2) * 0.5, y + h * 1.3, stars,
            ha="center", va="bottom", fontsize=9, color="#000000", fontweight="bold")


def draw_scatter(ax, df, metric="ecd_nm", scatter_y="aspect_ratio",
                 label_col="label", palette=None, log_x=False, log_y=False,
                 xlabel=None, ylabel=None):
    from acorn_plotting.style import PALETTE
    pal = palette or PALETTE
    grps = _groups(df, metric, label_col)
    for i, (lbl, _) in enumerate(grps):
        sub = df[df[label_col] == lbl] if (label_col and label_col in df.columns) else df
        kw = dict(color=pal[i % len(pal)], alpha=0.85, s=22,
                  edgecolors="#1a1a1a", linewidths=0.4)
        if len(grps) > 1:
            kw["label"] = lbl
        ax.scatter(sub[metric], sub[scatter_y], **kw)
    if len(grps) > 1:
        ax.legend()
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel or _XLABEL_MAP.get(metric, metric))
    ax.set_ylabel(ylabel or _XLABEL_MAP.get(scatter_y, scatter_y))


def draw_histogram(ax, df, metric="ecd_nm", n_bins=30, label_col="label",
                   palette=None, log_x=False, log_y=False,
                   xlabel=None, ylabel=None):
    from acorn_plotting.style import PALETTE
    pal = palette or PALETTE
    all_vals = df[metric].dropna().values
    if not len(all_vals):
        return
    lo, hi = all_vals.min(), all_vals.max()
    bins = np.linspace(lo, hi, n_bins + 1) if hi > lo else n_bins
    grps = _groups(df, metric, label_col)
    for i, (lbl, vals) in enumerate(grps):
        kw = dict(bins=bins, alpha=0.80, color=pal[i % len(pal)],
                  edgecolor="#1a1a1a", linewidth=0.5)
        if len(grps) > 1:
            kw["label"] = lbl
        ax.hist(vals, **kw)
    if len(grps) > 1:
        ax.legend()
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    n, mean, std, med = _stats(all_vals)
    ax.set_title(f"n = {n}    mean = {mean:.1f}    sd = {std:.1f}    median = {med:.1f}",
                 fontsize=8.5)
    ax.set_xlabel(xlabel or _XLABEL_MAP.get(metric, metric))
    ax.set_ylabel(ylabel or "Count")


def draw_box_jitter(ax, df, metric="ecd_nm", label_col="label", palette=None,
                    log_y=False, show_sig=True, xlabel=None, ylabel=None):
    from acorn_plotting.style import PALETTE
    from acorn_plotting.stats import run_statistics
    pal = palette or PALETTE

    grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
    data   = [v for _, v in grps]
    lbls   = [l for l, _ in grps]
    n_grps = len(grps)

    if not data:
        return

    positions = list(range(n_grps))

    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#000000", linewidth=2.0),
        whiskerprops=dict(color="#000000", linewidth=0.75),
        capprops=dict(color="#000000", linewidth=0.75),
        boxprops=dict(linewidth=0.75),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(pal[i % len(pal)])
        patch.set_alpha(0.20)

    rng = np.random.default_rng(42)
    for i, (_, vals) in enumerate(grps):
        c = pal[i % len(pal)]
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   color=c, alpha=0.90, s=14,
                   edgecolors="white", linewidths=0.3, zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(lbls)

    if log_y:
        ax.set_yscale("log")

    if show_sig and n_grps >= 2:
        try:
            result = run_statistics(df, metric, label_col)
            cmp = result.get("comparison", {})
            ph  = result.get("posthoc", [])

            if n_grps == 2 and cmp.get("p") is not None:
                ylim  = ax.get_ylim()
                y_top = ylim[1] * (1.30 if log_y else 1.07)
                _sig_bracket(ax, 0, 1, y_top, cmp["p"])
            elif ph:
                pos_map = {lbl: i for i, lbl in enumerate(lbls)}
                ylim  = ax.get_ylim()
                step  = (ylim[1] - ylim[0]) * 0.12
                for k, row in enumerate(ph):
                    ga, gb = row.get("group_a"), row.get("group_b")
                    if ga not in pos_map or gb not in pos_map:
                        continue
                    p_key = "p_bonferroni" if "p_bonferroni" in row else "p"
                    y_top = ylim[1] + step * (k + 1)
                    _sig_bracket(ax, pos_map[ga], pos_map[gb], y_top, row[p_key])
        except Exception:
            pass

    ax.set_ylabel(ylabel or _XLABEL_MAP.get(metric, metric))
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def draw_violin(ax, df, metric="ecd_nm", label_col="label", palette=None,
                log_y=False, xlabel=None, ylabel=None):
    from acorn_plotting.style import PALETTE
    pal = palette or PALETTE
    grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
    data = [v for _, v in grps]
    lbls = [l for l, _ in grps]
    if not data:
        return

    parts = ax.violinplot(data, positions=range(len(data)),
                          showmedians=False, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(pal[i % len(pal)])
        pc.set_edgecolor("#1a1a1a")
        pc.set_linewidth(0.75)
        pc.set_alpha(0.75)

    for i, vals in enumerate(data):
        q1, med, q3 = np.percentile(vals, [25, 50, 75])
        ax.vlines(i, q1, q3, color="#000000", linewidth=2.5, zorder=3)
        ax.scatter([i], [med], color="white", s=20,
                   edgecolors="#000000", linewidths=0.75, zorder=4)

    ax.set_xticks(range(len(lbls)))
    ax.set_xticklabels(lbls)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel or _XLABEL_MAP.get(metric, metric))
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def draw_box(ax, df, metric="ecd_nm", label_col="label", palette=None,
             log_y=False, xlabel=None, ylabel=None):
    from acorn_plotting.style import PALETTE
    pal = palette or PALETTE
    grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
    data = [v for _, v in grps]
    lbls = [l for l, _ in grps]
    if not data:
        return
    bp = ax.boxplot(
        data,
        patch_artist=True,
        labels=lbls,
        medianprops=dict(color="#000000", linewidth=2.0),
        whiskerprops=dict(color="#000000", linewidth=0.75),
        capprops=dict(color="#000000", linewidth=0.75),
        boxprops=dict(linewidth=0.75),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#888888", markeredgewidth=0.4),
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(pal[i % len(pal)])
        patch.set_alpha(0.40)
    if log_y:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel or _XLABEL_MAP.get(metric, metric))
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def draw_waterfall(ax_list, df, metric="ecd_nm", n_bins=30, label_col="label",
                   palette=None, log_x=False, xlabel=None):
    from acorn_plotting.style import PALETTE
    pal = palette or PALETTE
    grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
    all_vals = df[metric].dropna().values
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1) if len(all_vals) else n_bins
    for i, (ax, (lbl, vals)) in enumerate(zip(ax_list, grps)):
        c = pal[i % len(pal)]
        ax.hist(vals, bins=bins, color=c, alpha=0.80,
                edgecolor="#1a1a1a", linewidth=0.4)
        ax.set_ylabel(str(lbl), rotation=0, ha="right", va="center", fontsize=8)
        ax.yaxis.set_ticklabels([])
        if log_x:
            ax.set_xscale("log")
    ax_list[-1].set_xlabel(xlabel or _XLABEL_MAP.get(metric, metric))


def build_figure(df, fig, plot_type="scatter", metric="ecd_nm",
                 scatter_y="aspect_ratio", n_bins=30, label_col="label",
                 palette=None, log_x=False, log_y=False,
                 xlabel=None, ylabel=None, show_sig=True, output_path=None):
    from acorn_plotting.style import apply_acorn_style
    apply_acorn_style()
    fig.clf()

    if df is None or df.empty or metric not in df.columns:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No data — run particle analysis first.",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        if output_path:
            fig.savefig(output_path)
        return fig

    if plot_type == "waterfall":
        grps = [(l, v) for l, v in _groups(df, metric, label_col) if len(v) > 0]
        n_g  = max(len(grps), 1)
        axes = fig.subplots(n_g, 1, sharex=True)
        if n_g == 1:
            axes = [axes]
        draw_waterfall(axes, df, metric=metric, n_bins=n_bins,
                       label_col=label_col, palette=palette,
                       log_x=log_x, xlabel=xlabel)
    else:
        ax = fig.add_subplot(111)
        if plot_type == "scatter":
            draw_scatter(ax, df, metric=metric, scatter_y=scatter_y,
                         label_col=label_col, palette=palette,
                         log_x=log_x, log_y=log_y,
                         xlabel=xlabel, ylabel=ylabel)
        elif plot_type == "histogram":
            draw_histogram(ax, df, metric=metric, n_bins=n_bins,
                           label_col=label_col, palette=palette,
                           log_x=log_x, log_y=log_y,
                           xlabel=xlabel, ylabel=ylabel)
        elif plot_type == "box+jitter":
            draw_box_jitter(ax, df, metric=metric, label_col=label_col,
                            palette=palette, log_y=log_y,
                            show_sig=show_sig,
                            xlabel=xlabel, ylabel=ylabel)
        elif plot_type == "violin":
            draw_violin(ax, df, metric=metric, label_col=label_col,
                        palette=palette, log_y=log_y,
                        xlabel=xlabel, ylabel=ylabel)
        elif plot_type == "box":
            draw_box(ax, df, metric=metric, label_col=label_col,
                     palette=palette, log_y=log_y,
                     xlabel=xlabel, ylabel=ylabel)

    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=300)
    return fig


def build_figure_new(df, plot_type="scatter", metric="ecd_nm",
                     scatter_y="aspect_ratio", n_bins=30, label_col="label",
                     palette=None, log_x=False, log_y=False,
                     xlabel=None, ylabel=None, show_sig=True, output_path=None):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5.5, 4.5))
    return build_figure(df, fig, plot_type=plot_type, metric=metric,
                        scatter_y=scatter_y, n_bins=n_bins, label_col=label_col,
                        palette=palette, log_x=log_x, log_y=log_y,
                        xlabel=xlabel, ylabel=ylabel,
                        show_sig=show_sig, output_path=output_path)
