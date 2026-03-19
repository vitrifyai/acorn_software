"""
ACORN Surface Area Statistics Module  (surface_area_stats.py)
==============================================================
Accepts the DataFrame returned by batch_surface_area() and provides
population-level summary statistics, statistical tests with automatic
test selection, publication-quality figures, and batch export helpers.

Required dependencies: numpy, pandas, scipy, matplotlib, seaborn, statsmodels
Optional:  cupy (GPU-accelerated permutation tests for n > 10,000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Color palette used for groups (up to 8 groups; cycles if more)
_GROUP_PALETTE = [
    "#4878D0", "#EE854A", "#6ACC65", "#D65F5F",
    "#956CB4", "#8C613C", "#DC7EC0", "#797979",
]

# Colors by SA estimation method (imported from surface_area if available)
_METHOD_COLORS = {
    "ellipsoid": "#4878D0",
    "cauchy": "#6ACC65",
    "fourier": "#EE854A",
    "fourier_spiky": "#D65F5F",
    "unknown": "#999999",
}


# ── style helpers ─────────────────────────────────────────────────────────────

def _apply_publication_style(ax) -> None:
    """Remove top and right spines for a clean publication-ready appearance."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8)


def _p_to_stars(p: float) -> str:
    """Convert p-value to significance stars."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _save_figure(fig, output_path, dpi: int = 300) -> None:
    """Save a figure as both PNG and SVG, stripping any existing extension."""
    if output_path is None:
        return
    p = Path(output_path).with_suffix("")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(p.with_suffix(".png")), dpi=dpi, bbox_inches="tight")
    fig.savefig(str(p.with_suffix(".svg")), bbox_inches="tight")


def _group_colors(group_order: list[str]) -> dict[str, str]:
    return {g: _GROUP_PALETTE[i % len(_GROUP_PALETTE)] for i, g in enumerate(group_order)}


def _group_summary_stats(data: np.ndarray, alpha: float = 0.05) -> dict:
    """Descriptive statistics + 95% CI for a single group."""
    from scipy import stats

    n = len(data)
    if n == 0:
        return {"n": 0}
    mean = float(data.mean())
    std = float(data.std(ddof=1)) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 1 else 0.0
    median = float(np.median(data))
    q25, q75 = float(np.percentile(data, 25)), float(np.percentile(data, 75))
    if n > 1:
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=n - 1))
        ci_lo, ci_hi = mean - t_crit * sem, mean + t_crit * sem
    else:
        ci_lo, ci_hi = mean, mean
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "median": median,
        "iqr": q75 - q25,
        "q25": q25,
        "q75": q75,
        "ci95_lo": float(ci_lo),
        "ci95_hi": float(ci_hi),
    }


def _test_normality(data: np.ndarray) -> dict:
    """Shapiro-Wilk (n < 5000) or D'Agostino-Pearson (n >= 5000)."""
    from scipy import stats

    n = len(data)
    if n < 3:
        return {"statistic": float("nan"), "p_value": float("nan"), "is_normal": None, "test": "none"}
    if n < 5000:
        stat, p = stats.shapiro(data)
        test = "shapiro_wilk"
    else:
        stat, p = stats.normaltest(data)
        test = "dagostino_pearson"
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "is_normal": bool(p > 0.05),
        "test": test,
    }


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled Cohen's d effect size for two independent groups."""
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    pooled_var = (
        a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)
    ) / (len(a) + len(b) - 2)
    if pooled_var <= 0:
        return 0.0
    return float(abs(a.mean() - b.mean()) / np.sqrt(pooled_var))


def _dunns_test(groups: dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Dunn's post-hoc test for all pairwise group comparisons.

    Uses the standard tie-corrected formula (Dunn, 1964):
      z_ij = |R_i_bar - R_j_bar| / sqrt(sigma²·(1/n_i + 1/n_j))
    where sigma² = (N·(N+1)/12) - T/(12·(N-1)) and T = Σ(t³-t) over ties.

    Returns DataFrame with columns: group_a, group_b, z_statistic, p_raw.
    """
    from itertools import combinations
    from scipy.stats import norm, rankdata

    names = list(groups.keys())
    arrays = [groups[n] for n in names]
    ns = [len(a) for a in arrays]
    N = sum(ns)

    combined = np.concatenate(arrays)
    all_ranks = rankdata(combined)

    _, tie_counts = np.unique(combined, return_counts=True)
    T = float(np.sum(tie_counts.astype(float) ** 3 - tie_counts))

    idx = 0
    mean_ranks: dict[str, float] = {}
    for name, n in zip(names, ns):
        mean_ranks[name] = float(all_ranks[idx : idx + n].mean())
        idx += n

    ns_map = dict(zip(names, ns))
    rows = []
    for na, nb in combinations(names, 2):
        ni, nj = ns_map[na], ns_map[nb]
        variance = (N * (N + 1) / 12.0 - T / max(12.0 * (N - 1), 1e-12)) * (1.0 / ni + 1.0 / nj)
        if variance <= 0:
            z, p = 0.0, 1.0
        else:
            z = abs(mean_ranks[na] - mean_ranks[nb]) / np.sqrt(variance)
            p = float(2.0 * norm.sf(z))
        rows.append({"group_a": na, "group_b": nb, "z_statistic": float(z), "p_raw": float(p)})
    return pd.DataFrame(rows)


def _draw_stat_brackets(ax, pairwise_df: pd.DataFrame, group_order: list[str], alpha: float = 0.05) -> None:
    """
    Draw significance brackets above a categorical plot.

    Only pairs where p_fdr < alpha (or p_value if p_fdr is absent) are drawn.
    Brackets are stacked vertically to avoid overlap.
    """
    p_col = "p_fdr" if "p_fdr" in pairwise_df.columns else "p_value"
    sig = pairwise_df[pairwise_df[p_col] < alpha].copy()
    if sig.empty:
        return

    positions = {g: float(i) for i, g in enumerate(group_order)}
    ylim = ax.get_ylim()
    y_range = ylim[1] - ylim[0]
    step = y_range * 0.07
    y_cursor = ylim[1] - y_range * 0.04

    for _, row in sig.iterrows():
        ga, gb = row.get("group_a"), row.get("group_b")
        if ga not in positions or gb not in positions:
            continue
        p = float(row.get(p_col, 1.0))
        stars = _p_to_stars(p)
        if stars == "ns":
            continue
        x1, x2 = positions[ga], positions[gb]
        bar_h = y_cursor + step * 0.15
        ax.plot([x1, x1, x2, x2], [y_cursor, bar_h, bar_h, y_cursor],
                lw=0.8, color="k", clip_on=False)
        ax.text((x1 + x2) / 2.0, bar_h, stars,
                ha="center", va="bottom", fontsize=9, color="k", clip_on=False)
        y_cursor += step

    ax.set_ylim(ylim[0], max(ylim[1], y_cursor + step * 0.3))


# ── existing summary functions (kept) ─────────────────────────────────────────

def summarize(df: pd.DataFrame, column: str = "SA_nm2") -> dict:
    """
    Descriptive statistics for one numeric column (default: SA_nm2).

    Returns
    -------
    dict with keys: n, n_valid, mean, std, sem, median,
                    p5, p25, p75, p95, min, max, cv
    """
    series = df[column].dropna()
    n_valid = int((series > 0).sum())
    valid = series[series > 0]

    if valid.empty:
        return {"n": len(series), "n_valid": 0}

    return {
        "n": len(series),
        "n_valid": n_valid,
        "mean": float(valid.mean()),
        "std": float(valid.std()),
        "sem": float(valid.sem()),
        "median": float(valid.median()),
        "p5": float(valid.quantile(0.05)),
        "p25": float(valid.quantile(0.25)),
        "p75": float(valid.quantile(0.75)),
        "p95": float(valid.quantile(0.95)),
        "min": float(valid.min()),
        "max": float(valid.max()),
        "cv": float(valid.std() / valid.mean()) if valid.mean() > 0 else float("nan"),
    }


def method_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count of particles by estimation method.  Returns [method_used, count, fraction]."""
    counts = df["method_used"].value_counts().rename_axis("method_used").reset_index(name="count")
    counts["fraction"] = counts["count"] / max(len(df), 1)
    return counts


def flag_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary of flagged particles.

    Returns [flag_reason, count, fraction].  Unflagged → 'none'.
    """
    reasons = df["flag_reason"].fillna("none").apply(lambda s: s if s else "none")
    counts = reasons.value_counts().rename_axis("flag_reason").reset_index(name="count")
    counts["fraction"] = counts["count"] / max(len(df), 1)
    return counts


def hollow_report(df: pd.DataFrame) -> dict:
    """Return hollow-particle count, fraction, and mean shell thickness."""
    hollow = df[df["is_hollow"]]
    return {
        "n_hollow": int(len(hollow)),
        "fraction_hollow": float(len(hollow) / max(len(df), 1)),
        "mean_shell_thickness_nm": float(hollow["shell_thickness_estimate_nm"].mean())
        if not hollow.empty else float("nan"),
    }


def fit_lognormal(data: "Sequence | pd.Series") -> dict:
    """
    Fit a log-normal distribution to positive values.

    Returns mu, sigma (log-scale), mean/median/mode in original units, KS test.
    """
    from scipy import stats

    arr = np.asarray(data, dtype=float)
    arr = arr[arr > 0]
    if len(arr) < 3:
        return {}
    sigma, loc, scale = stats.lognorm.fit(arr, floc=0)
    mu = np.log(scale)
    ks_stat, ks_p = stats.kstest(arr, "lognorm", args=(sigma, 0, scale))
    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "mean_nm2": float(np.exp(mu + sigma ** 2 / 2)),
        "median_nm2": float(np.exp(mu)),
        "mode_nm2": float(np.exp(mu - sigma ** 2)),
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_p),
    }


def fit_normal(data: "Sequence | pd.Series") -> dict:
    """Fit a normal distribution.  Returns mean, std, KS test."""
    from scipy import stats

    arr = np.asarray(data, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return {}
    mean, std = float(arr.mean()), float(arr.std())
    ks_stat, ks_p = stats.kstest(arr, "norm", args=(mean, std))
    return {"mean": mean, "std": std, "ks_statistic": float(ks_stat), "ks_pvalue": float(ks_p)}


def per_method_summary(df: pd.DataFrame, column: str = "SA_nm2") -> pd.DataFrame:
    """Return summarize() statistics broken down by method_used."""
    rows = []
    for method, grp in df.groupby("method_used"):
        row = summarize(grp, column)
        row["method_used"] = method
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("method_used")


# ── enhanced group comparison ─────────────────────────────────────────────────

def compare_groups(
    results_df: pd.DataFrame,
    group_col: str,
    sa_col: str = "SA_nm2",
    include_flagged: bool = False,
    alpha: float = 0.05,
    use_gpu: bool = False,
) -> dict:
    """
    Automatically select and run appropriate statistical tests between groups.

    Test selection
    --------------
    2 groups:  Mann-Whitney U (non-parametric default) + Cohen's d effect size.
               When use_gpu=True and n > 10,000, a GPU-accelerated permutation
               test (10,000 permutations) is used instead; falls back to
               Mann-Whitney if cupy is not available.
               A two-sample t-test is also computed and stored in 'additional'.
    3+ groups: Kruskal-Wallis omnibus test + Dunn's post-hoc with
               Benjamini-Hochberg FDR correction + eta-squared effect size.
               use_gpu=True applies GPU permutation for each pairwise comparison.

    Normality is checked per group with Shapiro-Wilk (n < 5000) or
    D'Agostino-Pearson (n >= 5000).

    Parameters
    ----------
    results_df     : DataFrame from batch_surface_area()
    group_col      : column that identifies the group label
    sa_col         : numeric column to compare (default: 'SA_nm2')
    include_flagged: if False (default), exclude flagged=True rows
    alpha          : significance threshold for the FDR correction and bracket
                     annotation (default 0.05)
    use_gpu        : if True, use cupy permutation test for large n;
                     falls back gracefully to scipy if cupy is not available

    Returns
    -------
    dict with keys:
        test_used    : str — 'mann_whitney' | 'kruskal_wallis+dunn' | 'permutation'
        n_groups     : int
        group_stats  : dict[group_name -> {n, mean, std, sem, median, iqr,
                                           q25, q75, ci95_lo, ci95_hi}]
        normality    : dict[group_name -> {statistic, p_value, is_normal, test}]
        omnibus      : {statistic, p_value, eta_squared} or None (2-group case)
        pairwise     : DataFrame[group_a, group_b, statistic, p_raw, p_fdr,
                                 effect_size, significant]
        additional   : {t_statistic, t_p_value} for 2-group case, else None
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    # filter
    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    df = df[[group_col, sa_col]].dropna()

    groups: dict[str, np.ndarray] = {
        name: grp[sa_col].values
        for name, grp in df.groupby(group_col)
        if len(grp) >= 2
    }
    n_groups = len(groups)

    group_stats = {name: _group_summary_stats(arr, alpha) for name, arr in groups.items()}
    normality = {name: _test_normality(arr) for name, arr in groups.items()}

    if n_groups < 2:
        return {
            "test_used": "none",
            "n_groups": n_groups,
            "group_stats": group_stats,
            "normality": normality,
            "omnibus": None,
            "pairwise": pd.DataFrame(),
            "additional": None,
        }

    def _permutation_p(a: np.ndarray, b: np.ndarray, n_perm: int = 10_000) -> tuple[float, str]:
        """GPU or CPU permutation test; returns (p_value, test_name)."""
        if use_gpu:
            try:
                import cupy as cp
                ca, cb = cp.asarray(a), cp.asarray(b)
                combined = cp.concatenate([ca, cb])
                na = len(a)
                observed = float(abs(cp.mean(ca) - cp.mean(cb)).get())
                count = 0
                for _ in range(n_perm):
                    perm = cp.random.permutation(combined)
                    diff = float(abs(cp.mean(perm[:na]) - cp.mean(perm[na:])).get())
                    if diff >= observed:
                        count += 1
                return (count + 1) / (n_perm + 1), "permutation_gpu"
            except ImportError:
                logger.debug("cupy not available; falling back to Mann-Whitney")
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p), "mann_whitney"

    if n_groups == 2:
        names = list(groups.keys())
        a, b = groups[names[0]], groups[names[1]]
        N = len(a) + len(b)
        if use_gpu and N > 10_000:
            p_val, test_name = _permutation_p(a, b)
            stat = float("nan")
        else:
            u, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
            stat, test_name = float(u), "mann_whitney"

        d = _cohens_d(a, b)
        pairwise = pd.DataFrame([{
            "group_a": names[0], "group_b": names[1],
            "statistic": stat, "p_raw": float(p_val), "p_fdr": float(p_val),
            "effect_size": d, "significant": bool(p_val < alpha),
        }])

        t_stat, t_p = stats.ttest_ind(a, b, equal_var=False)
        return {
            "test_used": test_name,
            "n_groups": 2,
            "group_stats": group_stats,
            "normality": normality,
            "omnibus": None,
            "pairwise": pairwise,
            "additional": {"t_statistic": float(t_stat), "t_p_value": float(t_p)},
        }

    # 3+ groups: Kruskal-Wallis + Dunn + BH FDR
    kw_stat, kw_p = stats.kruskal(*groups.values())
    N_total = sum(len(v) for v in groups.values())
    eta_sq = max(0.0, (kw_stat - n_groups + 1) / max(N_total - n_groups, 1))

    pairwise = _dunns_test(groups)

    if use_gpu and N_total > 10_000:
        # GPU pairwise permutation replaces Dunn p_raw
        from itertools import combinations
        perm_rows = []
        for (na, a), (nb, b) in combinations(groups.items(), 2):
            p_perm, _ = _permutation_p(a, b)
            perm_rows.append({"group_a": na, "group_b": nb, "p_raw": p_perm})
        perm_df = pd.DataFrame(perm_rows)
        pairwise = pairwise.merge(perm_df, on=["group_a", "group_b"], suffixes=("_dunn", ""))
        test_name = "kruskal_wallis+permutation_gpu"
    else:
        test_name = "kruskal_wallis+dunn"

    reject, p_fdr, _, _ = multipletests(pairwise["p_raw"].values, alpha=alpha, method="fdr_bh")
    pairwise["p_fdr"] = p_fdr
    pairwise["significant"] = reject

    # effect size per pair: Cohen's d
    def _get_pair_d(row):
        a = groups.get(row["group_a"], np.array([]))
        b = groups.get(row["group_b"], np.array([]))
        return _cohens_d(a, b)

    pairwise["effect_size"] = pairwise.apply(_get_pair_d, axis=1)
    # add omnibus statistic as column for reference (pairwise is per-pair)
    pairwise["statistic"] = pairwise.get("z_statistic", pairwise.get("statistic", float("nan")))

    return {
        "test_used": test_name,
        "n_groups": n_groups,
        "group_stats": group_stats,
        "normality": normality,
        "omnibus": {
            "statistic": float(kw_stat),
            "p_value": float(kw_p),
            "eta_squared": float(eta_sq),
        },
        "pairwise": pairwise,
        "additional": None,
    }


# ── publication-quality plot functions ───────────────────────────────────────

def plot_sa_distribution(
    results_df: pd.DataFrame,
    group_col: str,
    output_path=None,
    style: str = "violin",
    sa_col: str = "SA_nm2",
    include_flagged: bool = False,
    alpha: float = 0.05,
    stats_results: Optional[dict] = None,
) -> "matplotlib.figure.Figure":
    """
    Publication-quality per-group SA distribution plot.

    Parameters
    ----------
    results_df     : DataFrame from batch_surface_area()
    group_col      : column containing group labels
    output_path    : base path for output files; saves as .png and .svg
    style          : 'violin' | 'box' | 'ridge'
        violin — seaborn violinplot + overlaid strip plot, median line, IQR.
        box    — seaborn boxplot + overlaid strip plot.
        ridge  — KDE ridgeline plot (no seaborn required).
    sa_col         : SA column to plot (default 'SA_nm2')
    include_flagged: include flagged particles (default False)
    alpha          : significance threshold for bracket annotation
    stats_results  : pre-computed output from compare_groups(); if None and
                     style in ('violin', 'box'), compare_groups() is run
                     automatically to annotate significant pairs.

    Returns
    -------
    matplotlib Figure.  Saved as PNG + SVG at 300 DPI if output_path is given.
    """
    import matplotlib.pyplot as plt

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    df = df[[group_col, sa_col]].dropna()
    df = df[df[sa_col] > 0]

    group_order = sorted(df[group_col].unique().tolist())
    n_groups = len(group_order)
    colors = _group_colors(group_order)
    palette = [colors[g] for g in group_order]

    if style == "ridge":
        return _plot_ridge(df, group_col, sa_col, group_order, palette, output_path)

    import seaborn as sns

    fig, ax = plt.subplots(figsize=(max(5.0, n_groups * 1.4), 4.5))

    if style == "violin":
        sns.violinplot(
            data=df, x=group_col, y=sa_col, order=group_order,
            hue=group_col, palette=palette, legend=False,
            inner="box", cut=0, ax=ax, linewidth=0.8,
        )
    else:  # box
        sns.boxplot(
            data=df, x=group_col, y=sa_col, order=group_order,
            hue=group_col, palette=palette, legend=False,
            width=0.5, linewidth=0.8, ax=ax,
            flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
        )
    sns.stripplot(
        data=df, x=group_col, y=sa_col, order=group_order,
        hue=group_col, palette=["k"] * len(group_order), legend=False,
        alpha=0.25, size=2.0, jitter=True, ax=ax,
    )

    # significance brackets
    if stats_results is None and n_groups >= 2:
        stats_results = compare_groups(
            results_df, group_col, sa_col=sa_col,
            include_flagged=include_flagged, alpha=alpha,
        )
    if stats_results is not None and not stats_results["pairwise"].empty:
        _draw_stat_brackets(ax, stats_results["pairwise"], group_order, alpha)

    ax.set_xlabel(group_col)
    ax.set_ylabel(f"{sa_col} (nm²)")
    ax.set_title(f"SA distribution by {group_col}")
    _apply_publication_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def _plot_ridge(df, group_col, sa_col, group_order, palette, output_path):
    """KDE ridgeline plot, pure matplotlib."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    fig, ax = plt.subplots(figsize=(6.5, max(3.0, len(group_order) * 1.0)))

    all_vals = df[sa_col].values
    x_min, x_max = float(all_vals.min()), float(all_vals.max())
    xs = np.linspace(x_min, x_max, 400)

    for i, (name, color) in enumerate(zip(group_order, palette)):
        data = df[df[group_col] == name][sa_col].dropna().values
        if len(data) < 3 or data.std() < 1e-9:
            # constant data: draw a single vertical spike
            ax.plot([data[0], data[0]], [i, i + 0.75], lw=2.0, color=color)
            ax.text(x_min, i + 0.06, name, va="bottom", ha="left", fontsize=9)
            continue
        try:
            kde = gaussian_kde(data)
        except Exception:
            ax.plot([data.mean(), data.mean()], [i, i + 0.75], lw=2.0, color=color)
            ax.text(x_min, i + 0.06, name, va="bottom", ha="left", fontsize=9)
            continue
        ys = kde(xs)
        ys_norm = ys / max(ys.max(), 1e-12) * 0.75
        ax.fill_between(xs, i, i + ys_norm, alpha=0.72, color=color, linewidth=0)
        ax.plot(xs, i + ys_norm, lw=1.4, color=color)
        ax.text(x_min, i + 0.06, name, va="bottom", ha="left", fontsize=9)

    ax.set_yticks([])
    ax.set_xlabel(f"{sa_col} (nm²)")
    ax.set_title(f"SA distribution by {group_col}")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_sa_ecdf(
    results_df: pd.DataFrame,
    group_col: str,
    output_path=None,
    sa_col: str = "SA_nm2",
    include_flagged: bool = False,
    confidence: float = 0.95,
) -> "matplotlib.figure.Figure":
    """
    Empirical CDF curves per group with Dvoretzky-Kiefer-Wolfowitz confidence bands.

    The DKW confidence band at level (1 - α) is:
      ε = sqrt(ln(2/α) / (2n)),  band = ECDF ± ε  (clipped to [0,1]).

    Parameters
    ----------
    confidence : confidence level for the DKW band (default 0.95)

    Returns matplotlib Figure.  Saved as PNG + SVG if output_path is given.
    """
    import matplotlib.pyplot as plt

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    df = df[[group_col, sa_col]].dropna()
    df = df[df[sa_col] > 0]

    group_order = sorted(df[group_col].unique().tolist())
    colors = _group_colors(group_order)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    alpha_dkw = 1.0 - confidence

    for name in group_order:
        data = np.sort(df[df[group_col] == name][sa_col].dropna().values)
        n = len(data)
        if n < 2:
            continue
        color = colors[name]
        y_ecdf = np.arange(1, n + 1) / n
        eps = float(np.sqrt(np.log(2.0 / alpha_dkw) / (2.0 * n)))
        ax.plot(data, y_ecdf, color=color, lw=2.0, label=name)
        ax.fill_between(
            data,
            np.clip(y_ecdf - eps, 0, 1),
            np.clip(y_ecdf + eps, 0, 1),
            alpha=0.15, color=color,
        )

    ax.set_xlabel(f"{sa_col} (nm²)")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(f"ECDF of {sa_col} by {group_col}")
    ax.legend(title=group_col, fontsize=8)
    _apply_publication_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_sa_histogram(
    results_df: pd.DataFrame,
    group_col: str,
    output_path=None,
    sa_col: str = "SA_nm2",
    bins: int = 40,
    include_flagged: bool = False,
) -> "matplotlib.figure.Figure":
    """
    Overlapping semi-transparent histograms with KDE overlay per group.

    Mean and median are annotated as solid / dashed vertical lines.

    Returns matplotlib Figure.  Saved as PNG + SVG if output_path is given.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    df = df[[group_col, sa_col]].dropna()
    df = df[df[sa_col] > 0]

    group_order = sorted(df[group_col].unique().tolist())
    colors = _group_colors(group_order)

    all_vals = df[sa_col].values
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name in group_order:
        data = df[df[group_col] == name][sa_col].dropna().values
        color = colors[name]
        ax.hist(data, bins=bin_edges, alpha=0.4, color=color, density=True, label=name)
        if len(data) > 3 and data.std() > 1e-9:
            try:
                kde = gaussian_kde(data)
                xs = np.linspace(all_vals.min(), all_vals.max(), 300)
                ax.plot(xs, kde(xs), color=color, lw=1.8)
            except Exception:
                pass
        ax.axvline(float(np.mean(data)), color=color, lw=1.2, linestyle="-")
        ax.axvline(float(np.median(data)), color=color, lw=1.2, linestyle="--")

    ax.set_xlabel(f"{sa_col} (nm²)")
    ax.set_ylabel("Density")
    ax.set_title(f"SA histogram by {group_col}")
    ax.legend(title=group_col, fontsize=8)
    # annotation for line styles
    from matplotlib.lines import Line2D
    ax.legend(
        handles=[
            *[plt.Rectangle((0, 0), 1, 1, color=colors[g], alpha=0.5, label=g) for g in group_order],
            Line2D([0], [0], color="k", lw=1.2, linestyle="-", label="mean"),
            Line2D([0], [0], color="k", lw=1.2, linestyle="--", label="median"),
        ],
        fontsize=7,
    )
    _apply_publication_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_sa_scatter(
    results_df: pd.DataFrame,
    x_col: str,
    y_col: str = "SA_nm2",
    group_col: Optional[str] = None,
    output_path=None,
    include_flagged: bool = False,
) -> "matplotlib.figure.Figure":
    """
    Scatter plot of SA vs any other morphometric column, with correlation annotation.

    Both Pearson r and Spearman ρ are computed on the combined data (ignoring
    group labels) and annotated in the top-left corner.

    Parameters
    ----------
    x_col      : column for the x-axis (e.g. 'a_nm', 'coverage_score')
    y_col      : column for the y-axis (default 'SA_nm2')
    group_col  : if given, points are coloured by group

    Returns matplotlib Figure.  Saved as PNG + SVG if output_path is given.
    """
    import matplotlib.pyplot as plt
    from scipy.stats import pearsonr, spearmanr

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    cols = [x_col, y_col] + ([group_col] if group_col else [])
    df = df[cols].dropna()
    df = df[df[y_col] > 0]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    if group_col:
        group_order = sorted(df[group_col].unique().tolist())
        colors = _group_colors(group_order)
        for name in group_order:
            sub = df[df[group_col] == name]
            ax.scatter(sub[x_col], sub[y_col], s=14, alpha=0.65,
                       color=colors[name], label=name, linewidths=0)
        ax.legend(title=group_col, fontsize=8)
    else:
        ax.scatter(df[x_col], df[y_col], s=14, alpha=0.6,
                   color=_GROUP_PALETTE[0], linewidths=0)

    x_vals, y_vals = df[x_col].values, df[y_col].values
    if len(x_vals) > 2:
        r_p, p_p = pearsonr(x_vals, y_vals)
        r_s, p_s = spearmanr(x_vals, y_vals)
        ax.annotate(
            f"Pearson r = {r_p:.3f}  (p = {p_p:.2g})\n"
            f"Spearman r = {r_s:.3f}  (p = {p_s:.2g})",
            xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, linewidth=0.5),
        )

    ax.set_xlabel(x_col)
    ax.set_ylabel(f"{y_col} (nm²)")
    ax.set_title(f"{y_col} vs {x_col}")
    _apply_publication_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


def plot_sa_summary_panel(
    results_df: pd.DataFrame,
    group_col: str,
    stats_results: dict,
    output_path=None,
    sa_col: str = "SA_nm2",
    include_flagged: bool = False,
    alpha: float = 0.05,
) -> "matplotlib.figure.Figure":
    """
    Multi-panel publication figure: violin + ECDF + histogram + stats table.

    Layout (2 rows × 2 columns):
      [0,0] violin plot         [0,1] ECDF with confidence bands
      [1,0] histogram + KDE     [1,1] group statistics table

    Parameters
    ----------
    stats_results : output dict from compare_groups(); used for the stats table
                    and significance brackets.

    Returns matplotlib Figure.  Saved as PNG + SVG if output_path is given.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import seaborn as sns
    from scipy.stats import gaussian_kde

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]
    df = df[[group_col, sa_col]].dropna()
    df = df[df[sa_col] > 0]

    group_order = sorted(df[group_col].unique().tolist())
    colors = _group_colors(group_order)
    palette = [colors[g] for g in group_order]

    fig = plt.figure(figsize=(11.0, 8.5))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

    # ── [0,0] violin ──────────────────────────────────────────────────────────
    ax_vln = fig.add_subplot(gs[0, 0])
    sns.violinplot(data=df, x=group_col, y=sa_col, order=group_order,
                   hue=group_col, palette=palette, legend=False,
                   inner="box", cut=0, ax=ax_vln, linewidth=0.8)
    sns.stripplot(data=df, x=group_col, y=sa_col, order=group_order,
                  hue=group_col, palette=["k"]*len(group_order), legend=False,
                  alpha=0.2, size=2.0, jitter=True, ax=ax_vln)
    if not stats_results["pairwise"].empty:
        _draw_stat_brackets(ax_vln, stats_results["pairwise"], group_order, alpha)
    ax_vln.set_xlabel(group_col)
    ax_vln.set_ylabel(f"{sa_col} (nm²)")
    ax_vln.set_title("Distribution")
    _apply_publication_style(ax_vln)

    # ── [0,1] ECDF ────────────────────────────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[0, 1])
    alpha_dkw = 0.05
    for name in group_order:
        data = np.sort(df[df[group_col] == name][sa_col].dropna().values)
        n = len(data)
        if n < 2:
            continue
        color = colors[name]
        y_ecdf = np.arange(1, n + 1) / n
        eps = float(np.sqrt(np.log(2.0 / alpha_dkw) / (2.0 * n)))
        ax_cdf.plot(data, y_ecdf, color=color, lw=2.0, label=name)
        ax_cdf.fill_between(data,
                            np.clip(y_ecdf - eps, 0, 1),
                            np.clip(y_ecdf + eps, 0, 1),
                            alpha=0.15, color=color)
    ax_cdf.set_xlabel(f"{sa_col} (nm²)")
    ax_cdf.set_ylabel("Cumulative probability")
    ax_cdf.set_title("ECDF (95% DKW band)")
    ax_cdf.legend(title=group_col, fontsize=7)
    _apply_publication_style(ax_cdf)

    # ── [1,0] histogram ───────────────────────────────────────────────────────
    ax_hist = fig.add_subplot(gs[1, 0])
    all_vals = df[sa_col].values
    bin_edges = np.linspace(all_vals.min(), all_vals.max(), 36)
    for name in group_order:
        data = df[df[group_col] == name][sa_col].dropna().values
        color = colors[name]
        ax_hist.hist(data, bins=bin_edges, alpha=0.38, color=color, density=True, label=name)
        if len(data) > 3 and data.std() > 1e-9:
            try:
                xs = np.linspace(all_vals.min(), all_vals.max(), 300)
                ax_hist.plot(xs, gaussian_kde(data)(xs), color=color, lw=1.6)
            except Exception:
                pass
        ax_hist.axvline(float(np.mean(data)), color=color, lw=1.0, ls="-")
        ax_hist.axvline(float(np.median(data)), color=color, lw=1.0, ls="--")
    ax_hist.set_xlabel(f"{sa_col} (nm²)")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Histogram + KDE  (solid=mean, dashed=median)")
    _apply_publication_style(ax_hist)

    # ── [1,1] stats table ─────────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, 1])
    ax_tbl.axis("off")

    gs_dict = stats_results.get("group_stats", {})
    pairwise = stats_results.get("pairwise", pd.DataFrame())
    omnibus = stats_results.get("omnibus")

    # build table rows
    header = ["Group", "n", "Median", "IQR", "95% CI"]
    rows_tbl = []
    for g in group_order:
        s = gs_dict.get(g, {})
        n = s.get("n", "—")
        med = f"{s['median']:.1f}" if "median" in s else "—"
        iqr = f"{s['q25']:.1f}–{s['q75']:.1f}" if "q25" in s else "—"
        ci = f"{s['ci95_lo']:.1f}–{s['ci95_hi']:.1f}" if "ci95_lo" in s else "—"
        rows_tbl.append([str(g), str(n), med, iqr, ci])

    tbl = ax_tbl.table(
        cellText=rows_tbl, colLabels=header,
        cellLoc="center", loc="upper center",
        bbox=[0.0, 0.45, 1.0, 0.5],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(header))))

    # summary test text
    test_used = stats_results.get("test_used", "")
    if omnibus:
        test_str = (
            f"Test: {test_used}\n"
            f"KW H = {omnibus['statistic']:.3f},  p = {omnibus['p_value']:.3g}\n"
            f"eta² = {omnibus['eta_squared']:.3f}"
        )
    elif not pairwise.empty:
        row0 = pairwise.iloc[0]
        test_str = (
            f"Test: {test_used}\n"
            f"p_fdr = {row0.get('p_fdr', row0.get('p_value', float('nan'))):.3g}\n"
            f"Cohen's d = {row0.get('effect_size', float('nan')):.3f}"
        )
    else:
        test_str = f"Test: {test_used}"

    ax_tbl.text(
        0.5, 0.18, test_str, transform=ax_tbl.transAxes,
        ha="center", va="bottom", fontsize=8,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", linewidth=0.5),
    )
    ax_tbl.set_title("Group statistics", fontsize=9)

    fig.suptitle(f"Surface Area Summary — {sa_col} by {group_col}", fontsize=11, y=1.01)
    _save_figure(fig, output_path)
    return fig


def plot_method_breakdown(
    results_df: pd.DataFrame,
    group_col: str,
    output_path=None,
    include_flagged: bool = True,
) -> "matplotlib.figure.Figure":
    """
    Stacked bar chart showing the fraction of particles in each group that used
    each SA estimation tier.

    Useful for QC and reporting in the methods section of a paper: if a large
    fraction of one sample used fourier_spiky, it indicates more irregular
    particle shapes than the other samples.

    Returns matplotlib Figure.  Saved as PNG + SVG if output_path is given.
    """
    import matplotlib.pyplot as plt

    df = results_df.copy()
    if not include_flagged and "flagged" in df.columns:
        df = df[~df["flagged"]]

    group_order = sorted(df[group_col].unique().tolist())
    methods = ["ellipsoid", "cauchy", "fourier", "fourier_spiky"]

    fractions = {}
    for m in methods:
        fractions[m] = [
            float((df[df[group_col] == g]["method_used"] == m).sum())
            / max(float((df[group_col] == g).sum()), 1.0)
            for g in group_order
        ]

    fig, ax = plt.subplots(figsize=(max(4.5, len(group_order) * 1.1), 4.0))
    bottom = np.zeros(len(group_order))
    x = np.arange(len(group_order))

    for m in methods:
        vals = np.array(fractions[m])
        ax.bar(x, vals, bottom=bottom, label=m,
               color=_METHOD_COLORS.get(m, "#999999"), linewidth=0.4, edgecolor="white")
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(group_order, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Fraction of particles")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Estimation method usage by {group_col}")
    ax.legend(title="Method", fontsize=8, loc="upper right")
    _apply_publication_style(ax)
    fig.tight_layout()
    _save_figure(fig, output_path)
    return fig


# ── export helpers ────────────────────────────────────────────────────────────

def export_csv(df: pd.DataFrame, path: str) -> None:
    """Write the DataFrame to a CSV file."""
    df.to_csv(path, index=False)


def export_json(df: pd.DataFrame, path: str) -> None:
    """Write the DataFrame to a JSON file (records orientation)."""
    df.to_json(path, orient="records", indent=2)


def export_summary(df: pd.DataFrame, path: str) -> None:
    """
    Write a human-readable text summary to *path*.

    Includes overall statistics, method breakdown, flag breakdown,
    and hollow-particle report.
    """
    lines = ["ACORN Surface Area Summary", "=" * 40]
    s = summarize(df)
    lines += [
        f"Particles analysed : {s.get('n', 0)}",
        f"Valid SA estimates : {s.get('n_valid', 0)}",
        f"Mean SA            : {s.get('mean', float('nan')):.1f} nm2",
        f"Std SA             : {s.get('std', float('nan')):.1f} nm2",
        f"Median SA          : {s.get('median', float('nan')):.1f} nm2",
        f"P5 / P95           : {s.get('p5', float('nan')):.1f} / {s.get('p95', float('nan')):.1f} nm2",
        f"CV                 : {s.get('cv', float('nan')):.3f}",
        "",
        "Method breakdown:",
    ]
    for _, row in method_report(df).iterrows():
        lines.append(f"  {row['method_used']:<20} {row['count']:>5}  ({row['fraction']:.1%})")
    lines += ["", "Flag breakdown:"]
    for _, row in flag_report(df).iterrows():
        lines.append(f"  {row['flag_reason']:<25} {row['count']:>5}  ({row['fraction']:.1%})")
    hr = hollow_report(df)
    lines += [
        "",
        f"Hollow particles   : {hr['n_hollow']} ({hr['fraction_hollow']:.1%})",
        f"Mean shell thick.  : {hr['mean_shell_thickness_nm']:.1f} nm",
    ]
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def export_stats_report(
    results_df: pd.DataFrame,
    group_col: str,
    output_dir: str,
    sample_name: str = "sample",
    sa_col: str = "SA_nm2",
    include_flagged: bool = False,
    alpha: float = 0.05,
) -> dict:
    """
    Run all statistical tests and generate all publication figures in one call.

    Saved files
    -----------
    {sample_name}_sa_violin.png/.svg
    {sample_name}_sa_ecdf.png/.svg
    {sample_name}_sa_histogram.png/.svg
    {sample_name}_sa_scatter.png/.svg         (a_nm vs SA_nm2)
    {sample_name}_sa_summary_panel.png/.svg
    {sample_name}_method_breakdown.png/.svg
    {sample_name}_stats_summary.csv           (all test results + group stats)

    Parameters
    ----------
    results_df  : DataFrame from batch_surface_area()
    group_col   : grouping column
    output_dir  : directory for all outputs (created if it does not exist)
    sample_name : prefix for all output filenames
    sa_col      : SA column (default 'SA_nm2')
    include_flagged : passed to all functions
    alpha       : significance threshold

    Returns
    -------
    stats dict from compare_groups() (same structure).
    """
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend for batch export

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    stats_result = compare_groups(
        results_df, group_col, sa_col=sa_col,
        include_flagged=include_flagged, alpha=alpha,
    )

    figs_and_paths = [
        (plot_sa_distribution,
         dict(results_df=results_df, group_col=group_col, sa_col=sa_col,
              include_flagged=include_flagged, alpha=alpha, stats_results=stats_result),
         out / f"{sample_name}_sa_violin"),
        (plot_sa_ecdf,
         dict(results_df=results_df, group_col=group_col, sa_col=sa_col,
              include_flagged=include_flagged),
         out / f"{sample_name}_sa_ecdf"),
        (plot_sa_histogram,
         dict(results_df=results_df, group_col=group_col, sa_col=sa_col,
              include_flagged=include_flagged),
         out / f"{sample_name}_sa_histogram"),
        (plot_sa_summary_panel,
         dict(results_df=results_df, group_col=group_col, stats_results=stats_result,
              sa_col=sa_col, include_flagged=include_flagged, alpha=alpha),
         out / f"{sample_name}_sa_summary_panel"),
        (plot_method_breakdown,
         dict(results_df=results_df, group_col=group_col, include_flagged=include_flagged),
         out / f"{sample_name}_method_breakdown"),
    ]

    import matplotlib.pyplot as plt
    for fn, kwargs, path in figs_and_paths:
        try:
            fig = fn(output_path=path, **kwargs)
            plt.close(fig)
        except Exception as exc:
            logger.warning("plot %s failed: %s", path.name, exc)

    # scatter: only if a_nm column is present
    if "a_nm" in results_df.columns:
        try:
            fig = plot_sa_scatter(
                results_df, x_col="a_nm", y_col=sa_col, group_col=group_col,
                include_flagged=include_flagged,
                output_path=out / f"{sample_name}_sa_scatter",
            )
            plt.close(fig)
        except Exception as exc:
            logger.warning("scatter plot failed: %s", exc)

    # stats summary CSV
    rows: list[dict] = []
    for group_name, gstats in stats_result.get("group_stats", {}).items():
        rows.append({"row_type": "group_stats", "group": group_name, **gstats})
    pairwise = stats_result.get("pairwise", pd.DataFrame())
    if not pairwise.empty:
        for _, row in pairwise.iterrows():
            rows.append({"row_type": "pairwise", **row.to_dict()})
    omnibus = stats_result.get("omnibus")
    if omnibus:
        rows.append({"row_type": "omnibus", **omnibus})
    pd.DataFrame(rows).to_csv(out / f"{sample_name}_stats_summary.csv", index=False)

    return stats_result
