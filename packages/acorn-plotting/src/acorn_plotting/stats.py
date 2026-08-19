"""Statistical analysis helpers — scipy only, no extra dependencies."""
from __future__ import annotations

from typing import Optional
import numpy as np


def _stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def run_statistics(df, metric: str, group_col: Optional[str] = "label") -> dict:
    """
    Run a full statistical summary for *metric* grouped by *group_col*.

    Returns a dict with keys:
        descriptive   : list of dicts, one per group
        normality     : list of dicts, one per group
        comparison    : dict with test name, statistic, p-value, significance
        posthoc       : list of pairwise dicts (if 3+ groups and significant)
        recommendation: plain-language string
    """
    from scipy import stats as scipy_stats

    if df is None or df.empty or metric not in df.columns:
        return {"error": f"Column '{metric}' not found in data."}

    # ── Build groups ─────────────────────────────────────────────────────────
    if group_col and group_col in df.columns:
        group_labels = sorted(df[group_col].dropna().unique().tolist())
        groups = {str(g): df[df[group_col] == g][metric].dropna().values
                  for g in group_labels}
    else:
        groups = {"all": df[metric].dropna().values}

    groups = {k: v for k, v in groups.items() if len(v) >= 3}
    if not groups:
        return {"error": "Need at least 3 values per group for statistics."}

    # ── Descriptive stats ────────────────────────────────────────────────────
    descriptive = []
    for name, vals in groups.items():
        q25, q75 = float(np.percentile(vals, 25)), float(np.percentile(vals, 75))
        descriptive.append({
            "group":  name,
            "n":      int(len(vals)),
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals, ddof=1)),
            "median": float(np.median(vals)),
            "iqr":    round(q75 - q25, 4),
            "min":    float(vals.min()),
            "max":    float(vals.max()),
            "q25":    q25,
            "q75":    q75,
        })

    # ── Normality tests ──────────────────────────────────────────────────────
    normality = []
    all_normal = True
    for name, vals in groups.items():
        if len(vals) < 3:
            normality.append({"group": name, "test": "skipped", "p": None, "normal": False})
            continue
        try:
            if len(vals) <= 5000:
                stat, p = scipy_stats.shapiro(vals)
                test_name = "Shapiro-Wilk"
            else:
                stat, p = scipy_stats.normaltest(vals)
                test_name = "D'Agostino-Pearson"
            normal = bool(p > 0.05)
            all_normal = all_normal and normal
            normality.append({
                "group": name, "test": test_name,
                "statistic": round(float(stat), 4),
                "p": round(float(p), 4),
                "normal": normal,
                "interpretation": "normal" if normal else "non-normal",
            })
        except Exception as exc:
            normality.append({"group": name, "test": "error", "p": None,
                              "normal": False, "error": str(exc)})
            all_normal = False

    # ── Comparison tests ─────────────────────────────────────────────────────
    group_list = list(groups.values())
    group_names = list(groups.keys())
    comparison = {}
    posthoc = []

    if len(groups) == 1:
        comparison = {"note": "Only one group — no comparison possible."}

    elif len(groups) == 2:
        a, b = group_list
        if all_normal:
            stat, p = scipy_stats.ttest_ind(a, b, equal_var=False)
            comparison = {
                "test": "Welch t-test", "statistic": round(float(stat), 4),
                "p": round(float(p), 4), "significance": _stars(p),
                "note": "Used because both groups appear normally distributed.",
            }
        else:
            stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
            comparison = {
                "test": "Mann-Whitney U", "statistic": round(float(stat), 4),
                "p": round(float(p), 4), "significance": _stars(p),
                "note": "Used because at least one group is non-normal.",
            }

    elif len(groups) >= 3:
        if all_normal:
            stat, p = scipy_stats.f_oneway(*group_list)
            comparison = {
                "test": "One-way ANOVA", "statistic": round(float(stat), 4),
                "p": round(float(p), 4), "significance": _stars(p),
                "note": "Used because all groups appear normally distributed.",
            }
            if p < 0.05:
                try:
                    from scipy.stats import tukey_hsd
                    res = tukey_hsd(*group_list)
                    for i in range(len(group_names)):
                        for j in range(i + 1, len(group_names)):
                            ph_p = float(res.pvalue[i, j])
                            posthoc.append({
                                "group_a": group_names[i],
                                "group_b": group_names[j],
                                "p": round(ph_p, 4),
                                "significance": _stars(ph_p),
                            })
                except Exception:
                    pass
        else:
            stat, p = scipy_stats.kruskal(*group_list)
            comparison = {
                "test": "Kruskal-Wallis", "statistic": round(float(stat), 4),
                "p": round(float(p), 4), "significance": _stars(p),
                "note": "Used because at least one group is non-normal.",
            }
            if p < 0.05:
                try:
                    from itertools import combinations
                    n_comp = len(list(combinations(range(len(group_list)), 2)))
                    for i, j in combinations(range(len(group_list)), 2):
                        _, ph_p = scipy_stats.mannwhitneyu(
                            group_list[i], group_list[j], alternative="two-sided")
                        ph_p_bonf = min(1.0, float(ph_p) * n_comp)
                        posthoc.append({
                            "group_a": group_names[i],
                            "group_b": group_names[j],
                            "p_bonferroni": round(ph_p_bonf, 4),
                            "significance": _stars(ph_p_bonf),
                        })
                except Exception:
                    pass

    # ── Correlation (if 2 numeric columns available) ─────────────────────────
    # (called separately via run_correlation)

    # ── Plain-language recommendation ────────────────────────────────────────
    n_groups = len(groups)
    if n_groups == 1:
        rec = "You have one group. Descriptive statistics and normality are shown above."
    elif n_groups == 2:
        test_used = comparison.get("test", "")
        sig = comparison.get("significance", "")
        p_val = comparison.get("p", None)
        if p_val is not None:
            rec = (
                f"Two groups compared using {test_used}. "
                f"p = {p_val} ({sig}). "
                + ("The difference is statistically significant." if sig != "ns"
                   else "No statistically significant difference detected.")
                + (" Both groups were normally distributed, so a parametric test was used."
                   if all_normal else
                   " At least one group was non-normal, so a non-parametric test was used.")
            )
        else:
            rec = "Could not run comparison."
    else:
        test_used = comparison.get("test", "")
        sig = comparison.get("significance", "")
        p_val = comparison.get("p", None)
        rec = (
            f"{n_groups} groups compared using {test_used}. "
            + (f"Overall p = {p_val} ({sig}). " if p_val is not None else "")
            + (f"Post-hoc pairwise comparisons shown below ({len(posthoc)} pairs)."
               if posthoc else "")
        )

    return {
        "metric":        metric,
        "descriptive":   descriptive,
        "normality":     normality,
        "all_normal":    all_normal,
        "comparison":    comparison,
        "posthoc":       posthoc,
        "recommendation": rec,
    }


def run_correlation(df, x: str, y: str) -> dict:
    """Pearson and Spearman correlation between two columns."""
    from scipy import stats as scipy_stats
    if x not in df.columns or y not in df.columns:
        return {"error": f"Column(s) not found: {x}, {y}"}
    sub = df[[x, y]].dropna()
    if len(sub) < 3:
        return {"error": "Need at least 3 paired values."}
    r, p_r = scipy_stats.pearsonr(sub[x], sub[y])
    rho, p_rho = scipy_stats.spearmanr(sub[x], sub[y])
    return {
        "x": x, "y": y, "n": len(sub),
        "pearson_r":   round(float(r),   4),
        "pearson_p":   round(float(p_r), 4),
        "spearman_rho": round(float(rho),   4),
        "spearman_p":  round(float(p_rho), 4),
        "interpretation": (
            f"Pearson r={r:.3f} (p={p_r:.4f}), Spearman ρ={rho:.3f} (p={p_rho:.4f}). "
            + ("Strong" if abs(r) > 0.7 else "Moderate" if abs(r) > 0.4 else "Weak")
            + f" {'positive' if r > 0 else 'negative'} linear correlation."
        ),
    }


def format_stats_report(result: dict) -> str:
    """Format a run_statistics result as plain text for display."""
    if "error" in result:
        return f"Error: {result['error']}"

    lines = [f"Statistical Analysis — {result['metric']}", "=" * 50, ""]

    # Descriptive
    lines.append("Descriptive Statistics")
    lines.append("-" * 30)
    for d in result.get("descriptive", []):
        lines.append(
            f"  {d['group']:20s}  n={d['n']:4d}  "
            f"mean={d['mean']:.2f}  std={d['std']:.2f}  "
            f"median={d['median']:.2f}  IQR={d['iqr']:.2f}  "
            f"[{d['min']:.2f}, {d['max']:.2f}]"
        )
    lines.append("")

    # Normality
    lines.append("Normality Tests")
    lines.append("-" * 30)
    for n in result.get("normality", []):
        p_str = f"p={n['p']:.4f}" if n.get("p") is not None else "skipped"
        lines.append(
            f"  {n['group']:20s}  {n.get('test',''):20s}  "
            f"{p_str}  → {n.get('interpretation', '')}"
        )
    lines.append("")

    # Comparison
    cmp = result.get("comparison", {})
    if cmp and "test" in cmp:
        lines.append("Comparison Test")
        lines.append("-" * 30)
        lines.append(
            f"  {cmp['test']}:  "
            f"statistic={cmp.get('statistic','?')}  "
            f"p={cmp.get('p','?')}  {cmp.get('significance','')}"
        )
        if cmp.get("note"):
            lines.append(f"  Note: {cmp['note']}")
        lines.append("")

    # Post-hoc
    ph = result.get("posthoc", [])
    if ph:
        lines.append("Post-hoc Pairwise Comparisons")
        lines.append("-" * 30)
        for row in ph:
            p_key = "p_bonferroni" if "p_bonferroni" in row else "p"
            lines.append(
                f"  {row['group_a']} vs {row['group_b']}:  "
                f"p={row[p_key]}  {row['significance']}"
            )
        lines.append("")

    # Summary
    lines.append("Summary")
    lines.append("-" * 30)
    lines.append(f"  {result.get('recommendation', '')}")

    return "\n".join(lines)
