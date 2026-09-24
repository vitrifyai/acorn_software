"""Plot measurement-bias benchmark summaries."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


def collect_score_summaries(scored_dir: Path, *, names: tuple[str, ...] | None = None) -> list[dict]:
    scored_dir = Path(scored_dir)
    name_filter = set(names) if names is not None else None
    rows: list[dict] = []
    for path in sorted(scored_dir.glob("*/bias_summary.json")):
        if name_filter is not None and path.parent.name not in name_filter:
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        row["score_dir"] = path.parent.name
        rows.append(row)
    return rows


def write_summary_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bias_summary(rows: list[dict], output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_rows = [r for r in rows if _finite(r.get("f1")) and _finite(r.get("total_bias_nm"))]
    if not plot_rows:
        raise ValueError("no finite F1/total-bias rows to plot")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax_scatter, ax_bar = axes

    colors = {"plga": "#3b7ddd", "vesicle": "#20a486", "spore": "#d95f02"}
    markers = {"cryoblob-log": "o", "log": "o", "threshold-watershed": "s", "yolo11n_seg_rgb": "^", "sam_point": "D"}

    for row in plot_rows:
        specimen = str(row.get("specimen", ""))
        method = str(row.get("method", ""))
        ax_scatter.scatter(
            float(row["f1"]),
            float(row["total_bias_nm"]),
            s=72,
            color=colors.get(specimen, "#666666"),
            marker=markers.get(method, "o"),
            edgecolor="black",
            linewidth=0.5,
            alpha=0.9,
        )
        ax_scatter.annotate(
            _short_label(row),
            (float(row["f1"]), float(row["total_bias_nm"])),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax_scatter.axhline(0, color="#555555", linewidth=0.8)
    ax_scatter.set_xlabel("Detection F1")
    ax_scatter.set_ylabel("Total size bias (nm)")
    ax_scatter.set_title("Detection score does not license size accuracy")
    ax_scatter.set_xlim(-0.03, 1.03)
    ax_scatter.grid(True, linewidth=0.4, alpha=0.3)

    bar_rows = [r for r in rows if _finite(r.get("B_det_nm")) and _finite(r.get("B_del_nm"))]
    labels = [_short_label(r) for r in bar_rows]
    x = list(range(len(bar_rows)))
    b_det = [float(r["B_det_nm"]) for r in bar_rows]
    b_del = [float(r["B_del_nm"]) for r in bar_rows]
    ax_bar.bar(x, b_det, label="B_det", color="#8da0cb")
    bottoms = [v if v > 0 else 0.0 for v in b_det]
    neg_bottoms = [v if v < 0 else 0.0 for v in b_det]
    pos_del = [v if v >= 0 else 0.0 for v in b_del]
    neg_del = [v if v < 0 else 0.0 for v in b_del]
    ax_bar.bar(x, pos_del, bottom=bottoms, label="B_del", color="#fc8d62")
    ax_bar.bar(x, neg_del, bottom=neg_bottoms, color="#fc8d62")
    ax_bar.axhline(0, color="#555555", linewidth=0.8)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax_bar.set_ylabel("Bias contribution (nm)")
    ax_bar.set_title("Total bias decomposes into missed-object and boundary terms")
    ax_bar.legend(frameon=False)
    ax_bar.grid(True, axis="y", linewidth=0.4, alpha=0.3)

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _short_label(row: dict) -> str:
    specimen = str(row.get("specimen", ""))
    method = str(row.get("method", ""))
    method_label = {
        "cryoblob-log": "CryoBlob",
        "threshold-watershed": "watershed",
        "yolo11n_seg_rgb": "YOLO",
        "sam_point": "SAM-point",
        "log": "LoG",
    }.get(method, method)
    specimen_label = {"plga": "PLGA", "vesicle": "ves", "spore": "spore"}.get(specimen, specimen)
    return f"{specimen_label} {method_label}"


def _finite(value: object) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot F1 versus size bias and B_det/B_del decomposition.")
    parser.add_argument("--scored-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--include", nargs="+", default=None, help="Score directory names to include.")
    args = parser.parse_args(argv)

    rows = collect_score_summaries(args.scored_dir, names=None if args.include is None else tuple(args.include))
    if args.summary_csv is not None:
        write_summary_csv(rows, args.summary_csv)
    plot_bias_summary(rows, args.output)
    print(f"Wrote {args.output}")
    if args.summary_csv is not None:
        print(f"Wrote {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
