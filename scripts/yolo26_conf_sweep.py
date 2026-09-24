"""Run a small YOLO26 confidence sweep for the analytical chemistry benchmark."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from acorn_sim_common.learned_pickers import predict_yolo
from acorn_sim_common.measurement_bias import (
    load_predictions,
    load_truth_dir,
    score_bias,
    write_score_outputs,
)


@dataclass(frozen=True)
class Spec:
    specimen_key: str
    specimen_label: str
    dataset: Path
    truth_dir: Path
    weights: Path
    prediction_class_name: str
    truth_class_name: str | None
    target_field: str
    class_id: int
    device: str


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path, default=Path("/home/vnw/analytical_chem_benchmarks/benchmark_20260914_120"))
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.35, 0.50])
    args = parser.parse_args()

    specs = build_specs(args.benchmark_root, args.run_root)
    rows: list[dict] = []
    for conf in args.confs:
        for spec in specs:
            pred_dir = args.run_root / "predictions" / f"{spec.specimen_key}_yolo26n_seg_conf{conf:.2f}"
            score_dir = args.run_root / "scored" / f"{spec.specimen_key}_yolo26n_seg_conf{conf:.2f}"
            predict_yolo(
                spec.dataset,
                spec.weights,
                pred_dir,
                class_name=spec.prediction_class_name,
                imgsz=256,
                conf=conf,
                iou=0.50,
                device=spec.device,
                class_id=spec.class_id,
                splits=("test",),
            )
            truths = load_truth_dir(
                spec.truth_dir,
                target_field=spec.target_field,
                class_name=spec.truth_class_name,
            )
            preds = load_predictions(
                pred_dir,
                truths,
                class_id=spec.class_id,
                class_name=spec.prediction_class_name,
            )
            summary, matched = score_bias(
                truths,
                preds,
                method=f"YOLO26n-seg conf={conf:.2f}",
                specimen=spec.specimen_label,
            )
            write_score_outputs(summary, matched, score_dir)
            row = {"confidence": conf, "score_dir": str(score_dir), **summary}
            rows.append(row)
            print(
                f"{spec.specimen_key} conf={conf:.2f}: "
                f"F1={summary['f1']:.3f}, total_bias={summary['total_bias_nm']:.3f} nm"
            )

    out_csv = args.run_root / "scored" / "yolo26_conf_sweep_summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_csv}")


def build_specs(benchmark_root: Path, run_root: Path) -> list[Spec]:
    weights_root = run_root / "models" / "yolo26_runs"
    return [
        Spec(
            specimen_key="plga",
            specimen_label="PLGA",
            dataset=benchmark_root / "plga",
            truth_dir=benchmark_root / "plga" / "truth" / "test",
            weights=weights_root / "plga_yolo26n_seg" / "weights" / "best.pt",
            prediction_class_name="nanoparticle",
            truth_class_name="nanoparticle",
            target_field="diameter_nm",
            class_id=0,
            device="1",
        ),
        Spec(
            specimen_key="vesicles",
            specimen_label="vesicles",
            dataset=benchmark_root / "vesicles",
            truth_dir=benchmark_root / "vesicles" / "truth" / "test",
            weights=weights_root / "vesicles_yolo26n_seg" / "weights" / "best.pt",
            prediction_class_name="vesicle",
            truth_class_name="vesicle",
            target_field="d_outer_leaflet_nm",
            class_id=0,
            device="2",
        ),
        Spec(
            specimen_key="spores",
            specimen_label="spores",
            dataset=benchmark_root / "spores",
            truth_dir=benchmark_root / "spores" / "truth" / "test",
            weights=weights_root / "spores_yolo26n_seg" / "weights" / "best.pt",
            prediction_class_name="spore",
            truth_class_name="spore",
            target_field="length_nm",
            class_id=0,
            device="3",
        ),
    ]


if __name__ == "__main__":
    main()
