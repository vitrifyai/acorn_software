"""Score detection and size-measurement bias against simulated object truth.

This is the bridge between segmentation outputs and the paper's measurement
claim. It separates population bias from missed objects (B_det) and boundary or
measurement bias on detected objects (B_del).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class TruthObject:
    stem: str
    object_id: str
    cx_px: float
    cy_px: float
    diameter_nm: float
    pixel_size_nm: float
    image_shape_px: tuple[int, int]
    class_name: str = ""
    fully_in_frame: bool = True


@dataclass(frozen=True)
class Prediction:
    stem: str
    prediction_id: str
    cx_px: float
    cy_px: float
    diameter_nm: float
    class_name: str = ""
    score: float | None = None
    truth_id: str | None = None


@dataclass(frozen=True)
class Match:
    truth: TruthObject
    prediction: Prediction
    center_distance_px: float


def score_bias(
    truth_objects: Iterable[TruthObject],
    predictions: Iterable[Prediction],
    *,
    method: str = "method",
    specimen: str = "",
    match_radius_fraction: float = 0.75,
) -> tuple[dict, list[dict]]:
    """Return summary and per-match rows for a method/specimen pair."""

    truths = list(truth_objects)
    preds = list(predictions)
    matches = _match_by_stem(truths, preds, match_radius_fraction=match_radius_fraction)
    true_all = np.asarray([t.diameter_nm for t in truths], dtype=float)
    true_detected = np.asarray([m.truth.diameter_nm for m in matches], dtype=float)
    measured_detected = np.asarray([m.prediction.diameter_nm for m in matches], dtype=float)

    b_det = _mean_or_nan(true_detected) - _mean_or_nan(true_all)
    b_del = _mean_or_nan(measured_detected) - _mean_or_nan(true_detected)
    total = _mean_or_nan(measured_detected) - _mean_or_nan(true_all)
    precision = len(matches) / len(preds) if preds else math.nan
    recall = len(matches) / len(truths) if truths else math.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else math.nan
    )
    summary = {
        "method": method,
        "specimen": specimen,
        "n_truth": len(truths),
        "n_predictions": len(preds),
        "n_matched": len(matches),
        "n_false_negative": len(truths) - len(matches),
        "n_false_positive": len(preds) - len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "truth_mean_all_nm": _mean_or_nan(true_all),
        "truth_mean_detected_nm": _mean_or_nan(true_detected),
        "measured_mean_detected_nm": _mean_or_nan(measured_detected),
        "B_det_nm": b_det,
        "B_del_nm": b_del,
        "total_bias_nm": total,
        "closure_error_nm": total - (b_det + b_del) if np.isfinite(total + b_det + b_del) else math.nan,
    }
    rows = [
        {
            "method": method,
            "specimen": specimen,
            "stem": m.truth.stem,
            "truth_id": m.truth.object_id,
            "prediction_id": m.prediction.prediction_id,
            "truth_diameter_nm": m.truth.diameter_nm,
            "measured_diameter_nm": m.prediction.diameter_nm,
            "measurement_error_nm": m.prediction.diameter_nm - m.truth.diameter_nm,
            "center_distance_px": m.center_distance_px,
            "truth_class": m.truth.class_name,
            "prediction_class": m.prediction.class_name,
            "prediction_score": m.prediction.score,
        }
        for m in matches
    ]
    return summary, rows


def load_truth_dir(
    truth_dir: Path,
    *,
    target_field: str = "diameter_nm",
    class_name: str | None = None,
    require_fully_in_frame: bool = True,
) -> list[TruthObject]:
    """Load per-image simulator truth JSON files."""

    truth_dir = Path(truth_dir)
    paths = sorted(truth_dir.glob("*.json"))
    if not paths:
        paths = sorted(truth_dir.glob("*/*.json"))
    objects: list[TruthObject] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stem = str(payload.get("stem") or path.stem)
        pixel_size_nm = float(payload.get("pixel_size_nm") or payload.get("px_nm"))
        shape = _shape_from_payload(payload)
        for obj in payload.get("objects", []):
            obj_class = str(obj.get("class_name") or obj.get("label") or "")
            if class_name is not None and obj_class and obj_class != class_name:
                continue
            fully_in_frame = bool(obj.get("fully_in_frame", True))
            if require_fully_in_frame and not fully_in_frame:
                continue
            diameter = _object_diameter_nm(obj, target_field)
            objects.append(
                TruthObject(
                    stem=stem,
                    object_id=str(obj.get("id", len(objects) + 1)),
                    cx_px=float(obj["cx_px"]),
                    cy_px=float(obj["cy_px"]),
                    diameter_nm=diameter,
                    pixel_size_nm=pixel_size_nm,
                    image_shape_px=shape,
                    class_name=obj_class,
                    fully_in_frame=fully_in_frame,
                )
            )
    return objects


def load_predictions(
    predictions: Path,
    truth_objects: Iterable[TruthObject],
    *,
    class_id: int | None = 0,
    class_name: str | None = None,
) -> list[Prediction]:
    """Load predictions from YOLO labels, ACORN sidecars, or simple JSON."""

    path = Path(predictions)
    truth_by_stem = _truth_by_stem(truth_objects)
    if path.is_dir():
        yolo_paths = sorted(path.glob("*.txt"))
        if not yolo_paths:
            yolo_paths = sorted(path.glob("*/*.txt"))
        if yolo_paths:
            return _load_yolo_predictions(yolo_paths, truth_by_stem, class_id=class_id)

        json_paths = sorted(path.glob("*.json")) + sorted(path.glob("*/*.json"))
        return _load_json_predictions(json_paths, truth_by_stem, class_name=class_name)

    if path.suffix.lower() == ".txt":
        return _load_yolo_predictions([path], truth_by_stem, class_id=class_id)
    return _load_json_predictions([path], truth_by_stem, class_name=class_name)


def write_score_outputs(summary: dict, rows: list[dict], output_dir: Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "bias_summary.json").write_text(json.dumps(_jsonable(summary), indent=2), encoding="utf-8")
    _write_csv(output_dir / "bias_summary.csv", [summary])
    _write_csv(output_dir / "matched_objects.csv", rows)


def _match_by_stem(
    truths: list[TruthObject],
    preds: list[Prediction],
    *,
    match_radius_fraction: float,
) -> list[Match]:
    matches: list[Match] = []
    stems = sorted({t.stem for t in truths} | {p.stem for p in preds})
    for stem in stems:
        t_stem = [t for t in truths if t.stem == stem]
        p_stem = [p for p in preds if p.stem == stem]
        if not t_stem or not p_stem:
            continue
        direct_matches, t_stem, p_stem = _match_by_declared_truth_id(t_stem, p_stem)
        matches.extend(direct_matches)
        if not t_stem or not p_stem:
            continue
        costs = np.full((len(t_stem), len(p_stem)), 1e9, dtype=float)
        distances = np.full_like(costs, np.nan)
        for i, t in enumerate(t_stem):
            radius_px = max(1.0, t.diameter_nm / (2.0 * t.pixel_size_nm))
            gate = radius_px * match_radius_fraction
            for j, p in enumerate(p_stem):
                dist = float(math.hypot(t.cx_px - p.cx_px, t.cy_px - p.cy_px))
                distances[i, j] = dist
                if dist <= gate:
                    costs[i, j] = dist / gate
        row_ind, col_ind = linear_sum_assignment(costs)
        for i, j in zip(row_ind, col_ind):
            if costs[i, j] >= 1e9:
                continue
            matches.append(Match(t_stem[int(i)], p_stem[int(j)], float(distances[i, j])))
    return matches


def _match_by_declared_truth_id(
    truths: list[TruthObject],
    preds: list[Prediction],
) -> tuple[list[Match], list[TruthObject], list[Prediction]]:
    truth_by_id = {t.object_id: t for t in truths}
    used_truth_ids: set[str] = set()
    used_pred_ids: set[str] = set()
    matches: list[Match] = []
    for pred in preds:
        if pred.truth_id is None:
            continue
        truth = truth_by_id.get(pred.truth_id)
        if truth is None or truth.object_id in used_truth_ids:
            continue
        used_truth_ids.add(truth.object_id)
        used_pred_ids.add(pred.prediction_id)
        dist = float(math.hypot(truth.cx_px - pred.cx_px, truth.cy_px - pred.cy_px))
        matches.append(Match(truth, pred, dist))
    remaining_truths = [t for t in truths if t.object_id not in used_truth_ids]
    remaining_preds = [p for p in preds if p.prediction_id not in used_pred_ids]
    return matches, remaining_truths, remaining_preds


def _load_yolo_predictions(
    paths: list[Path],
    truth_by_stem: dict[str, list[TruthObject]],
    *,
    class_id: int | None,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for path in paths:
        stem = path.stem
        ref = _reference_truth(stem, truth_by_stem)
        if ref is None:
            continue
        h, w = ref.image_shape_px
        for line_index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            if class_id is not None and cls != class_id:
                continue
            coords = [float(v) for v in parts[1:]]
            if len(coords) % 2 != 0:
                continue
            pts = np.asarray(coords, dtype=float).reshape(-1, 2)
            if pts.size == 0:
                continue
            if float(np.nanmax(pts)) <= 1.5:
                pts[:, 0] *= w
                pts[:, 1] *= h
            predictions.append(_prediction_from_polygon(stem, f"{stem}:{line_index}", pts, ref.pixel_size_nm, str(cls)))
    return predictions


def _load_json_predictions(
    paths: list[Path],
    truth_by_stem: dict[str, list[TruthObject]],
    *,
    class_name: str | None,
) -> list[Prediction]:
    predictions: list[Prediction] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = _prediction_records(payload)
        stem = str(payload.get("stem") or path.name.removesuffix(".annotations.json").removesuffix(".acorn.json").removesuffix(".json"))
        ref = _reference_truth(stem, truth_by_stem)
        if ref is None:
            continue
        for index, record in enumerate(records):
            label = str(record.get("class_name") or record.get("label") or record.get("name") or "")
            if class_name is not None and label and label != class_name:
                continue
            pred = _prediction_from_record(stem, f"{stem}:{index}", record, ref)
            if pred is not None:
                predictions.append(pred)
    return predictions


def _prediction_from_record(
    stem: str,
    prediction_id: str,
    record: dict,
    ref: TruthObject,
) -> Prediction | None:
    label = str(record.get("class_name") or record.get("label") or record.get("name") or "")
    score = record.get("score") or record.get("confidence")
    score = float(score) if score is not None else None
    diameter = record.get("diameter_nm") or record.get("ecd_nm") or record.get("equivalent_diameter_nm")
    if "cx_px" in record and "cy_px" in record and diameter is not None:
        return Prediction(
            stem=stem,
            prediction_id=prediction_id,
            cx_px=float(record["cx_px"]),
            cy_px=float(record["cy_px"]),
            diameter_nm=float(diameter),
            class_name=label,
            score=score,
            truth_id=None if record.get("truth_id") is None else str(record.get("truth_id")),
        )
    polygon = record.get("polygon") or record.get("points") or record.get("vertices")
    if polygon:
        return _prediction_from_polygon(stem, prediction_id, np.asarray(polygon, dtype=float), ref.pixel_size_nm, label, score)
    bbox = record.get("bbox_xyxy_px") or record.get("bbox") or record.get("box")
    if bbox and len(bbox) >= 4:
        x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        diam = float(diameter) if diameter is not None else 2.0 * math.sqrt(abs((x1 - x0) * (y1 - y0)) / math.pi) * ref.pixel_size_nm
        return Prediction(stem, prediction_id, (x0 + x1) / 2.0, (y0 + y1) / 2.0, diam, label, score)
    return None


def _prediction_from_polygon(
    stem: str,
    prediction_id: str,
    points_px: np.ndarray,
    pixel_size_nm: float,
    class_name: str,
    score: float | None = None,
) -> Prediction:
    area_px2 = abs(_polygon_area(points_px))
    diameter_nm = 2.0 * math.sqrt(max(area_px2, 0.0) / math.pi) * pixel_size_nm
    return Prediction(
        stem=stem,
        prediction_id=prediction_id,
        cx_px=float(np.mean(points_px[:, 0])),
        cy_px=float(np.mean(points_px[:, 1])),
        diameter_nm=diameter_nm,
        class_name=class_name,
        score=score,
    )


def _prediction_records(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("predictions", "detections", "objects", "annotations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [r for r in value if isinstance(r, dict)]
    return [payload]


def _object_diameter_nm(obj: dict, target_field: str) -> float:
    for key in (target_field, "diameter_nm", "ecd_nm", "d_outer_leaflet_nm", "length_nm"):
        if key in obj and obj[key] is not None:
            return float(obj[key])
    if "width_nm" in obj:
        return float(obj["width_nm"])
    raise KeyError(f"truth object has no diameter field {target_field!r}")


def _shape_from_payload(payload: dict) -> tuple[int, int]:
    shape = payload.get("image_shape_px") or payload.get("shape")
    if shape and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    size = payload.get("image_size_px") or payload.get("size_px")
    if size:
        return int(size), int(size)
    raise KeyError("truth JSON must include image_shape_px or image_size_px")


def _truth_by_stem(truth_objects: Iterable[TruthObject]) -> dict[str, list[TruthObject]]:
    by_stem: dict[str, list[TruthObject]] = {}
    for truth in truth_objects:
        by_stem.setdefault(truth.stem, []).append(truth)
    return by_stem


def _reference_truth(stem: str, truth_by_stem: dict[str, list[TruthObject]]) -> TruthObject | None:
    items = truth_by_stem.get(stem)
    if items:
        return items[0]
    return None


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _mean_or_nan(values: np.ndarray) -> float:
    if values.size == 0:
        return math.nan
    return float(np.mean(values))


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(_jsonable(row))


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score B_det, B_del, and total size bias from simulator truth.")
    parser.add_argument("--truth-dir", required=True, type=Path, help="Directory containing per-image truth JSON files.")
    parser.add_argument("--predictions", required=True, type=Path, help="YOLO labels, ACORN sidecars, or JSON predictions.")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for summary and matched rows.")
    parser.add_argument("--target-field", default="diameter_nm", help="Truth diameter field to score.")
    parser.add_argument("--method", default="method")
    parser.add_argument("--specimen", default="")
    parser.add_argument("--class-id", type=int, default=0, help="YOLO class id to score; use -1 for all classes.")
    parser.add_argument("--class-name", default=None, help="Class name to score in JSON/ACORN prediction files.")
    parser.add_argument("--include-edge-objects", action="store_true", help="Include truth objects clipped by the image edge.")
    parser.add_argument("--match-radius-fraction", type=float, default=0.75)
    args = parser.parse_args(argv)

    truths = load_truth_dir(
        args.truth_dir,
        target_field=args.target_field,
        class_name=args.class_name,
        require_fully_in_frame=not args.include_edge_objects,
    )
    predictions = load_predictions(
        args.predictions,
        truths,
        class_id=None if args.class_id < 0 else args.class_id,
        class_name=args.class_name,
    )
    summary, rows = score_bias(
        truths,
        predictions,
        method=args.method,
        specimen=args.specimen,
        match_radius_fraction=args.match_radius_fraction,
    )
    write_score_outputs(summary, rows, args.output)
    print(yaml.safe_dump(_jsonable(summary), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
