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
from dataclasses import dataclass, replace
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
    # Ellipse axes for non-circular truth (spores). None means a circle of
    # diameter_nm, which is also the shape used for overlap scoring.
    major_nm: float | None = None
    minor_nm: float | None = None
    angle_rad: float = 0.0
    # Size that sets the matching gate. Held at the object's outer extent so
    # that scoring a different edge definition changes the measured bias but
    # not which objects count as detected.
    match_diameter_nm: float | None = None
    # Hand-drawn reference outline (acquired data). When present it defines the
    # truth shape for overlap; diameter_nm is then its area-equivalent diameter.
    polygon_px: tuple[tuple[float, float], ...] | None = None


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
    # Outline in pixel coordinates when the method reports one; otherwise the
    # prediction is scored for overlap as a circle of diameter_nm.
    polygon_px: tuple[tuple[float, float], ...] | None = None


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
    match_mode: str = "iou",
    iou_threshold: float = 0.5,
    match_radius_fraction: float = 0.75,
) -> tuple[dict, list[dict]]:
    """Return summary and per-match rows for a method/specimen pair.

    ``match_mode="iou"`` pairs each truth with at most one prediction at
    IoU >= ``iou_threshold`` (maximum total IoU). ``"centroid"`` pairs by
    centre distance within ``match_radius_fraction`` of the true radius.
    Either way the overlap used for matching is taken against the object's
    outer extent, so the edge definition being scored cannot change which
    objects count as detected.
    """

    if match_mode not in ("iou", "centroid"):
        raise ValueError(f"match_mode must be 'iou' or 'centroid', got {match_mode!r}")
    truths = list(truth_objects)
    preds = list(predictions)
    matches = _match_by_stem(
        truths, preds, match_mode=match_mode, iou_threshold=iou_threshold, match_radius_fraction=match_radius_fraction
    )
    true_all = np.asarray([t.diameter_nm for t in truths], dtype=float)
    true_detected = np.asarray([m.truth.diameter_nm for m in matches], dtype=float)
    measured_detected = np.asarray([m.prediction.diameter_nm for m in matches], dtype=float)
    ious = np.asarray([object_iou(m.truth, m.prediction) for m in matches], dtype=float)

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
        "match_mode": match_mode,
        "match_iou_threshold": iou_threshold if match_mode == "iou" else None,
        "match_radius_fraction": match_radius_fraction if match_mode == "centroid" else None,
        "n_truth": len(truths),
        "n_predictions": len(preds),
        "n_matched": len(matches),
        "n_false_negative": len(truths) - len(matches),
        "n_false_positive": len(preds) - len(matches),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": _mean_or_nan(ious),
        "median_iou": float(np.median(ious)) if ious.size else math.nan,
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
            "iou": float(iou),
            "center_distance_px": m.center_distance_px,
            "truth_class": m.truth.class_name,
            "prediction_class": m.prediction.class_name,
            "prediction_score": m.prediction.score,
        }
        for m, iou in zip(matches, ious)
    ]
    return summary, rows


def object_iou(
    truth: TruthObject,
    prediction: Prediction,
    *,
    step_px: float = 0.25,
    samples_across: int = 100,
) -> float:
    """Overlap between one truth shape and one prediction, on a subpixel grid.

    Both shapes are rasterized on the same grid covering their union, so the
    result does not depend on the image pixel grid. The truth is the analytic
    circle or ellipse at the scored edge definition; the prediction is its
    outline if it has one, otherwise a circle of its reported diameter.

    The grid step is ``step_px`` or 1/``samples_across`` of the union's larger
    side, whichever is coarser: about 100 samples across any object keeps the
    area error near 1 %, and a fixed subpixel step would cost the square of the
    object size for large objects such as spores.
    """

    truth_inside, t_bounds = _truth_shape(truth)
    pred_inside, p_bounds = _prediction_shape(prediction, truth.pixel_size_nm)
    x0 = min(t_bounds[0], p_bounds[0])
    y0 = min(t_bounds[1], p_bounds[1])
    x1 = max(t_bounds[2], p_bounds[2])
    y1 = max(t_bounds[3], p_bounds[3])
    step_px = max(step_px, max(x1 - x0, y1 - y0) / samples_across)
    xs = np.arange(x0 + step_px / 2.0, x1, step_px)
    ys = np.arange(y0 + step_px / 2.0, y1, step_px)
    if xs.size == 0 or ys.size == 0:
        return math.nan
    gx, gy = np.meshgrid(xs, ys)
    a = truth_inside(gx, gy)
    b = pred_inside(gx, gy)
    union = np.count_nonzero(a | b)
    return float(np.count_nonzero(a & b) / union) if union else math.nan


def _polygon_tuple(points) -> tuple[tuple[float, float], ...] | None:
    if not points or len(points) < 3:
        return None
    return tuple((float(x), float(y)) for x, y in points)


def _truth_shape(truth: TruthObject):
    px = truth.pixel_size_nm
    if truth.polygon_px is not None:
        from matplotlib.path import Path as _MplPath

        pts = np.asarray(truth.polygon_px, dtype=float)
        path = _MplPath(pts)

        def inside(gx, gy):
            flat = np.column_stack([gx.ravel(), gy.ravel()])
            return path.contains_points(flat).reshape(gx.shape)

        return inside, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    if truth.major_nm is not None and truth.minor_nm is not None:
        a = truth.major_nm / (2.0 * px)
        b = truth.minor_nm / (2.0 * px)
        cos_t, sin_t = math.cos(truth.angle_rad), math.sin(truth.angle_rad)

        def inside(gx, gy):
            dx, dy = gx - truth.cx_px, gy - truth.cy_px
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            return (u / a) ** 2 + (v / b) ** 2 <= 1.0

        r = a
    else:
        r = truth.diameter_nm / (2.0 * px)

        def inside(gx, gy):
            return (gx - truth.cx_px) ** 2 + (gy - truth.cy_px) ** 2 <= r * r

    return inside, (truth.cx_px - r, truth.cy_px - r, truth.cx_px + r, truth.cy_px + r)


def _prediction_shape(prediction: Prediction, pixel_size_nm: float):
    if prediction.polygon_px is not None and len(prediction.polygon_px) >= 3:
        from matplotlib.path import Path as _MplPath

        pts = np.asarray(prediction.polygon_px, dtype=float)
        path = _MplPath(pts)

        def inside(gx, gy):
            flat = np.column_stack([gx.ravel(), gy.ravel()])
            return path.contains_points(flat).reshape(gx.shape)

        return inside, (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
    r = prediction.diameter_nm / (2.0 * pixel_size_nm)

    def inside(gx, gy):
        return (gx - prediction.cx_px) ** 2 + (gy - prediction.cy_px) ** 2 <= r * r

    return inside, (prediction.cx_px - r, prediction.cy_px - r, prediction.cx_px + r, prediction.cy_px + r)


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
                    major_nm=None if obj.get("major_nm") is None else float(obj["major_nm"]),
                    minor_nm=None if obj.get("minor_nm") is None else float(obj["minor_nm"]),
                    angle_rad=float(obj.get("angle_rad") or 0.0),
                    match_diameter_nm=float(obj.get("d_outer_leaflet_nm") or obj.get("ecd_nm") or diameter),
                    polygon_px=_polygon_tuple(obj.get("polygon_px")),
                )
            )
    return objects


def load_predictions(
    predictions: Path,
    truth_objects: Iterable[TruthObject],
    *,
    class_id: int | None = 0,
    class_name: str | None = None,
    exclude_border: bool = False,
) -> list[Prediction]:
    """Load predictions from YOLO labels, ACORN sidecars, or simple JSON.

    With ``exclude_border``, predictions reaching the image edge are dropped,
    matching the exclusion of truth objects that are not fully in frame.
    """

    path = Path(predictions)
    truth_by_stem = _truth_by_stem(truth_objects)
    if path.is_dir():
        yolo_paths = sorted(path.glob("*.txt"))
        if not yolo_paths:
            yolo_paths = sorted(path.glob("*/*.txt"))
        if yolo_paths:
            preds = _load_yolo_predictions(yolo_paths, truth_by_stem, class_id=class_id)
        else:
            json_paths = sorted(path.glob("*.json")) + sorted(path.glob("*/*.json"))
            preds = _load_json_predictions(json_paths, truth_by_stem, class_name=class_name)
    elif path.suffix.lower() == ".txt":
        preds = _load_yolo_predictions([path], truth_by_stem, class_id=class_id)
    else:
        preds = _load_json_predictions([path], truth_by_stem, class_name=class_name)
    if exclude_border:
        preds = [p for p in preds if not _touches_border(p, truth_by_stem[p.stem][0])]
    return preds


def _touches_border(prediction: Prediction, ref: TruthObject) -> bool:
    # Pixel i is centred at i, so the frame's outermost pixel centres are 0
    # and w - 1; an object reaching past them is clipped by the frame.
    h, w = ref.image_shape_px
    if prediction.polygon_px is not None and len(prediction.polygon_px) >= 3:
        pts = np.asarray(prediction.polygon_px, dtype=float)
        x0, y0 = pts.min(axis=0)
        x1, y1 = pts.max(axis=0)
    else:
        r = prediction.diameter_nm / (2.0 * ref.pixel_size_nm)
        x0, y0, x1, y1 = prediction.cx_px - r, prediction.cy_px - r, prediction.cx_px + r, prediction.cy_px + r
    return bool(x0 < 0.0 or y0 < 0.0 or x1 > w - 1.0 or y1 > h - 1.0)


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
    match_mode: str = "centroid",
    iou_threshold: float = 0.5,
    match_radius_fraction: float = 0.75,
) -> list[Match]:
    matches: list[Match] = []
    stems = sorted({t.stem for t in truths} | {p.stem for p in preds})
    for stem in stems:
        t_stem = [t for t in truths if t.stem == stem]
        p_stem = [p for p in preds if p.stem == stem]
        if not t_stem or not p_stem:
            continue
        accept = _acceptor(match_mode, iou_threshold, match_radius_fraction)
        direct_matches, t_stem, p_stem = _match_by_declared_truth_id(t_stem, p_stem, accept=accept)
        matches.extend(direct_matches)
        if not t_stem or not p_stem:
            continue
        costs = np.full((len(t_stem), len(p_stem)), 1e9, dtype=float)
        distances = np.full_like(costs, np.nan)
        for i, t in enumerate(t_stem):
            for j, p in enumerate(p_stem):
                dist = float(math.hypot(t.cx_px - p.cx_px, t.cy_px - p.cy_px))
                distances[i, j] = dist
                cost = accept(t, p, dist)
                if cost is not None:
                    costs[i, j] = cost
        row_ind, col_ind = linear_sum_assignment(costs)
        for i, j in zip(row_ind, col_ind):
            if costs[i, j] >= 1e9:
                continue
            matches.append(Match(t_stem[int(i)], p_stem[int(j)], float(distances[i, j])))
    return matches


def _acceptor(match_mode: str, iou_threshold: float, match_radius_fraction: float):
    """Return f(truth, prediction, centre distance) -> assignment cost, or None if not a candidate."""

    if match_mode == "centroid":

        def accept(t: TruthObject, p: Prediction, dist: float) -> float | None:
            gate = _gate_px(t, match_radius_fraction)
            return dist / gate if dist <= gate else None

        return accept

    def accept(t: TruthObject, p: Prediction, dist: float) -> float | None:
        # Cheap reject: shapes this far apart cannot overlap.
        r_t = (t.major_nm or t.match_diameter_nm or t.diameter_nm) / (2.0 * t.pixel_size_nm)
        if t.polygon_px is not None:
            pts_t = np.asarray(t.polygon_px, dtype=float)
            r_t = max(r_t, float(np.max(np.hypot(pts_t[:, 0] - t.cx_px, pts_t[:, 1] - t.cy_px))))
        r_p = p.diameter_nm / (2.0 * t.pixel_size_nm)
        if p.polygon_px is not None and len(p.polygon_px) >= 3:
            pts = np.asarray(p.polygon_px, dtype=float)
            r_p = max(r_p, float(np.max(np.hypot(pts[:, 0] - p.cx_px, pts[:, 1] - p.cy_px))))
        if dist > r_t + r_p:
            return None
        iou = object_iou(_outer_extent(t), p)
        return 1.0 - iou if np.isfinite(iou) and iou >= iou_threshold else None

    return accept


def _outer_extent(truth: TruthObject) -> TruthObject:
    if truth.match_diameter_nm is None or truth.major_nm is not None:
        return truth
    return replace(truth, diameter_nm=truth.match_diameter_nm)


def _match_by_declared_truth_id(
    truths: list[TruthObject],
    preds: list[Prediction],
    *,
    accept,
) -> tuple[list[Match], list[TruthObject], list[Prediction]]:
    # A prompted method names the object it was prompted on, but its mask
    # still has to land there: the same acceptance test applies as for every
    # other method, so a mask that wandered onto a neighbour is not credited.
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
        used_pred_ids.add(pred.prediction_id)
        dist = float(math.hypot(truth.cx_px - pred.cx_px, truth.cy_px - pred.cy_px))
        if accept(truth, pred, dist) is None:
            # Counted as a false positive; it is not offered to other truths.
            continue
        used_truth_ids.add(truth.object_id)
        matches.append(Match(truth, pred, dist))
    remaining_truths = [t for t in truths if t.object_id not in used_truth_ids]
    remaining_preds = [p for p in preds if p.prediction_id not in used_pred_ids]
    return matches, remaining_truths, remaining_preds


def _gate_px(truth: TruthObject, match_radius_fraction: float) -> float:
    gate_diameter_nm = truth.match_diameter_nm if truth.match_diameter_nm is not None else truth.diameter_nm
    return max(1.0, gate_diameter_nm / (2.0 * truth.pixel_size_nm)) * match_radius_fraction


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
        # ACORN writes its sidecar as a hidden file, .<stem>.acorn.json.
        fallback = path.name.removesuffix(".annotations.json").removesuffix(".acorn.json").removesuffix(".json").lstrip(".")
        stem = str(payload.get("stem") or fallback)
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
    polygon = record.get("polygon") or record.get("points") or record.get("vertices")
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
            polygon_px=tuple((float(x), float(y)) for x, y in polygon) if polygon and len(polygon) >= 3 else None,
        )
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
        polygon_px=tuple((float(x), float(y)) for x, y in points_px),
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
    parser.add_argument(
        "--include-edge-objects",
        action="store_true",
        help="Include truth objects clipped by the image edge, and predictions reaching it.",
    )
    parser.add_argument("--match-mode", choices=("iou", "centroid"), default="iou",
                        help="iou (primary): IoU >= --iou-threshold; centroid (SI check): centre distance.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
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
        exclude_border=not args.include_edge_objects,
    )
    summary, rows = score_bias(
        truths,
        predictions,
        method=args.method,
        specimen=args.specimen,
        match_mode=args.match_mode,
        iou_threshold=args.iou_threshold,
        match_radius_fraction=args.match_radius_fraction,
    )
    write_score_outputs(summary, rows, args.output)
    print(yaml.safe_dump(_jsonable(summary), sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
