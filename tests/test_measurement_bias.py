from __future__ import annotations

import json

import numpy as np
import pytest

from acorn_sim_common.measurement_bias import (
    Prediction,
    TruthObject,
    object_iou,
    load_predictions,
    load_truth_dir,
    score_bias,
    write_score_outputs,
)


def _write_truth(path):
    path.mkdir(parents=True)
    (path / "field_000.json").write_text(
        json.dumps(
            {
                "stem": "field_000",
                "image_shape_px": [100, 100],
                "pixel_size_nm": 1.0,
                "objects": [
                    {
                        "id": 1,
                        "class_name": "particle",
                        "cx_px": 25,
                        "cy_px": 50,
                        "diameter_nm": 10.0,
                        "fully_in_frame": True,
                    },
                    {
                        "id": 2,
                        "class_name": "particle",
                        "cx_px": 75,
                        "cy_px": 50,
                        "diameter_nm": 30.0,
                        "fully_in_frame": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_score_bias_decomposes_missed_and_boundary_bias(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    _write_truth(truth_dir)
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "field_000.json").write_text(
        json.dumps(
            {
                "stem": "field_000",
                "predictions": [
                    {
                        "class_name": "particle",
                        "cx_px": 75,
                        "cy_px": 50,
                        "diameter_nm": 36.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    truths = load_truth_dir(truth_dir, target_field="diameter_nm", class_name="particle")
    predictions = load_predictions(pred_dir, truths, class_name="particle")
    summary, rows = score_bias(truths, predictions, method="toy", specimen="particles")

    assert summary["n_truth"] == 2
    assert summary["n_predictions"] == 1
    assert summary["n_matched"] == 1
    assert summary["B_det_nm"] == pytest.approx(10.0)
    assert summary["B_del_nm"] == pytest.approx(6.0)
    assert summary["total_bias_nm"] == pytest.approx(16.0)
    assert summary["closure_error_nm"] == pytest.approx(0.0)
    assert rows[0]["measurement_error_nm"] == pytest.approx(6.0)


def test_load_yolo_predictions_and_write_outputs(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    _write_truth(truth_dir)
    labels = tmp_path / "labels"
    labels.mkdir()
    labels.joinpath("field_000.txt").write_text(
        "0 0.70000 0.50000 0.75000 0.45000 0.80000 0.50000 0.75000 0.55000\n",
        encoding="utf-8",
    )

    truths = load_truth_dir(truth_dir)
    predictions = load_predictions(labels, truths, class_id=0)
    summary, rows = score_bias(truths, predictions, method="yolo", specimen="particles", match_mode="centroid")
    out = tmp_path / "scored"
    write_score_outputs(summary, rows, out)

    assert len(predictions) == 1
    assert summary["n_matched"] == 1
    assert predictions[0].diameter_nm > 0
    assert (out / "bias_summary.json").exists()
    assert (out / "bias_summary.csv").exists()
    assert (out / "matched_objects.csv").exists()


def test_load_json_predictions_recurses_past_root_manifest(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    _write_truth(truth_dir)
    pred_dir = tmp_path / "pred"
    split_dir = pred_dir / "test"
    split_dir.mkdir(parents=True)
    (pred_dir / "manifest.json").write_text('{"n_images": 1}', encoding="utf-8")
    (split_dir / "field_000.json").write_text(
        json.dumps(
            {
                "stem": "field_000",
                "predictions": [
                    {
                        "class_name": "particle",
                        "cx_px": 75,
                        "cy_px": 50,
                        "diameter_nm": 30.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    truths = load_truth_dir(truth_dir, class_name="particle")
    predictions = load_predictions(pred_dir, truths, class_name="particle")

    assert len(predictions) == 1
    assert predictions[0].stem == "field_000"


def test_declared_truth_id_matches_prompted_predictions(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    _write_truth(truth_dir)
    pred_dir = tmp_path / "pred"
    pred_dir.mkdir()
    (pred_dir / "field_000.json").write_text(
        json.dumps(
            {
                "stem": "field_000",
                "predictions": [
                    {
                        "class_name": "particle",
                        "truth_id": "1",
                        "cx_px": 27,
                        "cy_px": 51,
                        "diameter_nm": 14.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    truths = load_truth_dir(truth_dir, target_field="diameter_nm", class_name="particle")
    predictions = load_predictions(pred_dir, truths, class_name="particle")
    summary, rows = score_bias(truths, predictions, method="sam-point", specimen="particles")

    assert summary["n_matched"] == 1
    assert rows[0]["truth_id"] == "1"
    assert rows[0]["measurement_error_nm"] == pytest.approx(4.0)


def _circle_truth(diameter_nm: float, **kwargs) -> TruthObject:
    return TruthObject(
        stem="f", object_id="1", cx_px=50.0, cy_px=50.0, diameter_nm=diameter_nm,
        pixel_size_nm=1.0, image_shape_px=(100, 100), **kwargs,
    )


def test_object_iou_concentric_circles() -> None:
    truth = _circle_truth(20.0)
    smaller = Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=18.0)
    assert object_iou(truth, smaller) == pytest.approx((18.0 / 20.0) ** 2, abs=0.01)
    same = Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=20.0)
    assert object_iou(truth, same) == pytest.approx(1.0)


def test_object_iou_uses_prediction_polygon_and_truth_ellipse() -> None:
    truth = _circle_truth(20.0, major_nm=40.0, minor_nm=20.0, angle_rad=0.0)
    theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
    outline = tuple((50.0 + 20.0 * np.cos(t), 50.0 + 10.0 * np.sin(t)) for t in theta)
    pred = Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=0.0, polygon_px=outline)
    assert object_iou(truth, pred) == pytest.approx(1.0, abs=0.01)
    rotated = TruthObject(**{**truth.__dict__, "angle_rad": np.pi / 2.0})
    assert object_iou(rotated, pred) < 0.5


def test_high_overlap_can_hide_directional_bias() -> None:
    # The paper's thought experiment: every boundary one pixel inside scores a
    # high IoU but biases the mean diameter by two pixels.
    truths = [_circle_truth(30.0)]
    inward = [Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=28.0)]
    summary, rows = score_bias(truths, inward)
    assert summary["mean_iou"] > 0.85
    assert summary["B_del_nm"] == pytest.approx(-2.0)
    assert rows[0]["iou"] == pytest.approx(summary["mean_iou"])


def test_edge_definition_changes_bias_not_detection(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    (truth_dir / "v.json").write_text(
        json.dumps(
            {
                "stem": "v",
                "image_shape_px": [100, 100],
                "pixel_size_nm": 1.0,
                "objects": [
                    {"id": 1, "class_name": "vesicle", "cx_px": 50, "cy_px": 50,
                     "d_outer_leaflet_nm": 40.0, "d_bilayer_mid_nm": 36.0, "d_inner_leaflet_nm": 32.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    # Off-centre by 13 px: inside 0.75 x the outer radius, outside it for the inner one.
    pred = [Prediction(stem="v", prediction_id="p", cx_px=63.0, cy_px=50.0, diameter_nm=40.0)]
    results = {}
    for field in ("d_outer_leaflet_nm", "d_inner_leaflet_nm"):
        truths = load_truth_dir(truth_dir, target_field=field)
        results[field], _ = score_bias(truths, pred, match_mode="centroid")
    assert results["d_outer_leaflet_nm"]["n_matched"] == results["d_inner_leaflet_nm"]["n_matched"] == 1
    assert results["d_inner_leaflet_nm"]["B_del_nm"] == pytest.approx(8.0)


def test_hidden_acorn_sidecar_scores_against_its_field(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    _write_truth(truth_dir)
    ann = tmp_path / "returned"
    ann.mkdir()
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    ring = [[75.0 + 15.0 * np.cos(t), 50.0 + 15.0 * np.sin(t)] for t in theta]
    (ann / ".field_000.acorn.json").write_text(
        json.dumps({"version": 4, "annotations": [{"type": "roi", "label": "particle", "vertices": ring}]}),
        encoding="utf-8",
    )
    truths = load_truth_dir(truth_dir, class_name="particle")
    preds = load_predictions(ann, truths, class_name="particle")
    summary, _ = score_bias(truths, preds)
    assert len(preds) == 1 and preds[0].stem == "field_000"
    assert summary["n_matched"] == 1
    assert summary["B_del_nm"] == pytest.approx(0.0, abs=0.2)
    assert summary["mean_iou"] > 0.97


def test_declared_truth_id_still_has_to_land_on_its_object() -> None:
    truths = [_circle_truth(20.0)]
    wandered = [Prediction(stem="f", prediction_id="p", cx_px=80.0, cy_px=80.0, diameter_nm=20.0, truth_id="1")]
    summary, _ = score_bias(truths, wandered)
    assert summary["n_matched"] == 0
    assert summary["n_false_positive"] == 1


def test_iou_matching_uses_outer_extent_for_every_edge_definition(tmp_path) -> None:
    truth_dir = tmp_path / "truth"
    truth_dir.mkdir()
    (truth_dir / "v.json").write_text(
        json.dumps(
            {
                "stem": "v",
                "image_shape_px": [100, 100],
                "pixel_size_nm": 1.0,
                "objects": [
                    {"id": 1, "class_name": "vesicle", "cx_px": 50, "cy_px": 50,
                     "d_outer_leaflet_nm": 40.0, "d_bilayer_mid_nm": 33.0, "d_inner_leaflet_nm": 26.0},
                ],
            }
        ),
        encoding="utf-8",
    )
    pred = [Prediction(stem="v", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=40.0)]
    inner = load_truth_dir(truth_dir, target_field="d_inner_leaflet_nm")
    summary, rows = score_bias(inner, pred, match_mode="iou", iou_threshold=0.5)
    # IoU against the inner leaflet is (26/40)^2 = 0.42, below threshold, yet
    # the object is detected because matching uses its outer extent.
    assert summary["n_matched"] == 1
    assert rows[0]["iou"] == pytest.approx((26.0 / 40.0) ** 2, abs=0.01)
    assert summary["B_del_nm"] == pytest.approx(14.0)


def test_iou_matching_rejects_low_overlap() -> None:
    truths = [_circle_truth(20.0)]
    offset = [Prediction(stem="f", prediction_id="p", cx_px=58.0, cy_px=50.0, diameter_nm=20.0)]
    assert score_bias(truths, offset, match_mode="iou")[0]["n_matched"] == 0
    assert score_bias(truths, offset, match_mode="centroid")[0]["n_matched"] == 0
    near = [Prediction(stem="f", prediction_id="p", cx_px=52.0, cy_px=50.0, diameter_nm=20.0)]
    assert score_bias(truths, near, match_mode="iou")[0]["n_matched"] == 1


def test_polygon_truth_defines_overlap() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
    square = ((40.0, 40.0), (60.0, 40.0), (60.0, 60.0), (40.0, 60.0))
    truth = TruthObject(stem="f", object_id="1", cx_px=50.0, cy_px=50.0,
                        diameter_nm=2.0 * np.sqrt(400.0 / np.pi), pixel_size_nm=1.0,
                        image_shape_px=(100, 100), polygon_px=square)
    same = Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=0.0, polygon_px=square)
    assert object_iou(truth, same) == pytest.approx(1.0, abs=0.01)
    circle = Prediction(stem="f", prediction_id="p", cx_px=50.0, cy_px=50.0, diameter_nm=truth.diameter_nm)
    # Equal-area circle vs square: IoU well below 1, unlike a circle truth.
    assert 0.8 < object_iou(truth, circle) < 0.95
