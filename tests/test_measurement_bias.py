from __future__ import annotations

import json

import pytest

from acorn_sim_common.measurement_bias import (
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
    summary, rows = score_bias(truths, predictions, method="yolo", specimen="particles")
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
                        "cx_px": 95,
                        "cy_px": 95,
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
