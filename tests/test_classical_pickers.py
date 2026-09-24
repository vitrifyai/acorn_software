from __future__ import annotations

import json

import numpy as np
import tifffile

from acorn_sim_common.classical_pickers import run_picker


def _toy_dataset(root, *, dark=True):
    image_dir = root / "images" / "test"
    truth_dir = root / "truth" / "test"
    image_dir.mkdir(parents=True)
    truth_dir.mkdir(parents=True)
    yy, xx = np.mgrid[:96, :96]
    image = np.full((96, 96), 220 if dark else 20, dtype=np.uint8)
    mask = (yy - 48) ** 2 + (xx - 48) ** 2 <= 10**2
    image[mask] = 30 if dark else 230
    tifffile.imwrite(image_dir / "field_000.tif", image)
    truth_dir.joinpath("field_000.json").write_text(
        json.dumps(
            {
                "stem": "field_000",
                "split": "test",
                "image_shape_px": [96, 96],
                "pixel_size_nm": 1.0,
                "objects": [
                    {
                        "id": 1,
                        "class_name": "particle",
                        "cx_px": 48,
                        "cy_px": 48,
                        "diameter_nm": 20.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_log_picker_writes_prediction_json(tmp_path) -> None:
    data = tmp_path / "data"
    _toy_dataset(data, dark=True)

    paths = run_picker(
        data,
        tmp_path / "pred",
        method="log",
        class_name="particle",
        polarity="dark",
        min_diameter_nm=12.0,
        max_diameter_nm=28.0,
        threshold_rel=0.02,
    )

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["predictions"]
    pred = payload["predictions"][0]
    assert pred["class_name"] == "particle"
    assert abs(pred["cx_px"] - 48) < 3
    assert abs(pred["cy_px"] - 48) < 3
    assert pred["diameter_nm"] > 0


def test_threshold_watershed_picker_writes_prediction_json(tmp_path) -> None:
    data = tmp_path / "data"
    _toy_dataset(data, dark=False)

    paths = run_picker(
        data,
        tmp_path / "pred",
        method="threshold-watershed",
        class_name="particle",
        polarity="light",
        min_diameter_nm=12.0,
        max_diameter_nm=28.0,
    )

    payload = json.loads(paths[0].read_text(encoding="utf-8"))
    assert payload["predictions"]
    pred = payload["predictions"][0]
    assert abs(pred["cx_px"] - 48) < 2
    assert abs(pred["cy_px"] - 48) < 2
