from __future__ import annotations

import numpy as np
import tifffile
import yaml
from PIL import Image

from acorn_sim_common.learned_pickers import _prediction_from_mask, prepare_yolo_rgb_dataset


def test_prediction_from_mask_reports_equivalent_diameter() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    mask[8:12, 7:13] = True

    pred = _prediction_from_mask(
        mask,
        2.0,
        class_name="particle",
        score=0.9,
        source="unit",
    )

    assert pred is not None
    assert pred["class_name"] == "particle"
    assert pred["score"] == 0.9
    assert pred["cx_px"] == 9.5
    assert pred["cy_px"] == 9.5
    assert pred["diameter_nm"] > 0


def test_prepare_yolo_rgb_dataset_converts_grayscale_images(tmp_path) -> None:
    dataset = tmp_path / "dataset"
    image_dir = dataset / "images" / "train"
    label_dir = dataset / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    tifffile.imwrite(image_dir / "img001.tif", np.arange(100, dtype=np.uint16).reshape(10, 10))
    (label_dir / "img001.txt").write_text("0 0.5 0.5 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    (dataset / "data.yaml").write_text(
        yaml.safe_dump({"path": str(dataset), "train": "images/train", "val": "images/train", "names": {0: "particle"}}),
        encoding="utf-8",
    )

    data_yaml = prepare_yolo_rgb_dataset(dataset, tmp_path / "rgb")

    assert data_yaml.exists()
    image = np.asarray(Image.open(tmp_path / "rgb" / "images" / "train" / "img001.png"))
    assert image.shape == (10, 10, 3)
    assert (tmp_path / "rgb" / "labels" / "train" / "img001.txt").exists()
    derived = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert derived["path"] == str(tmp_path / "rgb")
