from __future__ import annotations

import json
import math

import numpy as np
import pytest
import tifffile

from acorn_sim_common.unet_pickers import (
    BOUNDARY,
    INTERIOR,
    instances_from_probabilities,
    predict_unet,
    rasterize_truth_labels,
    train_unet,
)


def _payload(objects, shape=(64, 64)):
    return {"stem": "f", "image_shape_px": list(shape), "pixel_size_nm": 1.0, "objects": objects}


def test_rasterized_area_matches_truth_diameter() -> None:
    labels = rasterize_truth_labels(_payload([{"cx_px": 32, "cy_px": 32, "diameter_nm": 20.0}]))
    foreground = np.count_nonzero(labels != 0)
    assert foreground == pytest.approx(math.pi * 10.0**2, rel=0.03)
    assert np.count_nonzero(labels == BOUNDARY) > 0


def test_rasterize_uses_requested_edge_definition() -> None:
    obj = {"cx_px": 32, "cy_px": 32, "d_outer_leaflet_nm": 30.0, "d_inner_leaflet_nm": 20.0}
    outer = rasterize_truth_labels(_payload([obj]), target_field="d_outer_leaflet_nm")
    inner = rasterize_truth_labels(_payload([obj]), target_field="d_inner_leaflet_nm")
    assert np.count_nonzero(outer) > np.count_nonzero(inner)


def test_overlapping_objects_are_separated_by_boundary_label() -> None:
    labels = rasterize_truth_labels(
        _payload([{"cx_px": 24, "cy_px": 32, "diameter_nm": 20.0}, {"cx_px": 40, "cy_px": 32, "diameter_nm": 20.0}])
    )
    interior_components = len(np.unique(__import__("scipy.ndimage", fromlist=["label"]).label(labels == INTERIOR)[0])) - 1
    assert interior_components == 2


def test_watershed_splits_touching_objects() -> None:
    labels = rasterize_truth_labels(
        _payload([{"cx_px": 24, "cy_px": 32, "diameter_nm": 20.0}, {"cx_px": 40, "cy_px": 32, "diameter_nm": 20.0}])
    )
    prob = np.stack([(labels == k).astype(np.float32) for k in range(3)])
    instances = instances_from_probabilities(prob)
    assert instances.max() == 2
    # Together the two instances recover the full foreground, boundary included.
    assert np.count_nonzero(instances) == np.count_nonzero(labels)


def _tiny_dataset(root):
    rng = np.random.default_rng(0)
    for split, n in (("train", 4), ("val", 1), ("test", 1)):
        (root / "images" / split).mkdir(parents=True)
        (root / "truth" / split).mkdir(parents=True)
        for i in range(n):
            objs = [{"id": k + 1, "cx_px": float(c[0]), "cy_px": float(c[1]), "diameter_nm": 12.0, "fully_in_frame": True}
                    for k, c in enumerate([(16, 16), (44, 20), (30, 46)])]
            payload = {**_payload(objs), "stem": f"{split}_{i}", "split": split}
            img = (rasterize_truth_labels(payload) != 0).astype(np.float32) + 0.1 * rng.standard_normal((64, 64))
            tifffile.imwrite(root / "images" / split / f"{split}_{i}.tif", img.astype(np.float32))
            (root / "truth" / split / f"{split}_{i}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_train_and_predict_round_trip(tmp_path) -> None:
    _tiny_dataset(tmp_path / "ds")
    weights = train_unet(tmp_path / "ds", tmp_path / "unet.pt", epochs=40, batch=4, device="cpu", base=8)
    paths = predict_unet(tmp_path / "ds", weights, tmp_path / "pred", class_name="particle", device="cpu")
    preds = json.loads(paths[0].read_text())["predictions"]
    assert len(preds) == 3
    for p in preds:
        assert p["diameter_nm"] == pytest.approx(12.0, rel=0.25)
        assert len(p["polygon"]) >= 3
