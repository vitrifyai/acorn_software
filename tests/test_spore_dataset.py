from __future__ import annotations

import json

import tifffile

from acorn_sem_sim.spore_dataset import SporeCondition, generate_spore_dataset


def test_generate_spore_dataset_writes_exact_truth(tmp_path) -> None:
    paths = generate_spore_dataset(
        tmp_path,
        4,
        seed=40,
        image_size_px=160,
        conditions=(
            SporeCondition(
                size_class="small",
                clustering="isolated",
                pixel_size_nm=9.6,
                debris=False,
            ),
        ),
    )

    assert len(paths) == 4
    assert tifffile.imread(paths[0]).shape == (160, 160)
    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "manifest.json").exists()

    truth_files = sorted((tmp_path / "truth").glob("*/*.json"))
    label_files = sorted((tmp_path / "labels").glob("*/*.txt"))
    assert len(truth_files) == 4
    assert len(label_files) == 4

    truth = json.loads(truth_files[0].read_text(encoding="utf-8"))
    assert truth["geometry_targets_nm"] == ["major_nm", "minor_nm", "ecd_nm"]
    assert truth["objects"]
    obj = truth["objects"][0]
    assert obj["class_name"] == "spore"
    assert obj["major_nm"] >= obj["minor_nm"] > 0
    assert obj["ecd_nm"] > 0

    line = label_files[0].read_text(encoding="utf-8").splitlines()[0]
    parts = line.split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) >= 12
    assert all(0.0 <= v <= 1.0 for v in coords)
