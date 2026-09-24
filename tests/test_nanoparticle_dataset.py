from __future__ import annotations

import json

import tifffile

from acorn_tem_sim.nanoparticle_dataset import generate_nanoparticle_dataset


def test_generate_nanoparticle_dataset_writes_exact_truth(tmp_path) -> None:
    paths = generate_nanoparticle_dataset(
        tmp_path,
        4,
        seed=30,
        image_size_px=160,
        diameters_nm=(24.0,),
        pixel_sizes_a=(15.0,),
        densities=("sparse",),
        doses=("medium",),
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
    assert truth["geometry_targets_nm"] == ["diameter_nm", "ecd_nm"]
    assert truth["objects"]
    obj = truth["objects"][0]
    assert obj["class_name"] == "nanoparticle"
    assert obj["diameter_nm"] > 0
    assert obj["ecd_nm"] == obj["diameter_nm"]

    line = label_files[0].read_text(encoding="utf-8").splitlines()[0]
    parts = line.split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) == 96
    assert all(0.0 <= v <= 1.0 for v in coords)
