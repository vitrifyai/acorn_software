from __future__ import annotations

import json

import pytest
import tifffile

from acorn_tem_sim.simulator import Specimen, resolve
from acorn_tem_sim.vesicle_dataset import generate_vesicle_dataset, simulate_vesicle_micrograph


def test_simulate_vesicle_micrograph_preserves_geometry_truth() -> None:
    cfg = resolve({"image_size_px": 160, "pixel_size_a": 10.0, "total_dose_e_per_a2": 30})
    specimen = Specimen(
        kind="lipid_single",
        n_particles=4,
        diameter_nm_mean=24.0,
        diameter_nm_sd=1.0,
        membrane_thickness_nm=4.0,
        solvent_noise=2.0,
        seed=11,
    )

    sim = simulate_vesicle_micrograph(cfg, specimen, seed=12, defocus_um=-1.5)

    assert sim.micrograph.image.shape == (160, 160)
    assert sim.micrograph.image.std() > 0
    assert len(sim.objects) > 0
    obj = sim.objects[0]
    assert obj["d_bilayer_mid_nm"] == pytest.approx(obj["d_outer_leaflet_nm"] - 4.0)
    assert obj["d_inner_leaflet_nm"] == pytest.approx(obj["d_outer_leaflet_nm"] - 8.0)


def test_generate_vesicle_dataset_writes_images_labels_truth_and_manifests(tmp_path) -> None:
    paths = generate_vesicle_dataset(
        tmp_path,
        6,
        seed=20,
        image_size_px=192,
        outer_diameters_nm=(20.0,),
        pixel_sizes_a=(10.0,),
        densities=("sparse",),
        contrasts=("high",),
    )

    assert len(paths) == 6
    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "manifest_train.json").exists()
    assert (tmp_path / "manifest_val.json").exists()
    assert (tmp_path / "manifest_test.json").exists()

    image = tifffile.imread(paths[0])
    assert image.shape == (192, 192)
    assert image.std() > 0

    truth_files = sorted((tmp_path / "truth").glob("*/*.json"))
    label_files = sorted((tmp_path / "labels").glob("*/*.txt"))
    assert len(truth_files) == 6
    assert len(label_files) == 6

    truth = json.loads(truth_files[0].read_text(encoding="utf-8"))
    assert truth["geometry_targets_nm"] == [
        "d_outer_leaflet_nm",
        "d_bilayer_mid_nm",
        "d_inner_leaflet_nm",
    ]
    assert truth["objects"]
    obj = truth["objects"][0]
    assert obj["d_outer_leaflet_nm"] > obj["d_bilayer_mid_nm"] > obj["d_inner_leaflet_nm"]
    assert obj["d_bilayer_mid_nm"] == pytest.approx(obj["d_outer_leaflet_nm"] - 4.0)
    assert obj["d_inner_leaflet_nm"] == pytest.approx(obj["d_outer_leaflet_nm"] - 8.0)

    line = label_files[0].read_text(encoding="utf-8").splitlines()[0]
    parts = line.split()
    assert parts[0] == "0"
    coords = [float(v) for v in parts[1:]]
    assert len(coords) == 96
    assert all(0.0 <= v <= 1.0 for v in coords)
