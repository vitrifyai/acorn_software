"""Headless tests for the physics engine (no PyQt/GUI imports).

Covers the engine-backed generation in engine_io.py + reference.py: composed-scene
multislice TEM, 4D-STEM (scan-frame sidecars + the dose-0 guard), ground-truth
scene annotations per specimen, the scan-footprint coordinate mapping, and
reference-matched simulation. All PyQt-free so they run without a display.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from acorn_tem_sim.engine_io import (generate_tem_advanced, generate_4dstem,
                                     _build_scene, _scan_footprints)
from acorn_tem_sim.engine.scene import scene_annotations
from acorn_tem_sim.reference import simulate_from_reference


def _p(**over):
    """Small, fast defaults so the multislice tests stay quick."""
    p = dict(image_size_px=64, pixel_size_a=2.0, total_dose_e_per_a2=40.0, gpu=False)
    p.update(over)
    return p


def _sidecar_for(img_path):
    p = Path(img_path)
    return p.parent / f".{p.stem}.acorn.json"


# --------------------------------------------------------------------------
# TEM advanced (composed-scene multislice)
# --------------------------------------------------------------------------
def test_tem_advanced_writes_images_and_sidecars(tmp_path):
    paths = generate_tem_advanced(tmp_path, 2, _p(specimen_kind="plga", n_particles=8),
                                  seed=1, save_layers=False)
    assert len(paths) == 2
    for p in paths:
        assert p.exists()
        side = _sidecar_for(p)
        assert side.exists(), f"missing sidecar for {p.name}"
        d = json.loads(side.read_text())
        assert d["version"] == 3
        assert d["pixel_size_nm"] == pytest.approx(0.2)     # 2.0 A / 10
        assert all(a["type"] == "roi" for a in d["annotations"])


def test_tem_advanced_is_seeded(tmp_path):
    a = generate_tem_advanced(tmp_path / "a", 1, _p(n_particles=8), seed=7, save_layers=False)
    b = generate_tem_advanced(tmp_path / "b", 1, _p(n_particles=8), seed=7, save_layers=False)
    assert np.array_equal(tifffile.imread(a[0]), tifffile.imread(b[0]))


# --------------------------------------------------------------------------
# Ground-truth annotations per specimen type
# --------------------------------------------------------------------------
@pytest.mark.parametrize("kind,label", [
    ("plga", "nanoparticle"),
    ("contamination", "ice_contamination"),
    ("bacteria", "e_coli"),
])
def test_scene_annotations_label_per_specimen(kind, label):
    params = {"specimen_kind": kind}
    if kind == "bacteria":
        params["species"] = "e_coli"
    scene = _build_scene(params, seed=3)
    foot = scene_annotations((64, 64), 2.0, scene)
    assert label in {lab for lab, _ in foot}
    for _lab, verts in foot:
        assert len(verts) >= 3                              # real polygons, not points


# --------------------------------------------------------------------------
# Scan-footprint coordinate mapping (specimen px -> scan frame)
# --------------------------------------------------------------------------
def test_scan_footprints_maps_specimen_to_scan_frame():
    # Fake FourDSTEM: scan origin at (10,10) A, 5 A/step, 2 A/specimen-px.
    fd = SimpleNamespace(scan_pos_a=np.array([[[10.0, 10.0]]]), scan_step_a=5.0, px=2.0)
    # A specimen-pixel vertex at (10,10) -> position 20 A -> scan pixel (20-10)/5 = 2.
    out = _scan_footprints([("x", [(10.0, 10.0)])], fd)
    (lab, verts), = out
    assert lab == "x"
    assert verts[0] == pytest.approx([2.0, 2.0])


# --------------------------------------------------------------------------
# 4D-STEM
# --------------------------------------------------------------------------
def test_4dstem_writes_detectors_sidecars_and_cube(tmp_path):
    paths = generate_4dstem(tmp_path, _p(specimen_kind="plga", n_particles=10, scan_size=12),
                            seed=2)
    assert paths, "no detector images produced"
    assert (tmp_path / "datacube.npy").exists()
    for p in paths:
        side = _sidecar_for(p)
        assert side.exists()
        d = json.loads(side.read_text())
        assert d["pixel_size_nm"] > 0                       # scan-step calibration present


def test_4dstem_dose_zero_is_noise_free_not_crash(tmp_path):
    # dose=0 used to crash config validation (0 < min 0.5); now it means noise-free.
    p0 = generate_4dstem(tmp_path / "d0", _p(specimen_kind="plga", n_particles=6,
                         scan_size=8, total_dose_e_per_a2=0), seed=1)
    p40 = generate_4dstem(tmp_path / "d40", _p(specimen_kind="plga", n_particles=6,
                          scan_size=8, total_dose_e_per_a2=40.0), seed=1)
    assert p0 and p40
    c0 = np.load(tmp_path / "d0" / "datacube.npy")
    c40 = np.load(tmp_path / "d40" / "datacube.npy")
    assert c0.shape == c40.shape
    assert not np.array_equal(c0, c40)                      # dose actually changed the data


def test_4dstem_annotations_inside_scan_frame(tmp_path):
    scan = 16
    paths = generate_4dstem(tmp_path, _p(specimen_kind="plga", n_particles=12, scan_size=scan),
                            seed=1)
    d = json.loads(_sidecar_for(paths[0]).read_text())
    assert d["annotations"], "expected labeled particles in scan frame"
    for a in d["annotations"]:
        for x, y in a["vertices"]:
            assert 0 <= x <= scan - 1
            assert 0 <= y <= scan - 1


# --------------------------------------------------------------------------
# Reference-matched simulation
# --------------------------------------------------------------------------
def test_simulate_from_reference_cryoem(tmp_path):
    ref = tmp_path / "ref.tif"      # >=64 px: engine's image_size_px min is 64
    tifffile.imwrite(ref, (np.random.default_rng(0).random((96, 96)) * 255).astype("uint8"))
    out = Path(simulate_from_reference(str(ref), "cryoem",
                                       params={"pixel_size_a": 2.0}, n=2,
                                       out_dir=str(tmp_path / "out")))
    assert (out / "manifest.yaml").exists()
    assert len(list(out.glob("sim_*.png"))) == 2
    assert len(list(out.glob("label_*.npy"))) == 2
