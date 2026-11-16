"""acorn_3d tomogram tools: per-slice annotations, label volumes, prompt helpers, rendering."""
import os
import json
import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def _roi(verts, label):
    from acorn.core.annotations import ROIAnnotation
    return ROIAnnotation(vertices=verts, area_nm2=1.0, label=label)


def test_volume_annotations_roundtrip_and_rasterise(tmp_path):
    from acorn_3d.labels import VolumeAnnotations, sidecar_path_for_volume
    vol_path = tmp_path / "tomo.mrc"
    va = VolumeAnnotations(5, vol_path)
    va.set(1, [_roi([(2, 2), (8, 2), (8, 8), (2, 8)], "CNC")])
    va.set(3, [_roi([(0, 0), (4, 0), (4, 4), (0, 4)], "nanoparticle"),
               _roi([(5, 5), (9, 5), (9, 9), (5, 9)], "CNC")])
    va.prompts.append({"mode": "text", "label": "CNC"})
    p = va.save()
    assert p == sidecar_path_for_volume(vol_path) and p.exists()
    d = json.loads(p.read_text())
    assert d["version"] == 1 and set(d["slices"]) == {"1", "3"}

    va2 = VolumeAnnotations.load(vol_path, 5)
    assert va2.count() == 3 and va2.slices_with_annotations() == [1, 3]
    assert va2.labels() == ["CNC", "nanoparticle"]
    assert va2.prompts == [{"mode": "text", "label": "CNC"}]

    lab, names = va2.to_label_volume((10, 10), class_order=["CNC", "nanoparticle"])
    assert lab.shape == (5, 10, 10) and names == {1: "CNC", 2: "nanoparticle"}
    assert lab[1, 5, 5] == 1 and lab[3, 1, 1] == 2 and lab[3, 7, 7] == 1 and lab[0].sum() == 0


def test_masks_to_rois_and_back():
    from acorn_3d.labels import masks_to_rois, VolumeAnnotations
    m = np.zeros((40, 40), bool); m[10:30, 5:25] = True
    rois = masks_to_rois([m, np.zeros((40, 40), bool)], "CNC", "#ff0000", pixel_size_nm=2.0)
    assert len(rois) == 1 and rois[0].label == "CNC" and rois[0].area_nm2 == pytest.approx(400 * 4.0)
    va = VolumeAnnotations(1); va.set(0, rois)
    lab, _ = va.to_label_volume((40, 40))
    inter = (lab[0] > 0) & m; union = (lab[0] > 0) | m
    assert inter.sum() / union.sum() > 0.9


def test_write_label_mrc(tmp_path):
    import mrcfile
    from acorn_3d.labels import write_label_mrc
    lab = np.zeros((3, 8, 9), np.uint8); lab[1, 2:5, 3:6] = 2
    p = write_label_mrc(lab, tmp_path / "labels.mrc", (3.0, 2.0, 2.0))
    with mrcfile.open(str(p)) as mrc:
        assert mrc.data.shape == (3, 8, 9) and int(mrc.data.max()) == 2
        assert mrc.voxel_size.x == pytest.approx(20.0) and mrc.voxel_size.z == pytest.approx(30.0)


def test_propagation_result_volumes():
    from acorn_3d.propagate import PropagationResult, _seed_points, _correction_points, slices_to_uint8
    r = PropagationResult(n_slices=4)
    m = np.zeros((6, 6), bool); m[1:4, 1:4] = True
    r.masks = {0: {7: m}, 1: {7: m, 8: ~m}, 2: {8: ~m}}
    r.scores = {0: {7: 0.9}, 1: {7: 0.9, 8: 0.4}, 2: {8: 0.4}}
    assert r.object_ids() == [7, 8] and r.slices_for(7) == [0, 1]
    sem = r.to_label_volume((6, 6), class_id=3, min_slices=2)
    assert sem.max() == 3 and sem[3].sum() == 0
    only7 = r.to_label_volume((6, 6), score_min=0.5)
    assert only7[2].sum() == 0 and only7[0].sum() == 9
    inst = r.to_instance_volume((6, 6))
    assert set(np.unique(inst)) == {0, 8, 9}

    pts, labs = _seed_points(m, n_pos=3)
    assert 1 <= len(pts) <= 3 and all(m[int(y), int(x)] for x, y in pts) and set(labs) == {1}
    got = np.zeros_like(m); got[1:4, 1:3] = True; got[5, 5] = True
    cp, cl = _correction_points(m, got)
    assert cp == [] or set(cl) <= {0, 1}

    u8 = slices_to_uint8(np.random.default_rng(0).normal(size=(3, 16, 16)).astype(np.float32), denoise_sigma=0.5)
    assert u8.dtype == np.uint8 and u8.shape == (3, 16, 16)


def test_render_spin_movie_small(tmp_path):
    pytest.importorskip("pyrender")
    from acorn_3d.render import render_spin_movie, label_meshes
    lab = np.zeros((20, 40, 60), np.uint8)
    zz, yy, xx = np.indices(lab.shape)
    lab[((zz - 10) ** 2 + (yy - 20) ** 2 + (xx - 20) ** 2) < 36] = 2
    lab[8:12, 18:22, 30:55] = 1
    meshes = label_meshes(lab, (2.0, 2.0, 2.0))
    assert set(meshes) == {1, 2} and all(len(m.faces) > 0 for m in meshes.values())
    try:
        res = render_spin_movie(lab, 2.0, tmp_path / "spin.mp4", {1: "CNC", 2: "nanoparticle"},
                                style="paper", seconds=0.5, fps=4, size=(320, 240), gif=True)
    except Exception as exc:  # no EGL / GL on this machine
        pytest.skip(f"headless GL unavailable: {exc}")
    assert res["mp4"].exists() and res["mp4"].stat().st_size > 0
    assert res["gif"].exists() and all(p.exists() for p in res["stills"])


def test_volume_panel_offscreen():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from acorn_3d.panel import VolumePanel
    from acorn_3d.volume_loader import VolumeImage
    p = VolumePanel()
    assert p.class_label == "CNC" and p.prompt_text == "CNC"
    got = []
    p.propagate_requested.connect(lambda *a: got.append(a))
    vol = VolumeImage(); vol._data = np.zeros((7, 5, 5), np.float32); vol._shape_3d = (7, 5, 5)
    vol.meta.shape = (5, 5); vol.meta.filename = "t"
    p.set_volume(vol)
    p.set_z(3)
    assert p.current_z == 3
    p._prop_track_btn.click()
    assert got == [("track", "CNC", 0, 6)]
    p.add_class("membrane"); p.add_class("CNC")
    assert p._class_combo.count() == len(VolumePanel.DEFAULT_CLASSES)


def test_volume_image_matches_dm4image_api():
    """The canvas probes the loaded image like a DM4Image; a volume must answer every attribute."""
    from acorn.core.dm4_loader import DM4Image
    from acorn_3d.volume_loader import VolumeImage
    vol = VolumeImage(); vol._data = np.zeros((3, 4, 5), np.float32); vol._shape_3d = (3, 4, 5)
    for name in [a for a in dir(DM4Image) if not a.startswith("_") and a not in ("from_file", "open", "UNIT_TO_NM")]:
        assert hasattr(vol, name), f"VolumeImage lacks DM4Image attribute {name!r}"
    assert vol.is_color is False and vol.is_movie is False and vol.n_frames == 0
    assert vol.get_frame(2).shape == (4, 5) and vol.current_z == 2
