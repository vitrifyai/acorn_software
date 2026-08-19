"""
CryoBLOB end to end, through the same path CLU uses.

CryoBLOB injects its panel into the Annotate tab. Workspaces can remove that tab
from the tab bar, so the detector has to keep working when its own panel is not
on screen — otherwise "run cryoblob" silently does nothing outside the Annotate
workspace.

Skipped when the JAX/CryoBLOB stack is not installed.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")
pytest.importorskip("acorn_cryoblob")
np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def blobby_images(tmp_path_factory):
    """Two small images with obvious round blobs — no simulator dependency."""
    d = tmp_path_factory.mktemp("cryoblob")
    rng = np.random.RandomState(3)
    for n in range(2):
        img = rng.normal(0.5, 0.02, (256, 256)).astype("float32")
        yy, xx = np.mgrid[0:256, 0:256]
        # DARK blobs on a light ground — CryoBLOB looks for density minima, the way
        # real cryo-EM particles appear. Bright blobs return zero detections.
        for cy, cx in [(60, 60), (60, 180), (180, 60), (180, 180), (120, 120)]:
            img -= 0.4 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 7.0 ** 2)))
        tifffile.imwrite(d / f"blobs_{n}.tif", img)
    return d


@pytest.fixture(scope="module")
def window(blobby_images, tmp_path_factory):
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("cfg"))
    app = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    win.open_files(sorted(blobby_images.glob("*.tif")))
    for _ in range(50):
        app.processEvents()
    yield win, app
    win.close()
    app.processEvents()


def test_cryoblob_plugin_loads_and_targets_the_annotate_tab(window):
    win, _ = window
    plugin = next((p for p in win._plugins if p.PLUGIN_ID == "acorn_cryoblob"), None)
    assert plugin is not None, "CryoBLOB plugin did not load"
    assert plugin.WORKFLOW_STAGE == "Annotate"


def test_cryoblob_panel_survives_its_tab_being_hidden(window):
    win, _ = window
    plugin = next(p for p in win._plugins if p.PLUGIN_ID == "acorn_cryoblob")
    win.set_workspace("annotate")
    panel = getattr(plugin, "_panel", None)
    if panel is None:
        pytest.skip("CryoBLOB panel not created in this install")
    win.set_workspace("dataset")           # Annotate tab removed from the bar
    assert panel is not None
    assert panel.parent() is not None, "hiding the tab destroyed the CryoBLOB panel"


def test_cryoblob_run_reaches_the_canvas_from_a_workspace_without_its_tab(window):
    """The regression: detections written to disk but never drawn."""
    win, app = window
    win.set_workspace("dataset")
    img_dir = Path(win._image_paths[0]).parent
    for stale in img_dir.glob(".*.acorn.json"):
        stale.unlink()
    win._canvas_widget.canvas.store.clear()
    before = len(win._canvas_widget.canvas.store)

    win._context.action_requested.emit("run_cryoblob", dict(
        pixel_size_nm=1.0, min_sigma=3.0, max_sigma=14.0,
        threshold_rel=0.05, blob_downscale=1.0, detection_mode="log",
    ))

    def _populated_sidecars():
        # Clearing the store above makes autosave write an EMPTY sidecar, so waiting
        # for "a file exists" finishes before CryoBLOB has produced anything.
        out = []
        for f in img_dir.glob(".*.acorn.json"):
            try:
                if json.loads(f.read_text()).get("annotations"):
                    out.append(f)
            except Exception:
                pass
        return out

    deadline = time.time() + 180
    while time.time() < deadline and not _populated_sidecars():
        app.processEvents()
        time.sleep(0.05)
    for _ in range(80):
        app.processEvents()
        time.sleep(0.02)

    sidecars = _populated_sidecars()
    assert sidecars, "CryoBLOB wrote no sidecar with annotations"
    on_disk = json.loads(sidecars[0].read_text())["annotations"]

    after = len(win._canvas_widget.canvas.store)
    assert after > before, (
        "detections reached disk but not the canvas — the reload/redraw path is broken"
    )


# ── contrast polarity ─────────────────────────────────────────────────────────
# CryoBLOB finds density minima, which is right for cryo-EM but returns zero on
# inverted data with no error and nothing to explain why.

def _planted(polarity: str):
    """256x256 with five gaussian blobs, dark or light relative to the field."""
    yy, xx = np.mgrid[0:256, 0:256]
    img = np.random.RandomState(3).normal(0.5, 0.02, (256, 256)).astype("float32")
    blob = np.zeros((256, 256), dtype="float32")
    for cy, cx in [(60, 60), (60, 180), (180, 60), (180, 180), (120, 120)]:
        blob += 0.4 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 7.0 ** 2)))
    return img - blob if polarity == "dark" else img + blob


def test_polarity_detection_reads_dark_and_light_fields():
    from acorn_cryoblob.thread import detect_contrast_polarity
    assert detect_contrast_polarity(_planted("dark")) == "dark"
    assert detect_contrast_polarity(_planted("light")) == "light"


def test_polarity_detection_defaults_to_dark_on_a_featureless_field():
    """A symmetric histogram must not be flipped on a coin toss."""
    from acorn_cryoblob.thread import detect_contrast_polarity
    flat = np.random.RandomState(0).normal(0.5, 0.02, (256, 256))
    assert detect_contrast_polarity(flat) == "dark"


def test_polarity_detection_survives_nans_and_empty_input():
    from acorn_cryoblob.thread import detect_contrast_polarity
    assert detect_contrast_polarity(np.array([])) == "dark"
    arr = _planted("dark")
    arr[0, :10] = np.nan
    assert detect_contrast_polarity(arr) == "dark"


def test_inversion_preserves_the_intensity_range():
    """Sigma and threshold settings have to carry over unchanged."""
    from acorn_cryoblob.thread import _apply_contrast_polarity
    src = _planted("light")
    out, used = _apply_contrast_polarity(src, "light")
    assert used == "light"
    assert np.isclose(out.min(), src.min()) and np.isclose(out.max(), src.max())


def test_unknown_polarity_falls_back_to_auto():
    from acorn_cryoblob.thread import _apply_contrast_polarity
    _out, used = _apply_contrast_polarity(_planted("light"), "nonsense")
    assert used == "light"       # auto-detected rather than crashing


@pytest.mark.parametrize("polarity,expect_detections", [("dark", True), ("light", True)])
def test_detector_finds_planted_blobs_in_either_polarity(tmp_path, polarity, expect_detections):
    """The regression: 'light' data used to return zero, silently."""
    from acorn_cryoblob.thread import _process_single_file
    f = tmp_path / f"{polarity}.tif"
    tifffile.imwrite(f, _planted(polarity))
    recs = _process_single_file(
        str(f), detection_mode="log", use_watershed=False, contrast_polarity="auto",
        pixel_size_nm=1.0, run_mode="final", blob_downscale=1.0,
        min_sigma=3.0, max_sigma=14.0, blob_step=1.0, threshold_rel=0.05,
        max_detections=100, refine_sizes=True, size_scale=1.0,
        ridge_threshold=0.006, ridge_scales=20, min_marker_distance=4.0,
        use_ridge_detection=False, stream_large_files=False,
        exponential=False, logarizer=False, gblur=2, background=0,
        apply_filter=0, cache_results=False,
    )
    assert bool(recs) is expect_detections
    assert len(recs) >= 4, f"{polarity}: found {len(recs)} of 5 planted blobs"


def test_clu_can_set_the_polarity():
    """Without the key in _CLU_DEFAULTS the plugin silently drops CLU's override."""
    from acorn_cryoblob.plugin import _CLU_DEFAULTS
    assert _CLU_DEFAULTS["contrast_polarity"] == "auto"
    from acorn_llm.agent import _TOOLS
    tool = next(t for t in _TOOLS if t["name"] == "run_cryoblob")
    assert set(tool["properties"]["contrast_polarity"]["enum"]) == {"auto", "dark", "light"}
