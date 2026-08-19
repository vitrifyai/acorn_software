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
