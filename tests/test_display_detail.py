"""
Seeing native detail on a large micrograph.

The canvas stride-decimates to 1024 px so matplotlib redraws quickly. That was
done once when the image loaded and never revisited, so on a 4092x5760 frame you
were always looking at every sixth pixel — zooming in magnified the decimated
copy rather than revealing anything. Native detail was unreachable at any zoom.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from acorn.render.canvas import _DISPLAY_MAX_DIM, _make_display_array


def _fine_detail(h=4092, w=5760):
    """Alternating single-pixel columns — invisible under any decimation."""
    a = np.zeros((h, w), dtype="float32")
    a[:, ::2] = 1.0
    return a


def _columns_resolved(arr) -> bool:
    return abs(arr[:, ::2].mean() - arr[:, 1::2].mean()) > 0.5


def test_the_whole_frame_is_still_decimated_for_speed():
    arr, extent = _make_display_array(_fine_detail())
    assert max(arr.shape) <= _DISPLAY_MAX_DIM
    assert not _columns_resolved(arr), "decimation should lose single-pixel detail"
    assert extent == (-0.5, 5759.5, 4091.5, -0.5)


def test_zooming_in_resolves_single_pixel_detail():
    arr, _extent = _make_display_array(_fine_detail(), view=(2000, 2400, 2000, 2300))
    assert _columns_resolved(arr), "zoomed view is still decimated — no native detail"
    assert arr.shape[1] >= 400, f"expected ~1px per column, got {arr.shape}"


def test_the_extent_matches_the_region_so_annotations_stay_aligned():
    _arr, extent = _make_display_array(_fine_detail(), view=(2000, 2400, 2000, 2300))
    x0, x1, y1, y0 = extent
    assert x0 == pytest.approx(1999.5) and x1 == pytest.approx(2400.5)
    assert y0 == pytest.approx(1999.5) and y1 == pytest.approx(2300.5)


def test_a_large_zoomed_region_is_still_capped():
    """Half of a 24-megapixel frame must not be pushed to the canvas whole."""
    arr, _ = _make_display_array(_fine_detail(), view=(0, 3000, 0, 3000))
    assert max(arr.shape) <= _DISPLAY_MAX_DIM


def test_a_degenerate_view_falls_back_to_the_whole_image():
    for view in ((100, 100, 100, 100), (-50, -40, -50, -40)):
        arr, _ = _make_display_array(_fine_detail(256, 256), view=view)
        assert arr.size > 0


def test_the_canvas_re_renders_when_the_view_changes():
    from PyQt6.QtWidgets import QApplication
    from acorn.core.dm4_loader import DM4Image
    from acorn.render.canvas import CryoCanvas

    app = QApplication.instance() or QApplication([])
    img = DM4Image()
    img.raw = _fine_detail()
    img.meta.shape = img.raw.shape
    img.meta.pixel_size = 0.592
    img.meta.pixel_unit = "nm"

    canvas = CryoCanvas()
    loader = getattr(canvas, "load_image", None) or getattr(canvas, "set_image")
    loader(img)
    app.processEvents()
    wide = canvas._img_artist.get_array().shape

    canvas.ax.set_xlim(2000, 2400)
    canvas.ax.set_ylim(2300, 2000)
    app.processEvents()
    close = canvas._img_artist.get_array().shape

    assert close != wide, "the view changed but the image was not re-rendered"
    assert close[1] >= 400, f"zoomed render is still decimated: {close}"
