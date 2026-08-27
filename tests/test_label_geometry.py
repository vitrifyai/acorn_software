"""Where an annotation ends up, in pixels, after export.

The existing export tests cover splits, determinism and leakage between them --
everything except whether the label describes the same place in the image that
the operator drew. A coordinate flip or a missing tile offset would train a model
on systematically displaced labels and invalidate every benchmark built from it,
while every other test carried on passing.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from acorn.core.annotations import AnnotationStore, ROIAnnotation
from acorn.core.contrast import ContrastParams
from acorn.core.dm4_loader import DM4Image
from acorn.export.training_exporter import TrainingConfig, add_image


def _square_image(size, x0, y0, x1, y1):
    img = DM4Image()
    img.raw = np.zeros((size, size), np.float32)
    img.raw[y0:y1, x0:x1] = 1.0
    img.meta.shape = img.raw.shape
    img.meta.pixel_size = 0.5
    img.meta.filename = "geometry"

    store = AnnotationStore()
    store.add(ROIAnnotation(label="square",
                            vertices=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)]))
    return img, store


def _export(tmp_path, img, store, tile):
    add_image(tmp_path, img, store, ContrastParams(),
              TrainingConfig(tile_size=tile, tile_overlap=0, augment=False,
                             skip_empty_tiles=True))
    return json.loads((tmp_path / "annotations.json").read_text())


def test_an_untiled_export_puts_the_label_where_it_was_drawn(tmp_path):
    x0, y0, x1, y1 = 40, 60, 100, 140
    data = _export(tmp_path, *_square_image(256, x0, y0, x1, y1), tile=256)

    assert len(data["annotations"]) == 1
    bbox = data["annotations"][0]["bbox"]
    assert bbox == pytest.approx([x0, y0, x1 - x0, y1 - y0], abs=1.0)


def test_the_polygon_corners_survive_export(tmp_path):
    """Not just the bounding box -- the segmentation itself."""
    x0, y0, x1, y1 = 40, 60, 100, 140
    data = _export(tmp_path, *_square_image(256, x0, y0, x1, y1), tile=256)

    seg = data["annotations"][0]["segmentation"][0]
    xs = sorted({round(seg[i]) for i in range(0, len(seg), 2)})
    ys = sorted({round(seg[i]) for i in range(1, len(seg), 2)})
    assert xs == [x0, x1] and ys == [y0, y1]


def test_a_label_in_a_non_origin_tile_is_offset_into_tile_coordinates(tmp_path):
    """The case a whole-image test cannot catch: the tile's own origin has to be
    subtracted, and only a label outside the first tile proves it was."""
    size, tile = 256, 128
    x0, y0, x1, y1 = 150, 160, 200, 210        # wholly inside the tile at (128, 128)
    data = _export(tmp_path, *_square_image(size, x0, y0, x1, y1), tile=tile)

    assert len(data["annotations"]) == 1, "square spans more than one tile"
    bbox = data["annotations"][0]["bbox"]
    assert bbox == pytest.approx([x0 - tile, y0 - tile, x1 - x0, y1 - y0], abs=1.0)


def test_x_and_y_are_not_transposed(tmp_path):
    """A deliberately non-square shape: a transpose is invisible on a square."""
    x0, y0, x1, y1 = 30, 90, 130, 120          # 100 wide, 30 tall
    data = _export(tmp_path, *_square_image(256, x0, y0, x1, y1), tile=256)

    _, _, w, h = data["annotations"][0]["bbox"]
    assert w == pytest.approx(100, abs=1.0)
    assert h == pytest.approx(30, abs=1.0)


# --- annotation persistence -------------------------------------------------

def test_an_annotation_store_survives_a_json_round_trip():
    """Untested until now, and it runs on every autosave. Silent loss here would
    show up as annotations quietly changing between sessions."""
    original = AnnotationStore()
    original.add(ROIAnnotation(label="vesicle",
                               vertices=[(1.5, 2.5), (9.0, 2.5), (9.0, 8.25), (1.5, 8.25)]))
    original.add(ROIAnnotation(label="pore", vertices=[(0.0, 0.0), (4.0, 0.0), (2.0, 3.0)]))

    restored = AnnotationStore.from_json(original.to_json())

    assert len(list(restored)) == len(list(original))
    for a, b in zip(original, restored):
        assert a.label == b.label
        assert [tuple(map(float, v)) for v in a.vertices] == \
               [tuple(map(float, v)) for v in b.vertices]


def test_the_round_trip_is_stable_across_repeats():
    """Fractional vertices must not drift with each save/load cycle."""
    store = AnnotationStore()
    store.add(ROIAnnotation(label="vesicle",
                            vertices=[(1.5, 2.5), (9.0, 2.5), (9.0, 8.25)]))
    once = store.to_json()
    twice = AnnotationStore.from_json(once).to_json()
    thrice = AnnotationStore.from_json(twice).to_json()
    assert once == twice == thrice


def test_an_empty_store_round_trips_to_an_empty_store():
    assert list(AnnotationStore.from_json(AnnotationStore().to_json())) == []
