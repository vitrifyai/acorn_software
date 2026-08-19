"""
What gets drawn on top of a micrograph.

Two rules the app kept breaking: colours came from three unrelated hard-coded
lists, and every ROI printed its label even when thirty of them shared one name,
covering the data with the same word repeated.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from acorn.render import palette as P


# ── palette ───────────────────────────────────────────────────────────────────

def test_label_colour_is_stable_across_processes():
    """Python's str hash is salted per process; a label must not change colour
    every time the application restarts."""
    assert P.color_for_label("vesicle") == P.color_for_label("vesicle")
    assert P.color_for_label("Vesicle") == P.color_for_label("vesicle ")
    # value pinned so a future refactor cannot silently reshuffle every colour
    assert P.color_for_label("nanoparticle") == "#D2603C"


def test_palette_entries_are_valid_hex():
    for c in P.ANNOTATION_PALETTE + (P.PENDING, P.ACCEPTED, P.SELECTED, P.EXCLUDED):
        assert c.startswith("#") and len(c) == 7
        int(c[1:], 16)


def test_state_colours_are_not_in_the_categorical_palette():
    """Otherwise a label could accidentally look like 'pending' or 'accepted'."""
    for state in (P.PENDING, P.ACCEPTED, P.EXCLUDED):
        assert state not in P.ANNOTATION_PALETTE


def test_every_drawing_surface_uses_the_shared_palette():
    from acorn.gui.canvas_widget import CanvasWidget
    from acorn.gui.sam_controller import SAMControllerMixin
    assert list(CanvasWidget._SPATIAL_PALETTE) == list(P.ANNOTATION_PALETTE)
    assert SAMControllerMixin._sam_color_for_label("membrane") == P.color_for_label("membrane")


def test_simulator_labels_match_hand_drawn_ones():
    sim = pytest.importorskip("acorn_tem_sim.engine_io")
    assert sim._annotation_color("nanoparticle") == P.color_for_label("nanoparticle")


# ── label-drawing policy ──────────────────────────────────────────────────────

@pytest.fixture
def renderer():
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from acorn.render.annotation_renderer import AnnotationRenderer
    fig, ax = plt.subplots()
    yield AnnotationRenderer(ax)
    plt.close(fig)


def _rois(n, label):
    from acorn.core.annotations import ROIAnnotation
    out = []
    for i in range(n):
        x, y = (i % 10) * 20 + 5, (i // 10) * 20 + 5
        out.append(ROIAnnotation(
            vertices=[(x, y), (x + 8, y), (x + 8, y + 8), (x, y + 8)],
            area_nm2=1.0, stats={}, color=P.color_for_label(label),
            linewidth=1.5, label=label,
        ))
    return out


def _texts(ax):
    return [t.get_text() for t in ax.texts if t.get_text()]


def test_a_few_rois_still_print_their_labels(renderer):
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.replace_all(_rois(3, "vesicle"))
    renderer.render(store)
    assert _texts(renderer.ax).count("vesicle") == 3
    assert not renderer._legend_artists, "no legend needed when labels fit"


def test_many_identical_labels_collapse_into_one_legend_entry(renderer):
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.replace_all(_rois(30, "nanoparticle"))
    renderer.render(store)
    printed = _texts(renderer.ax)
    assert "nanoparticle" not in printed, "30 copies of the same word were drawn over the data"
    assert len(renderer._legend_artists) == 1
    legend = renderer._legend_artists[0].get_text()
    assert "nanoparticle" in legend and "30" in legend


def test_distinct_labels_each_get_a_legend_row(renderer):
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.replace_all(_rois(12, "vesicle") + _rois(20, "ice"))
    renderer.render(store)
    assert len(renderer._legend_artists) == 2
    rows = " ".join(a.get_text() for a in renderer._legend_artists)
    assert "vesicle" in rows and "ice" in rows


def test_unlabelled_rois_draw_no_text_and_no_legend(renderer):
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.replace_all(_rois(30, ""))
    renderer.render(store)
    assert not _texts(renderer.ax)
    assert not renderer._legend_artists


def test_clear_removes_the_legend(renderer):
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.replace_all(_rois(30, "nanoparticle"))
    renderer.render(store)
    assert renderer._legend_artists
    renderer.clear()
    assert not renderer._legend_artists
