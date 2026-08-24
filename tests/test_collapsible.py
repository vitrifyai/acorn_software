"""
Folding the control panels.

Fully expanded at a 1400x900 window the panels ran to 2181px (Annotate) and
1516px (Export) against a viewport of roughly 700px — three screens of scrolling
to reach a button, most of it belonging to a tool you are not using.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import (
    QApplication, QGroupBox, QLabel, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from acorn.gui import collapsible as C


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _group(app, n_children=3):
    w = QWidget()
    v = QVBoxLayout(w)
    g = QGroupBox("Presets")
    gv = QVBoxLayout(g)
    for i in range(n_children):
        gv.addWidget(QPushButton(f"b{i}"))
    v.addWidget(g)
    w.resize(320, 400)
    w.show()
    app.processEvents()
    return w, g


# ── the widget ────────────────────────────────────────────────────────────────

def test_folding_shrinks_the_group_to_its_title(app):
    _w, g = _group(app)
    open_height = g.sizeHint().height()
    C.make_collapsible(g, folded=True)
    app.processEvents()
    assert C.is_folded(g)
    assert g.maximumHeight() < open_height / 2, "folded group still takes real space"


def test_the_title_shows_which_way_it_is(app):
    _w, g = _group(app)
    C.make_collapsible(g, folded=False)
    assert g.title().startswith("▾")      # open
    C.set_folded(g, True)
    assert g.title().startswith("▸")      # shut
    assert "Presets" in g.title()


def test_folding_and_unfolding_returns_what_was_there(app):
    _w, g = _group(app)
    C.make_collapsible(g, folded=False)
    app.processEvents()
    buttons = g.findChildren(QPushButton)
    C.set_folded(g, True)
    assert all(not b.isVisibleTo(g) for b in buttons)
    C.set_folded(g, False)
    assert all(b.isVisibleTo(g) for b in buttons)


def test_unfolding_does_not_reveal_what_the_panel_hid_on_purpose(app):
    """
    Panels hide things deliberately — the inactive page of a stacked widget, a
    control that only applies to the current method. Showing every descendant on
    unfold made two panels TALLER than before they were made collapsible.
    """
    w = QWidget()
    v = QVBoxLayout(w)
    g = QGroupBox("Method")
    gv = QVBoxLayout(g)
    shown, hidden = QPushButton("shown"), QLabel("hidden on purpose")
    gv.addWidget(shown)
    gv.addWidget(hidden)
    v.addWidget(g)
    w.show()
    app.processEvents()
    hidden.setVisible(False)

    C.make_collapsible(g, folded=False)
    C.set_folded(g, True)
    C.set_folded(g, False)
    assert shown.isVisibleTo(g)
    assert not hidden.isVisibleTo(g), "unfolding revealed a deliberately hidden control"


def test_apply_to_panel_skips_what_it_is_told_to(app):
    w = QWidget()
    v = QVBoxLayout(w)
    keep, fold = QGroupBox("Run"), QGroupBox("Presets")
    for g in (keep, fold):
        QVBoxLayout(g).addWidget(QPushButton("x"))
        v.addWidget(g)
    w.show()
    app.processEvents()
    C.apply_to_panel(w, skip={"Run"})
    assert not keep.isCheckable(), "a skipped group was made foldable"
    assert fold.isCheckable()


def test_nested_groups_fold_too(app):
    """The Measure tab is ten of twelve groups deep inside a plugin section."""
    w = QWidget()
    v = QVBoxLayout(w)
    outer = QGroupBox("Particle Measurements")
    ov = QVBoxLayout(outer)
    inner = QGroupBox("Detector Geometry")
    QVBoxLayout(inner).addWidget(QPushButton("x"))
    ov.addWidget(inner)
    v.addWidget(outer)
    w.show()
    app.processEvents()
    done = C.apply_to_panel(w)
    titles = {C._title_without_arrow(g.title()) for g in done}
    assert "Detector Geometry" in titles, "a nested group was left unfoldable"


# ── in the application ────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def window(tmp_path_factory):
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("cfg"))
    a = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    for _ in range(6):
        a.processEvents()
    win.show_all_panels()
    for _ in range(8):
        a.processEvents()
    yield win, a
    win.close()
    a.processEvents()


def _panel_height(win, app, tab_name):
    tabs = win._control_tabs
    idx = [tabs.tabText(i) for i in range(tabs.count())].index(tab_name)
    tabs.setCurrentIndex(idx)
    for _ in range(8):
        app.processEvents()
    page = tabs.widget(idx)
    inner = page.widget() if isinstance(page, QScrollArea) else page
    visible = [g for g in inner.findChildren(QGroupBox) if g.isVisibleTo(inner)]
    return max((g.y() + g.height() for g in visible), default=0)


@pytest.mark.parametrize("tab,was", [
    ("Annotate", 2181), ("Export", 1516), ("Measure", 923), ("Contrast", 615),
])
def test_the_crowded_panels_now_fit_a_screen(window, tab, was):
    win, app = window
    height = _panel_height(win, app, tab)
    assert height < was, f"{tab} did not get shorter"
    assert height < 900, f"{tab} is still {height}px — more than a screen"


def test_the_action_a_panel_exists_for_never_folds(window):
    """Folding Style away would leave the Annotate tab with no drawing tools."""
    win, _ = window
    from PyQt6.QtWidgets import QGroupBox as _G
    for _label, page in win._all_tabs:
        for g in page.findChildren(_G):
            title = C._title_without_arrow(g.title())
            if title in win._NEVER_FOLDED:
                assert not g.isCheckable(), f"{title} was made foldable"


def test_which_groups_are_folded_is_remembered(window):
    win, _ = window
    win._folded_groups = set()
    from PyQt6.QtWidgets import QGroupBox as _G
    page = dict(win._all_tabs)["Contrast"]
    g = next(x for x in page.findChildren(_G)
             if C._title_without_arrow(x.title()) == "Presets")
    # Presets folds by default, so start from open or the toggle is a no-op
    g.setChecked(True)             # unfold
    assert "Presets" not in win._workspace_prefs.extra.get("folded_groups", [])
    g.setChecked(False)            # fold it
    assert "Presets" in win._workspace_prefs.extra.get("folded_groups", [])
