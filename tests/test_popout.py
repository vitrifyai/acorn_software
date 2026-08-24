"""
Detaching a control tab into its own window.

The case that needs it: in Analyze the Measure tab and the Spatial Analysis dock
share the right edge, each too narrow for its contents, and switching tabs hides
whichever you were reading.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture
def window(tmp_path):
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path)
    app = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    for _ in range(6):
        app.processEvents()
    yield win, app
    win.close()
    app.processEvents()


def _tabs(win):
    t = win._control_tabs
    return [t.tabText(i) for i in range(t.count())]


def _settle(app, n=8):
    for _ in range(n):
        app.processEvents()


def test_a_tab_becomes_a_floating_window(window):
    win, app = window
    win.set_workspace("annotate")
    _settle(app)
    before = _tabs(win)
    win.pop_out_current_tab()
    _settle(app)
    assert len(_tabs(win)) == len(before) - 1
    assert win._popped_out, "nothing was recorded as popped out"
    panel = next(iter(win._popped_out.values()))
    assert panel.isFloating()
    assert panel.widget() is not None


def test_closing_the_window_puts_the_tab_back(window):
    win, app = window
    win.set_workspace("annotate")
    _settle(app)
    before = _tabs(win)
    win.pop_out_current_tab()
    _settle(app)
    next(iter(win._popped_out.values())).close()
    _settle(app)
    assert _tabs(win) == before, "the tab did not return to where it was"


def test_the_only_tab_can_pop_out_and_the_panel_gets_out_of_the_way(window):
    """
    Analyze has one tab. Refusing to pop it would block the case this exists for,
    and an empty tab bar is dead space the image should have instead.
    """
    win, app = window
    win.set_workspace("analyze")
    _settle(app)
    assert _tabs(win) == ["Measure"]
    win.pop_out_current_tab()
    _settle(app)
    assert _tabs(win) == []
    assert not win._control_tabs.isVisible(), "an empty panel is still taking space"
    next(iter(win._popped_out.values())).close()
    _settle(app)
    assert _tabs(win) == ["Measure"]
    assert win._control_tabs.isVisible()


def test_a_workspace_switch_does_not_swallow_the_floating_window(window):
    """
    Rebuilding the tab bar re-adds widgets from the registry. Without excluding
    the popped-out ones, Qt re-parents the widget back into the tabs and the
    floating window is left empty.
    """
    win, app = window
    win.set_workspace("analyze")
    _settle(app)
    win.pop_out_current_tab()
    _settle(app)
    panel = next(iter(win._popped_out.values()))

    win.set_workspace("annotate")
    _settle(app, 10)
    assert panel.widget() is not None, "the workspace switch stole the panel"
    assert "Measure" not in _tabs(win)

    win.set_workspace("analyze")
    _settle(app, 10)
    assert panel.widget() is not None
    assert _tabs(win) == [], "the tab reappeared while still popped out"

    panel.close()
    _settle(app)
    assert _tabs(win) == ["Measure"]


def test_the_panel_keeps_working_after_being_popped_out(window):
    """It is the same widget re-parented, so its state has to survive."""
    win, app = window
    win.set_workspace("annotate")
    _settle(app)
    win._sam_panel.setProperty("_probe", "kept")
    win.pop_out_current_tab()
    _settle(app)
    next(iter(win._popped_out.values())).close()
    _settle(app)
    assert win._sam_panel.property("_probe") == "kept"


def test_there_is_a_visible_way_to_do_it(window):
    """A feature reachable only from a menu nobody opens may as well not exist."""
    win, _app = window
    from PyQt6.QtCore import Qt
    corner = win._control_tabs.cornerWidget(Qt.Corner.TopRightCorner)
    assert corner is not None and corner.toolTip()
    texts = [a.text() for a in win.findChildren(type(win.menuBar().actions()[0]))]
    assert any("Pop Out" in t for t in texts)


# ── remembered between sessions ───────────────────────────────────────────────

def _restart(tmp_path, app):
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    win.resize(1400, 900)
    win.show()
    _settle(app, 10)
    return win


def test_a_floating_window_reopens_next_launch(window, tmp_path):
    """Popping the same tab out every session is exactly the tedium to avoid."""
    from acorn.gui.workspaces import load_prefs
    win, app = window
    win.set_workspace("analyze")
    _settle(app)
    win.pop_out_current_tab()
    _settle(app)
    assert "Measure" in (load_prefs().extra.get("popped_out") or {})
    win.close()
    _settle(app)

    again = _restart(tmp_path, app)
    try:
        assert "Measure" in again._popped_out, "the floating window did not come back"
        panel = again._popped_out["Measure"]
        assert panel.isFloating() and panel.widget() is not None
    finally:
        again.close()
        _settle(app)


def test_where_the_window_was_is_remembered_too(window, tmp_path):
    """Reopening in the middle of the screen is barely better than not at all."""
    from acorn.gui.workspaces import load_prefs
    win, app = window
    win.set_workspace("analyze")
    _settle(app)
    win.pop_out_current_tab()
    _settle(app)
    panel = win._popped_out["Measure"]
    panel.move(300, 200)
    panel.resize(500, 600)
    _settle(app)
    win._save_popout_state()
    geom = load_prefs().extra["popped_out"]["Measure"]
    assert geom[2] > 400 and geom[3] > 500, f"geometry not recorded: {geom}"
    win.close()
    _settle(app)


def test_closing_a_window_stops_it_reopening(window, tmp_path):
    """
    QDockWidget.close() hides rather than destroys, so `destroyed` never fires
    and a closed window kept coming back every launch.
    """
    from acorn.gui.workspaces import load_prefs
    win, app = window
    win.set_workspace("analyze")
    _settle(app)
    win.pop_out_current_tab()
    _settle(app)
    win._popped_out["Measure"].close()
    _settle(app)
    assert not (load_prefs().extra.get("popped_out") or {}), \
        "a closed window is still recorded and would reopen"
    assert _tabs(win) == ["Measure"]


def test_a_saved_window_for_a_tab_that_no_longer_exists_is_ignored(window, tmp_path):
    """A plugin can be uninstalled between sessions."""
    win, app = window
    win._workspace_prefs.extra["popped_out"] = {"No Such Tab": [10, 10, 300, 300]}
    win._restore_popped_out()
    _settle(app)
    assert "No Such Tab" not in win._popped_out
