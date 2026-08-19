"""
Workspace layer — the guarantees users depend on.

The point of a workspace is that it changes what is *shown*, never what is
*loaded*. These tests pin that down, plus the escape hatch that makes hiding
anything safe in the first place.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from acorn.gui import workspaces as ws_mod
from acorn.gui.workspaces import (
    CORE_TAB_ORDER, DEFAULT_WORKSPACE, WORKSPACES, WorkspacePrefs, by_id,
)


# ── model (no Qt needed) ──────────────────────────────────────────────────────

def test_five_workspaces_with_unique_ids():
    assert len(WORKSPACES) == 5
    ids = [w.wid for w in WORKSPACES]
    assert len(set(ids)) == 5
    assert by_id(DEFAULT_WORKSPACE) is not None


def test_every_core_tab_is_reachable_from_some_workspace():
    """No core tab may become unreachable — that is the bug workspaces must not cause."""
    claimed = {tab for w in WORKSPACES for tab in w.tabs}
    missing = set(CORE_TAB_ORDER) - claimed
    assert not missing, f"core tabs no workspace shows: {sorted(missing)}"


def test_workspace_tabs_reference_real_tabs():
    for w in WORKSPACES:
        for tab in w.tabs:
            assert tab in CORE_TAB_ORDER, f"{w.wid} lists unknown tab {tab!r}"


def test_shortcuts_are_unique():
    keys = [w.shortcut for w in WORKSPACES if w.shortcut]
    assert len(keys) == len(set(keys))


def test_prefs_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    prefs = WorkspacePrefs(last_workspace="simulate", welcome_seen=True)
    ws_mod.save_prefs(prefs)
    assert ws_mod.load_prefs().last_workspace == "simulate"
    assert ws_mod.load_prefs().welcome_seen is True


def test_corrupt_prefs_fall_back_to_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    path = tmp_path / "acorn" / "workspace.json"
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert ws_mod.load_prefs().last_workspace == DEFAULT_WORKSPACE


# ── window behaviour ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def window(tmp_path_factory):
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("cfg"))
    app = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    yield win
    win.close()
    app.processEvents()


def _visible_tabs(win):
    t = win._control_tabs
    return [t.tabText(i) for i in range(t.count())]


def test_each_workspace_shows_only_its_tabs(window):
    for ws in WORKSPACES:
        window.set_workspace(ws.wid)
        shown = _visible_tabs(window)
        assert set(ws.tabs) <= set(shown), f"{ws.wid} is missing its own tabs"
        unexpected = set(shown) - set(ws.tabs)
        # only tabs no workspace claims (e.g. a third-party plugin tab) may leak in
        for label in unexpected:
            assert window._tab_is_unclaimed(label), f"{ws.wid} shows unrelated tab {label!r}"


def test_workspace_orders_its_own_tabs_first(window):
    ws = by_id("annotate")
    window.set_workspace("annotate")
    shown = _visible_tabs(window)
    assert shown[: len(ws.tabs)] == list(ws.tabs)


def test_switching_never_destroys_panel_widgets(window):
    """The whole design rests on this: panels are re-parented, never rebuilt."""
    before = dict(window._all_tabs)
    window.set_workspace("dataset")
    window.set_workspace("explore")
    window.set_workspace("annotate")
    after = dict(window._all_tabs)
    assert set(before) == set(after)
    for label in before:
        assert before[label] is after[label], f"{label} panel was recreated"


def test_switching_preserves_live_panel_state(window):
    window.set_workspace("annotate")
    window._sam_panel.setProperty("_probe", "kept")
    window.set_workspace("dataset")     # Segment hidden here
    window.set_workspace("annotate")
    assert window._sam_panel.property("_probe") == "kept"


def test_docks_follow_the_workspace(window):
    window.set_workspace("simulate")
    assert not window._plugin_docks["acorn_tem_sim"].isHidden()
    window.set_workspace("explore")
    assert window._plugin_docks["acorn_tem_sim"].isHidden()


def test_assistant_dock_is_never_forced_shut(window):
    """CLU is available in every workspace, so a switch must not close it."""
    window.set_workspace("analyze")
    window._plugin_docks["acorn_llm"].show()
    window.set_workspace("dataset")
    assert not window._plugin_docks["acorn_llm"].isHidden()


def test_show_all_panels_restores_everything(window):
    window.set_workspace("explore")
    window.show_all_panels()
    shown = _visible_tabs(window)
    for label in CORE_TAB_ORDER:
        assert label in shown, f"Show Every Panel left {label} hidden"
    for plugin_id, dock in window._plugin_docks.items():
        assert not dock.isHidden(), f"Show Every Panel left {plugin_id} hidden"


def test_unknown_workspace_falls_back_instead_of_crashing(window):
    window._apply_workspace("no-such-workspace", persist=False)
    assert window.active_workspace == DEFAULT_WORKSPACE


def test_multiple_docks_in_one_workspace_are_tabbed_not_stacked(window):
    """Stacked long panels leave each a sliver and squeeze the canvas."""
    window.resize(1400, 900)
    window.show()
    window.set_workspace("simulate")
    tem = window._plugin_docks.get("acorn_tem_sim")
    fib = window._plugin_docks.get("acorn_fib_sim")
    if tem is None or fib is None:
        import pytest
        pytest.skip("simulator plugins not installed")
    partners = [d.windowTitle() for d in window.tabifiedDockWidgets(tem)]
    assert fib.windowTitle() in partners


def test_the_workspaces_first_dock_is_the_one_raised(window):
    from acorn.gui.workspaces import by_id
    window.set_workspace("analyze")
    ws = by_id("analyze")
    first = window._plugin_docks.get(ws.docks[0])
    if first is None:
        import pytest
        pytest.skip("plugin not installed")
    assert not first.isHidden()


def test_control_panel_width_follows_the_workspace(window):
    """Explore's two short tabs should not claim Annotate's width."""
    window.resize(1600, 1000)
    window.show()
    from PyQt6.QtWidgets import QApplication
    widths = {}
    for ws in WORKSPACES:
        window.set_workspace(ws.wid)
        QApplication.processEvents()     # the sizing is deferred one turn
        QApplication.processEvents()
        widths[ws.wid] = window._main_splitter.sizes()[1]
    assert widths["annotate"] > widths["explore"], "Annotate needs more room than Explore"
    assert widths["simulate"] <= widths["annotate"], (
        "Simulate gives an edge to the simulator docks; the panel must not also be wide"
    )


def test_the_image_always_keeps_the_larger_share(window):
    window.resize(1600, 1000)
    window.show()
    from PyQt6.QtWidgets import QApplication
    for ws in WORKSPACES:
        window.set_workspace(ws.wid)
        QApplication.processEvents()
        QApplication.processEvents()
        canvas, panel = window._main_splitter.sizes()
        assert canvas > panel, f"{ws.wid}: control panel is wider than the image"


def test_no_workspace_makes_the_panel_narrower_than_its_own_controls(window):
    """
    A panel narrower than its buttons cuts them off, which reads as a broken
    window rather than a tight one. Simulate hit this: it gives an edge to the
    simulator docks, and a 320 px panel on top of that clipped Contrast.
    """
    from PyQt6.QtWidgets import QApplication
    window.resize(1600, 1000)
    window.show()
    for ws in WORKSPACES:
        window.set_workspace(ws.wid)
        QApplication.processEvents()
        QApplication.processEvents()
        panel = window._main_splitter.sizes()[1]
        assert panel >= window._control_tabs.minimumWidth(), (
            f"{ws.wid}: panel {panel}px is below the control panel's own minimum"
        )


def test_the_contrast_tab_can_scroll_rather_than_clip(window):
    """It was the one tab added without a scroll wrapper."""
    from PyQt6.QtWidgets import QScrollArea
    window.show_all_panels()
    tabs = window._control_tabs
    idx = [tabs.tabText(i) for i in range(tabs.count())].index("Contrast")
    assert isinstance(tabs.widget(idx), QScrollArea)
    assert window._contrast_panel.params().method   # still wired through the wrapper
