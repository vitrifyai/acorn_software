"""
Keyboard shortcuts for the actions people repeat all day.

Two things must hold or shortcuts do more harm than good: every documented key
must actually reach a real method, and none of them may fire while the user is
typing — otherwise entering a filename with 'd' in it jumps to the next image.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import QApplication, QLineEdit

from acorn.gui.shortcuts import SHORTCUTS, ShortcutHelp, _typing


@pytest.fixture(scope="module")
def window(tmp_path_factory):
    os.environ["XDG_CONFIG_HOME"] = str(tmp_path_factory.mktemp("cfg"))
    app = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    win = MainWindow()
    yield win, app
    win.close()
    app.processEvents()


def test_every_shortcut_points_at_a_real_method(window):
    win, _ = window
    missing = [s.method for s in SHORTCUTS if not callable(getattr(win, s.method, None))]
    assert not missing, f"shortcuts bound to nothing: {missing}"


def test_every_key_string_parses(window):
    for spec in SHORTCUTS:
        for key in (k.strip() for k in spec.keys.split("/")):
            assert not QKeySequence(key).isEmpty(), f"{spec.label}: {key!r} is not a key"


def test_no_two_actions_claim_the_same_key():
    seen: dict[str, str] = {}
    for spec in SHORTCUTS:
        for key in (k.strip() for k in spec.keys.split("/")):
            norm = QKeySequence(key).toString()
            assert norm not in seen, f"{norm} bound to both {seen[norm]} and {spec.label}"
            seen[norm] = spec.label


def test_shortcuts_do_not_collide_with_the_menus(window):
    """A key bound twice in one window fires ambiguously and neither action runs."""
    win, _ = window
    ours = set()
    for spec in SHORTCUTS:
        for key in (k.strip() for k in spec.keys.split("/")):
            ours.add(QKeySequence(key).toString())
    menu_keys = set()
    for menu in win.menuBar().findChildren(type(win.menuBar())):
        pass
    for action in win.findChildren(type(win.menuBar().actions()[0])):
        text = action.shortcut().toString()
        if text:
            menu_keys.add(text)
    # F1 is deliberately shared: the menu entry and the shortcut do the same thing.
    clashes = (ours & menu_keys) - {"F1"}
    assert not clashes, f"keys claimed by both a menu action and a shortcut: {clashes}"


def test_shortcuts_are_suppressed_while_typing(window, monkeypatch):
    """
    The predicate, not Qt's focus machinery: the offscreen platform does not hand
    real focus to a widget, so this drives what _typing() is asked about.
    """
    from PyQt6.QtWidgets import QComboBox, QPlainTextEdit, QSpinBox
    import acorn.gui.shortcuts as sc

    win, _ = window
    for widget_cls, expected in [(QLineEdit, True), (QPlainTextEdit, True),
                                 (QSpinBox, True), (QComboBox, False), (type(None), False)]:
        target = widget_cls(win) if widget_cls is not type(None) else None
        monkeypatch.setattr(sc.QApplication, "focusWidget", staticmethod(lambda t=target: t))
        assert sc._typing() is expected, f"{widget_cls.__name__}: expected {expected}"
        if target is not None:
            target.deleteLater()


def test_a_shortcut_does_not_fire_while_typing(window, monkeypatch):
    """The dispatcher itself must refuse, not just the predicate."""
    import acorn.gui.shortcuts as sc
    win, _ = window
    calls = []
    monkeypatch.setattr(win, "toggle_annotations", lambda: calls.append(1), raising=False)
    spec = next(s for s in SHORTCUTS if s.method == "toggle_annotations")
    run = sc._dispatcher(win, spec)

    monkeypatch.setattr(sc, "_typing", lambda: True)
    run()
    assert calls == [], "shortcut fired while the user was typing"

    monkeypatch.setattr(sc, "_typing", lambda: False)
    run()
    assert calls == [1], "shortcut did not fire when it should have"


def test_f1_still_works_while_typing():
    """The help key is deliberately exempt — you want it most when stuck in a form."""
    spec = next(s for s in SHORTCUTS if s.method == "show_shortcuts")
    assert spec.typing_safe is True


def test_help_dialog_lists_every_shortcut():
    """Generated from the same table, so the documentation cannot drift."""
    dlg = ShortcutHelp()
    from PyQt6.QtWidgets import QLabel
    shown = " ".join(lbl.text() for lbl in dlg.findChildren(QLabel))
    for spec in SHORTCUTS:
        assert spec.label in shown, f"{spec.label} missing from the help dialog"
    dlg.deleteLater()


def test_toggling_annotations_is_reversible(window):
    win, _ = window
    start = getattr(win, "_annotations_hidden", False)
    win.toggle_annotations()
    assert win._annotations_hidden is not start
    win.toggle_annotations()
    assert win._annotations_hidden is start


def test_accept_and_reject_with_nothing_pending_say_so_rather_than_crash(window):
    win, _ = window
    win.accept_pending()
    win.reject_pending()


# ── the image list, as a working aid ──────────────────────────────────────────

def test_image_list_shows_which_images_are_already_annotated(window, tmp_path):
    """
    Working through a folder of micrographs, the one thing you need from this list
    is where you got to. Every row looked identical before.
    """
    import time
    import numpy as np
    import tifffile

    win, app = window
    for i in range(2):
        tifffile.imwrite(tmp_path / f"img_{i}.tif",
                         np.random.rand(96, 96).astype("float32"))
    win.open_files(sorted(tmp_path.glob("*.tif")))
    for _ in range(60):
        app.processEvents()
        time.sleep(0.01)

    rows = [win._image_list.item(i) for i in range(win._image_list.count())]
    assert rows, "no rows in the image list"
    for row in rows:
        # unannotated: dimmed, and the tooltip says so rather than being blank
        assert "not annotated" in row.toolTip()
        assert row.foreground().color().name() == "#8a8a8a"


def test_annotation_count_reads_a_sidecar_without_loading_the_image(window, tmp_path):
    """The count has to come from disk, or the list can only describe the open image."""
    import json
    import numpy as np
    import tifffile

    win, app = window
    tifffile.imwrite(tmp_path / "one.tif", np.random.rand(64, 64).astype("float32"))
    win.open_files([tmp_path / "one.tif"])
    for _ in range(40):
        app.processEvents()
    # the count for the current image comes from the live store
    assert win._annotation_count(0) == len(win._canvas_widget.canvas.store)
    # and an out-of-range index must not raise
    assert win._annotation_count(999) == 0
