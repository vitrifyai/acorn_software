"""Removing an image must shift every per-image dict, not just some of them.

Seven dicts are keyed by image index. Three were being shifted. The four that
were not held the image cache, its fingerprints, SAM exclusion zones and crop
regions -- so after a removal, work done on one image silently applied to
whatever file inherited its index, and the cache could serve the wrong picture
entirely.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")


def test_every_per_image_dict_is_listed_for_reindexing():
    """The list must stay complete. A dict added later and forgotten here is
    exactly how this bug happened the first time."""
    import re
    from pathlib import Path

    from acorn.gui.main_window import MainWindow

    src = Path("src/acorn/gui/main_window.py").read_text()
    declared = set(re.findall(r"self\.(_[a-z_]+): dict\[int,", src))
    listed = set(MainWindow._PER_IMAGE_DICTS)
    missing = declared - listed
    assert not missing, f"per-image dicts not reindexed on removal: {sorted(missing)}"


class _Stand_in:
    """Holds the same per-image dicts without constructing a QMainWindow.

    MainWindow.__new__ is not usable here: getattr on a QObject whose
    super().__init__ never ran raises RuntimeError instead of returning a
    default, which tests the Qt binding rather than the shifting logic.
    """

    def __init__(self):
        from acorn.gui.main_window import MainWindow
        self._PER_IMAGE_DICTS = MainWindow._PER_IMAGE_DICTS


def _make(names, contents):
    obj = _Stand_in()
    for name in names:
        setattr(obj, name, dict(contents))
    return obj


def test_reindex_drops_the_row_and_shifts_the_rest():
    from acorn.gui.main_window import MainWindow

    obj = _make(MainWindow._PER_IMAGE_DICTS, {0: "a", 1: "b", 2: "c", 3: "d"})

    MainWindow._reindex_after_removal(obj, 1)

    for name in MainWindow._PER_IMAGE_DICTS:
        got = getattr(obj, name)
        assert got == {0: "a", 1: "c", 2: "d"}, (name, got)


def test_reindex_tolerates_a_dict_that_does_not_exist_yet():
    """Defensive: the guard must not raise if a dict has not been built."""
    from acorn.gui.main_window import MainWindow

    obj = _Stand_in()
    obj._ann_states = {0: "a", 1: "b"}
    MainWindow._reindex_after_removal(obj, 0)     # the other six are absent
    assert obj._ann_states == {0: "b"}


def test_removing_the_last_row_leaves_the_others_untouched():
    from acorn.gui.main_window import MainWindow

    obj = _make(MainWindow._PER_IMAGE_DICTS, {0: "a", 1: "b"})
    MainWindow._reindex_after_removal(obj, 1)
    for name in MainWindow._PER_IMAGE_DICTS:
        assert getattr(obj, name) == {0: "a"}, name
