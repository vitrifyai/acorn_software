"""A plugin that breaks must say so; one that is merely optional must not.

The failure this guards against is specific and has already happened once: a
stale signal connection made a dock fail to build, every test still passed, and
the only symptom was a panel that quietly was not there. Absence is
indistinguishable from a removed feature, which sends you looking in the wrong
place.

The opposite mistake matters just as much. Reporting an uninstalled optional
package as a fault on every launch teaches people to dismiss the notice, and
then the real one gets dismissed with it.
"""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from acorn.plugin_loader import PluginFailure, is_missing_dependency

pytest.importorskip("PyQt6")


@pytest.mark.parametrize("exc,expected", [
    (ImportError("no torch"), True),
    (ModuleNotFoundError("No module named 'mrcfile'"), True),
    (ImportError("cannot import name 'Foo' from 'bar'"), True),   # too-old package
    (RuntimeError("stale signal connection"), False),
    (AttributeError("'Panel' object has no attribute 'action_requested'"), False),
    (TypeError("bad argument"), False),
    (ValueError("nonsense"), False),
])
def test_only_import_failures_count_as_optional(exc, expected):
    assert is_missing_dependency(exc) is expected


def test_placeholder_is_offered_for_faults_and_withheld_for_missing_extras():
    from PyQt6.QtWidgets import QApplication

    from acorn.gui.plugin_failure import FailurePanel, placeholder_for

    app = QApplication.instance() or QApplication([])
    try:
        assert placeholder_for("3-D Viewer", ModuleNotFoundError("no mrcfile")) is None
        panel = placeholder_for("SEM Simulation", RuntimeError("boom"))
        assert isinstance(panel, FailurePanel)
        panel.deleteLater()
    finally:
        app.processEvents()


def test_placeholder_names_the_tool_and_the_error():
    from PyQt6.QtWidgets import QApplication, QLabel

    from acorn.gui.plugin_failure import FailurePanel

    app = QApplication.instance() or QApplication([])
    try:
        panel = FailurePanel("SEM Simulation", RuntimeError("stale signal connection"))
        text = " ".join(lbl.text() for lbl in panel.findChildren(QLabel))
        assert "SEM Simulation" in text
        assert "RuntimeError" in text and "stale signal connection" in text
        # the traceback belongs somewhere reachable but not in the reader's face
        assert "Traceback" in panel.toolTip() or "RuntimeError" in panel.toolTip()
        panel.deleteLater()
    finally:
        app.processEvents()


def test_loader_records_and_classifies_what_failed(monkeypatch):
    """Exercised through a fake entry point, since the real ones all load."""
    import acorn.plugin_loader as loader

    class _EP:
        def __init__(self, name, exc):
            self.name = name
            self._exc = exc

        def load(self):
            raise self._exc

    fake = [_EP("broken_plugin", RuntimeError("boom")),
            _EP("optional_plugin", ModuleNotFoundError("No module named 'torch'"))]
    monkeypatch.setattr(loader, "entry_points", lambda group: fake, raising=False)
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: fake)

    plugins = loader.discover_plugins(context=object())
    assert plugins == []

    failures = {f.name: f for f in loader.load_failures()}
    assert set(failures) == {"broken_plugin", "optional_plugin"}
    assert failures["broken_plugin"].missing_dependency is False
    assert failures["optional_plugin"].missing_dependency is True
    assert isinstance(failures["broken_plugin"], PluginFailure)


def test_load_failures_are_cleared_between_discoveries(monkeypatch):
    """Stale entries would report a plugin as broken long after it was fixed."""
    import acorn.plugin_loader as loader

    class _EP:
        name = "broken_plugin"

        def load(self):
            raise RuntimeError("boom")

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [_EP()])
    loader.discover_plugins(context=object())
    assert len(loader.load_failures()) == 1

    monkeypatch.setattr("importlib.metadata.entry_points", lambda group=None: [])
    loader.discover_plugins(context=object())
    assert loader.load_failures() == []


def test_a_broken_plugin_is_visible_and_an_optional_one_is_not(monkeypatch):
    """End to end, in a real window.

    Unit tests build panels directly; this one goes through the path the
    application actually takes, which is the path that broke.
    """
    from PyQt6.QtWidgets import QApplication

    import acorn.plugin_loader as loader
    from acorn.plugin_base import AcornPlugin

    class _Broken(AcornPlugin):
        TAB_LABEL, PLUGIN_ID, sort_order = "Broken Tool", "broken_tool", 99

        def create_panel(self):
            raise RuntimeError("stale signal connection")

    class _OptionalMissing(AcornPlugin):
        TAB_LABEL, PLUGIN_ID, sort_order = "Optional Tool", "optional_tool", 98

        def create_panel(self):
            raise ModuleNotFoundError("No module named 'torch'")

    real = loader.discover_plugins
    monkeypatch.setattr(
        loader, "discover_plugins",
        lambda ctx: [*real(ctx), _Broken(ctx), _OptionalMissing(ctx)])

    app = QApplication.instance() or QApplication([])
    from acorn.gui.main_window import MainWindow
    from acorn.gui.plugin_failure import FailurePanel

    window = MainWindow()
    try:
        app.processEvents()
        tabs = window._control_tabs
        labels = [tabs.tabText(i) for i in range(tabs.count())]
        stand_ins = [tabs.tabText(i) for i in range(tabs.count())
                     if isinstance(tabs.widget(i), FailurePanel)]

        assert "Broken Tool" in stand_ins, labels
        assert "Optional Tool" not in labels, labels
        # The rest of the application came up regardless. Checked as "some real
        # panel exists" rather than by naming one: which tabs are present
        # depends on the restored workspace, and an earlier test in the suite
        # leaves a different one saved.
        assert [t for t in labels if t not in stand_ins], labels
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()
