"""
The mixin split moved ~2,100 lines of method bodies between files. A method that
imports cleanly can still raise NameError the moment it runs, if the module it
landed in lost a module-level import along the way — which is exactly what
happened to numpy in movie.py and SAMThread in main_window.py.

These tests actually execute code from each extracted module rather than only
importing it.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_every_extracted_module_imports():
    import acorn.gui.detector_controller  # noqa: F401
    import acorn.gui.export_controller    # noqa: F401
    import acorn.gui.movie                # noqa: F401
    import acorn.gui.sam_controller       # noqa: F401
    import acorn.gui.threads              # noqa: F401


def test_no_function_references_a_name_its_module_cannot_supply():
    """
    The bug class the mixin split introduces: a method lands in a file whose
    imports do not cover it. The module still imports and the suite still
    collects; the NameError waits for a user to walk that path.

    acorn.gui._split_audit resolves every loaded name against its own scope, its
    enclosing scopes, the module, and builtins — reading the module scope from the
    source rather than from vars(module), which would happily pass a file whose
    import line had been deleted.
    """
    from acorn.gui._split_audit import audit
    report = audit()
    assert not report, "\n".join(
        f"{mod}.{fn} uses undefined {sorted(names)}"
        for mod, fns in report.items() for fn, names in fns.items()
    )


def test_the_audit_actually_catches_a_missing_import():
    """
    Guards the guard. An audit that passes because it looks in the wrong place is
    worse than none — the first version of this check read vars(module), which
    still holds a name after its import is deleted, so it verified nothing.
    """
    import textwrap
    from acorn.gui._split_audit import undefined_names

    sound = textwrap.dedent("""
        import numpy as np
        def f():
            return np.zeros(3)
    """)
    broken = textwrap.dedent("""
        def f():
            return np.zeros(3)
    """)
    closure = textwrap.dedent("""
        def outer(msg):
            def inner():
                return msg          # closure, not an undefined name
            return inner
    """)
    import tempfile, os
    def check(src):
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as fh:
            fh.write(src)
        try:
            return undefined_names(path)
        finally:
            os.unlink(path)

    assert not check(sound), "flagged a correct module"
    assert check(broken), "missed a deleted import — the check is vacuous"
    assert not check(closure), "flagged a closure variable as undefined"


def test_no_two_mixins_define_the_same_method():
    """
    Two mixins defining one name would have the MRO silently pick the first,
    and the other body would never run again.
    """
    import ast
    from acorn.gui._split_audit import MIXIN_MODULES
    import importlib

    seen: dict[str, str] = {}
    clashes = []
    for mod_name in MIXIN_MODULES:
        if mod_name == "acorn.gui.main_window":
            continue                      # MainWindow legitimately overrides
        mod = importlib.import_module(mod_name)
        tree = ast.parse(open(mod.__file__).read())
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef)
                    and n.name.endswith("Mixin")]:
            for m in cls.body:
                if isinstance(m, ast.FunctionDef):
                    if m.name in seen:
                        clashes.append(f"{m.name}: {seen[m.name]} and {cls.name}")
                    seen[m.name] = cls.name
    assert not clashes, "; ".join(clashes)


def test_motion_plot_dialog_runs_its_numpy_maths(app):
    """
    Exercises the drift-plot maths rather than only importing the module.

    (The module-level numpy import in movie.py is not load-bearing — every runtime
    use has its own local import. It is there because the string annotations
    reference np, which ruff reports as an undefined name.)
    """
    from acorn.gui.movie import MotionPlotDialog
    shifts = np.cumsum(np.random.RandomState(0).randn(12, 2), axis=0)
    dlg = MotionPlotDialog(shifts, start_frame=0)
    assert dlg is not None
    dlg.deleteLater()


def test_worker_threads_construct(app):
    from acorn.gui.threads import FrameProcessThread, LoadThread, SAMThread
    t = SAMThread(lambda: None, None)
    assert t is not None
    assert LoadThread([]) is not None


def test_mainwindow_still_composes_every_mixin():
    from acorn.gui.main_window import MainWindow
    names = [c.__name__ for c in MainWindow.__mro__]
    for mixin in ("SAMControllerMixin", "DetectorControllerMixin",
                  "ExportControllerMixin", "MovieControllerMixin"):
        assert mixin in names
