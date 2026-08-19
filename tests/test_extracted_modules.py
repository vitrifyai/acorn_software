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


def test_no_undefined_module_globals():
    """Catches an import stranded on the wrong side of the split."""
    import ast
    import importlib
    import builtins

    for mod_name in ("acorn.gui.movie", "acorn.gui.threads", "acorn.gui.sam_controller",
                     "acorn.gui.detector_controller", "acorn.gui.export_controller",
                     "acorn.gui.main_window"):
        mod = importlib.import_module(mod_name)
        tree = ast.parse(open(mod.__file__).read())
        defined = set(dir(builtins)) | set(vars(mod))
        missing = set()
        for node in ast.walk(tree):
            # only module-level calls we can attribute confidently
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in defined:
                    missing.add(node.func.id)
        # names bound locally inside functions are fine; filter those out
        local_binds = {n.id for n in ast.walk(tree)
                       if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
        local_binds |= {a.arg for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef) for a in n.args.args}
        # functions defined anywhere, including nested inside other functions
        local_binds |= {n.name for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
        local_binds |= {(al.asname or al.name).split(".")[0]
                        for n in ast.walk(tree)
                        if isinstance(n, (ast.Import, ast.ImportFrom)) for al in n.names}
        real = missing - local_binds
        assert not real, f"{mod_name} calls undefined names: {sorted(real)}"


def test_motion_plot_dialog_runs_its_numpy_maths(app):
    """movie.py lost `import numpy as np` in the split; this would have caught it."""
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
