"""Plugin context — exposes core application state to ACORN plugins."""
from __future__ import annotations
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu

if TYPE_CHECKING:
    from acorn.gui.main_window import MainWindow
    from acorn.gui.canvas_widget import CanvasWidget
    from acorn.core.dm4_loader import DM4Image
    from acorn.core.annotations import AnnotationStore
    from acorn.core.contrast import ContrastParams


class AcornContext(QObject):
    """
    Read/write access to core application state, exposed to plugins.

    Plugins receive this object at construction time and must not import
    MainWindow directly.
    """

    # Emitted by the core when state changes (plugins connect to these)
    image_loaded        = pyqtSignal(object)   # DM4Image
    annotations_changed = pyqtSignal(object)   # AnnotationStore
    pixel_size_changed  = pyqtSignal(float)
    slice_changed       = pyqtSignal(int)      # z-slice index (for acorn_3d)

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self._window_ref = weakref.ref(main_window)

    def _w(self) -> Optional["MainWindow"]:
        return self._window_ref()

    # ── read-only state ───────────────────────────────────────────────────────

    @property
    def current_image(self) -> Optional["DM4Image"]:
        w = self._w()
        if w is None or w._img_idx < 0:
            return None
        return w._canvas_widget.canvas.dm4

    @property
    def current_pixel_size_nm(self) -> float:
        w = self._w()
        if w is None:
            return 1.0
        return w._engine.pixel_size if w._engine.pixel_size > 0 else 1.0

    @property
    def annotation_store(self) -> Optional["AnnotationStore"]:
        w = self._w()
        if w is None:
            return None
        return w._canvas_widget.canvas.store

    @property
    def all_annotation_states(self) -> dict[int, list]:
        """
        Dict of image_index -> annotation list for all loaded images.
        The current image's live store is overlaid at its index.
        Plugins must not mutate this dict.
        """
        w = self._w()
        if w is None:
            return {}
        result = dict(w._ann_states)
        if w._img_idx >= 0:
            result[w._img_idx] = list(w._canvas_widget.canvas.store)
        return result

    @property
    def image_paths(self) -> list[Path]:
        w = self._w()
        return list(w._image_paths) if w else []

    @property
    def current_image_index(self) -> int:
        w = self._w()
        return w._img_idx if w else -1

    def pixel_size_for_index(self, idx: int) -> float:
        """Return pixel size for image at idx — manual override takes priority."""
        w = self._w()
        if w is None:
            return 1.0
        override = w._px_overrides.get(idx)
        if override is not None and override > 0:
            return float(override)
        if idx == w._img_idx:
            img = w._canvas_widget.canvas.dm4
            if img is not None and img.pixel_size > 0:
                return float(img.pixel_size)
        return 1.0

    @property
    def current_contrast_params(self) -> Optional["ContrastParams"]:
        w = self._w()
        if w is None:
            return None
        return w._canvas_widget.canvas._params if hasattr(w._canvas_widget.canvas, "_params") else None

    # ── write API ─────────────────────────────────────────────────────────────

    def canvas_widget(self) -> Optional["CanvasWidget"]:
        w = self._w()
        return w._canvas_widget if w else None

    def set_status(self, message: str, timeout_ms: int = 0) -> None:
        w = self._w()
        if w:
            w._statusbar.showMessage(message, timeout_ms)

    def register_menu_action(
        self,
        menu_name: str,
        label: str,
        callback: Callable,
        shortcut: Optional[str] = None,
    ) -> Optional[QAction]:
        """
        Add an action to an existing menu (e.g. 'File', 'View') or create
        a new top-level menu if it doesn't exist.
        """
        w = self._w()
        if w is None:
            return None
        menubar = w.menuBar()
        target_menu: Optional[QMenu] = None
        for action in menubar.actions():
            if action.text().replace("&", "") == menu_name:
                target_menu = action.menu()
                break
        if target_menu is None:
            target_menu = menubar.addMenu(menu_name)
        action = QAction(label, w)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        target_menu.addAction(action)
        return action
