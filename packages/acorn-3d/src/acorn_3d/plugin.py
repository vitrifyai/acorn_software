"""3D volume plugin for ACORN."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from acorn_3d.volume_loader import VolumeImage


class VolumePlugin(AcornPlugin):
    TAB_LABEL         = "3D"
    PLUGIN_ID         = "acorn_3d"
    sort_order        = 30
    FLOATING          = True
    FLOATING_TITLE    = "3D Viewer"
    FLOATING_SHORTCUT = "Ctrl+Shift+3"

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel: Optional["VolumePanel"] = None
        self._current_vol: Optional["VolumeImage"] = None

    def create_panel(self) -> QWidget:
        from acorn_3d.panel import VolumePanel
        self._panel = VolumePanel()
        self._panel.slice_changed.connect(self._on_slice_changed)
        self._panel.projection_requested.connect(self._on_projection_requested)
        return self._panel

    def setup_menus(self, menubar) -> None:
        self._context.register_menu_action(
            "File",
            "Open as Volume (MRC / TIFF)...",
            self._open_volume_dialog,
        )

    def _open_volume_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            None,
            "Open Volume",
            "",
            "Volume files (*.mrc *.mrcs *.tif *.tiff)",
        )
        if not path:
            return
        self._load_volume(Path(path))

    def _load_volume(self, path: Path) -> None:
        from acorn_3d.volume_loader import VolumeImage
        try:
            vol = VolumeImage.from_file(path)
        except Exception as exc:
            QMessageBox.critical(None, "Volume load error", str(exc))
            return

        self._current_vol = vol
        self._panel.set_volume(vol)

        cw = self._context.canvas_widget()
        if cw is not None:
            params = self._context.current_contrast_params
            cw.canvas.load_image(vol, params)

        self._context.set_status(
            f"Volume loaded: {vol.filename}  ({vol.n_slices} slices)"
        )

    def _on_slice_changed(self, z: int) -> None:
        if self._current_vol is None:
            return
        self._current_vol.set_slice(z)
        cw = self._context.canvas_widget()
        if cw is None:
            return
        params = self._context.current_contrast_params
        cw.canvas.load_image(self._current_vol, params)
        self._context.slice_changed.emit(z)

    def _on_projection_requested(self, method: str, z_from: int, z_to: int) -> None:
        if self._current_vol is None:
            return
        import numpy as np
        from acorn.core.dm4_loader import DM4Image
        try:
            arr = self._current_vol.projection(method=method, z_from=z_from, z_to=z_to)
        except Exception as exc:
            QMessageBox.critical(None, "Projection error", str(exc))
            return

        # Wrap in a DM4Image-like object for display
        vol = self._current_vol
        proj = DM4Image()
        proj.raw = arr.astype(np.float32)
        proj.meta.pixel_size = vol.pixel_size
        proj.meta.pixel_size_from_header = vol.meta.pixel_size_from_header
        proj.meta.shape = arr.shape
        proj.meta.filepath = vol.filepath
        proj.meta.filename = f"{vol.filename}_proj_{method}_{z_from}-{z_to}"

        cw = self._context.canvas_widget()
        if cw is not None:
            params = self._context.current_contrast_params
            cw.canvas.load_image(proj, params)
        self._context.set_status(
            f"Projection ({method}, z={z_from}-{z_to}) displayed. "
            "Use Export tab to save."
        )
