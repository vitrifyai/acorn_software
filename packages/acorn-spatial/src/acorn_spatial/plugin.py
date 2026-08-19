"""Spatial-statistics plugin — floating dock for clustering / hotspot / nearest-
neighbour / cross-label association analysis of detected features."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class SpatialPlugin(AcornPlugin):
    TAB_LABEL          = "Spatial"
    PLUGIN_ID          = "acorn_spatial"
    sort_order         = 25
    FLOATING           = True
    FLOATING_TITLE     = "Spatial Analysis"
    FLOATING_SHORTCUT  = "Ctrl+Shift+P"
    FLOATING_MIN_WIDTH = 360

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None

    def create_panel(self) -> QWidget:
        from acorn_spatial.panel import SpatialPanel
        self._panel = SpatialPanel(self._context)
        return self._panel
