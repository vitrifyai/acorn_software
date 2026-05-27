"""LLM Assistant plugin — dock-widget chat panel."""
from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from PyQt6.QtWidgets import QMenuBar


class LLMPlugin(AcornPlugin):
    TAB_LABEL = "Assistant"
    PLUGIN_ID = "acorn_llm"

    @property
    def sort_order(self) -> int:
        return 5

    def create_panel(self):
        return None  # uses a dock widget, not a tab

    def setup_menus(self, menubar: "QMenuBar") -> None:
        from acorn_llm.panel import AssistantPanel

        w = self._context._w()
        if w is None:
            return

        self._panel = AssistantPanel(self._context, w)
        self._dock = QDockWidget("AI Assistant", w)
        self._dock.setWidget(self._panel)
        self._dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable,
        )
        self._dock.setMinimumWidth(300)
        w.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.hide()

        self._context.register_menu_action(
            "View",
            "AI Assistant",
            lambda: self._dock.setVisible(not self._dock.isVisible()),
            shortcut="Ctrl+Shift+A",
        )
