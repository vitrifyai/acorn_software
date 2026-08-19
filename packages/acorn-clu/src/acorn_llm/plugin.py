"""LLM Assistant plugin — floating dock-widget chat panel."""
from __future__ import annotations
from typing import TYPE_CHECKING

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from PyQt6.QtWidgets import QMenuBar


class LLMPlugin(AcornPlugin):
    TAB_LABEL          = "Assistant"
    PLUGIN_ID          = "acorn_llm"
    FLOATING           = True
    FLOATING_TITLE     = "AI Assistant"
    FLOATING_SHORTCUT  = "Ctrl+Shift+A"
    FLOATING_MIN_WIDTH = 300

    @property
    def sort_order(self) -> int:
        return 5

    def create_panel(self):
        # Gate on credentials: no key/base_url → no dock (setup_menus adds a configure item)
        from acorn_llm.config import load_config
        cfg = load_config()
        if not cfg.api_key and not cfg.base_url:
            self._gated = True
            return None

        from acorn_llm.panel import AssistantPanel
        self._gated = False
        self._panel = AssistantPanel(self._context, self._context._w())
        return self._panel

    def setup_menus(self, menubar: "QMenuBar") -> None:
        # When credentials are missing, offer a settings shortcut in place of the toggle.
        if getattr(self, "_gated", False):
            self._context.register_menu_action(
                "View",
                "AI Assistant (configure API key in Settings)",
                self._open_settings,
                shortcut="Ctrl+Shift+A",
            )

    def _open_settings(self) -> None:
        from acorn_llm.settings_dialog import LLMSettingsDialog
        w = self._context._w()
        dlg = LLMSettingsDialog(w)
        dlg.exec()
