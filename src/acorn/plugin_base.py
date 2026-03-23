"""Base class for all ACORN plugins."""
from __future__ import annotations
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from PyQt6.QtWidgets import QMenuBar


class AcornPlugin:
    """
    Base class for all ACORN plugins.

    Every plugin package registers one subclass of this via an entry point
    in the 'acorn.plugins' group.

    Lifecycle:
      1. MainWindow calls discover_plugins(context) at startup.
      2. Each plugin class is instantiated with the AcornContext.
      3. MainWindow calls create_panel() and adds the result as a tab.
      4. MainWindow calls setup_menus(menubar) after the window is shown.
      5. On quit, MainWindow calls teardown().
    """

    TAB_LABEL: str = ""   # label shown on the tab
    PLUGIN_ID: str = ""   # unique slug, e.g. "acorn_analysis"

    def __init__(self, context: "AcornContext") -> None:
        self._context = context

    def create_panel(self) -> QWidget:
        """Return the QWidget to insert as a tab. Called once at startup."""
        raise NotImplementedError

    def setup_menus(self, menubar: "QMenuBar") -> None:
        """Called once after the window is shown. Default: no-op."""

    def teardown(self) -> None:
        """Called on application quit. Stop threads, release resources."""

    @property
    def sort_order(self) -> int:
        """Lower numbers appear first. Default 100."""
        return 100
