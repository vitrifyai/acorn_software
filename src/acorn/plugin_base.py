from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from PyQt6.QtWidgets import QWidget

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from PyQt6.QtWidgets import QMenuBar

# Valid workflow stage names plugins can declare themselves part of.
WORKFLOW_STAGES = ("Annotate", "Segment", "Measure", "Train", "Export")


class AcornPlugin:
    """
    Base class for all ACORN plugins.

    Every plugin package registers one subclass via the 'acorn.plugins' entry point.

    Lifecycle:
      1. MainWindow calls discover_plugins(context) at startup.
      2. Each plugin class is instantiated with the AcornContext.
      3. MainWindow calls create_panel() — result is routed based on the plugin's flags:
           - FLOATING True        → widget opens in a movable/floating dock toggled from View
           - WORKFLOW_STAGE set   → widget injected into that workflow tab as a section
           - WORKFLOW_STAGE None  → widget added as its own tab using TAB_LABEL
           - create_panel() returns None → no panel (plugin may still add menus/docks itself)
      4. MainWindow calls setup_menus(menubar) after the window is shown.
      5. On quit, MainWindow calls teardown().

    Workflow injection:
      Set WORKFLOW_STAGE to one of: "Annotate", "Segment", "Measure", "Train", "Export"
      Optionally set WORKFLOW_SECTION_LABEL for a visible header in the workflow tab.
      If the stage name is unknown or misspelled, the plugin falls back to its own tab.

    Floating dock mode:
      Set FLOATING = True and implement create_panel() as usual — the widget is placed
      in a movable/floating QDockWidget (hidden initially) with a View-menu toggle, instead
      of a tab. Use for side tools that aren't part of the main workflow (e.g. 3D viewer,
      tracking). Returning None from create_panel() skips the dock (e.g. when ungated).
    """

    TAB_LABEL:             str = ""     # tab label when not injected into a workflow stage
    PLUGIN_ID:             str = ""     # unique slug, e.g. "acorn_analysis"
    WORKFLOW_STAGE:        Optional[str] = None   # e.g. "Measure"
    WORKFLOW_SECTION_LABEL: str = ""    # section header shown inside the workflow tab

    # Floating dock mode (alternative to a tab) — toggled from the View menu
    FLOATING:           bool = False
    FLOATING_TITLE:     str  = ""        # dock title (defaults to TAB_LABEL or PLUGIN_ID)
    FLOATING_SHORTCUT:  str  = ""        # optional View-menu shortcut, e.g. "Ctrl+Shift+3"
    FLOATING_AREA:      str  = "right"   # initial dock area: left | right | top | bottom
    FLOATING_MIN_WIDTH: int  = 0         # optional minimum dock width in px (0 = no minimum)

    def __init__(self, context: "AcornContext") -> None:
        self._context = context

    def create_panel(self) -> Optional[QWidget]:
        """Return the QWidget to display, or None to skip tab/section creation."""
        raise NotImplementedError

    def setup_menus(self, menubar: "QMenuBar") -> None:
        """Called once after the window is shown. Default: no-op."""

    def teardown(self) -> None:
        """Called on application quit. Stop threads, release resources."""

    @property
    def sort_order(self) -> int:
        """Lower numbers appear first. Default 100."""
        return 100
