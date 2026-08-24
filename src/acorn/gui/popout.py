"""
Detaching a control tab into its own window.

The control panel shows one tab at a time, which is the point — but some work
needs two at once. In Analyze the Measure tab and the Spatial Analysis dock end
up sharing the right edge, each too narrow for its own contents, and switching to
another tab hides whichever you were reading.

Popping a tab out turns it into a floating window you can put anywhere, including
on a second monitor. The tab leaves the bar while it is out and returns to its
original position when the window is closed, so nothing is lost either way.

This is deliberately not a second layout system: the popped-out panel is the same
widget, just re-parented, so its state, signals and whatever it was showing carry
across untouched.
"""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDockWidget, QMainWindow, QTabWidget, QWidget


class PoppedOutPanel(QDockWidget):
    """A control tab living in its own floating window."""

    def __init__(self, title: str, widget: QWidget, tabs: QTabWidget,
                 index: int, parent: QMainWindow) -> None:
        super().__init__(title, parent)
        self.setObjectName(f"popout::{title}")
        self._tabs = tabs
        self._title = title
        self._widget = widget
        self._home_index = index
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable
            | QDockWidget.DockWidgetFeature.DockWidgetFloatable
            | QDockWidget.DockWidgetFeature.DockWidgetClosable
        )
        self.setWidget(widget)
        self.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

    def closeEvent(self, event) -> None:      # noqa: N802 - Qt name
        """Put the tab back where it came from."""
        self.return_to_tabs()
        event.accept()

    def return_to_tabs(self) -> None:
        """Re-insert the panel into the tab bar at roughly its old position."""
        widget = self.widget()
        if widget is None:
            return
        self.setWidget(None)          # release before re-parenting
        index = min(self._home_index, self._tabs.count())
        self._tabs.insertTab(index, widget, self._title)
        self._tabs.setCurrentIndex(index)


def pop_out(tabs: QTabWidget, index: int, window: QMainWindow) -> PoppedOutPanel | None:
    """
    Detach tab *index* into a floating window. Returns it, or None if it cannot.

    Popping the last tab is allowed and leaves the panel empty. That is the case
    this exists for: in Analyze the only tab is Measure, and floating it frees the
    whole right-hand side for the image and the Spatial Analysis dock. The caller
    hides the empty panel and shows it again when the tab comes back.
    """
    widget = tabs.widget(index)
    if widget is None:
        return None
    title = tabs.tabText(index)

    tabs.removeTab(index)
    panel = PoppedOutPanel(title, widget, tabs, index, window)
    window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, panel)
    panel.setFloating(True)
    # Somewhere visible and not exactly on top of the main window.
    geom = window.geometry()
    panel.resize(max(420, widget.sizeHint().width() + 40), min(720, geom.height() - 80))
    panel.move(geom.right() - panel.width() - 40, geom.top() + 60)
    panel.show()
    panel.raise_()
    return panel
