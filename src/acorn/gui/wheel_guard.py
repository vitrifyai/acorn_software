"""
Make the mouse wheel scroll the panel, not whatever happens to be under the cursor.

Two Qt defaults get in the way inside a dense control panel:

* A combo box, spin box or slider under the pointer eats the wheel and *changes
  its own value*. Scrolling past the Contrast tab could silently alter the
  normalisation method or the gamma, and nothing says it happened.
* A scrollable widget nested inside a scrollable panel consumes the wheel as
  soon as the pointer crosses it, so the panel stops moving halfway down.

Both become opt-in here: the inner widget takes the wheel only once you have
clicked into it. Until then the wheel belongs to the panel you are scrolling.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import (
    QAbstractScrollArea, QAbstractSlider, QAbstractSpinBox, QApplication,
    QComboBox, QScrollArea, QWidget,
)

# Value editors: scrolling over one must never change the value by accident.
_VALUE_WIDGETS = (QComboBox, QAbstractSpinBox, QAbstractSlider)


class WheelGuard(QObject):
    """Event filter that keeps wheel events with the enclosing scroll area."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt name
        if event.type() != QEvent.Type.Wheel or not isinstance(obj, QWidget):
            return False

        if isinstance(obj, _VALUE_WIDGETS):
            if obj.hasFocus():
                return False          # deliberately chosen — let it take the wheel
            self._forward(obj, event)
            return True

        if isinstance(obj, QAbstractScrollArea) and not obj.hasFocus():
            # Let a nested list or table scroll itself only while it still can;
            # at either end the wheel goes back to the panel instead of sticking.
            bar = obj.verticalScrollBar()
            if bar is None or bar.minimum() == bar.maximum():
                self._forward(obj, event)
                return True
            going_up = event.angleDelta().y() > 0
            if (going_up and bar.value() == bar.minimum()) or (
                not going_up and bar.value() == bar.maximum()
            ):
                self._forward(obj, event)
                return True
        return False

    @staticmethod
    def _forward(widget: QWidget, event: QEvent) -> None:
        """Hand the wheel to the nearest enclosing scroll area."""
        parent = widget.parentWidget()
        while parent is not None:
            if isinstance(parent, QAbstractScrollArea) and parent is not widget:
                QApplication.sendEvent(parent.viewport(), event)
                return
            parent = parent.parentWidget()


def install(root: QWidget, guard: WheelGuard | None = None) -> WheelGuard:
    """
    Apply the guard to *root* and everything currently inside it.

    Value editors also get StrongFocus so clicking one is what hands it the
    wheel; they default to WheelFocus, which is the behaviour being prevented.
    """
    guard = guard or WheelGuard(root)
    targets = [root, *root.findChildren(QWidget)]
    for widget in targets:
        if isinstance(widget, _VALUE_WIDGETS):
            widget.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            widget.installEventFilter(guard)
        elif isinstance(widget, QAbstractScrollArea) and not isinstance(widget, QScrollArea):
            widget.installEventFilter(guard)
        elif isinstance(widget, QScrollArea) and widget is not root:
            widget.installEventFilter(guard)
    return guard
