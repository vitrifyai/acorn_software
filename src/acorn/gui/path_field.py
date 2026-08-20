"""
Path inputs that show the useful end of a long path.

A QLineEdit holding a long path scrolls to its start, so a field showing
"/tmp/claude-26352/-home-vnw/8519d4f2-.../scratchpad/simout/images" displays the
opening characters of a temp directory and hides the part that identifies it. In
the Export panel it read as "f-a7fb-d25f750f916d/scratchpad/simout/images" —
which looks like a bug rather than a path.

`attach` keeps the widget's real text intact (callers still read .text() to get
the full path) and only changes what is shown when the field is not being edited:
the tail, which is the part that tells you where things are going, plus the full
path on hover.
"""
from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import QLineEdit


class _TailView(QObject):
    """Shows the end of the path while idle; the whole thing while focused."""

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt name
        if isinstance(obj, QLineEdit):
            if event.type() == QEvent.Type.FocusOut:
                _show_tail(obj)
            elif event.type() == QEvent.Type.FocusIn:
                obj.setCursorPosition(len(obj.text()))
        return False


def _show_tail(field: QLineEdit) -> None:
    """Scroll to the end and put the full path in the tooltip."""
    text = field.text()
    field.setToolTip(text or "")
    if text:
        field.setCursorPosition(len(text))
        field.deselect()


def attach(field: QLineEdit) -> QLineEdit:
    """Make *field* show the end of whatever path it holds. Returns the field."""
    view = _TailView(field)
    field.installEventFilter(view)
    field._tail_view = view          # keep the filter alive with the widget
    field.textChanged.connect(lambda _t, f=field: _show_tail(f))
    _show_tail(field)
    return field
