"""
Keyboard shortcuts for the actions people repeat all day.

Annotating a folder of micrographs means the same handful of actions hundreds of
times — accept, next image, accept, next image — and every one of them was
mouse-only. That is the difference between software that works and software that
is pleasant to use for a full day.

The table below is the single source of truth: the shortcuts are installed from
it and the Help dialog is generated from it, so the documented keys cannot drift
from the ones that actually fire.

Every shortcut is suppressed while a text field has focus, so typing a filename
containing "d" does not advance to the next image.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QAbstractSpinBox, QApplication, QDialog, QDialogButtonBox, QLabel,
    QLineEdit, QPlainTextEdit, QTextEdit, QVBoxLayout, QWidget,
)


@dataclass(frozen=True)
class Shortcut:
    keys:   str          # a QKeySequence string, or several separated by " / "
    label:  str          # what it does, in the user's words
    group:  str          # heading in the Help dialog
    method: str          # MainWindow method to call
    typing_safe: bool = False   # True = fires even while a text field has focus


SHORTCUTS: tuple[Shortcut, ...] = (
    # Moving through a folder — the two most-pressed keys in the application.
    Shortcut("Ctrl+Right / PgDown", "Next image",      "Navigating", "_on_next"),
    Shortcut("Ctrl+Left / PgUp",    "Previous image",  "Navigating", "_on_prev"),

    # The annotation loop.
    Shortcut("Ctrl+Return",    "Accept all pending annotations", "Annotating", "accept_pending"),
    Shortcut("Ctrl+Backspace", "Reject all pending annotations", "Annotating", "reject_pending"),
    Shortcut("Ctrl+Space",     "Keep this mask and start the next", "Annotating",
             "_on_sam_commit_new"),
    Shortcut("Ctrl+Z",         "Undo the last annotation",       "Annotating", "_on_undo"),
    Shortcut("Delete",         "Delete the selected annotation",  "Annotating",
             "_on_delete_selected"),

    # Looking at the image.
    Shortcut("Ctrl+H", "Hide or show annotations", "Viewing", "toggle_annotations"),
    Shortcut("Ctrl+0", "Reset zoom and pan",       "Viewing", "reset_view"),

    # Getting around the application.
    Shortcut("F1", "This list of shortcuts", "Application", "show_shortcuts", typing_safe=True),
)

_TEXT_WIDGETS = (QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox)


def _typing() -> bool:
    """True while the focus is in something you type into."""
    widget = QApplication.focusWidget()
    return isinstance(widget, _TEXT_WIDGETS)


def install(window) -> list[QShortcut]:
    """Bind every shortcut in SHORTCUTS to *window*. Returns the QShortcut objects."""
    created: list[QShortcut] = []
    for spec in SHORTCUTS:
        for key in (k.strip() for k in spec.keys.split("/")):
            seq = QKeySequence(key)
            if seq.isEmpty():
                continue
            sc = QShortcut(seq, window)
            sc.setContext(Qt.ShortcutContext.WindowShortcut)
            sc.activated.connect(_dispatcher(window, spec))
            created.append(sc)
    return created


def _dispatcher(window, spec: Shortcut) -> Callable[[], None]:
    def run() -> None:
        if _typing() and not spec.typing_safe:
            return
        handler = getattr(window, spec.method, None)
        if handler is None:
            return
        try:
            handler()
        except Exception as exc:                      # never let a key press crash the window
            import logging
            logging.getLogger(__name__).warning(
                "shortcut %s (%s) failed: %s", spec.keys, spec.method, exc)
    return run


class ShortcutHelp(QDialog):
    """The shortcut list, generated from SHORTCUTS so it cannot go stale."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Keyboard shortcuts")
        self.setMinimumWidth(430)
        self.setStyleSheet("""
            QDialog { background: #141414; }
            QLabel  { color: #e0e0e0; }
            QLabel#group {
                color: #4dbb78; font-weight: 700; font-size: 12px;
                padding-top: 10px;
            }
            QLabel#keys {
                color: #ffffff; font-family: monospace; font-size: 11px;
                background: #252525; border: 1px solid #363636;
                border-radius: 3px; padding: 2px 6px;
            }
            QLabel#what { color: #c8c8c8; font-size: 12px; }
        """)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 14)
        outer.setSpacing(4)

        from PyQt6.QtWidgets import QGridLayout

        seen_groups: list[str] = []
        for spec in SHORTCUTS:
            if spec.group not in seen_groups:
                seen_groups.append(spec.group)
                heading = QLabel(spec.group)
                heading.setObjectName("group")
                outer.addWidget(heading)
                grid = QGridLayout()
                grid.setColumnMinimumWidth(0, 150)
                grid.setHorizontalSpacing(14)
                grid.setVerticalSpacing(3)
                outer.addLayout(grid)
                row = 0
            keys = QLabel(spec.keys.replace(" / ", "  or  "))
            keys.setObjectName("keys")
            what = QLabel(spec.label)
            what.setObjectName("what")
            grid.addWidget(keys, row, 0)
            grid.addWidget(what, row, 1)
            row += 1

        note = QLabel("Shortcuts pause while you are typing in a text field.")
        note.setStyleSheet("color:#8a8a8a;font-size:11px;padding-top:12px;")
        outer.addWidget(note)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
