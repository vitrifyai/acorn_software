"""What to show where a plugin panel should have been.

A plugin that fails is skipped rather than allowed to take the application down,
which is right. But skipping it silently means the panel simply is not there,
and an absent panel is indistinguishable from a feature that was removed. That
cost us real time once already: a stale signal connection made the SEM dock fail
to build, the whole test suite still passed, and the only symptom was a dock
that quietly did not appear.

The rule this module encodes is that not every failure deserves to be shouted
about:

    missing dependency   expected, stay quiet
    anything else        a bug, and it should be visible

Several plugins are *supposed* to be absent sometimes -- the 3-D viewer without
mrcfile, CLU without an LLM client, SAM without torch. Announcing those on every
launch would train people to dismiss the notice, and then the one that matters
gets dismissed with it.
"""
from __future__ import annotations

import traceback

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from acorn.plugin_loader import is_missing_dependency

__all__ = ["FailurePanel", "placeholder_for"]


class FailurePanel(QWidget):
    """Stands in for a panel that raised while being built.

    Deliberately not a dialog. A modal at startup that returns every launch is
    worse than silence -- it gets clicked away without being read. Sitting in
    the panel's own place means the message is found exactly where someone goes
    looking for the missing tool.
    """

    def __init__(self, title: str, exc: BaseException, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        heading = QLabel(f"{title} failed to load")
        heading.setWordWrap(True)
        heading.setStyleSheet("font-weight: 600; color: #c65f5f;")
        layout.addWidget(heading)

        reason = QLabel(f"{type(exc).__name__}: {exc}")
        reason.setWordWrap(True)
        reason.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        reason.setStyleSheet("font-family: monospace; font-size: 11px;")
        layout.addWidget(reason)

        hint = QLabel(
            "This is a fault in the plugin, not a missing optional package. "
            "The rest of ACORN is unaffected. The full traceback is in the log."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: #6c7086;")
        layout.addWidget(hint)
        layout.addStretch()

        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        # The traceback is long and mostly framework frames; a tooltip keeps it
        # reachable without turning the panel into a wall of text.
        self.setToolTip("".join(traceback.format_exception(
            type(exc), exc, exc.__traceback__)).strip())


def placeholder_for(title: str, exc: BaseException, parent=None) -> QWidget | None:
    """A stand-in panel for `exc`, or None if the failure is not worth reporting.

    Returning None for a missing dependency is the whole point: those are
    expected states, not faults, and surfacing them is noise.
    """
    if is_missing_dependency(exc):
        return None
    return FailurePanel(title, exc, parent)
