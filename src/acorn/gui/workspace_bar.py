"""
Workspace switcher and first-launch welcome screen.

The switcher is a single exclusive row of buttons across the top of the window.
Picking one is the only decision a new user has to make before ACORN shows them
a window that fits what they came to do.
"""
from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
    QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from acorn.gui.workspaces import WORKSPACES, Workspace

# Matches the main-window stylesheet: ORNL green on a near-black ground.
_GREEN       = "#00703C"
_GREEN_LIGHT = "#4dbb78"
_GROUND      = "#141414"
_SURFACE     = "#252525"
_BORDER      = "#363636"
_TEXT        = "#e0e0e0"
_TEXT_DIM    = "#8a8a8a"

_BAR_QSS = f"""
QWidget#workspaceBar {{
    background: {_GROUND};
    border-bottom: 1px solid {_BORDER};
}}
QLabel#workspaceBarTitle {{
    color: {_TEXT_DIM};
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1.2px;
    padding-right: 4px;
}}
QPushButton#workspaceButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 4px;
    color: {_TEXT_DIM};
    font-size: 12px;
    font-weight: 600;
    padding: 5px 14px;
}}
QPushButton#workspaceButton:hover {{
    color: {_TEXT};
    background: {_SURFACE};
}}
QPushButton#workspaceButton:checked {{
    background: {_GREEN};
    border-color: {_GREEN};
    color: #ffffff;
}}
QPushButton#workspaceButton:focus {{
    border-color: {_GREEN_LIGHT};
}}
"""


class WorkspaceBar(QWidget):
    """Exclusive row of workspace buttons. Emits the workspace id when one is picked."""

    workspace_selected = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("workspaceBar")
        self.setStyleSheet(_BAR_QSS)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 5, 10, 5)
        row.setSpacing(4)

        title = QLabel("WORKSPACE")
        title.setObjectName("workspaceBarTitle")
        row.addWidget(title)

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons: dict[str, QPushButton] = {}

        for ws in WORKSPACES:
            btn = QPushButton(ws.label)
            btn.setObjectName("workspaceButton")
            btn.setCheckable(True)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            tip = f"{ws.tagline} — {ws.blurb}"
            if ws.shortcut:
                tip += f"  ({ws.shortcut})"
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _checked=False, w=ws.wid: self.workspace_selected.emit(w))
            self._group.addButton(btn)
            row.addWidget(btn)
            self._buttons[ws.wid] = btn

            if ws.shortcut and parent is not None:
                sc = QShortcut(QKeySequence(ws.shortcut), parent)
                sc.activated.connect(lambda w=ws.wid: self.workspace_selected.emit(w))

        row.addStretch(1)

        self._hint = QLabel("")
        self._hint.setObjectName("workspaceBarTitle")
        row.addWidget(self._hint)

    def set_active(self, wid: str) -> None:
        """Check the button for `wid` without re-emitting workspace_selected."""
        btn = self._buttons.get(wid)
        if btn is not None and not btn.isChecked():
            btn.setChecked(True)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)

    def set_all_shown(self, showing_all: bool) -> None:
        """Reflect 'every panel pinned open' — no workspace button is authoritative then."""
        if showing_all:
            self._group.setExclusive(False)
            for b in self._buttons.values():
                b.setChecked(False)
            self._group.setExclusive(True)
            self.set_hint("Showing every panel")
        else:
            self.set_hint("")


class WelcomeDialog(QDialog):
    """
    First-launch screen. Names the five jobs so nothing important stays hidden
    behind a menu the way it used to.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Welcome to ACORN")
        self.setModal(True)
        self.setMinimumWidth(520)
        self._chosen: str = WORKSPACES[0].wid

        self.setStyleSheet(f"""
            QDialog {{ background: {_GROUND}; }}
            QLabel {{ color: {_TEXT}; }}
            QLabel#welcomeTitle  {{ font-size: 20px; font-weight: 700; }}
            QLabel#welcomeLede   {{ color: {_TEXT_DIM}; font-size: 12px; }}
            QLabel#wsName        {{ font-size: 13px; font-weight: 700; color: {_GREEN_LIGHT}; }}
            QLabel#wsBlurb       {{ color: {_TEXT_DIM}; font-size: 11px; }}
            QFrame#wsCard {{
                background: {_SURFACE};
                border: 1px solid {_BORDER};
                border-radius: 5px;
            }}
            QFrame#wsCard:hover {{ border-color: {_GREEN_LIGHT}; }}
            QPushButton#wsPick {{
                background: {_GREEN}; color: #ffffff; border: none;
                border-radius: 4px; padding: 6px 16px; font-weight: 600;
            }}
            QPushButton#wsPick:hover {{ background: {_GREEN_LIGHT}; }}
            QCheckBox {{ color: {_TEXT_DIM}; font-size: 11px; }}
        """)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 20, 22, 16)
        outer.setSpacing(14)

        title = QLabel("What are you here to do?")
        title.setObjectName("welcomeTitle")
        outer.addWidget(title)

        lede = QLabel(
            "ACORN opens in one workspace at a time so you only see the tools for the "
            "job in front of you. You can switch at any time from the bar at the top of "
            "the window, and your image and annotations stay loaded."
        )
        lede.setObjectName("welcomeLede")
        lede.setWordWrap(True)
        outer.addWidget(lede)

        for ws in WORKSPACES:
            outer.addWidget(self._make_card(ws))

        self._want_clu = QCheckBox("Open CLU, the assistant, alongside my workspace")
        self._want_clu.setToolTip(
            "CLU can drive any part of ACORN from a chat panel: run a detector, "
            "queue images, start training, generate a simulation. It needs a model "
            "configured under Provider Settings."
        )
        outer.addWidget(self._want_clu)

        self._show_at_startup = QCheckBox("Show this when ACORN starts")
        self._show_at_startup.setChecked(True)
        outer.addWidget(self._show_at_startup)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)

    def _make_card(self, ws: Workspace) -> QFrame:
        card = QFrame()
        card.setObjectName("wsCard")
        lay = QHBoxLayout(card)
        lay.setContentsMargins(14, 10, 12, 10)
        lay.setSpacing(12)

        text = QVBoxLayout()
        text.setSpacing(2)
        name = QLabel(f"{ws.label} — {ws.tagline}")
        name.setObjectName("wsName")
        blurb = QLabel(ws.blurb)
        blurb.setObjectName("wsBlurb")
        blurb.setWordWrap(True)
        text.addWidget(name)
        text.addWidget(blurb)
        lay.addLayout(text, 1)

        pick = QPushButton("Start here" if ws is WORKSPACES[0] else "Open")
        pick.setObjectName("wsPick")
        pick.setCursor(Qt.CursorShape.PointingHandCursor)
        pick.clicked.connect(lambda _c=False, w=ws.wid: self._choose(w))
        lay.addWidget(pick, 0, Qt.AlignmentFlag.AlignVCenter)
        return card

    def _choose(self, wid: str) -> None:
        self._chosen = wid
        self.accept()

    @property
    def chosen_workspace(self) -> str:
        return self._chosen

    @property
    def show_at_startup(self) -> bool:
        """Whether to show this screen again next launch."""
        return self._show_at_startup.isChecked()

    @property
    def wants_assistant(self) -> bool:
        """Whether to open the CLU dock alongside the chosen workspace."""
        return self._want_clu.isChecked()


