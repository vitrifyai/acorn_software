from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)


class SegmentationPanel(QWidget):
    """
    Unified AI segmentation panel — SAM / YOLO / UNet selector with
    loaded indicators, per-tool controls, and a shared Accept/Reject footer.

    The three tool panels are passed in from MainWindow so all existing
    signal connections remain unchanged.
    """

    accept_all_requested = pyqtSignal(str)   # tool name: "sam" | "yolo" | "unet"
    reject_all_requested = pyqtSignal(str)

    _TOOLS = [("sam", "SAM"), ("yolo", "YOLO"), ("unet", "UNet")]

    def __init__(self, sam_panel, yolo_panel, unet_panel, parent=None):
        super().__init__(parent)
        self._panels = {"sam": sam_panel, "yolo": yolo_panel, "unet": unet_panel}
        self._active = "sam"
        self._loaded = {"sam": False, "yolo": False, "unet": False}
        self._sel_btns: dict[str, QPushButton] = {}
        self._ind_labels: dict[str, QLabel] = {}
        self._build_ui(sam_panel, yolo_panel, unet_panel)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self, sam_panel, yolo_panel, unet_panel) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # ── tool selector row ──────────────────────────────────────────
        sel_row = QHBoxLayout()
        sel_row.setSpacing(4)

        for key, label in self._TOOLS:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setFixedHeight(28)
            btn.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            btn.clicked.connect(lambda _, k=key: self._select(k))
            self._sel_btns[key] = btn

            ind = QLabel("●")
            ind.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            ind.setStyleSheet("font-size:8px; color: palette(mid);")
            ind.setToolTip(f"{label} model not loaded")
            self._ind_labels[key] = ind

            col.addWidget(btn)
            col.addWidget(ind)
            sel_row.addLayout(col)

        outer.addLayout(sel_row)

        # ── stacked tool area ──────────────────────────────────────────
        self._stack = QStackedWidget()
        for key, panel in (("sam", sam_panel), ("yolo", yolo_panel), ("unet", unet_panel)):
            panel.hide_footer()
            self._stack.addWidget(panel)
        outer.addWidget(self._stack, 1)

        # ── shared footer ──────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size:11px; color: palette(mid);")
        outer.addWidget(self._status)

        ar_row = QHBoxLayout()
        accept_btn = QPushButton("Accept All")
        accept_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        accept_btn.clicked.connect(self._on_accept)
        reject_btn = QPushButton("Reject All")
        reject_btn.setStyleSheet("background:#c0392b;color:white;")
        reject_btn.clicked.connect(self._on_reject)
        ar_row.addWidget(accept_btn)
        ar_row.addWidget(reject_btn)
        outer.addLayout(ar_row)

        self._select("sam")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_loaded(self, tool: str, loaded: bool) -> None:
        """Update the loaded indicator dot for the given tool."""
        if tool not in self._loaded:
            return
        self._loaded[tool] = loaded
        ind = self._ind_labels[tool]
        label = dict(self._TOOLS)[tool]
        if loaded:
            ind.setStyleSheet("font-size:8px; color: #4dbb78;")
            ind.setToolTip(f"{label} model loaded")
        else:
            ind.setStyleSheet("font-size:8px; color: palette(mid);")
            ind.setToolTip(f"{label} model not loaded")

    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    def active_tool(self) -> str:
        return self._active

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select(self, tool: str) -> None:
        self._active = tool
        for key, btn in self._sel_btns.items():
            btn.setChecked(key == tool)
            btn.setStyleSheet(
                "background:#1a5fa8;color:white;font-weight:bold;"
                if key == tool else ""
            )
        self._stack.setCurrentIndex(["sam", "yolo", "unet"].index(tool))

    def _on_accept(self) -> None:
        self.accept_all_requested.emit(self._active)

    def _on_reject(self) -> None:
        self.reject_all_requested.emit(self._active)
