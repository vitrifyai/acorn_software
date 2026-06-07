"""Annotation tool palette widget."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QSpinBox, QVBoxLayout, QWidget,
)

TOOLS = [
    ("none",        "Select / Move"),
    ("arrow",       "Arrow"),
    ("line",        "Line"),
    ("circle",      "Circle"),
    ("rectangle",   "Rectangle"),
    ("freehand",    "Freehand"),
    ("text",        "Text"),
    ("scalebar",    "Scale Bar"),
    ("distance",    "Distance"),
    ("line_profile","Line Profile"),
    ("angle",       "Angle"),
    ("roi",         "Area / ROI"),
]


class AnnotationPanel(QWidget):
    """
    Tool palette for annotations and measurements.

    Emits ``tool_changed(str)`` when the active tool changes.
    """

    tool_changed              = pyqtSignal(str)
    undo_requested            = pyqtSignal()
    clear_requested           = pyqtSignal()
    clear_profiles_requested  = pyqtSignal()
    delete_selected_requested = pyqtSignal()
    relabel_requested         = pyqtSignal(str)   # new label for selected annotation

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── tool selector (dropdown) ───────────────────────────────────────────
        tool_form = QFormLayout()
        tool_form.setSpacing(4)
        self._tool_combo = QComboBox()
        for key, label in TOOLS:
            self._tool_combo.addItem(label, key)
        tool_form.addRow("Tool:", self._tool_combo)
        layout.addLayout(tool_form)

        # ── style ─────────────────────────────────────────────────────────────
        style_box = QGroupBox("Style")
        style_layout = QFormLayout(style_box)

        self._color_btn = QPushButton("  Colour  ")
        self._color = QColor("#4dbb78")
        self._update_color_btn()
        self._color_btn.clicked.connect(self._pick_color)
        style_layout.addRow("Colour:", self._color_btn)

        self._lw = QDoubleSpinBox()
        self._lw.setRange(0.5, 8.0)
        self._lw.setValue(2.0)
        self._lw.setSingleStep(0.5)
        style_layout.addRow("Line width:", self._lw)

        self._linestyle = QComboBox()
        self._linestyle.addItem("Solid", "-")
        self._linestyle.addItem("Dashed", "--")
        self._linestyle.addItem("Dotted", ":")
        style_layout.addRow("Line style:", self._linestyle)

        self._fs = QSpinBox()
        self._fs.setRange(6, 48)
        self._fs.setValue(12)
        style_layout.addRow("Font size:", self._fs)

        layout.addWidget(style_box)

        # ── context fields ────────────────────────────────────────────────────
        ctx_box = QGroupBox("Context")
        ctx_layout = QFormLayout(ctx_box)

        self._text_val = QLineEdit("Label")
        ctx_layout.addRow("Text:", self._text_val)

        self._sb_nm = QDoubleSpinBox()
        self._sb_nm.setRange(0.1, 1e6)
        self._sb_nm.setValue(100.0)
        self._sb_nm.setSuffix(" nm")
        ctx_layout.addRow("Scale bar:", self._sb_nm)

        layout.addWidget(ctx_box)

        # ── region label (for ROI tool) ───────────────────────────────────────
        self._roi_label_box = QGroupBox("Region Label")
        roi_label_layout = QVBoxLayout(self._roi_label_box)

        self._roi_label = QLineEdit("")
        self._roi_label.setPlaceholderText("Custom label…")
        roi_label_layout.addWidget(self._roi_label)

        preset_row = QHBoxLayout()
        for preset, color, fg in [("Foreground", "#00703C", "white"), ("Background", "#c0392b", "white"), ("Ignore", "#555555", "white")]:
            btn = QPushButton(preset)
            btn.setStyleSheet(f"background:{color};color:{fg};font-size:10px;")
            btn.clicked.connect(lambda _, p=preset: self._roi_label.setText(p))
            preset_row.addWidget(btn)
        roi_label_layout.addLayout(preset_row)

        layout.addWidget(self._roi_label_box)

        # ── tool hint ─────────────────────────────────────────────────────────
        self._hint = QLabel("")
        self._hint.setWordWrap(True)
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self._hint)

        # ── selected annotation ───────────────────────────────────────────────
        self._selected_box = QGroupBox("Selected Annotation")
        sel_layout = QVBoxLayout(self._selected_box)
        sel_layout.setSpacing(4)

        self._sel_type_label = QLabel("None selected")
        self._sel_type_label.setStyleSheet("font-size: 11px; color: #888888;")
        sel_layout.addWidget(self._sel_type_label)

        label_row = QHBoxLayout()
        self._sel_label_edit = QLineEdit()
        self._sel_label_edit.setPlaceholderText("annotation label…")
        self._sel_label_edit.setEnabled(False)
        rename_btn = QPushButton("Rename")
        rename_btn.setFixedWidth(60)
        rename_btn.clicked.connect(self._on_rename)
        self._sel_label_edit.returnPressed.connect(self._on_rename)
        label_row.addWidget(self._sel_label_edit, 1)
        label_row.addWidget(rename_btn)
        sel_layout.addLayout(label_row)

        del_sel_btn = QPushButton("Delete Selected")
        del_sel_btn.setStyleSheet("background:#c0392b;color:white;")
        del_sel_btn.clicked.connect(self.delete_selected_requested)
        sel_layout.addWidget(del_sel_btn)

        layout.addWidget(self._selected_box)

        # ── actions ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        undo_btn = QPushButton("Undo Last")
        undo_btn.setStyleSheet("background:#c0392b;color:white;")
        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet("background:#c0392b;color:white;")
        btn_row.addWidget(undo_btn)
        btn_row.addWidget(clear_btn)
        layout.addLayout(btn_row)

        clear_prof_btn = QPushButton("Clear Profiles")
        clear_prof_btn.setStyleSheet("background:#1a5fa8;color:white;")
        clear_prof_btn.setToolTip("Remove all line profile overlays from the canvas")
        layout.addWidget(clear_prof_btn)
        layout.addStretch()

        # ── connect ───────────────────────────────────────────────────────────
        self._tool_combo.currentIndexChanged.connect(
            lambda _: self.tool_changed.emit(self._tool_combo.currentData())
        )
        undo_btn.clicked.connect(self.undo_requested)
        clear_btn.clicked.connect(self.clear_requested)
        clear_prof_btn.clicked.connect(self.clear_profiles_requested)

    # ── helpers ───────────────────────────────────────────────────────────────

    def set_selected_annotation(self, ann) -> None:
        """Update the Selected Annotation section when user clicks an annotation."""
        if ann is None:
            self._sel_type_label.setText("None selected")
            self._sel_label_edit.setText("")
            self._sel_label_edit.setEnabled(False)
            return
        t = ann.type
        label = getattr(ann, "label", None)
        self._sel_type_label.setText(f"Type: {t}")
        if label is not None:
            self._sel_label_edit.setText(label)
            self._sel_label_edit.setEnabled(True)
        else:
            self._sel_label_edit.setText("")
            self._sel_label_edit.setEnabled(False)

    def _on_rename(self) -> None:
        new_label = self._sel_label_edit.text().strip()
        if new_label:
            self.relabel_requested.emit(new_label)

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(self._color, self, "Pick annotation colour")
        if c.isValid():
            self._color = c
            self._update_color_btn()

    def _update_color_btn(self) -> None:
        hex_col = self._color.name()
        self._color_btn.setStyleSheet(
            f"background:{hex_col};color:{'black' if self._color.lightness()>128 else 'white'};"
        )
        self._color_btn.setText(hex_col.upper())

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def active_tool(self) -> str:
        return self._tool_combo.currentData() or "none"

    @property
    def color(self) -> str:
        return self._color.name()

    @property
    def linewidth(self) -> float:
        return self._lw.value()

    @property
    def linestyle(self) -> str:
        return self._linestyle.currentData()

    @property
    def fontsize(self) -> int:
        return self._fs.value()

    @property
    def text_value(self) -> str:
        return self._text_val.text()

    @property
    def scalebar_nm(self) -> float:
        return self._sb_nm.value()

    @property
    def roi_label(self) -> str:
        return self._roi_label.text()

    def set_scalebar_nm(self, nm: float) -> None:
        self._sb_nm.setValue(nm)

    def set_hint(self, text: str) -> None:
        self._hint.setText(text)
