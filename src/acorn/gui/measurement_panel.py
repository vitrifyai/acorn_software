"""Live measurement results panel."""

from __future__ import annotations

import csv
import io

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class MeasurementPanel(QWidget):
    """
    Displays accumulated measurement results in a table.

    Columns: Type | Value | Units | Detail
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        layout.addWidget(QLabel("<b>Measurement Results</b>"))

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Type", "Value", "Units", "Detail"])
        self._table.horizontalHeader().setSectionResizeMode(
            3, QHeaderView.ResizeMode.Stretch
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        copy_btn = QPushButton("Copy CSV")
        copy_btn.clicked.connect(self._copy_csv)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear)
        btn_row.addWidget(copy_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ── public API ────────────────────────────────────────────────────────────

    def add_distance(self, result) -> None:
        """Add a DistanceMeasurement row."""
        nm = result.distance_nm
        if nm < 1000:
            val, unit = f"{nm:.2f}", "nm"
        else:
            val, unit = f"{nm/1000:.3f}", "µm"
        p1, p2 = result.p1, result.p2
        detail = f"({p1[0]:.0f},{p1[1]:.0f}) → ({p2[0]:.0f},{p2[1]:.0f})"
        self._add_row("Distance", val, unit, detail)

    def add_angle(self, result) -> None:
        """Add an AngleMeasurement row."""
        self._add_row(
            "Angle", f"{result.angle_deg:.2f}", "°",
            f"vertex ({result.vertex[0]:.0f},{result.vertex[1]:.0f})"
        )

    def add_roi(self, result) -> None:
        """Add an ROI stats row."""
        area = result.area_nm2
        if area < 1e6:
            area_str, area_unit = f"{area:.0f}", "nm²"
        else:
            area_str, area_unit = f"{area/1e6:.3f}", "µm²"
        s = result.stats
        detail = (
            f"mean={s.get('mean',0):.4f}  std={s.get('std',0):.4f}  "
            f"min={s.get('min',0):.4f}  max={s.get('max',0):.4f}  "
            f"n={s.get('n_pixels',0)}"
        )
        self._add_row("Area/ROI", area_str, area_unit, detail)

    def clear(self) -> None:
        self._table.setRowCount(0)

    # ── internal ──────────────────────────────────────────────────────────────

    def _add_row(self, mtype: str, value: str, unit: str, detail: str) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        for col, text in enumerate([mtype, value, unit, detail]):
            item = QTableWidgetItem(text)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._table.setItem(row, col, item)
        self._table.scrollToBottom()

    def _copy_csv(self) -> None:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["Type", "Value", "Units", "Detail"])
        for r in range(self._table.rowCount()):
            row = []
            for c in range(self._table.columnCount()):
                item = self._table.item(r, c)
                row.append(item.text() if item else "")
            writer.writerow(row)
        QApplication.clipboard().setText(buf.getvalue())
