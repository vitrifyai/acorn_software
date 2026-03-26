"""Modal dialogs — line profile, about."""

from __future__ import annotations

import numpy as np
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
    QLabel, QPushButton, QVBoxLayout,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvasQtAgg
from matplotlib.figure import Figure


class LineProfileDialog(QDialog):
    """
    Shows an intensity profile plot in a popup window.

    Parameters
    ----------
    result : LineProfileResult from MeasurementEngine.line_profile()
    parent : parent widget
    """

    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Line Profile")
        self.resize(680, 380)
        self._result = result
        self._line_color = "#00AAFF"

        layout = QVBoxLayout(self)

        # info label
        lbl = QLabel(
            f"Length: {result.length_nm:.1f} nm  ·  "
            f"Points: {len(result.intensities)}  ·  "
            f"Pixel size: {result.pixel_size:.4f} nm/px"
        )
        layout.addWidget(lbl)

        # matplotlib canvas
        self._fig = Figure(figsize=(6, 2.8), facecolor="none")
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasQtAgg(self._fig)
        layout.addWidget(self._canvas)

        self._plot(result)

        # buttons
        btn_row = QHBoxLayout()
        color_btn = QPushButton("Line color")
        color_btn.clicked.connect(self._pick_color)
        self._color_swatch = QPushButton()
        self._color_swatch.setFixedWidth(28)
        self._color_swatch.setStyleSheet(
            f"background-color: {self._line_color}; border: 1px solid #888;"
        )
        self._color_swatch.clicked.connect(self._pick_color)
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        export_img_btn = QPushButton("Export PNG")
        export_img_btn.clicked.connect(self._export_png)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._color_swatch)
        btn_row.addWidget(color_btn)
        btn_row.addSpacing(12)
        btn_row.addWidget(export_btn)
        btn_row.addWidget(export_img_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _plot(self, result) -> None:
        ax = self._ax
        ax.clear()
        ax.plot(result.distances_nm, result.intensities, color=self._line_color, lw=1.2)
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title("Line Profile")
        ax.set_xlim(0, result.length_nm)
        ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()

    def _pick_color(self) -> None:
        qcol = QColorDialog.getColor(
            QColor(self._line_color), self, "Line color"
        )
        if qcol.isValid():
            self._line_color = qcol.name()
            self._color_swatch.setStyleSheet(
                f"background-color: {self._line_color}; border: 1px solid #888;"
            )
            self._plot(self._result)

    def _export_csv(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CSV", "line_profile.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        with open(path, "w") as f:
            f.write("distance_nm,intensity\n")
            for d, i in zip(self._result.distances_nm, self._result.intensities):
                f.write(f"{d:.6f},{i:.8f}\n")

    def update(self, result) -> None:
        """Refresh the plot with new profile data (live drag update)."""
        self._result = result
        self._plot(result)

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", "line_profile.png", "PNG files (*.png)"
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
