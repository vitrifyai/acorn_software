"""Modal dialogs — line profile, about."""

from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout,
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
        self.resize(620, 340)
        self._result = result

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
        export_btn = QPushButton("Export CSV")
        export_btn.clicked.connect(self._export_csv)
        export_img_btn = QPushButton("Export PNG")
        export_img_btn.clicked.connect(self._export_png)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(export_btn)
        btn_row.addWidget(export_img_btn)
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

    def _plot(self, result) -> None:
        ax = self._ax
        ax.clear()
        ax.plot(result.distances_nm, result.intensities, color="#00AAFF", lw=1.2)
        ax.set_xlabel("Distance (nm)")
        ax.set_ylabel("Normalised intensity")
        ax.set_title("Line Profile")
        ax.set_xlim(0, result.length_nm)
        ax.grid(True, alpha=0.3)
        self._fig.tight_layout()
        self._canvas.draw()

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

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PNG", "line_profile.png", "PNG files (*.png)"
        )
        if not path:
            return
        self._fig.savefig(path, dpi=150, bbox_inches="tight")
