"""
ACORN Tracking Panel  (tracking_panel.py)
==========================================
GUI panel for particle tracking across image sequences (time series or
z-stacks).

The panel operates on the images and annotations already loaded in the main
window.  After tracking, a summary table is shown and results can be exported
as CSV.

Signals
-------
track_requested(dict)
    Emitted when the user clicks "Track".  The dict contains:
      max_displacement_nm (float)
      min_frames (int)
      max_gap (int)

export_requested(str)
    Emitted when the user clicks "Export CSV".  str is the chosen file path.
"""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


class TrackingPanel(QWidget):
    """Control panel for particle tracking across loaded image frames."""

    track_requested = pyqtSignal(dict)
    export_requested = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._tracks_df = None  # last result DataFrame
        self._n_frames = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ── status ────────────────────────────────────────────────────────────
        status_box = QGroupBox("Image sequence")
        s_lay = QVBoxLayout(status_box)
        self._status_label = QLabel("No images loaded.")
        self._status_label.setWordWrap(True)
        s_lay.addWidget(self._status_label)
        layout.addWidget(status_box)

        # ── parameters ────────────────────────────────────────────────────────
        param_box = QGroupBox("Tracking parameters")
        p_lay = QVBoxLayout(param_box)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Max displacement (nm):"))
        self._max_disp_spin = QDoubleSpinBox()
        self._max_disp_spin.setRange(1.0, 100000.0)
        self._max_disp_spin.setValue(500.0)
        self._max_disp_spin.setDecimals(1)
        self._max_disp_spin.setSingleStep(50.0)
        row1.addWidget(self._max_disp_spin)
        p_lay.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Min track length (frames):"))
        self._min_frames_spin = QSpinBox()
        self._min_frames_spin.setRange(1, 999)
        self._min_frames_spin.setValue(2)
        row2.addWidget(self._min_frames_spin)
        p_lay.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("Max gap (frames):"))
        self._max_gap_spin = QSpinBox()
        self._max_gap_spin.setRange(0, 10)
        self._max_gap_spin.setValue(1)
        row3.addWidget(self._max_gap_spin)
        p_lay.addLayout(row3)

        layout.addWidget(param_box)

        # ── actions ───────────────────────────────────────────────────────────
        self._track_btn = QPushButton("Track annotations across frames")
        self._track_btn.clicked.connect(self._on_track_clicked)
        layout.addWidget(self._track_btn)

        # ── results ───────────────────────────────────────────────────────────
        result_box = QGroupBox("Track summary")
        r_lay = QVBoxLayout(result_box)

        self._result_label = QLabel("No tracks computed yet.")
        self._result_label.setWordWrap(True)
        r_lay.addWidget(self._result_label)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels([
            "Track", "Frames", "First", "Last",
            "Total disp. (nm)", "Net disp. (nm)",
        ])
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMaximumHeight(220)
        r_lay.addWidget(self._table)

        layout.addWidget(result_box)

        # ── export ────────────────────────────────────────────────────────────
        self._export_btn = QPushButton("Export tracks as CSV")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export_clicked)
        layout.addWidget(self._export_btn)

        layout.addStretch()

    # ── public API ────────────────────────────────────────────────────────────

    def update_status(self, n_images: int, annotated: int) -> None:
        """Called by main_window when the image list changes."""
        self._n_frames = n_images
        self._status_label.setText(
            f"{n_images} image(s) loaded, "
            f"{annotated} with annotations."
        )
        self._track_btn.setEnabled(n_images >= 2)

    def set_tracks(self, df, stats_df) -> None:
        """
        Receive tracking results from main_window.

        df       -- full per-point DataFrame (stored for CSV export)
        stats_df -- per-track summary DataFrame shown in the table
        """
        self._tracks_df = df
        self._table.setRowCount(0)

        if df is None or df.empty:
            self._result_label.setText("No tracks found with current parameters.")
            self._export_btn.setEnabled(False)
            return

        n_tracks = len(stats_df)
        n_points = len(df)
        self._result_label.setText(
            f"{n_tracks} track(s), {n_points} total detection(s)."
        )

        self._table.setRowCount(len(stats_df))
        for row_idx, row in stats_df.iterrows():
            self._table.setItem(row_idx, 0, QTableWidgetItem(str(int(row["track_id"]))))
            self._table.setItem(row_idx, 1, QTableWidgetItem(str(int(row["n_frames"]))))
            self._table.setItem(row_idx, 2, QTableWidgetItem(str(int(row["first_frame"]))))
            self._table.setItem(row_idx, 3, QTableWidgetItem(str(int(row["last_frame"]))))
            self._table.setItem(row_idx, 4, QTableWidgetItem(f"{row['total_displacement_nm']:.1f}"))
            self._table.setItem(row_idx, 5, QTableWidgetItem(f"{row['net_displacement_nm']:.1f}"))

        self._export_btn.setEnabled(True)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_track_clicked(self) -> None:
        params = {
            "max_displacement_nm": self._max_disp_spin.value(),
            "min_frames": self._min_frames_spin.value(),
            "max_gap": self._max_gap_spin.value(),
        }
        self.track_requested.emit(params)

    def _on_export_clicked(self) -> None:
        if self._tracks_df is None or self._tracks_df.empty:
            QMessageBox.warning(self, "No data", "No tracks to export.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export tracks", "tracks.csv", "CSV files (*.csv)"
        )
        if path:
            self.export_requested.emit(path)
