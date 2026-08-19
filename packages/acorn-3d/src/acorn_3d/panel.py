"""Volume navigation panel for the acorn_3d plugin."""
from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from acorn_3d.volume_loader import VolumeImage


class VolumePanel(QWidget):
    """
    Z-stack navigation panel.

    Signals
    -------
    slice_changed(int)           -- user changed the z slice
    projection_requested(str, int, int)  -- method, z_from, z_to
    """

    slice_changed          = pyqtSignal(int)
    projection_requested   = pyqtSignal(str, int, int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._n_slices = 0
        self._updating = False  # guard against slider/spinbox feedback loops

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── volume info ───────────────────────────────────────────────────────
        info_box = QGroupBox("Volume Info")
        info_lay = QVBoxLayout(info_box)
        self._info_label = QLabel("No volume loaded.\nUse File > Open as Volume...")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet("font-size: 11px; color: #6c7086;")
        info_lay.addWidget(self._info_label)
        layout.addWidget(info_box)

        # ── z navigation ──────────────────────────────────────────────────────
        nav_box = QGroupBox("Z Navigation")
        nav_lay = QVBoxLayout(nav_box)
        nav_lay.setSpacing(6)

        spin_row = QHBoxLayout()
        spin_row.addWidget(QLabel("Slice:"))
        self._z_spin = QSpinBox()
        self._z_spin.setRange(0, 0)
        self._z_spin.setEnabled(False)
        spin_row.addWidget(self._z_spin)
        self._of_label = QLabel("of 0")
        spin_row.addWidget(self._of_label)
        spin_row.addStretch()
        nav_lay.addLayout(spin_row)

        self._z_slider = QSlider(Qt.Orientation.Horizontal)
        self._z_slider.setRange(0, 0)
        self._z_slider.setEnabled(False)
        nav_lay.addWidget(self._z_slider)

        btn_row = QHBoxLayout()
        self._prev_btn = QPushButton("Prev")
        self._prev_btn.setEnabled(False)
        self._next_btn = QPushButton("Next")
        self._next_btn.setEnabled(False)
        btn_row.addWidget(self._prev_btn)
        btn_row.addWidget(self._next_btn)
        nav_lay.addLayout(btn_row)
        layout.addWidget(nav_box)

        # ── projection ────────────────────────────────────────────────────────
        proj_box = QGroupBox("Projection")
        proj_lay = QVBoxLayout(proj_box)
        proj_lay.setSpacing(6)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self._proj_method = QComboBox()
        self._proj_method.addItem("Max intensity", "max")
        self._proj_method.addItem("Mean", "mean")
        self._proj_method.addItem("Min intensity", "min")
        method_row.addWidget(self._proj_method)
        proj_lay.addLayout(method_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("Z range:"))
        self._proj_from = QSpinBox()
        self._proj_from.setRange(0, 0)
        range_row.addWidget(self._proj_from)
        range_row.addWidget(QLabel("to"))
        self._proj_to = QSpinBox()
        self._proj_to.setRange(0, 0)
        range_row.addWidget(self._proj_to)
        proj_lay.addLayout(range_row)

        self._proj_btn = QPushButton("Run Projection")
        self._proj_btn.setEnabled(False)
        self._proj_btn.clicked.connect(self._on_projection_clicked)
        proj_lay.addWidget(self._proj_btn)
        layout.addWidget(proj_box)

        layout.addStretch()

        # ── wire signals ──────────────────────────────────────────────────────
        self._z_slider.valueChanged.connect(self._on_slider_changed)
        self._z_spin.valueChanged.connect(self._on_spin_changed)
        self._prev_btn.clicked.connect(lambda: self._step(-1))
        self._next_btn.clicked.connect(lambda: self._step(1))

        # ── scroll wrapper ────────────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    # ── public API ────────────────────────────────────────────────────────────

    def set_volume(self, vol: "VolumeImage") -> None:
        n = vol.n_slices
        self._n_slices = n
        px = vol.pixel_size
        depth = vol.meta.voxel_depth_nm if hasattr(vol.meta, "voxel_depth_nm") else px
        h, w = vol.shape
        self._info_label.setText(
            f"File: {vol.filename}\n"
            f"Dimensions (Z x H x W): {n} x {h} x {w}\n"
            f"Pixel size: {px:.4f} nm/px\n"
            f"Voxel depth: {depth:.4f} nm"
        )
        self._updating = True
        self._z_slider.setRange(0, max(0, n - 1))
        self._z_slider.setValue(0)
        self._z_spin.setRange(0, max(0, n - 1))
        self._z_spin.setValue(0)
        self._of_label.setText(f"of {n - 1}")
        self._proj_from.setRange(0, max(0, n - 1))
        self._proj_from.setValue(0)
        self._proj_to.setRange(0, max(0, n - 1))
        self._proj_to.setValue(max(0, n - 1))
        enabled = n > 1
        self._z_slider.setEnabled(enabled)
        self._z_spin.setEnabled(enabled)
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)
        self._proj_btn.setEnabled(True)
        self._updating = False

    @property
    def current_z(self) -> int:
        return self._z_slider.value()

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_slider_changed(self, z: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._z_spin.setValue(z)
        self._updating = False
        self.slice_changed.emit(z)

    def _on_spin_changed(self, z: int) -> None:
        if self._updating:
            return
        self._updating = True
        self._z_slider.setValue(z)
        self._updating = False
        self.slice_changed.emit(z)

    def _step(self, delta: int) -> None:
        self._z_slider.setValue(self._z_slider.value() + delta)

    def _on_projection_clicked(self) -> None:
        method = self._proj_method.currentData()
        z_from = self._proj_from.value()
        z_to   = self._proj_to.value() + 1
        self.projection_requested.emit(method, z_from, z_to)
