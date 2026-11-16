"""Volume navigation + tomogram segmentation panel for the acorn_3d plugin."""
from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
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
    Z-stack navigation plus the tomogram annotation workflow.

    Signals
    -------
    slice_changed(int)                     -- user changed the z slice
    projection_requested(str, int, int)    -- method, z_from, z_to (exclusive)
    segment_slice_requested(str, str, float)
        -- text prompt, class label, confidence: SAM 3 text prompt on the current slice
    propagate_requested(str, str, int, int)
        -- mode ("text" | "track"), class label, z_from, z_to (inclusive)
    clear_class_requested(str)             -- remove every ROI of this class on all slices
    export_requested()                     -- write the label volume (MRC)
    render_requested(str, float, float)    -- style, seconds, z exaggeration
    """

    slice_changed = pyqtSignal(int)
    view_changed = pyqtSignal(int, float)          # slab thickness (slices), smoothing sigma (px)
    projection_requested = pyqtSignal(str, int, int)
    segment_slice_requested = pyqtSignal(str, str, float)
    propagate_requested = pyqtSignal(str, str, int, int)
    clear_class_requested = pyqtSignal(str)
    export_requested = pyqtSignal()
    render_requested = pyqtSignal(str, float, float)

    DEFAULT_CLASSES = ("CNC", "nanoparticle", "cell", "membrane")

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
        view_row = QHBoxLayout()
        view_row.addWidget(QLabel("Slab:"))
        self._slab_spin = QSpinBox()
        self._slab_spin.setRange(1, 99)
        self._slab_spin.setSingleStep(2)
        self._slab_spin.setValue(1)
        self._slab_spin.setSuffix(" slices")
        self._slab_spin.setToolTip("Average this many slices around z. 9 to 15 makes a noisy tomogram far easier to read.")
        view_row.addWidget(self._slab_spin)
        view_row.addWidget(QLabel("Smooth:"))
        self._smooth_spin = QDoubleSpinBox()
        self._smooth_spin.setRange(0.0, 5.0)
        self._smooth_spin.setSingleStep(0.5)
        self._smooth_spin.setValue(0.0)
        self._smooth_spin.setSuffix(" px")
        view_row.addWidget(self._smooth_spin)
        view_row.addStretch()
        nav_lay.addLayout(view_row)
        self._ann_label = QLabel("")
        self._ann_label.setStyleSheet("font-size: 11px; color: #6c7086;")
        nav_lay.addWidget(self._ann_label)
        layout.addWidget(nav_box)

        # ── segmentation ──────────────────────────────────────────────────────
        seg_box = QGroupBox("Tomogram Segmentation (SAM 3)")
        seg_lay = QVBoxLayout(seg_box)
        seg_lay.setSpacing(6)

        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel("Class:"))
        self._class_combo = QComboBox()
        self._class_combo.setEditable(True)
        for c in self.DEFAULT_CLASSES:
            self._class_combo.addItem(c)
        cls_row.addWidget(self._class_combo, 1)
        seg_lay.addLayout(cls_row)

        prompt_row = QHBoxLayout()
        prompt_row.addWidget(QLabel("Text prompt:"))
        self._prompt_edit = QLineEdit()
        self._prompt_edit.setPlaceholderText("e.g. rod, fiber, sphere, dark particle")
        prompt_row.addWidget(self._prompt_edit, 1)
        seg_lay.addLayout(prompt_row)

        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel("Confidence:"))
        self._conf_spin = QDoubleSpinBox()
        self._conf_spin.setRange(0.05, 0.95)
        self._conf_spin.setSingleStep(0.05)
        self._conf_spin.setValue(0.4)
        conf_row.addWidget(self._conf_spin)
        conf_row.addStretch()
        seg_lay.addLayout(conf_row)

        self._segment_btn = QPushButton("Segment this slice (text prompt)")
        self._segment_btn.setEnabled(False)
        self._segment_btn.setToolTip("Run the SAM 3 text prompt on the current slice and add the masks "
                                     "as ROIs of the chosen class. Edit or delete them before propagating.")
        seg_lay.addWidget(self._segment_btn)

        rng_row = QHBoxLayout()
        rng_row.addWidget(QLabel("Propagate z:"))
        self._prop_from = QSpinBox()
        self._prop_from.setRange(0, 0)
        rng_row.addWidget(self._prop_from)
        rng_row.addWidget(QLabel("to"))
        self._prop_to = QSpinBox()
        self._prop_to.setRange(0, 0)
        rng_row.addWidget(self._prop_to)
        rng_row.addStretch()
        seg_lay.addLayout(rng_row)

        self._prop_text_btn = QPushButton("Propagate text prompt through Z")
        self._prop_text_btn.setEnabled(False)
        self._prop_text_btn.setToolTip("Find every instance of the text prompt on the current slice and "
                                       "track them up and down the stack (one SAM 3 video pass).")
        seg_lay.addWidget(self._prop_text_btn)

        self._prop_track_btn = QPushButton("Track this slice's ROIs through Z")
        self._prop_track_btn.setEnabled(False)
        self._prop_track_btn.setToolTip("Track the ROIs of the chosen class drawn on the current slice "
                                        "through the stack (two SAM 3 passes).")
        seg_lay.addWidget(self._prop_track_btn)

        self._clear_btn = QPushButton("Clear this class on all slices")
        self._clear_btn.setEnabled(False)
        seg_lay.addWidget(self._clear_btn)

        self._seg_status = QLabel("")
        self._seg_status.setWordWrap(True)
        self._seg_status.setStyleSheet("font-size: 11px; color: #6c7086;")
        seg_lay.addWidget(self._seg_status)
        layout.addWidget(seg_box)

        # ── export / render ───────────────────────────────────────────────────
        out_box = QGroupBox("Label Volume and 3D Movie")
        out_lay = QVBoxLayout(out_box)
        out_lay.setSpacing(6)

        self._export_btn = QPushButton("Export label volume (MRC)")
        self._export_btn.setEnabled(False)
        out_lay.addWidget(self._export_btn)

        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Style:"))
        self._style_combo = QComboBox()
        self._style_combo.addItem("Talk (dark)", "talk")
        self._style_combo.addItem("Paper (white)", "paper")
        style_row.addWidget(self._style_combo)
        style_row.addWidget(QLabel("Seconds:"))
        self._secs_spin = QDoubleSpinBox()
        self._secs_spin.setRange(2.0, 60.0)
        self._secs_spin.setValue(20.0)
        style_row.addWidget(self._secs_spin)
        out_lay.addLayout(style_row)

        zex_row = QHBoxLayout()
        zex_row.addWidget(QLabel("Z exaggeration:"))
        self._zexag_spin = QDoubleSpinBox()
        self._zexag_spin.setRange(1.0, 5.0)
        self._zexag_spin.setSingleStep(0.5)
        self._zexag_spin.setValue(1.0)
        zex_row.addWidget(self._zexag_spin)
        zex_row.addStretch()
        out_lay.addLayout(zex_row)

        self._render_btn = QPushButton("Render spinning movie")
        self._render_btn.setEnabled(False)
        out_lay.addWidget(self._render_btn)
        layout.addWidget(out_box)

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
        self._slab_spin.valueChanged.connect(lambda _v: self.view_changed.emit(self._slab_spin.value(), self._smooth_spin.value()))
        self._smooth_spin.valueChanged.connect(lambda _v: self.view_changed.emit(self._slab_spin.value(), self._smooth_spin.value()))
        self._segment_btn.clicked.connect(
            lambda: self.segment_slice_requested.emit(self.prompt_text, self.class_label, self._conf_spin.value()))
        self._prop_text_btn.clicked.connect(
            lambda: self.propagate_requested.emit("text", self.class_label, self._prop_from.value(), self._prop_to.value()))
        self._prop_track_btn.clicked.connect(
            lambda: self.propagate_requested.emit("track", self.class_label, self._prop_from.value(), self._prop_to.value()))
        self._clear_btn.clicked.connect(lambda: self.clear_class_requested.emit(self.class_label))
        self._export_btn.clicked.connect(self.export_requested.emit)
        self._render_btn.clicked.connect(
            lambda: self.render_requested.emit(self._style_combo.currentData(), self._secs_spin.value(),
                                               self._zexag_spin.value()))

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
        for sp in (self._z_slider, self._z_spin, self._proj_from, self._proj_to, self._prop_from, self._prop_to):
            sp.setRange(0, max(0, n - 1))
        self._z_slider.setValue(0)
        self._z_spin.setValue(0)
        self._of_label.setText(f"of {n - 1}")
        self._proj_from.setValue(0)
        self._proj_to.setValue(max(0, n - 1))
        self._prop_from.setValue(0)
        self._prop_to.setValue(max(0, n - 1))
        enabled = n > 1
        for wdg in (self._z_slider, self._z_spin, self._prev_btn, self._next_btn,
                    self._prop_text_btn, self._prop_track_btn):
            wdg.setEnabled(enabled)
        for wdg in (self._proj_btn, self._segment_btn, self._clear_btn, self._export_btn, self._render_btn):
            wdg.setEnabled(True)
        self._updating = False

    def set_z(self, z: int) -> None:
        self._updating = True
        self._z_slider.setValue(z)
        self._z_spin.setValue(z)
        self._updating = False

    def set_annotation_summary(self, text: str) -> None:
        self._ann_label.setText(text)

    def set_seg_status(self, text: str) -> None:
        self._seg_status.setText(text)

    def set_busy(self, busy: bool) -> None:
        for wdg in (self._segment_btn, self._prop_text_btn, self._prop_track_btn, self._clear_btn,
                    self._export_btn, self._render_btn):
            wdg.setEnabled(not busy and self._n_slices > 0)

    def add_class(self, label: str) -> None:
        if label and self._class_combo.findText(label, Qt.MatchFlag.MatchFixedString) < 0:
            self._class_combo.addItem(label)

    @property
    def current_z(self) -> int:
        return self._z_slider.value()

    @property
    def class_label(self) -> str:
        return self._class_combo.currentText().strip() or "Unlabelled"

    @property
    def prompt_text(self) -> str:
        return self._prompt_edit.text().strip() or self.class_label

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
