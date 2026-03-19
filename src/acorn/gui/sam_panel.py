"""SAM 3 / SAM 2 / micro-SAM semi-automatic annotation panel."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

_SAM3_AVAILABLE = importlib.util.find_spec("sam3") is not None
_USAM_AVAILABLE = importlib.util.find_spec("micro_sam") is not None


def _find_local_checkpoints() -> list[tuple[str, str]]:
    """Scan common locations for SAM checkpoints.

    Returns list of (display_name, absolute_path) tuples.
    """
    home = Path.home()
    found: list[tuple[str, str]] = []

    # SAM3: HuggingFace cache
    sam3_cache = home / ".cache" / "huggingface" / "hub" / "models--facebook--sam3"
    if sam3_cache.exists():
        for p in sorted(sam3_cache.glob("snapshots/*/sam3.pt")):
            found.append(("SAM 3 (cached)", str(p)))

    # SAM2: local source checkout
    for sam2_dir in [home / "sam2" / "checkpoints", home / "sam2_checkpoints"]:
        if sam2_dir.exists():
            for p in sorted(sam2_dir.glob("*.pt")):
                found.append((f"SAM 2 — {p.stem}", str(p)))

    # SAM2: HuggingFace cache
    sam2_cache = home / ".cache" / "huggingface" / "hub"
    if sam2_cache.exists():
        for p in sorted(sam2_cache.glob("models--facebook--sam2*/snapshots/**/*.pt")):
            found.append((f"SAM 2 — {p.stem} (cached)", str(p)))

    # micro-SAM: local user checkpoint cache
    usam_cache = home / ".cache" / "micro_sam"
    if usam_cache.exists():
        for p in sorted(usam_cache.glob("*/*.pt")):
            found.append((f"micro-SAM — {p.parent.name} (cached)", str(p)))

    # micro-SAM: shared system-wide models (all users)
    usam_shared = Path("/opt/acorn/models/micro_sam")
    if usam_shared.exists():
        for p in sorted(usam_shared.glob("*/*.pt")):
            if str(p) not in {f for _, f in found}:  # skip if already listed above
                found.append((f"micro-SAM — {p.parent.name} (shared)", str(p)))

    return found


class SAMPanel(QWidget):
    """
    Controls for SAM 3 / SAM 2 semi-automatic annotation.

    SAM 3 is used automatically when installed; falls back to SAM 2.

    Workflow
    --------
    1. Load Model (downloads from HuggingFace Hub if no checkpoint set)
    2. Click "Run Auto Segment" or switch to Point/Box prompt mode and click on canvas
    3. Predicted masks appear as candidate ROI annotations
    4. Use Undo to remove unwanted predictions, or Accept All to keep them all

    Signals
    -------
    load_model_requested(checkpoint, model_cfg)  — load the SAM model
    auto_segment_requested()                     — run automatic mask generation
    point_prompt_requested(x, y, label)          — add a point prompt (1=pos, 0=neg)
    box_prompt_requested(x0, y0, x1, y1)         — add a box prompt
    accept_all_requested()                       — confirm all pending masks as ROIs
    reject_all_requested()                       — discard all pending masks
    settings_changed()                           — any parameter changed
    """

    load_model_requested    = pyqtSignal(str, str, str)      # checkpoint_path, model_cfg, backend
    auto_segment_requested  = pyqtSignal()
    point_prompt_mode_set   = pyqtSignal(bool)               # True = positive, False = negative
    box_prompt_mode_set     = pyqtSignal()
    scribble_mode_set       = pyqtSignal()                   # freehand stroke → SAM prompts
    prompt_mode_cleared     = pyqtSignal()                   # all mode buttons unchecked
    commit_new_requested    = pyqtSignal()                   # keep current preview, start next object
    clear_points_requested  = pyqtSignal()                   # discard accumulated points + preview
    undo_point_requested    = pyqtSignal()                   # remove the last added point
    accept_all_requested    = pyqtSignal()
    reject_all_requested    = pyqtSignal()
    accept_and_queue_requested = pyqtSignal()   # accept masks then queue image for export
    exclude_zone_mode_set   = pyqtSignal()                   # user clicked Draw Exclude Zone
    exclude_zone_cleared    = pyqtSignal()                   # user clicked Clear Exclude
    crop_region_mode_set    = pyqtSignal()                   # user clicked Draw Crop Region
    crop_region_cleared     = pyqtSignal()                   # user clicked Clear Crop

    def __init__(self, parent=None):
        super().__init__(parent)
        _content = QWidget()
        layout = QVBoxLayout(_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── model settings ────────────────────────────────────────────────────
        if _SAM3_AVAILABLE:
            model_box = QGroupBox("Model  (SAM 3 detected)")
        elif _USAM_AVAILABLE:
            model_box = QGroupBox("Model  (micro-SAM available)")
        else:
            model_box = QGroupBox("Model  (SAM 2)")
        model_layout = QFormLayout(model_box)

        # Backend selector
        self._backend_combo = QComboBox()
        if _SAM3_AVAILABLE:
            self._backend_combo.addItem("Auto (SAM 3 preferred)", "auto")
            self._backend_combo.addItem("SAM 3 only", "sam3")
            self._backend_combo.addItem("SAM 2 only", "sam2")
        else:
            self._backend_combo.addItem("SAM 2", "sam2")
        if _USAM_AVAILABLE:
            self._backend_combo.addItem("micro-SAM (\u03bcSAM)", "usam")
        self._backend_combo.setToolTip(
            "Which model backend to use.\n"
            "micro-SAM uses SAM1 checkpoints fine-tuned on microscopy data\n"
            "(light microscopy and electron microscopy organelle models)."
        )
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        model_layout.addRow("Backend:", self._backend_combo)

        # Config dropdown — only relevant for SAM 2
        self._model_cfg = QComboBox()
        for cfg in ["sam2_hiera_large", "sam2_hiera_base_plus",
                    "sam2_hiera_small", "sam2_hiera_tiny"]:
            self._model_cfg.addItem(cfg)
        self._cfg_label = QLabel("Config:")
        model_layout.addRow(self._cfg_label, self._model_cfg)
        if _SAM3_AVAILABLE:
            self._cfg_label.hide()
            self._model_cfg.hide()

        # micro-SAM model type selector
        self._usam_model_combo = QComboBox()
        if _USAM_AVAILABLE:
            from acorn.core.usam_predictor import available_models
            for model_type, display in available_models():
                self._usam_model_combo.addItem(display, model_type)
        self._usam_model_label = QLabel("Model:")
        model_layout.addRow(self._usam_model_label, self._usam_model_combo)
        self._usam_model_label.hide()
        self._usam_model_combo.hide()

        ckpt_row = QHBoxLayout()
        self._ckpt_combo = QComboBox()
        self._ckpt_combo.addItem("Auto (download from HuggingFace Hub)", "")
        for display, path in _find_local_checkpoints():
            self._ckpt_combo.addItem(display, path)
        # Auto-select the first local checkpoint only for SAM2/3 backends
        default_backend = self._backend_combo.currentData()
        if self._ckpt_combo.count() > 1 and default_backend != "usam":
            self._ckpt_combo.setCurrentIndex(1)
        self._ckpt_combo.setToolTip(
            "Select a locally cached checkpoint or let the model download automatically."
        )
        browse_ckpt = QPushButton("Browse…")
        browse_ckpt.setFixedWidth(80)
        browse_ckpt.clicked.connect(self._browse_checkpoint)
        ckpt_row.setSpacing(4)
        ckpt_row.addWidget(self._ckpt_combo, 1)
        ckpt_row.addWidget(browse_ckpt)
        model_layout.addRow("Checkpoint:", ckpt_row)

        load_btn = QPushButton("Load Model")
        load_btn.setStyleSheet("background:#2980b9;color:white;font-weight:bold;")
        if _SAM3_AVAILABLE:
            load_btn.setToolTip(
                "Load SAM 3.  If no checkpoint is set, downloads from "
                "HuggingFace Hub on first use.  SAM 3 takes priority over SAM 2."
            )
        else:
            load_btn.setToolTip(
                "Load SAM 2.  If no checkpoint is set, downloads sam2-hiera-large "
                "from HuggingFace Hub on first use."
            )
        load_btn.clicked.connect(self._on_load_model)
        model_layout.addRow("", load_btn)

        if _SAM3_AVAILABLE:
            default_status = "SAM 3 installed — model not loaded yet"
        else:
            default_status = "Model not loaded"
        self._model_status = QLabel(default_status)
        self._model_status.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._model_status.setWordWrap(True)
        model_layout.addRow("", self._model_status)

        layout.addWidget(model_box)

        # ── prompt mode ───────────────────────────────────────────────────────
        prompt_box = QGroupBox("Prompt Mode")
        prompt_layout = QVBoxLayout(prompt_box)
        prompt_layout.setSpacing(4)

        info = QLabel(
            "Select a mode, then click on the canvas.\n"
            "You can add as many positive and negative points\n"
            "as needed — SAM updates the preview after each click.\n"
            "Box: drag two corners around the object.\n"
            "When satisfied, hit Commit & New to lock the mask\n"
            "and start annotating the next object."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: palette(mid);")
        prompt_layout.addWidget(info)

        self._mode_pos_btn = QPushButton("+ Positive Point")
        self._mode_pos_btn.setCheckable(True)
        self._mode_pos_btn.setStyleSheet("background:#27ae60;color:white;")
        self._mode_pos_btn.setToolTip("Click on canvas to add a positive (foreground) point")
        self._mode_pos_btn.clicked.connect(self._on_pos_clicked)

        self._mode_neg_btn = QPushButton("- Negative Point")
        self._mode_neg_btn.setCheckable(True)
        self._mode_neg_btn.setStyleSheet("background:#e74c3c;color:white;")
        self._mode_neg_btn.setToolTip("Click on canvas to add a negative (background) point")
        self._mode_neg_btn.clicked.connect(self._on_neg_clicked)

        self._mode_box_btn = QPushButton("Box")
        self._mode_box_btn.setCheckable(True)
        self._mode_box_btn.setStyleSheet("background:#8e44ad;color:white;")
        self._mode_box_btn.setToolTip("Click two corners to define a bounding box prompt")
        self._mode_box_btn.clicked.connect(self._on_box_clicked)

        self._mode_scribble_btn = QPushButton("Scribble")
        self._mode_scribble_btn.setCheckable(True)
        self._mode_scribble_btn.setStyleSheet("background:#0e7490;color:white;")
        self._mode_scribble_btn.setToolTip(
            "Draw a freehand stroke along a feature (e.g. membrane layer).\n"
            "Points sampled along the stroke become positive SAM prompts.\n"
            "You can draw multiple strokes to refine the mask."
        )
        self._mode_scribble_btn.clicked.connect(self._on_scribble_clicked)

        mode_row1 = QHBoxLayout()
        mode_row1.addWidget(self._mode_pos_btn)
        mode_row1.addWidget(self._mode_neg_btn)
        mode_row2 = QHBoxLayout()
        mode_row2.addWidget(self._mode_box_btn)
        mode_row2.addWidget(self._mode_scribble_btn)
        prompt_layout.addLayout(mode_row1)
        prompt_layout.addLayout(mode_row2)

        # ── label for the resulting mask ───────────────────────────────────────
        label_row = QHBoxLayout()
        label_row.addWidget(QLabel("Label:"))
        self._point_label = QLineEdit("Foreground")
        self._point_label.setPlaceholderText("mask label…")
        self._point_label.setToolTip("Label assigned to the mask when committed")
        label_row.addWidget(self._point_label, 1)
        prompt_layout.addLayout(label_row)

        # Quick-select label buttons — first two are fixed, rest are user-defined
        self._label_btn_row = QHBoxLayout()
        self._label_buttons: list[QPushButton] = []
        self._user_label_colors = ["#8e44ad", "#1a6fa8", "#d35400", "#16a085", "#2c3e50"]
        for preset, color in [("Foreground", "#27ae60"), ("Background", "#e74c3c")]:
            btn = self._make_label_btn(preset, color)
            self._label_btn_row.addWidget(btn)
        prompt_layout.addLayout(self._label_btn_row)

        add_label_row = QHBoxLayout()
        self._new_label_edit = QLineEdit()
        self._new_label_edit.setPlaceholderText("new class name…")
        add_label_btn = QPushButton("+ Add")
        add_label_btn.setFixedWidth(56)
        add_label_btn.setToolTip("Add a quick-select button for this class name")
        add_label_btn.clicked.connect(self._on_add_label)
        self._new_label_edit.returnPressed.connect(self._on_add_label)
        add_label_row.addWidget(self._new_label_edit, 1)
        add_label_row.addWidget(add_label_btn)
        prompt_layout.addLayout(add_label_row)

        # ── commit / clear ────────────────────────────────────────────────────
        action_info = QLabel(
            "Commit & New locks this mask. No need to confirm between\n"
            "individual points — only press this when you are done with\n"
            "the current object and want to annotate the next one."
        )
        action_info.setWordWrap(True)
        action_info.setStyleSheet("font-size: 11px; color: palette(mid);")
        prompt_layout.addWidget(action_info)

        action_row = QHBoxLayout()
        commit_btn = QPushButton("Commit & New")
        commit_btn.setStyleSheet("background:#e67e22;color:white;font-weight:bold;")
        commit_btn.setToolTip(
            "Lock the current preview mask, switch to Select mode for vertex editing,\n"
            "then click + Positive Point to start the next object."
        )
        commit_btn.clicked.connect(self.commit_new_requested)

        undo_pt_btn = QPushButton("Undo Point")
        undo_pt_btn.setToolTip("Remove the last added point and re-run SAM with the remaining points")
        undo_pt_btn.clicked.connect(self.undo_point_requested)

        clear_pts_btn = QPushButton("Clear Points")
        clear_pts_btn.setToolTip("Discard all accumulated points and the current preview mask")
        clear_pts_btn.clicked.connect(self.clear_points_requested)

        action_row.addWidget(commit_btn)
        action_row.addWidget(undo_pt_btn)
        action_row.addWidget(clear_pts_btn)
        prompt_layout.addLayout(action_row)

        layout.addWidget(prompt_box)

        # ── region controls ───────────────────────────────────────────────────
        region_box = QGroupBox("Region Controls")
        region_layout = QVBoxLayout(region_box)
        region_layout.setSpacing(4)

        region_info = QLabel(
            "Exclude Zone: drag a box around areas SAM should ignore (e.g. scale bar).\n"
            "Crop Region: SAM runs only inside this rectangle; results map back to the full image."
        )
        region_info.setWordWrap(True)
        region_info.setStyleSheet("font-size: 11px; color: palette(mid);")
        region_layout.addWidget(region_info)

        excl_row = QHBoxLayout()
        self._exclude_btn = QPushButton("Draw Exclude Zone")
        self._exclude_btn.setCheckable(True)
        self._exclude_btn.setStyleSheet("background:#8b0000;color:white;")
        self._exclude_btn.setToolTip("Drag on canvas to mark a region SAM will ignore")
        self._exclude_btn.clicked.connect(self._on_exclude_clicked)
        excl_clear_btn = QPushButton("Clear")
        excl_clear_btn.setFixedWidth(56)
        excl_clear_btn.setToolTip("Remove the current exclude zone")
        excl_clear_btn.clicked.connect(self.exclude_zone_cleared)
        excl_row.addWidget(self._exclude_btn, 1)
        excl_row.addWidget(excl_clear_btn)
        region_layout.addLayout(excl_row)

        crop_row = QHBoxLayout()
        self._crop_btn = QPushButton("Draw Crop Region")
        self._crop_btn.setCheckable(True)
        self._crop_btn.setStyleSheet("background:#005f7a;color:white;")
        self._crop_btn.setToolTip("Drag on canvas to restrict SAM to this sub-region")
        self._crop_btn.clicked.connect(self._on_crop_clicked)
        crop_clear_btn = QPushButton("Clear")
        crop_clear_btn.setFixedWidth(56)
        crop_clear_btn.setToolTip("Remove the crop restriction — SAM will run on the full image")
        crop_clear_btn.clicked.connect(self.crop_region_cleared)
        crop_row.addWidget(self._crop_btn, 1)
        crop_row.addWidget(crop_clear_btn)
        region_layout.addLayout(crop_row)

        self._keep_regions_chk = QCheckBox("Keep regions across images")
        self._keep_regions_chk.setToolTip(
            "When checked, the exclude zone and crop region persist when you\n"
            "switch to the next image.  Useful when the scale bar is always in\n"
            "the same position.  Uncheck to clear automatically on image switch."
        )
        region_layout.addWidget(self._keep_regions_chk)

        layout.addWidget(region_box)

        # ── automatic segmentation ────────────────────────────────────────────
        auto_box = QGroupBox("Automatic Segmentation")
        auto_layout = QFormLayout(auto_box)
        auto_layout.setSpacing(4)

        self._pts_per_side = QSpinBox()
        self._pts_per_side.setRange(4, 128)
        self._pts_per_side.setValue(32)
        self._pts_per_side.setToolTip("Grid density for automatic mask generation")
        auto_layout.addRow("Points/side:", self._pts_per_side)

        self._iou_thresh = QDoubleSpinBox()
        self._iou_thresh.setRange(0.0, 1.0)
        self._iou_thresh.setSingleStep(0.01)
        self._iou_thresh.setValue(0.88)
        self._iou_thresh.setToolTip("Minimum predicted IoU score to keep a mask")
        auto_layout.addRow("IoU threshold:", self._iou_thresh)

        self._stability_thresh = QDoubleSpinBox()
        self._stability_thresh.setRange(0.0, 1.0)
        self._stability_thresh.setSingleStep(0.01)
        self._stability_thresh.setValue(0.95)
        self._stability_thresh.setToolTip("Minimum stability score to keep a mask")
        auto_layout.addRow("Stability:", self._stability_thresh)

        self._min_area = QSpinBox()
        self._min_area.setRange(1, 100000)
        self._min_area.setValue(200)
        self._min_area.setSuffix(" px")
        self._min_area.setToolTip("Discard masks smaller than this area")
        auto_layout.addRow("Min area:", self._min_area)

        self._label_combo = QComboBox()
        for lbl in ["Foreground", "Background", "Ignore"]:
            self._label_combo.addItem(lbl)
        auto_layout.addRow("Label:", self._label_combo)

        auto_btn = QPushButton("Run Auto-Segment")
        auto_btn.setStyleSheet("background:#1a6fa8;color:white;font-weight:bold;")
        auto_btn.setToolTip(
            "Run SAM automatic mask generation on the full image.\n"
            "All predicted masks are added as ROI annotations."
        )
        auto_btn.clicked.connect(self.auto_segment_requested)
        auto_layout.addRow("", auto_btn)

        layout.addWidget(auto_box)

        # ── accept / reject ───────────────────────────────────────────────────
        self._sam_status = QLabel("")
        self._sam_status.setWordWrap(True)
        layout.addWidget(self._sam_status)

        ar_row = QHBoxLayout()
        accept_btn = QPushButton("Accept All")
        accept_btn.setStyleSheet("background:#27ae60;color:white;font-weight:bold;")
        accept_btn.setToolTip("Keep all predicted masks as permanent ROI annotations")
        accept_btn.clicked.connect(self.accept_all_requested)

        reject_btn = QPushButton("Reject All")
        reject_btn.setStyleSheet("background:#c0392b;color:white;")
        reject_btn.setToolTip("Remove all pending SAM mask annotations")
        reject_btn.clicked.connect(self.reject_all_requested)

        ar_row.addWidget(accept_btn)
        ar_row.addWidget(reject_btn)
        layout.addLayout(ar_row)

        accept_queue_btn = QPushButton("Accept & Queue for Export")
        accept_queue_btn.setStyleSheet("background:#0e7490;color:white;font-weight:bold;")
        accept_queue_btn.setToolTip(
            "Accept all SAM masks and immediately add this image to the export queue.\n"
            "Saves a tab switch — continue to the next image straight away."
        )
        accept_queue_btn.clicked.connect(self.accept_and_queue_requested)
        layout.addWidget(accept_queue_btn)

        layout.addStretch()
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setWidget(_content)
        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(_scroll)

    # ── public API ────────────────────────────────────────────────────────────

    def set_model_status(self, msg: str, loaded: bool = False) -> None:
        color = "#27ae60" if loaded else "palette(mid)"
        self._model_status.setStyleSheet(f"font-size: 11px; color: {color};")
        self._model_status.setText(msg)

    def set_sam_status(self, msg: str) -> None:
        self._sam_status.setText(msg)

    @property
    def label(self) -> str:
        return self._label_combo.currentText()

    @property
    def point_label(self) -> str:
        """Label assigned to masks created by point/box prompts."""
        return self._point_label.text().strip() or "Foreground"

    @property
    def auto_params(self) -> dict:
        return {
            "points_per_side":         self._pts_per_side.value(),
            "pred_iou_thresh":         self._iou_thresh.value(),
            "stability_score_thresh":  self._stability_thresh.value(),
            "min_mask_region_area":    self._min_area.value(),
        }

    # ── public API ────────────────────────────────────────────────────────────

    def reset_prompt_mode(self) -> None:
        """Uncheck all prompt-mode buttons without emitting any signals."""
        for btn in (self._mode_pos_btn, self._mode_neg_btn,
                    self._mode_box_btn, self._mode_scribble_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def set_positive_mode(self) -> None:
        """Activate positive-point mode programmatically (e.g. from keyboard shortcut)."""
        self._activate_mode(self._mode_pos_btn)

    def set_negative_mode(self) -> None:
        """Activate negative-point mode programmatically (e.g. from keyboard shortcut)."""
        self._activate_mode(self._mode_neg_btn)

    # ── slots ─────────────────────────────────────────────────────────────────

    def _activate_mode(self, active_btn: QPushButton) -> None:
        """Check active_btn and silently uncheck the others."""
        for btn in (self._mode_pos_btn, self._mode_neg_btn,
                    self._mode_box_btn, self._mode_scribble_btn):
            btn.blockSignals(True)
            btn.setChecked(btn is active_btn)
            btn.blockSignals(False)

    def _on_pos_clicked(self) -> None:
        self._activate_mode(self._mode_pos_btn)
        self.point_prompt_mode_set.emit(True)

    def _on_neg_clicked(self) -> None:
        self._activate_mode(self._mode_neg_btn)
        self.point_prompt_mode_set.emit(False)

    def _on_box_clicked(self) -> None:
        self._activate_mode(self._mode_box_btn)
        self.box_prompt_mode_set.emit()

    def _on_scribble_clicked(self) -> None:
        self._activate_mode(self._mode_scribble_btn)
        self.scribble_mode_set.emit()

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SAM checkpoint", "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            label = Path(path).name
            # Add to combo if not already present
            for i in range(self._ckpt_combo.count()):
                if self._ckpt_combo.itemData(i) == path:
                    self._ckpt_combo.setCurrentIndex(i)
                    return
            self._ckpt_combo.addItem(label, path)
            self._ckpt_combo.setCurrentIndex(self._ckpt_combo.count() - 1)

    def _on_backend_changed(self, index: int) -> None:
        backend = self._backend_combo.itemData(index)
        sam2_visible = (backend == "sam2")
        usam_visible = (backend == "usam")
        self._cfg_label.setVisible(sam2_visible)
        self._model_cfg.setVisible(sam2_visible)
        self._usam_model_label.setVisible(usam_visible)
        self._usam_model_combo.setVisible(usam_visible)
        # Checkpoints are not interchangeable across backends — reset to Auto
        self._ckpt_combo.setCurrentIndex(0)

    def _on_load_model(self) -> None:
        ckpt    = self._ckpt_combo.currentData() or ""
        cfg     = self._model_cfg.currentText() + ".yaml"
        backend = self._backend_combo.currentData()
        self._model_status.setText("Loading…")
        self.load_model_requested.emit(ckpt, cfg, backend)

    @property
    def usam_model_type(self) -> str:
        """Currently selected micro-SAM model type key (e.g. 'vit_b_lm')."""
        return self._usam_model_combo.currentData() or "vit_b_lm"

    def _make_label_btn(self, name: str, color: str) -> QPushButton:
        btn = QPushButton(name)
        btn.setStyleSheet(f"background:{color};color:white;font-size:10px;")
        btn.clicked.connect(lambda _, n=name: self._on_label_btn_clicked(n))
        self._label_buttons.append(btn)
        return btn

    def _on_label_btn_clicked(self, name: str) -> None:
        if name.strip().lower() == "background":
            self._on_neg_clicked()
        else:
            self._point_label.setText(name)
            self._on_pos_clicked()

    @property
    def keep_regions_across_images(self) -> bool:
        return self._keep_regions_chk.isChecked()

    def reset_region_btns(self) -> None:
        """Uncheck both region-draw buttons (called after a drag completes)."""
        for btn in (self._exclude_btn, self._crop_btn):
            btn.blockSignals(True)
            btn.setChecked(False)
            btn.blockSignals(False)

    def _on_exclude_clicked(self) -> None:
        self._exclude_btn.setChecked(True)
        self._crop_btn.blockSignals(True)
        self._crop_btn.setChecked(False)
        self._crop_btn.blockSignals(False)
        self.exclude_zone_mode_set.emit()

    def _on_crop_clicked(self) -> None:
        self._crop_btn.setChecked(True)
        self._exclude_btn.blockSignals(True)
        self._exclude_btn.setChecked(False)
        self._exclude_btn.blockSignals(False)
        self.crop_region_mode_set.emit()

    def _on_add_label(self) -> None:
        name = self._new_label_edit.text().strip()
        if not name:
            return
        # Don't add duplicates
        for btn in self._label_buttons:
            if btn.text() == name:
                self._point_label.setText(name)
                self._new_label_edit.clear()
                return
        color = self._user_label_colors[
            (len(self._label_buttons) - 2) % len(self._user_label_colors)
        ]
        btn = self._make_label_btn(name, color)
        self._label_btn_row.addWidget(btn)
        btn.show()
        self._point_label.setText(name)
        self._new_label_edit.clear()
