"""YOLO detection and segmentation annotation panel."""

from __future__ import annotations

import os
from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QVBoxLayout, QWidget,
)

_YOLO_PRESETS = [
    # YOLO26 (latest — NMS-free)
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    "yolo26n-seg.pt", "yolo26s-seg.pt", "yolo26m-seg.pt", "yolo26l-seg.pt", "yolo26x-seg.pt",
    # YOLO11
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt", "yolo11l-seg.pt", "yolo11x-seg.pt",
    # YOLOv10
    "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolov10l.pt", "yolov10x.pt",
    # YOLOv9
    "yolov9c.pt", "yolov9e.pt",
    # YOLOv8
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolov8n-seg.pt", "yolov8s-seg.pt", "yolov8m-seg.pt", "yolov8l-seg.pt", "yolov8x-seg.pt",
    # YOLOv5
    "yolov5n.pt", "yolov5s.pt", "yolov5m.pt", "yolov5l.pt", "yolov5x.pt",
]


def _find_local_yolo_models() -> list[tuple[str, str]]:
    """Scan common cache locations for downloaded YOLO checkpoints.

    Returns list of (display_name, absolute_path) tuples.
    """
    home = Path.home()
    found: list[tuple[str, str]] = []
    seen: set[str] = set()

    _shared = os.environ.get("ACORN_MODELS_DIR")
    search_dirs = [
        *(  [Path(_shared) / "yolo"] if _shared else []),
        Path("/opt/acorn/models/yolo"),        # shared system-wide (all users)
        home / ".cache" / "ultralytics",
        home / "ultralytics",
        home / "weights",
        home / "models",
    ]

    preset_stems = {Path(n).stem for n in _YOLO_PRESETS}

    for d in search_dirs:
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.pt")):
            key = str(p)
            if key in seen:
                continue
            seen.add(key)
            label = f"{p.name} (local)" if p.stem in preset_stems else p.name
            found.append((label, key))

    return found


class YOLOPanel(QWidget):
    """
    Controls for YOLO-based object detection and segmentation.

    Supports any ultralytics YOLO model — pre-trained or custom-trained on
    any EM modality (cryo-EM SPA, STEM, TEM, EDX, materials science, etc.).

    Workflow
    --------
    1. Load Model (local .pt path or ultralytics model name, e.g. yolo11n.pt)
    2. Click Run Detection or Run Detection + Segmentation
    3. Results appear as ROI annotations
    4. Optionally pipe detected boxes to SAM for precise masks
    5. Undo unwanted, or Accept All

    Signals
    -------
    load_model_requested(model_path)   — load a YOLO model checkpoint
    detect_requested()                 — run box detection on current image
    detect_seg_requested()             — run detection + segmentation
    accept_all_requested()             — confirm all pending annotations
    reject_all_requested()             — discard all pending annotations
    """

    load_model_requested  = pyqtSignal(str)
    detect_requested      = pyqtSignal()
    detect_seg_requested  = pyqtSignal()
    accept_all_requested  = pyqtSignal()
    reject_all_requested  = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── model ─────────────────────────────────────────────────────────────
        model_box = QGroupBox("Model")
        model_layout = QFormLayout(model_box)

        # Unified model selector: local files first, then downloadable presets
        model_row = QHBoxLayout()
        self._model_combo = QComboBox()

        local_models = _find_local_yolo_models()
        for display, path in local_models:
            self._model_combo.addItem(display, path)

        if local_models:
            self._model_combo.insertSeparator(self._model_combo.count())

        for name in _YOLO_PRESETS:
            self._model_combo.addItem(f"{name}  (download)", name)

        self._model_combo.setToolTip(
            "Local checkpoints are listed first.\n"
            "Preset names (marked 'download') are fetched automatically on Load."
        )
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse_model)
        model_row.setSpacing(4)
        model_row.addWidget(self._model_combo, 1)
        model_row.addWidget(browse_btn)
        model_layout.addRow("Model:", model_row)

        load_btn = QPushButton("Load Model")
        load_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        load_btn.setToolTip(
            "Load a YOLO model.  Provide a local .pt path, or a model name "
            "(e.g. yolo11n.pt, yolo11n-seg.pt) to download automatically."
        )
        load_btn.clicked.connect(self._on_load_model)
        model_layout.addRow("", load_btn)

        self._model_status = QLabel("Model not loaded")
        self._model_status.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._model_status.setWordWrap(True)
        model_layout.addRow("", self._model_status)

        layout.addWidget(model_box)

        # ── detection parameters ───────────────────────────────────────────────
        param_box = QGroupBox("Detection Parameters")
        param_layout = QFormLayout(param_box)

        self._conf_thresh = QDoubleSpinBox()
        self._conf_thresh.setRange(0.01, 1.0)
        self._conf_thresh.setSingleStep(0.05)
        self._conf_thresh.setValue(0.25)
        self._conf_thresh.setToolTip("Minimum confidence score to keep a detection")
        param_layout.addRow("Confidence:", self._conf_thresh)

        self._iou_thresh = QDoubleSpinBox()
        self._iou_thresh.setRange(0.01, 1.0)
        self._iou_thresh.setSingleStep(0.05)
        self._iou_thresh.setValue(0.45)
        self._iou_thresh.setToolTip(
            "NMS IoU threshold — lower values suppress more overlapping boxes"
        )
        param_layout.addRow("NMS IoU:", self._iou_thresh)

        self._label_combo = QComboBox()
        for lbl in ["Foreground", "Background", "Ignore"]:
            self._label_combo.addItem(lbl)
        param_layout.addRow("Label:", self._label_combo)

        self._as_rects = QCheckBox("Add as rectangles (not polygons)")
        self._as_rects.setToolTip(
            "If checked, detected boxes are added as rectangle outlines "
            "instead of 4-vertex ROI polygons."
        )
        param_layout.addRow("", self._as_rects)

        layout.addWidget(param_box)

        # ── run buttons ────────────────────────────────────────────────────────
        run_box = QGroupBox("Run")
        run_layout = QVBoxLayout(run_box)

        detect_btn = QPushButton("Run Detection")
        detect_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        detect_btn.setToolTip("Run YOLO detection — adds bounding boxes as annotations")
        detect_btn.clicked.connect(self.detect_requested)
        run_layout.addWidget(detect_btn)

        seg_btn = QPushButton("Detect + Segment (YOLO-seg)")
        seg_btn.setStyleSheet("background:#1a5fa8;color:white;")
        seg_btn.setToolTip(
            "Run YOLO segmentation (requires a YOLO-seg .pt model).\n"
            "Each detected object gets a precise polygon mask."
        )
        seg_btn.clicked.connect(self.detect_seg_requested)
        run_layout.addWidget(seg_btn)

        layout.addWidget(run_box)

        # ── status and accept / reject ─────────────────────────────────────────
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        ar_row = QHBoxLayout()
        self._accept_btn = QPushButton("Accept All")
        self._accept_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        self._accept_btn.setToolTip("Keep all pending YOLO annotations as permanent ROIs")
        self._accept_btn.clicked.connect(self.accept_all_requested)

        self._reject_btn = QPushButton("Reject All")
        self._reject_btn.setStyleSheet("background:#c0392b;color:white;")
        self._reject_btn.setToolTip("Remove all pending YOLO annotations")
        self._reject_btn.clicked.connect(self.reject_all_requested)

        ar_row.addWidget(self._accept_btn)
        ar_row.addWidget(self._reject_btn)
        layout.addLayout(ar_row)
        layout.addStretch()

    # ── public API ────────────────────────────────────────────────────────────

    def hide_footer(self) -> None:
        """Hide accept/reject/status — used when a parent panel owns shared controls."""
        self._status.setVisible(False)
        self._accept_btn.setVisible(False)
        self._reject_btn.setVisible(False)

    def set_model_status(self, msg: str, loaded: bool = False) -> None:
        color = "#4dbb78" if loaded else "palette(mid)"
        self._model_status.setStyleSheet(f"font-size: 11px; color: {color};")
        self._model_status.setText(msg)

    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    @property
    def model_path(self) -> str:
        return self._model_combo.currentData() or ""

    @property
    def conf_thresh(self) -> float:
        return self._conf_thresh.value()

    @property
    def iou_thresh(self) -> float:
        return self._iou_thresh.value()

    @property
    def label(self) -> str:
        return self._label_combo.currentText()

    @property
    def as_rectangles(self) -> bool:
        return self._as_rects.isChecked()

    # ── slots ──────────────────────────────────────────────────────────────────

    def _browse_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model checkpoint", "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            # Add to combo if not already present, then select it
            for i in range(self._model_combo.count()):
                if self._model_combo.itemData(i) == path:
                    self._model_combo.setCurrentIndex(i)
                    return
            self._model_combo.insertItem(0, Path(path).name, path)
            self._model_combo.setCurrentIndex(0)

    def _on_load_model(self) -> None:
        path = self._model_combo.currentData() or ""
        if not path:
            self._model_status.setText("Select a model first.")
            return
        self._model_status.setText("Loading…")
        self.load_model_requested.emit(path)
