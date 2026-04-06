"""Main application window."""

from __future__ import annotations

import os
# Must be set before any matplotlib import (canvas.py imports pyplot at module level)
os.environ.setdefault("MPLBACKEND", "QtAgg")

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog,
    QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox,
    QHBoxLayout, QPushButton, QSplitter, QStatusBar,
    QTabWidget, QVBoxLayout, QWidget,
)

from acorn.core.dm4_loader import DM4Image, scan_folder
from acorn.core.contrast import ContrastParams
from acorn.core.annotations import (
    AnnotationStore, ArrowAnnotation, LineAnnotation, CircleAnnotation,
    RectangleAnnotation, TextAnnotation, ScalebarAnnotation,
    DistanceMeasurement, AngleMeasurement, ROIAnnotation,
)
from acorn.core.measurements import MeasurementEngine
from acorn.render.scalebar import nice_scalebar_nm

from acorn.gui.canvas_widget import CanvasWidget
from acorn.gui.contrast_panel import ContrastPanel
from acorn.gui.annotation_panel import AnnotationPanel
from acorn.gui.measurement_panel import MeasurementPanel
from acorn.gui.export_panel import ExportPanel
from acorn.gui.sam_panel import SAMPanel
from acorn.gui.yolo_panel import YOLOPanel
from acorn.gui.unet_panel import UNetPanel
from acorn.gui.train_panel import TrainPanel


# ── folder file-picker dialog ─────────────────────────────────────────────────

class FolderPickerDialog(QDialog):
    """
    Shows all supported files found in a folder and lets the user
    choose which ones to open.  Includes Select All / Deselect All helpers.
    """

    def __init__(self, files: list[Path], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select files to open")
        self.resize(520, 400)

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        layout.addWidget(QLabel(f"Found {len(files)} supported file(s).  Select which to open:"))

        self._list = QListWidget()
        self._list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        for p in files:
            item = QListWidgetItem(p.name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setCheckState(Qt.CheckState.Checked)
            self._list.addItem(item)
        layout.addWidget(self._list, 1)

        btn_row = QHBoxLayout()
        sel_all = QPushButton("Select All")
        sel_all.clicked.connect(self._select_all)
        desel_all = QPushButton("Deselect All")
        desel_all.clicked.connect(self._deselect_all)
        btn_row.addWidget(sel_all)
        btn_row.addWidget(desel_all)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._list.count()):
            self._list.item(i).setCheckState(Qt.CheckState.Unchecked)

    def selected_paths(self) -> list[Path]:
        result = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result


# ── module-level helpers ──────────────────────────────────────────────────────

def _sample_path(pts: list, spacing: float = 20.0) -> list:
    """Return a subset of pts sampled at roughly `spacing`-pixel intervals."""
    import math
    if len(pts) < 2:
        return list(pts)
    result = [pts[0]]
    accumulated = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        accumulated += math.sqrt(dx * dx + dy * dy)
        if accumulated >= spacing:
            result.append(pts[i])
            accumulated = 0.0
    if result[-1] != pts[-1]:
        result.append(pts[-1])
    return result


# ── background loader thread ──────────────────────────────────────────────────

class LoadThread(QThread):
    """Validate image paths in a background thread; actual loading is lazy."""

    progress = pyqtSignal(int, int, str)            # n_done, n_total, filename
    finished = pyqtSignal(list, list)               # valid_paths, error_pairs

    def __init__(self, paths: list[Path], parent=None):
        super().__init__(parent)
        self._paths = paths

    def run(self) -> None:
        valid = []
        errors = []
        n = len(self._paths)
        for i, p in enumerate(self._paths):
            self.progress.emit(i, n, p.name)
            if p.exists():
                valid.append(p)
            else:
                errors.append((p, "File not found"))
        self.finished.emit(valid, errors)


# ── background training-export thread ────────────────────────────────────────

class TrainingThread(QThread):
    """Run training export on a background thread so the GUI stays responsive."""

    progress     = pyqtSignal(str)       # status message
    progress_int = pyqtSignal(int, int)  # (current_tile, total_tiles)
    finished     = pyqtSignal(dict)      # result summary dict
    error        = pyqtSignal(str)       # error message string

    def __init__(self, dataset_dir, dm4img, store_snapshot, params, config, parent=None):
        super().__init__(parent)
        self._dataset_dir    = dataset_dir
        self._dm4img         = dm4img
        self._store_snapshot = store_snapshot
        self._params         = params
        self._config         = config

    def run(self) -> None:
        try:
            from acorn.export.training_exporter import add_image, TrainingConfig
            from acorn.core.annotations import AnnotationStore
            # Reconstruct a store from the snapshot
            store = AnnotationStore()
            store.replace_all(self._store_snapshot)

            def _progress_cb(current: int, total: int) -> None:
                self.progress.emit(f"Exporting tile {current}/{total}…")
                self.progress_int.emit(current, total)

            result = add_image(
                self._dataset_dir, self._dm4img, store, self._params, self._config,
                progress_callback=_progress_cb,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── batch training export thread ──────────────────────────────────────────────

class BatchExportThread(QThread):
    """Process a queue of annotated images into a training dataset sequentially."""

    image_status    = pyqtSignal(str)        # human-readable status per image
    image_progress  = pyqtSignal(int, int)  # (current_image, total_images) overall
    tile_progress   = pyqtSignal(int, int)  # (current_tile, total_tiles) within image
    item_done       = pyqtSignal(int, str)  # (item_idx, stem) when one image finishes
    finished        = pyqtSignal(list)      # list of result dicts (one per image)
    error           = pyqtSignal(int, str)  # (item_idx, error message) — continues

    def __init__(self, items: list[dict], dataset_dir: str, config, parent=None):
        super().__init__(parent)
        self._items       = items        # list of {dm4img, store_snapshot, params, stem}
        self._dataset_dir = dataset_dir
        self._config      = config

    def run(self) -> None:
        from acorn.export.training_exporter import add_image
        from acorn.core.annotations import AnnotationStore

        results: list[dict] = []
        n = len(self._items)

        for i, item in enumerate(self._items):
            stem = item["stem"]
            self.image_progress.emit(i + 1, n)
            self.image_status.emit(f"[{i + 1}/{n}] {stem} — preparing…")

            # Negative images imported by path are loaded on demand here
            dm4img = item["dm4img"]
            if dm4img is None:
                try:
                    dm4img = DM4Image.from_file(Path(item["path"]))
                except Exception as exc:
                    self.error.emit(i, f"{stem}: could not load image — {exc}")
                    continue

            store = AnnotationStore()
            store.replace_all(item["store_snapshot"])

            # Unannotated images (negatives) must never skip empty tiles —
            # that would produce zero output. Override skip_empty_tiles for them.
            n_rois = sum(1 for a in store if getattr(a, "type", None) == "roi")
            if n_rois == 0 and self._config.skip_empty_tiles:
                from dataclasses import replace as _dc_replace
                cfg = _dc_replace(self._config, skip_empty_tiles=False)
                self.image_status.emit(
                    f"[{i + 1}/{n}] {stem} — no annotations, exporting as negative tiles…"
                )
            else:
                cfg = self._config

            def _cb(current: int, total: int, _s=stem, _i=i, _n=n) -> None:
                self.image_status.emit(f"[{_i + 1}/{_n}] {_s} — tile {current}/{total}")
                self.tile_progress.emit(current, total)

            try:
                result = add_image(
                    self._dataset_dir,
                    dm4img,
                    store,
                    item["params"],
                    cfg,
                    progress_callback=_cb,
                )
                results.append(result)
                self.item_done.emit(i, stem)
            except Exception as exc:
                self.error.emit(i, f"{stem}: {exc}")

        self.finished.emit(results)


# ── background SAM thread ─────────────────────────────────────────────────────

class SAMThread(QThread):
    """Run any SAM inference or model-load call on a background thread."""

    finished = pyqtSignal(object)   # result (type depends on task)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)      # intermediate status messages (e.g. download progress)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            self.error.emit(str(exc))


# ── background image-load thread ──────────────────────────────────────────────

class ImageLoadThread(QThread):
    """Load a DM4Image and pre-compute contrast normalisation in a background thread.

    Both the file I/O (DM4Image.from_file) and the contrast calculation
    (apply_contrast, which can involve a slow bandpass FFT) happen off the
    main thread so the GUI stays responsive throughout.
    """

    finished = pyqtSignal(int, object, object)  # (idx, DM4Image, norm_array)
    error    = pyqtSignal(int, str)             # (idx, error message)

    def __init__(self, idx: int, path: Path, contrast_params, parent=None):
        super().__init__(parent)
        self._idx     = idx
        self._path    = path
        self._params  = contrast_params

    def run(self) -> None:
        try:
            from acorn.core.contrast import apply_contrast
            from acorn.render.canvas import _DISPLAY_MAX_DIM
            img = DM4Image.from_file(self._path)

            # Pre-downsample large images before contrast computation.
            # The display never uses more than _DISPLAY_MAX_DIM px per side,
            # so computing contrast on the full-res raw is wasted work.
            # img.raw stays full-resolution for all analysis and export.
            raw = img.raw
            h, w = raw.shape[:2]
            step = max(1, (max(h, w) + _DISPLAY_MAX_DIM - 1) // _DISPLAY_MAX_DIM)
            raw_display = raw[::step, ::step] if step > 1 else raw

            norm = apply_contrast(raw_display, self._params)
            self.finished.emit(self._idx, img, norm)
        except Exception as exc:
            self.error.emit(self._idx, str(exc))


# ── main window ───────────────────────────────────────────────────────────────

class _PngMaskMapDialog(QDialog):
    """Dialog for assigning label names to colors found in a PNG mask."""

    def __init__(self, colors: list, pixels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Map Mask Colors to Labels")
        self.setMinimumWidth(400)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Assign a label name to each color region found in the mask.\n"
            "Leave blank to skip that color."
        ))

        import numpy as np
        self._rows: list[tuple] = []   # (color_tuple, QLineEdit)

        for color in colors:
            r, g, b = color
            count = int(np.all(pixels == np.array([r, g, b], dtype=np.uint8), axis=1).sum())
            row = QHBoxLayout()
            swatch = QLabel()
            swatch.setFixedSize(24, 24)
            swatch.setStyleSheet(
                f"background: rgb({r},{g},{b}); border: 1px solid #555; border-radius: 3px;"
            )
            count_lbl = QLabel(f"{count:,} px")
            count_lbl.setFixedWidth(80)
            edit = QLineEdit()
            edit.setPlaceholderText("label name…")
            row.addWidget(swatch)
            row.addWidget(count_lbl)
            row.addWidget(edit, 1)
            layout.addLayout(row)
            self._rows.append((color, edit))

        btns = QHBoxLayout()
        ok_btn = QPushButton("Import")
        ok_btn.setStyleSheet("background:#27ae60;color:white;font-weight:bold;")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def label_map(self) -> dict:
        return {color: edit.text() for color, edit in self._rows}


class MainWindow(QMainWindow):
    """
    ACORN main window.

    Layout
    ------
    QSplitter (horizontal):
        CanvasWidget (expanding) | ControlPanel (QTabWidget, 300px)
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ACORN")
        self.resize(1380, 880)
        self.setStyleSheet("""
            * {
                font-family: "Ubuntu", "Noto Sans", "Segoe UI", "Helvetica Neue", sans-serif;
                font-size: 12px;
            }
            QMainWindow, QDialog {
                background-color: #1e1e2e;
            }
            QWidget {
                color: #cdd6f4;
            }
            QSplitter {
                background: #1e1e2e;
            }
            QSplitter::handle {
                background: #313244;
                width: 2px;
                height: 2px;
            }
            QTabWidget::pane {
                background: #1e1e2e;
                border: 1px solid #45475a;
                border-top: none;
            }
            QTabBar {
                background: #1e1e2e;
            }
            QTabBar::tab {
                background: #313244;
                color: #a6adc8;
                padding: 6px 13px;
                min-width: 58px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #7c3aed;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #45475a;
                color: #cdd6f4;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 4px 4px 4px;
                color: #89b4fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px;
            }
            QLabel {
                color: #cdd6f4;
            }
            QPushButton {
                background: #313244;
                color: #cdd6f4;
                padding: 4px 12px;
                min-height: 26px;
                border: 1px solid #45475a;
                border-radius: 5px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #45475a;
                border-color: #89b4fa;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #181825;
            }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background: #313244;
                color: #cdd6f4;
                padding: 3px 6px;
                min-height: 24px;
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
                border-color: #a6e3a1;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 4px;
            }
            QRadioButton {
                color: #cdd6f4;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid #585b70;
                background: #313244;
            }
            QRadioButton::indicator:checked {
                background: #cba6f7;
                border-color: #cba6f7;
            }
            QRadioButton::indicator:hover {
                border-color: #89b4fa;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 2px solid #585b70;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                background: #a6e3a1;
                border-color: #a6e3a1;
            }
            QCheckBox::indicator:hover {
                border-color: #89b4fa;
            }
            QSlider::groove:horizontal {
                height: 5px;
                border-radius: 2px;
                background: #45475a;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #89b4fa;
            }
            QSlider::sub-page:horizontal {
                background: #7c3aed;
                border-radius: 2px;
            }
            QScrollBar:vertical {
                background: #1e1e2e;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #585b70;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #1e1e2e;
                height: 10px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #585b70;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }
            QMenuBar {
                background: #181825;
                color: #cdd6f4;
                border-bottom: 1px solid #45475a;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #45475a;
            }
            QMenu {
                background: #1e1e2e;
                color: #cdd6f4;
                border: 1px solid #45475a;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #313244;
                color: #cba6f7;
            }
            QMenu::separator {
                height: 1px;
                background: #45475a;
                margin: 4px 8px;
            }
            QStatusBar {
                background: #181825;
                color: #a6adc8;
                border-top: 1px solid #45475a;
                font-size: 11px;
            }
            QMessageBox {
                background: #1e1e2e;
            }
            QAbstractItemView {
                background: #313244;
                color: #cdd6f4;
                border: 1px solid #45475a;
                selection-background-color: #7c3aed;
                selection-color: #ffffff;
                alternate-background-color: #1e1e2e;
            }
            QHeaderView::section {
                background: #1e1e2e;
                color: #89b4fa;
                border: none;
                border-bottom: 1px solid #45475a;
                padding: 4px 6px;
                font-weight: bold;
            }
            QToolBar {
                background: #181825;
                border: none;
                spacing: 3px;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 3px;
                color: #cdd6f4;
            }
            QToolButton:hover {
                background: #45475a;
            }
            QToolButton:checked, QToolButton:pressed {
                background: #313244;
            }
            QDockWidget {
                color: #cdd6f4;
                font-weight: bold;
            }
            QDockWidget::title {
                background: #181825;
                padding: 4px 6px;
                border-bottom: 1px solid #45475a;
            }
            QListWidget {
                background: #1e1e2e;
                border: none;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #313244;
            }
            QListWidget::item:selected {
                background: #7c3aed;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background: #313244;
            }
        """)

        # ── application state ─────────────────────────────────────────────────
        self._image_paths: list[Path] = []          # all file paths (no data held)
        self._image_cache: dict[int, DM4Image] = {} # at most _MAX_CACHE loaded at once
        self._MAX_CACHE = 3
        self._img_idx: int = -1          # -1 = no image loaded yet
        self._click_buffer: list[tuple[float, float]] = []
        self._engine: MeasurementEngine = MeasurementEngine(pixel_size=1.0)
        self._px_overrides: dict[int, float] = {}  # manually set pixel size per image index
        self._last_distance_px: float = 0.0        # last measured distance in pixels
        self._contrast_states: dict[int, ContrastParams] = {}
        self._ann_states: dict[int, list] = {}   # per-image annotation snapshots
        self._image_load_thread: Optional[ImageLoadThread] = None  # active background load
        self._pending_contrast: Optional[ContrastParams] = None
        self._contrast_timer = QTimer()
        self._contrast_timer.setSingleShot(True)
        self._contrast_timer.setInterval(150)   # ms to wait after last slider move
        self._contrast_timer.timeout.connect(self._apply_contrast_debounced)

        self._autosave_timer = QTimer()
        self._autosave_timer.setSingleShot(True)
        self._autosave_timer.setInterval(2000)  # 2 s debounce after last annotation change
        self._autosave_timer.timeout.connect(self._do_autosave)

        # SAM state
        self._sam_predictor = None      # SAMPredictor, loaded on demand
        self._sam_thread: Optional[SAMThread] = None   # active background thread
        self._sam_mode: Optional[str] = None   # "pos_point" | "neg_point" | "box" | "exclude_zone" | "crop_region"
        self._sam_box_click: Optional[tuple[float, float]] = None
        self._pending_sam_masks: list = []
        # accumulated point-prompt state — cleared by Commit & New / Accept / Reject
        self._sam_prompt_points: list = []     # [(x, y), ...] always in full image coords
        self._sam_prompt_labels: list = []     # [1|0, ...]
        self._sam_current_preview = None       # ROIAnnotation currently shown as preview
        self._sam_point_artists: list = []     # matplotlib dot artists for visual feedback
        # region state — current image (live)
        self._sam_exclude_zone: Optional[tuple] = None   # (x0, y0, x1, y1) full image px
        self._sam_crop_region: Optional[tuple] = None    # (x0, y0, x1, y1) full image px
        # region state — persisted per image index
        self._sam_exclude_zones: dict[int, tuple] = {}
        self._sam_crop_regions_saved: dict[int, tuple] = {}

        # YOLO state
        self._yolo_predictor = None     # YOLOPredictor, loaded on demand
        self._yolo_thread: Optional[SAMThread] = None
        self._last_yolo_detections: list = []   # kept to pipe to SAM
        self._pending_yolo_anns: list = []

        # UNet state
        self._unet_predictor = None     # UNetPredictor, loaded on demand
        self._unet_thread: Optional[SAMThread] = None
        self._pending_unet_masks: list = []

        # ── central splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # canvas
        self._canvas_widget = CanvasWidget()
        self._canvas_widget.canvas.store.on_change(self._on_store_changed_autosave)
        splitter.addWidget(self._canvas_widget)

        # right control panel
        control = QTabWidget()
        control.setMinimumWidth(320)

        self._contrast_panel = ContrastPanel()
        self._ann_panel = AnnotationPanel()
        self._meas_panel = MeasurementPanel()
        self._export_panel = ExportPanel()
        self._sam_panel = SAMPanel()
        self._yolo_panel = YOLOPanel()
        self._unet_panel = UNetPanel()
        self._train_panel    = TrainPanel()

        control.addTab(self._contrast_panel,  "Contrast")
        control.addTab(self._ann_panel,       "Annotate")
        control.addTab(self._meas_panel,      "Measure")
        control.addTab(self._export_panel,    "Export")
        control.addTab(self._sam_panel,       "SAM")
        control.addTab(self._yolo_panel,      "YOLO")
        control.addTab(self._unet_panel,      "UNet")
        control.addTab(self._train_panel,     "Train")

        # ── plugin tabs ────────────────────────────────────────────────────────────
        from acorn.gui.context import AcornContext
        from acorn.plugin_loader import discover_plugins
        self._context = AcornContext(self)
        self._plugins = discover_plugins(self._context)
        for plugin in self._plugins:
            try:
                panel = plugin.create_panel()
                control.addTab(panel, plugin.TAB_LABEL)
            except Exception as _plugin_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Plugin %s failed to create panel: %s", plugin.PLUGIN_ID, _plugin_exc
                )

        splitter.addWidget(control)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([9999, 400])   # initial: canvas gets all extra, panel starts at 400px

        self.setCentralWidget(splitter)

        # ── status bar ────────────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

        # Permanent pixel-size widget — always visible on the right of the status bar
        self._px_btn = QPushButton("px: —")
        self._px_btn.setFlat(True)
        self._px_btn.setToolTip("Click to set pixel size manually")
        self._px_btn.setStyleSheet(
            "QPushButton { color: #aaaaaa; font-size: 11px; padding: 0 6px; border: none; }"
            "QPushButton:hover { color: #ffffff; text-decoration: underline; }"
        )
        self._px_btn.clicked.connect(self._on_edit_pixel_size)
        self._statusbar.addPermanentWidget(self._px_btn)

        # ── image list dock ───────────────────────────────────────────────────
        self._image_list = QListWidget()
        self._image_list.setFixedWidth(220)
        self._image_list.currentRowChanged.connect(self._on_image_list_select)
        self._image_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._image_list.customContextMenuRequested.connect(self._on_image_list_context_menu)
        dock = QDockWidget("Images", self)
        dock.setWidget(self._image_list)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable |
            QDockWidget.DockWidgetFeature.DockWidgetClosable,
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        self._image_list_dock = dock

        # ── menus ─────────────────────────────────────────────────────────────
        self._build_menus()
        # Let plugins register menu items after core menus are built
        for plugin in self._plugins:
            try:
                plugin.setup_menus(self.menuBar())
            except Exception as _plugin_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Plugin %s menu setup failed: %s", plugin.PLUGIN_ID, _plugin_exc
                )

        # ── signals ───────────────────────────────────────────────────────────
        self._contrast_panel.contrast_changed.connect(self._on_contrast_changed)
        self._ann_panel.undo_requested.connect(self._on_undo)
        self._ann_panel.clear_requested.connect(self._on_clear_annotations)
        self._ann_panel.clear_profiles_requested.connect(
            self._canvas_widget.clear_line_profiles
        )
        self._canvas_widget.line_profile_preview.connect(self._on_line_profile_preview)
        self._live_profile_dlg = None
        self._ann_panel.delete_selected_requested.connect(self._on_delete_selected)
        self._ann_panel.relabel_requested.connect(self._on_relabel_selected)
        self._ann_panel.tool_changed.connect(self._canvas_widget.set_tool)
        self._ann_panel.tool_changed.connect(self._on_tool_changed)
        self._canvas_widget.click_event.connect(self._on_canvas_click)
        self._canvas_widget.drag_commit.connect(self._on_drag_commit)
        self._canvas_widget.freehand_commit.connect(self._on_freehand_commit)
        self._canvas_widget.sam_box_commit.connect(self._on_sam_box_drag)
        self._canvas_widget.prev_requested.connect(self._on_prev)
        self._canvas_widget.next_requested.connect(self._on_next)
        self._canvas_widget.annotation_selected.connect(self._on_annotation_selected)
        self._canvas_widget.annotation_selected.connect(self._ann_panel.set_selected_annotation)
        self._canvas_widget.annotation_delete_requested.connect(self._on_annotation_delete)
        self._export_panel.export_requested.connect(self._on_export)
        self._export_panel.raw_export_requested.connect(self._on_export_raw)
        self._export_panel.mask_export_requested.connect(self._on_export_masks)
        self._export_panel.training_export_requested.connect(self._on_training_export)
        self._export_panel.queue_requested.connect(self._on_queue_image)
        self._export_panel.batch_export_requested.connect(self._on_batch_export)
        self._export_panel.clear_queue_requested.connect(self._on_clear_queue)
        self._export_panel.import_negatives_requested.connect(self._on_import_negatives)
        self._export_panel.finalize_requested.connect(self._on_finalize_dataset)
        self._export_panel.quality_check_requested.connect(self._on_check_quality)
        self._export_panel.hub_push_requested.connect(self._on_push_hub)
        self._export_panel.display_export_requested.connect(self._on_display_export)
        self._export_queue: list[dict] = []
        self._batch_export_thread: Optional[BatchExportThread] = None

        # SAM panel signals
        self._sam_panel.load_model_requested.connect(self._on_sam_load_model)
        self._sam_panel.auto_segment_requested.connect(self._on_sam_auto_segment)
        self._sam_panel.point_prompt_mode_set.connect(self._on_sam_point_mode)
        self._sam_panel.box_prompt_mode_set.connect(self._on_sam_box_mode)
        self._sam_panel.scribble_mode_set.connect(self._on_sam_scribble_mode)
        self._sam_panel.prompt_mode_cleared.connect(lambda: setattr(self, "_sam_mode", None))
        self._sam_panel.commit_new_requested.connect(self._on_sam_commit_new)
        self._sam_panel.undo_point_requested.connect(self._on_sam_undo_point)
        self._sam_panel.clear_points_requested.connect(self._on_sam_clear_points)
        self._sam_panel.accept_all_requested.connect(self._on_sam_accept)
        self._sam_panel.reject_all_requested.connect(self._on_sam_reject)
        self._sam_panel.accept_and_queue_requested.connect(self._on_sam_accept_and_queue)
        self._sam_panel.exclude_zone_mode_set.connect(self._on_sam_exclude_mode)
        self._sam_panel.exclude_zone_cleared.connect(self._on_sam_exclude_clear)
        self._sam_panel.crop_region_mode_set.connect(self._on_sam_crop_mode)
        self._sam_panel.crop_region_cleared.connect(self._on_sam_crop_clear)

        # YOLO panel signals
        self._yolo_panel.load_model_requested.connect(self._on_yolo_load_model)
        self._yolo_panel.detect_requested.connect(self._on_yolo_detect)
        self._yolo_panel.detect_seg_requested.connect(self._on_yolo_detect_seg)
        self._yolo_panel.pipe_to_sam_requested.connect(self._on_yolo_pipe_to_sam)
        self._yolo_panel.accept_all_requested.connect(self._on_yolo_accept)
        self._yolo_panel.reject_all_requested.connect(self._on_yolo_reject)

        # UNet panel signals
        self._unet_panel.load_model_requested.connect(self._on_unet_load_model)
        self._unet_panel.segment_requested.connect(self._on_unet_segment)
        self._unet_panel.accept_all_requested.connect(self._on_unet_accept)
        self._unet_panel.reject_all_requested.connect(self._on_unet_reject)

        # Train panel signals
        self._train_panel.train_requested.connect(self._on_train_requested)
        self._train_panel.cancel_requested.connect(self._on_train_cancel)
        self._train_panel.load_yolo_requested.connect(self._on_train_load_yolo)
        self._train_panel.load_unet_requested.connect(self._on_train_load_unet)
        self._train_thread: Optional[SAMThread] = None
        self._train_proc = None

        # Intercept key events from all child widgets (e.g. matplotlib canvas
        # consumes key events and never lets them reach MainWindow.keyPressEvent)
        QApplication.instance().installEventFilter(self)

    # ── menu setup ────────────────────────────────────────────────────────────

    def _build_menus(self) -> None:
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("File")
        open_a = file_menu.addAction("Open File(s)…")
        open_a.setShortcut("Ctrl+O")
        open_a.triggered.connect(self._open_files_dialog)

        open_dir_a = file_menu.addAction("Open Folder…")
        open_dir_a.setShortcut("Ctrl+Shift+O")
        open_dir_a.triggered.connect(self._open_folder_dialog)

        file_menu.addSeparator()
        save_sess_a = file_menu.addAction("Save Session…")
        save_sess_a.setShortcut("Ctrl+S")
        save_sess_a.triggered.connect(self._save_session)

        load_sess_a = file_menu.addAction("Load Session…")
        load_sess_a.setShortcut("Ctrl+L")
        load_sess_a.triggered.connect(self._load_session)

        file_menu.addSeparator()
        star_a = file_menu.addAction("Import Particle Picks (.star)…")
        star_a.setShortcut("Ctrl+I")
        star_a.triggered.connect(self._import_star)

        import_ann_menu = file_menu.addMenu("Import Annotations")
        png_mask_a = import_ann_menu.addAction("PNG / TIFF Mask…")
        png_mask_a.setToolTip("Import a colour or binary mask image as ROI annotations")
        png_mask_a.triggered.connect(self._import_png_mask)
        imagej_a = import_ann_menu.addAction("ImageJ ROI Set (.zip)…")
        imagej_a.setToolTip("Import a .zip of ImageJ/FIJI .roi files")
        imagej_a.triggered.connect(self._import_imagej_roi)

        file_menu.addSeparator()
        quit_a = file_menu.addAction("Quit")
        quit_a.setShortcut("Ctrl+Q")
        quit_a.triggered.connect(QApplication.quit)

        # View
        view_menu = mb.addMenu("View")
        toggle_list_a = view_menu.addAction("Show Image List")
        toggle_list_a.setCheckable(True)
        toggle_list_a.setChecked(True)
        toggle_list_a.setShortcut("Ctrl+Shift+L")
        toggle_list_a.toggled.connect(self._image_list_dock.setVisible)
        self._image_list_dock.visibilityChanged.connect(toggle_list_a.setChecked)

        # Help
        help_menu = mb.addMenu("Help")
        about_a = help_menu.addAction("About ACORN")
        about_a.triggered.connect(self._show_about)

    # ── file opening ──────────────────────────────────────────────────────────

    def _open_files_dialog(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open image file(s)", "",
            "All supported (*.dm4 *.tif *.tiff *.mrc *.mrcs *.png *.jpg *.jpeg);;"
            "DM4 (*.dm4);;"
            "TIFF (*.tif *.tiff);;"
            "MRC (*.mrc *.mrcs);;"
            "Images (*.png *.jpg *.jpeg);;"
            "All files (*)",
            options=QFileDialog.Option.DontUseNativeDialog,
        )
        if paths:
            self.open_files([Path(p) for p in paths])

    def _pick_directory(self, title: str = "Select folder") -> str:
        """
        Directory picker that avoids hanging on NFS/slow mounts at /.
        Shows a text field for pasting a path directly, plus a Browse button
        that starts from the home directory.
        """
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(520)
        layout = QVBoxLayout(dlg)

        layout.addWidget(QLabel("Paste or type the full path, or use Browse:"))

        row = QHBoxLayout()
        edit = QLineEdit(str(Path.home()))
        browse = QPushButton("Browse…")
        row.addWidget(edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        def _browse():
            start = edit.text().strip() or str(Path.home())
            if not Path(start).is_dir():
                start = str(Path.home())
            d = QFileDialog.getExistingDirectory(
                dlg, title, start,
                QFileDialog.Option.DontUseNativeDialog,
            )
            if d:
                edit.setText(d)

        browse.clicked.connect(_browse)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return ""
        return edit.text().strip()

    def _open_folder_dialog(self) -> None:
        d = self._pick_directory("Open folder of images")
        if not d:
            return
        files = scan_folder(d)
        if not files:
            QMessageBox.information(
                self, "No supported files",
                f"No supported image files found in:\n{d}\n\n"
                "Supported: .dm4, .tif/.tiff, .mrc/.mrcs, .png, .jpg/.jpeg"
            )
            return
        dlg = FolderPickerDialog(files, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            chosen = dlg.selected_paths()
            if chosen:
                self.open_files(chosen)

    # ── auto-save ──────────────────────────────────────────────────────────────

    def _autosave_path(self, idx: int) -> Optional[Path]:
        """Return the sidecar auto-save path for image at idx, or None if unavailable."""
        if 0 <= idx < len(self._image_paths):
            p = self._image_paths[idx]
            return p.parent / f".{p.stem}.acorn.json"
        return None

    def _on_store_changed_autosave(self, _items) -> None:
        """Called on every store change — arms the debounce timer."""
        if self._img_idx >= 0:
            self._autosave_timer.start()
        # Skip plugin notifications during image loading (canvas._loading guards renders;
        # plugins get a single image_loaded emit at the end of _finish_switch instead).
        if hasattr(self, "_context") and not self._canvas_widget.canvas._loading:
            store = self._canvas_widget.canvas.store
            self._context.annotations_changed.emit(store)

    def _do_autosave(self) -> None:
        """Write current annotations to the sidecar file (debounced)."""
        idx = self._img_idx
        path = self._autosave_path(idx)
        if path is None:
            return
        anns = list(self._canvas_widget.canvas.store)
        self._ann_states[idx] = anns
        try:
            ez = self._sam_exclude_zones.get(idx)
            cr = self._sam_crop_regions_saved.get(idx)
            data = {
                "version": 3,
                "annotations": [asdict(a) for a in anns],
                "pixel_size_nm": self._px_overrides.get(idx),
                "exclude_zone": list(ez) if ez else None,
                "crop_region": list(cr) if cr else None,
            }
            path.write_text(json.dumps(data))
        except OSError:
            pass  # NAS write failure — silently skip, in-memory state is preserved
        self._save_annotated_overlay(idx)

    def _save_annotated_overlay(self, idx: int) -> None:
        """Render the current image with annotations overlaid and save to annotated/ subfolder.

        Saves to: {image_parent}/annotated/{stem}_annotated.png
        Only runs when idx matches the currently displayed image (norm array is in memory).
        Silently skips if the norm image is unavailable or PIL is not installed.
        """
        if idx != self._img_idx:
            return
        norm = self._canvas_widget.canvas.norm_image
        if norm is None:
            return
        if idx < 0 or idx >= len(self._image_paths):
            return
        src_path = self._image_paths[idx]
        try:
            import numpy as np
            from PIL import Image as _PILImage, ImageDraw as _ImageDraw
        except ImportError:
            return

        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        if img8.ndim == 2:
            pil_img = _PILImage.fromarray(img8, "L").convert("RGB")
        else:
            pil_img = _PILImage.fromarray(img8).convert("RGB")

        draw = _ImageDraw.Draw(pil_img)

        def _hex_to_rgb(hex_color: str) -> tuple:
            h = hex_color.lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            try:
                return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))
            except ValueError:
                return (255, 140, 0)

        for ann in self._canvas_widget.canvas.store:
            color = _hex_to_rgb(getattr(ann, "color", "#FF8C00"))
            lw = max(1, int(getattr(ann, "linewidth", 2.0)))
            t = ann.type
            if t == "roi":
                if len(ann.vertices) >= 2:
                    pts = [tuple(v) for v in ann.vertices]
                    draw.line(pts + [pts[0]], fill=color, width=lw)
            elif t == "circle":
                x0, y0 = ann.cx - ann.r, ann.cy - ann.r
                x1, y1 = ann.cx + ann.r, ann.cy + ann.r
                draw.ellipse([x0, y0, x1, y1], outline=color, width=lw)
            elif t in ("line", "arrow", "distance", "angle"):
                if hasattr(ann, "p1") and hasattr(ann, "p2"):
                    draw.line([tuple(ann.p1), tuple(ann.p2)], fill=color, width=lw)
            elif t == "rectangle":
                draw.rectangle([ann.x0, ann.y0, ann.x1, ann.y1], outline=color, width=lw)

        try:
            out_dir = src_path.parent / "annotated"
            out_dir.mkdir(exist_ok=True)
            out_path = out_dir / f"{src_path.stem}_annotated.png"
            pil_img.save(str(out_path))
        except OSError:
            pass  # NAS write failure — silently skip

    def _autoload_sidecar(self, idx: int) -> Optional[tuple]:
        """Load sidecar file for idx. Returns (annotations, pixel_size_nm_or_None) or None."""
        path = self._autosave_path(idx)
        if path is None or not path.exists():
            return None
        try:
            raw = json.loads(path.read_text())
            from acorn.core.annotations import AnnotationStore
            if isinstance(raw, list):
                ann_data = raw
                px_nm = None
                ez = None
                cr = None
            else:
                ann_data = raw.get("annotations", [])
                px_nm = raw.get("pixel_size_nm")
                ez_raw = raw.get("exclude_zone")
                cr_raw = raw.get("crop_region")
                ez = tuple(ez_raw) if ez_raw else None
                cr = tuple(cr_raw) if cr_raw else None
            store = AnnotationStore.from_json(json.dumps(ann_data))
            return (list(store), px_nm, ez, cr)
        except Exception:
            return None

    def _save_session(self) -> None:
        """Save annotations for all loaded images to a JSON session file."""
        if not self._image_paths:
            self._statusbar.showMessage("No images loaded — nothing to save.")
            return
        # Snapshot the currently-displayed image's annotations before saving
        if 0 <= self._img_idx < len(self._image_paths):
            self._ann_states[self._img_idx] = list(
                self._canvas_widget.canvas.store
            )
        default_dir = str(self._image_paths[0].parent)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session",
            str(Path(default_dir) / "session.json"),
            "Session files (*.json);;All files (*)",
        )
        if not path:
            return
        data: dict = {"version": 2, "images": {}}
        for idx, img_path in enumerate(self._image_paths):
            anns = self._ann_states.get(idx, [])
            data["images"][str(img_path)] = {
                "annotations": [asdict(a) for a in anns],
                "pixel_size_nm": self._px_overrides.get(idx),
            }
        try:
            Path(path).write_text(json.dumps(data, indent=2))
            n = sum(1 for v in data["images"].values() if v)
            self._statusbar.showMessage(
                f"Session saved → {path}  ({n} image(s) with annotations)"
            )
        except OSError as e:
            QMessageBox.critical(self, "Save error", str(e))

    def _load_session(self) -> None:
        """Restore annotations from a previously saved JSON session file."""
        if not self._image_paths:
            QMessageBox.information(self, "No images loaded",
                                    "Open the same image files first, then load the session.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "Session files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return
        if data.get("version") not in (1, 2):
            QMessageBox.warning(self, "Session error",
                                "Unknown session format — expected version 1 or 2.")
            return
        images_data: dict = data.get("images", {})
        restored = 0
        for idx, img_path in enumerate(self._image_paths):
            key = str(img_path)
            if key in images_data:
                entry = images_data[key]
                if isinstance(entry, list):
                    ann_list = entry
                    px_nm = None
                else:
                    ann_list = entry.get("annotations", [])
                    px_nm = entry.get("pixel_size_nm")
                store = AnnotationStore.from_json(json.dumps(ann_list))
                self._ann_states[idx] = list(store)
                if px_nm is not None:
                    self._px_overrides[idx] = px_nm
                restored += 1
        # Refresh what's currently on screen
        saved = self._ann_states.get(self._img_idx)
        if saved is not None:
            self._canvas_widget.canvas.store.replace_all(saved)
        self._statusbar.showMessage(
            f"Session loaded: {restored} / {len(self._image_paths)} image(s) restored"
        )

    def _import_star(self) -> None:
        """Import particle picks from a RELION STAR file as ROI annotations."""
        if not self._image_paths:
            QMessageBox.information(self, "No image loaded",
                                    "Open an image first, then import particle picks.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Import STAR particle picks", "",
            "STAR files (*.star);;All files (*)"
        )
        if not path:
            return

        from PyQt6.QtWidgets import QInputDialog
        radius_px, ok = QInputDialog.getDouble(
            self, "Particle radius",
            "Circle radius (pixels):", 50.0, 1.0, 5000.0, 1
        )
        if not ok:
            return

        try:
            from acorn.core.star_loader import load_star_picks, picks_to_roi_annotations
            picks = load_star_picks(path)
            if not picks:
                QMessageBox.warning(self, "No picks found",
                                    "No coordinate columns found in the STAR file.\n"
                                    "Expected _rlnCoordinateX and _rlnCoordinateY.")
                return
            store = self._canvas_widget.canvas.store
            n = picks_to_roi_annotations(picks, store, radius_px=radius_px)
            self._statusbar.showMessage(
                f"Imported {n} particle picks from {Path(path).name} "
                f"(radius {radius_px:.0f} px)"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import error", str(exc))

    def open_files(self, paths: list[Path]) -> None:
        """Public entry point — also called from CLI view subcommand."""
        self._statusbar.showMessage(f"Loading {len(paths)} file(s)…")
        self._thread = LoadThread(paths, parent=self)
        self._thread.progress.connect(
            lambda n, tot, name: self._statusbar.showMessage(f"Loading [{n+1}/{tot}] {name}…")
        )
        self._thread.finished.connect(self._on_load_finished)
        self._thread.start()

    def _on_load_finished(self, paths: list[Path], errors: list) -> None:
        if errors:
            msgs = "\n".join(f"{p.name}: {e}" for p, e in errors)
            QMessageBox.warning(self, "Load errors", f"{len(errors)} file(s) failed:\n{msgs}")
        if not paths:
            self._statusbar.showMessage("No images loaded.")
            return
        self._image_paths = paths
        self._image_cache.clear()
        self._img_idx = -1           # sentinel so _switch_to doesn't save stale data
        self._contrast_states.clear()
        self._ann_states.clear()
        self._populate_image_list()
        self._switch_to(0)
        n = len(paths)
        self._statusbar.showMessage(f"{n} image(s) loaded. Use Prev/Next to navigate.")

    # ── image navigation ──────────────────────────────────────────────────────

    def _switch_to(self, idx: int) -> None:
        if not self._image_paths:
            return

        # Ignore if a load is already in progress for this same index.
        if self._image_load_thread is not None and self._image_load_thread.isRunning():
            return

        # ── save annotations for the image we're leaving ──────────────────────
        if 0 <= self._img_idx < len(self._image_paths):
            self._autosave_timer.stop()
            self._do_autosave()   # flush immediately before switching

        self._img_idx = idx
        self._click_buffer.clear()
        self._canvas_widget.reset_interaction()
        self._canvas_widget.set_nav_enabled(False)
        self._sync_image_list(idx)
        self._canvas_widget.update_nav_label(idx + 1, len(self._image_paths))

        if idx in self._image_cache:
            # Already cached — complete immediately without a thread.
            self._finish_switch(idx, self._image_cache[idx])
            return

        path = self._image_paths[idx]
        # Resolve contrast params now (before thread starts) so the
        # background thread can pre-compute the normalised image.
        contrast_params = self._contrast_states.get(idx) or self._contrast_panel.params()
        self._statusbar.showMessage(f"Loading {path.name}…")
        self._image_load_thread = ImageLoadThread(idx, path, contrast_params, parent=self)
        self._image_load_thread.finished.connect(self._on_image_loaded)
        self._image_load_thread.error.connect(self._on_image_load_error)
        self._image_load_thread.start()

    def _on_image_loaded(self, idx: int, img: DM4Image, norm) -> None:
        """Called on the main thread when ImageLoadThread finishes successfully."""
        if len(self._image_cache) >= self._MAX_CACHE:
            oldest = next(iter(self._image_cache))
            del self._image_cache[oldest]
        self._image_cache[idx] = img
        # Only render if this is still the current image (user may not have switched).
        if idx == self._img_idx:
            self._finish_switch(idx, img, precomputed_norm=norm)
        else:
            self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)

    def _on_image_load_error(self, idx: int, message: str) -> None:
        self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)
        self._statusbar.showMessage(f"Error loading image {idx}: {message}")

    def _finish_switch(self, idx: int, img: DM4Image, precomputed_norm=None) -> None:
        """Complete the image switch once the DM4Image is available."""
        # Reapply any manually set pixel size for this image (survives cache eviction)
        if idx in self._px_overrides:
            img.meta.pixel_size = self._px_overrides[idx]
        self._engine = MeasurementEngine(pixel_size=img.pixel_size)

        canvas = self._canvas_widget.canvas

        # Suppress intermediate _on_store_change renders while we set up state.
        # load_image() will do the single authoritative render at the end.
        canvas._loading = True
        try:
            canvas.store.clear()
            saved_anns = self._ann_states.get(idx)
            if saved_anns is None:
                sidecar = self._autoload_sidecar(idx)
                if sidecar is not None:
                    saved_anns, px_nm, ez, cr = sidecar
                    self._ann_states[idx] = saved_anns
                    if px_nm is not None and idx not in self._px_overrides:
                        self._px_overrides[idx] = px_nm
                        img.meta.pixel_size = px_nm
                        self._engine = MeasurementEngine(pixel_size=px_nm)
                    if ez is not None:
                        self._sam_exclude_zones[idx] = ez
                    if cr is not None:
                        self._sam_crop_regions_saved[idx] = cr
                    self._statusbar.showMessage(
                        f"Auto-saved annotations restored for {self._image_paths[idx].name}"
                    )
            if saved_anns is not None:
                canvas.store.replace_all(saved_anns)
        finally:
            canvas._loading = False

        saved_contrast = self._contrast_states.get(idx)
        if saved_contrast is not None:
            self._contrast_panel.set_params(saved_contrast)
        # Pass pre-computed norm so canvas.load_image skips apply_contrast on the main thread
        canvas.load_image(img, self._contrast_panel.params(), precomputed_norm=precomputed_norm)

        # Invalidate SAM embedding cache — new image, old embedding is stale
        if self._sam_predictor is not None:
            self._sam_predictor.invalidate_cache()
            self._sam_warmup_encode()

        # Restore per-image exclude/crop zones, or clear if none saved
        if self._sam_panel.keep_regions_across_images:
            # keep whatever is currently drawn — don't touch it
            pass
        else:
            ez = self._sam_exclude_zones.get(idx)
            if ez:
                self._sam_exclude_zone = ez
                self._canvas_widget.set_exclude_zone(*ez)
            else:
                self._sam_exclude_zone = None
                self._canvas_widget.clear_exclude_zone()
            cr = self._sam_crop_regions_saved.get(idx)
            if cr:
                self._sam_crop_region = cr
                self._canvas_widget.set_crop_region(*cr)
            else:
                self._sam_crop_region = None
                self._canvas_widget.clear_crop_region()

        # Sync panels
        ps = img.pixel_size
        w = img.shape[1] if img.shape else 512
        self._ann_panel.set_scalebar_nm(nice_scalebar_nm(ps, w))
        self._export_panel.set_defaults(img.filename, str(img.filepath.parent))

        self.setWindowTitle("ACORN")

        meta_parts = []
        if img.mag:
            meta_parts.append(f"{int(img.mag):,}×")
        if img.voltage_kV:
            meta_parts.append(f"{img.voltage_kV} kV")
        meta_parts.append(f"{img.shape[1]}×{img.shape[0]} px")
        self._statusbar.showMessage("  |  ".join(meta_parts))
        self._update_px_btn(ps, img.meta.pixel_size_from_header)

        self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)

        if hasattr(self, "_context"):
            self._context.image_loaded.emit(img)
            self._context.pixel_size_changed.emit(img.pixel_size)
            self._context.annotations_changed.emit(self._canvas_widget.canvas.store)

    # ── pixel size helpers ────────────────────────────────────────────────────

    def _update_px_btn(self, px_nm: float, from_header: bool) -> None:
        """Refresh the status-bar pixel size button label."""
        manually_set = self._img_idx in self._px_overrides
        if from_header:
            text  = f"{px_nm:.4f} nm/px  (header)"
            color = "#88ccff"
        elif manually_set:
            text  = f"{px_nm:.4f} nm/px  (manual)"
            color = "#ffcc66"
        else:
            text  = "px: not set  (click to enter)"
            color = "#ff6b6b"
        self._px_btn.setText(text)
        self._px_btn.setStyleSheet(
            f"QPushButton {{ color: {color}; font-size: 11px; padding: 0 6px; border: none; }}"
            f"QPushButton:hover {{ color: #ffffff; text-decoration: underline; }}"
        )

    def _on_edit_pixel_size(self) -> None:
        """Open a dialog for the user to enter a custom pixel size for this image."""
        img_idx = self._img_idx
        if img_idx < 0:
            return
        img = self._image_cache.get(img_idx)
        if img is None:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Set Pixel Size")
        layout = QVBoxLayout(dlg)

        source = "file header" if img.meta.pixel_size_from_header else "not in header / manual"
        note = QLabel(f"Image {img_idx + 1} of {len(self._image_paths)}  —  current source: {source}")
        note.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(note)

        # ── direct entry ──────────────────────────────────────────────────────
        entry_box = QGroupBox("Enter pixel size directly")
        entry_form = QFormLayout(entry_box)
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(0.0001, 1_000_000.0)
        spin.setSingleStep(0.001)
        spin.setValue(img.pixel_size if img.meta.pixel_size_from_header or img_idx in self._px_overrides else 0.0)
        unit_combo = QComboBox()
        unit_combo.addItems(["nm/px", "Angstrom/px", "um/px", "pm/px"])
        entry_form.addRow("Pixel size:", spin)
        entry_form.addRow("Unit:", unit_combo)
        layout.addWidget(entry_box)

        # ── calculate from scale bar ──────────────────────────────────────────
        cal_box = QGroupBox("Calculate from image scale bar")
        cal_box.setToolTip(
            "Use the Measure tool (Annotate tab) to draw a line along the printed\n"
            "scale bar in the image. When pixel size is unset the readout equals pixels."
        )
        cal_form = QFormLayout(cal_box)

        bar_px_spin = QDoubleSpinBox()
        bar_px_spin.setDecimals(2)
        bar_px_spin.setRange(1.0, 1_000_000.0)
        bar_px_spin.setSuffix(" px")
        bar_px_spin.setToolTip("Pixel length of the scale bar — read from the measure tool")
        if self._last_distance_px > 0:
            bar_px_spin.setValue(self._last_distance_px)

        use_last_btn = QPushButton(
            f"Use last measurement ({self._last_distance_px:.1f} px)"
            if self._last_distance_px > 0 else "No measurement yet"
        )
        use_last_btn.setEnabled(self._last_distance_px > 0)
        use_last_btn.setToolTip(
            "Fills the pixel length field with the most recent distance measurement.\n"
            "Use the Distance tool in the Annotate tab to measure the scale bar first."
        )
        use_last_btn.clicked.connect(lambda: bar_px_spin.setValue(self._last_distance_px))

        bar_len_spin = QDoubleSpinBox()
        bar_len_spin.setDecimals(4)
        bar_len_spin.setRange(0.0001, 1_000_000.0)
        bar_len_spin.setValue(100.0)

        bar_unit_combo = QComboBox()
        bar_unit_combo.addItems(["nm", "Angstrom", "um", "pm"])

        cal_btn = QPushButton("Calculate  →  fill above")
        cal_btn.setToolTip("Divides the known length by the pixel count to give nm/px")

        def _do_calc():
            px = bar_px_spin.value()
            length = bar_len_spin.value()
            _TO_NM_CAL = {"nm": 1.0, "Angstrom": 0.1, "um": 1000.0, "pm": 0.001}
            length_nm = length * _TO_NM_CAL[bar_unit_combo.currentText()]
            if px > 0:
                spin.setValue(length_nm / px)
                unit_combo.setCurrentText("nm/px")

        cal_btn.clicked.connect(_do_calc)
        cal_form.addRow("Scale bar length in image:", bar_px_spin)
        cal_form.addRow("", use_last_btn)
        cal_form.addRow("Known physical length:", bar_len_spin)
        cal_form.addRow("Unit:", bar_unit_combo)
        cal_form.addRow("", cal_btn)
        layout.addWidget(cal_box)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        value = spin.value()
        unit = unit_combo.currentText()
        _TO_NM = {"nm/px": 1.0, "Angstrom/px": 0.1, "um/px": 1000.0, "pm/px": 0.001}
        ps_nm = value * _TO_NM[unit]
        if ps_nm <= 0:
            return

        # Persist so this survives cache eviction and reload
        self._px_overrides[img_idx] = ps_nm
        self._autosave_timer.start()

        img.meta.pixel_size = ps_nm
        img.meta.pixel_size_from_header = False
        self._engine = MeasurementEngine(pixel_size=ps_nm)
        self._canvas_widget.canvas.set_pixel_size(ps_nm)
        w = img.shape[1] if img.shape else 512
        self._ann_panel.set_scalebar_nm(nice_scalebar_nm(ps_nm, w))
        self._update_px_btn(ps_nm, False)

    # ── image list ────────────────────────────────────────────────────────────

    def _populate_image_list(self) -> None:
        """Fill the image list dock with filenames."""
        self._image_list.blockSignals(True)
        self._image_list.clear()
        for p in self._image_paths:
            self._image_list.addItem(QListWidgetItem(p.name))
        self._image_list.blockSignals(False)

    def _sync_image_list(self, idx: int) -> None:
        """Highlight the row matching *idx* without triggering navigation."""
        self._image_list.blockSignals(True)
        self._image_list.setCurrentRow(idx)
        self._image_list.blockSignals(False)

    def _on_image_list_select(self, row: int) -> None:
        if row >= 0 and row != self._img_idx:
            self._switch_to(row)

    def _on_image_list_context_menu(self, pos) -> None:
        """Right-click menu on the image list — clear annotations for any image."""
        from PyQt6.QtWidgets import QMenu
        item = self._image_list.itemAt(pos)
        if item is None:
            return
        row = self._image_list.row(item)
        menu = QMenu(self)
        clear_act = menu.addAction(f"Clear all annotations for: {item.text()}")
        action = menu.exec(self._image_list.mapToGlobal(pos))
        if action == clear_act:
            reply = QMessageBox.question(
                self, "Clear annotations",
                f"Clear all annotations for {item.text()}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._ann_states[row] = []
                if row == self._img_idx:
                    self._canvas_widget.canvas.store.clear()
                self._autosave_timer.start()

    def _on_prev(self) -> None:
        if self._image_paths:
            self._switch_to((self._img_idx - 1) % len(self._image_paths))

    def _on_next(self) -> None:
        if self._image_paths:
            self._switch_to((self._img_idx + 1) % len(self._image_paths))

    # ── tool hints ────────────────────────────────────────────────────────────

    _TOOL_HINTS: dict[str, str] = {
        "none":         "Click to select  |  Drag selected to move  |  Delete to remove  |  Esc to deselect",
        "line":         "Drag to draw a line — hold Shift to snap to 0/45/90°",
        "arrow":        "Drag to draw an arrow — hold Shift to snap to 0/45/90°",
        "circle":       "Drag from centre outward to draw a circle",
        "rectangle":    "Drag to draw a rectangle — hold Shift to constrain to square",
        "freehand":     "Click and drag to draw freehand — release to commit as ROI",
        "text":         "Click to place text",
        "scalebar":     "Click to place a scale bar at that position",
        "distance":     "Click two points to measure distance",
        "line_profile": "Click two points to plot a line profile",
        "angle":        "Click three points: ray 1, vertex, ray 2",
        "roi":          "Click to add polygon vertices — right-click to close",
    }

    def _on_tool_changed(self, tool: str) -> None:
        self._click_buffer.clear()
        self._canvas_widget.clear_rubber_band()
        hint = self._TOOL_HINTS.get(tool, "")
        self._ann_panel.set_hint(hint)
        if hint:
            self._statusbar.showMessage(hint)

    def _on_annotation_selected(self, ann) -> None:
        if ann is None:
            self._statusbar.showMessage(self._TOOL_HINTS.get("none", ""))
            return
        t = ann.type
        if ann in self._pending_sam_masks or ann in self._pending_unet_masks:
            suffix = "  |  Press Delete to remove this mask, or Accept All to keep all"
        else:
            suffix = "  |  Drag to move, Delete to remove, Esc to deselect"
        if t in ("arrow", "line"):
            msg = f"Selected {t}: ({ann.p1[0]:.0f},{ann.p1[1]:.0f}) -> ({ann.p2[0]:.0f},{ann.p2[1]:.0f})"
        elif t == "circle":
            msg = f"Selected circle: centre ({ann.cx:.0f},{ann.cy:.0f}), r={ann.r:.0f}px"
        elif t == "rectangle":
            msg = f"Selected rectangle: ({ann.x0:.0f},{ann.y0:.0f}) – ({ann.x1:.0f},{ann.y1:.0f})"
        elif t == "text":
            msg = f"Selected text: \"{ann.label}\""
        elif t == "roi":
            msg = f"Selected ROI: {len(ann.vertices)} vertices, area {ann.area_nm2:.0f} nm²"
        elif t == "distance":
            if not getattr(ann, "calibrated", True):
                msg = f"Selected distance: {ann.distance_px:.1f} px (uncalibrated)"
            else:
                msg = f"Selected distance: {ann.distance_nm:.2f} nm"
        elif t == "scalebar":
            msg = f"Selected scale bar: {ann.nm:.0f} nm"
        elif t == "angle":
            msg = f"Selected angle: {ann.angle_deg:.1f}°"
        else:
            msg = f"Selected {t}"
        self._statusbar.showMessage(msg + suffix)

    def _on_annotation_delete(self, ann) -> None:
        self._canvas_widget.canvas.store.remove(ann)
        self._ann_panel.set_selected_annotation(None)
        # Keep pending SAM/UNet lists in sync if a pending mask is deleted individually
        for lst in (self._pending_sam_masks, self._pending_unet_masks):
            if ann in lst:
                lst.remove(ann)
                break

    def _on_delete_selected(self) -> None:
        """Delete whichever annotation is currently selected on the canvas."""
        renderer = self._canvas_widget.canvas.renderer
        if renderer is None:
            return
        ann = renderer.selected_annotation()
        if ann is not None:
            self._on_annotation_delete(ann)

    def _on_relabel_selected(self, new_label: str) -> None:
        """Rename the label of the currently selected annotation."""
        renderer = self._canvas_widget.canvas.renderer
        if renderer is None:
            return
        ann = renderer.selected_annotation()
        if ann is not None and hasattr(ann, "label"):
            ann.label = new_label
            self._canvas_widget.canvas.store._notify()
            self._statusbar.showMessage(f"Renamed to: {new_label}")

    # ── contrast ──────────────────────────────────────────────────────────────

    def _on_contrast_changed(self, params: ContrastParams) -> None:
        self._contrast_states[self._img_idx] = params
        self._pending_contrast = params
        self._contrast_timer.start()  # restarts the timer on every change

    def _apply_contrast_debounced(self) -> None:
        if self._pending_contrast is None:
            return
        canvas = self._canvas_widget.canvas
        if canvas.dm4 is not None:
            canvas.update_contrast(self._pending_contrast)

    # ── annotation & measurement click dispatch ────────────────────────────────

    def _on_canvas_click(self, x: float, y: float, button: int) -> None:
        # SAM prompt modes intercept canvas clicks only when tool is still "sam".
        # If the user switched to an annotation tool (e.g. select), let that through.
        if self._canvas_widget.current_tool == "sam":
            if self._sam_mode == "pos_point":
                self._sam_point_prompt(x, y, positive=True)
                return
            if self._sam_mode == "neg_point":
                self._sam_point_prompt(x, y, positive=False)
                return
            if self._sam_mode == "box":
                if self._sam_box_click is None:
                    self._sam_box_prompt_first_click(x, y)
                else:
                    self._sam_box_prompt_second_click(x, y)
                return

        tool = self._ann_panel.active_tool
        col  = self._ann_panel.color
        lw   = self._ann_panel.linewidth
        fs   = self._ann_panel.fontsize
        store = self._canvas_widget.canvas.store
        img   = self._canvas_widget.canvas.dm4
        norm  = self._canvas_widget.canvas.norm_image

        # Drag tools (line/arrow/circle/rectangle/freehand) are handled by
        # _on_drag_commit / _on_freehand_commit; ignore any stray click_events.
        if tool in ("none", "line", "arrow", "circle", "rectangle", "freehand", "line_profile"):
            return

        # ── single-click tools ────────────────────────────────────────────────
        if tool == "text":
            store.add(TextAnnotation(x=x, y=y, label=self._ann_panel.text_value,
                                     color=col, fontsize=fs))
            return

        if tool == "scalebar":
            ax = self._canvas_widget.canvas.ax
            w_ax = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
            h_ax = abs(ax.get_ylim()[0] - ax.get_ylim()[1])
            if w_ax > 1 and h_ax > 1:
                store.add(ScalebarAnnotation(
                    nm=self._ann_panel.scalebar_nm,
                    x_frac=x / w_ax, y_frac=y / h_ax,
                    color=col, linewidth=lw, fontsize=fs,
                ))
            return

        # ── two-click measurement tools ───────────────────────────────────────
        self._click_buffer.append((x, y))

        if tool == "distance":
            if len(self._click_buffer) == 1:
                self._canvas_widget.set_rubber_band_pts(list(self._click_buffer))
                self._statusbar.showMessage(
                    "Distance: click 1/2 placed — click endpoint"
                )
                return
            p1, p2 = self._click_buffer[0], self._click_buffer[1]
            self._click_buffer.clear()
            self._canvas_widget.clear_rubber_band()
            if tool == "distance":
                img = self._canvas_widget.canvas.dm4
                is_cal = (img is not None and
                          (img.meta.pixel_size_from_header or self._img_idx in self._px_overrides))
                m = self._engine.distance(p1, p2, color=col, calibrated=is_cal)
                store.add(m)
                self._meas_panel.add_distance(m)
                self._last_distance_px = m.distance_px
                if is_cal:
                    self._statusbar.showMessage(f"Distance: {m.distance_nm:.2f} nm")
                else:
                    self._statusbar.showMessage(
                        f"Distance: {m.distance_px:.1f} px  "
                        "(pixel size not set — click the px button to calibrate)"
                    )

        # ── three-click tools ─────────────────────────────────────────────────
        elif tool == "angle":
            if len(self._click_buffer) < 3:
                n = len(self._click_buffer)
                self._canvas_widget.set_rubber_band_pts(list(self._click_buffer))
                self._statusbar.showMessage(f"Angle: click {n}/3 placed")
                return
            p1, vertex, p2 = (self._click_buffer[0],
                               self._click_buffer[1],
                               self._click_buffer[2])
            self._click_buffer.clear()
            self._canvas_widget.clear_rubber_band()
            m = self._engine.angle(p1, vertex, p2, color=col)
            store.add(m)
            self._meas_panel.add_angle(m)
            self._statusbar.showMessage(f"Angle: {m.angle_deg:.2f}°")

        # ── polygon / ROI (click-to-add vertices, right-click to close) ───────
        elif tool == "roi":
            if button == 3 and len(self._click_buffer) >= 3:
                self._click_buffer.pop()  # drop the right-click coord
                vertices = list(self._click_buffer)
                self._click_buffer.clear()
                self._canvas_widget.clear_rubber_band()
                if norm is not None:
                    m = self._engine.roi_stats(vertices, norm, color=col)
                    m.label = self._ann_panel.roi_label
                    store.add(m)
                    self._meas_panel.add_roi(m)
                    area = m.area_nm2
                    label_str = f" [{m.label}]" if m.label else ""
                    self._statusbar.showMessage(
                        f"ROI{label_str} area: {area:.0f} nm²  mean: {m.stats.get('mean',0):.4f}"
                    )
            else:
                self._canvas_widget.set_rubber_band_pts(list(self._click_buffer))
                n = len(self._click_buffer)
                self._statusbar.showMessage(
                    f"ROI: {n} point(s) — right-click to close (min 3)"
                )

    # ── drag annotation commit ─────────────────────────────────────────────────

    def _on_drag_commit(
        self, tool: str,
        x1: float, y1: float, x2: float, y2: float,
        shift: bool,
    ) -> None:
        """Called when the user finishes a drag on line/arrow/circle/rectangle."""
        import math
        col   = self._ann_panel.color
        lw    = self._ann_panel.linewidth
        store = self._canvas_widget.canvas.store

        if tool == "arrow":
            store.add(ArrowAnnotation(p1=(x1, y1), p2=(x2, y2), color=col, linewidth=lw))
        elif tool == "line":
            store.add(LineAnnotation(p1=(x1, y1), p2=(x2, y2), color=col, linewidth=lw,
                                     linestyle=self._ann_panel.linestyle))
        elif tool == "circle":
            r = math.hypot(x2 - x1, y2 - y1)
            store.add(CircleAnnotation(cx=x1, cy=y1, r=r, color=col, linewidth=lw,
                                       linestyle=self._ann_panel.linestyle))
        elif tool == "rectangle":
            # local (x1,y1)=start, local (x2,y2)=end → dataclass (x0,y0,x1,y1)
            rx0, ry0, rx1, ry1 = x1, y1, x2, y2
            store.add(RectangleAnnotation(x0=rx0, y0=ry0, x1=rx1, y1=ry1,
                                          color=col, linewidth=lw,
                                          linestyle=self._ann_panel.linestyle))
        elif tool == "line_profile":
            norm = self._canvas_widget.canvas.norm_image
            if norm is not None:
                result = self._engine.line_profile((x1, y1), (x2, y2), norm)
                # Clear live preview, commit as permanent overlay
                self._canvas_widget._clear_live_profile()
                self._canvas_widget.add_line_profile_overlay(
                    (x1, y1), (x2, y2), result.intensities, color=col
                )
                self._statusbar.showMessage(
                    f"Line profile: {result.length_nm:.1f} nm  "
                    f"({len(result.intensities)} points)"
                )
                # Update the live dialog or open it if closed
                from acorn.gui.dialogs import LineProfileDialog
                if self._live_profile_dlg is None or not self._live_profile_dlg.isVisible():
                    self._live_profile_dlg = LineProfileDialog(result, parent=self)
                    self._live_profile_dlg.show()
                else:
                    self._live_profile_dlg.update(result)

    def _on_line_profile_preview(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Called on every drag-motion when line_profile tool is active."""
        norm = self._canvas_widget.canvas.norm_image
        if norm is None:
            return
        col = self._ann_panel.color
        result = self._engine.line_profile((x1, y1), (x2, y2), norm)
        self._canvas_widget.update_live_profile(
            (x1, y1), (x2, y2), result.intensities, color=col
        )
        from acorn.gui.dialogs import LineProfileDialog
        if self._live_profile_dlg is None or not self._live_profile_dlg.isVisible():
            self._live_profile_dlg = LineProfileDialog(result, parent=self)
            self._live_profile_dlg.show()
        else:
            self._live_profile_dlg.update(result)

    def _on_freehand_commit(self, pts: list) -> None:
        """Called when a freehand stroke is released."""
        if len(pts) < 2:
            return
        # SAM scribble mode: convert stroke to point prompts
        if (self._sam_mode == "scribble"
                and self._sam_predictor is not None
                and self._sam_predictor.is_loaded):
            self._on_sam_scribble_commit(pts)
            return
        if len(pts) < 3:
            return
        norm  = self._canvas_widget.canvas.norm_image
        col   = self._ann_panel.color
        lw    = self._ann_panel.linewidth
        store = self._canvas_widget.canvas.store
        if norm is not None:
            m = self._engine.roi_stats(pts, norm, color=col)
            m.linewidth = lw
            m.label = self._ann_panel.roi_label
            store.add(m)
            self._meas_panel.add_roi(m)
            area = m.area_nm2
            label_str = f" [{m.label}]" if m.label else ""
            self._statusbar.showMessage(
                f"Freehand ROI{label_str} area: {area:.0f} nm²  mean: {m.stats.get('mean',0):.4f}"
            )
        else:
            roi = ROIAnnotation(
                vertices=[(float(x), float(y)) for x, y in pts],
                area_nm2=0.0, stats={},
                color=col, linewidth=lw, label=self._ann_panel.roi_label,
            )
            store.add(roi)
            label_str = f" [{roi.label}]" if roi.label else ""
            self._statusbar.showMessage(
                f"Freehand ROI{label_str} added (no image — stats unavailable)"
            )

    def _on_sam_scribble_commit(self, pts: list) -> None:
        """Convert a freehand scribble stroke into SAM positive point prompts."""
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return

        sampled = _sample_path(pts, spacing=20.0)
        if not sampled:
            return

        point_label = self._sam_panel.point_label
        for x, y in sampled:
            self._sam_prompt_points.append((x, y))
            self._sam_prompt_labels.append(1)
            markers = self._canvas_widget.add_sam_point_marker(x, y, True, label="")
            self._sam_point_artists.append(markers)

        points_snap = list(self._sam_prompt_points)
        labels_snap = list(self._sam_prompt_labels)
        points_for_sam = [(px - ox, py - oy) for px, py in points_snap]
        n_pts = len(sampled)
        self._sam_panel.set_sam_status(f"Running SAM with {len(points_snap)} point(s)…")

        def _run():
            return self._sam_predictor.predict_points(img8, points_for_sam, labels=labels_snap)

        def _done(masks):
            if not masks:
                self._sam_panel.set_sam_status(
                    "No mask returned — try adding more strokes or a positive point."
                )
                return
            store = self._canvas_widget.canvas.store
            if self._sam_current_preview is not None:
                store.undo()
                if self._sam_current_preview in self._pending_sam_masks:
                    self._pending_sam_masks.remove(self._sam_current_preview)
                self._sam_current_preview = None
            vertices = self._sam_predictor.mask_to_polygon(masks[0])
            if ox != 0 or oy != 0:
                vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
            if len(vertices) >= 3:
                from acorn.core.annotations import ROIAnnotation
                roi = ROIAnnotation(
                    vertices=vertices, area_nm2=0.0, stats={},
                    color=self._sam_color_for_label(point_label), linewidth=1.5,
                    label=point_label,
                )
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            self._sam_panel.set_sam_status(
                f"Preview updated ({len(points_snap)} prompt point(s)).  "
                "Draw more strokes to refine, or Commit & New / Accept All."
            )
            # Stay in scribble/freehand mode so next stroke adds more prompts
            self._canvas_widget.set_tool("freehand")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()

    # ── annotation actions ────────────────────────────────────────────────────

    def _on_undo(self) -> None:
        removed = self._canvas_widget.canvas.store.undo()
        if removed:
            self._statusbar.showMessage(f"Removed: {removed.type}")

    def _on_clear_annotations(self) -> None:
        self._canvas_widget.canvas.store.clear()
        self._ann_states[self._img_idx] = []   # mark visited so auto-scalebar won't re-add
        self._click_buffer.clear()
        self._canvas_widget.clear_rubber_band()
        self._canvas_widget.force_redraw()     # synchronous — clears immediately on ThinLinc
        self._statusbar.showMessage("Annotations cleared")

    # ── export ────────────────────────────────────────────────────────────────

    @staticmethod
    def _format_write_error(e: Exception, dest: str) -> str:
        """Return a user-friendly error string for file-write failures."""
        import errno as _errno
        if isinstance(e, MemoryError):
            return "Out of RAM — reduce tile size or image DPI and try again"
        if isinstance(e, OSError):
            if e.errno == _errno.ENOSPC:
                return f"Disk full — check available space on {Path(dest).anchor}"
            if e.errno == _errno.EACCES:
                return f"Permission denied writing to {dest}"
            if e.errno == _errno.EROFS:
                return f"Read-only filesystem: {dest}"
        return f"Error: {e}"

    def _on_export(self, path: str, fmt: str, dpi: int) -> None:
        try:
            out = self._canvas_widget.canvas.save(path, dpi=dpi, fmt=fmt)
            self._export_panel.set_status(f"Saved: {out.name}")
            self._statusbar.showMessage(f"Exported → {out}")
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, path))

    def _on_export_raw(self, path: str) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return
        try:
            import tifffile
            tifffile.imwrite(path, img.raw,
                             metadata={"pixel_size_nm": str(img.pixel_size)})
            self._export_panel.set_status(f"Raw TIFF saved: {Path(path).name}")
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, path))

    def _on_export_masks(self, stem: str) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_status("No image loaded.")
            return
        store = self._canvas_widget.canvas.store
        rois = [a for a in store if getattr(a, "type", None) == "roi"]
        if not rois:
            self._export_panel.set_status("No ROI regions to export.")
            return
        try:
            from acorn.export.mask_exporter import export_masks
            result = export_masks(store, img.shape, stem)
            n = result["n_regions"]
            self._export_panel.set_status(f"Masks saved: {n} region(s)")
            self._statusbar.showMessage(
                f"Mask → {result['mask_path']}  Labels → {result['json_path']}"
            )
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, stem))

    def _on_training_export(self, dataset_dir: str) -> None:
        import shutil
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_train_status("No image loaded.")
            return

        # Disk-space pre-check
        ds_path = Path(dataset_dir)
        ds_path.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(str(ds_path)).free
        if free_bytes < 500 * 1024 * 1024:
            free_mb = free_bytes // (1024 * 1024)
            self._export_panel.set_train_status(
                f"Low disk space: only {free_mb} MB free on "
                f"{ds_path.anchor}. Export cancelled — free at least 500 MB."
            )
            return

        store_snapshot = list(self._canvas_widget.canvas.store)

        # Warn immediately if there are no ROI annotations on this image
        n_rois = sum(1 for a in store_snapshot if getattr(a, "type", None) == "roi")
        if n_rois == 0:
            self._export_panel.set_train_status(
                "No ROI annotations on this image — pick particles in the SAM tab first, "
                "then Commit & New (or Accept All) before exporting."
            )
            return

        params   = self._contrast_panel.params()
        cfg_dict = self._export_panel.training_config()

        from acorn.export.training_exporter import TrainingConfig
        config = TrainingConfig(**cfg_dict)

        self._export_panel.set_train_status(
            f"Exporting {n_rois} ROI(s)… (running in background)"
        )
        self._statusbar.showMessage(f"Training export started -> {dataset_dir}")

        self._train_thread = TrainingThread(
            dataset_dir, img, store_snapshot, params, config, parent=self
        )
        self._train_thread.progress.connect(self._export_panel.set_train_status)
        self._train_thread.progress_int.connect(self._export_panel.set_train_progress)
        self._train_thread.finished.connect(self._on_training_export_done)
        self._train_thread.error.connect(
            lambda msg: (
                self._export_panel.set_train_status(f"Error: {msg}"),
                self._export_panel.reset_train_progress(),
            )
        )
        self._train_thread.start()

    def _on_training_export_done(self, result: dict) -> None:
        n_aug   = result["n_augmented"]
        n_tiles = result["n_tiles"]
        n_inst  = result["n_instances_total"]
        n_skip  = result["n_skipped_tiles"]
        self._export_panel.set_train_status(
            f"Done. {n_aug} entries ({n_tiles} tiles, {n_inst} instance(s), "
            f"{n_skip} empty skipped)."
        )
        self._export_panel.reset_train_progress()
        self._statusbar.showMessage(
            f"Training export done  ({n_aug} tiles/augs, {n_inst} masks)"
        )

    def _on_queue_image(self, _dataset_dir: str) -> None:
        """Snapshot the current image + annotations into the export queue."""
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_train_status("No image loaded.")
            return

        store_snapshot = list(self._canvas_widget.canvas.store)
        n_rois = sum(1 for a in store_snapshot if getattr(a, "type", None) == "roi")

        stem = (
            self._image_paths[self._img_idx].stem
            if 0 <= self._img_idx < len(self._image_paths)
            else "image"
        )
        if any(item["stem"] == stem for item in self._export_queue):
            self._export_panel.set_train_status(f"{stem} is already queued.")
            return

        self._export_queue.append({
            "dm4img":         img,
            "store_snapshot": store_snapshot,
            "params":         self._contrast_panel.params(),
            "stem":           stem,
            "path":           str(img.filepath) if img is not None else "",
            "n_rois":         n_rois,
        })
        names = [item["stem"] for item in self._export_queue]
        self._export_panel.set_queue_status(len(self._export_queue), names)
        self._export_panel.update_queue_table(self._export_queue)
        ann_note = f"{n_rois} ROI(s)" if n_rois > 0 else "no annotations — will contribute negative tiles"
        self._export_panel.set_train_status(
            f"Queued: {stem} ({ann_note})  —  {len(self._export_queue)} total in queue."
        )

    def _on_clear_queue(self) -> None:
        self._export_queue.clear()
        self._export_panel.set_queue_status(0, [])
        self._export_panel.update_queue_table([])
        self._export_panel.set_train_status("Queue cleared.")

    def _on_import_negatives(self) -> None:
        """Open a file/folder picker and add selected images to the queue as negatives."""
        from acorn.core.dm4_loader import scan_folder

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select negative images",
            str(Path.home()),
            "Images (*.dm4 *.tif *.tiff *.mrc *.mrcs *.png *.jpg *.jpeg)",
        )

        # Also offer folder import
        if not paths:
            folder = QFileDialog.getExistingDirectory(
                self, "Or select a folder of negative images", str(Path.home())
            )
            if folder:
                paths = [str(p) for p in scan_folder(Path(folder))]

        if not paths:
            return

        params = self._contrast_panel.params()
        added = 0
        skipped = 0
        for p in paths:
            stem = Path(p).stem
            if any(item["stem"] == stem for item in self._export_queue):
                skipped += 1
                continue
            self._export_queue.append({
                "dm4img":         None,        # loaded on demand in BatchExportThread
                "path":           p,
                "store_snapshot": [],          # no annotations — pure negative
                "params":         params,
                "stem":           stem,
            })
            added += 1

        names = [item["stem"] for item in self._export_queue]
        self._export_panel.set_queue_status(len(self._export_queue), names)
        self._export_panel.update_queue_table(self._export_queue)
        msg = f"Added {added} negative image(s) to queue."
        if skipped:
            msg += f"  {skipped} already queued, skipped."
        self._export_panel.set_train_status(msg)

    def _on_batch_export(self, dataset_dir: str) -> None:
        """Export all queued images to the training dataset."""
        if not self._export_queue:
            return

        if self._batch_export_thread and self._batch_export_thread.isRunning():
            self._export_panel.set_train_status("Export already running — please wait.")
            return

        from acorn.export.training_exporter import TrainingConfig
        cfg_dict = self._export_panel.training_config()
        config   = TrainingConfig(**cfg_dict)

        n = len(self._export_queue)
        self._export_panel.set_train_status(f"Starting batch export of {n} image(s)…")
        self._statusbar.showMessage(f"Batch training export started — {n} image(s)")

        self._batch_export_thread = BatchExportThread(
            items=list(self._export_queue),
            dataset_dir=dataset_dir,
            config=config,
            parent=self,
        )
        self._batch_export_thread.image_status.connect(self._export_panel.set_train_status)
        self._batch_export_thread.image_progress.connect(self._export_panel.set_image_progress)
        self._batch_export_thread.tile_progress.connect(self._export_panel.set_train_progress)
        self._batch_export_thread.item_done.connect(
            lambda idx, stem: self._statusbar.showMessage(
                f"Exported {stem} ({idx + 1}/{n})"
            )
        )
        self._batch_export_thread.error.connect(
            lambda idx, msg: self._export_panel.set_train_status(f"Error: {msg}")
        )
        self._batch_export_thread.finished.connect(self._on_batch_export_done)
        self._batch_export_thread.start()

    def _on_batch_export_done(self, results: list) -> None:
        self._export_panel.reset_train_progress()
        total_aug   = sum(r.get("n_augmented", 0) for r in results)
        total_inst  = sum(r.get("n_instances_total", 0) for r in results)
        n_images    = len(results)
        self._export_panel.set_train_status(
            f"Batch export done — {n_images} image(s), {total_aug} tiles/augs, "
            f"{total_inst} instance(s) total."
        )
        self._statusbar.showMessage(
            f"Batch export complete: {n_images} images, {total_aug} tiles"
        )
        # Clear queue after successful export
        self._export_queue.clear()
        self._export_panel.set_queue_status(0, [])
        self._export_panel.update_queue_table([])

    def _on_push_hub(self, dataset_dir: str, repo_id: str, token: str) -> None:
        try:
            from acorn.export.hub_exporter import push_to_hub
            token_arg = token if token else None
            url = push_to_hub(dataset_dir, repo_id, token=token_arg)
            self._export_panel.set_hub_status(f"Pushed. URL: {url}")
            self._statusbar.showMessage(f"Dataset pushed to HuggingFace Hub: {url}")
        except Exception as exc:
            self._export_panel.set_hub_status(f"Error: {exc}")

    def _on_display_export(self) -> None:
        """Export 8-bit contrast-normalised PNG next to the source file for external annotation."""
        img = self._canvas_widget.canvas.dm4
        norm = self._canvas_widget.canvas.norm_image
        if img is None or norm is None:
            self._export_panel.set_status("No image loaded.")
            return
        import numpy as np
        from PIL import Image as _PILImage
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        if img8.ndim == 2:
            pil_img = _PILImage.fromarray(img8, mode="L")
        else:
            pil_img = _PILImage.fromarray(img8)
        out_path = img.filepath.parent / f"{img.filepath.stem}_display.png"
        pil_img.save(str(out_path))
        self._export_panel.set_status(f"Saved: {out_path.name}")
        self._statusbar.showMessage(f"Display image saved: {out_path}")

    def _import_png_mask(self) -> None:
        """Import a PNG/TIFF mask as ROI annotations with user-defined label mapping."""
        if not self._image_paths:
            QMessageBox.information(self, "Import Annotations", "Open an image first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select mask image", "",
            "Images (*.png *.tif *.tiff *.jpg);;All files (*)"
        )
        if not path:
            return
        import numpy as np
        try:
            from PIL import Image as _PILImage
            mask_img = np.array(_PILImage.open(path).convert("RGB"))
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Could not load mask:\n{exc}")
            return

        # Find unique colors, skip black (background)
        h, w, _ = mask_img.shape
        pixels = mask_img.reshape(-1, 3)
        unique_colors = [
            tuple(int(c) for c in color)
            for color in np.unique(pixels, axis=0)
            if not (color[0] < 15 and color[1] < 15 and color[2] < 15)
        ]
        if not unique_colors:
            QMessageBox.information(self, "Import Annotations",
                                    "No non-black regions found in the mask.")
            return

        # Show color mapping dialog
        dlg = _PngMaskMapDialog(unique_colors, pixels, self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        label_map = dlg.label_map()   # {(r,g,b): label_str}

        from acorn.core.annotations import ROIAnnotation
        try:
            from skimage.measure import find_contours, label as sk_label
        except ImportError:
            QMessageBox.warning(self, "Import Error",
                                "scikit-image is required:\n  pip install scikit-image")
            return

        store = self._canvas_widget.canvas.store
        n_added = 0
        for color, lbl in label_map.items():
            if not lbl.strip():
                continue
            binary = np.all(mask_img == np.array(color, dtype=np.uint8), axis=2).astype(np.uint8)
            labeled = sk_label(binary)
            for region_id in range(1, labeled.max() + 1):
                region_mask = (labeled == region_id)
                if region_mask.sum() < 9:
                    continue
                contours = find_contours(region_mask.astype(float), 0.5)
                if not contours:
                    continue
                contour = max(contours, key=len)
                if len(contour) < 3:
                    continue
                vertices = [(float(c[1]), float(c[0])) for c in contour]
                roi = ROIAnnotation(
                    vertices=vertices, area_nm2=0.0, stats={},
                    color=self._ann_panel.color, linewidth=1.5, label=lbl.strip(),
                )
                store.add(roi)
                n_added += 1

        self._statusbar.showMessage(
            f"Imported {n_added} annotation(s) from {Path(path).name}"
        )

    def _import_imagej_roi(self) -> None:
        """Import an ImageJ/FIJI ROI .zip as ROI annotations."""
        if not self._image_paths:
            QMessageBox.information(self, "Import Annotations", "Open an image first.")
            return
        try:
            import roifile
        except ImportError:
            QMessageBox.warning(
                self, "Missing dependency",
                "roifile is required to read ImageJ ROI sets:\n"
                "  pip install roifile"
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Select ImageJ ROI set", "",
            "ImageJ ROI zip (*.zip);;ROI file (*.roi);;All files (*)"
        )
        if not path:
            return
        try:
            rois = roifile.roiread(path)
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Could not read ROI file:\n{exc}")
            return

        from acorn.core.annotations import ROIAnnotation
        store = self._canvas_widget.canvas.store
        col = self._ann_panel.color
        lw  = self._ann_panel.linewidth
        n_added = 0
        for roi in rois:
            name = getattr(roi, "name", "") or ""
            try:
                coords = roi.coordinates()   # (N,2) array of (x, y)
            except Exception:
                continue
            if coords is None or len(coords) < 3:
                continue
            vertices = [(float(c[0]), float(c[1])) for c in coords]
            ann = ROIAnnotation(
                vertices=vertices, area_nm2=0.0, stats={},
                color=col, linewidth=lw, label=name,
            )
            store.add(ann)
            n_added += 1

        self._statusbar.showMessage(
            f"Imported {n_added} ROI(s) from {Path(path).name}"
        )

    # ── SAM 2 handlers ────────────────────────────────────────────────────────

    def _sam_busy(self) -> bool:
        """Return True if a SAM thread is currently running."""
        return self._sam_thread is not None and self._sam_thread.isRunning()

    def _sam_warmup_encode(self) -> None:
        """Encode the current image into the SAM embedding cache in the background.

        Called after model load and after image switch so the first point/box
        prompt skips the expensive ViT encoder pass.
        """
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return
        if self._sam_busy():
            return
        norm = self._canvas_widget.canvas.norm_image
        if norm is None:
            return
        import numpy as np
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        self._sam_panel.set_sam_status("Encoding image…")

        def _run():
            result = self._sam_predictor.encode_image(img8)
            return result  # True = loaded from disk, False = recomputed

        def _done(from_cache):
            if from_cache is True:
                self._sam_panel.set_sam_status("Ready  (embedding loaded from cache).")
            else:
                self._sam_panel.set_sam_status("Ready.")

        def _err(msg):
            self._sam_panel.set_sam_status(f"Encode failed: {msg}")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()

    def _clear_sam_point_artists(self) -> None:
        """Remove all SAM point-marker dots from the canvas."""
        for group in self._sam_point_artists:
            for a in group:
                try:
                    a.remove()
                except (ValueError, AttributeError):
                    pass
        self._sam_point_artists.clear()
        self._canvas_widget.canvas._overlay_artists.clear()
        self._canvas_widget.canvas.blit_annotations()

    @staticmethod
    def _sam_color_for_label(label: str) -> str:
        """Return a consistent colour for a SAM mask label.

        Matches the quick-select button colours in SAMPanel so the mask on
        canvas always corresponds visually to the button used to create it.
        """
        _FIXED = {
            "foreground": "#27ae60",
            "background": "#e74c3c",
        }
        key = label.strip().lower()
        if key in _FIXED:
            return _FIXED[key]
        # Deterministic colour for any custom label — cycle through the same
        # palette used by SAMPanel._user_label_colors
        _PALETTE = ["#8e44ad", "#1a6fa8", "#d35400", "#16a085", "#2c3e50"]
        return _PALETTE[hash(key) % len(_PALETTE)]

    def _on_sam_load_model(self, checkpoint: str, model_cfg: str, backend: str) -> None:
        if self._sam_busy():
            return

        ckpt_arg = checkpoint if checkpoint else None

        if backend == "usam":
            from acorn.core.usam_predictor import MicroSAMPredictor
            model_type = self._sam_panel.usam_model_type
            predictor  = MicroSAMPredictor(
                model_type=model_type,
                checkpoint_path=ckpt_arg,
            )
            label = f"micro-SAM ({model_type})"
        else:
            from acorn.core.sam_predictor import SAMPredictor
            predictor = SAMPredictor(
                checkpoint_path=ckpt_arg, model_cfg=model_cfg, backend=backend
            )
            label = backend

        self._sam_panel.set_model_status("Loading model…", loaded=False)

        # For usam, emit download progress through the thread's status signal.
        # _emit is filled after the thread is constructed (thread-safe via Qt signal queue).
        _emit: list = [None]

        def _run():
            if backend == "usam":
                def _progress(pct):
                    fn = _emit[0]
                    if fn:
                        fn(f"Downloading {model_type}… {pct}%")
                predictor.load_model(progress_cb=_progress)
            else:
                predictor.load_model()
            return predictor

        def _done(p):
            self._sam_predictor = p
            active = getattr(p, "backend", None) or label
            self._sam_panel.set_model_status(f"Model loaded ({active}).", loaded=True)
            self._statusbar.showMessage(f"SAM model loaded ({active}).")
            self._sam_warmup_encode()

        def _err(msg):
            self._sam_panel.set_model_status(f"Load failed: {msg}", loaded=False)
            QMessageBox.critical(self, "SAM model failed to load",
                f"The model could not be loaded:\n\n{msg}\n\n"
                "Check that the model file exists and you have read access to it.")

        self._sam_thread = SAMThread(_run, self)
        if backend == "usam":
            _emit[0] = self._sam_thread.status.emit
            self._sam_thread.status.connect(
                lambda msg: self._sam_panel.set_model_status(msg, loaded=False)
            )
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()

    def _get_sam_working_image(self):
        """Return (img8, offset_x, offset_y).

        If a crop region is set, img8 is the cropped sub-image and (offset_x, offset_y)
        is the top-left corner in full-image pixel coordinates.  Callers must add the
        offset to all polygon vertices returned by SAM.
        """
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return None, 0, 0
        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm = apply_contrast(img.raw, self._contrast_panel.params())
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        if self._sam_crop_region is None:
            return img8, 0, 0
        x0, y0, x1, y1 = self._sam_crop_region
        h, w = img8.shape[:2]
        cx0 = max(0, int(round(min(x0, x1))))
        cy0 = max(0, int(round(min(y0, y1))))
        cx1 = min(w, int(round(max(x0, x1))))
        cy1 = min(h, int(round(max(y0, y1))))
        if cx1 <= cx0 or cy1 <= cy0:
            return img8, 0, 0
        return img8[cy0:cy1, cx0:cx1], cx0, cy0

    def _on_sam_exclude_mode(self) -> None:
        self._sam_mode = "exclude_zone"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM exclude zone: drag to mark region SAM should ignore")

    def _on_sam_exclude_clear(self) -> None:
        self._sam_exclude_zone = None
        self._canvas_widget.clear_exclude_zone()
        if self._img_idx >= 0:
            self._sam_exclude_zones.pop(self._img_idx, None)
            self._autosave_timer.start()
        self._sam_panel.reset_region_btns()
        if self._sam_mode == "exclude_zone":
            self._sam_mode = None
        self._statusbar.showMessage("SAM exclude zone cleared.")

    def _on_sam_crop_mode(self) -> None:
        self._sam_mode = "crop_region"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM crop region: drag to restrict SAM to a sub-area")

    def _on_sam_crop_clear(self) -> None:
        self._sam_crop_region = None
        self._canvas_widget.clear_crop_region()
        if self._img_idx >= 0:
            self._sam_crop_regions_saved.pop(self._img_idx, None)
            self._autosave_timer.start()
        self._sam_panel.reset_region_btns()
        if self._sam_mode == "crop_region":
            self._sam_mode = None
        self._statusbar.showMessage("SAM crop region cleared — SAM will use the full image.")

    def _on_sam_auto_segment(self) -> None:
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is busy — please wait.")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._sam_panel.set_sam_status("Load the SAM model first.")
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return
        params = self._sam_panel.auto_params
        active = self._sam_predictor.backend or "SAM"
        crop_note = " (cropped region)" if self._sam_crop_region is not None else ""
        self._sam_panel.set_sam_status(f"Running {active}{crop_note}…")

        def _run():
            return self._sam_predictor.predict_everything(img8, **params)

        def _done(masks):
            self._add_sam_masks_to_store(masks, offset=(ox, oy))
            n = len(self._pending_sam_masks)
            # Switch to select mode so user can click masks to delete individually.
            # Also reset prompt-mode buttons so they can be re-activated cleanly.
            self._canvas_widget.set_tool("none")
            self._sam_mode = None
            self._sam_panel.reset_prompt_mode()
            self._sam_panel.set_sam_status(
                f"{n} mask(s) added.  Click a mask to select it, then press Delete to remove it.  "
                "Accept All to keep all remaining masks."
            )
            self._statusbar.showMessage(f"{active} auto-segment: {n} masks found — click to select, Delete to remove.")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()

    def _on_sam_point_mode(self, positive: bool) -> None:
        self._sam_mode      = "pos_point" if positive else "neg_point"
        self._sam_box_click = None
        self._canvas_widget.set_tool("sam")
        label = "positive" if positive else "negative"
        self._statusbar.showMessage(f"SAM: click on canvas to add a {label} point prompt")

    def _on_sam_box_mode(self) -> None:
        self._sam_mode      = "box"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM box: drag around object, or click two corners")

    def _on_sam_scribble_mode(self) -> None:
        self._sam_mode      = "scribble"
        self._sam_box_click = None
        self._canvas_widget.set_tool("freehand")   # reuse freehand canvas tool
        self._statusbar.showMessage(
            "SAM scribble: draw along the feature — stroke points become positive prompts"
        )

    def _on_sam_accept(self) -> None:
        self._pending_sam_masks.clear()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")
        self._sam_panel.set_sam_status("Masks accepted as ROI annotations.")

    def _on_sam_accept_and_queue(self) -> None:
        """Accept all pending SAM masks then immediately queue the image for export."""
        self._on_sam_accept()
        ds_dir = self._export_panel.dataset_dir
        if not ds_dir:
            self._sam_panel.set_sam_status(
                "Masks accepted. Set a dataset directory in the Export tab to enable queuing."
            )
            return
        self._on_queue_image(ds_dir)
        n = len(self._export_queue)
        self._sam_panel.set_sam_status(
            f"Masks accepted and image queued ({n} total in queue)."
        )

    def _on_sam_reject(self) -> None:
        store = self._canvas_widget.canvas.store
        for _ in range(len(self._pending_sam_masks)):
            store.undo()
        self._pending_sam_masks.clear()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")
        self._sam_panel.set_sam_status("SAM masks removed.")

    def _on_sam_commit_new(self) -> None:
        """Lock current preview, switch to select mode for vertex editing."""
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")   # select mode — user can now edit vertices
        count = len(self._pending_sam_masks)
        self._sam_panel.set_sam_status(
            f"{count} mask(s) committed. Edit vertices if needed, then click + Positive Point for the next object."
        )

    def _on_sam_undo_point(self) -> None:
        """Remove the last added point and re-run SAM with the remaining points."""
        if not self._sam_prompt_points:
            self._sam_panel.set_sam_status("No points to undo.")
            return
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return

        # Remove the last point from state
        self._sam_prompt_points.pop()
        self._sam_prompt_labels.pop()

        # Remove its canvas marker
        if self._sam_point_artists:
            for a in self._sam_point_artists.pop():
                self._canvas_widget.remove_artist(a)

        # Remove the current preview mask from the store
        if self._sam_current_preview is not None:
            self._canvas_widget.canvas.store.undo()
            if self._sam_current_preview in self._pending_sam_masks:
                self._pending_sam_masks.remove(self._sam_current_preview)
            self._sam_current_preview = None

        # No points left — just report and stop
        if not self._sam_prompt_points:
            self._sam_panel.set_sam_status("All points removed. Click to start a new prompt.")
            return

        # Re-run SAM with the remaining points
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        points_for_sam = [(px - ox, py - oy) for px, py in self._sam_prompt_points]
        labels_snap    = list(self._sam_prompt_labels)
        point_label    = self._sam_panel.point_label
        self._sam_panel.set_sam_status("Re-running SAM…")

        def _run():
            return self._sam_predictor.predict_points(img8, points_for_sam, labels=labels_snap)

        def _done(masks):
            if not masks:
                self._sam_panel.set_sam_status("SAM returned no mask — add more points.")
                return
            store = self._canvas_widget.canvas.store
            vertices = self._sam_predictor.mask_to_polygon(masks[0])
            if ox != 0 or oy != 0:
                vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
            if len(vertices) >= 3:
                from acorn.core.annotations import ROIAnnotation
                roi = ROIAnnotation(
                    vertices=vertices, area_nm2=0.0, stats={},
                    color=self._sam_color_for_label(point_label), linewidth=1.5, label=point_label,
                )
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            n_pos = labels_snap.count(1)
            n_neg = labels_snap.count(0)
            self._sam_panel.set_sam_status(
                f"Preview: {n_pos} pos + {n_neg} neg point(s).  "
                "Add more points, Commit & New to lock and edit, or Accept All."
            )

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()

    def _on_sam_clear_points(self) -> None:
        """Discard accumulated point prompts and remove the current preview mask."""
        if self._sam_current_preview is not None:
            self._canvas_widget.canvas.store.undo()
            if self._sam_current_preview in self._pending_sam_masks:
                self._pending_sam_masks.remove(self._sam_current_preview)
            self._sam_current_preview = None
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._clear_sam_point_artists()
        self._sam_panel.set_sam_status("Points cleared. Click to start a new prompt.")

    def _add_sam_masks_to_store(self, masks, offset: tuple = (0, 0)) -> None:
        """Convert SAM masks to ROIAnnotations and add to the store.

        Parameters
        ----------
        masks  : list of masks returned by SAMPredictor
        offset : (ox, oy) pixel offset to add to all polygon vertices.
                 Non-zero when SAM was run on a cropped sub-image.
        """
        store = self._canvas_widget.canvas.store
        label = self._sam_panel.label
        ox, oy = offset
        self._pending_sam_masks.clear()
        for mask in masks:
            vertices = self._sam_predictor.mask_to_polygon(mask)
            if len(vertices) < 3:
                continue
            if ox != 0 or oy != 0:
                vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
            # Filter: discard masks whose centroid falls inside the exclude zone
            if self._sam_exclude_zone is not None:
                ex0, ey0, ex1, ey1 = self._sam_exclude_zone
                cx = sum(v[0] for v in vertices) / len(vertices)
                cy = sum(v[1] for v in vertices) / len(vertices)
                if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                    continue
            from acorn.core.annotations import ROIAnnotation
            roi = ROIAnnotation(
                vertices  = vertices,
                area_nm2  = 0.0,
                stats     = {},
                color     = self._sam_color_for_label(label),
                linewidth = 1.5,
                label     = label,
            )
            store.add(roi)
            self._pending_sam_masks.append(roi)

    def _sam_point_prompt(self, x: float, y: float, positive: bool) -> None:
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._sam_panel.set_sam_status("Load the SAM model first (click 'Load Model').")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return

        # Accumulate point (stored in full image coords), draw marker immediately
        point_label = self._sam_panel.point_label
        self._sam_prompt_points.append((x, y))
        self._sam_prompt_labels.append(1 if positive else 0)
        display_label = point_label if positive else ""
        markers = self._canvas_widget.add_sam_point_marker(x, y, positive, label=display_label)
        self._sam_point_artists.append(markers)   # list-of-lists: one group per point

        # Snapshot in crop-space (full coords minus crop offset)
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        points_snap_full = list(self._sam_prompt_points)
        labels_snap      = list(self._sam_prompt_labels)
        points_for_sam   = [(px - ox, py - oy) for px, py in points_snap_full]
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return self._sam_predictor.predict_points(img8, points_for_sam, labels=labels_snap)

        def _done(masks):
            if not masks:
                n_pos = labels_snap.count(1)
                n_neg = labels_snap.count(0)
                if n_pos == 0:
                    self._sam_panel.set_sam_status(
                        "No mask — add at least one positive point first, "
                        "then use negative points to refine."
                    )
                else:
                    self._sam_panel.set_sam_status(
                        "SAM returned no mask for these points — try repositioning."
                    )
                return
            store = self._canvas_widget.canvas.store
            if self._sam_current_preview is not None:
                store.undo()
                if self._sam_current_preview in self._pending_sam_masks:
                    self._pending_sam_masks.remove(self._sam_current_preview)
                self._sam_current_preview = None
            vertices = self._sam_predictor.mask_to_polygon(masks[0])
            if ox != 0 or oy != 0:
                vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
            if len(vertices) >= 3:
                from acorn.core.annotations import ROIAnnotation
                roi = ROIAnnotation(
                    vertices=vertices, area_nm2=0.0, stats={},
                    color=self._sam_color_for_label(point_label), linewidth=1.5, label=point_label,
                )
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            n_pos = labels_snap.count(1)
            n_neg = labels_snap.count(0)
            self._sam_panel.set_sam_status(
                f"Preview: {n_pos} pos + {n_neg} neg point(s).  "
                "Add more points, Commit & New to lock and edit, or Accept All."
            )

        def _err(msg):
            self._sam_prompt_points.pop()
            self._sam_prompt_labels.pop()
            if self._sam_point_artists:
                for a in self._sam_point_artists.pop():
                    self._canvas_widget.remove_artist(a)
            self._sam_panel.set_sam_status(f"Error: {msg}")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()

    def _sam_box_prompt_first_click(self, x: float, y: float) -> None:
        self._sam_box_click = (x, y)
        self._canvas_widget.set_sam_box_anchor(x, y)
        self._statusbar.showMessage(
            f"SAM box: first corner at ({x:.0f}, {y:.0f}) — drag or click second corner"
        )

    def _sam_box_prompt_second_click(self, x: float, y: float) -> None:
        if self._sam_box_click is None:
            return
        if self._sam_busy():
            return
        x0, y0 = self._sam_box_click
        x1, y1 = x, y
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        box = (min(x0, x1) - ox, min(y0, y1) - oy, max(x0, x1) - ox, max(y0, y1) - oy)
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return self._sam_predictor.predict_box(img8, box)

        def _done(mask):
            self._add_sam_masks_to_store([mask], offset=(ox, oy))
            self._sam_panel.set_sam_status("Box prompt: 1 mask. Undo if wrong, or Accept All.")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()

    def _on_sam_box_drag(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Handle a drag-drawn SAM box (from canvas sam_box_commit signal).

        Routes to exclude-zone, crop-region, or normal box-prompt handling
        depending on the current _sam_mode.
        """
        if self._sam_mode == "exclude_zone":
            self._sam_exclude_zone = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self._canvas_widget.set_exclude_zone(*self._sam_exclude_zone)
            if self._img_idx >= 0:
                self._sam_exclude_zones[self._img_idx] = self._sam_exclude_zone
                self._autosave_timer.start()
            self._sam_panel.reset_region_btns()
            self._sam_mode = None
            self._canvas_widget.set_tool("none")
            self._statusbar.showMessage(
                f"Exclude zone set: ({self._sam_exclude_zone[0]:.0f}, {self._sam_exclude_zone[1]:.0f}) — "
                f"({self._sam_exclude_zone[2]:.0f}, {self._sam_exclude_zone[3]:.0f})"
            )
            return

        if self._sam_mode == "crop_region":
            self._sam_crop_region = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self._canvas_widget.set_crop_region(*self._sam_crop_region)
            if self._img_idx >= 0:
                self._sam_crop_regions_saved[self._img_idx] = self._sam_crop_region
                self._autosave_timer.start()
            self._sam_panel.reset_region_btns()
            self._sam_mode = None
            self._canvas_widget.set_tool("none")
            self._statusbar.showMessage(
                f"Crop region set: ({self._sam_crop_region[0]:.0f}, {self._sam_crop_region[1]:.0f}) — "
                f"({self._sam_crop_region[2]:.0f}, {self._sam_crop_region[3]:.0f})"
            )
            return

        if self._sam_mode != "box":
            return
        if self._sam_busy():
            return
        self._sam_box_click = None   # cancel any pending two-click state

        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        box = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return self._sam_predictor.predict_box(img8, box)

        def _done(mask):
            self._add_sam_masks_to_store([mask], offset=(ox, oy))
            self._sam_panel.set_sam_status("Box prompt: 1 mask. Undo if wrong, or Accept All.")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()

    def _on_check_quality(self) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._export_panel.set_quality_status("No image loaded.", ok=False)
            return
        try:
            from acorn.core.quality import assess_quality
            report = assess_quality(img.raw)
            lines = [f"Blur: {report.blur_score:.1f}  CV: {report.cv_score:.3f}  "
                     f"Sat: {report.saturation_frac*100:.1f}%  "
                     f"LowFreq: {report.low_freq_frac*100:.1f}%"]
            if report.warnings:
                lines += report.warnings
            self._export_panel.set_quality_status("\n".join(lines), ok=report.ok)
            status = "Quality OK" if report.ok else f"Quality: {len(report.warnings)} warning(s)"
            self._statusbar.showMessage(status)
        except Exception as exc:
            self._export_panel.set_quality_status(f"Error: {exc}", ok=False)

    def _on_finalize_dataset(self, dataset_dir: str, val_frac: float, test_frac: float, assignments: dict) -> None:
        try:
            from acorn.export.dataset_finalizer import finalize_dataset
            result = finalize_dataset(
                dataset_dir, val_frac=val_frac, test_frac=test_frac,
                explicit_splits=assignments or None,
            )
            sc = result["split_counts"]
            self._export_panel.set_fin_status(
                f"Done. Train={sc['train']}  Val={sc['val']}  Test={sc['test']} tiles."
            )
            self._statusbar.showMessage(
                f"Dataset finalized -> {dataset_dir}/splits/  "
                f"train={sc['train']} val={sc['val']} test={sc['test']}"
            )
        except Exception as e:
            self._export_panel.set_fin_status(f"Error: {e}")

    # ── YOLO handlers ─────────────────────────────────────────────────────────

    def _yolo_busy(self) -> bool:
        return self._yolo_thread is not None and self._yolo_thread.isRunning()

    def _on_yolo_load_model(self, model_path: str) -> None:
        if self._yolo_busy():
            return
        from acorn.core.yolo_predictor import YOLOPredictor
        predictor = YOLOPredictor()
        self._yolo_panel.set_model_status("Loading…", loaded=False)

        def _run():
            predictor.load_model(model_path)
            return predictor

        def _done(p):
            self._yolo_predictor = p
            seg_note = " (seg)" if p.is_seg else ""
            self._yolo_panel.set_model_status(
                f"Loaded{seg_note}: {model_path}", loaded=True
            )
            self._statusbar.showMessage(f"YOLO model loaded: {model_path}")

        def _err(msg):
            self._yolo_panel.set_model_status(f"Load failed: {msg}", loaded=False)

        self._yolo_thread = SAMThread(_run, self)
        self._yolo_thread.finished.connect(_done)
        self._yolo_thread.error.connect(_err)
        self._yolo_thread.start()

    def _on_yolo_detect(self) -> None:
        self._run_yolo(segmentation=False)

    def _on_yolo_detect_seg(self) -> None:
        self._run_yolo(segmentation=True)

    def _run_yolo(self, segmentation: bool) -> None:
        if self._yolo_busy():
            self._yolo_panel.set_status("YOLO is running — please wait.")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._yolo_panel.set_status("No image loaded.")
            return
        if self._yolo_predictor is None or not self._yolo_predictor.is_loaded:
            self._yolo_panel.set_status("Load a YOLO model first.")
            return

        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm = apply_contrast(img.raw, self._contrast_panel.params())
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        conf = self._yolo_panel.conf_thresh
        iou  = self._yolo_panel.iou_thresh
        self._yolo_panel.set_status("Running…")

        def _run():
            if segmentation:
                return self._yolo_predictor.detect_and_segment(
                    img8, conf_thresh=conf, iou_thresh=iou
                )
            return self._yolo_predictor.detect(img8, conf_thresh=conf, iou_thresh=iou)

        def _done(detections):
            self._last_yolo_detections = detections
            self._add_yolo_detections_to_store(detections, segmentation)
            self._yolo_panel.set_status(
                f"{len(detections)} detection(s). Undo unwanted, then Accept All."
            )
            self._statusbar.showMessage(f"YOLO: {len(detections)} detection(s).")

        def _err(msg):
            self._yolo_panel.set_status(f"Error: {msg}")

        self._yolo_thread = SAMThread(_run, self)
        self._yolo_thread.finished.connect(_done)
        self._yolo_thread.error.connect(_err)
        self._yolo_thread.start()

    def _add_yolo_detections_to_store(
        self, detections: list, use_masks: bool
    ) -> None:
        store = self._canvas_widget.canvas.store
        label = self._yolo_panel.label
        color = "#FFD700"
        self._pending_yolo_anns.clear()

        has_masks = use_masks and any("mask" in d for d in detections)
        if has_masks:
            from acorn.core.yolo_predictor import masks_to_roi_annotations
            n = masks_to_roi_annotations(detections, store, label=label, color=color)
            for _ in range(n):
                self._pending_yolo_anns.append(True)
        else:
            from acorn.core.yolo_predictor import boxes_to_roi_annotations
            n = boxes_to_roi_annotations(
                detections, store, label=label, color=color,
                as_rectangles=self._yolo_panel.as_rectangles,
            )
            for _ in range(n):
                self._pending_yolo_anns.append(True)

    def _on_yolo_pipe_to_sam(self) -> None:
        if self._yolo_busy() or self._sam_busy():
            self._yolo_panel.set_status("Please wait — inference in progress.")
            return
        if not self._last_yolo_detections:
            self._yolo_panel.set_status("Run detection first, then pipe to SAM.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._yolo_panel.set_status("Load SAM model in the SAM tab first.")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return

        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm  = apply_contrast(img.raw, self._contrast_panel.params())
        img8  = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        boxes = [d["box"] for d in self._last_yolo_detections]
        self._yolo_panel.set_status(f"SAM refining {len(boxes)} box(es)…")

        def _run():
            masks = []
            for box in boxes:
                masks.append(self._sam_predictor.predict_box(img8, box))
            return masks

        def _done(masks):
            self._add_sam_masks_to_store(masks)
            self._yolo_panel.set_status(
                f"SAM refined {len(masks)} mask(s). Accept/Reject in SAM tab."
            )
            self._statusbar.showMessage(
                f"YOLO boxes -> SAM masks: {len(masks)} generated."
            )

        def _err(msg):
            self._yolo_panel.set_status(f"Error: {msg}")

        self._yolo_thread = SAMThread(_run, self)
        self._yolo_thread.finished.connect(_done)
        self._yolo_thread.error.connect(_err)
        self._yolo_thread.start()

    def _on_yolo_accept(self) -> None:
        self._pending_yolo_anns.clear()
        self._yolo_panel.set_status("Detections accepted as ROI annotations.")

    def _on_yolo_reject(self) -> None:
        store = self._canvas_widget.canvas.store
        for _ in range(len(self._pending_yolo_anns)):
            store.undo()
        self._pending_yolo_anns.clear()
        self._yolo_panel.set_status("YOLO detections removed.")

    # ── UNet handlers ─────────────────────────────────────────────────────────

    def _unet_busy(self) -> bool:
        return self._unet_thread is not None and self._unet_thread.isRunning()

    def _on_unet_load_model(
        self, arch: str, encoder: str, in_channels: int,
        n_classes: int, ckpt_path: str,
    ) -> None:
        if self._unet_busy():
            return
        from acorn.core.unet_predictor import UNetPredictor
        tile_size = self._unet_panel.tile_size
        predictor = UNetPredictor(
            architecture=arch, encoder=encoder,
            in_channels=in_channels, n_classes=n_classes,
            tile_size=tile_size,
        )
        self._unet_panel.set_model_status("Loading…", loaded=False)

        def _run():
            predictor.load_model(ckpt_path)
            return predictor

        def _done(p):
            self._unet_predictor = p
            self._unet_panel.set_model_status(
                f"Loaded ({arch}/{encoder}, {in_channels}ch, {n_classes} cls)",
                loaded=True,
            )
            self._statusbar.showMessage(f"UNet model loaded: {ckpt_path}")

        def _err(msg):
            self._unet_panel.set_model_status(f"Load failed: {msg}", loaded=False)

        self._unet_thread = SAMThread(_run, self)
        self._unet_thread.finished.connect(_done)
        self._unet_thread.error.connect(_err)
        self._unet_thread.start()

    def _on_unet_segment(self) -> None:
        if self._unet_busy():
            self._unet_panel.set_status("UNet is running — please wait.")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._unet_panel.set_status("No image loaded.")
            return
        if self._unet_predictor is None or not self._unet_predictor.is_loaded:
            self._unet_panel.set_status("Load a UNet model first.")
            return

        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm  = apply_contrast(img.raw, self._contrast_panel.params())
        img8  = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        threshold = self._unet_panel.threshold
        fg_class  = self._unet_panel.foreground_class
        min_area  = self._unet_panel.min_area
        self._unet_panel.set_status("Running segmentation…")

        def _run():
            return self._unet_predictor.predict(
                img8, threshold=threshold,
                foreground_class=fg_class, min_area=min_area,
            )

        def _done(masks):
            self._add_unet_masks_to_store(masks)
            self._unet_panel.set_status(
                f"{len(masks)} mask(s) found. Undo unwanted, then Accept All."
            )
            self._statusbar.showMessage(f"UNet: {len(masks)} instance mask(s).")

        def _err(msg):
            self._unet_panel.set_status(f"Error: {msg}")

        self._unet_thread = SAMThread(_run, self)
        self._unet_thread.finished.connect(_done)
        self._unet_thread.error.connect(_err)
        self._unet_thread.start()

    def _add_unet_masks_to_store(self, masks: list) -> None:
        store  = self._canvas_widget.canvas.store
        label  = self._unet_panel.label
        color  = "#00CED1"
        self._pending_unet_masks.clear()
        for mask in masks:
            vertices = self._unet_predictor.mask_to_polygon(mask)
            if len(vertices) < 3:
                continue
            from acorn.core.annotations import ROIAnnotation
            roi = ROIAnnotation(
                vertices=vertices, area_nm2=0.0, stats={},
                color=color, linewidth=1.5, label=label,
            )
            store.add(roi)
            self._pending_unet_masks.append(roi)

    def _on_unet_accept(self) -> None:
        self._pending_unet_masks.clear()
        self._unet_panel.set_status("Masks accepted as ROI annotations.")

    def _on_unet_reject(self) -> None:
        store = self._canvas_widget.canvas.store
        for _ in range(len(self._pending_unet_masks)):
            store.undo()
        self._pending_unet_masks.clear()
        self._unet_panel.set_status("UNet masks removed.")

    # ── training ──────────────────────────────────────────────────────────────

    def _on_train_requested(self, config: dict) -> None:
        if getattr(self, "_train_proc", None) is not None:
            import os
            try:
                os.kill(self._train_proc.pid, 0)
                self._train_panel.append_log("Training already in progress.")
                return
            except (ProcessLookupError, PermissionError):
                self._train_proc = None

        import json as _json
        import subprocess as _sp
        import sys as _sys
        from pathlib import Path as _Path

        dataset_dir = _Path(config["dataset_dir"])
        config_path = dataset_dir / "_training_config.json"
        log_path    = dataset_dir / "_training.log"
        config_path.write_text(_json.dumps(config))
        log_path.write_text("")   # clear / create

        self._train_panel.append_log(
            f"Launching training as a detached background process.\n"
            f"Training will continue even if this window is closed.\n"
            f"Log file: {log_path}"
        )

        self._train_proc = _sp.Popen(
            [_sys.executable, "-m", "acorn.core._train_worker", str(config_path)],
            stdout=open(log_path, "w"),
            stderr=_sp.STDOUT,
            start_new_session=True,   # detach — survives GUI close
        )
        self._train_log_path = log_path
        self._train_log_pos  = 0
        self._train_model_type = config["model_type"]

        self._train_tail_timer = QTimer(self)
        self._train_tail_timer.setInterval(500)
        self._train_tail_timer.timeout.connect(self._tail_train_log)
        self._train_tail_timer.start()

    def _tail_train_log(self) -> None:
        """Read new lines from training log file and update the UI."""
        import os
        try:
            with open(self._train_log_path) as f:
                f.seek(self._train_log_pos)
                new_text = f.read()
                self._train_log_pos = f.tell()
        except OSError:
            return

        for line in new_text.splitlines():
            if line.startswith("PROGRESS:"):
                try:
                    ep, total = line[9:].split("/")
                    self._train_panel.set_progress(int(ep), int(total))
                except Exception:
                    pass
            elif line.startswith("METRIC:"):
                try:
                    ep, loss, metric = line[7:].split(",")
                    self._train_panel.update_loss_curve(int(ep), float(loss), float(metric))
                except Exception:
                    pass
            elif line.startswith("DONE:"):
                model_path = line[5:]
                self._train_tail_timer.stop()
                self._train_proc = None
                self._train_panel.training_finished(self._train_model_type, model_path)
            elif line.startswith("ERROR:"):
                self._train_tail_timer.stop()
                self._train_proc = None
                self._train_panel.training_failed(line[6:])
            elif line.strip():
                self._train_panel.append_log(line)

        # Also check if process died without writing DONE/ERROR
        if self._train_proc is not None:
            try:
                os.kill(self._train_proc.pid, 0)
            except (ProcessLookupError, PermissionError):
                self._train_tail_timer.stop()
                self._train_proc = None
                self._train_panel.set_training(False)
                self._train_panel.append_log("Training process ended.")

    def _on_train_cancel(self) -> None:
        import os, signal
        if getattr(self, "_train_proc", None) is not None:
            try:
                os.kill(self._train_proc.pid, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            self._train_proc = None
        if hasattr(self, "_train_tail_timer"):
            self._train_tail_timer.stop()
        self._train_panel.set_training(False)
        self._train_panel.append_log("Training cancelled.")

    def _on_train_load_yolo(self, model_path: str) -> None:
        """Auto-load the freshly trained YOLO model into the YOLO tab."""
        from acorn.core.yolo_predictor import YOLOPredictor
        self._yolo_panel.set_model_status("Loading trained model…", loaded=False)

        def _run():
            predictor = YOLOPredictor(model_path=model_path)
            predictor.load_model()
            return predictor

        def _done(p):
            self._yolo_predictor = p
            self._yolo_panel.set_model_status(
                f"Trained model loaded: {Path(model_path).name}", loaded=True
            )
            self._statusbar.showMessage("Trained YOLO model loaded into YOLO tab.")

        def _err(msg):
            self._yolo_panel.set_model_status(f"Auto-load failed: {msg}", loaded=False)

        t = SAMThread(_run, self)
        t.finished.connect(_done)
        t.error.connect(_err)
        t.start()

    def _on_train_load_unet(self, model_path: str) -> None:
        """Auto-load the freshly trained UNet model into the UNet tab."""
        info_path = Path(model_path).parent / "training_info.json"
        arch, encoder, n_classes = "Unet", "resnet34", 2
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                arch      = info.get("arch", arch)
                encoder   = info.get("encoder", encoder)
                n_classes = info.get("n_classes", n_classes)
            except Exception:
                pass

        from acorn.core.unet_predictor import UNetPredictor
        predictor = UNetPredictor(
            architecture=arch, encoder=encoder,
            in_channels=1, n_classes=n_classes,
        )
        self._unet_panel.set_model_status("Loading trained model…", loaded=False)

        def _run():
            predictor.load_model(model_path)
            return predictor

        def _done(p):
            self._unet_predictor = p
            self._unet_panel.set_model_status(
                f"Trained model loaded: {Path(model_path).name}", loaded=True
            )
            self._statusbar.showMessage("Trained UNet model loaded into UNet tab.")

        def _err(msg):
            self._unet_panel.set_model_status(f"Auto-load failed: {msg}", loaded=False)

        t = SAMThread(_run, self)
        t.finished.connect(_done)
        t.error.connect(_err)
        t.start()

    # ── application quit ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Tear down plugins on quit."""
        for plugin in getattr(self, "_plugins", []):
            plugin.teardown()
        super().closeEvent(event)

    # ── keyboard shortcuts ────────────────────────────────────────────────────

    _TEXT_INPUT_TYPES = None   # populated lazily

    @staticmethod
    def _focus_is_text_input() -> bool:
        """Return True if a text-entry widget currently has keyboard focus."""
        from PyQt6.QtWidgets import (
            QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox,
        )
        return isinstance(
            QApplication.focusWidget(),
            (QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox),
        )

    def _dispatch_shortcut(self, key: int) -> bool:
        """Execute the action for *key*.  Returns True if the key was handled.

        N / Right arrow : next image
        B / Left arrow  : previous image
        A               : Accept All (SAM → YOLO → UNet, whichever has pending masks)
        R               : Reject All (same priority)
        P               : SAM positive-point mode
        X               : SAM negative-point mode
        C               : SAM Commit & New
        U               : SAM Undo Last Point
        """
        from PyQt6.QtCore import Qt as _Qt
        if key in (_Qt.Key.Key_N, _Qt.Key.Key_Right):
            self._on_next()
        elif key in (_Qt.Key.Key_B, _Qt.Key.Key_Left):
            self._on_prev()
        elif key == _Qt.Key.Key_A:
            if self._pending_sam_masks:
                self._on_sam_accept()
            elif self._pending_yolo_anns:
                self._on_yolo_accept()
            elif self._pending_unet_masks:
                self._on_unet_accept()
        elif key == _Qt.Key.Key_R:
            if self._pending_sam_masks:
                self._on_sam_reject()
            elif self._pending_yolo_anns:
                self._on_yolo_reject()
            elif self._pending_unet_masks:
                self._on_unet_reject()
        elif key == _Qt.Key.Key_P:
            self._sam_panel.set_positive_mode()
            self._on_sam_point_mode(positive=True)
        elif key == _Qt.Key.Key_X:
            self._sam_panel.set_negative_mode()
            self._on_sam_point_mode(positive=False)
        elif key == _Qt.Key.Key_C:
            self._on_sam_commit_new()
        elif key == _Qt.Key.Key_U:
            self._on_sam_undo_point()
        else:
            return False
        return True

    def keyPressEvent(self, event) -> None:
        if self._focus_is_text_input():
            super().keyPressEvent(event)
            return
        if not self._dispatch_shortcut(event.key()):
            super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        """Catch key presses that land on child widgets (e.g. the matplotlib canvas)
        so shortcuts work without needing to click away from the image first."""
        from PyQt6.QtCore import QEvent
        if (event.type() == QEvent.Type.KeyPress
                and obj is not self
                and not self._focus_is_text_input()):
            self._dispatch_shortcut(event.key())
        # Always return False — let the event continue to its original target too.
        return False

    # ── about dialog ──────────────────────────────────────────────────────────

    def _show_about(self) -> None:
        QMessageBox.about(
            self, "About ACORN",
            "<b>ACORN v0.1.0</b><br><br>"
            "Interactive DM4 cryo-EM image viewer, annotator, and exporter.<br><br>"
            "Features:<br>"
            "• Best-in-class contrast for low-dose cryo-EM (bandpass default)<br>"
            "• Publication-ready annotations (arrows, scale bars, text, shapes)<br>"
            "• Fiji-style measurements: distance, angle, area, line profiles<br>"
            "• Headless CLI for server-side batch processing<br><br>"
            "pip install acorn[gui]<br>"
            "uv tool install acorn",
        )


# ── entry point ───────────────────────────────────────────────────────────────

def launch(files: list[str] | None = None) -> None:
    """GUI entry point — called by `acorn-gui` script and `acorn view`."""
    import os
    import time
    import traceback
    import matplotlib

    _t0 = time.time()

    def _log(msg: str) -> None:
        print(f"  [{time.time() - _t0:5.2f}s] {msg}", flush=True)

    print("ACORN starting...", flush=True)

    def _excepthook(exc_type, exc_value, exc_tb):
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)
    sys.excepthook = _excepthook

    os.environ.setdefault("QT_FILESYSTEMMODEL_WATCH_FILES", "0")

    # ── shared model cache ─────────────────────────────────────────────────
    # Point all model libraries at the shared /opt/acorn/models/ directory
    # so every user reads from the same pre-downloaded weights instead of
    # re-downloading to their own home directory.
    # Individual users can still override these by setting the env vars
    # before launching (e.g. in their ~/.bashrc).
    _shared_models = "/opt/acorn/models"
    if os.path.isdir(_shared_models):
        os.environ.setdefault("MICROSAM_CACHEDIR",      f"{_shared_models}/micro_sam")
        # HUGGINGFACE_HUB_CACHE covers model weights only — deliberately NOT
        # setting HF_HOME so each user's login token stays in their own
        # ~/.cache/huggingface/token (needed for personal Hub pushes).
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_shared_models}/huggingface/hub")
        os.environ.setdefault("ACORN_MODELS_DIR",       _shared_models)
        # Ultralytics (YOLO) settings and weight cache
        # YOLO_CONFIG_DIR intentionally not set — each user keeps their own
        # ~/.config/Ultralytics settings; only model weights are shared.

    _log("setting matplotlib backend")
    matplotlib.use("QtAgg")

    # Qt tries to add inotify watches on NFS-mounted paths (home dir, network
    # mounts) at startup. inotify doesn't support NFS and prints
    # "inotify_add_watch(...) failed: (No space left on device)" directly to
    # stderr fd. Suppress at the fd level during QApplication init, then restore.
    _log("initialising Qt application")
    _stderr_fd = sys.stderr.fileno()
    _saved_stderr = os.dup(_stderr_fd)
    _devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(_devnull, _stderr_fd)
    os.close(_devnull)
    try:
        app = QApplication(sys.argv)
        app.setApplicationName("ACORN")
    finally:
        os.dup2(_saved_stderr, _stderr_fd)
        os.close(_saved_stderr)
    _log("Qt application ready")

    from PyQt6.QtWidgets import QStyleFactory
    from PyQt6.QtGui import QPalette, QColor
    app.setStyle(QStyleFactory.create("Fusion"))

    pal = QPalette()
    _c = QColor  # shorthand
    pal.setColor(QPalette.ColorRole.Window,          _c("#1e1e2e"))
    pal.setColor(QPalette.ColorRole.WindowText,      _c("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.Base,            _c("#313244"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   _c("#1e1e2e"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     _c("#313244"))
    pal.setColor(QPalette.ColorRole.ToolTipText,     _c("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.Text,            _c("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.Button,          _c("#313244"))
    pal.setColor(QPalette.ColorRole.ButtonText,      _c("#cdd6f4"))
    pal.setColor(QPalette.ColorRole.BrightText,      _c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight,       _c("#7c3aed"))
    pal.setColor(QPalette.ColorRole.HighlightedText, _c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link,            _c("#89b4fa"))
    pal.setColor(QPalette.ColorRole.Mid,             _c("#45475a"))
    pal.setColor(QPalette.ColorRole.Shadow,          _c("#11111b"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       _c("#6c7086"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, _c("#6c7086"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, _c("#6c7086"))
    app.setPalette(pal)

    _log("building main window")
    window = MainWindow()
    _log("showing window")
    window.show()

    if files:
        from pathlib import Path
        window.open_files([Path(f) for f in files if Path(f).is_file()])

    _log("entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
