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


def _atomic_write_text(path: Path, text: str) -> None:
    """Write text by writing to a temp file in the same dir then os.replace().

    Avoids leaving a truncated/empty file if the process is killed or a network
    (NAS) write is interrupted mid-write — the previous good file is only
    replaced once the new one is fully written.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDockWidget, QFileDialog,
    QDoubleSpinBox, QFormLayout, QGroupBox, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox,
    QHBoxLayout, QPushButton, QSpinBox, QSplitter, QStatusBar,
    QTabWidget, QVBoxLayout, QWidget,
)

from acorn.core.dm4_loader import DM4Image, scan_folder, EM_EXTS, DEFAULT_EM_CONTRAST
from acorn.core.contrast import ContrastParams
from acorn.core.annotations import (
    AnnotationStore, ArrowAnnotation, LineAnnotation, CircleAnnotation,
    RectangleAnnotation, TextAnnotation, ScalebarAnnotation,
    DistanceMeasurement, AngleMeasurement, ROIAnnotation,
)
from acorn.core.measurements import MeasurementEngine
from acorn.export import measurements_dir as _meas_dir, MEASUREMENTS_CSV as _MEAS_CSV
from acorn.render.scalebar import nice_scalebar_nm

from acorn.gui.canvas_widget import CanvasWidget
from acorn.gui.contrast_panel import ContrastPanel
from acorn.gui.annotation_panel import AnnotationPanel
from acorn.gui.measurement_panel import MeasurementPanel
from acorn.gui.export_panel import ExportPanel
from acorn.gui.sam_panel import SAMPanel
from acorn.gui.yolo_panel import YOLOPanel
from acorn.gui.unet_panel import UNetPanel
from acorn.gui.segmentation_panel import SegmentationPanel
from acorn.gui.train_panel import TrainPanel
from acorn.gui.detector_controller import DetectorControllerMixin
from acorn.gui.export_controller import ExportControllerMixin
from acorn.gui.movie import DoseSeriesDialog, MovieControllerMixin
from acorn.gui.sam_controller import SAMControllerMixin
from acorn.gui.threads import (
    BatchExportThread, FrameProcessThread, ImageLoadThread, LoadThread, SAMThread,
)
from acorn.gui.workspace_bar import WelcomeDialog, WorkspaceBar
from acorn.gui.workspaces import (
    ALWAYS_AVAILABLE_DOCKS, DEFAULT_WORKSPACE, WORKSPACES,
    by_id as workspace_by_id,
    load_prefs as load_workspace_prefs,
    save_prefs as save_workspace_prefs,
)


ImageFingerprint = tuple[str, int, int, int]


def _image_file_fingerprint(path: Path) -> ImageFingerprint | None:
    """Cheap identity for deciding whether an already-open image file changed."""
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return (
        str(Path(path).resolve()),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(getattr(stat, "st_ino", 0)),
    )


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

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("File type:"))
        self._type_filter = QComboBox()
        self._type_filter.addItem("All supported", None)
        self._type_filter.addItem("DM4",           {".dm4"})
        self._type_filter.addItem("TIFF",          {".tif", ".tiff"})
        self._type_filter.addItem("MRC / MRCS",    {".mrc", ".mrcs"})
        self._type_filter.addItem("EMD / HDF5",    {".emd", ".h5", ".hdf5"})
        self._type_filter.addItem("PNG / JPEG",    {".png", ".jpg", ".jpeg"})
        self._type_filter.currentIndexChanged.connect(self._apply_filter)
        filter_row.addWidget(self._type_filter)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self._count_label = QLabel(f"Found {len(files)} supported file(s).  Select which to open:")
        layout.addWidget(self._count_label)

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

    def _apply_filter(self) -> None:
        exts = self._type_filter.currentData()
        visible = 0
        for i in range(self._list.count()):
            item = self._list.item(i)
            p: Path = item.data(Qt.ItemDataRole.UserRole)
            show = exts is None or p.suffix.lower() in exts
            item.setHidden(not show)
            if show:
                visible += 1
        self._count_label.setText(f"{visible} file(s) shown.  Select which to open:")

    def _select_all(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Checked)

    def _deselect_all(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.CheckState.Unchecked)

    def selected_paths(self) -> list[Path]:
        result = []
        for i in range(self._list.count()):
            item = self._list.item(i)
            if not item.isHidden() and item.checkState() == Qt.CheckState.Checked:
                result.append(item.data(Qt.ItemDataRole.UserRole))
        return result


# ── module-level helpers ──────────────────────────────────────────────────────




# ── motion plot dialog ────────────────────────────────────────────────────────



# ── dose series dialog ────────────────────────────────────────────────────────



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
                f"background: rgb({r},{g},{b}); border: 1px solid #363636; border-radius: 3px;"
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
        ok_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(ok_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def label_map(self) -> dict:
        return {color: edit.text() for color, edit in self._rows}


class MainWindow(
    SAMControllerMixin,
    DetectorControllerMixin,
    ExportControllerMixin,
    MovieControllerMixin,
    QMainWindow,
):
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
                background-color: #1a1a1a;
            }
            QWidget {
                color: #e0e0e0;
            }
            QSplitter {
                background: #1a1a1a;
            }
            QSplitter::handle {
                background: #252525;
                width: 2px;
                height: 2px;
            }
            QTabWidget::pane {
                background: #1a1a1a;
                border: 1px solid #363636;
                border-top: none;
            }
            QTabBar {
                background: #1a1a1a;
            }
            QTabBar::tab {
                background: #252525;
                color: #888888;
                padding: 6px 13px;
                min-width: 58px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #00703C;
                color: #ffffff;
                font-weight: bold;
            }
            QTabBar::tab:hover:!selected {
                background: #363636;
                color: #e0e0e0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #363636;
                border-radius: 6px;
                margin-top: 10px;
                padding: 8px 4px 4px 4px;
                color: #4dbb78;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 6px;
            }
            QLabel {
                color: #e0e0e0;
            }
            QPushButton {
                background: #252525;
                color: #e0e0e0;
                padding: 4px 12px;
                min-height: 26px;
                border: 1px solid #363636;
                border-radius: 5px;
                font-weight: 500;
            }
            QPushButton:hover {
                background: #363636;
                border-color: #4dbb78;
                color: #ffffff;
            }
            QPushButton:pressed {
                background: #141414;
            }
            QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit {
                background: #252525;
                color: #e0e0e0;
                padding: 3px 6px;
                min-height: 24px;
                border: 1px solid #363636;
                border-radius: 4px;
            }
            QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus, QLineEdit:focus {
                border-color: #4dbb78;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 4px;
            }
            QRadioButton {
                color: #e0e0e0;
                spacing: 6px;
            }
            QRadioButton::indicator {
                width: 14px;
                height: 14px;
                border-radius: 7px;
                border: 2px solid #454545;
                background: #252525;
            }
            QRadioButton::indicator:checked {
                background: #1a5fa8;
                border-color: #1a5fa8;
            }
            QRadioButton::indicator:hover {
                border-color: #4dbb78;
            }
            QCheckBox {
                color: #e0e0e0;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
                border-radius: 3px;
                border: 2px solid #454545;
                background: #252525;
            }
            QCheckBox::indicator:checked {
                background: #00703C;
                border-color: #00703C;
            }
            QCheckBox::indicator:hover {
                border-color: #4dbb78;
            }
            QSlider::groove:horizontal {
                height: 5px;
                border-radius: 2px;
                background: #363636;
            }
            QSlider::handle:horizontal {
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: #4dbb78;
            }
            QSlider::sub-page:horizontal {
                background: #00703C;
                border-radius: 2px;
            }
            QScrollBar:vertical {
                background: #1a1a1a;
                width: 10px;
                margin: 0;
            }
            QScrollBar::handle:vertical {
                background: #454545;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0;
            }
            QScrollBar:horizontal {
                background: #1a1a1a;
                height: 10px;
                margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #454545;
                border-radius: 5px;
                min-width: 20px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0;
            }
            QMenuBar {
                background: #141414;
                color: #e0e0e0;
                border-bottom: 1px solid #363636;
                padding: 2px;
            }
            QMenuBar::item {
                padding: 4px 10px;
                border-radius: 4px;
            }
            QMenuBar::item:selected {
                background: #363636;
            }
            QMenu {
                background: #1a1a1a;
                color: #e0e0e0;
                border: 1px solid #363636;
                border-radius: 6px;
                padding: 4px;
            }
            QMenu::item {
                padding: 5px 20px;
                border-radius: 4px;
            }
            QMenu::item:selected {
                background: #252525;
                color: #4d8ec4;
            }
            QMenu::separator {
                height: 1px;
                background: #363636;
                margin: 4px 8px;
            }
            QStatusBar {
                background: #141414;
                color: #888888;
                border-top: 1px solid #363636;
                font-size: 11px;
            }
            QMessageBox {
                background: #1a1a1a;
            }
            QAbstractItemView {
                background: #252525;
                color: #e0e0e0;
                border: 1px solid #363636;
                selection-background-color: #00703C;
                selection-color: #ffffff;
                alternate-background-color: #1a1a1a;
            }
            QHeaderView::section {
                background: #1a1a1a;
                color: #4dbb78;
                border: none;
                border-bottom: 1px solid #363636;
                padding: 4px 6px;
                font-weight: bold;
            }
            QToolBar {
                background: #141414;
                border: none;
                spacing: 3px;
            }
            QToolButton {
                background: transparent;
                border: none;
                border-radius: 4px;
                padding: 3px;
                color: #e0e0e0;
            }
            QToolButton:hover {
                background: #363636;
            }
            QToolButton:checked, QToolButton:pressed {
                background: #252525;
            }
            QDockWidget {
                color: #e0e0e0;
                font-weight: bold;
            }
            QDockWidget::title {
                background: #141414;
                padding: 4px 6px;
                border-bottom: 1px solid #363636;
            }
            QListWidget {
                background: #1a1a1a;
                border: none;
                font-size: 11px;
            }
            QListWidget::item {
                padding: 5px 8px;
                border-bottom: 1px solid #252525;
            }
            QListWidget::item:selected {
                background: #00703C;
                color: #ffffff;
            }
            QListWidget::item:hover:!selected {
                background: #252525;
            }
        """)

        # ── application state ─────────────────────────────────────────────────
        self._image_paths: list[Path] = []          # all file paths (no data held)
        self._image_cache: dict[int, DM4Image] = {} # at most _MAX_CACHE loaded at once
        self._image_cache_fingerprints: dict[int, ImageFingerprint | None] = {}
        self._MAX_CACHE = 3
        self._img_idx: int = -1          # -1 = no image loaded yet
        self._click_buffer: list[tuple[float, float]] = []
        self._engine: MeasurementEngine = MeasurementEngine(pixel_size=1.0)
        self._px_overrides: dict[int, float] = {}  # manually set pixel size per image index
        self._last_distance_px: float = 0.0        # last measured distance in pixels
        self._contrast_states: dict[int, ContrastParams] = {}
        self._ann_states: dict[int, list] = {}   # per-image annotation snapshots
        self._image_load_thread: Optional[ImageLoadThread] = None  # active background load
        self._frame_proc_thread: Optional[FrameProcessThread] = None
        self._last_motion_shifts = None    # (n_frames, 2) shifts from last motion correction
        self._last_motion_start_frame = 1  # which frame the plot starts at
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
        self._batch_proc: Optional[dict] = None        # active batch-SAM state machine
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
        self._batch_model_proc = None   # state for batch_run_yolo / batch_run_unet
        # ── provenance / study instrumentation ───────────────────────────────
        self._current_invocation = "direct_gui"   # or "clu_nl"; set per action
        self._current_turn_id = None               # CLU turn id when clu_nl
        self._yolo_source_model = None             # {"path","sha256"} cached at load
        self._unet_source_model = None
        self._sam_source_model = None
        self._pending_yolo_anns: list = []

        # UNet state
        self._unet_predictor = None     # UNetPredictor, loaded on demand
        self._unet_thread: Optional[SAMThread] = None
        self._pending_unet_masks: list = []

        # ── central splitter ──────────────────────────────────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # canvas + movie bar
        self._canvas_widget = CanvasWidget()
        self._canvas_widget.canvas.store.on_change(self._on_store_changed_autosave)
        _canvas_container = QWidget()
        _canvas_vbox = QVBoxLayout(_canvas_container)
        _canvas_vbox.setContentsMargins(0, 0, 0, 0)
        _canvas_vbox.setSpacing(0)
        _canvas_vbox.addWidget(self._canvas_widget, 1)
        self._movie_bar = self._build_movie_bar()
        _canvas_vbox.addWidget(self._movie_bar)
        splitter.addWidget(_canvas_container)

        # right control panel
        control = QTabWidget()
        control.setMinimumWidth(320)

        self._contrast_panel = ContrastPanel()
        self._ann_panel      = AnnotationPanel()
        self._meas_panel     = MeasurementPanel()
        self._export_panel   = ExportPanel()
        self._sam_panel      = SAMPanel()
        self._yolo_panel     = YOLOPanel()
        self._unet_panel     = UNetPanel()
        self._train_panel    = TrainPanel()

        # Unified segmentation panel (SAM / YOLO / UNet selector)
        self._seg_panel = SegmentationPanel(
            self._sam_panel, self._yolo_panel, self._unet_panel
        )
        self._seg_panel.accept_all_requested.connect(self._on_seg_accept)
        self._seg_panel.reject_all_requested.connect(self._on_seg_reject)

        def _make_tab_wrapper(panel):
            """Wrap a panel in a scrollable container pre-tagged for plugin injection."""
            from PyQt6.QtWidgets import QScrollArea as _SA2
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.setSpacing(0)
            lay.addWidget(panel)
            lay.addStretch()
            inner._plugin_inject_layout = lay
            scroll = _SA2()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)
            scroll._plugin_inject_layout = lay   # tag on scroll too so lookup finds it
            return scroll

        # Annotate tab: manual annotation tools (CryoBLOB and other detectors inject here)
        _annotate_wrapper = _make_tab_wrapper(self._ann_panel)

        # Segment tab: AI-assisted segmentation (SAM / YOLO / UNet)
        _segment_wrapper = _make_tab_wrapper(self._seg_panel)

        _measure_wrapper = _make_tab_wrapper(self._meas_panel)
        _export_wrapper  = _make_tab_wrapper(self._export_panel)
        _train_wrapper   = _make_tab_wrapper(self._train_panel)

        control.addTab(self._contrast_panel, "Contrast")
        control.addTab(_annotate_wrapper,    "Annotate")
        control.addTab(_segment_wrapper,     "Segment")
        control.addTab(_measure_wrapper,     "Measure")
        control.addTab(_export_wrapper,      "Export")
        control.addTab(_train_wrapper,       "Train")

        # ── plugin tabs / workflow injection ──────────────────────────────────────
        from acorn.gui.context import AcornContext
        from acorn.plugin_loader import discover_plugins
        from acorn.plugin_base import WORKFLOW_STAGES
        from PyQt6.QtWidgets import QGroupBox, QVBoxLayout as _VBox, QScrollArea as _SA
        self._context = AcornContext(self)
        self._plugins = discover_plugins(self._context)

        # Build a tab-index map so we can inject into workflow tabs by name
        _tab_index = {control.tabText(i): i for i in range(control.count())}
        self._floating_plugins = []   # [(plugin, panel)] wired into docks after menus exist

        for plugin in self._plugins:
            try:
                panel = plugin.create_panel()
                if panel is None:
                    continue
                if getattr(plugin, "FLOATING", False):
                    # Defer dock + View-menu toggle until after _build_menus()
                    self._floating_plugins.append((plugin, panel))
                    continue
                stage = plugin.WORKFLOW_STAGE
                if stage in WORKFLOW_STAGES and stage in _tab_index:
                    # Inject into the existing workflow tab as a labeled section
                    target_widget = control.widget(_tab_index[stage])
                    # Wrap target in a scroll+vbox if it isn't already one we control
                    if not hasattr(target_widget, "_plugin_inject_layout"):
                        wrapper = _SA()
                        wrapper.setWidgetResizable(True)
                        inner = QWidget()
                        inner._plugin_inject_layout = _VBox(inner)
                        inner._plugin_inject_layout.setContentsMargins(0, 0, 0, 0)
                        inner._plugin_inject_layout.addWidget(target_widget)
                        inner._plugin_inject_layout.addStretch()
                        wrapper.setWidget(inner)
                        # Replace tab with wrapper
                        idx = _tab_index[stage]
                        control.removeTab(idx)
                        control.insertTab(idx, wrapper, stage)
                        control.setCurrentIndex(idx)
                        target_widget = inner
                        _tab_index[stage] = idx
                    inject_layout = target_widget._plugin_inject_layout
                    # Remove trailing stretch, add section, re-add stretch
                    count = inject_layout.count()
                    if count and inject_layout.itemAt(count - 1).spacerItem():
                        inject_layout.takeAt(count - 1)
                    if plugin.WORKFLOW_SECTION_LABEL:
                        box = QGroupBox(plugin.WORKFLOW_SECTION_LABEL)
                        bl = _VBox(box)
                        bl.setContentsMargins(4, 4, 4, 4)
                        bl.addWidget(panel)
                        inject_layout.addWidget(box)
                    else:
                        inject_layout.addWidget(panel)
                    inject_layout.addStretch()
                else:
                    # Unknown or no stage — own tab
                    control.addTab(panel, plugin.TAB_LABEL or plugin.PLUGIN_ID)
            except Exception as _plugin_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Plugin %s failed to create panel: %s", plugin.PLUGIN_ID, _plugin_exc
                )

        self._control_tabs = control
        splitter.addWidget(control)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([9999, 400])   # initial: canvas gets all extra, panel starts at 400px

        # ── workspace switcher ────────────────────────────────────────────────
        # Snapshot every tab now that core tabs and plugin tabs are both in place.
        # Switching workspaces removes and re-inserts tabs from this registry, so
        # the panel widgets themselves are never rebuilt and never lose state.
        self._all_tabs = [
            (control.tabText(i), control.widget(i)) for i in range(control.count())
        ]
        self._workspace_prefs = load_workspace_prefs()
        self._active_workspace = self._workspace_prefs.last_workspace
        self._workspace_bar = WorkspaceBar(self)
        self._workspace_bar.workspace_selected.connect(self.set_workspace)

        _shell = QWidget()
        _shell_v = QVBoxLayout(_shell)
        _shell_v.setContentsMargins(0, 0, 0, 0)
        _shell_v.setSpacing(0)
        _shell_v.addWidget(self._workspace_bar)
        _shell_v.addWidget(splitter, 1)

        self.setCentralWidget(_shell)

        # ── status bar ────────────────────────────────────────────────────────
        self._statusbar = QStatusBar()
        self.setStatusBar(self._statusbar)
        self._statusbar.showMessage("Ready")

        # Permanent pixel-size widget — always visible on the right of the status bar
        self._px_btn = QPushButton("px: —")
        self._px_btn.setFlat(True)
        self._px_btn.setToolTip("Click to set pixel size manually")
        self._px_btn.setStyleSheet(
            "QPushButton { color: #888888; font-size: 11px; padding: 0 6px; border: none; }"
            "QPushButton:hover { color: #ffffff; text-decoration: underline; }"
        )
        self._px_btn.clicked.connect(self._on_edit_pixel_size)
        self._statusbar.addPermanentWidget(self._px_btn)

        # ── image list dock ───────────────────────────────────────────────────
        self._image_list = QListWidget()
        self._image_list.setMinimumWidth(120)
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
        # Wire floating-dock plugins now that the View menu exists
        self._wire_floating_plugins()
        # Let plugins register menu items after core menus are built
        for plugin in self._plugins:
            try:
                plugin.setup_menus(self.menuBar())
            except Exception as _plugin_exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Plugin %s menu setup failed: %s", plugin.PLUGIN_ID, _plugin_exc
                )

        # Apply the saved workspace now that tabs, docks and menus all exist.
        self._apply_workspace(self._active_workspace, persist=False)

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
        self._export_panel.batch_cancel_requested.connect(self._on_batch_cancel)
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
        self._sam_panel.neg_box_prompt_mode_set.connect(self._on_sam_neg_box_mode)
        self._sam_panel.scribble_mode_set.connect(self._on_sam_scribble_mode)
        self._sam_panel.scribble_neg_mode_set.connect(self._on_sam_neg_scribble_mode)
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
        self._yolo_panel.accept_all_requested.connect(self._on_yolo_accept)
        self._yolo_panel.reject_all_requested.connect(self._on_yolo_reject)
        self._yolo_panel.batch_requested.connect(self._on_yolo_batch)

        # UNet panel signals
        self._unet_panel.load_model_requested.connect(self._on_unet_load_model)
        self._unet_panel.segment_requested.connect(self._on_unet_segment)
        self._unet_panel.accept_all_requested.connect(self._on_unet_accept)
        self._unet_panel.reject_all_requested.connect(self._on_unet_reject)
        self._unet_panel.batch_requested.connect(self._on_unet_batch)

        # Train panel signals
        self._train_panel.train_requested.connect(self._on_train_requested)
        self._train_panel.cancel_requested.connect(self._on_train_cancel)
        self._train_panel.load_yolo_requested.connect(self._on_train_load_yolo)
        self._train_panel.load_unet_requested.connect(self._on_train_load_unet)
        self._train_thread: Optional[SAMThread] = None
        self._train_proc = None

        # LLM assistant action dispatcher
        self._context.action_requested.connect(self._on_action_requested)

        # Intercept key events from all child widgets (e.g. matplotlib canvas
        # consumes key events and never lets them reach MainWindow.keyPressEvent)
        QApplication.instance().installEventFilter(self)

    # ── menu setup ────────────────────────────────────────────────────────────

    def _wire_floating_plugins(self) -> None:
        """Place FLOATING plugin panels in movable docks with a View-menu toggle."""
        _areas = {
            "left":   Qt.DockWidgetArea.LeftDockWidgetArea,
            "right":  Qt.DockWidgetArea.RightDockWidgetArea,
            "top":    Qt.DockWidgetArea.TopDockWidgetArea,
            "bottom": Qt.DockWidgetArea.BottomDockWidgetArea,
        }
        self._plugin_docks = {}
        for plugin, panel in getattr(self, "_floating_plugins", []):
            try:
                title = plugin.FLOATING_TITLE or plugin.TAB_LABEL or plugin.PLUGIN_ID
                d = QDockWidget(title, self)
                d.setWidget(panel)
                d.setFeatures(
                    QDockWidget.DockWidgetFeature.DockWidgetMovable
                    | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                    | QDockWidget.DockWidgetFeature.DockWidgetClosable,
                )
                if plugin.FLOATING_MIN_WIDTH:
                    d.setMinimumWidth(plugin.FLOATING_MIN_WIDTH)
                area = _areas.get(plugin.FLOATING_AREA or "right",
                                  Qt.DockWidgetArea.RightDockWidgetArea)
                self.addDockWidget(area, d)
                d.hide()
                self._context.register_menu_action(
                    "View", title,
                    lambda checked=False, dock=d: dock.setVisible(not dock.isVisible()),
                    shortcut=plugin.FLOATING_SHORTCUT or None,
                )
                self._plugin_docks[plugin.PLUGIN_ID] = d
            except Exception as _exc:
                import logging as _logging
                _logging.getLogger(__name__).warning(
                    "Floating plugin %s failed to dock: %s", plugin.PLUGIN_ID, _exc
                )

    # ── workspaces ────────────────────────────────────────────────────────────
    # A workspace decides which control tabs are visible and which tool docks are
    # open. It never touches loaded images, annotations, or models held in memory.

    @property
    def active_workspace(self) -> str:
        """Id of the workspace currently applied (see acorn.gui.workspaces)."""
        return self._active_workspace

    def set_workspace(self, wid: str) -> None:
        """Switch to workspace `wid` and remember the choice for next launch."""
        self._apply_workspace(wid, persist=True)

    def _tab_is_unclaimed(self, label: str) -> bool:
        """True if no workspace lists this tab — such tabs stay visible everywhere."""
        return not any(label in ws.tabs for ws in WORKSPACES)

    def _apply_workspace(self, wid: str, persist: bool = True) -> None:
        ws = workspace_by_id(wid)
        if ws is None:
            ws = workspace_by_id(DEFAULT_WORKSPACE)
            wid = ws.wid
        self._active_workspace = wid
        self._show_all_panels = False

        self._rebuild_tabs(visible=[
            label for label, _w in self._all_tabs
            if label in ws.tabs or self._tab_is_unclaimed(label)
        ], order=ws.tabs)
        self._apply_workspace_docks(ws)

        self._workspace_bar.set_all_shown(False)
        self._workspace_bar.set_active(wid)
        if hasattr(self, "_statusbar"):
            self._statusbar.showMessage(f"{ws.label} — {ws.tagline}", 4000)

        if persist:
            self._workspace_prefs.last_workspace = wid
            self._workspace_prefs.show_all = False
            save_workspace_prefs(self._workspace_prefs)

    def _rebuild_tabs(self, visible: list[str], order: tuple[str, ...] = ()) -> None:
        """
        Show exactly `visible`, with the labels in `order` first.

        Tabs are removed from and re-inserted into the same QTabWidget; the panel
        widgets are held in self._all_tabs and are never recreated, so scroll
        position, entered values and loaded models all survive a switch.
        """
        tabs = self._control_tabs
        current_label = tabs.tabText(tabs.currentIndex()) if tabs.count() else ""

        tabs.blockSignals(True)
        while tabs.count():
            tabs.removeTab(0)

        def sort_key(item):
            label = item[0]
            return (order.index(label) if label in order else len(order))

        for label, widget in sorted(self._all_tabs, key=sort_key):
            if label in visible:
                tabs.addTab(widget, label)
        tabs.blockSignals(False)

        # Keep the user on the same tab if it survived the switch.
        for i in range(tabs.count()):
            if tabs.tabText(i) == current_label:
                tabs.setCurrentIndex(i)
                break
        else:
            if tabs.count():
                tabs.setCurrentIndex(0)

    def _apply_workspace_docks(self, ws) -> None:
        """Open the docks this workspace owns; put other workspaces' docks away."""
        owned_by_any = {d for w in WORKSPACES for d in w.docks}
        for plugin_id, dock in getattr(self, "_plugin_docks", {}).items():
            if plugin_id in ALWAYS_AVAILABLE_DOCKS:
                continue                       # e.g. the assistant — user's choice stands
            if plugin_id in ws.docks:
                dock.show()
            elif plugin_id in owned_by_any:
                dock.hide()

    def show_all_panels(self) -> None:
        """Escape hatch: every tab and every dock at once, ignoring the workspace."""
        self._show_all_panels = True
        self._rebuild_tabs(visible=[label for label, _w in self._all_tabs])
        for dock in getattr(self, "_plugin_docks", {}).values():
            dock.show()
        self._workspace_bar.set_all_shown(True)
        self._workspace_prefs.show_all = True
        save_workspace_prefs(self._workspace_prefs)
        self._statusbar.showMessage(
            "Showing every panel — pick a workspace above to narrow it down again", 6000
        )

    def _show_welcome_again(self) -> None:
        """View ▸ Welcome Screen — reopen the chooser on demand."""
        dlg = WelcomeDialog(self)
        if dlg.exec():
            self._apply_workspace(dlg.chosen_workspace, persist=True)
        if dlg.dont_show_again:
            self._workspace_prefs.welcome_seen = True
            save_workspace_prefs(self._workspace_prefs)

    def _maybe_show_welcome(self) -> None:
        """First launch only: name the five workspaces instead of hiding them."""
        if self._workspace_prefs.welcome_seen:
            return
        dlg = WelcomeDialog(self)
        accepted = dlg.exec()
        self._workspace_prefs.welcome_seen = dlg.dont_show_again or bool(accepted)
        if accepted:
            self._apply_workspace(dlg.chosen_workspace, persist=True)
        else:
            save_workspace_prefs(self._workspace_prefs)

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

        # Workspaces first — the same five choices as the bar at the top of the window.
        ws_menu = view_menu.addMenu("Workspace")
        for _ws in WORKSPACES:
            _a = ws_menu.addAction(f"{_ws.label} — {_ws.tagline}")
            if _ws.shortcut:
                _a.setShortcut(_ws.shortcut)
            _a.triggered.connect(lambda _c=False, w=_ws.wid: self.set_workspace(w))
        view_menu.addSeparator()

        show_all_a = view_menu.addAction("Show Every Panel")
        show_all_a.setShortcut("Ctrl+Shift+E")
        show_all_a.setStatusTip("Ignore the workspace and show every tab and tool at once")
        show_all_a.triggered.connect(self.show_all_panels)

        welcome_a = view_menu.addAction("Welcome Screen…")
        welcome_a.setStatusTip("Show the five workspaces again")
        welcome_a.triggered.connect(self._show_welcome_again)
        view_menu.addSeparator()

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
        dlg = QFileDialog(self, "Open image file(s)", str(Path.home()))
        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
        dlg.setNameFilters([
            "All supported (*.dm4 *.tif *.tiff *.mrc *.mrcs *.emd *.h5 *.hdf5 *.png *.jpg *.jpeg)",
            "DM4 (*.dm4)",
            "TIFF (*.tif *.tiff)",
            "MRC (*.mrc *.mrcs)",
            "EMD / HDF5 (*.emd *.h5 *.hdf5)",
            "Images (*.png *.jpg *.jpeg)",
            "All files (*)",
        ])
        if dlg.exec():
            paths = dlg.selectedFiles()
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
                "Supported: .dm4, .tif/.tiff, .mrc/.mrcs, .emd, .h5/.hdf5, .png, .jpg/.jpeg"
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
        return self._context.sidecar_path_for_index(idx)

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
        # Don't autosave while an image load is in flight: _img_idx has advanced
        # but the canvas store may still hold the previous image's annotations,
        # which would write them to the new index's sidecar.
        if self._image_load_thread is not None and self._image_load_thread.isRunning():
            return
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
                "version": 4,   # v4 adds the per-annotation provenance block
                "annotations": [asdict(a) for a in anns],
                "pixel_size_nm": self._px_overrides.get(idx),
                "exclude_zone": list(ez) if ez else None,
                "crop_region": list(cr) if cr else None,
            }
            _atomic_write_text(path, json.dumps(data))
        except OSError:
            pass  # NAS write failure — silently skip, in-memory state is preserved

        # PNG overlay save is slow (PIL encode + disk write); run off the main thread
        # so the GUI never freezes waiting for it.
        norm = self._canvas_widget.canvas.norm_image
        if norm is None or idx < 0 or idx >= len(self._image_paths):
            return
        src_path  = self._image_paths[idx]
        ann_snap  = list(self._canvas_widget.canvas.store)  # snapshot for the thread
        norm_copy = norm.copy()
        import threading
        threading.Thread(
            target=self._save_annotated_overlay_bg,
            args=(norm_copy, src_path, ann_snap),
            daemon=True,
        ).start()

    @staticmethod
    def _save_annotated_overlay_bg(norm, src_path, anns) -> None:
        """Save annotated overlay PNG — runs on a background thread, no GUI access."""
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

        for ann in anns:
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
            _atomic_write_text(Path(path), json.dumps(data, indent=2))
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
        self._image_cache_fingerprints.clear()
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
        self._reset_pending_segmentation()
        self._canvas_widget.set_nav_enabled(False)
        self._sync_image_list(idx)
        self._canvas_widget.update_nav_label(idx + 1, len(self._image_paths))

        if idx in self._image_cache and self._cached_image_is_current(idx):
            # Already cached — complete immediately without a thread.
            self._finish_switch(idx, self._image_cache[idx])
            return
        if idx in self._image_cache:
            self._image_cache.pop(idx, None)
            self._image_cache_fingerprints.pop(idx, None)

        path = self._image_paths[idx]
        # Resolve contrast params now (before thread starts) so the
        # background thread can pre-compute the normalised image.
        # For cryo-EM formats with no saved state, default to bandpass so the
        # background thread computes the norm with the correct params from the start.
        contrast_params = (
            self._contrast_states.get(idx)
            or (ContrastParams(method=DEFAULT_EM_CONTRAST) if path.suffix.lower() in EM_EXTS else self._contrast_panel.params())
        )
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
            self._image_cache_fingerprints.pop(oldest, None)
        self._image_cache[idx] = img
        self._image_cache_fingerprints[idx] = _image_file_fingerprint(img.filepath) if img.filepath else None
        # Only render if this is still the current image (user may not have switched).
        if idx == self._img_idx:
            self._finish_switch(idx, img, precomputed_norm=norm)
        else:
            self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)

    def _on_image_load_error(self, idx: int, message: str) -> None:
        self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)
        self._statusbar.showMessage(f"Error loading image {idx}: {message}")

    def _cached_image_is_current(self, idx: int) -> bool:
        if idx not in self._image_cache or idx >= len(self._image_paths):
            return False
        cached = self._image_cache[idx]
        path = cached.filepath if cached.filepath else self._image_paths[idx]
        cached_fp = self._image_cache_fingerprints.get(idx)
        current_fp = _image_file_fingerprint(path)
        return cached_fp is not None and cached_fp == current_fp

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
        elif img.filepath and img.filepath.suffix.lower() in EM_EXTS:
            _em_default = ContrastParams(method=DEFAULT_EM_CONTRAST)
            self._contrast_panel.set_params(_em_default)
            self._contrast_states[idx] = _em_default
        elif img.is_color:
            pass  # color images — display as-is, don't touch the contrast panel
        # Pass pre-computed norm so canvas.load_image skips apply_contrast on the main thread
        canvas.load_image(img, self._contrast_panel.params(), precomputed_norm=precomputed_norm)
        self._update_movie_bar(img)

        # Invalidate SAM embedding cache — new image, old embedding is stale
        self._sam_img8_cache = None
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
            color = "#4dbb78"
        elif manually_set:
            text  = f"{px_nm:.4f} nm/px  (manual)"
            color = "#4d8ec4"
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
        """Right-click menu on the image list."""
        from PyQt6.QtWidgets import QMenu
        item = self._image_list.itemAt(pos)
        if item is None:
            return
        row = self._image_list.row(item)
        menu = QMenu(self)
        clear_act  = menu.addAction(f"Clear annotations: {item.text()}")
        remove_act = menu.addAction(f"Remove from session: {item.text()}")
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

        elif action == remove_act:
            reply = QMessageBox.question(
                self, "Remove image",
                f"Remove {item.text()} from this session?\n(The file will not be deleted.)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self._remove_image_at(row)

    def _remove_image_at(self, row: int) -> None:
        """Remove image at *row* from the session without deleting the file."""
        if row < 0 or row >= len(self._image_paths):
            return

        # Rebuild index-keyed dicts shifting keys above row down by one
        self._image_paths.pop(row)
        self._ann_states     = {(k if k < row else k - 1): v
                                 for k, v in self._ann_states.items() if k != row}
        self._contrast_states = {(k if k < row else k - 1): v
                                  for k, v in self._contrast_states.items() if k != row}
        self._px_overrides    = {(k if k < row else k - 1): v
                                  for k, v in getattr(self, "_px_overrides", {}).items()
                                  if k != row}
        stem = None
        if hasattr(self, "_export_queue"):
            # Remove from export queue if present (queue stores dicts with 'stem' key)
            pass  # queue is stem-based, not index-based; no action needed

        self._populate_image_list()

        if not self._image_paths:
            self._img_idx = -1
            self._canvas_widget.canvas.clear()
            self._canvas_widget.update_nav_label(0, 0)
            self._canvas_widget.set_nav_enabled(False)
            return

        # Navigate to a valid index
        new_idx = min(row, len(self._image_paths) - 1)
        self._img_idx = -1  # force reload
        self._switch_to(new_idx)
        self._canvas_widget.set_nav_enabled(len(self._image_paths) > 1)

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

    # ── movie bar ─────────────────────────────────────────────────────────────












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
        if (self._sam_mode in ("scribble", "scribble_neg")
                and self._sam_predictor is not None
                and self._sam_predictor.is_loaded):
            self._on_sam_scribble_commit(pts, positive=(self._sam_mode == "scribble"))
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


    # ── annotation actions ────────────────────────────────────────────────────

    def _on_undo(self) -> None:
        removed = self._canvas_widget.canvas.store.undo()
        if removed:
            self._statusbar.showMessage(f"Removed: {removed.type}")

    def _on_clear_annotations(self) -> None:
        store = self._canvas_widget.canvas.store
        n = len(store)
        if n == 0:
            return
        resp = QMessageBox.question(
            self, "Clear all annotations?",
            f"Remove all {n} annotation(s) on this image? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if resp != QMessageBox.StandardButton.Yes:
            return
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








    def _on_import_negatives(self) -> None:
        """Open a file/folder picker and add selected images to the queue as negatives."""
        from acorn.core.dm4_loader import scan_folder

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select negative images",
            str(Path.home()),
            "Images (*.dm4 *.tif *.tiff *.mrc *.mrcs *.emd *.h5 *.hdf5 *.png *.jpg *.jpeg)",
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


    def _on_batch_cancel(self) -> None:
        if self._batch_export_thread is not None and self._batch_export_thread.isRunning():
            self._batch_export_thread.cancel()
            self._export_panel.set_train_status("Cancelling export — finishing current image…")




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












    # ── batch SAM state machine ───────────────────────────────────────────────


    def _batch_next_image(self) -> None:
        bp = self._batch_proc
        if bp is None:
            return
        if not bp["queue"]:
            n, total = bp["processed"], bp["total"]
            self._batch_proc = None
            self._statusbar.showMessage(f"Batch SAM complete — {n}/{total} images processed and queued.")
            return
        target = bp["queue"][0]
        # Use the image_loaded signal as the trigger to start SAM on this image
        self._context.image_loaded.connect(self._batch_on_image_loaded)
        self._on_image_list_select(target)

    def _batch_on_image_loaded(self, img) -> None:
        bp = self._batch_proc
        try:
            self._context.image_loaded.disconnect(self._batch_on_image_loaded)
        except Exception:
            pass
        if bp is None or not bp["queue"]:
            return
        if self._img_idx != bp["queue"][0]:
            return  # spurious signal from a different navigation
        # Configure SAM label and points_per_side
        label = bp["label"]
        if label:
            combo = self._sam_panel._label_combo
            txt_idx = combo.findText(label)
            if txt_idx < 0:
                combo.addItem(label)
                txt_idx = combo.count() - 1
            combo.setCurrentIndex(txt_idx)
        self._sam_panel._pts_per_side.setValue(bp["points_per_side"])
        n, total = bp["processed"], bp["total"]
        self._statusbar.showMessage(f"Batch SAM: running on image {n+1}/{total} — {img.filename if hasattr(img,'filename') else ''}…")
        self._on_sam_auto_segment(_batch_done_cb=self._batch_after_sam)


    # ── batch YOLO / UNet state machine ───────────────────────────────────────
    # Reuses the proven per-image navigate → run → accept path (same as manual and
    # as batch SAM), so predictions are identical to a manual run and auto-save to
    # each image's .acorn.json sidecar as editable annotations. Kept separate from
    # the SAM batch methods so that path is untouched.

    def _start_batch_model(self, params: dict, model: str) -> None:
        """Run a loaded YOLO/UNet model over all images, adding editable predictions."""
        if self._batch_model_proc is not None:
            self._statusbar.showMessage("A batch model run is already in progress.")
            return
        if model == "yolo":
            if self._yolo_predictor is None or not self._yolo_predictor.is_loaded:
                self._statusbar.showMessage("Batch YOLO: load a YOLO model first.")
                self._report_clu("Batch YOLO could not run — no YOLO model is loaded. Tell the "
                                 "user to load_yolo first. Do NOT claim anything was detected.")
                return
        elif model == "unet":
            if self._unet_predictor is None or not self._unet_predictor.is_loaded:
                self._statusbar.showMessage("Batch UNet: load a UNet model first.")
                self._report_clu("Batch UNet could not run — no UNet model is loaded. Tell the "
                                 "user to load_unet first. Do NOT claim anything was segmented.")
                return
        else:
            return

        label          = params.get("label", "")
        segmentation   = bool(params.get("segmentation", False))
        skip_annotated = bool(params.get("skip_annotated", True))
        queue_after    = bool(params.get("queue_after", False))

        # Set the label on the relevant panel so predictions are labelled correctly.
        if label:
            combo = (self._yolo_panel if model == "yolo" else self._unet_panel)._label_combo
            idx = combo.findText(label)
            if idx < 0:
                combo.addItem(label)
                idx = combo.count() - 1
            combo.setCurrentIndex(idx)

        queue: list[int] = []
        for i in range(len(self._image_paths)):
            if skip_annotated:
                existing = self._ann_states.get(i, [])
                if i == self._img_idx:
                    existing = list(self._canvas_widget.canvas.store)
                if existing:
                    continue
            queue.append(i)

        if not queue:
            self._statusbar.showMessage(
                f"Batch {model.upper()}: all images already annotated — nothing to do."
            )
            self._report_clu(f"Batch {model.upper()}: every image already has annotations, so "
                             "nothing was run. Do NOT claim new predictions were made.")
            return

        self._batch_model_proc = {
            "queue":        queue,
            "model":        model,
            "segmentation": segmentation,
            "queue_after":  queue_after,
            "label":        label,
            "processed":    0,
            "total":        len(queue),
            "detections":   0,
        }
        self._statusbar.showMessage(
            f"Batch {model.upper()}: starting — {len(queue)} image(s) to process…"
        )
        self._batch_model_next()

    def _batch_model_next(self) -> None:
        bp = self._batch_model_proc
        if bp is None:
            return
        if not bp["queue"]:
            n, total, det, model = bp["processed"], bp["total"], bp["detections"], bp["model"]
            queued = bp["queue_after"]
            self._batch_model_proc = None
            msg = (f"Batch {model.upper()} complete — {det} object(s) added as editable "
                   f"annotations across {n}/{total} image(s).")
            if not queued:
                msg += " Review and correct them, then queue for training."
            self._statusbar.showMessage(msg)
            self._report_clu(
                f"Batch {model.upper()} finished: {det} prediction(s) were added as EDITABLE "
                f"annotations to {n} image(s) and saved to their sidecars. "
                + ("They were auto-queued for training export." if queued else
                   "They were NOT queued for training — a fresh model produces false positives, so "
                   "tell the user to review/correct the predictions (they can edit or reject them per "
                   "image) BEFORE finalizing the dataset. Do NOT claim the predictions are correct.")
            )
            return
        target = bp["queue"][0]
        if target == self._img_idx:
            # Already on this image — navigation would no-op (no image_loaded emit),
            # so run directly instead of waiting for a signal that never fires.
            self._batch_model_on_loaded(None)
        else:
            self._context.image_loaded.connect(self._batch_model_on_loaded)
            self._on_image_list_select(target)

    def _batch_model_on_loaded(self, _img) -> None:
        bp = self._batch_model_proc
        try:
            self._context.image_loaded.disconnect(self._batch_model_on_loaded)
        except Exception:
            pass
        if bp is None or not bp["queue"]:
            return
        if self._img_idx != bp["queue"][0]:
            return  # spurious signal from a different navigation
        n, total = bp["processed"], bp["total"]
        self._statusbar.showMessage(f"Batch {bp['model'].upper()}: image {n+1}/{total}…")
        if bp["model"] == "yolo":
            self._run_yolo(bp["segmentation"], on_complete=self._batch_model_after)
        else:
            self._on_unet_segment(on_complete=self._batch_model_after)

    def _batch_model_after(self) -> None:
        bp = self._batch_model_proc
        if bp is None:
            return
        # Predictions are already in the store as pending; accept keeps them (editable).
        if bp["model"] == "yolo":
            bp["detections"] += len(self._pending_yolo_anns)
            self._on_yolo_accept()
        else:
            bp["detections"] += len(self._pending_unet_masks)
            self._on_unet_accept()
        # Persist this image's predictions to its sidecar now (don't rely on debounce).
        self._do_autosave()
        if bp["queue_after"]:
            self._on_queue_image("")
        if bp["queue"]:
            bp["queue"].pop(0)
        bp["processed"] += 1
        self._batch_model_next()






    def _on_seg_accept(self, tool: str) -> None:
        """Shared Accept All button in SegmentationPanel — dispatch to active tool."""
        if tool == "sam":
            self._on_sam_accept()
        elif tool == "yolo":
            self._on_yolo_accept()
        elif tool == "unet":
            self._on_unet_accept()

    def _on_seg_reject(self, tool: str) -> None:
        """Shared Reject All button in SegmentationPanel — dispatch to active tool."""
        if tool == "sam":
            self._on_sam_reject()
        elif tool == "yolo":
            self._on_yolo_reject()
        elif tool == "unet":
            self._on_unet_reject()



    def _reset_pending_segmentation(self) -> None:
        """Forget all pending SAM/YOLO/UNet state when switching images.

        The pending masks themselves stay in (and are autosaved with) the image
        being left; we only clear the tracking lists/preview/prompt points so a
        later Reject All on the new image can't act on stale references.
        """
        self._pending_sam_masks.clear()
        self._pending_yolo_anns.clear()
        self._pending_unet_masks.clear()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._clear_sam_point_artists(blit=False)

    def _remove_pending_annotations(self, pending: list) -> None:
        """Remove specific pending annotations by identity (not via the undo stack).

        Using the undo stack would pop the most-recent annotations, which may be
        the user's own edits made after a detection ran — not the pending masks.
        """
        if not pending:
            return
        store = self._canvas_widget.canvas.store
        pending_ids = {id(a) for a in pending}
        store.replace_all([a for a in store if id(a) not in pending_ids])










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


    # ── YOLO handlers ─────────────────────────────────────────────────────────






    def _pred_context(self, origin: str):
        """Provenance context for a predictor's store.add calls — stamps origin,
        the cached source_model, the current invocation (direct_gui/clu_nl), and
        the active batch_id (if a batch of this origin is running)."""
        from acorn.core import provenance as _prov
        sm = {
            "yolo": self._yolo_source_model,
            "unet": self._unet_source_model,
            "sam":  self._sam_source_model,
        }.get(origin)
        bid = None
        bp = self._batch_model_proc
        if bp is not None and bp.get("model") == origin:
            bid = bp.get("batch_id")
        return _prov.provenance_context(
            origin,
            invocation=self._current_invocation,
            source_model=sm,
            batch_id=bid,
            turn_id=self._current_turn_id,
        )





    def _on_detect_atoms(self, params: dict) -> None:
        """Detect atomic columns (incl. moire lattices) and overlay them as circles."""
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._statusbar.showMessage("Detect atoms: no image loaded.")
            self._report_clu("detect_atoms could not run — no image is loaded. Tell the user to "
                             "open an atomic-resolution image first. Do NOT claim atoms were found.")
            return
        import numpy as np
        from acorn.core.contrast import apply_contrast
        try:
            from acorn.core.atom_detect import detect_atoms as _detect, estimate_lattice_spacing
        except Exception as e:
            self._statusbar.showMessage(f"Detect atoms: {e}")
            self._report_clu(f"Atom detection unavailable: {e}")
            return
        norm = np.clip(np.asarray(apply_contrast(img.raw, self._contrast_panel.params()), float), 0.0, 1.0)
        spacing = params.get("lattice_spacing_px")
        thr = float(params.get("threshold", 2.0))
        bright = bool(params.get("bright_atoms", True))
        cap = int(params.get("max_annotations", 4000))
        model_path = params.get("model_path")
        fill = bool(params.get("fill_gaps", True))   # lattice-guided recovery of dim-region atoms
        self._statusbar.showMessage("Detecting atomic columns…")
        try:
            if model_path:
                from acorn.core.atom_model import detect_atoms_model
                md = float(spacing) * 0.6 if spacing else 6.0
                coords = detect_atoms_model(norm, model_path, min_distance_px=md)
            else:
                coords = _detect(norm, lattice_spacing_px=spacing,
                                 min_distance_px=params.get("min_distance_px"),
                                 threshold=thr, bright_atoms=bright, fill_gaps=fill)
        except Exception as e:
            self._statusbar.showMessage(f"Detect atoms failed: {e}")
            self._report_clu(f"Atom detection failed: {e}")
            return
        n = len(coords)
        if spacing is None:
            try:
                spacing = float(estimate_lattice_spacing(norm))
            except Exception:
                spacing = 8.0
        r = max(2.0, float(spacing) / 4.0)
        from acorn.core.annotations import CircleAnnotation
        from acorn.core import provenance as _prov
        canvas = self._canvas_widget.canvas
        store = canvas.store
        canvas._loading = True
        try:
            with _prov.batch_transaction(origin="atom_detect",
                                         invocation=self._current_invocation,
                                         human_interaction_count=1):
                for (yy, xx) in coords[:cap]:
                    store.add(CircleAnnotation(cx=float(xx), cy=float(yy), r=r,
                                               color="#00E5FF", linewidth=1.0))
        finally:
            canvas._loading = False
        if canvas.renderer is not None:
            canvas.renderer.render_noblit(canvas.store, canvas)
        else:
            canvas.fig.canvas.draw_idle()
        self._autosave_timer.start()
        extra = "" if n <= cap else f" (showing {cap}; raise max_annotations to draw all)"
        self._statusbar.showMessage(f"Detected {n} atomic columns (est. spacing {spacing:.1f} px).{extra}")
        self._report_clu(f"Atom detection found {n} atomic columns (estimated lattice spacing "
                         f"{spacing:.1f} px) and added {min(n, cap)} as editable circle annotations."
                         f"{extra}")

    def _on_atom_statistics(self, params: dict) -> None:
        """Compute lattice statistics from detected atomic columns and save a CSV."""
        import numpy as np
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._statusbar.showMessage("Atom statistics: no image loaded.")
            self._report_clu("atom_statistics could not run — no image is loaded.")
            return
        try:
            from acorn.core import atom_stats as _stats
            from acorn.core.atom_detect import detect_atoms as _detect
            from acorn.core.annotations import CircleAnnotation
        except Exception as e:
            self._report_clu(f"Atom statistics unavailable: {e}")
            return
        # gather atoms already detected (origin=atom_detect), else detect now
        redetect = bool(params.get("redetect", False))
        coords = None
        if not redetect:
            pts = [(a.cy, a.cx) for a in self._canvas_widget.canvas.store
                   if isinstance(a, CircleAnnotation)
                   and getattr(getattr(a, "provenance", None), "origin", "") == "atom_detect"]
            if pts:
                coords = np.asarray(pts, float)
        from acorn.core.contrast import apply_contrast
        norm_img = np.clip(np.asarray(apply_contrast(img.raw, self._contrast_panel.params()), float), 0, 1)
        if coords is None or len(coords) < 5:
            coords = _detect(norm_img, threshold=2.0, fill_gaps=True)
        if len(coords) < 5:
            self._statusbar.showMessage("Atom statistics: too few atoms.")
            self._report_clu("Atom statistics: fewer than 5 atoms — nothing to compute. Run detect_atoms first.")
            return

        px_nm = params.get("pixel_size_nm")
        if px_nm is None:
            px_nm = float(getattr(self._engine, "pixel_size", 0.0)) or None
        self._statusbar.showMessage("Computing atom statistics…")
        try:
            summ = _stats.summarize(coords, pixel_size_nm=px_nm, image=norm_img)
            nn_med, _, d, _ = _stats.nearest_neighbour(coords)
            psi6 = _stats.bond_orientational_order(coords)
            cn = _stats.voronoi_coordination(coords)
            st = _stats.strain_field(coords)
        except Exception as e:
            self._statusbar.showMessage(f"Atom statistics failed: {e}")
            self._report_clu(f"Atom statistics failed: {e}")
            return

        # save per-atom CSV next to the image
        saved = ""
        try:
            import csv
            if 0 <= self._img_idx < len(self._image_paths):
                out = self._image_paths[self._img_idx].with_name(
                    self._image_paths[self._img_idx].stem + "_atom_stats.csv")
                with open(out, "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["x_px", "y_px", "nn_dist_px", "psi6", "coord_num",
                                "exx", "eyy", "exy", "rotation_deg"])
                    for i in range(len(coords)):
                        w.writerow([f"{coords[i,1]:.3f}", f"{coords[i,0]:.3f}",
                                    f"{d[i,0]:.3f}", f"{psi6[i]:.4f}", cn[i],
                                    f"{st['exx'][i]:.4f}", f"{st['eyy'][i]:.4f}",
                                    f"{st['exy'][i]:.4f}", f"{st['rotation_deg'][i]:.3f}"])
                saved = f" Per-atom CSV saved to {out.name}."
                # also save the 6-panel statistics maps next to the image
                try:
                    figp = self._image_paths[self._img_idx].with_name(
                        self._image_paths[self._img_idx].stem + "_atom_stats.png")
                    _stats.save_stats_figure(coords, str(figp), pixel_size_nm=px_nm, image=norm_img)
                    saved += f" Maps saved to {figp.name}."
                except Exception:
                    pass
        except OSError:
            pass

        sp_a = summ.get("lattice_spacing_A")
        sp_txt = f"{sp_a:.2f} A" if sp_a else f"{summ['lattice_spacing_px']:.1f} px"
        moire = (f", moire period {summ['moire_period_nm']:.1f} nm, twist ~{summ['twist_deg']:.1f}deg"
                 if summ.get("moire_period_nm") else "")
        self._statusbar.showMessage(
            f"{summ['n_atoms']} atoms | spacing {sp_txt} | psi6 {summ['psi6_mean']:.2f}{moire}.")
        self._report_clu(
            f"Atom statistics ({summ['n_atoms']} atoms): lattice spacing {sp_txt}; "
            f"nearest-neighbour {summ.get('nn_distance_A_mean', summ['nn_distance_px_mean']):.2f} "
            f"{'A' if sp_a else 'px'}; hexagonal order psi6 {summ['psi6_mean']:.2f}; median "
            f"coordination {summ['coord_number_median']:.0f}{moire}.{saved} "
            "NOTE: psi6 and nn-distance are reliable; the per-atom strain/coordination magnitudes are "
            "sensitive to missed atoms in dim regions and should be treated as qualitative.")

    def _on_train_atom_model(self, params: dict) -> None:
        """Self-label + train a U-Net atom-finder (background thread)."""
        import numpy as np
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._report_clu("train_atom_model: no image loaded. Open a STEM image first.")
            return
        from acorn.core.contrast import apply_contrast
        from acorn.core.annotations import CircleAnnotation
        norm = np.clip(np.asarray(apply_contrast(img.raw, self._contrast_panel.params()), float), 0, 1)
        pts = [(a.cy, a.cx) for a in self._canvas_widget.canvas.store
               if isinstance(a, CircleAnnotation)
               and getattr(getattr(a, "provenance", None), "origin", "") == "atom_detect"]
        if len(pts) >= 20:
            coords = np.asarray(pts, float)
            src = "corrected atom annotations"
        else:
            from acorn.core.atom_detect import detect_atoms as _detect
            coords = _detect(norm, threshold=2.0)
            src = "classical detector"
        if len(coords) < 20:
            self._report_clu("train_atom_model: too few seed atoms — run detect_atoms first.")
            return
        out = params.get("output_path")
        if not out and 0 <= self._img_idx < len(self._image_paths):
            p = self._image_paths[self._img_idx]
            out = str(p.with_name(p.stem + "_atom_unet.pt"))
        out = out or "/tmp/atom_unet.pt"
        epochs = int(params.get("epochs", 40))
        sigma = float(params.get("atom_sigma", 2.0))
        self._statusbar.showMessage(f"Training atom model on {len(coords)} seed atoms…")
        self._report_clu(f"Started training a U-Net atom-finder on {len(coords)} seed atoms (from the "
                         f"{src}, {epochs} epochs). This is a dedicated atom-heatmap trainer (separate "
                         f"from the UNet Train tab); live epoch/loss progress shows in the status bar at "
                         f"the bottom of the window (and the UNet panel's status line). Runs in the "
                         f"background (~1-2 min on GPU); the model saves to {out}. When done, detect with "
                         f"model_path set to that path.")

        def _run():
            from acorn.core.atom_model import train_atom_model

            def _log(m):
                th = getattr(self, "_atom_train_thread", None)
                if th is not None:
                    try:
                        th.status.emit(m)   # cross-thread signal -> status bar (safe)
                    except Exception:
                        pass

            return train_atom_model(norm, coords, out, atom_sigma=sigma, epochs=epochs, log=_log)

        def _done(info):
            self._statusbar.showMessage(f"Atom model trained → {out} (loss {info['final_loss']:.4f})")
            self._report_clu(f"Atom-finder training finished: saved to {out} (final loss "
                             f"{info['final_loss']:.5f}, {info['n_tiles']} tiles). Use it via "
                             f"detect_atoms or analyze_stem_atoms with model_path='{out}'.")

        def _err(msg):
            self._statusbar.showMessage(f"Atom model training failed: {msg}")
            self._report_clu(f"Atom model training failed: {msg}")

        self._atom_train_thread = SAMThread(_run, self)
        # live epoch progress -> status bar (and the YOLO/UNet-style status label if present)
        self._atom_train_thread.status.connect(lambda m: self._statusbar.showMessage(m))
        try:
            self._atom_train_thread.status.connect(self._unet_panel.set_status)
        except Exception:
            pass
        self._atom_train_thread.finished.connect(_done)
        self._atom_train_thread.error.connect(_err)
        self._atom_train_thread.start()

    # ── UNet handlers ─────────────────────────────────────────────────────────








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



    # ── LLM assistant action dispatcher ──────────────────────────────────────

    _CLU_RESULT_ACTIONS = {
        "spatial_analysis", "run_yolo_detect", "run_yolo_segment", "run_unet",
        "run_sam_auto", "export_masks", "finalize_dataset",
    }

    def _report_clu(self, message: str) -> None:
        """Report a real outcome back to a waiting CLU agent (once per action)."""
        if getattr(self, "_clu_result_pending", False):
            self._clu_result_pending = False
            try:
                self._context.report_action_result(message)
            except Exception:
                pass

    def _on_action_requested(self, action: str, params: dict) -> None:
        print(f"[_on_action_requested] action={action} params={params}", flush=True)
        self._clu_result_pending = action in self._CLU_RESULT_ACTIONS
        if action == "switch_workspace":
            # CLU was asked for something this workspace does not cover.
            wid = str(params.get("workspace") or "").strip().lower()
            if workspace_by_id(wid) is None:
                self._statusbar.showMessage(f"No workspace called '{wid}'", 4000)
            else:
                self.set_workspace(wid)
            return
        if action == "reload_annotations_from_disk":
            # A plugin (e.g. CryoBLOB) wrote new sidecars to disk. Evict our in-memory
            # per-image annotation cache for those images so the next switch reloads the
            # updated sidecar, and reload the CURRENT image immediately so it shows now.
            from pathlib import Path      # Path is imported locally elsewhere in this fn
            raw_paths = params.get("paths") or []
            wanted = {str(Path(p).expanduser().resolve()) for p in raw_paths} or None
            for idx, ip in enumerate(self._image_paths):
                try:
                    rp = str(Path(ip).expanduser().resolve())
                except Exception:
                    continue
                if wanted is None or rp in wanted:
                    self._ann_states.pop(idx, None)
            cur = self._img_idx
            if 0 <= cur < len(self._image_paths):
                sidecar = self._autoload_sidecar(cur)
                if sidecar is not None:
                    anns = sidecar[0]
                    self._ann_states[cur] = anns
                    self._canvas_widget.canvas.store.replace_all(anns)
                    self._canvas_widget.canvas.force_redraw()
            return
        if action == "run_sam_auto":
            label = params.get("label", "")
            if label:
                combo = self._sam_panel._label_combo
                idx = combo.findText(label)
                if idx < 0:
                    combo.addItem(label)
                    idx = combo.count() - 1
                combo.setCurrentIndex(idx)
            pts = params.get("points_per_side")
            if pts is not None:
                self._sam_panel._pts_per_side.setValue(int(pts))
            self._on_sam_auto_segment()

        elif action == "run_yolo_detect":
            label = params.get("label", "")
            if label:
                combo = self._yolo_panel._label_combo
                idx = combo.findText(label)
                if idx < 0:
                    combo.addItem(label)
                    idx = combo.count() - 1
                combo.setCurrentIndex(idx)
            self._on_yolo_detect()

        elif action == "run_yolo_segment":
            label = params.get("label", "")
            if label:
                combo = self._yolo_panel._label_combo
                idx = combo.findText(label)
                if idx < 0:
                    combo.addItem(label)
                    idx = combo.count() - 1
                combo.setCurrentIndex(idx)
            self._on_yolo_detect_seg()

        elif action == "accept_annotations":
            model = params.get("model", "all")
            if model in ("sam", "all") and self._pending_sam_masks:
                self._on_sam_accept()
            if model in ("yolo", "all") and self._pending_yolo_anns:
                self._on_yolo_accept()
            if model in ("unet", "all") and self._pending_unet_masks:
                self._on_unet_accept()

        elif action == "queue_for_export":
            self._on_queue_image("")

        elif action == "start_training":
            self._train_panel._on_train_clicked()

        elif action == "finalize_dataset":
            val_frac  = params.get("val_frac")
            test_frac = params.get("test_frac")
            if val_frac is not None:
                self._export_panel._val_frac.setValue(float(val_frac))
            if test_frac is not None:
                self._export_panel._test_frac.setValue(float(test_frac))
            self._export_panel._on_finalize()

        elif action == "load_sam":
            backend = params.get("backend")
            if backend:
                combo = self._sam_panel._backend_combo
                idx = combo.findData(backend)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._sam_panel._on_load_model()

        elif action == "spatial_analysis":
            dock = getattr(self, "_plugin_docks", {}).get("acorn_spatial")
            if dock is not None:
                dock.show(); dock.raise_()
                panel = dock.widget()
                if panel is not None and hasattr(panel, "run_from_clu"):
                    panel.run_from_clu(params.get("labels"))
                    self._report_clu("Spatial analysis complete (also shown in the "
                                     "Spatial panel):\n" + panel.clu_result_text())

        elif action == "load_yolo":
            self._yolo_panel._on_load_model()

        elif action == "load_unet":
            self._unet_panel._on_load_model()

        elif action == "run_unet":
            label = params.get("label", "")
            if label:
                combo = self._unet_panel._label_combo
                idx = combo.findText(label)
                if idx < 0:
                    combo.addItem(label)
                    idx = combo.count() - 1
                combo.setCurrentIndex(idx)
            self._on_unet_segment()

        elif action == "reject_annotations":
            model = params.get("model", "all")
            if model in ("sam", "all"):
                self._on_sam_reject()
            if model in ("yolo", "all"):
                self._on_yolo_reject()
            if model in ("unet", "all"):
                self._on_unet_reject()

        elif action == "undo_annotation":
            self._on_undo()

        elif action == "clear_annotations":
            self._on_clear_annotations()

        elif action == "set_contrast":
            import dataclasses
            from acorn.core.contrast import ContrastParams
            method = params.get("method", "percentile")
            p = self._contrast_panel.params()
            kwargs: dict = {"method": method}
            low  = params.get("low")
            high = params.get("high")
            if method == "percentile":
                if low  is not None: kwargs["low_pct"]  = float(low)
                if high is not None: kwargs["high_pct"] = float(high)
            elif method == "sigma":
                if low  is not None: kwargs["n_sigma"] = float(low)
            elif method == "adaptive":
                if low  is not None: kwargs["clip_limit"] = float(low)
            elif method == "bandpass":
                if low  is not None: kwargs["bp_low_sigma"]  = float(low)
                if high is not None: kwargs["bp_high_sigma"] = float(high)
            elif method == "fourier":
                if low  is not None: kwargs["fbp_hp_px"] = float(low)
                if high is not None: kwargs["fbp_lp_px"] = float(high)
            new_p = dataclasses.replace(p, **kwargs)
            self._contrast_panel.set_params(new_p)
            self._on_contrast_changed(new_p)

        elif action == "check_quality":
            self._on_check_quality()

        elif action == "next_image":
            self._on_next()

        elif action == "prev_image":
            self._on_prev()

        elif action == "go_to_image":
            target = int(params.get("index", 1)) - 1
            if 0 <= target < len(self._image_paths):
                self._on_image_list_select(target)

        elif action == "configure_training":
            tp = self._train_panel
            model_type = params.get("model_type")
            if model_type == "yolo":
                tp._yolo_radio.setChecked(True)
            elif model_type == "unet":
                tp._unet_radio.setChecked(True)
            dataset_dir = params.get("dataset_dir")
            if dataset_dir:
                tp._dir_edit.setText(dataset_dir)
                from pathlib import Path
                tp._scan_dataset(Path(dataset_dir))
            epochs = params.get("epochs")
            if epochs is not None:
                tp._epochs.setValue(int(epochs))
            batch = params.get("batch")
            if batch is not None:
                tp._batch.setValue(int(batch))
            # YOLO-specific
            yolo_base = params.get("yolo_base_model")
            if yolo_base:
                existing = [tp._yolo_base.itemText(i) for i in range(tp._yolo_base.count())]
                if yolo_base not in existing:
                    tp._yolo_base.addItem(yolo_base)
                tp._yolo_base.setCurrentText(yolo_base)
            # UNet-specific
            unet_arch = params.get("unet_arch")
            if unet_arch:
                idx = tp._unet_arch.findText(unet_arch)
                if idx >= 0:
                    tp._unet_arch.setCurrentIndex(idx)
            unet_encoder = params.get("unet_encoder")
            if unet_encoder:
                idx = tp._unet_encoder.findText(unet_encoder)
                if idx >= 0:
                    tp._unet_encoder.setCurrentIndex(idx)

        elif action == "set_pixel_size":
            ps_nm = float(params.get("pixel_size_nm", 0))
            if ps_nm > 0 and self._img_idx >= 0:
                img = self._image_cache.get(self._img_idx)
                if img is not None:
                    self._px_overrides[self._img_idx] = ps_nm
                    self._autosave_timer.start()
                    img.meta.pixel_size = ps_nm
                    img.meta.pixel_size_from_header = False
                    self._engine = MeasurementEngine(pixel_size=ps_nm)
                    self._canvas_widget.canvas.set_pixel_size(ps_nm)
                    w = img.shape[1] if img.shape else 512
                    self._ann_panel.set_scalebar_nm(nice_scalebar_nm(ps_nm, w))
                    self._update_px_btn(ps_nm, False)
                    self._statusbar.showMessage(f"Pixel size set to {ps_nm:.4f} nm/px by AI assistant.")

        elif action == "apply_contrast_preset":
            name = params.get("preset_name", "")
            if name:
                all_presets = self._contrast_panel._all_presets()
                if name in all_presets:
                    preset_p = all_presets[name]
                    self._contrast_panel.set_params(preset_p)
                    self._on_contrast_changed(preset_p)
                    self._statusbar.showMessage(f"Contrast preset applied: {name}")
                else:
                    self._statusbar.showMessage(f"Preset not found: {name}")

        elif action == "export_masks":
            stem = ""
            img = self._canvas_widget.canvas.dm4
            if img is not None and img.filepath:
                stem = str(img.filepath.parent / img.filepath.stem)
            self._on_export_masks(stem)

        elif action == "export_display_image":
            self._on_display_export()

        elif action == "push_to_hub":
            repo_id = params.get("repo_id", "")
            token   = params.get("token", "")
            dataset_dir = self._export_panel.dataset_dir
            if not repo_id:
                self._statusbar.showMessage("push_to_hub: repo_id is required.")
            elif not dataset_dir:
                self._statusbar.showMessage("push_to_hub: no export dataset dir configured.")
            else:
                self._on_push_hub(dataset_dir, repo_id, token)

        elif action == "import_star_file":
            self._import_star()

        elif action == "compress_frames":
            method      = params.get("method", "mean")
            dose        = float(params.get("dose_per_frame", 1.0))
            start_frame = int(params.get("start_frame", 1))
            end_frame   = int(params.get("end_frame", 0))
            img = self._canvas_widget.canvas.dm4
            print(f"[compress_frames] method={method} is_movie={img.is_movie if img else None} n_frames={img.n_frames if img and img.is_movie else 0}", flush=True)
            self._compress_frames(method, dose, start_frame, end_frame)

        elif action == "dose_comparison":
            img = self._canvas_widget.canvas.dm4
            print(f"[dose_comparison] is_movie={img.is_movie if img else None} n_frames={img.n_frames if img and img.is_movie else 0}", flush=True)
            if img is None or not img.is_movie:
                self._statusbar.showMessage("No movie loaded for dose comparison — open a multi-frame DM4/TIFF/MRC file first.", 6000)
                return
            n_bins = int(params.get("n_bins", 4))
            dose   = float(params.get("dose_per_frame", self._movie_dose_spin.value()))
            s = max(0, self._movie_start_spin.value() - 1)
            e = min(img.n_frames, self._movie_end_spin.value())
            frames_slice = img._frames[s:e]
            print(f"[dose_comparison] s={s} e={e} frames_slice.shape={frames_slice.shape}", flush=True)
            if len(frames_slice) >= 2:
                dlg = DoseSeriesDialog(
                    frames_slice,
                    pixel_size_nm=img.pixel_size,
                    dose_per_frame=dose,
                    start_frame=s + 1,
                    parent=self,
                )
                dlg._n_bins_spin.setValue(min(n_bins, len(frames_slice)))
                dlg._update_figure()
                dlg.exec()
            else:
                self._statusbar.showMessage(f"Not enough frames for dose comparison ({len(frames_slice)} selected, need at least 2).", 6000)

        elif action == "batch_run_sam":
            self._start_batch_sam(params)

        elif action == "detect_atoms":
            self._on_detect_atoms(params)

        elif action == "atom_statistics":
            self._on_atom_statistics(params)

        elif action == "analyze_stem_atoms":
            if params.get("pixel_size_nm"):
                self._on_action_requested("set_pixel_size",
                                          {"pixel_size_nm": params["pixel_size_nm"]})
            self._on_detect_atoms(params)
            self._on_atom_statistics(params)

        elif action == "train_atom_model":
            self._on_train_atom_model(params)

        elif action == "batch_run_yolo":
            self._start_batch_model(params, "yolo")

        elif action == "batch_run_unet":
            self._start_batch_model(params, "unet")

        elif action == "add_scalebar":
            color = params.get("color", "#FFFFFF")
            self._canvas_widget.canvas.add_default_scalebar(color=color)
            self._autosave_timer.start()
            self._statusbar.showMessage("Scale bar added.")

        elif action == "rename_label":
            old_lbl = params.get("old_label", "")
            new_lbl = params.get("new_label", "")
            if old_lbl and new_lbl:
                store = self._canvas_widget.canvas.store
                modified = 0
                for ann in list(store):
                    if getattr(ann, "label", None) == old_lbl:
                        ann.label = new_lbl
                        modified += 1
                if modified:
                    # Fire store change so annotations re-render with updated labels
                    store.replace_all(list(store))
                    self._autosave_timer.start()
                    self._statusbar.showMessage(f"Renamed {modified} annotation(s): '{old_lbl}' → '{new_lbl}'.")
                else:
                    self._statusbar.showMessage(f"No annotations found with label '{old_lbl}'.")

        elif action == "save_contrast_preset":
            preset_name = params.get("name", "").strip()
            if preset_name:
                try:
                    from acorn.gui.contrast_panel import (
                        _load_user_presets, _save_user_presets, _params_to_dict, _BUILTIN_PRESETS,
                    )
                    if preset_name in _BUILTIN_PRESETS:
                        self._statusbar.showMessage(f"Cannot overwrite built-in preset '{preset_name}'.")
                    else:
                        user = _load_user_presets()
                        user[preset_name] = _params_to_dict(self._contrast_panel.params())
                        _save_user_presets(user)
                        self._contrast_panel._refresh_preset_combo(select=preset_name)
                        self._statusbar.showMessage(f"Contrast preset saved: {preset_name}")
                except Exception as exc:
                    self._statusbar.showMessage(f"save_contrast_preset error: {exc}")

        elif action == "export_measurements":
            if not self._image_paths:
                self._statusbar.showMessage("No images loaded — cannot export measurements.")
            else:
                import csv
                from pathlib import Path as _Path
                # Collect annotations from ALL loaded images
                all_ann_states = dict(self._ann_states)
                if self._img_idx >= 0:
                    all_ann_states[self._img_idx] = list(self._canvas_widget.canvas.store)
                rows: list[dict] = []
                for idx, anns in sorted(all_ann_states.items()):
                    if idx >= len(self._image_paths):
                        continue
                    img_path = self._image_paths[idx]
                    px_nm = self._px_overrides.get(idx) or 1.0
                    if idx == self._img_idx:
                        loaded = self._canvas_widget.canvas.dm4
                        if loaded and loaded.pixel_size > 0:
                            px_nm = loaded.pixel_size
                    elif idx in self._image_cache:
                        cached = self._image_cache[idx]
                        if cached.pixel_size > 0:
                            px_nm = cached.pixel_size
                    try:
                        from acorn_analysis.particle_panel import _polygon_metrics
                        _have_metrics = True
                    except ImportError:
                        _have_metrics = False
                    for ann in (anns or []):
                        ann_type  = getattr(ann, "type", "")
                        ann_label = getattr(ann, "label", "") or getattr(ann, "text", "")
                        row: dict = {
                            "image":         img_path.name,
                            "pixel_size_nm": px_nm,
                            "type":          ann_type,
                            "label":         ann_label,
                        }
                        # Full shape metrics for ROI annotations
                        verts = getattr(ann, "vertices", None)
                        if _have_metrics and verts and len(verts) >= 3 and px_nm > 0:
                            row.update(_polygon_metrics(verts, px_nm))
                        else:
                            row["area_nm2"]    = getattr(ann, "area_nm2", "")
                            row["distance_nm"] = getattr(ann, "distance_nm", "")
                            row["distance_px"] = getattr(ann, "distance_px", "")
                            row["calibrated"]  = getattr(ann, "calibrated", "")
                        rows.append(row)
                if rows:
                    # Save to acorn_measurements/ inside the same folder the images live in
                    img_dir   = _Path(self._image_paths[0]).parent
                    meas_root = _meas_dir(img_dir)
                    meas_root.mkdir(parents=True, exist_ok=True)
                    out_path  = meas_root / _MEAS_CSV
                    # Overwrite with full combined dataset each time
                    with open(out_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
                    self._statusbar.showMessage(
                        f"Measurements → acorn_measurements/measurements.csv  ({len(rows)} rows, {len(all_ann_states)} images)"
                    )
                    self._context.action_requested.emit("show_measurements", {"csv_path": str(out_path)})
                else:
                    self._statusbar.showMessage("No annotations found across loaded images.")

        elif action == "export_nexus":
            self._on_export_nexus(params)

    # ── NeXus export ──────────────────────────────────────────────────────────


    # ── application quit ──────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        """Stop background threads, then tear down plugins on quit.

        Destroying a running QThread aborts the process ("QThread: Destroyed
        while thread is still running"), so wait for each in-flight worker to
        finish (bounded) before shutting down.
        """
        for attr in ("_thread", "_image_load_thread", "_frame_proc_thread",
                     "_sam_thread", "_yolo_thread", "_unet_thread",
                     "_train_thread", "_batch_export_thread"):
            t = getattr(self, attr, None)
            if t is None:
                continue
            try:
                if t.isRunning():
                    t.quit()                 # stop event-loop threads (no-op for run-once)
                    if not t.wait(3000):     # give run() up to 3s to return
                        t.terminate()
                        t.wait(1000)
            except RuntimeError:
                pass  # underlying C++ object already deleted

        # Stop the training subprocess + its log-tail timer if active
        proc = getattr(self, "_train_proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
        timer = getattr(self, "_train_tail_timer", None)
        if timer is not None:
            try:
                timer.stop()
            except RuntimeError:
                pass

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
            "<b>ACORN v0.2.0</b><br><br>"
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
        import os as _os
        _icon_path = _os.path.join(_os.path.dirname(__file__), "acorn.png")
        if _os.path.exists(_icon_path):
            from PyQt6.QtGui import QIcon
            app.setWindowIcon(QIcon(_icon_path))
    finally:
        os.dup2(_saved_stderr, _stderr_fd)
        os.close(_saved_stderr)
    _log("Qt application ready")

    from PyQt6.QtWidgets import QStyleFactory
    from PyQt6.QtGui import QPalette, QColor
    app.setStyle(QStyleFactory.create("Fusion"))

    pal = QPalette()
    _c = QColor  # shorthand
    pal.setColor(QPalette.ColorRole.Window,          _c("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.WindowText,      _c("#e0e0e0"))
    pal.setColor(QPalette.ColorRole.Base,            _c("#252525"))
    pal.setColor(QPalette.ColorRole.AlternateBase,   _c("#1a1a1a"))
    pal.setColor(QPalette.ColorRole.ToolTipBase,     _c("#252525"))
    pal.setColor(QPalette.ColorRole.ToolTipText,     _c("#e0e0e0"))
    pal.setColor(QPalette.ColorRole.Text,            _c("#e0e0e0"))
    pal.setColor(QPalette.ColorRole.Button,          _c("#252525"))
    pal.setColor(QPalette.ColorRole.ButtonText,      _c("#e0e0e0"))
    pal.setColor(QPalette.ColorRole.BrightText,      _c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight,       _c("#00703C"))
    pal.setColor(QPalette.ColorRole.HighlightedText, _c("#ffffff"))
    pal.setColor(QPalette.ColorRole.Link,            _c("#4dbb78"))
    pal.setColor(QPalette.ColorRole.Mid,             _c("#363636"))
    pal.setColor(QPalette.ColorRole.Shadow,          _c("#0d0d0d"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,       _c("#888888"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, _c("#888888"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, _c("#888888"))
    app.setPalette(pal)

    _log("building main window")
    window = MainWindow()
    _log("showing window")
    window.show()

    if files:
        from pathlib import Path
        window.open_files([Path(f) for f in files if Path(f).is_file()])

    # First launch only — name the five workspaces rather than leave tools hidden.
    # Deferred so it appears over a drawn window, not a grey rectangle.
    QTimer.singleShot(0, window._maybe_show_welcome)

    _log("entering event loop")
    sys.exit(app.exec())


if __name__ == "__main__":
    launch()
