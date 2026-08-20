"""CryoBLOB plugin panel for ACORN."""

from __future__ import annotations

from acorn.gui import path_field

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QComboBox,
)


class CryoBlobPanel(QWidget):
    """Control panel for running CryoBLOB inside ACORN."""

    run_requested = pyqtSignal(dict)
    clear_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results_df = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_workflow_group())
        layout.addWidget(self._build_source_group())
        layout.addWidget(self._build_controls_tabs())
        layout.addWidget(self._build_output_group())

        action_row = QHBoxLayout()
        self._run_btn = QPushButton("Run CryoBLOB")
        self._run_btn.clicked.connect(self._on_run_clicked)
        self._clear_btn = QPushButton("Clear CryoBLOB")
        self._clear_btn.clicked.connect(self._on_clear_clicked)
        action_row.addWidget(self._run_btn, 1)
        action_row.addWidget(self._clear_btn)
        layout.addLayout(action_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        layout.addWidget(self._progress)

        self._status = QLabel("Load an image file or choose a folder to begin.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: palette(mid);")
        layout.addWidget(self._status)

        layout.addWidget(self._build_results_group(), 1)

    # ------------------------------------------------------------------
    # Workflow presets
    # ------------------------------------------------------------------

    _WORKFLOWS = {
        "custom": {
            "label": "Custom",
            "description": "Manually configure all detection parameters.",
        },
        "filament": {
            "label": "Filament Tracing",
            "description": (
                "Traces filaments and filament-like structures using multi-scale "
                "ridge detection (Sato + Meijering filters). Each detected segment "
                "is traced to a full polyline skeleton in Final mode. "
                "Tune Min/Max blob size to match your filament width."
            ),
            "params": {
                "run_mode": "final",
                "detection_mode": "ridge",
                "blob_downscale": 2.0,
                "min_sigma": 3.0,
                "max_sigma": 28.0,
                "ridge_threshold": 0.006,
                "ridge_scales": 20,
                "gblur": 2,
                "background": 10,
                "max_detections": 300,
                "refine_sizes": False,
            },
        },
        "vesicle_overlap": {
            "label": "Overlapping Vesicles",
            "description": (
                "Picks vesicles that are touching or stacked on top of each other. "
                "Hessian blob detection seeds a distance-transform watershed that "
                "splits overlapping clusters into individual vesicles. "
                "Lower Min marker distance to separate more tightly packed clusters."
            ),
            "params": {
                "run_mode": "final",
                "detection_mode": "enhanced",
                "blob_downscale": 3.0,
                "min_sigma": 10.0,
                "max_sigma": 100.0,
                "blob_step": 1.0,
                "threshold_rel": 6.0,
                "min_marker_distance": 4.0,
                "use_ridge_detection": False,
                "use_watershed": True,
                "gblur": 2,
                "background": 0,
                "max_detections": 200,
                "refine_sizes": False,
            },
        },
    }

    def _build_workflow_group(self) -> QGroupBox:
        box = QGroupBox("Workflow")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        self._workflow = QComboBox()
        for key, wf in self._WORKFLOWS.items():
            self._workflow.addItem(wf["label"], key)
        self._workflow.currentIndexChanged.connect(self._on_workflow_changed)
        layout.addWidget(self._workflow)

        self._workflow_description = QLabel(self._WORKFLOWS["custom"]["description"])
        self._workflow_description.setWordWrap(True)
        self._workflow_description.setStyleSheet("font-size: 11px; color: palette(mid);")
        layout.addWidget(self._workflow_description)

        return box

    def _on_workflow_changed(self) -> None:
        key = self._workflow.currentData()
        wf = self._WORKFLOWS.get(key, {})
        self._workflow_description.setText(wf.get("description", ""))

        params = wf.get("params")
        if not params:
            return

        if "run_mode" in params:
            idx = self._run_mode.findData(params["run_mode"])
            if idx >= 0:
                self._run_mode.setCurrentIndex(idx)

        if "detection_mode" in params:
            idx = self._detection_mode.findData(params["detection_mode"])
            if idx >= 0:
                self._detection_mode.setCurrentIndex(idx)

        if "blob_downscale" in params:
            self._blob_downscale.setValue(params["blob_downscale"])
        if "min_sigma" in params:
            self._min_sigma.setValue(params["min_sigma"])
        if "max_sigma" in params:
            self._max_sigma.setValue(params["max_sigma"])
        if "blob_step" in params:
            self._blob_step.setValue(params["blob_step"])
        if "threshold_rel" in params:
            self._threshold_rel.setValue(params["threshold_rel"])
        if "max_detections" in params:
            self._max_detections.setValue(params["max_detections"])
        if "refine_sizes" in params:
            self._refine_sizes.setChecked(params["refine_sizes"])
        if "ridge_threshold" in params:
            self._ridge_threshold.setValue(params["ridge_threshold"])
        if "ridge_scales" in params:
            self._ridge_scales.setValue(params["ridge_scales"])
        if "min_marker_distance" in params:
            self._min_marker_distance.setValue(params["min_marker_distance"])
        if "use_ridge_detection" in params:
            self._use_ridge_detection.setChecked(params["use_ridge_detection"])
        if "use_watershed" in params:
            self._use_watershed.setChecked(params["use_watershed"])
        if "gblur" in params:
            self._gblur.setValue(params["gblur"])
        if "background" in params:
            self._background.setValue(params["background"])

        self._update_mode_visibility()

    def _build_source_group(self) -> QGroupBox:
        box = QGroupBox("Source")
        form = QFormLayout(box)

        self._mode = QComboBox()
        self._mode.addItem("Current image", "current")
        self._mode.addItem("Loaded images", "loaded")
        self._mode.addItem("Folder", "folder")
        self._mode.currentIndexChanged.connect(self._on_mode_changed)
        form.addRow("Mode:", self._mode)

        self._current_image = QLabel("No image loaded.")
        self._current_image.setWordWrap(True)
        form.addRow("Current:", self._current_image)

        self._folder_widget = QWidget()
        row = QHBoxLayout(self._folder_widget)
        row.setContentsMargins(0, 0, 0, 0)
        self._folder_edit = QLineEdit()
        path_field.attach(self._folder_edit)
        self._folder_edit.setPlaceholderText("Folder containing MRC, TIFF, PNG, or JPG files")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_folder)
        row.addWidget(self._folder_edit, 1)
        row.addWidget(browse)
        form.addRow("Folder:", self._folder_widget)
        self._folder_widget.setVisible(False)
        return box

    def _build_controls_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        basic = QWidget()
        basic_layout = QVBoxLayout(basic)
        basic_layout.setContentsMargins(6, 6, 6, 6)
        basic_layout.setSpacing(8)
        basic_layout.addWidget(self._build_basic_group())
        basic_layout.addWidget(self._build_measurement_group())
        basic_layout.addStretch(1)

        advanced = QWidget()
        advanced_layout = QVBoxLayout(advanced)
        advanced_layout.setContentsMargins(6, 6, 6, 6)
        advanced_layout.setSpacing(8)
        advanced_layout.addWidget(self._build_advanced_mode_group())
        advanced_layout.addWidget(self._build_preprocess_group())
        advanced_layout.addStretch(1)

        tabs.addTab(basic, "Simple")
        tabs.addTab(advanced, "Advanced")
        return tabs

    def _build_basic_group(self) -> QGroupBox:
        box = QGroupBox("Detection")
        form = QFormLayout(box)

        self._run_mode = QComboBox()
        self._run_mode.addItem("Preview (fast)", "preview")
        self._run_mode.addItem("Final (accurate)", "final")
        self._run_mode.setToolTip(
            "Preview uses lighter refinement for tuning settings quickly. "
            "Final uses the more accurate full refinement path."
        )
        form.addRow("Run mode:", self._run_mode)

        self._detection_mode = QComboBox()
        self._detection_mode.addItem("LoG blobs (ACORN GPU)", "log")
        self._detection_mode.addItem("LoG + Watershed (splits overlaps)", "log_watershed")
        self._detection_mode.addItem("Hessian blobs (CryoBLOB)", "hessian")
        self._detection_mode.addItem("Ridges only (CryoBLOB)", "ridge")
        self._detection_mode.addItem("Enhanced: Hessian + Ridge + Watershed", "enhanced")
        self._detection_mode.currentIndexChanged.connect(self._update_mode_visibility)
        form.addRow("Detection mode:", self._detection_mode)

        self._blob_downscale = QDoubleSpinBox()
        self._blob_downscale.setRange(1.0, 64.0)
        self._blob_downscale.setDecimals(1)
        self._blob_downscale.setValue(3.0)
        form.addRow("Blob downscale:", self._blob_downscale)

        self._min_sigma = QDoubleSpinBox()
        self._min_sigma.setRange(0.5, 200.0)
        self._min_sigma.setDecimals(1)
        self._min_sigma.setValue(18.0)
        self._min_sigma.setToolTip("Minimum blob radius scale after downscaling. Raise this to ignore tiny specks.")
        form.addRow("Min blob size:", self._min_sigma)

        self._max_sigma = QDoubleSpinBox()
        self._max_sigma.setRange(0.5, 500.0)
        self._max_sigma.setDecimals(1)
        self._max_sigma.setValue(90.0)
        self._max_sigma.setToolTip("Maximum blob radius scale after downscaling. Raise this if large particles are missed.")
        form.addRow("Max blob size:", self._max_sigma)

        self._threshold_rel = QDoubleSpinBox()
        self._threshold_rel.setRange(0.001, 20.0)
        self._threshold_rel.setDecimals(3)
        self._threshold_rel.setSingleStep(0.025)
        self._threshold_rel.setValue(0.22)
        self._threshold_rel.setToolTip("LoG threshold uses relative response. Hessian/enhanced use this as a standard-deviation multiplier.")
        form.addRow("Detection threshold:", self._threshold_rel)

        self._max_detections = QSpinBox()
        self._max_detections.setRange(1, 10000)
        self._max_detections.setValue(80)
        self._max_detections.setToolTip("Safety cap for annotations and CSV rows per image.")
        form.addRow("Max detections:", self._max_detections)
        return box

    def _build_measurement_group(self) -> QGroupBox:
        box = QGroupBox("Measurement")
        form = QFormLayout(box)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.0001, 100000.0)
        self._pixel_size.setDecimals(4)
        self._pixel_size.setValue(1.0)
        self._pixel_size.setSuffix(" nm/px")
        self._pixel_size.setToolTip(
            "Used for PNG/JPG/TIFF output units. MRC files use voxel size from the file header."
        )
        form.addRow("Pixel size:", self._pixel_size)

        self._gblur = QSpinBox()
        self._gblur.setRange(0, 50)
        self._gblur.setValue(2)
        form.addRow("Gaussian blur:", self._gblur)

        self._background = QSpinBox()
        self._background.setRange(0, 100)
        self._background.setValue(0)
        form.addRow("Background:", self._background)

        self._apply_filter = QSpinBox()
        self._apply_filter.setRange(0, 20)
        self._apply_filter.setValue(0)
        form.addRow("Wiener filter:", self._apply_filter)

        self._size_scale = QDoubleSpinBox()
        self._size_scale.setRange(0.25, 3.0)
        self._size_scale.setDecimals(2)
        self._size_scale.setSingleStep(0.05)
        self._size_scale.setValue(1.0)
        self._size_scale.setToolTip("Calibration multiplier applied to measured particle radius.")
        form.addRow("Size scale:", self._size_scale)

        self._refine_sizes = QCheckBox("Refine particle sizes from image edge")
        self._refine_sizes.setChecked(True)
        form.addRow("", self._refine_sizes)

        self._add_annotations = QCheckBox("Add detections to current image")
        self._add_annotations.setChecked(True)
        form.addRow("", self._add_annotations)
        return box

    def _build_advanced_mode_group(self) -> QGroupBox:
        box = QGroupBox("Mode Details")
        form = QFormLayout(box)

        self._blob_step = QDoubleSpinBox()
        self._blob_step.setRange(0.1, 50.0)
        self._blob_step.setDecimals(2)
        self._blob_step.setSingleStep(0.25)
        self._blob_step.setValue(1.0)
        self._blob_step.setToolTip("Scale step used by Hessian and enhanced CryoBLOB detectors.")
        self._blob_step_label = QLabel("Blob step:")
        form.addRow(self._blob_step_label, self._blob_step)

        self._ridge_threshold = QDoubleSpinBox()
        self._ridge_threshold.setRange(0.0001, 10.0)
        self._ridge_threshold.setDecimals(4)
        self._ridge_threshold.setSingleStep(0.0025)
        self._ridge_threshold.setValue(0.01)
        self._ridge_threshold.setToolTip("Sensitivity for CryoBLOB ridge detection.")
        self._ridge_threshold_label = QLabel("Ridge threshold:")
        form.addRow(self._ridge_threshold_label, self._ridge_threshold)

        self._ridge_scales = QSpinBox()
        self._ridge_scales.setRange(2, 100)
        self._ridge_scales.setValue(15)
        self._ridge_scales.setToolTip("Number of scales tested by the multi-scale ridge detector.")
        self._ridge_scales_label = QLabel("Ridge scales:")
        form.addRow(self._ridge_scales_label, self._ridge_scales)

        self._min_marker_distance = QDoubleSpinBox()
        self._min_marker_distance.setRange(0.5, 500.0)
        self._min_marker_distance.setDecimals(1)
        self._min_marker_distance.setSingleStep(0.5)
        self._min_marker_distance.setValue(5.0)
        self._min_marker_distance.setToolTip("Minimum distance between watershed markers for overlapping blobs.")
        self._min_marker_distance_label = QLabel("Min marker distance:")
        form.addRow(self._min_marker_distance_label, self._min_marker_distance)

        self._use_ridge_detection = QCheckBox("Use ridge detection in enhanced mode")
        self._use_ridge_detection.setChecked(True)
        form.addRow("", self._use_ridge_detection)

        self._use_watershed = QCheckBox("Use watershed in enhanced mode")
        self._use_watershed.setChecked(True)
        form.addRow("", self._use_watershed)
        return box

    def _build_preprocess_group(self) -> QGroupBox:
        box = QGroupBox("Image Prep")
        form = QFormLayout(box)

        self._exponential = QCheckBox("Use exponential preprocessing")
        self._exponential.setChecked(False)
        form.addRow("", self._exponential)

        self._logarizer = QCheckBox("Use logarithmic preprocessing")
        form.addRow("", self._logarizer)

        self._stream = QCheckBox("Stream large files")
        self._stream.setChecked(True)
        form.addRow("", self._stream)

        self._contrast_polarity = QComboBox()
        self._contrast_polarity.addItem("Auto-detect", "auto")
        self._contrast_polarity.addItem("Particles are darker (cryo-EM)", "dark")
        self._contrast_polarity.addItem("Particles are lighter (inverted)", "light")
        self._contrast_polarity.setToolTip(
            "CryoBLOB looks for density minima, so it finds particles that are DARKER "
            "than the surrounding field. On inverted data - a rendered mask, an ADF-STEM "
            "frame, an already-flipped TIFF - it would otherwise return nothing at all. "
            "Auto-detect measures the image and flips it when needed."
        )
        form.addRow("Particle contrast:", self._contrast_polarity)

        self._cache_results = QCheckBox("Reuse preprocessing between reruns")
        self._cache_results.setChecked(True)
        self._cache_results.setToolTip(
            "Keeps normalized images and ridge-response maps in memory so tuning "
            "thresholds and sizes is much faster on the same file."
        )
        form.addRow("", self._cache_results)

        self._update_mode_visibility()
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        row = QHBoxLayout(box)
        self._output_edit = QLineEdit()
        path_field.attach(self._output_edit)
        self._output_edit.setPlaceholderText("Default: next to the image or inside the chosen folder")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row.addWidget(self._output_edit, 1)
        row.addWidget(browse)
        return box

    def _build_results_group(self) -> QGroupBox:
        box = QGroupBox("Results")
        layout = QVBoxLayout(box)

        self._summary = QLabel("No CryoBLOB results yet.")
        self._summary.setWordWrap(True)
        layout.addWidget(self._summary)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(
            ["File", "Type", "Center Y (nm)", "Center X (nm)", "Diameter (nm)"]
        )
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self._table, 1)
        return box

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose image folder")
        if folder:
            self._folder_edit.setText(folder)

    def _browse_output(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "CryoBLOB output CSV",
            "cryoblob_results.csv",
            "CSV files (*.csv)",
        )
        if path:
            self._output_edit.setText(path)

    def _on_mode_changed(self) -> None:
        self._folder_widget.setVisible(self._mode.currentData() == "folder")

    def _update_mode_visibility(self) -> None:
        mode = self._detection_mode.currentData()
        # log_watershed shares LoG's controls (min/max sigma, relative threshold,
        # downscale, refine); it just adds a distance-transform watershed pass.
        is_log = mode in {"log", "log_watershed"}
        is_ridge = mode == "ridge"
        is_enhanced = mode == "enhanced"
        is_hessian_family = mode in {"hessian", "enhanced"}

        self._blob_step.setEnabled(not is_log and not is_ridge)
        self._threshold_rel.setEnabled(not is_ridge)
        self._ridge_threshold.setEnabled(is_ridge or is_enhanced)
        self._ridge_scales.setEnabled(is_ridge)
        self._min_marker_distance.setEnabled(is_enhanced)
        self._use_ridge_detection.setEnabled(is_enhanced)
        self._use_watershed.setEnabled(is_enhanced)
        self._refine_sizes.setEnabled(not is_ridge)
        self._blob_step_label.setVisible(not is_log and not is_ridge)
        self._blob_step.setVisible(not is_log and not is_ridge)
        self._ridge_threshold_label.setVisible(is_ridge or is_enhanced)
        self._ridge_threshold.setVisible(is_ridge or is_enhanced)
        self._ridge_scales_label.setVisible(is_ridge)
        self._ridge_scales.setVisible(is_ridge)
        self._min_marker_distance_label.setVisible(is_enhanced)
        self._min_marker_distance.setVisible(is_enhanced)
        self._use_ridge_detection.setVisible(is_enhanced)
        self._use_watershed.setVisible(is_enhanced)
        if is_hessian_family and self._threshold_rel.value() <= 1.0:
            self._threshold_rel.setValue(6.0)
        elif is_log and self._threshold_rel.value() > 1.0:
            self._threshold_rel.setValue(0.22)

    def current_params(self) -> dict:
        """The full detection-parameter dict reflecting the panel's current state.

        Exposed so callers other than the Run button (e.g. the CLU assistant via
        the plugin) can launch a run with the user's on-screen settings plus any
        overrides they supply."""
        return {
            "mode": self._mode.currentData(),
            "run_mode": self._run_mode.currentData(),
            "detection_mode": self._detection_mode.currentData(),
            "folder": self._folder_edit.text().strip(),
            "output_csv": self._output_edit.text().strip(),
            "pixel_size_nm": self._pixel_size.value(),
            "blob_downscale": self._blob_downscale.value(),
            "min_sigma": self._min_sigma.value(),
            "max_sigma": self._max_sigma.value(),
            "blob_step": self._blob_step.value(),
            "threshold_rel": self._threshold_rel.value(),
            "max_detections": self._max_detections.value(),
            "refine_sizes": self._refine_sizes.isChecked(),
            "size_scale": self._size_scale.value(),
            "ridge_threshold": self._ridge_threshold.value(),
            "ridge_scales": self._ridge_scales.value(),
            "min_marker_distance": self._min_marker_distance.value(),
            "use_ridge_detection": self._use_ridge_detection.isChecked(),
            "use_watershed": self._use_watershed.isChecked(),
            "gblur": self._gblur.value(),
            "background": self._background.value(),
            "apply_filter": self._apply_filter.value(),
            "exponential": self._exponential.isChecked(),
            "logarizer": self._logarizer.isChecked(),
            "stream_large_files": self._stream.isChecked(),
            "cache_results": self._cache_results.isChecked(),
            "contrast_polarity": self._contrast_polarity.currentData(),
            "add_annotations": self._add_annotations.isChecked(),
        }

    def _on_run_clicked(self) -> None:
        if self._mode.currentData() == "folder" and not self._folder_edit.text().strip():
            QMessageBox.information(self, "CryoBLOB", "Choose a folder before running.")
            return
        self.run_requested.emit(self.current_params())

    def _on_clear_clicked(self) -> None:
        self.clear_requested.emit()

    def set_current_image(self, image_path: str | None, is_supported: bool) -> None:
        if not image_path:
            self._current_image.setText("No image loaded.")
        elif is_supported:
            self._current_image.setText(str(Path(image_path)))
        else:
            self._current_image.setText(f"{image_path} (unsupported format)")

    def set_pixel_size(self, pixel_size_nm: float) -> None:
        if pixel_size_nm > 0:
            self._pixel_size.setValue(float(pixel_size_nm))

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)
        if not running:
            self._progress.setValue(0)

    def show_progress(self, pct: int, message: str) -> None:
        self._progress.setValue(pct)
        self._status.setText(message)

    def clear_results(self) -> None:
        self._results_df = None
        self._table.setRowCount(0)
        self._summary.setText("No CryoBLOB results yet.")
        self._status.setText("Cleared CryoBLOB detections from the current image.")

    def show_results(self, df, output_csv: str, failures: list[tuple[str, str]]) -> None:
        self._results_df = df
        self._table.setRowCount(len(df))
        for row_idx, (_, row) in enumerate(df.iterrows()):
            file_name = Path(str(row["File Location"])).name
            values = [
                file_name,
                str(row.get("Detection Type", "")),
                f"{row['Center Y (nm)']:.3f}",
                f"{row['Center X (nm)']:.3f}",
                f"{row['Size (nm)']:.3f}",
            ]
            for col_idx, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col_idx > 1:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row_idx, col_idx, item)

        summary = f"{len(df)} detection(s) saved to {output_csv}."
        if failures:
            first_path, first_error = failures[0]
            summary += (
                f" {len(failures)} file(s) failed. "
                f"First failure: {Path(first_path).name}: {first_error}"
            )
        self._summary.setText(summary)
        self._status.setText(summary)
