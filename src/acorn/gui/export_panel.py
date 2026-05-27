"""Export panel widget."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QProgressBar, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class ExportPanel(QWidget):
    """Controls for saving the current view to a file."""

    export_requested          = pyqtSignal(str, str, int)  # path, format, dpi
    raw_export_requested      = pyqtSignal(str)            # path for raw TIFF
    display_export_requested  = pyqtSignal()               # export 8-bit display PNG for external annotation
    mask_export_requested     = pyqtSignal(str)            # stem for mask PNG + JSON
    training_export_requested = pyqtSignal(str)            # dataset directory (add now)
    queue_requested           = pyqtSignal(str)            # dataset directory (queue image)
    batch_export_requested    = pyqtSignal(str)            # dataset directory (export all queued)
    clear_queue_requested     = pyqtSignal()               # clear the export queue
    import_negatives_requested = pyqtSignal()              # open file picker for negative images
    finalize_requested        = pyqtSignal(str, float, float, object)  # dir, val_frac, test_frac, assignments
    quality_check_requested   = pyqtSignal()               # assess current image
    hub_push_requested        = pyqtSignal(str, str, str)  # dataset_dir, repo_id, token

    def __init__(self, parent=None):
        super().__init__(parent)
        _content = QWidget()
        layout = QVBoxLayout(_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── filename + directory ──────────────────────────────────────────────
        path_box = QGroupBox("Destination")
        path_layout = QFormLayout(path_box)

        self._name = QLineEdit("image")
        path_layout.addRow("Filename:", self._name)

        dir_row = QHBoxLayout()
        dir_row.setSpacing(4)
        self._dir = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(80)
        browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(self._dir)
        dir_row.addWidget(browse_btn)
        path_layout.addRow("Directory:", dir_row)

        layout.addWidget(path_box)

        # ── format ────────────────────────────────────────────────────────────
        fmt_box = QGroupBox("Format")
        fmt_layout = QHBoxLayout(fmt_box)
        fmt_layout.addWidget(QLabel("Format:"))
        self._fmt_combo = QComboBox()
        for fmt in ["PNG", "TIFF", "JPEG", "SVG", "EPS", "PDF"]:
            self._fmt_combo.addItem(fmt)
        fmt_layout.addWidget(self._fmt_combo, 1)
        layout.addWidget(fmt_box)

        # ── DPI ───────────────────────────────────────────────────────────────
        dpi_row = QHBoxLayout()
        dpi_row.addWidget(QLabel("DPI:"))
        self._dpi = QSpinBox()
        self._dpi.setRange(72, 1200)
        self._dpi.setValue(300)
        self._dpi.setSingleStep(50)
        dpi_row.addWidget(self._dpi)
        dpi_row.addStretch()
        layout.addLayout(dpi_row)

        # ── standard save buttons ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save Image")
        save_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        save_btn.clicked.connect(self._on_save)
        raw_btn = QPushButton("Save Raw TIFF")
        raw_btn.clicked.connect(self._on_save_raw)
        btn_row.addWidget(save_btn)
        btn_row.addWidget(raw_btn)
        layout.addLayout(btn_row)

        display_btn = QPushButton("Export Display Image (for external annotation)")
        display_btn.setToolTip(
            "Save a contrast-normalized 8-bit PNG next to the source file.\n"
            "Use this to annotate on an iPad or other tool, then import\n"
            "the resulting mask via File > Import Annotations."
        )
        display_btn.clicked.connect(lambda: self.display_export_requested.emit())
        layout.addWidget(display_btn)

        mask_btn = QPushButton("Export ROI Masks")
        mask_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        mask_btn.setToolTip("Save labelled mask PNG + labels.json for segmentation")
        mask_btn.clicked.connect(self._on_save_masks)
        layout.addWidget(mask_btn)

        self._status = QLabel("")
        layout.addWidget(self._status)

        # ── image quality ─────────────────────────────────────────────────────
        quality_btn = QPushButton("Check Image Quality")
        quality_btn.setToolTip(
            "Assess motion blur, thick ice, saturation, and low-frequency artefacts"
        )
        quality_btn.clicked.connect(self._on_check_quality)
        layout.addWidget(quality_btn)
        self._quality_status = QLabel("")
        self._quality_status.setWordWrap(True)
        layout.addWidget(self._quality_status)

        # ── AI training export ────────────────────────────────────────────────
        train_box = QGroupBox("AI Training Export")
        train_layout = QVBoxLayout(train_box)
        train_layout.setSpacing(6)

        info = QLabel(
            "Annotate ROIs first, then add this image to the dataset. "
            "Each image is tiled and optionally augmented. "
            "When done annotating all images, click Finalize."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size: 11px; color: palette(mid);")
        train_layout.addWidget(info)

        # Dataset directory
        ds_row = QHBoxLayout()
        ds_row.setSpacing(4)
        ds_row.addWidget(QLabel("Dataset dir:"))
        self._dataset_dir = QLineEdit()
        self._dataset_dir.setPlaceholderText("Select output folder…")
        ds_browse = QPushButton("Browse…")
        ds_browse.setFixedWidth(80)
        ds_browse.clicked.connect(self._browse_dataset)
        ds_row.addWidget(self._dataset_dir, 1)
        ds_row.addWidget(ds_browse)
        train_layout.addLayout(ds_row)

        # Tiling options
        tile_form = QFormLayout()
        tile_form.setSpacing(4)

        self._tile_size = QComboBox()
        for label, val in [("Full image (no tiling)", 0), ("512 px", 512),
                            ("1024 px (SAM default)", 1024), ("2048 px", 2048)]:
            self._tile_size.addItem(label, userData=val)
        self._tile_size.setCurrentIndex(2)   # default 1024
        tile_form.addRow("Tile size:", self._tile_size)

        self._tile_overlap = QComboBox()
        for label, val in [("0%", 0.0), ("25%", 0.25), ("50%", 0.5)]:
            self._tile_overlap.addItem(label, userData=val)
        self._tile_overlap.setCurrentIndex(1)   # default 25%
        tile_form.addRow("Tile overlap:", self._tile_overlap)

        self._augment_check = QCheckBox("Augmentation (8 orientations: 4 rotations x flip)")
        self._augment_check.setChecked(True)
        tile_form.addRow("", self._augment_check)

        self._neg_prompts = QSpinBox()
        self._neg_prompts.setRange(0, 20)
        self._neg_prompts.setValue(3)
        self._neg_prompts.setToolTip("Random negative SAM prompt points sampled outside all masks per tile")
        tile_form.addRow("Neg prompts/instance:", self._neg_prompts)

        self._skip_empty = QCheckBox("Skip tiles with no annotations")
        self._skip_empty.setChecked(True)
        tile_form.addRow("", self._skip_empty)

        train_layout.addLayout(tile_form)

        # Queue / Add-now row
        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        queue_btn = QPushButton("Queue Image")
        queue_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        queue_btn.setToolTip(
            "Snapshot this image's annotations into the export queue.\n"
            "Unannotated images are queued as pure negative examples.\n"
            "Continue annotating other images, then click 'Export All Queued'."
        )
        queue_btn.clicked.connect(self._on_queue)
        add_now_btn = QPushButton("Add Now")
        add_now_btn.setToolTip(
            "Immediately tile + augment this image and append to the dataset\n"
            "(same as before — does not use the queue)."
        )
        add_now_btn.clicked.connect(self._on_training_export)
        add_row.addWidget(queue_btn, 3)
        add_row.addWidget(add_now_btn, 1)
        train_layout.addLayout(add_row)

        # Import negatives button
        neg_btn = QPushButton("Import Images as Negatives")
        neg_btn.setToolTip(
            "Select image files or a folder to add directly to the queue as\n"
            "unannotated negative examples (no annotation required)."
        )
        neg_btn.clicked.connect(self._on_import_negatives)
        train_layout.addWidget(neg_btn)

        # Split assignment table
        self._queue_table = QTableWidget(0, 3)
        self._queue_table.setHorizontalHeaderLabels(["Image", "Type", "Split"])
        self._queue_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._queue_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self._queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self._queue_table.setColumnWidth(1, 74)
        self._queue_table.setColumnWidth(2, 92)
        self._queue_table.verticalHeader().hide()
        self._queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._queue_table.setMinimumHeight(70)
        self._queue_table.setMaximumHeight(200)
        self._queue_table.setStyleSheet("font-size: 11px;")
        train_layout.addWidget(self._queue_table)

        auto_assign_btn = QPushButton("Auto-assign splits from fractions")
        auto_assign_btn.setToolTip(
            "Randomly assign Train / Val / Test to queued images\n"
            "using the fraction sliders below as a starting point.\n"
            "You can then adjust individual rows by hand."
        )
        auto_assign_btn.clicked.connect(self._on_auto_assign)
        train_layout.addWidget(auto_assign_btn)

        # Export queued + clear row
        batch_row = QHBoxLayout()
        batch_row.setSpacing(4)
        self._batch_btn = QPushButton("Export All Queued (0)")
        self._batch_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        self._batch_btn.setToolTip("Export all queued images to the dataset in one pass.")
        self._batch_btn.setEnabled(False)
        self._batch_btn.clicked.connect(self._on_batch_export)
        clear_btn = QPushButton("Clear Queue")
        clear_btn.setToolTip("Remove all images from the queue without exporting.")
        clear_btn.clicked.connect(self._on_clear_queue)
        batch_row.addWidget(self._batch_btn, 3)
        batch_row.addWidget(clear_btn, 1)
        train_layout.addLayout(batch_row)

        self._train_status = QLabel("")
        self._train_status.setWordWrap(True)
        train_layout.addWidget(self._train_status)

        # Overall image progress (Image N / M)
        self._image_progress = QProgressBar()
        self._image_progress.setRange(0, 1)
        self._image_progress.setValue(0)
        self._image_progress.setTextVisible(True)
        self._image_progress.setFormat("Image %v / %m")
        self._image_progress.setFixedHeight(18)
        self._image_progress.setVisible(False)
        train_layout.addWidget(self._image_progress)

        # Per-image tile progress
        self._train_progress = QProgressBar()
        self._train_progress.setRange(0, 100)
        self._train_progress.setValue(0)
        self._train_progress.setTextVisible(True)
        self._train_progress.setFormat("Tile %v / %m")
        self._train_progress.setFixedHeight(14)
        self._train_progress.setVisible(False)
        train_layout.addWidget(self._train_progress)

        # Finalize section
        fin_form = QFormLayout()
        fin_form.setSpacing(4)

        self._val_frac = QDoubleSpinBox()
        self._val_frac.setRange(0.0, 0.4)
        self._val_frac.setSingleStep(0.05)
        self._val_frac.setValue(0.1)
        self._val_frac.setSuffix("  (10%)")
        self._val_frac.valueChanged.connect(
            lambda v: self._val_frac.setSuffix(f"  ({int(round(v*100))}%)")
        )
        fin_form.addRow("Val split:", self._val_frac)

        self._test_frac = QDoubleSpinBox()
        self._test_frac.setRange(0.0, 0.4)
        self._test_frac.setSingleStep(0.05)
        self._test_frac.setValue(0.1)
        self._test_frac.setSuffix("  (10%)")
        self._test_frac.valueChanged.connect(
            lambda v: self._test_frac.setSuffix(f"  ({int(round(v*100))}%)")
        )
        fin_form.addRow("Test split:", self._test_frac)

        train_layout.addLayout(fin_form)

        fin_btn = QPushButton("Finalize Dataset (Create Splits + Stats)")
        fin_btn.setStyleSheet("background:#c0392b;color:white;font-weight:bold;")
        fin_btn.setToolTip(
            "Session-aware train/val/test split.\n"
            "All tiles from the same source image stay in the same split.\n"
            "Writes splits/ and dataset_stats.json."
        )
        fin_btn.clicked.connect(self._on_finalize)
        train_layout.addWidget(fin_btn)

        self._fin_status = QLabel("")
        self._fin_status.setWordWrap(True)
        train_layout.addWidget(self._fin_status)

        # Hub push section
        hub_form = QFormLayout()
        hub_form.setSpacing(4)
        self._hub_repo = QLineEdit()
        self._hub_repo.setPlaceholderText("myorg/cryoem-particles")
        hub_form.addRow("Repo ID:", self._hub_repo)

        self._hub_token = QLineEdit()
        self._hub_token.setPlaceholderText("hf_... (or set HF_TOKEN env var)")
        self._hub_token.setEchoMode(QLineEdit.EchoMode.Password)
        hub_form.addRow("HF Token:", self._hub_token)
        train_layout.addLayout(hub_form)

        hub_btn = QPushButton("Push Dataset to HuggingFace Hub")
        hub_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        hub_btn.setToolTip(
            "Upload the training dataset to HuggingFace Hub.\n"
            "Requires: pip install datasets huggingface_hub"
        )
        hub_btn.clicked.connect(self._on_push_hub)
        train_layout.addWidget(hub_btn)

        self._hub_status = QLabel("")
        self._hub_status.setWordWrap(True)
        train_layout.addWidget(self._hub_status)

        layout.addWidget(train_box)

        # ── dataset stats ─────────────────────────────────────────────────────
        stats_box = QGroupBox("Dataset Stats")
        stats_layout = QVBoxLayout(stats_box)
        stats_layout.setSpacing(4)

        stats_btn = QPushButton("Refresh Stats")
        stats_btn.setToolTip(
            "Count annotations per class in the current dataset directory."
        )
        stats_btn.clicked.connect(self._on_refresh_stats)
        stats_layout.addWidget(stats_btn)

        self._stats_label = QLabel("")
        self._stats_label.setWordWrap(True)
        self._stats_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        stats_layout.addWidget(self._stats_label)

        layout.addWidget(stats_box)
        layout.addStretch()
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setWidget(_content)
        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(_scroll)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def dataset_dir(self) -> str:
        return self._dataset_dir.text().strip()

    def set_defaults(self, filename: str, directory: str) -> None:
        self._name.setText(filename)
        self._dir.setText(directory)
        if not self._dataset_dir.text():
            self._dataset_dir.setText(str(Path(directory) / "training_data"))

    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    def set_train_status(self, msg: str) -> None:
        self._train_status.setText(msg)

    def set_image_progress(self, current: int, total: int) -> None:
        self._image_progress.setRange(0, total)
        self._image_progress.setValue(current)
        self._image_progress.setFormat(f"Image {current} / {total}")
        self._image_progress.setVisible(True)

    def set_train_progress(self, current: int, total: int) -> None:
        self._train_progress.setRange(0, total)
        self._train_progress.setValue(current)
        self._train_progress.setFormat(f"Tile {current} / {total}")
        self._train_progress.setVisible(True)

    def reset_train_progress(self) -> None:
        self._image_progress.setValue(0)
        self._image_progress.setVisible(False)
        self._train_progress.setValue(0)
        self._train_progress.setVisible(False)

    def set_queue_status(self, n: int, names: list[str]) -> None:
        """Update Export All button state (table is updated via update_queue_table)."""
        self._batch_btn.setText(f"Export All Queued ({n})")
        self._batch_btn.setEnabled(n > 0)

    def update_queue_table(self, items: list[dict]) -> None:
        """Refresh the split assignment table from the current queue.

        Preserves any split assignments the user has already set —
        only new rows default to Train.
        """
        existing = self.queue_split_assignments()
        self._queue_table.setRowCount(len(items))
        for row, item in enumerate(items):
            path = item.get("path", "")
            name = item.get("stem") or (Path(path).stem if path else f"image_{row}")
            n_rois = item.get("n_rois", len(item.get("annotations", [])))
            item_type = "Annotated" if n_rois > 0 else "Negative"

            self._queue_table.setItem(row, 0, QTableWidgetItem(name))
            self._queue_table.setItem(row, 1, QTableWidgetItem(item_type))

            combo = QComboBox()
            for label in ["Train", "Validation", "Test", "Exclude"]:
                combo.addItem(label)
            prev = existing.get(path, "Train")
            combo.setCurrentIndex({"Train": 0, "Validation": 1, "Test": 2, "Exclude": 3}.get(prev, 0))
            combo.setProperty("item_path", path)
            self._queue_table.setCellWidget(row, 2, combo)

        self._batch_btn.setText(f"Export All Queued ({len(items)})")
        self._batch_btn.setEnabled(len(items) > 0)

    def queue_split_assignments(self) -> dict[str, str]:
        """Return {path: split_name} for every row currently in the table."""
        result = {}
        label_map = {0: "Train", 1: "Validation", 2: "Test", 3: "Exclude"}
        for row in range(self._queue_table.rowCount()):
            combo = self._queue_table.cellWidget(row, 2)
            if combo is None:
                continue
            path = combo.property("item_path") or ""
            result[path] = label_map.get(combo.currentIndex(), "Train")
        return result

    def set_fin_status(self, msg: str) -> None:
        self._fin_status.setText(msg)

    def set_hub_status(self, msg: str) -> None:
        self._hub_status.setText(msg)

    def set_quality_status(self, msg: str, ok: bool = True) -> None:
        color = "palette(windowText)" if ok else "#c0392b"
        self._quality_status.setStyleSheet(f"color: {color}; font-size: 11px;")
        self._quality_status.setText(msg)

    def training_config(self) -> dict:
        """Return current training settings as a plain dict."""
        tile_val = self._tile_size.currentData()
        return {
            "tile_size":        tile_val if tile_val > 0 else None,
            "tile_overlap":     self._tile_overlap.currentData(),
            "augment":          self._augment_check.isChecked(),
            "n_neg_prompts":    self._neg_prompts.value(),
            "skip_empty_tiles": self._skip_empty.isChecked(),
            "encode_rle":       True,
        }

    # ── slots ─────────────────────────────────────────────────────────────────

    def _browse(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory", self._dir.text())
        if d:
            self._dir.setText(d)

    def _browse_dataset(self) -> None:
        d = QFileDialog.getExistingDirectory(
            self, "Select dataset directory", self._dataset_dir.text()
        )
        if d:
            self._dataset_dir.setText(d)

    def _on_save(self) -> None:
        fmt = self._active_fmt().lower()
        ext_map = {"jpeg": "jpg", "tiff": "tif"}
        ext = ext_map.get(fmt, fmt)
        path = str(Path(self._dir.text()) / f"{self._name.text()}.{ext}")
        self.export_requested.emit(path, fmt, self._dpi.value())

    def _on_save_raw(self) -> None:
        path = str(Path(self._dir.text()) / f"{self._name.text()}_raw.tif")
        self.raw_export_requested.emit(path)

    def _on_save_masks(self) -> None:
        stem = str(Path(self._dir.text()) / self._name.text())
        self.mask_export_requested.emit(stem)

    def _on_training_export(self) -> None:
        d = self._dataset_dir.text().strip()
        if not d:
            self._train_status.setText("Set a dataset directory first.")
            return
        self.training_export_requested.emit(d)

    def _on_queue(self) -> None:
        d = self._dataset_dir.text().strip()
        if not d:
            self._train_status.setText("Set a dataset directory first.")
            return
        self.queue_requested.emit(d)

    def _on_batch_export(self) -> None:
        d = self._dataset_dir.text().strip()
        if not d:
            self._train_status.setText("Set a dataset directory first.")
            return
        self.batch_export_requested.emit(d)

    def _on_clear_queue(self) -> None:
        self.clear_queue_requested.emit()

    def _on_import_negatives(self) -> None:
        self.import_negatives_requested.emit()

    def _on_auto_assign(self) -> None:
        """Randomly fill split dropdowns based on val/test fraction sliders."""
        import random
        n = self._queue_table.rowCount()
        if n == 0:
            return
        val_frac  = self._val_frac.value()
        test_frac = self._test_frac.value()
        n_test = round(n * test_frac) if test_frac > 0 else 0
        n_val  = round(n * val_frac)  if val_frac  > 0 else 0
        # ensure at least 1 train image
        while n_test + n_val >= n and n > 1:
            if n_test > n_val:
                n_test -= 1
            else:
                n_val -= 1
        indices = list(range(n))
        random.shuffle(indices)
        split_for_row = {}
        for i, row in enumerate(indices):
            if i < n_test:
                split_for_row[row] = 2   # Test
            elif i < n_test + n_val:
                split_for_row[row] = 1   # Validation
            else:
                split_for_row[row] = 0   # Train
        for row, idx in split_for_row.items():
            combo = self._queue_table.cellWidget(row, 2)
            if combo:
                combo.setCurrentIndex(idx)

    def _on_finalize(self) -> None:
        d = self._dataset_dir.text().strip()
        if not d:
            self._fin_status.setText("Set a dataset directory first.")
            return
        assignments = self.queue_split_assignments()
        self.finalize_requested.emit(d, self._val_frac.value(), self._test_frac.value(), assignments)

    def _on_push_hub(self) -> None:
        d     = self._dataset_dir.text().strip()
        repo  = self._hub_repo.text().strip()
        token = self._hub_token.text().strip()
        if not d:
            self._hub_status.setText("Set a dataset directory first.")
            return
        if not repo:
            self._hub_status.setText("Enter a HuggingFace repo ID (e.g. myorg/mydata).")
            return
        self._hub_status.setText("Pushing to Hub…")
        self.hub_push_requested.emit(d, repo, token)

    def _on_refresh_stats(self) -> None:
        d = self._dataset_dir.text().strip()
        if not d:
            self._stats_label.setText("Set a dataset directory first.")
            return
        import json
        from pathlib import Path as _Path
        ann_path = _Path(d) / "annotations.json"
        if not ann_path.exists():
            self._stats_label.setText("No annotations.json found in that directory.")
            return
        try:
            coco = json.loads(ann_path.read_text())
        except Exception as exc:
            self._stats_label.setText(f"Could not read annotations.json: {exc}")
            return

        _skip = {"background", "ignore"}
        cat_map = {c["id"]: c["name"] for c in coco.get("categories", [])}
        counts: dict[str, int] = {}
        for ann in coco.get("annotations", []):
            name = cat_map.get(ann.get("category_id"), "Unknown")
            if name.lower() in _skip:
                continue
            counts[name] = counts.get(name, 0) + 1

        n_images = len(coco.get("images", []))
        if not counts:
            self._stats_label.setText(
                f"Images: {n_images}  |  No foreground annotations yet."
            )
            return

        col_w = max(len(n) for n in counts) + 2
        lines = [f"Images: {n_images}   Annotations: {sum(counts.values())}"]
        lines.append("-" * (col_w + 8))
        for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"{name:<{col_w}} {cnt:>6}")
        self._stats_label.setText("\n".join(lines))

    def _on_check_quality(self) -> None:
        self.quality_check_requested.emit()

    def _active_fmt(self) -> str:
        return self._fmt_combo.currentText() or "PNG"
