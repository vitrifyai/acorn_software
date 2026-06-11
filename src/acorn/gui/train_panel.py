"""
Training panel — Train tab in the ACORN GUI.

Lets the user pick a dataset directory, choose YOLO or UNet, configure
hyperparameters, select CPU/GPU(s), and launch training.  Progress is
shown via a progress bar and a scrolling log.

Signals
-------
train_requested(dict)   — emitted when the user clicks Train; dict contains
                          all settings needed by YOLOTrainer / UNetTrainer.
cancel_requested()      — emitted when the user clicks Cancel.
load_yolo_requested(str)  — emitted after training to auto-load best.pt into
                            the YOLO tab.
load_unet_requested(str)  — emitted after training to auto-load into UNet tab.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QSplitter,
    QCheckBox, QComboBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QProgressBar, QPushButton, QRadioButton, QScrollArea, QSizePolicy,
    QSpinBox, QStackedWidget, QVBoxLayout, QWidget,
)

_YOLO_MODELS_DIR = (
    Path(os.environ["ACORN_MODELS_DIR"]) / "yolo"
    if "ACORN_MODELS_DIR" in os.environ
    else Path.home() / ".acorn" / "models" / "yolo"
)

# Pull URLs from download_models.py — single source of truth for YOLO assets.
try:
    import sys as _sys, importlib.util as _ilu
    _dm_path = str(Path(__file__).resolve().parents[3] / "download_models.py")
    _spec = _ilu.spec_from_file_location("download_models", _dm_path)
    _dm   = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_dm)
    _YOLO_DOWNLOAD_URLS = {k: v[0] for k, v in _dm.YOLO_MODELS.items()
                           if k.endswith("-seg.pt")}
    del _dm, _spec, _dm_path, _ilu, _sys
except Exception:
    # Fallback: hardcode only as last resort if download_models.py is unavailable
    _GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
    _YOLO_DOWNLOAD_URLS = {
        f"yolo26{s}-seg.pt": f"{_GH}/yolo26{s}-seg.pt" for s in "nsmzlx"
    } | {f"yolo11{s}-seg.pt": f"{_GH}/yolo11{s}-seg.pt" for s in "nsmlx"}


def _resolve_yolo_model(tag: str) -> str:
    """Return full path to a YOLO model if it lives in the local cache, else tag."""
    local = _YOLO_MODELS_DIR / tag
    return str(local) if local.exists() else tag

_MPL_OK: bool | None = None  # None = not yet checked; True/False = result


def _detect_gpus() -> list[tuple[int, str]]:
    """Return list of (index, name) for available CUDA GPUs."""
    try:
        import torch
        n = torch.cuda.device_count()
        return [(i, torch.cuda.get_device_name(i)) for i in range(n)]
    except Exception:
        return []


class TrainPanel(QWidget):
    train_requested    = pyqtSignal(dict)
    cancel_requested   = pyqtSignal()
    load_yolo_requested = pyqtSignal(str)
    load_unet_requested = pyqtSignal(str)
    _gpus_detected     = pyqtSignal(list)   # emitted from bg thread, received on main thread

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._gpus: list[tuple[int, str]] = []
        self._gpu_checks: list[QCheckBox] = []
        self._training = False
        self._build_ui()
        self._gpus_detected.connect(self._apply_gpus)
        # Detect GPUs in background to avoid blocking startup (torch import + 8x GPU query ~2s)
        import threading
        threading.Thread(target=self._detect_gpus_bg, daemon=True).start()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        outer.addWidget(splitter)

        # ── top: scrollable config ─────────────────────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(scroll.Shape.NoFrame)
        splitter.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addWidget(self._build_dataset_group())
        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_hardware_group())
        layout.addWidget(self._build_common_group())
        layout.addStretch()

        # ── bottom: progress + log (drag splitter handle to resize) ───────────
        bottom_widget = QWidget()
        bottom = QVBoxLayout(bottom_widget)
        bottom.setContentsMargins(6, 4, 6, 6)
        bottom.setSpacing(4)
        splitter.addWidget(bottom_widget)
        splitter.setStretchFactor(0, 2)   # config gets 2/3
        splitter.setStretchFactor(1, 1)   # log gets 1/3

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train")
        self._train_btn.setStyleSheet(
            "background:#00703C;color:white;font-weight:bold;"
        )
        self._train_btn.clicked.connect(self._on_train_clicked)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_requested)

        btn_row.addWidget(self._train_btn)
        btn_row.addWidget(self._cancel_btn)
        bottom.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("Idle")
        bottom.addWidget(self._progress)

        # ── live loss curve ───────────────────────────────────────────────────
        self._loss_epochs: list[int] = []
        self._loss_train: list[float] = []
        self._loss_metric: list[float] = []  # seg_mAP50 (YOLO) or mF1 (UNet)
        self._loss_canvas = None
        self._loss_chart_layout = bottom  # stored so canvas can be added lazily

        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumBlockCount(2000)
        self._log.setMinimumHeight(60)
        self._log.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._log.setStyleSheet("font-family: monospace; font-size: 11px;")
        bottom.addWidget(self._log)

    def _build_dataset_group(self) -> QGroupBox:
        box = QGroupBox("Dataset")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        dir_row = QHBoxLayout()
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Path to exported dataset directory…")
        self._dir_edit.editingFinished.connect(self._on_dir_typed)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(75)
        browse_btn.clicked.connect(self._browse_dataset)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedWidth(75)
        refresh_btn.setToolTip("Re-scan dataset for updated annotation counts")
        refresh_btn.clicked.connect(self._on_refresh_dataset)
        dir_row.addWidget(self._dir_edit, 1)
        dir_row.addWidget(browse_btn)
        dir_row.addWidget(refresh_btn)
        layout.addLayout(dir_row)

        self._dataset_info = QPlainTextEdit()
        self._dataset_info.setReadOnly(True)
        self._dataset_info.setFixedHeight(120)
        self._dataset_info.setStyleSheet(
            "font-family: monospace; font-size: 11px; "
            "background: palette(base); border: 1px solid palette(mid);"
        )
        self._dataset_info.setPlainText("No dataset selected.")
        layout.addWidget(self._dataset_info)

        return box

    def _build_model_group(self) -> QGroupBox:
        box = QGroupBox("Model")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        radio_row = QHBoxLayout()
        self._yolo_radio = QRadioButton("YOLO (instance segmentation)")
        self._unet_radio = QRadioButton("UNet (semantic segmentation)")
        self._yolo_radio.setChecked(True)
        self._yolo_radio.toggled.connect(self._on_model_toggled)
        radio_row.addWidget(self._yolo_radio)
        radio_row.addWidget(self._unet_radio)
        layout.addLayout(radio_row)

        self._stack = QStackedWidget()
        self._stack.addWidget(self._build_yolo_config())   # index 0
        self._stack.addWidget(self._build_unet_config())   # index 1
        layout.addWidget(self._stack)

        return box

    def _build_yolo_config(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        form = QFormLayout()
        form.setSpacing(4)

        model_row = QHBoxLayout()
        self._yolo_base = QComboBox()
        for tag in [
            "yolo26n-seg.pt",
            "yolo26s-seg.pt",
            "yolo26m-seg.pt",
            "yolo26l-seg.pt",
            "yolo26x-seg.pt",
            "yolo11n-seg.pt",
            "yolo11s-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt",
            "yolo11x-seg.pt",
        ]:
            self._yolo_base.addItem(tag)
        self._yolo_base.setToolTip(
            "n=nano (fastest), s=small, m=medium, l=large, x=xlarge (most accurate)"
        )
        self._yolo_base.currentTextChanged.connect(self._on_yolo_model_changed)
        model_row.addWidget(self._yolo_base, 1)

        browse_model_btn = QPushButton("Browse…")
        browse_model_btn.setMinimumWidth(90)
        browse_model_btn.setToolTip("Use a locally downloaded .pt file")
        browse_model_btn.clicked.connect(self._browse_yolo_model)
        model_row.addWidget(browse_model_btn)
        form.addRow("Base model:", model_row)

        self._yolo_imgsz = QSpinBox()
        self._yolo_imgsz.setRange(320, 1280)
        self._yolo_imgsz.setSingleStep(32)
        self._yolo_imgsz.setValue(640)
        self._yolo_imgsz.setToolTip("Input image size (square). 640 is standard.")
        form.addRow("Image size:", self._yolo_imgsz)

        layout.addLayout(form)

        self._yolo_download_hint = QLabel("")
        self._yolo_download_hint.setWordWrap(True)
        self._yolo_download_hint.setStyleSheet("font-size: 10px; color: #4d8ec4;")
        self._yolo_download_hint.setVisible(False)
        layout.addWidget(self._yolo_download_hint)

        # Trigger hint check for the default selection
        self._yolo_base.currentTextChanged.emit(self._yolo_base.currentText())

        return w

    def _build_unet_config(self) -> QWidget:
        w = QWidget()
        form = QFormLayout(w)
        form.setSpacing(4)

        self._unet_arch = QComboBox()
        for arch in ["Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus", "MAnet", "PAN"]:
            self._unet_arch.addItem(arch)
        self._unet_arch.setToolTip(
            "Unet / UnetPlusPlus: best for small objects.\n"
            "FPN / DeepLabV3Plus: good for multi-scale structure."
        )
        form.addRow("Architecture:", self._unet_arch)

        self._unet_encoder = QComboBox()
        for enc in [
            "resnet34", "resnet50",
            "efficientnet-b0", "efficientnet-b3",
            "mobilenet_v2",
            "mit_b0", "mit_b2",
        ]:
            self._unet_encoder.addItem(enc)
        self._unet_encoder.setToolTip(
            "resnet34: fast, good baseline.\n"
            "efficientnet-b3: higher accuracy, more memory.\n"
            "mit_b2: transformer encoder, best for textures."
        )
        form.addRow("Encoder:", self._unet_encoder)

        # Learning rate is logarithmic — an editable combo of common values is
        # far more usable than a linear spinbox (which needed ~99 clicks to go
        # 1e-4 → 1e-2). Users can still type an exact value.
        self._unet_lr = QComboBox()
        self._unet_lr.setEditable(True)
        self._unet_lr.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        for lr in ["1e-5", "5e-5", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2"]:
            self._unet_lr.addItem(lr)
        self._unet_lr.setCurrentText("1e-4")
        self._unet_lr.setToolTip("Pick a common learning rate or type your own (e.g. 3e-4).")
        form.addRow("Learning rate:", self._unet_lr)

        self._unet_imgsz = QSpinBox()
        self._unet_imgsz.setRange(128, 1024)
        self._unet_imgsz.setSingleStep(32)
        self._unet_imgsz.setValue(512)
        self._unet_imgsz.setToolTip(
            "Resize tiles to this size before training. "
            "Smaller = faster but less detail."
        )
        form.addRow("Image size:", self._unet_imgsz)

        return w

    def _build_hardware_group(self) -> QGroupBox:
        box = QGroupBox("Hardware")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)
        self._hw_layout = layout

        self._cpu_check = QCheckBox("CPU")
        self._cpu_check.setChecked(True)
        self._cpu_check.setEnabled(False)
        layout.addWidget(self._cpu_check)

        self._hw_status_label = QLabel("Detecting GPUs...")
        self._hw_status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        layout.addWidget(self._hw_status_label)

        return box

    def _detect_gpus_bg(self) -> None:
        """Run GPU detection off the main thread, then update UI via signal."""
        gpus = _detect_gpus()
        self._gpus_detected.emit(gpus)

    def _apply_gpus(self, gpus: list[tuple[int, str]]) -> None:
        """Called on main thread once GPU detection is complete."""
        self._gpus = gpus
        # Remove the "Detecting GPUs..." label
        self._hw_status_label.deleteLater()
        if gpus:
            gpu_label = QLabel("GPUs:")
            self._hw_layout.addWidget(gpu_label)
            for idx, name in gpus:
                cb = QCheckBox(f"GPU {idx}: {name}")
                cb.setChecked(idx == 0)
                cb.toggled.connect(self._on_gpu_toggled)
                self._gpu_checks.append(cb)
                self._hw_layout.addWidget(cb)
            self._cpu_check.setEnabled(True)
            self._cpu_check.toggled.connect(self._on_cpu_toggled)
            # GPU 0 is auto-selected, so CPU must not also be checked
            self._cpu_check.blockSignals(True)
            self._cpu_check.setChecked(False)
            self._cpu_check.blockSignals(False)
        else:
            no_gpu = QLabel("No CUDA GPUs detected — CPU only.")
            no_gpu.setStyleSheet("font-size: 11px; color: palette(mid);")
            self._hw_layout.addWidget(no_gpu)

    @staticmethod
    def _parse_lr(text: str) -> float:
        """Parse the learning-rate combo text, falling back to 1e-4 if invalid."""
        try:
            lr = float(text)
            return lr if lr > 0 else 1e-4
        except (ValueError, TypeError):
            return 1e-4

    def _on_cpu_toggled(self, checked: bool) -> None:
        """CPU and GPU are mutually exclusive: checking CPU clears all GPU boxes."""
        if checked:
            for cb in self._gpu_checks:
                cb.blockSignals(True)
                cb.setChecked(False)
                cb.blockSignals(False)

    def _on_gpu_toggled(self, checked: bool) -> None:
        """Checking any GPU clears the CPU box."""
        if checked and self._cpu_check.isChecked():
            self._cpu_check.blockSignals(True)
            self._cpu_check.setChecked(False)
            self._cpu_check.blockSignals(False)

    def _build_common_group(self) -> QGroupBox:
        box = QGroupBox("Training")
        form = QFormLayout(box)
        form.setSpacing(4)

        self._epochs = QSpinBox()
        self._epochs.setRange(1, 10000)
        self._epochs.setValue(100)
        form.addRow("Epochs:", self._epochs)

        self._batch = QSpinBox()
        self._batch.setRange(1, 256)
        self._batch.setValue(8)
        form.addRow("Batch size:", self._batch)

        return box

    # ── slots ─────────────────────────────────────────────────────────────────

    def _on_model_toggled(self, yolo_checked: bool) -> None:
        self._stack.setCurrentIndex(0 if yolo_checked else 1)

    def _on_yolo_model_changed(self, tag: str) -> None:
        """Show a download hint if the selected preset model isn't cached locally."""
        if tag not in _YOLO_DOWNLOAD_URLS:
            # Custom path from Browse — no hint needed
            self._yolo_download_hint.setVisible(False)
            return
        resolved = _resolve_yolo_model(tag)
        if resolved != tag:
            # Found in local cache
            self._yolo_download_hint.setVisible(False)
        else:
            url = _YOLO_DOWNLOAD_URLS[tag]
            self._yolo_download_hint.setText(
                f"Model not found in local cache — will attempt download. "
                f"If offline, download manually and use Browse:\n{url}"
            )
            self._yolo_download_hint.setVisible(True)

    def _browse_yolo_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO model weights", "", "PyTorch weights (*.pt)"
        )
        if path:
            # Add as a custom entry if not already present
            existing = [self._yolo_base.itemText(i) for i in range(self._yolo_base.count())]
            if path not in existing:
                self._yolo_base.addItem(path)
            self._yolo_base.setCurrentText(path)
            self._yolo_download_hint.setVisible(False)

    def _on_refresh_dataset(self) -> None:
        path = self._dir_edit.text().strip()
        if path:
            self._scan_dataset(Path(path))

    def _on_dir_typed(self) -> None:
        path = self._dir_edit.text().strip()
        if path:
            self._scan_dataset(Path(path))

    def _browse_dataset(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self, "Select Dataset Directory", ""
        )
        if path:
            self._dir_edit.setText(path)
            self._scan_dataset(Path(path))

    def _scan_dataset(self, path: Path) -> None:
        ann_path   = path / "annotations.json"
        splits_dir = path / "splits"

        if not ann_path.exists():
            self._dataset_info.setPlainText(
                "No annotations.json found — export training data first."
            )
            return

        try:
            coco = json.loads(ann_path.read_text())
        except Exception as exc:
            self._dataset_info.setPlainText(f"Could not read annotations.json: {exc}")
            return

        _skip = {"background", "ignore"}
        cat_map = {
            c["id"]: c["name"] for c in coco.get("categories", [])
            if c["name"].lower() not in _skip
        }
        class_names = list(dict.fromkeys(cat_map.values()))

        # Count unique source images (before tiling/augmentation)
        source_ids = {im.get("source_image_id", im["id"]) for im in coco.get("images", [])}
        n_source = len(source_ids)
        n_tiles  = len(coco.get("images", []))
        n_total_anns = sum(
            1 for a in coco.get("annotations", [])
            if cat_map.get(a.get("category_id"))
        )

        finalized = splits_dir.exists() and (splits_dir / "train.json").exists()
        lines = []

        if finalized:
            col_w = max((len(n) for n in class_names), default=5) + 2
            col_w = max(col_w, len("Tiles (aug)") + 2)
            header = f"{'Class':<{col_w}}"
            rows: dict[str, dict[str, int]] = {n: {} for n in class_names}
            split_totals: dict[str, int] = {}
            split_sources: dict[str, int] = {}

            for split in ("train", "val", "test"):
                sf = splits_dir / f"{split}.json"
                if not sf.exists():
                    continue
                sc = json.loads(sf.read_text())
                counts: dict[str, int] = {n: 0 for n in class_names}
                for ann in sc.get("annotations", []):
                    name = cat_map.get(ann.get("category_id"))
                    if name:
                        counts[name] = counts.get(name, 0) + 1
                split_totals[split] = len(sc.get("images", []))
                split_sources[split] = len({
                    im.get("source_image_id", im["id"])
                    for im in sc.get("images", [])
                })
                header += f"  {split.capitalize():>7}"
                for name in class_names:
                    rows[name][split] = counts.get(name, 0)

            sep = "-" * (col_w + len(split_totals) * 10)
            lines.append(header)
            lines.append(sep)
            for name in class_names:
                row_str = f"{name:<{col_w}}"
                for split in split_totals:
                    row_str += f"  {rows[name].get(split, 0):>7}"
                lines.append(row_str)
            lines.append(sep)
            src_row = f"{'Source imgs':<{col_w}}"
            for split in split_totals:
                src_row += f"  {split_sources.get(split, 0):>7}"
            lines.append(src_row)
            tile_row = f"{'Tiles (aug)':<{col_w}}"
            for split, n in split_totals.items():
                tile_row += f"  {n:>7}"
            lines.append(tile_row)
        else:
            lines.append(f"Source images : {n_source}")
            lines.append(f"Tiles (aug)   : {n_tiles}")
            lines.append(f"Annotations   : {n_total_anns}")
            lines.append(f"Classes       : {', '.join(class_names) or 'none'}")
            lines.append("")
            lines.append("Not finalized — run Finalize Dataset in the Export tab.")

        self._dataset_info.setPlainText("\n".join(lines))

    def _on_train_clicked(self) -> None:
        dataset_dir = self._dir_edit.text().strip()
        if not dataset_dir:
            self.append_log("Select a dataset directory first.")
            return

        # Determine device list
        devices: list[int] | str
        selected_gpus = [
            self._gpus[i][0]
            for i, cb in enumerate(self._gpu_checks)
            if cb.isChecked()
        ]
        # Any selected GPU wins; CPU only when no GPU is selected.
        devices = selected_gpus if selected_gpus else "cpu"

        config: dict = {
            "dataset_dir": dataset_dir,
            "epochs":      self._epochs.value(),
            "batch":       self._batch.value(),
            "devices":     devices,
        }

        if self._yolo_radio.isChecked():
            config["model_type"] = "yolo"
            config["base_model"] = _resolve_yolo_model(self._yolo_base.currentText())
            config["imgsz"]      = self._yolo_imgsz.value()
        else:
            config["model_type"] = "unet"
            config["arch"]       = self._unet_arch.currentText()
            config["encoder"]    = self._unet_encoder.currentText()
            config["lr"]         = self._parse_lr(self._unet_lr.currentText())
            config["imgsz"]      = self._unet_imgsz.value()

        self.set_training(True)
        self._progress.setValue(0)
        self._progress.setFormat("Starting…")
        self._loss_epochs.clear()
        self._loss_train.clear()
        self._loss_metric.clear()
        self._reset_loss_plot()
        self.train_requested.emit(config)

    # ── public API ────────────────────────────────────────────────────────────

    def _ensure_loss_canvas(self) -> bool:
        """Create the matplotlib canvas on first use (deferred to avoid slow init at startup)."""
        global _MPL_OK
        if self._loss_canvas is not None:
            return True
        if _MPL_OK is False:
            return False
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as _FigCanvas
            _MPL_OK = True
        except Exception:
            _MPL_OK = False
            return False
        self._loss_fig = Figure(figsize=(4, 1.8), dpi=80, facecolor="#1a1a1a")
        self._loss_ax = self._loss_fig.add_subplot(111)
        self._loss_fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.22)
        self._loss_canvas = _FigCanvas(self._loss_fig)
        self._loss_canvas.setFixedHeight(145)
        self._loss_canvas.setStyleSheet("background:#1a1a1a;")
        self._loss_chart_layout.insertWidget(
            self._loss_chart_layout.count() - 1,  # before the log widget
            self._loss_canvas,
        )
        return True

    def _reset_loss_plot(self) -> None:
        """Clear loss-curve axes and apply dark styling."""
        if not self._ensure_loss_canvas():
            return
        ax = self._loss_ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#363636")
        ax.set_xlabel("Epoch", color="#888888", fontsize=8)
        ax.set_ylabel("Loss / Metric", color="#888888", fontsize=8)
        ax.set_title("Training Progress", color="#e0e0e0", fontsize=9, pad=4)
        self._loss_fig.canvas.draw_idle()

    def update_loss_curve(self, epoch: int, train_loss: float, metric: float) -> None:
        """Append one epoch's data and refresh the chart.

        Called on the main thread from the log-tail QTimer (training itself runs
        in a separate subprocess), so direct widget/canvas updates are safe.

        Parameters
        ----------
        epoch      : 1-based epoch number
        train_loss : training loss for that epoch
        metric     : seg_mAP50 (YOLO) or mean F1 (UNet)
        """
        if not self._ensure_loss_canvas():
            return
        self._loss_epochs.append(epoch)
        self._loss_train.append(train_loss)
        self._loss_metric.append(metric)

        ax = self._loss_ax
        ax.cla()
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="#888888", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#363636")
        ax.set_xlabel("Epoch", color="#888888", fontsize=8)
        ax.set_title("Training Progress", color="#e0e0e0", fontsize=9, pad=4)

        ax.plot(self._loss_epochs, self._loss_train,
                color="#4dbb78", lw=1.5, label="loss")
        if any(v == v for v in self._loss_metric):  # skip if all NaN
            ax2 = ax.twinx()
            ax2.set_facecolor("#1a1a1a")
            ax2.tick_params(colors="#4d8ec4", labelsize=8)
            ax2.spines["right"].set_edgecolor("#363636")
            ax2.plot(self._loss_epochs, self._loss_metric,
                     color="#4d8ec4", lw=1.5, linestyle="--", label="metric")
            ax2.set_ylabel("mAP50 / mF1", color="#4d8ec4", fontsize=8)

        self._loss_fig.canvas.draw_idle()

    def append_log(self, msg: str) -> None:
        self._log.appendPlainText(msg)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def set_progress(self, current: int, total: int) -> None:
        pct = int(100 * current / max(total, 1))
        self._progress.setValue(pct)
        self._progress.setFormat(f"Epoch {current}/{total}  ({pct}%)")

    def set_training(self, active: bool) -> None:
        self._training = active
        self._train_btn.setEnabled(not active)
        self._cancel_btn.setEnabled(active)
        if not active:
            self._progress.setFormat(
                "Complete" if self._progress.value() == 100 else "Stopped"
            )

    def training_finished(self, model_type: str, model_path: str) -> None:
        self.set_training(False)
        self._progress.setValue(100)
        self._progress.setFormat("Complete")
        self.append_log(f"Model saved: {model_path}")
        if model_type == "yolo":
            self.load_yolo_requested.emit(model_path)
        else:
            self.load_unet_requested.emit(model_path)

    def training_failed(self, error: str) -> None:
        self.set_training(False)
        self.append_log(f"ERROR: {error}")
        self._progress.setFormat("Failed")
