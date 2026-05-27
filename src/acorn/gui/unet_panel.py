"""UNet semantic / instance segmentation annotation panel."""

from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QSpinBox, QVBoxLayout, QWidget,
)

from acorn.core.unet_predictor import SMP_ARCHITECTURES, SMP_ENCODERS


class UNetPanel(QWidget):
    """
    Controls for UNet-based semantic/instance segmentation.

    Works with any segmentation_models_pytorch architecture or a raw
    PyTorch .pt exported from a custom training pipeline.  Suitable for
    any EM modality: cryo-EM, STEM, TEM, EDX, tomography, materials EM, etc.

    Workflow
    --------
    1. Select architecture and encoder (or leave defaults for Unet/resnet34)
    2. Browse to a .pt checkpoint and click Load Model
    3. Set inference parameters (threshold, foreground class, tile size)
    4. Click Run Segmentation
    5. Predicted instance masks appear as ROI annotations
    6. Undo unwanted masks, or Accept All

    Signals
    -------
    load_model_requested(arch, encoder, in_channels, n_classes, ckpt_path)
    segment_requested()
    accept_all_requested()
    reject_all_requested()
    """

    load_model_requested = pyqtSignal(str, str, int, int, str)
    segment_requested    = pyqtSignal()
    accept_all_requested = pyqtSignal()
    reject_all_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        _content = QWidget()
        layout = QVBoxLayout(_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── architecture ───────────────────────────────────────────────────────
        arch_box = QGroupBox("Architecture")
        arch_layout = QFormLayout(arch_box)

        self._arch_combo = QComboBox()
        for a in SMP_ARCHITECTURES:
            self._arch_combo.addItem(a)
        self._arch_combo.setToolTip(
            "segmentation_models_pytorch architecture.\n"
            "Unet and UnetPlusPlus are the most common choices."
        )
        arch_layout.addRow("Architecture:", self._arch_combo)

        self._enc_combo = QComboBox()
        for e in SMP_ENCODERS:
            self._enc_combo.addItem(e)
        self._enc_combo.setToolTip("Encoder backbone network")
        arch_layout.addRow("Encoder:", self._enc_combo)

        self._in_channels = QSpinBox()
        self._in_channels.setRange(1, 4)
        self._in_channels.setValue(1)
        self._in_channels.setToolTip(
            "Input channels: 1 for grayscale EM images, 3 for RGB"
        )
        arch_layout.addRow("Input channels:", self._in_channels)

        self._n_classes = QSpinBox()
        self._n_classes.setRange(1, 32)
        self._n_classes.setValue(2)
        self._n_classes.setToolTip(
            "Number of output classes.\n"
            "2 = background + foreground (standard binary segmentation)."
        )
        arch_layout.addRow("Output classes:", self._n_classes)

        layout.addWidget(arch_box)

        # ── checkpoint ─────────────────────────────────────────────────────────
        ckpt_box = QGroupBox("Checkpoint")
        ckpt_layout = QFormLayout(ckpt_box)

        ckpt_row = QHBoxLayout()
        self._ckpt = QLineEdit()
        self._ckpt.setPlaceholderText("Path to .pt checkpoint (required)")
        browse_btn = QPushButton("Browse…")
        browse_btn.setFixedWidth(75)
        browse_btn.clicked.connect(self._browse_checkpoint)
        ckpt_row.addWidget(self._ckpt, 1)
        ckpt_row.addWidget(browse_btn)
        ckpt_layout.addRow("Checkpoint:", ckpt_row)

        load_btn = QPushButton("Load Model")
        load_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        load_btn.setToolTip(
            "Load the model weights from the selected checkpoint.\n"
            "The architecture and encoder must match the checkpoint."
        )
        load_btn.clicked.connect(self._on_load_model)
        ckpt_layout.addRow("", load_btn)

        self._model_status = QLabel("Model not loaded")
        self._model_status.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._model_status.setWordWrap(True)
        ckpt_layout.addRow("", self._model_status)

        layout.addWidget(ckpt_box)

        # ── inference parameters ───────────────────────────────────────────────
        infer_box = QGroupBox("Inference")
        infer_layout = QFormLayout(infer_box)

        self._threshold = QDoubleSpinBox()
        self._threshold.setRange(0.01, 0.99)
        self._threshold.setSingleStep(0.05)
        self._threshold.setValue(0.50)
        self._threshold.setToolTip("Foreground probability threshold")
        infer_layout.addRow("Threshold:", self._threshold)

        self._fg_class = QSpinBox()
        self._fg_class.setRange(0, 31)
        self._fg_class.setValue(1)
        self._fg_class.setToolTip(
            "Class index to treat as foreground.\n"
            "For a 2-class model: 0=background, 1=foreground."
        )
        infer_layout.addRow("Foreground class:", self._fg_class)

        self._min_area = QSpinBox()
        self._min_area.setRange(1, 1000000)
        self._min_area.setValue(50)
        self._min_area.setSuffix(" px")
        self._min_area.setToolTip(
            "Discard connected components smaller than this area.\n"
            "Increase to filter out noise predictions."
        )
        infer_layout.addRow("Min area:", self._min_area)

        self._tile_size = QSpinBox()
        self._tile_size.setRange(64, 8192)
        self._tile_size.setValue(512)
        self._tile_size.setSuffix(" px")
        self._tile_size.setToolTip(
            "Images larger than this are split into overlapping tiles.\n"
            "Smaller tiles use less GPU memory but may miss large objects."
        )
        infer_layout.addRow("Tile size:", self._tile_size)

        self._label_combo = QComboBox()
        for lbl in ["Foreground", "Background", "Ignore"]:
            self._label_combo.addItem(lbl)
        infer_layout.addRow("Label:", self._label_combo)

        run_btn = QPushButton("Run Segmentation")
        run_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        run_btn.setToolTip("Run UNet segmentation on the current image")
        run_btn.clicked.connect(self.segment_requested)
        infer_layout.addRow("", run_btn)

        layout.addWidget(infer_box)

        # ── status and accept / reject ─────────────────────────────────────────
        self._status = QLabel("")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        ar_row = QHBoxLayout()
        accept_btn = QPushButton("Accept All")
        accept_btn.setStyleSheet("background:#00703C;color:white;font-weight:bold;")
        accept_btn.setToolTip("Keep all pending UNet masks as permanent ROI annotations")
        accept_btn.clicked.connect(self.accept_all_requested)

        reject_btn = QPushButton("Reject All")
        reject_btn.setStyleSheet("background:#c0392b;color:white;")
        reject_btn.setToolTip("Remove all pending UNet mask annotations")
        reject_btn.clicked.connect(self.reject_all_requested)

        ar_row.addWidget(accept_btn)
        ar_row.addWidget(reject_btn)
        layout.addLayout(ar_row)
        layout.addStretch()
        _scroll = QScrollArea()
        _scroll.setWidgetResizable(True)
        _scroll.setWidget(_content)
        _outer = QVBoxLayout(self)
        _outer.setContentsMargins(0, 0, 0, 0)
        _outer.addWidget(_scroll)

    # ── public API ────────────────────────────────────────────────────────────

    def set_model_status(self, msg: str, loaded: bool = False) -> None:
        color = "#4dbb78" if loaded else "palette(mid)"
        self._model_status.setStyleSheet(f"font-size: 11px; color: {color};")
        self._model_status.setText(msg)

    def set_status(self, msg: str) -> None:
        self._status.setText(msg)

    @property
    def architecture(self) -> str:
        return self._arch_combo.currentText()

    @property
    def encoder(self) -> str:
        return self._enc_combo.currentText()

    @property
    def in_channels(self) -> int:
        return self._in_channels.value()

    @property
    def n_classes(self) -> int:
        return self._n_classes.value()

    @property
    def checkpoint_path(self) -> str:
        return self._ckpt.text().strip()

    @property
    def threshold(self) -> float:
        return self._threshold.value()

    @property
    def foreground_class(self) -> int:
        return self._fg_class.value()

    @property
    def min_area(self) -> int:
        return self._min_area.value()

    @property
    def tile_size(self) -> int:
        return self._tile_size.value()

    @property
    def label(self) -> str:
        return self._label_combo.currentText()

    # ── slots ──────────────────────────────────────────────────────────────────

    def _browse_checkpoint(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select UNet checkpoint", "",
            "PyTorch checkpoints (*.pt *.pth);;All files (*)"
        )
        if path:
            self._ckpt.setText(path)

    def _on_load_model(self) -> None:
        ckpt = self._ckpt.text().strip()
        if not ckpt:
            self._model_status.setText("Browse to a .pt checkpoint first.")
            return
        self._model_status.setText("Loading…")
        self.load_model_requested.emit(
            self.architecture,
            self.encoder,
            self.in_channels,
            self.n_classes,
            ckpt,
        )
