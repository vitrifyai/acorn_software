"""SEM 3D surface area analysis panel."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QProgressBar,
    QScrollArea, QSpinBox, QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)


class SEMPanel(QWidget):
    """
    Physics-informed SEM 3D surface area estimation panel.

    Workflow:
      1. Select annotation labels.
      2. Set detector geometry (alpha/phi) from instrument docs, or Auto-calibrate.
      3. Optionally load a trained U-Net checkpoint for residual correction.
      4. Click Run — shape-from-shading recovers h(x,y) per particle,
         then SA = integral sqrt(1 + p^2 + q^2) * px^2.
    """

    sem_requested   = pyqtSignal(dict)   # run SA estimation
    train_requested = pyqtSignal(dict)   # train U-Net on synthetic data

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._px_nm = 1.0
        self._checkpoint: Optional[str] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        _content = QWidget()
        layout = QVBoxLayout(_content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Labels ────────────────────────────────────────────────────
        lbl_box = QGroupBox("Annotation Labels")
        lbl_layout = QVBoxLayout(lbl_box)
        lbl_layout.setSpacing(3)
        self._label_checks: list[QCheckBox] = []
        self._label_container = QVBoxLayout()
        lbl_layout.addLayout(self._label_container)
        lbl_hint = QLabel("(no ROI annotations loaded)")
        lbl_hint.setStyleSheet("color:#888; font-size:11px;")
        self._label_hint = lbl_hint
        lbl_layout.addWidget(lbl_hint)
        layout.addWidget(lbl_box)

        # ── Detector geometry ─────────────────────────────────────────
        det_box = QGroupBox("Detector Geometry")
        det_form = QFormLayout(det_box)
        det_form.setSpacing(4)

        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0, 90)
        self._alpha.setValue(25.0)
        self._alpha.setSuffix("°")
        self._alpha.setToolTip(
            "Detector elevation angle from vertical (beam axis).\n"
            "Typical ET detector: 20–35°.  Check instrument documentation."
        )
        det_form.addRow("Elevation (α):", self._alpha)

        self._phi = QDoubleSpinBox()
        self._phi.setRange(0, 360)
        self._phi.setValue(0.0)
        self._phi.setSuffix("°")
        self._phi.setToolTip(
            "Detector azimuth angle (0° = right, 90° = top of image).\n"
            "Depends on instrument orientation."
        )
        det_form.addRow("Azimuth (φ):", self._phi)

        self._I_bg = QDoubleSpinBox()
        self._I_bg.setRange(-1e6, 1e6)
        self._I_bg.setDecimals(4)
        self._I_bg.setValue(0.0)
        self._I_bg.setToolTip(
            "Background intensity (dark substrate level).\n"
            "Click Auto-calibrate to estimate from the current image."
        )
        det_form.addRow("I background:", self._I_bg)

        self._eta0 = QDoubleSpinBox()
        self._eta0.setRange(0.001, 1e6)
        self._eta0.setDecimals(4)
        self._eta0.setValue(1.0)
        self._eta0.setToolTip(
            "Contrast scale (SE yield factor).\n"
            "Auto-calibrate estimates this from image intensity range."
        )
        det_form.addRow("η₀ contrast:", self._eta0)

        self._lam = QDoubleSpinBox()
        self._lam.setRange(0.0, 1.0)
        self._lam.setSingleStep(0.05)
        self._lam.setDecimals(2)
        self._lam.setValue(0.30)
        self._lam.setToolTip(
            "Detector asymmetry factor λ ∈ [0, 1].\n"
            "0 = isotropic SE emission, 1 = full Lambertian shading toward detector."
        )
        det_form.addRow("λ asymmetry:", self._lam)

        calib_btn = QPushButton("Auto-calibrate from image")
        calib_btn.setToolTip(
            "Estimates I_bg and η₀ from the current image intensity distribution."
        )
        calib_btn.clicked.connect(self._on_auto_calibrate)
        det_form.addRow("", calib_btn)

        self._learn_det = QCheckBox("Alternating optimisation (experimental)")
        self._learn_det.setToolTip(
            "Also optimise α and φ during shape-from-shading.\n"
            "Only reliable for particles with strong bilateral asymmetry."
        )
        det_form.addRow("", self._learn_det)

        layout.addWidget(det_box)

        # ── Shape-from-shading settings ───────────────────────────────
        sfs_box = QGroupBox("Shape-from-Shading")
        sfs_form = QFormLayout(sfs_box)
        sfs_form.setSpacing(4)

        self._n_iters = QSpinBox()
        self._n_iters.setRange(50, 2000)
        self._n_iters.setValue(300)
        self._n_iters.setToolTip("Adam optimisation iterations per particle.")
        sfs_form.addRow("Iterations:", self._n_iters)

        self._smooth = QDoubleSpinBox()
        self._smooth.setRange(0.001, 10.0)
        self._smooth.setDecimals(3)
        self._smooth.setSingleStep(0.01)
        self._smooth.setValue(0.10)
        self._smooth.setToolTip(
            "Laplacian smoothness weight.  Higher = smoother height field,\n"
            "lower = more detail but noisier."
        )
        sfs_form.addRow("Smoothness:", self._smooth)

        self._lr = QDoubleSpinBox()
        self._lr.setRange(1e-5, 0.5)
        self._lr.setDecimals(4)
        self._lr.setSingleStep(0.001)
        self._lr.setValue(0.005)
        sfs_form.addRow("Learning rate:", self._lr)

        layout.addWidget(sfs_box)

        # ── Neural network correction ─────────────────────────────────
        nn_box = QGroupBox("U-Net Residual Correction (optional)")
        nn_layout = QVBoxLayout(nn_box)
        nn_layout.setSpacing(4)

        self._use_nn = QCheckBox("Apply U-Net height correction")
        self._use_nn.setEnabled(False)
        self._use_nn.setToolTip("Load a checkpoint below to enable.")
        nn_layout.addWidget(self._use_nn)

        ckpt_row = QHBoxLayout()
        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("No checkpoint loaded")
        self._ckpt_edit.setReadOnly(True)
        ckpt_browse = QPushButton("Browse…")
        ckpt_browse.setFixedWidth(75)
        ckpt_browse.clicked.connect(self._on_browse_ckpt)
        ckpt_row.addWidget(self._ckpt_edit, 1)
        ckpt_row.addWidget(ckpt_browse)
        nn_layout.addLayout(ckpt_row)

        train_btn = QPushButton("Train U-Net on synthetic data…")
        train_btn.setToolTip(
            "Train a small U-Net on synthetically generated SEM images.\n"
            "Generates spheres / ellipsoids / rough surfaces, renders them\n"
            "with the physics model, and trains on the residual corrections."
        )
        train_btn.clicked.connect(self._on_train_nn)
        nn_layout.addWidget(train_btn)

        layout.addWidget(nn_box)

        # ── Pixel size + mode ─────────────────────────────────────────
        run_box = QGroupBox("Run")
        run_form = QFormLayout(run_box)
        run_form.setSpacing(4)

        self._px = QDoubleSpinBox()
        self._px.setRange(0.001, 1e6)
        self._px.setDecimals(4)
        self._px.setValue(1.0)
        self._px.setSuffix(" nm/px")
        run_form.addRow("Pixel size:", self._px)

        out_row = QHBoxLayout()
        self._out_dir = QLineEdit()
        self._out_dir.setPlaceholderText("Auto (next to image file)")
        out_browse = QPushButton("Browse…")
        out_browse.setFixedWidth(75)
        out_browse.clicked.connect(self._on_browse_out)
        out_row.addWidget(self._out_dir, 1)
        out_row.addWidget(out_browse)
        run_form.addRow("Output dir:", out_row)

        self._run_btn = QPushButton("Run SEM 3D Analysis")
        self._run_btn.setStyleSheet("background:#2e86c1;color:white;font-weight:bold;")
        self._run_btn.clicked.connect(self._on_run)
        run_form.addRow("", self._run_btn)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size:11px;")
        run_form.addRow("", self._status)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setFixedHeight(16)
        self._progress.setVisible(False)
        run_form.addRow("", self._progress)

        layout.addWidget(run_box)
        layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(_content)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        # ── Results tab ───────────────────────────────────────────────
        self._results_tabs = QTabWidget()
        self._results_tabs.setVisible(False)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._results_tabs.addTab(self._table, "Particles")

        # Outer splitter-like layout
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll, 2)
        outer.addWidget(self._results_tabs, 1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_pixel_size(self, ps_nm: float) -> None:
        if ps_nm and ps_nm > 0:
            self._px.setValue(ps_nm)

    def refresh_labels(self, labels: list[str]) -> None:
        unique = sorted(set(l for l in labels if l))
        # Clear existing
        for cb in self._label_checks:
            self._label_container.removeWidget(cb)
            cb.deleteLater()
        self._label_checks.clear()

        if not unique:
            self._label_hint.setVisible(True)
            return
        self._label_hint.setVisible(False)
        for lbl in unique:
            cb = QCheckBox(lbl)
            cb.setChecked(True)
            self._label_checks.append(cb)
            self._label_container.addWidget(cb)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)
        if not running:
            self._progress.setValue(0)

    def show_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status.setText(msg)

    def show_results(self, df, out_dir) -> None:
        import pandas as pd
        self._results_tabs.setVisible(True)

        cols = ["particle_id", "label", "image_name",
                "SA_sem_nm2", "SA_2d_nm2", "roughness_rms", "method"]
        present = [c for c in cols if c in df.columns]
        self._table.setColumnCount(len(present))
        self._table.setHorizontalHeaderLabels(present)
        self._table.setRowCount(len(df))

        for r, row in enumerate(df.itertuples(index=False)):
            for c, col in enumerate(present):
                val = getattr(row, col, "")
                if isinstance(val, float):
                    text = f"{val:.4g}"
                else:
                    text = str(val)
                self._table.setItem(r, c, QTableWidgetItem(text))

        self._table.resizeColumnsToContents()
        self._results_tabs.setCurrentIndex(0)

        if out_dir:
            self._status.setText(f"Done — {len(df)} particles, results in {out_dir}")
        else:
            self._status.setText(f"Done — {len(df)} particles")

    # ------------------------------------------------------------------
    # Internal callbacks
    # ------------------------------------------------------------------

    def _on_auto_calibrate(self) -> None:
        """Estimate I_bg and eta0 from current image via context signal (no-op if no image)."""
        # The plugin will call _do_calibrate(image_array) when it receives this request.
        # For now, emit a dummy request; the plugin overrides this after construction.
        if hasattr(self, "_calibrate_cb") and self._calibrate_cb:
            self._calibrate_cb()

    def set_calibrate_callback(self, cb) -> None:
        self._calibrate_cb = cb

    def apply_calibration(self, I_bg: float, eta0: float) -> None:
        self._I_bg.setValue(I_bg)
        self._eta0.setValue(eta0)

    def _on_browse_ckpt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select U-Net checkpoint", "", "PyTorch weights (*.pt)"
        )
        if path:
            self._checkpoint = path
            self._ckpt_edit.setText(path)
            self._use_nn.setEnabled(True)
            self._use_nn.setChecked(True)

    def _on_browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select output directory")
        if d:
            self._out_dir.setText(d)

    def _on_run(self) -> None:
        selected = [cb.text() for cb in self._label_checks if cb.isChecked()]
        if not selected:
            self._status.setText("Select at least one label.")
            return

        config = {
            "selected_labels": selected,
            "pixel_size_nm":   self._px.value(),
            "alpha_deg":       self._alpha.value(),
            "phi_deg":         self._phi.value(),
            "I_bg":            self._I_bg.value(),
            "eta0":            self._eta0.value(),
            "lam":             self._lam.value(),
            "learn_detector":  self._learn_det.isChecked(),
            "n_iters":         self._n_iters.value(),
            "smoothness":      self._smooth.value(),
            "lr":              self._lr.value(),
            "use_nn":          self._use_nn.isChecked(),
            "checkpoint":      self._checkpoint or "",
            "output_dir":      self._out_dir.text().strip(),
        }
        self.sem_requested.emit(config)

    def _on_train_nn(self) -> None:
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        dlg = _TrainDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.train_requested.emit(dlg.config())


# ---------------------------------------------------------------------------
# Training dialog
# ---------------------------------------------------------------------------

class _TrainDialog(QWidget):
    def __init__(self, parent=None) -> None:
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox
        super().__init__(parent)
        self._dlg = QDialog(parent)
        self._dlg.setWindowTitle("Train SEM U-Net")
        layout = QVBoxLayout(self._dlg)

        form = QFormLayout()
        self._n_samples = QSpinBox()
        self._n_samples.setRange(200, 20000)
        self._n_samples.setValue(2000)
        form.addRow("Synthetic samples:", self._n_samples)

        self._epochs = QSpinBox()
        self._epochs.setRange(5, 500)
        self._epochs.setValue(50)
        form.addRow("Epochs:", self._epochs)

        self._batch = QSpinBox()
        self._batch.setRange(1, 256)
        self._batch.setValue(16)
        form.addRow("Batch size:", self._batch)

        self._imgsz = QSpinBox()
        self._imgsz.setRange(64, 512)
        self._imgsz.setSingleStep(64)
        self._imgsz.setValue(128)
        form.addRow("Tile size (px):", self._imgsz)

        out_row = QHBoxLayout()
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Choose output directory…")
        browse = QPushButton("Browse…")
        browse.setFixedWidth(75)
        browse.clicked.connect(self._browse)
        out_row.addWidget(self._out_edit, 1)
        out_row.addWidget(browse)
        form.addRow("Save checkpoint:", out_row)

        layout.addLayout(form)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._dlg.accept)
        btns.rejected.connect(self._dlg.reject)
        layout.addWidget(btns)

    def exec(self):
        return self._dlg.exec()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self._dlg, "Output directory for checkpoint")
        if d:
            self._out_edit.setText(d)

    def config(self) -> dict:
        return {
            "n_samples":  self._n_samples.value(),
            "epochs":     self._epochs.value(),
            "batch_size": self._batch.value(),
            "image_size": self._imgsz.value(),
            "output_dir": self._out_edit.text().strip(),
        }
