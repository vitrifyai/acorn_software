"""Control panel for focused FIB-SEM simulation."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class FibSimPanel(QWidget):
    generate_requested = pyqtSignal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_sample_group())
        layout.addWidget(self._build_milling_group())
        layout.addWidget(self._build_detector_group())

        self._generate_btn = QPushButton("Generate FIB Simulation")
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #6c7086;")
        layout.addWidget(self._status)
        layout.addStretch()

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self._output_dir = QLineEdit(str(Path.home() / "fib_sim_output"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row.addWidget(self._output_dir, 1)
        row.addWidget(browse)
        form.addRow("Folder:", row)

        self._count = QSpinBox()
        self._count.setRange(1, 100000)
        self._count.setValue(10)
        form.addRow("Images:", self._count)

        self._open_generated = QCheckBox("Open generated images in Acorn")
        self._open_generated.setChecked(True)
        form.addRow("", self._open_generated)

        self._save_layers = QCheckBox("Save truth/milled/edge/artifact layers")
        self._save_layers.setChecked(True)
        form.addRow("", self._save_layers)
        return box

    def _build_sample_group(self) -> QGroupBox:
        box = QGroupBox("Phantom and Scene")
        form = QFormLayout(box)

        self._sample = QComboBox()
        self._sample.addItem("Biological surface", "bio")
        self._sample.addItem("Materials surface", "material")
        form.addRow("Sample:", self._sample)

        self._width = QSpinBox()
        self._width.setRange(32, 8192)
        self._width.setValue(640)
        self._height = QSpinBox()
        self._height.setRange(32, 8192)
        self._height.setValue(512)
        size_row = QHBoxLayout()
        size_row.addWidget(self._width)
        size_row.addWidget(QLabel("x"))
        size_row.addWidget(self._height)
        form.addRow("Width x height:", size_row)

        self._pixel_size = QDoubleSpinBox()
        self._pixel_size.setRange(0.001, 100000.0)
        self._pixel_size.setDecimals(3)
        self._pixel_size.setValue(5.0)
        form.addRow("Pixel size nm/px:", self._pixel_size)

        self._seed = QSpinBox()
        self._seed.setRange(0, 2147483647)
        self._seed.setValue(1)
        form.addRow("Seed:", self._seed)

        self._liftout = QCheckBox("Lift-out geometry")
        self._liftout.toggled.connect(self._on_liftout_changed)
        form.addRow("", self._liftout)
        self._trench = QCheckBox("Milled trenches")
        self._trench.setChecked(True)
        form.addRow("", self._trench)
        self._pt_cap = QCheckBox("Pt cap")
        self._pt_cap.setChecked(True)
        form.addRow("", self._pt_cap)
        self._needle = QCheckBox("Needle / manipulator")
        form.addRow("", self._needle)
        self._on_liftout_changed(False)
        return box

    def _build_milling_group(self) -> QGroupBox:
        box = QGroupBox("FIB Milling Artifacts")
        form = QFormLayout(box)

        self._mill_axis = QComboBox()
        self._mill_axis.addItem("Top-down / y", "y")
        self._mill_axis.addItem("Left-right / x", "x")
        form.addRow("Mill axis:", self._mill_axis)
        self._curtain = self._spin(0.25, 0.0, 1.0, 0.05)
        form.addRow("Curtaining:", self._curtain)
        self._edge_gain = self._spin(2.0, 0.0, 8.0, 0.25)
        form.addRow("Edge-gated curtain gain:", self._edge_gain)
        self._gradient = self._spin(0.15, 0.0, 1.0, 0.05)
        form.addRow("Mill gradient:", self._gradient)
        self._redeposition = self._spin(0.06, 0.0, 1.0, 0.02)
        form.addRow("Redeposition:", self._redeposition)
        self._charging = self._spin(0.0, 0.0, 1.0, 0.05)
        form.addRow("Charging:", self._charging)
        self._lamella = self._spin(200.0, 1.0, 100000.0, 25.0)
        self._lamella.setDecimals(1)
        form.addRow("Lamella thickness nm:", self._lamella)
        return box

    def _build_detector_group(self) -> QGroupBox:
        box = QGroupBox("Detector")
        form = QFormLayout(box)
        self._electrons = self._spin(400.0, 1.0, 1000000.0, 50.0)
        self._electrons.setDecimals(1)
        form.addRow("Electrons / pixel:", self._electrons)
        self._mtf = self._spin(0.7, 0.0, 25.0, 0.1)
        form.addRow("MTF sigma px:", self._mtf)
        self._read_noise = self._spin(3.0, 0.0, 1000.0, 0.5)
        form.addRow("Read noise e-:", self._read_noise)
        self._jitter = self._spin(0.4, 0.0, 100.0, 0.1)
        form.addRow("Scan-line jitter %:", self._jitter)
        return box

    def _spin(self, value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(2)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose FIB simulation output folder", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    def _on_liftout_changed(self, checked: bool) -> None:
        self._trench.setEnabled(checked)
        self._pt_cap.setEnabled(checked)
        self._needle.setEnabled(checked)

    def _on_generate(self) -> None:
        output_dir = self._output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "FIB Simulation", "Choose an output folder.")
            return
        self.generate_requested.emit(
            {
                "output_dir": output_dir,
                "count": self._count.value(),
                "open_generated": self._open_generated.isChecked(),
                "save_layers": self._save_layers.isChecked(),
                "sample": self._sample.currentData(),
                "width": self._width.value(),
                "height": self._height.value(),
                "pixel_size_nm": self._pixel_size.value(),
                "seed": self._seed.value(),
                "liftout": self._liftout.isChecked(),
                "trench": self._trench.isChecked(),
                "pt_cap": self._pt_cap.isChecked(),
                "needle": self._needle.isChecked(),
                "mill_axis": self._mill_axis.currentData(),
                "curtain_strength": self._curtain.value(),
                "curtain_edge_gain": self._edge_gain.value(),
                "mill_gradient": self._gradient.value(),
                "redeposition": self._redeposition.value(),
                "charging": self._charging.value(),
                "lamella_thickness_nm": self._lamella.value(),
                "electrons_per_pixel": self._electrons.value(),
                "mtf_sigma_px": self._mtf.value(),
                "read_noise_e": self._read_noise.value(),
                "scan_line_jitter": self._jitter.value(),
            }
        )

    def set_running(self, running: bool) -> None:
        self._generate_btn.setEnabled(not running)
        self._generate_btn.setText("Generating..." if running else "Generate FIB Simulation")

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def apply_params(self, params: dict) -> None:
        self._output_dir.setText(str(params.get("output_dir", self._output_dir.text())))
        self._count.setValue(int(params.get("count", self._count.value())))
        self._open_generated.setChecked(bool(params.get("open_generated", self._open_generated.isChecked())))
        self._save_layers.setChecked(bool(params.get("save_layers", self._save_layers.isChecked())))
        self._set_combo(self._sample, params.get("sample", self._sample.currentData()))
        self._width.setValue(int(params.get("width", self._width.value())))
        self._height.setValue(int(params.get("height", self._height.value())))
        self._pixel_size.setValue(float(params.get("pixel_size_nm", self._pixel_size.value())))
        self._seed.setValue(int(params.get("seed", self._seed.value())))
        self._liftout.setChecked(bool(params.get("liftout", self._liftout.isChecked())))
        self._trench.setChecked(bool(params.get("trench", self._trench.isChecked())))
        self._pt_cap.setChecked(bool(params.get("pt_cap", self._pt_cap.isChecked())))
        self._needle.setChecked(bool(params.get("needle", self._needle.isChecked())))
        self._set_combo(self._mill_axis, params.get("mill_axis", self._mill_axis.currentData()))
        self._curtain.setValue(float(params.get("curtain_strength", self._curtain.value())))
        self._edge_gain.setValue(float(params.get("curtain_edge_gain", self._edge_gain.value())))
        self._gradient.setValue(float(params.get("mill_gradient", self._gradient.value())))
        self._redeposition.setValue(float(params.get("redeposition", self._redeposition.value())))
        self._charging.setValue(float(params.get("charging", self._charging.value())))
        self._lamella.setValue(float(params.get("lamella_thickness_nm", self._lamella.value())))
        self._electrons.setValue(float(params.get("electrons_per_pixel", self._electrons.value())))
        self._mtf.setValue(float(params.get("mtf_sigma_px", self._mtf.value())))
        self._read_noise.setValue(float(params.get("read_noise_e", self._read_noise.value())))
        self._jitter.setValue(float(params.get("scan_line_jitter", self._jitter.value())))

    def _set_combo(self, combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
