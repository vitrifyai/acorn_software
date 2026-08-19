"""Control panel for focused cryo-TEM simulation."""

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


class TemSimPanel(QWidget):
    generate_requested = pyqtSignal(dict)
    action_requested = pyqtSignal(str, dict)   # (action_name, params) — engine/4D-STEM/reference

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        layout.addWidget(self._build_output_group())
        layout.addWidget(self._build_path_group())
        layout.addWidget(self._build_optics_group())
        layout.addWidget(self._build_specimen_group())
        layout.addWidget(self._build_advanced_group())

        self._generate_btn = QPushButton("Generate TEM Simulation")
        self._generate_btn.clicked.connect(self._on_generate)
        layout.addWidget(self._generate_btn)

        self._fourd_btn = QPushButton("Generate 4D-STEM")
        self._fourd_btn.clicked.connect(lambda: self._emit_action("generate_4dstem"))
        layout.addWidget(self._fourd_btn)
        self._ref_btn = QPushButton("Match Reference Image…")
        self._ref_btn.clicked.connect(lambda: self._emit_action("simulate_from_reference"))
        layout.addWidget(self._ref_btn)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #6c7086;")
        layout.addWidget(self._status)
        layout.addStretch()

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)
        row = QHBoxLayout()
        self._output_dir = QLineEdit(str(Path.home() / "tem_sim_output"))
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
        self._save_layers = QCheckBox("Save potential/ideal/counts/label layers")
        self._save_layers.setChecked(True)
        form.addRow("", self._save_layers)
        return box

    def _build_path_group(self) -> QGroupBox:
        box = QGroupBox("Simulation Path")
        form = QFormLayout(box)
        self._simulation_path = QComboBox()
        for label, value in (
            ("Fast (quick preview)", "fast"),
            ("Full physics (multislice + detector)", "multislice"),
            ("Custom Python script", "custom"),
        ):
            self._simulation_path.addItem(label, value)
        form.addRow("Quality:", self._simulation_path)
        script_row = QHBoxLayout()
        self._custom_script = QLineEdit("")
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_custom_script)
        script_row.addWidget(self._custom_script, 1)
        script_row.addWidget(browse)
        form.addRow("Custom script:", script_row)
        self._slice_thickness = self._spin(5.0, 1.0, 50.0, 1.0)
        form.addRow("Slice thickness Å:", self._slice_thickness)
        return box

    def _build_optics_group(self) -> QGroupBox:
        box = QGroupBox("Optics, CTF, and Detector")
        form = QFormLayout(box)
        self._preset = QComboBox()
        for label, value in (
            ("Krios + K3", "krios-k3"),
            ("Glacios + Falcon4", "glacios-falcon4"),
            ("Talos + Ceta", "talos-ceta"),
            ("Cs corrected", "cs-corrected"),
        ):
            self._preset.addItem(label, value)
        form.addRow("Instrument preset:", self._preset)
        self._size = QSpinBox()
        self._size.setRange(64, 8192)
        self._size.setValue(512)
        form.addRow("Image size px:", self._size)
        self._pixel_size = self._spin(1.5, 0.1, 50, 0.1)
        form.addRow("Pixel size Å/px:", self._pixel_size)
        self._voltage = QComboBox()
        for kv in ("300", "200", "120", "100"):
            self._voltage.addItem(f"{kv} kV", kv)
        form.addRow("Voltage:", self._voltage)
        self._cs = self._spin(2.7, 0.0, 10.0, 0.1)
        form.addRow("Cs mm:", self._cs)
        self._amp = self._spin(0.10, 0.0, 0.5, 0.01)
        form.addRow("Amplitude contrast:", self._amp)
        self._dose = self._spin(40.0, 0.5, 300, 1.0)
        form.addRow("Dose e-/Å²:", self._dose)
        self._detector = QComboBox()
        for model in ("K3", "Falcon4", "K2", "Ceta"):
            self._detector.addItem(model, model)
        form.addRow("Detector:", self._detector)
        self._defocus_min = self._spin(-1.0, -8.0, 2.0, 0.1)
        form.addRow("Defocus start µm:", self._defocus_min)
        self._defocus_max = self._spin(-2.5, -8.0, 2.0, 0.1)
        form.addRow("Defocus end µm:", self._defocus_max)
        self._phase_plate = QCheckBox("Volta phase plate")
        form.addRow("", self._phase_plate)
        self._seed = QSpinBox()
        self._seed.setRange(0, 2147483647)
        self._seed.setValue(1)
        form.addRow("Seed:", self._seed)
        return box

    def _build_specimen_group(self) -> QGroupBox:
        box = QGroupBox("Specimen in Vitreous Ice")
        form = QFormLayout(box)
        self._specimen_kind = QComboBox()
        for label, value in (
            ("Solid particles / PLGA", "plga"),
            ("Single lipid vesicles", "lipid_single"),
            ("Multilamellar lipid vesicles", "lipid_multi"),
            ("Protein particles", "protein"),
            ("Protein from local PDB", "pdb"),
            ("Bacteria / cells", "bacteria"),
        ):
            self._specimen_kind.addItem(label, value)
        form.addRow("Specimen:", self._specimen_kind)
        self._particles = QSpinBox()
        self._particles.setRange(0, 10000)
        self._particles.setValue(45)
        form.addRow("Objects:", self._particles)
        self._diam_mean = self._spin(28.0, 1.0, 1000, 1.0)
        form.addRow("Size mean nm:", self._diam_mean)
        self._diam_sd = self._spin(7.0, 0.0, 1000, 1.0)
        form.addRow("Size SD nm:", self._diam_sd)
        self._membrane = self._spin(4.0, 0.5, 50, 0.5)
        form.addRow("Membrane thickness nm:", self._membrane)
        pdb_row = QHBoxLayout()
        self._pdb_path = QLineEdit("")
        pdb_browse = QPushButton("Browse")
        pdb_browse.clicked.connect(self._browse_pdb)
        pdb_row.addWidget(self._pdb_path, 1)
        pdb_row.addWidget(pdb_browse)
        form.addRow("Local PDB:", pdb_row)
        self._oligomer = QSpinBox()
        self._oligomer.setRange(1, 64)
        self._oligomer.setValue(1)
        form.addRow("Oligomer copies:", self._oligomer)
        self._ice = self._spin(45.0, 1.0, 10000, 5.0)
        form.addRow("Ice thickness nm:", self._ice)
        self._solvent_noise = self._spin(6.0, 0.0, 100, 0.5)
        form.addRow("Ice potential noise:", self._solvent_noise)
        self._overlap = QCheckBox("Allow particle overlap")
        form.addRow("", self._overlap)
        return box

    def _build_advanced_group(self) -> QGroupBox:
        box = QGroupBox("Advanced engine / 4D-STEM / Reference")
        form = QFormLayout(box)
        self._microscope = QComboBox()
        for m in ["krios", "krios-cfeg", "glacios", "talos-arctica", "talos-l120c", "cs-corrected"]:
            self._microscope.addItem(m, m)
        form.addRow("Microscope:", self._microscope)
        self._species = QComboBox()
        for s in ["e_coli", "b_subtilis", "s_aureus", "caulobacter", "p_aeruginosa",
                  "vibrio_cholerae", "streptococcus", "mycoplasma"]:
            self._species.addItem(s, s)
        form.addRow("Bacterium species:", self._species)
        self._add_np = QCheckBox("Add nanoparticles to scene")
        form.addRow("", self._add_np)
        self._add_contam = QCheckBox("Add crystalline-ice contamination")
        form.addRow("", self._add_contam)
        self._energy_filter = self._spin(0.0, 0.0, 100.0, 5.0)
        form.addRow("Energy filter eV (0=off):", self._energy_filter)
        self._conv_mrad = self._spin(20.0, 0.5, 60.0, 1.0)
        form.addRow("4D-STEM convergence mrad:", self._conv_mrad)
        self._scan_size = QSpinBox()
        self._scan_size.setRange(8, 256)
        self._scan_size.setValue(64)
        form.addRow("4D-STEM scan size:", self._scan_size)
        ref_row = QHBoxLayout()
        self._reference_path = QLineEdit("")
        rb = QPushButton("Browse")
        rb.clicked.connect(self._browse_reference)
        ref_row.addWidget(self._reference_path, 1)
        ref_row.addWidget(rb)
        form.addRow("Reference image:", ref_row)
        self._modality = QComboBox()
        for m in ["cryoem", "fib"]:
            self._modality.addItem(m, m)
        form.addRow("Reference modality:", self._modality)
        self._calibrate = QCheckBox("Calibrate from reference (CTF / curtaining)")
        self._calibrate.setChecked(True)
        form.addRow("", self._calibrate)
        return box

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose reference image", str(Path.home()),
            "Images (*.mrc *.mrcs *.dm4 *.tif *.tiff *.png);;All files (*)")
        if path:
            self._reference_path.setText(path)

    def _emit_action(self, action: str) -> None:
        if not self._output_dir.text().strip():
            QMessageBox.warning(self, "TEM Simulation", "Choose an output folder.")
            return
        p = self.params()
        if action == "generate_4dstem":
            p.update({"conv_mrad": self._conv_mrad.value(), "scan_size": self._scan_size.value(),
                      "detectors": ["BF", "ADF", "HAADF", "CoM_mag", "iDPC"]})
        elif action == "simulate_from_reference":
            p.update({"reference_path": self._reference_path.text().strip(),
                      "modality": self._modality.currentData(),
                      "calibrate": self._calibrate.isChecked()})
        self.action_requested.emit(action, p)

    def _spin(self, value: float, minimum: float, maximum: float, step: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(3)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Choose TEM simulation output folder", self._output_dir.text())
        if path:
            self._output_dir.setText(path)

    def _browse_pdb(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose local PDB file", str(Path.home()), "PDB files (*.pdb *.ent);;All files (*)")
        if path:
            self._pdb_path.setText(path)

    def _browse_custom_script(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose custom TEM Python script", str(Path.home()), "Python files (*.py);;All files (*)")
        if path:
            self._custom_script.setText(path)

    def _on_generate(self) -> None:
        output_dir = self._output_dir.text().strip()
        if not output_dir:
            QMessageBox.warning(self, "TEM Simulation", "Choose an output folder.")
            return
        # One Generate button: the specimen + Path (fast vs full-physics) auto-select
        # the fast projected-potential path or the full multislice engine.
        self.action_requested.emit("generate_tem_simulation", self.params())

    def params(self) -> dict:
        return {
            "output_dir": self._output_dir.text().strip(),
            "count": self._count.value(),
            "open_generated": self._open_generated.isChecked(),
            "save_layers": self._save_layers.isChecked(),
            "simulation_path": self._simulation_path.currentData(),
            "custom_script_path": self._custom_script.text().strip(),
            "slice_thickness_a": self._slice_thickness.value(),
            "preset": self._preset.currentData(),
            "image_size_px": self._size.value(),
            "pixel_size_a": self._pixel_size.value(),
            "voltage_kv": self._voltage.currentData(),
            "cs_mm": self._cs.value(),
            "amplitude_contrast": self._amp.value(),
            "total_dose_e_per_a2": self._dose.value(),
            "detector_model": self._detector.currentData(),
            "defocus_min_um": self._defocus_min.value(),
            "defocus_max_um": self._defocus_max.value(),
            "phase_plate": self._phase_plate.isChecked(),
            "seed": self._seed.value(),
            "specimen_kind": self._specimen_kind.currentData(),
            "n_particles": self._particles.value(),
            "diameter_nm_mean": self._diam_mean.value(),
            "diameter_nm_sd": self._diam_sd.value(),
            "membrane_thickness_nm": self._membrane.value(),
            "pdb_path": self._pdb_path.text().strip(),
            "oligomer_count": self._oligomer.value(),
            "ice_thickness_nm": self._ice.value(),
            "solvent_noise": self._solvent_noise.value(),
            "allow_overlap": self._overlap.isChecked(),
            "microscope": self._microscope.currentData(),
            "species": self._species.currentData(),
            "add_nanoparticles": self._add_np.isChecked(),
            "add_contamination": self._add_contam.isChecked(),
            "energy_filter_ev": self._energy_filter.value(),
        }

    def apply_params(self, params: dict) -> None:
        self._output_dir.setText(str(params.get("output_dir", self._output_dir.text())))
        self._count.setValue(int(params.get("count", self._count.value())))
        self._open_generated.setChecked(bool(params.get("open_generated", self._open_generated.isChecked())))
        self._save_layers.setChecked(bool(params.get("save_layers", self._save_layers.isChecked())))
        self._set_combo(self._simulation_path, params.get("simulation_path", self._simulation_path.currentData()))
        self._custom_script.setText(str(params.get("custom_script_path", self._custom_script.text())))
        self._slice_thickness.setValue(float(params.get("slice_thickness_a", self._slice_thickness.value())))
        self._set_combo(self._preset, params.get("preset", self._preset.currentData()))
        self._size.setValue(int(params.get("image_size_px", self._size.value())))
        self._pixel_size.setValue(float(params.get("pixel_size_a", self._pixel_size.value())))
        self._set_combo(self._voltage, str(params.get("voltage_kv", self._voltage.currentData())))
        self._cs.setValue(float(params.get("cs_mm", self._cs.value())))
        self._amp.setValue(float(params.get("amplitude_contrast", self._amp.value())))
        self._dose.setValue(float(params.get("total_dose_e_per_a2", self._dose.value())))
        self._set_combo(self._detector, params.get("detector_model", self._detector.currentData()))
        self._defocus_min.setValue(float(params.get("defocus_min_um", self._defocus_min.value())))
        self._defocus_max.setValue(float(params.get("defocus_max_um", self._defocus_max.value())))
        self._phase_plate.setChecked(bool(params.get("phase_plate", self._phase_plate.isChecked())))
        self._seed.setValue(int(params.get("seed", self._seed.value())))
        self._set_combo(self._specimen_kind, params.get("specimen_kind", params.get("kind", self._specimen_kind.currentData())))
        self._particles.setValue(int(params.get("n_particles", self._particles.value())))
        self._diam_mean.setValue(float(params.get("diameter_nm_mean", self._diam_mean.value())))
        self._diam_sd.setValue(float(params.get("diameter_nm_sd", self._diam_sd.value())))
        self._membrane.setValue(float(params.get("membrane_thickness_nm", self._membrane.value())))
        self._pdb_path.setText(str(params.get("pdb_path", self._pdb_path.text())))
        self._oligomer.setValue(int(params.get("oligomer_count", self._oligomer.value())))
        self._ice.setValue(float(params.get("ice_thickness_nm", self._ice.value())))
        self._solvent_noise.setValue(float(params.get("solvent_noise", self._solvent_noise.value())))
        self._overlap.setChecked(bool(params.get("allow_overlap", self._overlap.isChecked())))

    def _set_combo(self, combo: QComboBox, value) -> None:
        idx = combo.findData(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)

    def set_running(self, running: bool) -> None:
        for b in (self._generate_btn, self._fourd_btn, self._ref_btn):
            b.setEnabled(not running)
        self._generate_btn.setText("Generating..." if running else "Generate TEM Simulation")

    def set_status(self, text: str) -> None:
        self._status.setText(text)
