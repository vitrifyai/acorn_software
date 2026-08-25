"""Control panel for physics-based SEM simulation.

Laid out so the two decisions that actually govern image quality -- beam energy
and field of view -- sit next to the gauge that shows their consequence, rather
than being buried among parameters that matter far less.
"""
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
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .gauge import InteractionVolumeGauge
from .materials import MATERIALS
from .scenes import SCENE_LABELS

# Plain-language starting points. Each is a claim about what the settings are
# *for*, not just a bundle of numbers -- the tooltip says which contrast
# mechanism it isolates, because that is what makes a preset worth having.
PRESETS: dict[str, dict] = {
    "Nanoparticles on carbon": dict(
        scene="nanoparticles", E0_kev=5.0, pixel_size_nm=2.0, image_size_px=512,
        detector="ETD", particle="gold", substrate="carbon", electrons_per_px=500,
        _hint="Heavy particles on a light support. Strong Z contrast plus relief."),
    "Alloy phases (Z contrast only)": dict(
        scene="grains", E0_kev=10.0, pixel_size_nm=8.0, image_size_px=512,
        detector="BSE", phase_a="iron", phase_b="copper", electrons_per_px=800,
        _hint="Polished flat, so all contrast is compositional. BSE separates "
              "the phases where SE barely does."),
    "Cells in resin (topography only)": dict(
        scene="biological", E0_kev=2.0, pixel_size_nm=8.0, image_size_px=512,
        detector="ETD", electrons_per_px=300,
        _hint="Almost no atomic-number difference. The hard, honest case for "
              "life-science SEM -- all the signal is surface relief."),
    "Porous ceramic": dict(
        scene="porous", E0_kev=5.0, pixel_size_nm=5.0, image_size_px=512,
        detector="ETD", matrix="alumina", electrons_per_px=600,
        _hint="Pores are vacuum and emit nothing, so they read as truly black."),
    "FIB cross-section": dict(
        scene="cross_section", E0_kev=5.0, pixel_size_nm=8.0, image_size_px=512,
        detector="BSE", electrons_per_px=600,
        _hint="Pt cap over stacked layers, each with its own interaction volume."),
    "Low dose / noisy": dict(
        scene="nanoparticles", E0_kev=3.0, pixel_size_nm=4.0, image_size_px=512,
        detector="ETD", electrons_per_px=30, read_noise_e=6.0,
        _hint="Beam-sensitive conditions. Contrast is unchanged; only the noise "
              "grows, exactly as dose works in the cryo-TEM engine."),
}

_DETECTORS = [
    ("Everhart-Thornley (SE + some BSE)", "ETD"),
    ("Through-lens / in-lens (sharp SE)", "TLD"),
    ("Annular backscatter (Z contrast)", "BSE"),
]

_MATERIAL_NAMES = [m for m in sorted(MATERIALS) if m != "vacuum"]


class SemSimPanel(QWidget):
    generate_requested = pyqtSignal(dict)
    action_requested = pyqtSignal(str, dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._build_ui()
        self._on_preset_changed(self._preset.currentText())
        self._refresh_gauge()

    # -- construction --------------------------------------------------------
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        layout.addWidget(self._build_preset_row())
        layout.addWidget(self._build_specimen_group())
        layout.addWidget(self._build_beam_group())

        self._gauge = InteractionVolumeGauge()
        layout.addWidget(self._gauge)
        self._advice = QLabel("")
        self._advice.setWordWrap(True)
        self._advice.setStyleSheet("font-size: 11px; color: #d4a24c;")
        layout.addWidget(self._advice)

        layout.addWidget(self._build_detector_group())
        layout.addWidget(self._build_output_group())

        self._generate_btn = QPushButton("Generate SEM Simulation")
        self._generate_btn.setToolTip(
            "Trace electrons, build interaction-volume kernels, and render.\n"
            "The first run for a new material and energy spends a few seconds "
            "on Monte Carlo; after that the kernels are cached.")
        self._generate_btn.clicked.connect(self._on_generate)
        _style_primary(self._generate_btn)
        layout.addWidget(self._generate_btn)

        self._status = QLabel("Ready.")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("font-size: 11px; color: #6c7086;")
        layout.addWidget(self._status)
        layout.addStretch()

        _install_collapsible(self)
        _guard_wheel(self)

    def _build_preset_row(self) -> QGroupBox:
        box = QGroupBox("Start from")
        form = QFormLayout(box)
        self._preset = QComboBox()
        self._preset.addItems(PRESETS.keys())
        self._preset.currentTextChanged.connect(self._on_preset_changed)
        form.addRow("Preset:", self._preset)
        self._preset_hint = QLabel("")
        self._preset_hint.setWordWrap(True)
        self._preset_hint.setStyleSheet("font-size: 11px; color: #6c7086;")
        form.addRow(self._preset_hint)
        return box

    def _build_specimen_group(self) -> QGroupBox:
        box = QGroupBox("Specimen")
        form = QFormLayout(box)

        self._scene = QComboBox()
        for key, label in SCENE_LABELS.items():
            self._scene.addItem(label, key)
        self._scene.currentIndexChanged.connect(self._on_scene_changed)
        form.addRow("Scene:", self._scene)

        self._particle = QComboBox()
        self._particle.addItems(_MATERIAL_NAMES)
        self._particle.setCurrentText("gold")
        self._particle.currentTextChanged.connect(self._refresh_gauge)
        form.addRow("Particle / phase A:", self._particle)

        self._substrate = QComboBox()
        self._substrate.addItems(_MATERIAL_NAMES)
        self._substrate.setCurrentText("carbon")
        self._substrate.currentTextChanged.connect(self._refresh_gauge)
        self._substrate.setToolTip(
            "The support or second phase. For particles on a support the gauge "
            "follows the particle, not this -- a feature's edge is limited by "
            "its own interaction volume, and the support only contributes a "
            "smooth background.")
        form.addRow("Substrate / phase B:", self._substrate)

        self._n_particles = _spin_int(1, 5000, 40)
        form.addRow("Particles:", self._n_particles)
        self._diameter = _spin(1.0, 5000.0, 40.0, 1.0)
        form.addRow("Diameter nm:", self._diameter)
        self._diameter_sd = _spin(0.0, 1000.0, 12.0, 1.0)
        form.addRow("Diameter spread nm:", self._diameter_sd)

        self._relief = QCheckBox("Particles stand proud of the surface")
        self._relief.setChecked(True)
        self._relief.setToolTip(
            "Uncheck to flatten them. That removes topographic contrast and "
            "leaves pure atomic-number contrast -- the control condition for "
            "asking what a model is really keying on.")
        form.addRow(self._relief)
        return box

    def _build_beam_group(self) -> QGroupBox:
        box = QGroupBox("Beam")
        form = QFormLayout(box)

        self._kv = _spin(0.2, 30.0, 5.0, 0.5)
        self._kv.setToolTip(
            "Beam energy. The dominant control on the interaction volume: "
            "range grows roughly as E^1.67, so 20 kV probes ~10x deeper than "
            "5 kV. Low kV is how SEM gets surface sensitivity.")
        self._kv.valueChanged.connect(self._refresh_gauge)
        form.addRow("Energy kV:", self._kv)

        self._pixel = _spin(0.1, 500.0, 4.0, 0.5)
        self._pixel.valueChanged.connect(self._refresh_gauge)
        form.addRow("Pixel size nm:", self._pixel)

        self._size = _spin_int(64, 4096, 512)
        self._size.valueChanged.connect(self._refresh_gauge)
        form.addRow("Image size px:", self._size)

        self._electrons = _spin(1.0, 1e6, 500.0, 50.0)
        self._electrons.setToolTip(
            "Electrons landing per pixel (probe current x dwell time). Affects "
            "noise only, never contrast.")
        form.addRow("Electrons / px:", self._electrons)

        self._probe = _spin(0.0, 100.0, 1.0, 0.1)
        self._probe.valueChanged.connect(self._refresh_gauge)
        form.addRow("Probe size nm:", self._probe)
        return box

    def _build_detector_group(self) -> QGroupBox:
        box = QGroupBox("Detector")
        form = QFormLayout(box)

        self._detector = QComboBox()
        for label, key in _DETECTORS:
            self._detector.addItem(label, key)
        form.addRow("Type:", self._detector)

        self._bse_mix = _spin(0.0, 1.0, 0.15, 0.05)
        self._bse_mix.setToolTip("Backscatter fraction reaching an SE detector.")
        form.addRow("BSE admixture:", self._bse_mix)

        self._elevation = _spin(0.0, 90.0, 25.0, 5.0)
        form.addRow("Elevation deg:", self._elevation)
        self._azimuth = _spin(0.0, 360.0, 0.0, 15.0)
        form.addRow("Azimuth deg:", self._azimuth)
        self._asymmetry = _spin(0.0, 1.0, 0.30, 0.05)
        self._asymmetry.setToolTip(
            "Directional shading: how much brighter a surface facing the "
            "detector looks. Set 0 for flat, shadowless illumination.")
        form.addRow("Shading:", self._asymmetry)

        self._read_noise = _spin(0.0, 100.0, 3.0, 0.5)
        form.addRow("Read noise e-:", self._read_noise)
        self._jitter = _spin(0.0, 10.0, 0.0, 0.1)
        self._jitter.setToolTip("Scan-line jitter in pixels; a stage-drift stand-in.")
        form.addRow("Scan jitter px:", self._jitter)
        return box

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output")
        form = QFormLayout(box)

        row = QHBoxLayout()
        self._output_dir = QLineEdit(str(Path.home() / "sem_sim_output"))
        _attach_path(self._output_dir)
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse_output)
        row.addWidget(self._output_dir, 1)
        row.addWidget(browse)
        form.addRow("Folder:", row)

        self._count = _spin_int(1, 100000, 10)
        form.addRow("Images:", self._count)

        self._quality = QComboBox()
        for label, n in (("Fast (15k electrons)", 15000),
                         ("Balanced (40k)", 40000),
                         ("Careful (120k)", 120000)):
            self._quality.addItem(label, n)
        self._quality.setCurrentIndex(1)
        self._quality.setToolTip(
            "Monte Carlo trajectories per kernel. Affects only how smooth the "
            "kernels are, not the physics -- and it is paid once per material "
            "and energy, then cached.")
        form.addRow("Kernel quality:", self._quality)

        self._save_layers = QCheckBox("Save truth mask, SE and BSE channels")
        self._save_layers.setChecked(True)
        form.addRow(self._save_layers)

        self._open_generated = QCheckBox("Open generated images in Acorn")
        self._open_generated.setChecked(True)
        form.addRow(self._open_generated)
        return box

    # -- behaviour -----------------------------------------------------------
    def _on_preset_changed(self, name: str) -> None:
        preset = PRESETS.get(name)
        if not preset:
            return
        self._preset_hint.setText(preset.get("_hint", ""))
        self.apply_params({k: v for k, v in preset.items() if not k.startswith("_")})

    def _on_scene_changed(self) -> None:
        kind = self._scene.currentData()
        particles = kind == "nanoparticles"
        for w in (self._n_particles, self._diameter, self._diameter_sd, self._relief):
            w.setEnabled(particles)
        self._refresh_gauge()

    def _gauge_material(self) -> str:
        """The material of the feature being annotated, not the worst present.

        This distinction matters and getting it wrong gives actively bad advice.
        Gauging the largest interaction volume in the scene flags gold particles
        on carbon at 5 kV as marginal -- carbon's volume is ~445 nm -- when that
        is a completely routine experiment that produces sharp particles,
        because a particle's edge is set by ITS OWN interaction volume (gold at
        5 kV: ~85 nm), not by the support's. The support contributes a smooth
        background, not blur of the feature.

        So gauge whatever carries the ground-truth label. Where two labelled
        phases compete, take the larger, since that one limits the pair.
        """
        from .materials import get, kanaya_okayama_nm

        kind = self._scene.currentData()
        if kind == "nanoparticles":
            candidates = [self._particle.currentText()]
        elif kind == "grains":
            candidates = [self._particle.currentText(), self._substrate.currentText()]
        elif kind == "porous":
            candidates = [self._substrate.currentText()]     # pores are vacuum
        elif kind == "biological":
            candidates = ["biology"]
        else:
            # The cross-section scene builds its own layer stack, so gauge those
            # materials rather than the substrate combo, which it ignores.
            candidates = ["carbon", "silica", "silicon", "platinum"]

        kv = self._kv.value()
        best, best_r = candidates[0], -1.0
        for name in candidates:
            try:
                r = kanaya_okayama_nm(get(name), kv)
            except (KeyError, ZeroDivisionError):
                continue
            if r > best_r:
                best, best_r = name, r
        return best

    def _refresh_gauge(self) -> None:
        self._gauge.update_scales(
            material=self._gauge_material(),
            kv=self._kv.value(),
            pixel_nm=self._pixel.value(),
            field_px=self._size.value(),
            probe_nm=max(self._probe.value(), 0.1),
        )
        self._advice.setText(self._gauge.advice())

    def _browse_output(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self, "Choose output folder", self._output_dir.text())
        if chosen:
            self._output_dir.setText(chosen)

    def _on_generate(self) -> None:
        self.generate_requested.emit(self.params())

    # -- state ---------------------------------------------------------------
    def params(self) -> dict:
        return {
            "output_dir": self._output_dir.text().strip(),
            "count": self._count.value(),
            "open_generated": self._open_generated.isChecked(),
            "save_layers": self._save_layers.isChecked(),
            "n_electrons": self._quality.currentData(),
            "scene": self._scene.currentData(),
            "particle": self._particle.currentText(),
            "substrate": self._substrate.currentText(),
            "phase_a": self._particle.currentText(),
            "phase_b": self._substrate.currentText(),
            "matrix": self._substrate.currentText(),
            "n_particles": self._n_particles.value(),
            "diameter_nm_mean": self._diameter.value(),
            "diameter_nm_sd": self._diameter_sd.value(),
            "relief": self._relief.isChecked(),
            "E0_kev": self._kv.value(),
            "pixel_size_nm": self._pixel.value(),
            "image_size_px": self._size.value(),
            "electrons_per_px": self._electrons.value(),
            "probe_nm": self._probe.value(),
            "detector": self._detector.currentData(),
            "bse_mix": self._bse_mix.value(),
            "elevation_deg": self._elevation.value(),
            "azimuth_deg": self._azimuth.value(),
            "asymmetry": self._asymmetry.value(),
            "read_noise_e": self._read_noise.value(),
            "scan_jitter_px": self._jitter.value(),
            "seed": 0,
        }

    def apply_params(self, params: dict) -> None:
        """Push CLU- or preset-supplied values into the widgets.

        Blocks signals while setting so the gauge refreshes once at the end
        rather than flickering through partially-applied states.
        """
        widgets = {
            "count": self._count, "n_particles": self._n_particles,
            "diameter_nm_mean": self._diameter, "diameter_nm_sd": self._diameter_sd,
            "E0_kev": self._kv, "pixel_size_nm": self._pixel,
            "image_size_px": self._size, "electrons_per_px": self._electrons,
            "probe_nm": self._probe, "bse_mix": self._bse_mix,
            "elevation_deg": self._elevation, "azimuth_deg": self._azimuth,
            "asymmetry": self._asymmetry, "read_noise_e": self._read_noise,
            "scan_jitter_px": self._jitter,
        }
        blocked = list(widgets.values()) + [self._scene, self._particle,
                                            self._substrate, self._detector]
        for w in blocked:
            w.blockSignals(True)
        try:
            for key, widget in widgets.items():
                if key in params and params[key] is not None:
                    try:
                        widget.setValue(type(widget.value())(params[key]))
                    except (TypeError, ValueError):
                        pass
            if params.get("output_dir"):
                self._output_dir.setText(str(params["output_dir"]))
            if "scene" in params:
                i = self._scene.findData(str(params["scene"]))
                if i >= 0:
                    self._scene.setCurrentIndex(i)
            if "detector" in params:
                i = self._detector.findData(str(params["detector"]).upper())
                if i >= 0:
                    self._detector.setCurrentIndex(i)
            for key, combo in (("particle", self._particle),
                               ("phase_a", self._particle),
                               ("substrate", self._substrate),
                               ("phase_b", self._substrate),
                               ("matrix", self._substrate)):
                if params.get(key) and combo.findText(str(params[key])) >= 0:
                    combo.setCurrentText(str(params[key]))
            for key, box in (("relief", self._relief),
                             ("save_layers", self._save_layers),
                             ("open_generated", self._open_generated)):
                if key in params:
                    box.setChecked(bool(params[key]))
        finally:
            for w in blocked:
                w.blockSignals(False)
        self._on_scene_changed()

    def set_status(self, text: str) -> None:
        self._status.setText(text)

    def set_running(self, running: bool) -> None:
        self._generate_btn.setEnabled(not running)
        self._generate_btn.setText(
            "Simulating..." if running else "Generate SEM Simulation")


# ---------------------------------------------------------------------------
# Small helpers, and optional ACORN chrome applied only if it is available.
# ---------------------------------------------------------------------------

def _spin(lo, hi, value, step) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(2 if step < 1 else 1)
    s.setValue(value)
    return s


def _spin_int(lo, hi, value) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(value)
    return s


def _style_primary(button) -> None:
    try:
        from acorn.gui.buttons import primary
        primary(button)
    except Exception:
        pass


def _attach_path(field) -> None:
    try:
        from acorn.gui.path_field import attach
        attach(field)
    except Exception:
        pass


def _install_collapsible(panel) -> None:
    try:
        from acorn.gui.collapsible import apply_to_panel
        apply_to_panel(panel, folded_titles={"Detector"}, panel_key="acorn_sem_sim")
    except Exception:
        pass


def _guard_wheel(panel) -> None:
    try:
        from acorn.gui.wheel_guard import install
        install(panel)
    except Exception:
        pass
