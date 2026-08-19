"""ACORN plugin for focused cryo-TEM simulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin
from acorn_tem_sim.io import generate_tem_dataset
from acorn_tem_sim.simulator import Specimen, resolve
from acorn_tem_sim.engine_io import generate_tem_advanced, generate_4dstem

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class _TemSimThread(QThread):
    finished_ok = pyqtSignal(int, str, list)
    failed = pyqtSignal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            cfg, specimen = _config_from_params(self.params)
            output_dir = _fresh_run_dir(Path(self.params["output_dir"]), "tem_run")
            paths = generate_tem_dataset(
                output_dir,
                int(self.params["count"]),
                cfg,
                specimen,
                save_layers=bool(self.params.get("save_layers", True)),
                seed=int(self.params.get("seed", 1)),
            )
            self.finished_ok.emit(len(paths), str(output_dir), [str(p) for p in paths])
        except Exception as exc:
            self.failed.emit(str(exc))


class _EngineThread(QThread):
    """Runs an arbitrary engine generation callable(params)->list[str] off-thread."""
    finished_ok = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, params: dict, work) -> None:
        super().__init__()
        self.params = params
        self._work = work

    def run(self) -> None:
        try:
            self.finished_ok.emit(self._work(self.params))
        except Exception as exc:  # surface to the UI
            self.failed.emit(str(exc))


class TemSimulationPlugin(AcornPlugin):
    TAB_LABEL = "TEM Simulation"
    PLUGIN_ID = "acorn_tem_sim"
    sort_order = 37
    FLOATING = True
    FLOATING_TITLE = "TEM Simulation"
    FLOATING_SHORTCUT = "Ctrl+Shift+M"
    FLOATING_MIN_WIDTH = 380

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._thread: _TemSimThread | None = None
        context.action_requested.connect(self._on_action_requested)

    def create_panel(self) -> QWidget:
        from acorn_tem_sim.panel import TemSimPanel

        self._panel = TemSimPanel()
        self._panel.generate_requested.connect(self._on_generate_requested)
        self._panel.action_requested.connect(self._on_action_requested)   # advanced/4D-STEM/reference buttons
        return self._panel

    def _on_action_requested(self, action: str, params: dict) -> None:
        if action in ("generate_tem_simulation", "generate_tem_advanced"):
            self._run_tem(dict(params))          # single entry; specimen decides the path
        elif action == "generate_4dstem":
            self._run_engine_action("generate_4dstem", dict(params))
        elif action == "simulate_from_reference":
            self._run_reference_action(dict(params))

    def _run_tem(self, params: dict) -> None:
        """Route to the advanced engine (plga/bacteria/contamination/scenes/advanced
        physics) or the legacy fast path (lipid/protein/PDB). Whichever TEM tool CLU
        picked, the specimen + params decide — so a wrong tool choice can't misroute."""
        if _use_engine(params):
            self._run_engine_action("generate_tem_advanced", params)
        else:
            request = _params_from_clu(params)
            if self._panel is not None:
                self._panel.apply_params(request)
            self._on_generate_requested(request)

    def _run_reference_action(self, params: dict) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(None, "TEM Simulation", "Generation is already running.")
            return
        ref = params.get("reference_path") or params.get("image_path") or self._current_image_path()
        if not ref:
            QMessageBox.warning(None, "Reference simulation",
                                "No reference image given (reference_path) or open in the viewer.")
            return
        self._context.set_status("Simulating from reference image...")
        if self._panel is not None:
            self._panel.set_running(True)

        def work(p):
            from acorn_tem_sim.reference import simulate_from_reference
            out = _fresh_run_dir(Path(p.get("output_dir") or (Path.home() / "tem_sim_output")), "ref_run")
            simulate_from_reference(ref, str(p.get("modality", "cryoem")), params=p,
                                    n=int(p.get("count", p.get("n", 8))),
                                    calibrate=_as_bool(p.get("calibrate", True)), out_dir=str(out))
            return [str(x) for x in sorted(Path(out).glob("sim_*.png"))]

        self._thread = _EngineThread(params, work)
        self._thread.finished_ok.connect(lambda paths: self._on_engine_finished("reference sim", paths))
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(
            lambda: self._panel.set_running(False) if self._panel is not None else None)
        self._thread.start()

    def _current_image_path(self):
        """Best-effort path of the image currently open in ACORN (reference fallback)."""
        try:
            getter = getattr(self._context, "_w", None)
            win = getter() if getter else None
            for attr in ("current_path", "current_image_path", "image_path"):
                p = getattr(win, attr, None)
                if p:
                    return str(p)
        except Exception:
            pass
        return None

    def _run_engine_action(self, action: str, params: dict) -> None:
        """CLU-driven engine capabilities (advanced multislice TEM, 4D-STEM)."""
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(None, "TEM Simulation", "Generation is already running.")
            return
        label = "advanced TEM" if action == "generate_tem_advanced" else "4D-STEM"
        self._context.set_status(f"Generating {label} simulation (engine)...")
        if self._panel is not None:
            self._panel.set_running(True)

        def work(p):
            out = _fresh_run_dir(Path(p.get("output_dir") or (Path.home() / "tem_sim_output")),
                                 "fourd_run" if action == "generate_4dstem" else "tem_adv_run")
            if action == "generate_4dstem":
                return [str(x) for x in generate_4dstem(out, p, seed=int(p.get("seed", 1)))]
            return [str(x) for x in generate_tem_advanced(
                out, int(p.get("count", 5)), p, seed=int(p.get("seed", 1)),
                save_layers=_as_bool(p.get("save_layers", True)))]

        self._thread = _EngineThread(params, work)
        self._thread.finished_ok.connect(lambda paths: self._on_engine_finished(label, paths))
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(
            lambda: self._panel.set_running(False) if self._panel is not None else None)
        self._thread.start()

    def _on_engine_finished(self, label: str, image_paths: list) -> None:
        self._context.set_status(f"Generated {label}: {len(image_paths)} image(s)", timeout_ms=8000)
        if image_paths:
            _open_paths_in_acorn(self._context, image_paths)

    def _on_generate_requested(self, params: dict) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(None, "TEM Simulation", "Generation is already running.")
            return
        if self._panel is not None:
            self._panel.set_running(True)
            self._panel.set_status("Generating TEM simulation...")
        self._context.set_status("Generating TEM simulation...")

        self._thread = _TemSimThread(params)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(lambda: self._panel.set_running(False) if self._panel is not None else None)
        self._thread.start()

    def _on_finished(self, count: int, output_dir: str, image_paths: list) -> None:
        message = f"Generated {count} TEM simulation image(s): {output_dir}"
        if self._panel is not None:
            self._panel.set_status(message)
        self._context.set_status(message, timeout_ms=8000)
        params = self._thread.params if self._thread is not None else {}
        if params.get("open_generated", True) and image_paths:
            opened, detail = _open_paths_in_acorn(self._context, image_paths)
            if self._panel is not None:
                self._panel.set_status(f"Opening {opened} generated image(s): {output_dir}" if opened else detail)
            self._context.set_status(
                f"Opening {opened} generated TEM image(s)..." if opened else detail,
                timeout_ms=8000,
            )
        elif self._panel is not None:
            self._panel.set_status(f"Generated {count} image(s); auto-open is off: {output_dir}")

    def _on_failed(self, message: str) -> None:
        if self._panel is not None:
            self._panel.set_status(f"Error: {message}")
        self._context.set_status(f"TEM simulation failed: {message}", timeout_ms=8000)
        QMessageBox.critical(None, "TEM Simulation", message)

    def teardown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.wait(1500)


def _config_from_params(params: dict) -> tuple[dict, Specimen]:
    cfg = resolve(
        answers={
            "simulation_path": str(params.get("simulation_path", "fast")),
            "custom_script_path": str(params.get("custom_script_path", "")),
            "slice_thickness_a": float(params.get("slice_thickness_a", 5.0)),
            "pixel_size_a": float(params.get("pixel_size_a", 1.5)),
            "voltage_kv": str(params.get("voltage_kv", "300")),
            "cs_mm": float(params.get("cs_mm", 2.7)),
            "amplitude_contrast": float(params.get("amplitude_contrast", 0.10)),
            "total_dose_e_per_a2": float(params.get("total_dose_e_per_a2", 40.0)),
            "detector_model": str(params.get("detector_model", "K3")),
            "image_size_px": int(params.get("image_size_px", 512)),
            "defocus_min_um": float(params.get("defocus_min_um", -1.0)),
            "defocus_max_um": float(params.get("defocus_max_um", -2.5)),
            "phase_plate": bool(params.get("phase_plate", False)),
        },
        preset=str(params.get("preset", "krios-k3")),
    )
    specimen = Specimen(
        kind=str(params.get("specimen_kind", params.get("kind", "plga"))),
        ice_thickness_nm=float(params.get("ice_thickness_nm", 45.0)),
        n_particles=int(params.get("n_particles", 45)),
        diameter_nm_mean=float(params.get("diameter_nm_mean", 28.0)),
        diameter_nm_sd=float(params.get("diameter_nm_sd", 7.0)),
        membrane_thickness_nm=float(params.get("membrane_thickness_nm", 4.0)),
        oligomer_count=int(params.get("oligomer_count", 1)),
        pdb_path=str(params.get("pdb_path", "")),
        solvent_noise=float(params.get("solvent_noise", 6.0)),
        allow_overlap=bool(params.get("allow_overlap", False)),
        seed=int(params.get("seed", 1)),
    )
    return cfg, specimen


def _params_from_clu(params: dict) -> dict:
    preset = str(params.get("preset") or "krios-k3")
    low_dose = "low" in preset.lower() or bool(params.get("low_dose", False))
    return {
        "output_dir": params.get("output_dir") or str(Path.home() / "tem_sim_output"),
        "count": int(params.get("count", params.get("images", 10))),
        "open_generated": _as_bool(params.get("open_generated", True)),
        "save_layers": _as_bool(params.get("save_layers", True)),
        "simulation_path": _simulation_path_from_text(params.get("simulation_path", params.get("backend", params.get("path", "fast")))),
        "custom_script_path": str(params.get("custom_script_path", params.get("script_path", ""))),
        "slice_thickness_a": float(params.get("slice_thickness_a", 5.0)),
        "preset": preset if preset in {"krios-k3", "glacios-falcon4", "talos-ceta", "cs-corrected"} else "krios-k3",
        "image_size_px": int(params.get("image_size_px", params.get("size", 512))),
        "pixel_size_a": float(params.get("pixel_size_a", 1.5)),
        "voltage_kv": str(params.get("voltage_kv", "300")),
        "cs_mm": float(params.get("cs_mm", 2.7)),
        "amplitude_contrast": float(params.get("amplitude_contrast", 0.10)),
        "total_dose_e_per_a2": float(params.get("total_dose_e_per_a2", 8.0 if low_dose else 40.0)),
        "detector_model": str(params.get("detector_model", "K3")),
        "defocus_min_um": float(params.get("defocus_min_um", -1.0)),
        "defocus_max_um": float(params.get("defocus_max_um", -2.5)),
        "phase_plate": _as_bool(params.get("phase_plate", False)),
        "seed": int(params.get("seed", 1)),
        "specimen_kind": _specimen_kind_from_text(params.get("specimen_kind", params.get("kind", params.get("sample", "plga")))),
        "n_particles": int(params.get("n_particles", params.get("particles", 45))),
        "diameter_nm_mean": float(params.get("diameter_nm_mean", params.get("diameter_nm", 28.0))),
        "diameter_nm_sd": float(params.get("diameter_nm_sd", 7.0)),
        "membrane_thickness_nm": float(params.get("membrane_thickness_nm", 4.0)),
        "pdb_path": str(params.get("pdb_path", "")),
        "oligomer_count": int(params.get("oligomer_count", 1)),
        "ice_thickness_nm": float(params.get("ice_thickness_nm", 45.0)),
        "solvent_noise": float(params.get("solvent_noise", 6.0)),
        "allow_overlap": _as_bool(params.get("allow_overlap", False)),
    }


def _use_engine(params: dict) -> bool:
    """Decide advanced engine vs legacy fast path from the specimen + params.

    Engine: plga / bacteria / contamination / composed scenes + any advanced
    request (detector/microscope/energy-filter/multislice). Legacy: its unique
    kinds (lipid vesicles, protein, PDB) which the engine does not model, and an
    explicit 'fast' PLGA preview."""
    kind = str(params.get("specimen_kind", params.get("kind", "plga"))).strip().lower()
    if kind in ("lipid_single", "lipid_multi", "lipid", "vesicle", "liposome", "protein", "pdb"):
        return False
    if kind in ("bacteria", "cell", "bacterium", "contamination"):
        return True
    if any(params.get(k) for k in ("add_contamination", "add_nanoparticles", "microscope", "species")):
        return True
    if str(params.get("detector_model", "K3")) not in ("K3", "Falcon4", "K2", "Ceta"):
        return True
    try:
        if float(params.get("energy_filter_ev", 0) or 0) > 0:
            return True
    except (TypeError, ValueError):
        pass
    sp = str(params.get("simulation_path", params.get("backend", ""))).strip().lower()
    if sp in ("multislice", "advanced", "engine"):
        return True
    if sp == "fast":
        return False                       # explicit quick PLGA preview -> legacy
    return kind in ("plga", "nanoparticles", "")   # default PLGA -> engine (better physics, GPU)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def _specimen_kind_from_text(value) -> str:
    text = str(value or "plga").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"solid", "solid_particles", "nanoparticle", "nanoparticles", "plga"}:
        return "plga"
    if text in {"single_lipid", "single_lipid_vesicle", "lipid_single", "vesicle", "liposome"}:
        return "lipid_single"
    if text in {"multi_lipid", "multilamellar", "multilamellar_vesicle", "lipid_multi", "mlv"}:
        return "lipid_multi"
    if text in {"protein", "proteins"}:
        return "protein"
    if text in {"pdb", "protein_from_pdb", "local_pdb"}:
        return "pdb"
    if text in {"bacteria", "bacterium", "cell", "cells"}:
        return "bacteria"
    return "plga"


def _simulation_path_from_text(value) -> str:
    text = str(value or "fast").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"physics", "multislice", "advanced", "electron_scattering", "scattering"}:
        return "multislice"
    if text in {"custom", "script", "python", "custom_python"}:
        return "custom"
    return "fast"


def _fresh_run_dir(base_dir: Path, prefix: str) -> Path:
    base_dir = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"{prefix}_{stamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"{prefix}_{stamp}_{counter}"
        counter += 1
    return run_dir


def _open_paths_in_acorn(context: "AcornContext", paths: list[str]) -> tuple[int, str]:
    real_paths = [Path(p) for p in paths if p and Path(p).exists()]
    if not real_paths:
        return 0, "Generated images, but no output TIFF files were found to open."
    window_getter = getattr(context, "_w", None)
    if window_getter is None:
        return 0, "Generated images, but Acorn window is unavailable for auto-open."
    window = window_getter()
    if window is None or not hasattr(window, "open_files"):
        return 0, "Generated images, but Acorn viewer is unavailable for auto-open."
    QTimer.singleShot(0, lambda paths=real_paths: window.open_files(paths))
    return len(real_paths), f"Opening {len(real_paths)} generated image(s)."
