"""ACORN plugin for focused FIB-SEM simulation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from acorn_sim_common import as_bool, fresh_run_dir, open_paths_in_acorn
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin
from acorn_fib_sim.io import generate_fib_dataset
from acorn_fib_sim.simulator import ImagingConfig, MillingConfig, SceneConfig, SimConfig

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class _FibSimThread(QThread):
    finished_ok = pyqtSignal(int, str, list)
    failed = pyqtSignal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            config = _config_from_params(self.params)
            output_dir = fresh_run_dir(Path(self.params["output_dir"]), "fib_run")
            paths = generate_fib_dataset(
                output_dir,
                int(self.params["count"]),
                config,
                save_layers=bool(self.params.get("save_layers", True)),
            )
            self.finished_ok.emit(len(paths), str(output_dir), [str(p) for p in paths])
        except Exception as exc:
            self.failed.emit(str(exc))


class FibSimulationPlugin(AcornPlugin):
    TAB_LABEL = "FIB Simulation"
    PLUGIN_ID = "acorn_fib_sim"
    sort_order = 36
    FLOATING = True
    FLOATING_TITLE = "FIB Simulation"
    FLOATING_SHORTCUT = "Ctrl+Shift+F"
    FLOATING_MIN_WIDTH = 380

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._thread: _FibSimThread | None = None
        context.action_requested.connect(self._on_action_requested)

    def create_panel(self) -> QWidget:
        from acorn_fib_sim.panel import FibSimPanel

        self._panel = FibSimPanel()
        self._panel.generate_requested.connect(self._on_generate_requested)
        return self._panel

    def _on_action_requested(self, action: str, params: dict) -> None:
        if action != "generate_fib_simulation":
            return
        request = _params_from_clu(params)
        if self._panel is not None:
            self._panel.apply_params(request)
        self._on_generate_requested(request)

    def _on_generate_requested(self, params: dict) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(None, "FIB Simulation", "Generation is already running.")
            return
        if self._panel is not None:
            self._panel.set_running(True)
            self._panel.set_status("Generating FIB simulation...")
        self._context.set_status("Generating FIB simulation...")

        self._thread = _FibSimThread(params)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(lambda: self._panel.set_running(False) if self._panel is not None else None)
        self._thread.start()

    def _on_finished(self, count: int, output_dir: str, image_paths: list) -> None:
        message = f"Generated {count} FIB simulation image(s): {output_dir}"
        if self._panel is not None:
            self._panel.set_status(message)
        self._context.set_status(message, timeout_ms=8000)
        params = self._thread.params if self._thread is not None else {}
        if params.get("open_generated", True) and image_paths:
            opened, detail = open_paths_in_acorn(self._context, image_paths)
            if self._panel is not None:
                self._panel.set_status(f"Opening {opened} generated image(s): {output_dir}" if opened else detail)
            self._context.set_status(
                f"Opening {opened} generated FIB image(s)..." if opened else detail,
                timeout_ms=8000,
            )
        elif self._panel is not None:
            self._panel.set_status(f"Generated {count} image(s); auto-open is off: {output_dir}")

    def _on_failed(self, message: str) -> None:
        if self._panel is not None:
            self._panel.set_status(f"Error: {message}")
        self._context.set_status(f"FIB simulation failed: {message}", timeout_ms=8000)
        QMessageBox.critical(None, "FIB Simulation", message)

    def teardown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.wait(1500)


def _config_from_params(params: dict) -> SimConfig:
    return SimConfig(
        sample=str(params.get("sample", "bio")),
        imaging=ImagingConfig(
            shape=(int(params.get("height", 512)), int(params.get("width", 640))),
            pixel_size_nm=float(params.get("pixel_size_nm", 5.0)),
            electrons_per_pixel=float(params.get("electrons_per_pixel", 400.0)),
            mtf_sigma_px=float(params.get("mtf_sigma_px", 0.7)),
            read_noise_e=float(params.get("read_noise_e", 3.0)),
            scan_line_jitter=float(params.get("scan_line_jitter", 0.4)),
            seed=int(params.get("seed", 1)),
        ),
        milling=MillingConfig(
            mill_axis=str(params.get("mill_axis", "y")),
            curtain_strength=float(params.get("curtain_strength", 0.25)),
            curtain_edge_gain=float(params.get("curtain_edge_gain", 2.0)),
            mill_gradient=float(params.get("mill_gradient", 0.15)),
            redeposition=float(params.get("redeposition", 0.06)),
            charging=float(params.get("charging", 0.0)),
            lamella_thickness_nm=float(params.get("lamella_thickness_nm", 200.0)),
        ),
        scene=SceneConfig(
            liftout=bool(params.get("liftout", False)),
            trench=bool(params.get("trench", True)),
            pt_cap=bool(params.get("pt_cap", True)),
            needle=bool(params.get("needle", False)),
        ),
    )


def _params_from_clu(params: dict) -> dict:
    sample = _sample_from_text(params.get("sample") or params.get("fib_sample") or params.get("phantom") or "bio")
    preset = str(params.get("preset") or "").lower()
    low_dose = "low" in preset or bool(params.get("low_dose", False))
    liftout = "lift" in preset or bool(params.get("liftout", False))
    return {
        "output_dir": params.get("output_dir") or str(Path.home() / "fib_sim_output"),
        "count": int(params.get("count", params.get("images", 10))),
        "open_generated": as_bool(params.get("open_generated", True)),
        "save_layers": as_bool(params.get("save_layers", True)),
        "sample": sample,
        "width": int(params.get("width", 640)),
        "height": int(params.get("height", 512)),
        "pixel_size_nm": float(params.get("pixel_size_nm", 5.0)),
        "seed": int(params.get("seed", 1)),
        "liftout": liftout,
        "trench": as_bool(params.get("trench", True)),
        "pt_cap": as_bool(params.get("pt_cap", True)),
        "needle": as_bool(params.get("needle", liftout)),
        "mill_axis": str(params.get("mill_axis", "y")),
        "curtain_strength": float(params.get("curtain_strength", params.get("curtaining", 0.28 if sample == "bio" else 0.18))),
        "curtain_edge_gain": float(params.get("curtain_edge_gain", 2.0)),
        "mill_gradient": float(params.get("mill_gradient", 0.15)),
        "redeposition": float(params.get("redeposition", 0.06)),
        "charging": float(params.get("charging", 0.12 if sample == "bio" else 0.0)),
        "lamella_thickness_nm": float(params.get("lamella_thickness_nm", 200.0)),
        "electrons_per_pixel": float(params.get("electrons_per_pixel", 80.0 if low_dose else (300.0 if sample == "bio" else 800.0))),
        "mtf_sigma_px": float(params.get("mtf_sigma_px", 0.7)),
        "read_noise_e": float(params.get("read_noise_e", 3.0)),
        "scan_line_jitter": float(params.get("scan_line_jitter", 0.4)),
    }


def _sample_from_text(value) -> str:
    text = str(value or "bio").strip().lower()
    if text in {"material", "materials", "grain", "grains", "oxide", "metal"}:
        return "material"
    return "bio"
