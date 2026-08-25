"""ACORN plugin for physics-based SEM simulation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, QTimer, pyqtSignal
from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class _SemSimThread(QThread):
    finished_ok = pyqtSignal(int, str, list)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, params: dict) -> None:
        super().__init__()
        self.params = params

    def run(self) -> None:
        try:
            from .io import generate_sem_dataset
            out = _fresh_run_dir(
                Path(self.params.get("output_dir") or (Path.home() / "sem_sim_output")),
                "sem_run")
            self.progress.emit(
                "Tracing electrons for the interaction-volume kernels "
                "(cached after the first run)...")
            paths = generate_sem_dataset(
                out, int(self.params.get("count", 10)), self.params,
                save_layers=bool(self.params.get("save_layers", True)),
                seed=int(self.params.get("seed", 0)))
            self.finished_ok.emit(len(paths), str(out), [str(p) for p in paths])
        except Exception as exc:
            self.failed.emit(str(exc))


class SemSimulationPlugin(AcornPlugin):
    TAB_LABEL = "SEM Simulation"
    PLUGIN_ID = "acorn_sem_sim"
    sort_order = 38
    FLOATING = True
    FLOATING_TITLE = "SEM Simulation"
    FLOATING_SHORTCUT = "Ctrl+Shift+N"
    FLOATING_MIN_WIDTH = 380

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._thread: _SemSimThread | None = None
        context.action_requested.connect(self._on_action_requested)

    def create_panel(self) -> QWidget:
        from .panel import SemSimPanel

        self._panel = SemSimPanel()
        self._panel.generate_requested.connect(self._on_generate_requested)
        self._panel.action_requested.connect(self._on_action_requested)
        return self._panel

    def _on_action_requested(self, action: str, params: dict) -> None:
        if action != "generate_sem_simulation":
            return
        request = _params_from_clu(dict(params))
        if self._panel is not None:
            self._panel.apply_params(request)
        self._on_generate_requested(request)

    def _on_generate_requested(self, params: dict) -> None:
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(None, "SEM Simulation",
                                    "Generation is already running.")
            return

        warning = _field_of_view_warning(params)
        if warning and self._panel is not None:
            # Surfaced rather than silently produced: an interaction volume
            # wider than the field gives a uniformly blurred image that looks
            # like a bug in the simulator rather than a consequence of the kV.
            self._panel.set_status(warning)

        if self._panel is not None:
            self._panel.set_running(True)
            self._panel.set_status(warning or "Generating SEM simulation...")
        self._context.set_status("Generating SEM simulation...")

        self._thread = _SemSimThread(params)
        self._thread.progress.connect(
            lambda m: self._panel.set_status(m) if self._panel is not None else None)
        self._thread.finished_ok.connect(self._on_finished)
        self._thread.failed.connect(self._on_failed)
        self._thread.finished.connect(
            lambda: self._panel.set_running(False) if self._panel is not None else None)
        self._thread.start()

    def _on_finished(self, count: int, output_dir: str, image_paths: list) -> None:
        message = f"Generated {count} SEM image(s) with ground truth: {output_dir}"
        if self._panel is not None:
            self._panel.set_status(message)
        self._context.set_status(message, timeout_ms=8000)
        params = self._thread.params if self._thread is not None else {}
        if params.get("open_generated", True) and image_paths:
            opened, detail = _open_paths_in_acorn(self._context, image_paths)
            if self._panel is not None and not opened:
                self._panel.set_status(detail)

    def _on_failed(self, message: str) -> None:
        if self._panel is not None:
            self._panel.set_status(f"Error: {message}")
        self._context.set_status(f"SEM simulation failed: {message}", timeout_ms=8000)
        QMessageBox.critical(None, "SEM Simulation", message)

    def teardown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.wait(1500)


def _field_of_view_warning(params: dict) -> str:
    """Warn when the interaction volume is a large fraction of the field."""
    try:
        from .materials import get, kanaya_okayama_nm
        kv = float(params.get("E0_kev", 5.0))
        px = float(params.get("pixel_size_nm", 4.0))
        field = px * int(params.get("image_size_px", 512))
        worst = 0.0
        for key in ("substrate", "particle", "matrix", "phase_a", "phase_b"):
            name = params.get(key)
            if not name:
                continue
            try:
                worst = max(worst, kanaya_okayama_nm(get(str(name)), kv))
            except KeyError:
                continue
        if worst > 0 and field / worst < 2:
            return (f"Note: at {kv:g} kV the interaction volume (~{worst / 1000:.1f} um) "
                    f"is comparable to the {field / 1000:.1f} um field, so the image "
                    f"will be heavily blurred. That is real physics, not a fault -- "
                    f"lower the kV to sharpen it.")
    except Exception:
        pass
    return ""


def _params_from_clu(params: dict) -> dict:
    """Normalise a CLU tool call into panel parameters.

    Accepts the loose synonyms a language model will reach for -- kv/voltage for
    beam energy, sample/specimen for scene -- so a reasonable phrasing does not
    fail on vocabulary.
    """
    scene = _scene_from_text(params.get("scene") or params.get("sample")
                             or params.get("specimen"))
    detector = _detector_from_text(params.get("detector") or params.get("signal"))
    kv = _first_float(params, ("E0_kev", "kv", "voltage_kv", "beam_kv", "energy_kev"), 5.0)
    quality = {"fast": 15000, "balanced": 40000, "careful": 120000}
    return {
        "output_dir": params.get("output_dir") or str(Path.home() / "sem_sim_output"),
        "count": int(params.get("count", params.get("images", 10))),
        "open_generated": _as_bool(params.get("open_generated", True)),
        "save_layers": _as_bool(params.get("save_layers", True)),
        "n_electrons": quality.get(str(params.get("quality", "balanced")).lower(), 40000),
        "scene": scene,
        "particle": str(params.get("particle") or params.get("phase_a") or "gold"),
        "substrate": str(params.get("substrate") or params.get("phase_b")
                         or params.get("matrix") or "carbon"),
        "phase_a": str(params.get("phase_a") or params.get("particle") or "iron"),
        "phase_b": str(params.get("phase_b") or params.get("substrate") or "copper"),
        "matrix": str(params.get("matrix") or params.get("substrate") or "alumina"),
        "n_particles": int(params.get("n_particles", params.get("particles", 40))),
        "diameter_nm_mean": _first_float(params, ("diameter_nm_mean", "diameter_nm",
                                                  "diameter"), 40.0),
        "diameter_nm_sd": _first_float(params, ("diameter_nm_sd",), 12.0),
        "relief": _as_bool(params.get("relief", True)),
        "porosity": _first_float(params, ("porosity",), 0.25),
        "n_grains": int(params.get("n_grains", 24)),
        "n_cells": int(params.get("n_cells", 14)),
        "n_layers": int(params.get("n_layers", 4)),
        "E0_kev": kv,
        "pixel_size_nm": _first_float(params, ("pixel_size_nm", "pixel_nm",
                                               "pixel_size"), 4.0),
        "image_size_px": int(params.get("image_size_px", params.get("size", 512))),
        "electrons_per_px": _first_float(params, ("electrons_per_px",
                                                  "electrons_per_pixel", "dose"), 500.0),
        "probe_nm": _first_float(params, ("probe_nm", "probe"), 1.0),
        "detector": detector,
        "bse_mix": _first_float(params, ("bse_mix",), 0.15),
        "elevation_deg": _first_float(params, ("elevation_deg",), 25.0),
        "azimuth_deg": _first_float(params, ("azimuth_deg",), 0.0),
        "asymmetry": _first_float(params, ("asymmetry", "shading"), 0.30),
        "read_noise_e": _first_float(params, ("read_noise_e", "read_noise"), 3.0),
        "scan_jitter_px": _first_float(params, ("scan_jitter_px", "jitter"), 0.0),
        "seed": int(params.get("seed", 0)),
    }


def _first_float(params: dict, keys: tuple, default: float) -> float:
    for k in keys:
        if params.get(k) is not None:
            try:
                return float(params[k])
            except (TypeError, ValueError):
                continue
    return float(default)


def _scene_from_text(value) -> str:
    text = str(value or "nanoparticles").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"grains", "alloy", "two_phase", "phases", "grain", "polished"}:
        return "grains"
    if text in {"porous", "pores", "pore", "ceramic", "foam"}:
        return "porous"
    if text in {"biological", "bio", "cells", "cell", "resin", "tissue"}:
        return "biological"
    if text in {"cross_section", "crosssection", "fib", "lamella", "layers", "stack"}:
        return "cross_section"
    return "nanoparticles"


def _detector_from_text(value) -> str:
    text = str(value or "ETD").strip().lower()
    if text in {"bse", "backscatter", "backscattered", "z_contrast", "compositional",
                "annular"}:
        return "BSE"
    if text in {"tld", "inlens", "in_lens", "in-lens", "through_lens", "immersion"}:
        return "TLD"
    return "ETD"


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


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
        return 0, "Generated images, but no output files were found to open."
    window_getter = getattr(context, "_w", None)
    if window_getter is None:
        return 0, "Generated images, but the Acorn window is unavailable for auto-open."
    window = window_getter()
    if window is None or not hasattr(window, "open_files"):
        return 0, "Generated images, but the Acorn viewer is unavailable for auto-open."
    QTimer.singleShot(0, lambda p=real_paths: window.open_files(p))
    return len(real_paths), f"Opening {len(real_paths)} generated image(s)."
