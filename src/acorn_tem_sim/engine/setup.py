#!/usr/bin/env python3
"""
Setup / import phase for the cryo-TEM simulator (CryoSPARC-style).

Header metadata from MRC/DM4 is only partially reliable: MRC usually gives a
trustworthy pixel size but NOT voltage/Cs; DM4 often has voltage + pixel size
but units vary; dose is inconsistent everywhere. So this module:

  1. PROBES a file for whatever it can trust (pixel size, kV, shape, ...),
  2. ASKS the user for the physics the header can't be trusted for
     (kV, Cs, pixel size, amplitude contrast, dose, coherence, detector),
     pre-filled from the probe or an instrument PRESET,
  3. VALIDATES ranges, DERIVES wavelength + interaction parameter, and ECHOES
     the fully-resolved config back so the user can catch a wrong value before
     a single image is simulated.

Run interactively:   python -m cryotem.setup --file micrograph.dm4
Scripted/tested:     resolve(answers={...})   /   run_setup(answers={...})
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path

from .detectors import MICROSCOPES, DETECTORS

# ---- physical constants (SI) ---------------------------------------------
_H = 6.62607015e-34      # Planck (J s)
_ME = 9.1093837015e-31   # electron rest mass (kg)
_E = 1.602176634e-19     # elementary charge (C)
_C = 299792458.0         # speed of light (m/s)
_E0_EV = _ME * _C ** 2 / _E   # rest energy in eV (~511 keV)


def wavelength_pm(voltage_kv: float) -> float:
    """Relativistic electron wavelength in picometres."""
    V = voltage_kv * 1e3
    lam_m = _H / math.sqrt(2 * _ME * _E * V * (1 + _E * V / (2 * _ME * _C ** 2)))
    return lam_m * 1e12


def sigma_rad_per_VA(voltage_kv: float) -> float:
    """Interaction parameter sigma in rad/(V*Angstrom): phase = sigma * V_proj.

    Kirkland form  sigma = 2*pi*m*e*lambda / h^2  with the relativistic mass.
    (~6.53e-4 at 300 kV, ~9.24e-4 at 100 kV.)
    """
    V = voltage_kv * 1e3
    gamma = 1 + _E * V / (_ME * _C ** 2)
    m = gamma * _ME
    lam_m = wavelength_pm(voltage_kv) * 1e-12
    sigma_per_Vm = 2 * math.pi * m * _E * lam_m / _H ** 2
    return sigma_per_Vm * 1e-10   # rad/(V*m) -> rad/(V*Angstrom)


# ---- the form fields ------------------------------------------------------
@dataclass
class Field:
    name: str
    default: object
    unit: str
    help: str
    critical: bool = False          # physics-critical: contrast breaks if wrong
    kind: str = "float"             # 'float' | 'int' | 'bool' | 'choice'
    choices: tuple = ()
    lo: float | None = None
    hi: float | None = None

    def cast(self, raw):
        if self.kind == "bool":
            return str(raw).strip().lower() in ("1", "true", "yes", "y", "on")
        if self.kind == "int":
            return int(raw)
        if self.kind == "choice":
            v = str(raw).strip()
            if self.choices and v not in self.choices:
                raise ValueError(f"choose one of {self.choices}")
            return v
        return float(raw)

    def validate(self, v):
        if self.kind in ("float", "int"):
            if self.lo is not None and v < self.lo:
                raise ValueError(f"{v} < min {self.lo}")
            if self.hi is not None and v > self.hi:
                raise ValueError(f"{v} > max {self.hi}")
        return v


# Grouped so the wizard prints sections like CryoSPARC's import form.
FIELDS = {
    "Optics (physics-critical)": [
        Field("pixel_size_a", None, "Å/px", "Calibrated pixel size at the specimen. "
              "MRC often stores this; ALWAYS confirm it.", critical=True, lo=0.1, hi=50),
        Field("voltage_kv", 300, "kV", "Accelerating voltage.", critical=True,
              kind="choice", choices=("300", "200", "120", "100")),
        Field("cs_mm", 2.7, "mm", "Spherical aberration (2.7 for Krios/Glacios; "
              "~0.01 if Cs-corrected).", critical=True, lo=0.0, hi=10.0),
        Field("amplitude_contrast", 0.10, "-", "Amplitude-contrast ratio Q "
              "(~0.07-0.10 for cryo).", critical=True, lo=0.0, hi=0.5),
    ],
    "Exposure": [
        Field("total_dose_e_per_a2", 40.0, "e-/Å²", "Total electron dose.", lo=0.5, hi=300),
    ],
    "Coherence (envelopes — advanced)": [
        Field("cc_mm", 2.7, "mm", "Chromatic aberration (temporal-coherence envelope).", lo=0, hi=10),
        Field("energy_spread_ev", 0.9, "eV", "Energy spread (FEG ~0.7-1.0).", lo=0.1, hi=3.0),
        Field("convergence_mrad", 0.10, "mrad", "Illumination semi-angle "
              "(spatial-coherence envelope).", lo=0.0, hi=5.0),
    ],
    "Instrument": [
        Field("microscope", "krios", "-", "Sets voltage / Cs / Cc / energy spread.",
              kind="choice", choices=("krios", "krios-cfeg", "glacios",
                                      "talos-arctica", "talos-l120c", "cs-corrected")),
        Field("detector_model", "K3", "-", "Sets the DQE / MTF / noise model.",
              kind="choice", choices=("K3", "K2", "Falcon4", "Falcon4i",
                                      "Falcon3EC", "Apollo", "DE64", "Ceta", "ideal")),
    ],
    "Canvas": [
        Field("image_size_px", 1024, "px", "Simulated image size (square).", kind="int", lo=64, hi=8192),
    ],
    "Simulation": [
        Field("defocus_min_um", -1.0, "µm", "Defocus range start (negative = underfocus).", lo=-8, hi=2),
        Field("defocus_max_um", -2.5, "µm", "Defocus range end.", lo=-8, hi=2),
        Field("phase_plate", False, "-", "Volta phase plate (in-focus phase contrast).", kind="bool"),
        Field("energy_filter_ev", 0.0, "eV", "Energy-filter slit width (0 = none).", lo=0, hi=100),
    ],
}

# Instrument presets now just pick a (microscope, detector) combo; the physics
# comes from the MICROSCOPES/DETECTORS tables. 'krios-k3' is the user's default.
PRESETS = {
    "krios-k3":        {"microscope": "krios",         "detector_model": "K3"},
    "krios-falcon4":   {"microscope": "krios",         "detector_model": "Falcon4"},
    "glacios-falcon4": {"microscope": "glacios",       "detector_model": "Falcon4"},
    "talos-ceta":      {"microscope": "talos-arctica", "detector_model": "Ceta"},
    "cs-corrected":    {"microscope": "cs-corrected",  "detector_model": "K3"},
}
DEFAULT_PRESET = "krios-k3"


# ---- header probe ---------------------------------------------------------
@dataclass
class ProbedMeta:
    pixel_size_a: float | None = None
    voltage_kv: float | None = None
    cs_mm: float | None = None
    dose_e_per_a2: float | None = None
    shape: tuple | None = None
    dtype: str | None = None
    source: dict = field(default_factory=dict)   # field -> where it came from
    warnings: list = field(default_factory=list)

    def prefill(self) -> dict:
        d = {}
        if self.pixel_size_a is not None:
            d["pixel_size_a"] = self.pixel_size_a
        if self.voltage_kv is not None:
            d["voltage_kv"] = str(int(self.voltage_kv))
        if self.cs_mm is not None:
            d["cs_mm"] = self.cs_mm
        if self.dose_e_per_a2 is not None:
            d["total_dose_e_per_a2"] = self.dose_e_per_a2
        return d


def _flag_pixel(px, meta):
    if px is None:
        return
    if abs(px - 1.0) < 1e-6:
        meta.warnings.append("pixel size is exactly 1.0 Å — likely a placeholder, confirm it.")
    elif px <= 0 or px > 20:
        meta.warnings.append(f"pixel size {px:g} Å looks out of range — confirm it.")


def _probe_mrc(path: Path) -> ProbedMeta:
    import mrcfile
    m = ProbedMeta()
    with mrcfile.open(path, permissive=True, header_only=True) as mrc:
        h = mrc.header
        nx = int(h.mx) or int(h.nx)
        px = float(h.cella.x) / nx if nx else None
        m.pixel_size_a = px
        m.shape = (int(h.ny), int(h.nx))
        m.dtype = str(mrc.data.dtype) if mrc.data is not None else f"mode {int(h.mode)}"
        if px:
            m.source["pixel_size_a"] = "MRC header cella/mx"
        # Voltage/Cs/dose live only in extended (FEI/SerialEM) headers if at all.
        m.warnings.append("MRC base header has no kV/Cs — using preset/your input.")
    _flag_pixel(m.pixel_size_a, m)
    return m


def _probe_dm(path: Path) -> ProbedMeta:
    from ncempy.io import dm
    m = ProbedMeta()
    with dm.fileDM(str(path)) as f:
        ds = f.getDataset(0)
        # Pixel size from calibration; check units (nm/µm -> Å).
        scale = ds.get("pixelSize", [None])[0]
        unit = (ds.get("pixelUnit", [""]) or [""])[0]
        if scale:
            u = str(unit).lower()
            factor = {"nm": 10.0, "µm": 1e4, "um": 1e4, "a": 1.0, "å": 1.0}.get(u, 1.0)
            m.pixel_size_a = float(scale) * factor
            m.source["pixel_size_a"] = f"DM calibration ({scale} {unit})"
            if u not in ("a", "å", "nm", "um", "µm"):
                m.warnings.append(f"DM pixel unit '{unit}' unrecognised — verify pixel size.")
        # Voltage / Cs from the microscope tag tree (paths vary by version).
        tags = f.allTags
        for k, v in tags.items():
            kl = k.lower()
            if m.voltage_kv is None and kl.endswith("microscope info.voltage"):
                try:
                    m.voltage_kv = float(v) / 1000.0  # stored in volts
                    m.source["voltage_kv"] = "DM Microscope Info.Voltage"
                except (TypeError, ValueError):
                    pass
            if m.cs_mm is None and "cs(mm)" in kl:
                try:
                    m.cs_mm = float(v); m.source["cs_mm"] = "DM Microscope Info.Cs(mm)"
                except (TypeError, ValueError):
                    pass
        if hasattr(ds["data"], "shape"):
            m.shape = tuple(ds["data"].shape[-2:])
    _flag_pixel(m.pixel_size_a, m)
    return m


def probe_metadata(path) -> ProbedMeta:
    """Read whatever a file can be trusted to provide; degrade gracefully."""
    path = Path(path)
    ext = path.suffix.lower()
    try:
        if ext in (".mrc", ".mrcs", ".map", ".st", ".ali", ".rec"):
            return _probe_mrc(path)
        if ext in (".dm4", ".dm3"):
            return _probe_dm(path)
    except Exception as exc:  # never let a bad header abort the whole setup
        m = ProbedMeta()
        m.warnings.append(f"could not parse {ext} metadata ({exc}); enter values manually.")
        return m
    m = ProbedMeta()
    m.warnings.append(f"no probe for '{ext}'; enter values manually.")
    return m


# ---- resolve + echo -------------------------------------------------------
def resolve(answers: dict | None = None, meta: ProbedMeta | None = None,
            preset: str = DEFAULT_PRESET) -> dict:
    """Merge preset < header-probe < user answers; validate; add derived physics."""
    answers = dict(answers or {})
    values = {}
    # Resolve the microscope selector first, then expand it into the physics
    # fields (voltage/Cs/Cc/energy spread) as the lowest-precedence layer so a
    # header value or explicit user answer still overrides it.
    microscope = (answers.get("microscope")
                  or PRESETS.get(preset, {}).get("microscope") or "krios")
    scope_expand = dict(MICROSCOPES.get(microscope, {}))
    layers = [scope_expand,
              PRESETS.get(preset, {}),
              meta.prefill() if meta else {},
              answers]
    for fld in (f for group in FIELDS.values() for f in group):
        raw = None
        for layer in layers:
            if fld.name in layer and layer[fld.name] is not None:
                raw = layer[fld.name]
        if raw is None:
            raw = fld.default
        if raw is None:
            raise ValueError(f"'{fld.name}' has no value and no default — must be supplied "
                             f"({fld.help})")
        values[fld.name] = fld.validate(fld.cast(raw))

    if values["defocus_max_um"] > values["defocus_min_um"]:
        values["defocus_min_um"], values["defocus_max_um"] = \
            values["defocus_max_um"], values["defocus_min_um"]

    kv = float(values["voltage_kv"])
    det = DETECTORS.get(values["detector_model"], {})
    values["_derived"] = {
        "wavelength_pm": round(wavelength_pm(kv), 4),
        "sigma_rad_per_VA": sigma_rad_per_VA(kv),
        "preset": preset,
        "microscope": microscope,
        "detector_kind": det.get("kind"),
        "detector_dqe0": det.get("dqe0"),
    }
    return values


def format_echo(cfg: dict, meta: ProbedMeta | None = None) -> str:
    d = cfg["_derived"]
    kv = float(cfg["voltage_kv"])
    lines = ["", "Resolved simulation setup", "=" * 40]
    if meta and meta.source:
        lines.append("From file header: " + ", ".join(
            f"{k}={cfg.get(k)} ({src})" for k, src in meta.source.items()))
    lines += [
        f"  {kv:.0f} kV  ->  λ = {d['wavelength_pm']} pm,  "
        f"σ = {d['sigma_rad_per_VA']:.3e} rad/(V·Å)",
        f"  pixel size      : {cfg['pixel_size_a']} Å/px",
        f"  Cs / Q          : {cfg['cs_mm']} mm  /  {cfg['amplitude_contrast']}",
        f"  dose            : {cfg['total_dose_e_per_a2']} e-/Å²",
        f"  coherence       : Cc {cfg['cc_mm']} mm, ΔE {cfg['energy_spread_ev']} eV, "
        f"α {cfg['convergence_mrad']} mrad",
        f"  microscope      : {d['microscope']}",
        f"  detector        : {cfg['detector_model']} ({d['detector_kind']}, "
        f"DQE₀≈{d['detector_dqe0']}), canvas {cfg['image_size_px']} px",
        f"  defocus range   : {cfg['defocus_min_um']} … {cfg['defocus_max_um']} µm"
        + ("  [Volta phase plate]" if cfg["phase_plate"] else "")
        + (f"  [E-filter {cfg['energy_filter_ev']} eV]" if cfg["energy_filter_ev"] else ""),
        f"  preset          : {d['preset']}",
    ]
    if meta and meta.warnings:
        lines.append("Warnings:")
        lines += [f"  ! {w}" for w in meta.warnings]
    lines.append("=" * 40)
    return "\n".join(lines)


# ---- interactive wizard ---------------------------------------------------
def _ask(fld: Field, current):
    """Prompt for one field; empty input accepts `current`. Re-asks on error."""
    show = current if current is not None else "(required)"
    choices = f" {list(fld.choices)}" if fld.kind == "choice" else ""
    unit = f" [{fld.unit}]" if fld.unit not in ("-", "") else ""
    while True:
        raw = input(f"  {fld.name}{unit}{choices} [{show}]: ").strip()
        if raw == "":
            if current is None:
                print("    ! required — please enter a value")
                continue
            return current
        try:
            return fld.validate(fld.cast(raw))
        except ValueError as e:
            print(f"    ! {e}")


def run_setup(file: str | None = None, preset: str = DEFAULT_PRESET,
              answers: dict | None = None, out: str | None = "sim_config.yaml") -> dict:
    """The CryoSPARC-style setup phase.

    If `answers` is given, runs non-interactively (for scripts/tests); otherwise
    prompts field-by-field, pre-filled from the header probe and the preset.
    """
    meta = probe_metadata(file) if file else None
    if meta:
        print(f"\nProbed {file}:")
        print(f"  shape={meta.shape} dtype={meta.dtype} "
              f"pixel_size={meta.pixel_size_a} kV={meta.voltage_kv}")
        for w in meta.warnings:
            print(f"  ! {w}")

    if answers is None:  # interactive
        prefill = {**PRESETS.get(preset, {}), **(meta.prefill() if meta else {})}
        print(f"\nSetup form (preset '{preset}'). Enter = keep the shown value.\n")
        answers = {}
        for group, flds in FIELDS.items():
            print(f"[{group}]")
            for fld in flds:
                cur = prefill.get(fld.name, fld.default)
                answers[fld.name] = _ask(fld, cur)
            print()

    cfg = resolve(answers, meta=meta, preset=preset)
    print(format_echo(cfg, meta))

    if out:
        import yaml
        payload = {k: v for k, v in cfg.items() if k != "_derived"}
        payload["_derived"] = cfg["_derived"]
        payload["_source_file"] = file
        Path(out).write_text(yaml.safe_dump(payload, sort_keys=False))
        print(f"\nWrote {out}")
    return cfg


def _cli():
    ap = argparse.ArgumentParser(description="cryo-TEM simulation setup phase")
    ap.add_argument("--file", help="MRC/DM4 image to probe for pixel size/kV")
    ap.add_argument("--preset", default=DEFAULT_PRESET, choices=list(PRESETS))
    ap.add_argument("--out", default="sim_config.yaml")
    args = ap.parse_args()
    run_setup(file=args.file, preset=args.preset, out=args.out)


if __name__ == "__main__":
    _cli()
