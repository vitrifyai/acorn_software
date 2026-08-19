"""Cryo-TEM forward pipeline: specimen -> potential -> CTF -> dose/detector.

Consumes a resolved setup config (from cryotem.setup) so the microscope
parameters flow straight from the import form into a simulated micrograph.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .potential import Specimen, make_potential
from .optics import ctf, apply_ctf
from .detector import expose, to_display
from .setup import sigma_rad_per_VA


@dataclass
class Micrograph:
    image: np.ndarray        # final noisy micrograph, [0,1] for display
    counts: np.ndarray       # raw electron counts (physics units)
    ideal: np.ndarray        # noiseless CTF image (mean ~1)
    potential: np.ndarray    # projected potential relative to ice (V*Angstrom)
    label: np.ndarray        # per-pixel projected particle thickness (GT)
    defocus_um: float
    config: dict


def _pick_defocus(cfg, rng, defocus_um):
    if defocus_um is not None:
        return defocus_um
    lo, hi = cfg["defocus_min_um"], cfg["defocus_max_um"]
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def simulate_micrograph(cfg: dict, specimen: Specimen | None = None,
                        defocus_um: float | None = None,
                        bfactor: float = 40.0, seed: int = 0) -> Micrograph:
    rng = np.random.default_rng(seed)
    n = int(cfg["image_size_px"])
    shape = (n, n)
    px = cfg["pixel_size_a"]

    specimen = specimen or Specimen(seed=seed)
    pot = make_potential(shape, px, specimen)

    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))     # rad/(V*Angstrom)
    df = _pick_defocus(cfg, rng, defocus_um)
    if cfg.get("phase_plate"):
        # Volta phase plate ~ pi/2 phase shift -> strong near-focus contrast.
        df = df if abs(df) > 0.05 else -0.05
    ctf_2d = ctf(shape, px, cfg, df, bfactor=bfactor)

    ideal = apply_ctf(sigma * pot.v_proj, ctf_2d)
    counts = expose(ideal, cfg, rng)

    return Micrograph(image=to_display(counts), counts=counts, ideal=ideal,
                      potential=pot.v_proj, label=pot.label,
                      defocus_um=df, config=cfg)
