"""Microscope and detector tables + parameterised DQE / MTF models.

DQE(q) and MTF(q) are approximations of each detector's published curves as a
function of q = k / k_Nyquist in [0, 1]:

    MTF(q) = mtf_floor + (1 - mtf_floor) / (1 + (q/mtf_q0)^mtf_n)      # MTF(0)=1
    DQE(q) = dqe_ny + (dqe0 - dqe_ny) * (1 - q)^dqe_shape             # DQE(0)=dqe0

These are realistic *shapes* (counting detectors: high, flat-ish DQE falling
toward Nyquist; integrating CMOS: low DQE, soft MTF), not vendor-exact curves.
Swap in measured curves per detector when you have them.
"""
from __future__ import annotations

import numpy as np

# Microscope presets: accelerating voltage + aberrations + energy spread.
MICROSCOPES = {
    "krios":         dict(voltage_kv="300", cs_mm=2.7,  cc_mm=2.7, energy_spread_ev=0.9),
    "krios-cfeg":    dict(voltage_kv="300", cs_mm=2.7,  cc_mm=2.7, energy_spread_ev=0.35),
    "glacios":       dict(voltage_kv="200", cs_mm=2.7,  cc_mm=2.7, energy_spread_ev=0.9),
    "talos-arctica": dict(voltage_kv="200", cs_mm=2.7,  cc_mm=2.7, energy_spread_ev=0.9),
    "talos-l120c":   dict(voltage_kv="120", cs_mm=2.7,  cc_mm=2.0, energy_spread_ev=1.0),
    "cs-corrected":  dict(voltage_kv="300", cs_mm=0.01, cc_mm=2.7, energy_spread_ev=0.7),
}

# Detector table. kind: 'counting' (direct electron) or 'integrating' (CMOS).
# native_px_um informs pixel scale; DQE/MTF params define the response.
DETECTORS = {
    "K3":        dict(kind="counting",    native_px_um=5.0,  dqe0=0.92, dqe_ny=0.34, dqe_shape=1.3, mtf_floor=0.05, mtf_q0=0.50, mtf_n=2.2),
    "K2":        dict(kind="counting",    native_px_um=5.0,  dqe0=0.82, dqe_ny=0.24, dqe_shape=1.5, mtf_floor=0.05, mtf_q0=0.42, mtf_n=2.0),
    "Falcon4":   dict(kind="counting",    native_px_um=14.0, dqe0=0.93, dqe_ny=0.42, dqe_shape=1.1, mtf_floor=0.06, mtf_q0=0.58, mtf_n=2.2),
    "Falcon4i":  dict(kind="counting",    native_px_um=14.0, dqe0=0.94, dqe_ny=0.44, dqe_shape=1.1, mtf_floor=0.06, mtf_q0=0.60, mtf_n=2.2),
    "Falcon3EC": dict(kind="counting",    native_px_um=14.0, dqe0=0.72, dqe_ny=0.30, dqe_shape=1.3, mtf_floor=0.05, mtf_q0=0.50, mtf_n=2.0),
    "Apollo":    dict(kind="counting",    native_px_um=8.0,  dqe0=0.90, dqe_ny=0.40, dqe_shape=1.1, mtf_floor=0.05, mtf_q0=0.55, mtf_n=2.1),
    "DE64":      dict(kind="counting",    native_px_um=6.5,  dqe0=0.88, dqe_ny=0.38, dqe_shape=1.2, mtf_floor=0.05, mtf_q0=0.53, mtf_n=2.1),
    "Ceta":      dict(kind="integrating", native_px_um=14.0, dqe0=0.42, dqe_ny=0.05, dqe_shape=2.0, mtf_floor=0.03, mtf_q0=0.35, mtf_n=1.8),
    "ideal":     dict(kind="counting",    native_px_um=5.0,  dqe0=1.0,  dqe_ny=1.0,  dqe_shape=1.0, mtf_floor=1.0,  mtf_q0=1e6,  mtf_n=2.0),
}


def detector_mtf(q, det):
    """MTF(q), q = k/k_Nyquist, MTF(0)=1."""
    c = det["mtf_floor"]
    return c + (1.0 - c) / (1.0 + (q / det["mtf_q0"]) ** det["mtf_n"])


def detector_dqe(q, det):
    """DQE(q), q = k/k_Nyquist, DQE(0)=dqe0."""
    return det["dqe_ny"] + (det["dqe0"] - det["dqe_ny"]) * np.clip(1.0 - q, 0, 1) ** det["dqe_shape"]
