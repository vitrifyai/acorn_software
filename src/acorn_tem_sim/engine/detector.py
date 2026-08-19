"""Dose fractionation + detector response, applied LAST.

Detector-specific model (replaces the old single Gaussian MTF). The detector is
characterised by its MTF(k) and DQE(k) curves (from cryotem.detectors). The
recorded image is formed in the standard frequency-domain DQE model:

    lam      = ideal * dose * pixel_area          # incident electrons / pixel (mean)
    signal   = IFFT( FFT(lam) * MTF(k) )          # detector blurs the signal
    shot     = Poisson(lam) - lam                 # white shot noise (variance = lam)
    noise    = IFFT( FFT(shot) * MTF(k)/sqrt(DQE(k)) )   # coloured so NPS = MTF^2/DQE
    recorded = signal + noise

so the output MTF and DQE match the detector: DQE = MTF^2 / NNPS. Counting
detectors keep high DQE to Nyquist; a scintillator CMOS (Ceta) blurs more and
adds excess noise -> visibly worse SNR at the same dose.
"""
from __future__ import annotations

import numpy as np

from .detectors import DETECTORS, detector_mtf, detector_dqe
from .optics import _freq_grid


def _detector_transfers(shape, pixel_size_a, det):
    """Return (MTF, noise-transfer) arrays on the fft grid for this detector."""
    k = _freq_grid(shape, pixel_size_a)          # cycles/Å
    k_ny = 0.5 / pixel_size_a
    q = np.clip(k / k_ny, 0, 1.0)
    mtf = detector_mtf(q, det).astype(np.float32)
    dqe = np.clip(detector_dqe(q, det), 1e-3, None).astype(np.float32)
    ntf = (mtf / np.sqrt(dqe)).astype(np.float32)   # noise transfer function
    return mtf, ntf


def expose(i_ideal, cfg, rng, dose_scale=1.0):
    dose = cfg["total_dose_e_per_a2"] * dose_scale     # dose_scale<1: energy-filtered thick specimen
    px = cfg["pixel_size_a"]
    det = DETECTORS.get(cfg.get("detector_model", "K3"), DETECTORS["K3"])

    lam = np.clip(i_ideal, 0, None) * dose * px ** 2      # mean e-/pixel
    mtf, ntf = _detector_transfers(lam.shape, px, det)

    # Signal blurred by the MTF.
    signal = np.fft.ifft2(np.fft.fft2(lam) * mtf).real

    # White shot noise, then coloured by MTF/sqrt(DQE) so output DQE matches.
    shot = rng.poisson(lam).astype(np.float32) - lam
    noise = np.fft.ifft2(np.fft.fft2(shot) * ntf).real

    counts = np.clip(signal + noise, 0, None).astype(np.float32)
    return counts


def to_display(counts):
    lo, hi = np.percentile(counts, [0.5, 99.5])
    return np.clip((counts - lo) / (hi - lo + 1e-8), 0, 1).astype(np.float32)
