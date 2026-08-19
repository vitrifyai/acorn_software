"""Contrast transfer function: aberrations + partial-coherence envelopes.

A pure phase object is invisible in focus; defocus and Cs turn phase into
recordable amplitude contrast. We use the standard linear weak-phase model
(the same one CTFFIND/RELION assume), so simulated Thon rings match a real
power spectrum at the same defocus:

    chi(k)  = pi*lambda*df*k^2  -  0.5*pi*Cs*lambda^3*k^4      (+ astigmatism)
    CTF(k)  = -( sqrt(1-Q^2)*sin chi  +  Q*cos chi ) * E_t(k) * E_s(k) * E_b(k)
    contrast(x) = IFFT( FFT(sigma*V_proj) * CTF )
    I_ideal(x)  = 1 + 2*contrast

Sign convention: defocus df is in Angstrom with POSITIVE = underfocus
(so a config value of -1.5 um underfocus -> df = +15000 A).
"""
from __future__ import annotations

import numpy as np


def _freq_grid(shape, pixel_size_a):
    """Radial spatial frequency k (cycles/Angstrom) on an fft grid."""
    ky = np.fft.fftfreq(shape[0], d=pixel_size_a)
    kx = np.fft.fftfreq(shape[1], d=pixel_size_a)
    kxx, kyy = np.meshgrid(kx, ky)
    return np.sqrt(kxx ** 2 + kyy ** 2).astype(np.float32)


def dose_damage_envelope(shape, pixel_size_a, cfg, dose_e_per_a2=None):
    """Frequency-dependent radiation-damage envelope (Grant & Grigorieff 2015).

    High resolution fades first as dose accumulates: the critical exposure is
    N_e(q) = 0.245·q^-1.665 + 2.81 (e/Å², q in 1/Å), and the retained signal
    amplitude after dose D is exp(-D / (2·N_e(q))). This replaces the ad-hoc
    fixed B-factor with a physical, dose-driven envelope — so raising the dose
    both adds electrons AND destroys high-frequency signal (the real trade-off).
    """
    D = cfg["total_dose_e_per_a2"] if dose_e_per_a2 is None else dose_e_per_a2
    k = np.clip(_freq_grid(shape, pixel_size_a), 1e-4, None)
    n_e = 0.245 * k ** (-1.665) + 2.81                  # critical exposure (e/Å²)
    return np.exp(-D / (2.0 * n_e)).astype(np.float32)


def ctf(shape, pixel_size_a, cfg, defocus_um, bfactor=40.0, dose_damage=True):
    """Build the 2-D CTF (with envelopes) from a resolved setup `cfg`.

    `bfactor` is now a RESIDUAL envelope (optics/specimen imperfection). The
    dominant high-frequency falloff comes from the physical dose-dependent
    radiation-damage envelope when `dose_damage=True`."""
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2        # pm -> Angstrom
    cs_a = cfg["cs_mm"] * 1e7                             # mm -> Angstrom
    q = cfg["amplitude_contrast"]
    df_a = -defocus_um * 1e4                              # um underfocus(neg) -> +A

    k = _freq_grid(shape, pixel_size_a)
    k2 = k ** 2
    chi = np.pi * lam * df_a * k2 - 0.5 * np.pi * cs_a * lam ** 3 * k2 ** 2

    ctf = -(np.sqrt(max(0.0, 1 - q ** 2)) * np.sin(chi) + q * np.cos(chi))

    # Temporal (chromatic) envelope from focus spread d = Cc*(dE/E).
    cc_a = cfg["cc_mm"] * 1e7
    e_acc = float(cfg["voltage_kv"]) * 1e3
    d_spread = cc_a * (cfg["energy_spread_ev"] / e_acc)
    e_t = np.exp(-0.5 * (np.pi * lam * d_spread * k2) ** 2)

    # Spatial (source-size) envelope from illumination semi-angle alpha.
    alpha = cfg["convergence_mrad"] * 1e-3
    grad = cs_a * lam ** 3 * k2 * k - df_a * lam * k       # (d chi / dk)/(2 pi)
    e_s = np.exp(-(np.pi * alpha / lam) ** 2 * grad ** 2)

    # Residual B-factor envelope (specimen/optics imperfection).
    e_b = np.exp(-bfactor * k2 / 4.0)
    # Physical dose-dependent radiation-damage envelope.
    e_dmg = dose_damage_envelope(shape, pixel_size_a, cfg) if dose_damage else 1.0

    return (ctf * e_t * e_s * e_b * e_dmg).astype(np.float32)


def _tukey2d(shape, alpha=0.1):
    """Separable 2-D Tukey (tapered-cosine) window; tapers the outer `alpha`
    fraction of each edge smoothly to zero."""
    def win1d(n):
        w = np.ones(n)
        edge = int(alpha * n / 2)
        if edge > 0:
            t = np.arange(edge)
            ramp = 0.5 * (1 + np.cos(np.pi * (t / edge - 1)))
            w[:edge] = ramp
            w[-edge:] = ramp[::-1]
        return w
    return np.outer(win1d(shape[0]), win1d(shape[1])).astype(np.float32)


def apply_ctf(sigma_vproj, ctf_2d, apodize=0.1):
    """Linear weak-phase image contrast: I = 1 + 2*IFFT(FFT(sigma*V) * CTF).

    The CTF is applied as a periodic (circular) convolution, so a non-periodic
    input leaks wrap-around banding along the image edges. We taper the edges
    with a Tukey window first (the input is zero-mean, so edges fade to plain
    ice, which is physically benign) to remove the discontinuity.
    """
    s = sigma_vproj
    if apodize and apodize > 0:
        s = s * _tukey2d(s.shape, alpha=apodize)
    contrast = np.fft.ifft2(np.fft.fft2(s) * ctf_2d).real
    return (1.0 + 2.0 * contrast).astype(np.float32)
