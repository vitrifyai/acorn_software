#!/usr/bin/env python3
"""
Level-2 calibration: MEASURE statistics from a real image so the simulator can
reproduce them, rather than only replicating supplied settings.

  cryo-EM : fit the CTF (defocus) from the rotationally-averaged power spectrum
            -- a lightweight CTFFIND -- plus a rough B-factor and an SNR proxy.
  FIB-SEM : measure curtaining strength + mill orientation from the 2-D power
            spectrum anisotropy, plus a noise/contrast proxy.

Honest scope: this recovers GLOBAL style (defocus, envelope decay, striping,
noise level), not the exact scene. Measured values feed the simulator knobs;
any parameter the user set explicitly still wins over a measured one.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

from .optics import _freq_grid, _tukey2d


# --------------------------------------------------------------------------
# Shared: rotationally-averaged power spectrum
# --------------------------------------------------------------------------
def radial_power(img, pixel_size_a, nbins=200):
    """Return (k [cycles/Å], P(k)) rotationally averaged, DC removed."""
    im = (img - img.mean()).astype(np.float32)
    im = im * _tukey2d(im.shape, 0.25)                 # reduce edge leakage
    P = np.abs(np.fft.fft2(im)) ** 2
    k = _freq_grid(img.shape, pixel_size_a)            # cycles/Å (unshifted)
    kmax = k.max() / np.sqrt(2)                        # ignore spectrum corners
    bins = np.linspace(0, kmax, nbins + 1)
    idx = np.digitize(k.ravel(), bins) - 1
    valid = (idx >= 0) & (idx < nbins)
    Pr = np.zeros(nbins); cnt = np.zeros(nbins)
    np.add.at(Pr, idx[valid], P.ravel()[valid])
    np.add.at(cnt, idx[valid], 1)
    kc = 0.5 * (bins[:-1] + bins[1:])
    Pr /= np.maximum(cnt, 1)
    Pr[0] = 0.0                                         # kill residual DC bin
    return kc, Pr


def fit_spectral_decay(kc, Pr, k_lo=1 / 40.0, k_hi=1 / 4.0):
    """Effective spectral B-factor from the overall signal power falloff.

    Models P(k) = C_noise + S*exp(-B_eff k^2 / 2). We estimate the white noise
    floor C from the highest frequencies, subtract it, and log-linear fit the
    signal vs k^2 over the band where signal stays well above the floor. This is
    a MONOTONIC proxy for how fast the reference loses contrast with resolution
    (microscope + specimen + detector combined) -- the quantity a closed-loop
    calibration then matches. Not a microscope-only B; named accordingly.
    """
    hi = min(k_hi, kc.max() * 0.9)
    floor = float(np.median(Pr[(kc > hi) & (kc < kc.max() * 0.98)]) or 0.0)
    band = (kc > k_lo) & (kc < hi)
    k = kc[band]
    sig = Pr[band] - floor
    # Keep only where signal is safely above the noise floor (avoids slope bias).
    good = sig > max(0.05 * sig.max(), 0.0)
    if good.sum() < 8:
        return 0.0, False, 0.0
    A = np.vstack([k[good] ** 2, np.ones(good.sum())]).T
    sol = np.linalg.lstsq(A, np.log(sig[good]), rcond=None)[0]
    b_eff = float(-2.0 * sol[0])
    pred = A @ sol
    lg = np.log(sig[good])
    r2 = 1.0 - float(np.sum((lg - pred) ** 2)) / (float(np.sum((lg - lg.mean()) ** 2)) + 1e-9)
    return max(0.0, b_eff), (r2 > 0.4 and b_eff > 0), r2


def fit_envelope_bfactor(kc, Pr, cfg, defocus_um, k_lo=1 / 40.0, k_hi=1 / 5.0):
    """Effective spectral B-factor from Thon-ring peak decay.

    Given the fitted defocus we know exactly where the CTF peaks (|CTF|~1) and
    zeros (CTF~0) fall. Sample the 1-D power spectrum at the ZEROS to trace the
    noise floor N(k), at the PEAKS to trace N(k)+S*Env(k); the ring contrast
    (peak - floor) decays as exp(-B_eff k^2 / 2), so a log-linear fit vs k^2
    gives B_eff. This measures the TOTAL envelope decay of the reference
    (microscope + specimen + detector) -- exactly what matching needs.
    """
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    cs_a = cfg["cs_mm"] * 1e7
    q = cfg["amplitude_contrast"]
    df_a = -defocus_um * 1e4

    band = (kc > k_lo) & (kc < min(k_hi, kc.max() * 0.95))
    k, P = kc[band], Pr[band]
    if k.size < 12:
        return 0.0, False, 0.0
    chi = np.pi * lam * df_a * k ** 2 - 0.5 * np.pi * cs_a * lam ** 3 * k ** 4
    ctf2 = (np.sqrt(max(0.0, 1 - q ** 2)) * np.sin(chi) + q * np.cos(chi)) ** 2

    turn = np.where(np.diff(np.sign(np.diff(ctf2))) != 0)[0] + 1
    peaks = [i for i in turn if ctf2[i] > 0.5]
    zeros = [i for i in turn if ctf2[i] < 0.1]
    if len(peaks) < 3 or len(zeros) < 2:
        return 0.0, False, 0.0

    floor = np.interp(k, k[zeros], P[zeros])           # noise floor under peaks
    env = np.clip(P[peaks] - floor[peaks], 1e-12, None)
    kp = k[peaks]
    A = np.vstack([kp ** 2, np.ones_like(kp)]).T
    sol = np.linalg.lstsq(A, np.log(env), rcond=None)[0]
    b_eff = float(-2.0 * sol[0])

    pred = A @ sol
    ss_res = float(np.sum((np.log(env) - pred) ** 2))
    ss_tot = float(np.sum((np.log(env) - np.log(env).mean()) ** 2) + 1e-12)
    r2 = 1.0 - ss_res / ss_tot
    reliable = (r2 > 0.5) and (5.0 < b_eff < 3000.0)
    return b_eff, reliable, r2


# --------------------------------------------------------------------------
# Cryo-EM: CTF fit
# --------------------------------------------------------------------------
def characterize_cryo(img, cfg, df_search_um=(0.2, 5.0), n_df=400,
                      k_lo=1 / 40.0, k_hi=1 / 6.0):
    px = cfg["pixel_size_a"]
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2      # Å
    cs_a = cfg["cs_mm"] * 1e7
    q = cfg["amplitude_contrast"]

    kc, Pr = radial_power(img, px)
    k_hi = min(k_hi, kc.max() * 0.95)
    band = (kc > k_lo) & (kc < k_hi)
    k = kc[band]
    # Background-subtract the slow envelope to isolate the ring oscillation.
    bg = uniform_filter1d(Pr, size=max(3, int(len(Pr) * 0.06)))
    # Flatten by DIVISION, not subtraction: the dose-damage envelope makes the
    # spectrum fall off steeply, and subtraction leaves a residual slope that
    # swamps the ring signal (fit collapses to df~0 at high dose). Dividing
    # normalises the ring oscillation uniformly across k so rings survive.
    sub = (Pr / (bg + 1e-12) - 1.0)[band]
    sub = (sub - sub.mean()) / (sub.std() + 1e-8)

    def ctf2(df_a):
        chi = np.pi * lam * df_a * k ** 2 - 0.5 * np.pi * cs_a * lam ** 3 * k ** 4
        model = (np.sqrt(max(0.0, 1 - q ** 2)) * np.sin(chi) + q * np.cos(chi)) ** 2
        return (model - model.mean()) / (model.std() + 1e-8)

    dfs = np.linspace(df_search_um[0], df_search_um[1], n_df) * 1e4   # Å underfocus
    scores = np.array([float(np.dot(sub, ctf2(df))) for df in dfs])
    best = int(np.argmax(scores))
    df_a = dfs[best]
    defocus_um = -df_a / 1e4

    # Effective spectral B-factor from the overall signal power falloff.
    bfactor, b_reliable, b_r2 = fit_spectral_decay(kc, Pr)

    # SNR proxy: in-band ring contrast vs high-frequency noise floor.
    hf = Pr[kc > k_hi]
    snr = float(np.percentile(Pr[band], 90) / (np.median(hf) + 1e-8)) if hf.size else float("nan")

    return {
        "defocus_um": round(float(defocus_um), 3),     # negative = underfocus
        "fit_score": round(float(scores[best] / len(sub)), 4),
        "bfactor": round(bfactor, 1),                  # effective TOTAL envelope B
        "bfactor_reliable": bool(b_reliable),
        "bfactor_r2": round(b_r2, 3),
        "snr_proxy": round(snr, 3),
        "_spectrum": (kc, Pr),
    }


# --------------------------------------------------------------------------
# FIB-SEM: curtaining + noise
# --------------------------------------------------------------------------
def characterize_fib(img, curtain_gain=0.5):
    im = (img - img.mean()).astype(np.float32)
    im = im * _tukey2d(im.shape, 0.2)
    P = np.abs(np.fft.fft2(im)) ** 2
    P[0, 0] = 0.0
    total = P.sum() + 1e-12

    # Vertical stripes vary along x, constant along y -> FT energy concentrated
    # on the ky=0 line (row 0). Horizontal stripes -> kx=0 line (col 0).
    e_vert = P[0, 1:].sum()          # energy of vertical stripes (mill axis 'y')
    e_horz = P[1:, 0].sum()          # energy of horizontal stripes (mill axis 'x')
    curtain_energy = max(e_vert, e_horz)
    mill_axis = "y" if e_vert >= e_horz else "x"
    curtain_index = float(curtain_energy / total)               # 0..~1
    anisotropy = float(curtain_energy / (min(e_vert, e_horz) + 1e-12))

    # Noise proxy: high-frequency power fraction (flat noise floor).
    ny, nx = im.shape
    k = _freq_grid(im.shape, 1.0)                                # cycles/px
    hf_frac = float(P[k > 0.35].sum() / total)

    # Heuristic map to simulator knobs. NOTE: unlike the cryo defocus fit, this
    # is content-dependent (grain boundaries add axial energy too) -> treat the
    # measured curtain_index as a monotonic proxy, not an absolute calibration.
    curtain_strength = float(np.clip(curtain_gain * curtain_index, 0, 0.5))
    # More HF power (noisier) -> lower effective dose.
    electrons_per_pixel = float(np.clip(400.0 / (1 + 40 * hf_frac), 40, 2000))

    return {
        "mill_axis": mill_axis,
        "curtain_index": round(curtain_index, 5),
        "curtain_anisotropy": round(anisotropy, 2),
        "hf_noise_frac": round(hf_frac, 4),
        "contrast_std": round(float(img.std()), 4),
        # mapped knobs:
        "curtain_strength": round(curtain_strength, 3),
        "electrons_per_pixel": round(electrons_per_pixel, 0),
    }
