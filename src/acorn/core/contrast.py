"""Contrast normalisation functions optimised for low-dose cryo-EM."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

try:
    import cv2 as _cv2
    def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
        return _cv2.GaussianBlur(arr.astype(np.float32), (0, 0), sigmaX=sigma)
except ImportError:
    _cv2 = None
    def _gaussian(arr: np.ndarray, sigma: float) -> np.ndarray:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr.astype(np.float32), sigma=sigma)

ContrastMethod = Literal["fourier", "percentile", "sigma", "adaptive", "bandpass"]


@dataclass
class ContrastParams:
    """Serialisable parameters for any contrast method."""
    method: ContrastMethod = "percentile"
    # percentile
    low_pct: float = 0.5
    high_pct: float = 99.5
    # sigma
    n_sigma: float = 3.0
    # adaptive (CLAHE)
    clip_limit: float = 0.03
    # Fourier-space bandpass — standard for low-dose cryo-EM structural biology
    # hp_px: remove ice/support gradients at scales larger than this (px)
    # lp_px: suppress shot noise at scales smaller than this (px)
    fbp_hp_px: float = 100.0
    fbp_lp_px: float = 4.0
    # spatial Gaussian bandpass (legacy)
    bp_low_sigma: float = 100.0  # px radius for background subtraction
    bp_high_sigma: float = 1.0   # px radius for noise smoothing
    # post-processing
    gamma: float = 1.0
    colormap: str = "gray"


def normalize_percentile(
    arr: np.ndarray, low_pct: float = 0.5, high_pct: float = 99.5
) -> np.ndarray:
    """Robust percentile clip — handles ice/contamination intensity outliers."""
    lo = np.percentile(arr, low_pct)
    hi = np.percentile(arr, high_pct)
    out = np.clip(arr, lo, hi).astype(np.float32)
    return (out - lo) / (hi - lo + 1e-12)


def normalize_sigma(arr: np.ndarray, n_sigma: float = 3.0) -> np.ndarray:
    """Mean ± N×std clipping — good for tilt-series consistency."""
    mu, sig = arr.mean(), arr.std()
    lo, hi = mu - n_sigma * sig, mu + n_sigma * sig
    out = np.clip(arr, lo, hi).astype(np.float32)
    return (out - lo) / (hi - lo + 1e-12)


def normalize_adaptive(arr: np.ndarray, clip_limit: float = 0.03) -> np.ndarray:
    """CLAHE local histogram equalisation — reveals fine structure."""
    from skimage import exposure
    img8 = ((arr - arr.min()) / (arr.max() - arr.min() + 1e-12) * 255).astype(np.uint8)
    return exposure.equalize_adapthist(img8, clip_limit=clip_limit).astype(np.float32)


def normalize_fourier_bandpass(
    arr: np.ndarray,
    hp_px: float = 100.0,
    lp_px: float = 4.0,
) -> np.ndarray:
    """
    Fourier-space Gaussian bandpass — standard for low-dose cryo-EM display.

    This is the approach used in RELION, cryoSPARC, and CTFFIND for micrograph
    visualisation. Operates in frequency space so there are no ringing artifacts
    and the filter is fully separable.

    hp_px : highpass cutoff (pixels).  Removes ice-thickness/support gradients at
            spatial scales *larger* than this value (the 50% transmission point).
            Typical: 100-300 px.  Set 0 to disable.
    lp_px : lowpass cutoff (pixels).  Suppresses shot noise at scales *smaller*
            than this (50% transmission point).  Typical: 2-6 px.  Set 0 to disable.

    The filter is:
        H(f) = HP(f) * LP(f)
        HP(f) = 1 - exp(-f² / (2 σ_hp²))     σ_hp = 1 / (hp_px * sqrt(2 ln 2))
        LP(f) = exp(-f² / (2 σ_lp²))          σ_lp = 1 / (lp_px * sqrt(2 ln 2))
    where f is the radial spatial frequency in cycles/px (0 … 0.5).
    """
    f = arr.astype(np.float32)
    H, W = f.shape[:2]

    F = np.fft.rfft2(f)

    # Radial frequency grid (cycles/px), shape matches rfft2 output (H, W//2+1)
    fy = np.fft.fftfreq(H)[:, None]      # (H, 1)
    fx = np.fft.rfftfreq(W)[None, :]     # (1, W//2+1)
    freq2 = (fy ** 2 + fx ** 2).astype(np.float32)

    _ln2x2 = 2.0 * math.log(2.0)

    if hp_px > 0:
        sigma_hp2 = (1.0 / (hp_px * math.sqrt(_ln2x2))) ** 2
        hp_filter = (1.0 - np.exp(-freq2 / (2.0 * sigma_hp2))).astype(np.float32)
    else:
        hp_filter = np.ones(freq2.shape, dtype=np.float32)

    if lp_px > 0:
        sigma_lp2 = (1.0 / (lp_px * math.sqrt(_ln2x2))) ** 2
        lp_filter = np.exp(-freq2 / (2.0 * sigma_lp2)).astype(np.float32)
    else:
        lp_filter = np.ones(freq2.shape, dtype=np.float32)

    result = np.fft.irfft2(F * hp_filter * lp_filter, s=(H, W))

    sub = result[::4, ::4]
    lo, hi = np.percentile(sub, 0.5), np.percentile(sub, 99.5)
    result = np.clip(result, lo, hi)
    return (result - lo) / (hi - lo + 1e-12)


_bp_cache: dict = {}   # (array_id, shape, low_sigma) -> background-subtracted float32 array


def normalize_bandpass(
    arr: np.ndarray,
    low_sigma: float = 20.0,
    high_sigma: float = 1.0,
) -> np.ndarray:
    """
    Gaussian bandpass — best contrast method for low-dose cryo-EM.

    Steps:
    1. Subtract a large-radius Gaussian (low_sigma px) to remove slow
       ice-thickness / support-gradient background (low-frequency content).
       Larger values remove broader background. Set 0 to skip.
    2. Apply a small-radius Gaussian (high_sigma px) to suppress high-frequency
       shot noise. Set 0 to skip.
    3. Percentile clip [0.5, 99.5] then scale to [0, 1].

    The background subtraction is computed at 1/8 resolution and upsampled,
    giving ~25x speedup with negligible quality loss at display resolution.
    The result is cached so repeated calls with the same low_sigma are free.
    """
    from scipy.ndimage import zoom

    f = arr.astype(np.float32)
    cache_key = (id(arr), arr.shape, round(low_sigma, 3))

    if low_sigma > 0:
        if cache_key not in _bp_cache:
            # Clear stale entries from other images to avoid unbounded growth
            _bp_cache.clear()
            scale = min(1.0, 8.0 / low_sigma)   # target sigma ≥ 8px in downsampled image
            if scale < 1.0:
                h, w = f.shape[:2]
                sh, sw = max(1, int(h * scale)), max(1, int(w * scale))
                if _cv2 is not None:
                    small = _cv2.resize(f, (sw, sh), interpolation=_cv2.INTER_LINEAR)
                    bg_small = _gaussian(small, sigma=low_sigma * scale)
                    bg = _cv2.resize(bg_small, (w, h), interpolation=_cv2.INTER_LINEAR)
                else:
                    from scipy.ndimage import zoom
                    small = zoom(f, scale, order=1)
                    bg_small = _gaussian(small, sigma=low_sigma * scale)
                    bg = zoom(bg_small, 1.0 / scale, order=1)
                    bg = bg[:h, :w]
                    pad = [(0, max(0, f.shape[i] - bg.shape[i])) for i in range(2)]
                    if any(p[1] > 0 for p in pad):
                        bg = np.pad(bg, pad, mode="edge")
            else:
                bg = _gaussian(f, sigma=low_sigma)
            _bp_cache[cache_key] = f - bg
        f = _bp_cache[cache_key].copy()

    if high_sigma > 0:
        f = _gaussian(f, sigma=high_sigma)

    # Subsample for percentile estimation — statistically equivalent at 1/4 res
    sub = f[::4, ::4]
    lo, hi = np.percentile(sub, 0.5), np.percentile(sub, 99.5)
    f = np.clip(f, lo, hi)
    return (f - lo) / (hi - lo + 1e-12)


def apply_contrast(arr: np.ndarray, params: ContrastParams) -> np.ndarray:
    """Apply contrast normalisation from a ContrastParams object → float32 in [0, 1]."""
    f = arr.astype(np.float32)
    # Sanitize NaN/inf (common in masked / gain-corrected micrographs) so they
    # don't poison percentile/FFT normalisation → a blank image and NaN ROI stats.
    if not np.isfinite(f).all():
        finite = f[np.isfinite(f)]
        if finite.size:
            med, fmin, fmax = float(np.median(finite)), float(finite.min()), float(finite.max())
        else:
            med = fmin = fmax = 0.0
        f = np.nan_to_num(f, nan=med, posinf=fmax, neginf=fmin)
    dispatch = {
        "fourier":    lambda: normalize_fourier_bandpass(f, params.fbp_hp_px, params.fbp_lp_px),
        "percentile": lambda: normalize_percentile(f, params.low_pct, params.high_pct),
        "sigma":      lambda: normalize_sigma(f, params.n_sigma),
        "adaptive":   lambda: normalize_adaptive(f, params.clip_limit),
        "bandpass":   lambda: normalize_bandpass(f, params.bp_low_sigma, params.bp_high_sigma),
    }
    if params.method not in dispatch:
        raise ValueError(
            f"Unknown contrast method: {params.method!r}. "
            f"Choose from: {list(dispatch)}"
        )
    result = dispatch[params.method]()
    if params.gamma != 1.0:
        result = np.power(np.clip(result, 0.0, 1.0), params.gamma)
    return result
