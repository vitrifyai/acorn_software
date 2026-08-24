"""
Denoising for low-dose electron microscopy.

Low-dose cryo-EM and TEM images are shot-noise limited: the signal is a few
electrons per pixel and the noise is Poisson. Every method here trades spatial
resolution for signal-to-noise, and the right trade depends on what you are
looking for — a 30 nm particle survives aggressive smoothing that would erase a
lattice fringe.

Ordered by what they cost you:

    gaussian     fastest, blurs everything equally
    median       removes hot pixels and X-ray spikes without blurring edges
    butterworth  low-pass at a stated resolution in angstroms — the cryo-EM
                 convention, because the cutoff means something physical
    bilateral    smooths within regions, keeps boundaries
    tv           piecewise-flat result; good for segmentation, bad for texture
    wavelet      general purpose, scale-aware
    nlmeans      best quality on repetitive structure, and much the slowest
    bm3d         state of the art classical, optional dependency
    topaz        learned, trained on real cryo-EM micrographs, optional

A caution that belongs in the code rather than only in a manual: ACORN applies
contrast and denoising before handing an image to a detector AND before writing
training tiles, so whatever is set here is baked into detections and datasets.
Measurements taken from a denoised image are biased — smoothing pulls particle
boundaries inward and raises apparent circularity. Denoise to see and to detect;
measure from raw, or record what was applied. `DenoiseParams.describe()` exists
so that record can be written into provenance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass(frozen=True)
class DenoiseParams:
    """How to denoise. `method="none"` is the default and does nothing."""

    method: str = "none"
    # 0..1, mapped onto whatever each method's natural parameter is, so the
    # control means the same thing to the user whichever method is chosen.
    strength: float = 0.5
    # Butterworth only: cutoff in angstroms. Needs the image's pixel size.
    resolution_a: float = 20.0

    def describe(self) -> str:
        """One line for provenance, so a reader knows what was applied."""
        if self.method == "none":
            return "none"
        if self.method == "butterworth":
            return f"butterworth low-pass {self.resolution_a:.1f} A"
        return f"{self.method} strength {self.strength:.2f}"


# ── the methods ───────────────────────────────────────────────────────────────

def _gaussian(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma=max(0.1, p.strength * 4.0))


def _median(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from scipy.ndimage import median_filter
    size = int(round(1 + p.strength * 6)) | 1        # odd, 1..7
    return median_filter(a, size=max(3, size))


def _butterworth(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    """
    Low-pass at a stated resolution, the way cryo-EM states it.

    Without a pixel size the cutoff cannot be expressed in angstroms, so it falls
    back to treating `strength` as a fraction of Nyquist rather than guessing a
    calibration.
    """
    from skimage.filters import butterworth

    if px_nm and px_nm > 0 and p.resolution_a > 0:
        px_a = px_nm * 10.0
        # cutoff as a fraction of Nyquist: 2*pixel/resolution
        cutoff = min(0.49, max(0.005, (2.0 * px_a) / p.resolution_a / 2.0))
    else:
        cutoff = min(0.49, max(0.005, 1.0 - p.strength))
    return butterworth(a, cutoff_frequency_ratio=cutoff, high_pass=False, order=4)


def _bilateral(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from skimage.restoration import denoise_bilateral
    return denoise_bilateral(a, sigma_color=max(0.01, p.strength * 0.3),
                             sigma_spatial=max(1.0, p.strength * 8.0))


def _tv(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from skimage.restoration import denoise_tv_chambolle
    return denoise_tv_chambolle(a, weight=max(0.01, p.strength * 0.3))


def _wavelet(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from skimage.restoration import denoise_wavelet, estimate_sigma
    sigma = float(np.mean(estimate_sigma(a))) * (0.5 + p.strength * 1.5)
    return denoise_wavelet(a, sigma=sigma, mode="soft",
                           method="BayesShrink", rescale_sigma=True)


def _nlmeans(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    from skimage.restoration import denoise_nl_means, estimate_sigma
    sigma = float(np.mean(estimate_sigma(a)))
    return denoise_nl_means(a, h=max(1e-4, p.strength * 1.5 * sigma), sigma=sigma,
                            patch_size=5, patch_distance=6, fast_mode=True)


def _bm3d(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    import bm3d
    from skimage.restoration import estimate_sigma
    sigma = float(np.mean(estimate_sigma(a))) * (0.5 + p.strength * 1.5)
    return bm3d.bm3d(a, sigma_psd=max(1e-4, sigma))


def _topaz(a: np.ndarray, p: DenoiseParams, px_nm: Optional[float]) -> np.ndarray:
    """Topaz-Denoise, trained on real cryo-EM micrographs."""
    from topaz.denoise import denoise as topaz_denoise      # noqa: F401
    from topaz.denoise import load_model as topaz_load

    model = topaz_load("unet")
    return np.asarray(topaz_denoise(model, a.astype(np.float32)), dtype=np.float32)


@dataclass(frozen=True)
class Method:
    key: str
    label: str
    note: str
    fn: Callable[[np.ndarray, DenoiseParams, Optional[float]], np.ndarray]
    requires: tuple[str, ...] = ()          # importable modules this needs
    uses_resolution: bool = False           # shows the angstrom control instead of strength
    slow: bool = False


METHODS: tuple[Method, ...] = (
    Method("none", "None", "Show the data as it is.", lambda a, p, s: a),
    Method("gaussian", "Gaussian blur",
           "Fastest. Blurs signal and noise equally.", _gaussian),
    Method("median", "Median",
           "Removes hot pixels and X-ray spikes without blurring edges.", _median),
    Method("butterworth", "Low-pass (resolution)",
           "Cuts everything finer than a stated resolution — the cryo-EM convention.",
           _butterworth, uses_resolution=True),
    Method("bilateral", "Bilateral",
           "Smooths inside regions while keeping their boundaries.", _bilateral, slow=True),
    Method("tv", "Total variation",
           "Piecewise-flat result. Helps segmentation, flattens real texture.", _tv),
    Method("wavelet", "Wavelet",
           "General purpose and scale-aware. A good first choice.",
           _wavelet, requires=("pywt",)),
    Method("nlmeans", "Non-local means",
           "Best on repetitive structure, and much the slowest.",
           _nlmeans, requires=("pywt",), slow=True),
    Method("bm3d", "BM3D",
           "State of the art classical denoiser. Needs the bm3d package.",
           _bm3d, requires=("bm3d",), slow=True),
    Method("topaz", "Topaz-Denoise",
           "Learned, trained on real cryo-EM micrographs. Needs topaz.",
           _topaz, requires=("topaz",), slow=True),
)

_BY_KEY = {m.key: m for m in METHODS}


def is_available(method: str) -> bool:
    """True when this method's dependencies are importable."""
    import importlib.util

    m = _BY_KEY.get(method)
    if m is None:
        return False
    return all(importlib.util.find_spec(r) is not None for r in m.requires)


def available_methods() -> tuple[Method, ...]:
    """Methods that can actually run in this environment."""
    return tuple(m for m in METHODS if is_available(m.key))


def missing_note(method: str) -> str:
    """What to install, when a method is chosen but unavailable."""
    m = _BY_KEY.get(method)
    if m is None or is_available(method):
        return ""
    return (f"{m.label} needs: {', '.join(m.requires)}. "
            f"Install it, or choose another method.")


def apply_denoise(
    arr: np.ndarray,
    params: DenoiseParams,
    pixel_size_nm: Optional[float] = None,
) -> np.ndarray:
    """
    Denoise *arr*, which is expected already normalised to roughly [0, 1].

    Never raises: an unavailable or failing method returns the input unchanged,
    because a denoiser is an aid and must not be able to stop an image being
    shown. The caller can check `is_available` to say something useful first.
    """
    if params is None or params.method in ("none", "", None):
        return arr
    method = _BY_KEY.get(params.method)
    if method is None or not is_available(params.method):
        return arr
    try:
        out = method.fn(np.asarray(arr, dtype=np.float32), params, pixel_size_nm)
        return np.asarray(out, dtype=np.float32)
    except Exception:
        import logging
        logging.getLogger(__name__).warning(
            "denoise %s failed; showing the image undenoised", params.method,
            exc_info=True)
        return arr
