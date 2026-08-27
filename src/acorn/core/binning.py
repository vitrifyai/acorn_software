"""Reduce an image by an integer factor, for analysis rather than for display.

The renderer already decimates: it picks a stride for whatever is on screen so
zooming reveals real detail. That is a drawing optimisation and nothing else --
segmentation backends, detectors and measurements all receive the full-resolution
array. This module is the other thing: a reduction of the data those consumers
actually see.

What binning does NOT do, despite the intuition, is make faint objects
detectable. Measured on a low-dose 42 kX micrograph of lipid nanoparticles,
matching a filter to the 30 nm particle scale in nanometres at every factor:

    bin   nm/px     detectability (particles / blank ice)
    1     0.2081    5.80
    2     0.4162    5.80
    4     0.8324    5.80
    8     1.6649    5.76

Detectability is flat. Per-pixel noise does fall as 1/factor, which is why a
binned image looks so much better, but a matched filter over the particle area
extracts exactly the same signal from the full-resolution data. Binning moves
information around; it does not add any.

What it is genuinely for:

    scale       Deep models resize to a fixed input anyway (SAM works at ~1024).
                Feeding a 4092 x 5760 array means something downsamples it --
                better an anti-aliased mean than whatever ad-hoc resize is
                buried in the backend.
    tractability 23.6 megapixels needs tiling for SAM and makes pixel-scale
                parameters in classical detectors (LoG radii, structuring
                elements) awkward to reason about.
    calibration Binning here rescales the pixel size in the same step, so a
                diameter in nanometres is unchanged by the choice.

That last point is the hard requirement. A 4x bin without rescaling makes every
distance, area and diameter wrong by a factor of four while still carrying
units, which is worse than having no calibration at all. Nothing in this module
returns a binned array without also returning what its pixel size became.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Powers of two only. Non-power-of-two factors are legal arithmetic but make the
# relationship to a detector's own hardware binning harder to reason about, and
# every real acquisition bins by 1, 2, 4 or 8.
VALID_FACTORS = (1, 2, 4, 8)


@dataclass(frozen=True)
class BinResult:
    """A binned array and everything that changed with it."""

    data:           np.ndarray
    pixel_size_nm:  float
    factor:         int
    cropped_px:     tuple[int, int]     # rows, columns discarded before binning

    @property
    def was_cropped(self) -> bool:
        return any(self.cropped_px)


def validate_factor(factor: int) -> int:
    """Coerce `factor` to a supported bin factor, or raise saying what is allowed."""
    try:
        f = int(factor)
    except (TypeError, ValueError):
        raise ValueError(f"bin factor must be an integer, got {factor!r}") from None
    if f not in VALID_FACTORS:
        raise ValueError(
            f"unsupported bin factor {f}. Supported: "
            + ", ".join(str(v) for v in VALID_FACTORS))
    return f


def _mean_bin(arr: np.ndarray, factor: int, ay: int) -> np.ndarray:
    """Mean-bin the adjacent spatial axes (`ay`, `ay + 1`) by `factor`.

    Both callers pass adjacent ascending axes -- (0, 1) for an image, (1, 2) for
    a movie stack -- so the reshape is a single split of each spatial axis into
    (blocks, factor) followed by averaging the two factor axes. Written for the
    cases that occur rather than in general: the general version needed enough
    index arithmetic to be worth getting wrong.

    Mean rather than sum: summing rescales intensity by factor^2 and breaks
    every contrast setting and threshold downstream. The mean leaves the signal
    level alone and reduces the noise, which is the entire point.
    """
    ax = ay + 1
    h, w = arr.shape[ay], arr.shape[ax]
    keep_y, keep_x = (h // factor) * factor, (w // factor) * factor

    sl = [slice(None)] * arr.ndim
    sl[ay], sl[ax] = slice(0, keep_y), slice(0, keep_x)
    arr = arr[tuple(sl)]

    shape = (arr.shape[:ay]
             + (keep_y // factor, factor, keep_x // factor, factor)
             + arr.shape[ax + 1:])
    return arr.reshape(shape).mean(axis=(ay + 1, ay + 3), dtype=np.float32)


def bin_image(arr: np.ndarray, factor: int, pixel_size_nm: float = 1.0) -> BinResult:
    """Bin a 2-D image, or an (H, W, 3) colour image, by `factor`.

    A trailing colour axis is left alone -- binning it would average red into
    green.
    """
    factor = validate_factor(factor)
    arr = np.asarray(arr)
    if arr.ndim not in (2, 3):
        raise ValueError(f"expected a 2-D or (H, W, 3) array, got shape {arr.shape}")
    if arr.ndim == 3 and arr.shape[-1] not in (3, 4):
        raise ValueError(
            f"3-D input must be (H, W, 3) or (H, W, 4) colour; got {arr.shape}. "
            "Use bin_frames() for an (N, H, W) movie stack.")

    if factor == 1:
        return BinResult(arr, float(pixel_size_nm), 1, (0, 0))

    h, w = arr.shape[0], arr.shape[1]
    out = _mean_bin(arr.astype(np.float32, copy=False), factor, 0)
    return BinResult(out, float(pixel_size_nm) * factor, factor,
                     (h % factor, w % factor))


def bin_frames(arr: np.ndarray, factor: int, pixel_size_nm: float = 1.0) -> BinResult:
    """Bin every frame of an (N, H, W) movie stack, leaving the frame axis alone."""
    factor = validate_factor(factor)
    arr = np.asarray(arr)
    if arr.ndim != 3:
        raise ValueError(f"expected an (N, H, W) stack, got shape {arr.shape}")

    if factor == 1:
        return BinResult(arr, float(pixel_size_nm), 1, (0, 0))

    h, w = arr.shape[1], arr.shape[2]
    out = _mean_bin(arr.astype(np.float32, copy=False), factor, 1)
    return BinResult(out, float(pixel_size_nm) * factor, factor,
                     (h % factor, w % factor))


def per_pixel_noise_gain(factor: int) -> float:
    """Factor by which per-pixel noise falls when binning by `factor`.

    Averaging f^2 independent pixels divides the noise by f. Deliberately NOT
    called an SNR gain: per-pixel noise is not detectability, and a filter
    matched to the object size recovers the same signal without binning. See
    the module docstring for the measurement.
    """
    return float(validate_factor(factor))


def describe(factor: int, pixel_size_nm: float | None = None) -> str:
    """One line for the UI saying what binning is doing to the data."""
    factor = validate_factor(factor)
    if factor == 1:
        return "No binning — analysis uses full resolution."
    parts = [f"{factor}x{factor} bin"]
    if pixel_size_nm:
        parts.append(f"pixel size {pixel_size_nm:.4g} nm")
    return (", ".join(parts)
            + ". Detection and measurements use the binned image; "
              "sizes stay in real units.")
