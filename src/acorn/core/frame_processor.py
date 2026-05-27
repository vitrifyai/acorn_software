"""Frame averaging and motion correction for multi-frame cryo-EM movies."""
from __future__ import annotations

import numpy as np


def mean_average(frames: np.ndarray) -> np.ndarray:
    """Simple mean of all frames. (n, h, w) -> (h, w)."""
    return frames.mean(axis=0).astype(np.float32)


def motion_correct_frames(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Two-pass motion correction via phase cross-correlation.

    Returns
    -------
    averaged : (H, W) float32
        Mean of all aligned frames.
    shifts : (n_frames, 2) float32
        Total (dy, dx) offset applied to each frame to bring it into alignment.
        Negate these to get the sample drift trajectory relative to the aligned average.
    """
    from skimage.registration import phase_cross_correlation
    from scipy.ndimage import shift as nd_shift

    n = frames.shape[0]
    aligned = frames.astype(np.float32).copy()
    total_shifts = np.zeros((n, 2), dtype=np.float32)

    # Pass 1: align each frame to frame 0
    ref = aligned[0]
    for i in range(1, n):
        s, _, _ = phase_cross_correlation(ref, aligned[i], upsample_factor=10)
        total_shifts[i] += s
        aligned[i] = nd_shift(aligned[i], s)

    # Pass 2: align each frame to the pass-1 mean
    ref2 = aligned.mean(axis=0)
    for i in range(n):
        s, _, _ = phase_cross_correlation(ref2, aligned[i], upsample_factor=10)
        total_shifts[i] += s
        aligned[i] = nd_shift(aligned[i], s)

    return aligned.mean(axis=0).astype(np.float32), total_shifts


def motion_corrected_average(frames: np.ndarray) -> np.ndarray:
    """Convenience wrapper — returns the averaged image only."""
    averaged, _ = motion_correct_frames(frames)
    return averaged


def dose_series(
    frames: np.ndarray,
    n_bins: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """
    Split frames into n_bins equal groups and return per-bin averages.

    Returns
    -------
    averages : list of (H, W) float32 arrays, one per bin
    ranges   : list of (start, end) 0-based exclusive-end index pairs into frames
    """
    n = len(frames)
    n_bins = max(1, min(n_bins, n))
    averages: list[np.ndarray] = []
    ranges:   list[tuple[int, int]] = []
    for i in range(n_bins):
        s = int(round(i       * n / n_bins))
        e = int(round((i + 1) * n / n_bins))
        e = max(e, s + 1)
        averages.append(frames[s:e].mean(axis=0).astype(np.float32))
        ranges.append((s, e))
    return averages, ranges


def dose_weighted_average(
    frames: np.ndarray,
    dose_per_frame: float = 1.0,
    pixel_size_nm: float = 1.0,
) -> np.ndarray:
    """
    Dose-weighted average using the critical exposure formula from Grant &
    Grigorieff (2015, eLife). Each frame is weighted in Fourier space:
      w_i(q) = exp(-0.5 * cumulative_dose_i / Ne(q))
      Ne(q)  = 0.245 * q^(-1.665) + 2.81   [e/A^2]
    where q is spatial frequency in 1/Angstrom.
    """
    n, h, w = frames.shape
    pixel_size_A = max(pixel_size_nm * 10.0, 1e-6)

    fy = np.fft.fftfreq(h, d=pixel_size_A)
    fx = np.fft.fftfreq(w, d=pixel_size_A)
    FX, FY = np.meshgrid(fx, fy)
    q = np.clip(np.sqrt(FX**2 + FY**2), 1e-6, None)

    ne = 0.245 * np.power(q, -1.665) + 2.81

    fsum = np.zeros((h, w), dtype=np.complex128)
    wsum = np.zeros((h, w), dtype=np.float64)

    for i, frame in enumerate(frames):
        w_i = np.exp(-0.5 * (i + 1) * float(dose_per_frame) / ne)
        fsum += np.fft.fft2(frame.astype(np.float64)) * w_i
        wsum += w_i

    return np.real(np.fft.ifft2(fsum / np.maximum(wsum, 1e-12))).astype(np.float32)
