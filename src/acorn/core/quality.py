"""
Image quality assessment for cryo-EM micrographs.

Detects common acquisition problems before adding images to a training dataset:
  - Motion blur (beam-induced motion)
  - Thick ice / over-focus artefacts
  - Empty frames (no sample, air)
  - Carbon contamination (large dark patches)

Usage
-----
from acorn.core.quality import assess_quality

report = assess_quality(img.raw)
print(report.summary())
if not report.ok:
    print("Warnings:", report.warnings)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


# ── tunable thresholds ────────────────────────────────────────────────────────

# Laplacian variance: images below this are likely motion-blurred.
# Calibrated on typical 4096x4096 low-dose cryo-EM images (200–300 kV).
BLUR_THRESHOLD = 5.0

# Coefficient of variation (std/mean): images below this are near-uniform
# (thick ice, beam stop, or empty frame).
CV_THRESHOLD = 0.05

# Fraction of pixels that are saturated (at raw min or max).
# High saturation usually means detector issues or massive over-exposure.
SATURATION_THRESHOLD = 0.05

# Power ratio: fraction of FFT power in the very-low-frequency band
# (< 2% of Nyquist). Thick ice concentrates power at low frequencies.
LOW_FREQ_THRESHOLD = 0.70


# ── report dataclass ──────────────────────────────────────────────────────────

@dataclass
class QualityReport:
    """Quality assessment result for a single cryo-EM image."""

    # Scores (higher = better for blur/cv; lower = better for saturation/low_freq)
    blur_score:       float = 0.0   # Laplacian variance
    cv_score:         float = 0.0   # coefficient of variation
    saturation_frac:  float = 0.0   # fraction of saturated pixels
    low_freq_frac:    float = 0.0   # fraction of FFT power at low freq

    # Flags
    motion_blurred:   bool = False
    near_uniform:     bool = False  # thick ice or empty
    saturated:        bool = False
    low_freq_heavy:   bool = False  # thick ice signature in frequency domain

    # Human-readable warnings
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True if no quality issues were flagged."""
        return not (self.motion_blurred or self.near_uniform
                    or self.saturated or self.low_freq_heavy)

    def summary(self) -> str:
        lines = [
            f"Quality:  {'OK' if self.ok else 'WARNINGS'}",
            f"  Blur score  : {self.blur_score:.2f}  (threshold > {BLUR_THRESHOLD})",
            f"  CV          : {self.cv_score:.4f}  (threshold > {CV_THRESHOLD})",
            f"  Saturation  : {self.saturation_frac*100:.2f}%  (threshold < {SATURATION_THRESHOLD*100:.0f}%)",
            f"  Low-freq    : {self.low_freq_frac*100:.1f}%  (threshold < {LOW_FREQ_THRESHOLD*100:.0f}%)",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING: {w}")
        return "\n".join(lines)


# ── helpers ───────────────────────────────────────────────────────────────────

def _laplacian_variance(arr: np.ndarray) -> float:
    """Variance of the Laplacian — proxy for focus sharpness."""
    from scipy.ndimage import laplace
    lap = laplace(arr.astype(np.float32))
    return float(lap.var())


def _low_freq_power_fraction(arr: np.ndarray, low_frac: float = 0.02) -> float:
    """
    Fraction of total FFT power contained in the central low-frequency region.
    low_frac : radius as fraction of the smaller image dimension.
    """
    h, w = arr.shape[:2]
    f = np.fft.fft2(arr.astype(np.float32))
    power = np.abs(np.fft.fftshift(f)) ** 2
    total = power.sum()
    if total == 0:
        return 0.0
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * low_frac)
    radius = max(radius, 3)
    ys, xs = np.ogrid[:h, :w]
    mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= radius ** 2
    return float(power[mask].sum() / total)


# ── main function ─────────────────────────────────────────────────────────────

def assess_quality(
    raw: np.ndarray,
    blur_threshold:        float = BLUR_THRESHOLD,
    cv_threshold:          float = CV_THRESHOLD,
    saturation_threshold:  float = SATURATION_THRESHOLD,
    low_freq_threshold:    float = LOW_FREQ_THRESHOLD,
) -> QualityReport:
    """
    Assess image quality for a cryo-EM micrograph.

    Parameters
    ----------
    raw                  : 2-D float array (raw image data, any range)
    blur_threshold       : Laplacian variance below this flags motion blur
    cv_threshold         : CV below this flags near-uniform / thick-ice image
    saturation_threshold : fraction of saturated pixels above this flags problem
    low_freq_threshold   : low-frequency FFT power fraction above this flags thick ice

    Returns
    -------
    QualityReport
    """
    arr = raw.astype(np.float32)
    report = QualityReport()

    # ── blur ──────────────────────────────────────────────────────────────────
    try:
        report.blur_score = _laplacian_variance(arr)
    except Exception:
        report.blur_score = float("nan")

    if report.blur_score < blur_threshold:
        report.motion_blurred = True
        report.warnings.append(
            f"Possible motion blur (Laplacian variance {report.blur_score:.2f} < {blur_threshold})"
        )

    # ── near-uniform / empty ──────────────────────────────────────────────────
    rng = float(arr.max()) - float(arr.min())
    mean = float(arr.mean())
    std  = float(arr.std())
    report.cv_score = std / abs(mean) if abs(mean) > 1e-9 else 0.0

    if report.cv_score < cv_threshold or rng < 1e-6:
        report.near_uniform = True
        report.warnings.append(
            f"Near-uniform image (CV {report.cv_score:.4f} < {cv_threshold}) — "
            "possible empty frame or extremely thick ice"
        )

    # ── saturation ────────────────────────────────────────────────────────────
    vmin, vmax = float(arr.min()), float(arr.max())
    n_pixels = arr.size
    n_sat = int((arr == vmin).sum()) + int((arr == vmax).sum())
    report.saturation_frac = n_sat / max(n_pixels, 1)

    if report.saturation_frac > saturation_threshold:
        report.saturated = True
        report.warnings.append(
            f"High pixel saturation ({report.saturation_frac*100:.1f}% at extremes) — "
            "detector issue or severe over-exposure"
        )

    # ── low-frequency power ───────────────────────────────────────────────────
    try:
        report.low_freq_frac = _low_freq_power_fraction(arr)
    except Exception:
        report.low_freq_frac = 0.0

    if report.low_freq_frac > low_freq_threshold:
        report.low_freq_heavy = True
        report.warnings.append(
            f"Excessive low-frequency power ({report.low_freq_frac*100:.1f}%) — "
            "possible thick ice or large-scale contamination"
        )

    return report


# ── batch helper ──────────────────────────────────────────────────────────────

def batch_assess(
    images: list,  # list of DM4Image
    **kwargs,
) -> list[tuple]:
    """
    Assess quality for a list of DM4Image objects.

    Returns list of (dm4img, QualityReport) tuples, sorted worst-first.
    """
    results = []
    for img in images:
        try:
            report = assess_quality(img.raw, **kwargs)
        except Exception as exc:
            report = QualityReport(warnings=[f"Assessment failed: {exc}"])
        results.append((img, report))
    # Sort: images with warnings first, then by blur score ascending
    results.sort(key=lambda t: (t[1].ok, t[1].blur_score))
    return results
