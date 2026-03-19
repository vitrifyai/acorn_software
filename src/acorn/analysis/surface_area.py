"""
ACORN Surface Area Estimation Module  (surface_area.py)
=========================================================
Estimates 3D surface area of particles from 2D instance segmentation masks
and calibrated pixel size (nm/px), or directly from 3D tomographic volumes.

2D projection methods (auto-selected from shape metrics):
  ellipsoid      -- fit ellipse, compute prolate/oblate spheroid SA analytically
  cauchy         -- convex hull + Cauchy-Crofton theorem with aspect-ratio k
  fourier        -- radial Fourier decomposition including roughness contribution
  fourier_spiky  -- fourier + cone-modeled spikes for fractal-dimension particles
  capsule        -- fit minimal bounding rect; model as spherocylinder (2 hemicaps + cylinder)
  perimeter      -- isoperimetric SA = P²/(π·circularity) from projected 2D perimeter
  monte_carlo    -- Cauchy-Crofton random test line perimeter → SA via isoperimetric formula
  richardson     -- multiscale Richardson plot perimeter extrapolation

3D volume methods (ground-truth tier for cryo-ET / tomographic data):
  marching_cubes -- triangulated isosurface via marching cubes (Lorensen & Cline 1987)
                    Requires: scikit-image
  poisson        -- smooth watertight mesh via Poisson surface reconstruction
                    (Kazhdan & Hoppe 2013); more robust for low-SNR volumes
                    Requires: open3d, scipy

Additional diagnostics computed for every particle (all 2D methods):
  perimeter_nm           -- 2D projected perimeter (Richardson extrapolation to pixel scale)
  lacunarity             -- gliding-box lacunarity at r=10 px (texture heterogeneity)
  richardson_fractal_dim -- fractal dimension from Richardson plot slope

Auto-selection logic (2D)
--------------------------
  fractal_dim > 1.15                          -> fourier_spiky
  aspect_ratio > 2.5 AND convexity > 0.85    -> capsule
  circularity > 0.75 AND convexity > 0.85    -> ellipsoid
  convexity > 0.85                            -> cauchy
  else                                        -> fourier

Batch processing uses multiprocessing distributed across available GPUs (or
CPU cores when no CUDA device is present).

Requirements (all optional; graceful ImportError if missing):
  opencv-python, scipy, torch, scikit-image, open3d
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict, dataclass
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# Fractal-dimension threshold that triggers fourier_spiky mode
_FRACTAL_DIM_THRESHOLD = 1.15


# ── result dataclass ──────────────────────────────────────────────────────────

@dataclass
class SurfaceAreaResult:
    """Per-particle surface area estimate and metadata."""

    particle_id: int = 0
    a_nm: float = 0.0                        # semi-major axis (nm)
    b_nm: float = 0.0                        # semi-minor axis (nm)
    SA_nm2: float = 0.0                      # estimated 3D surface area (nm²)
    SA_nm2_uncertainty: float = 0.0          # propagated uncertainty (nm²)
    coverage_score: float = 1.0             # mask area / fitted ellipse area
    method_used: str = "unknown"
    flagged: bool = False
    flag_reason: str = ""
    is_hollow: bool = False
    shell_thickness_estimate_nm: float = 0.0
    SA_outer_nm2: float = 0.0   # outer surface only (always populated when hollow)
    SA_inner_nm2: float = 0.0   # inner surface (populated when shell thickness known)
    aggregate_score: float = 0.0            # 0 = clean, 1 = likely aggregate
    sem_roughness_index: float = 0.0        # std(r)/mean(r) of radial profile
    perimeter_nm: float = 0.0               # 2D projected perimeter (nm, Richardson scale-1)
    lacunarity: float = 0.0                 # gliding-box lacunarity at r=10 px
    richardson_fractal_dim: float = 0.0     # fractal dimension from Richardson plot


# ── device resolution ─────────────────────────────────────────────────────────

def _resolve_device(device: str) -> str:
    """Return a canonical device string ('cpu', 'cuda', 'cuda:N')."""
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return device


def _available_gpu_ids(n_gpus: Optional[int] = None) -> list[int]:
    """Return list of GPU indices to use; falls back to [] for CPU."""
    try:
        import torch
        total = torch.cuda.device_count()
        if total == 0:
            return []
        ids = list(range(total))
        if n_gpus is not None:
            ids = ids[: n_gpus]
        return ids
    except ImportError:
        return []


# ── shape metrics ─────────────────────────────────────────────────────────────

def _shape_metrics(mask_u8: np.ndarray):
    """
    Compute circularity, convexity (solidity), and fractal dimension.

    Returns (circularity, convexity, fractal_dim, largest_contour, all_contours).
    """
    import cv2

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0, 0.0, 1.0, None, [], 1.0

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area < 4 or perimeter < 4:
        return 0.0, 0.0, 1.0, cnt, contours, 1.0

    # circularity = 4π·area / perimeter²  (1.0 = perfect circle)
    circularity = float(min(1.0, 4 * np.pi * area / perimeter ** 2))

    # convexity (solidity) = mask area / convex hull area
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    convexity = float(area / hull_area) if hull_area > 0 else 0.0

    fractal_dim = _box_counting_fractal(mask_u8, cnt)

    x, y, w, h = cv2.boundingRect(cnt) if cnt is not None else (0, 0, 1, 1)
    aspect_ratio = max(w, h) / max(min(w, h), 1.0)
    return circularity, convexity, fractal_dim, cnt, contours, aspect_ratio


def _box_counting_fractal(mask_u8: np.ndarray, contour) -> float:
    """Estimate fractal dimension of a contour via box-counting."""
    import cv2

    h, w = mask_u8.shape
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(canvas, [contour], -1, 1, 1)

    ys, xs = np.where(canvas > 0)
    if len(ys) < 10:
        return 1.0

    sizes, counts = [], []
    box_size = max(2, min(h, w) // 2)
    while box_size >= 2:
        n_boxes = len(set(zip((ys // box_size).tolist(), (xs // box_size).tolist())))
        sizes.append(box_size)
        counts.append(n_boxes)
        box_size //= 2

    if len(sizes) < 3:
        return 1.0

    log_s = np.log(1.0 / np.array(sizes, dtype=float))
    log_n = np.log(np.array(counts, dtype=float))
    coeffs = np.polyfit(log_s, log_n, 1)
    return float(np.clip(coeffs[0], 1.0, 2.0))


# ── coverage / occlusion ──────────────────────────────────────────────────────

def _coverage_score(mask_u8: np.ndarray, contour) -> float:
    """Ratio of mask area to fitted ellipse area; values < 0.6 indicate occlusion."""
    import cv2

    if contour is None or len(contour) < 5:
        return 1.0

    mask_area = float(np.sum(mask_u8 > 0))
    try:
        _, (ma, mb), _ = cv2.fitEllipse(contour)
        ellipse_area = np.pi * (ma / 2) * (mb / 2)
        if ellipse_area <= 0:
            return 1.0
        return float(min(1.0, mask_area / ellipse_area))
    except Exception:
        return 1.0


def _complete_occluded_contour(contour, mask_u8: np.ndarray):
    """
    Use low-frequency Fourier interpolation to estimate the missing portion
    of a partially occluded contour.
    """
    if contour is None or len(contour) < 10:
        return contour

    pts = contour[:, 0, :].astype(float)
    cx, cy = pts.mean(axis=0)
    # represent in complex plane relative to centroid
    pts_c = (pts[:, 0] - cx) + 1j * (pts[:, 1] - cy)

    F = np.fft.fft(pts_c)
    n_keep = max(5, len(F) // 8)
    F_lp = np.zeros_like(F)
    F_lp[:n_keep] = F[:n_keep]
    F_lp[-n_keep:] = F[-n_keep:]

    smooth = np.fft.ifft(F_lp)
    x_out = smooth.real + cx
    y_out = smooth.imag + cy
    pts_out = np.stack([x_out, y_out], axis=1).reshape(-1, 1, 2).astype(np.float32)
    return pts_out


# ── aggregate detection ───────────────────────────────────────────────────────

def _aggregate_score(contour) -> float:
    """
    Score 0–1 for likelihood of being an aggregate (fused particles).
    Uses number of deep convexity defects relative to particle size.
    """
    import cv2

    if contour is None or len(contour) < 5:
        return 0.0

    hull_idx = cv2.convexHull(contour, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return 0.0

    try:
        defects = cv2.convexityDefects(contour, hull_idx)
    except Exception:
        return 0.0

    if defects is None:
        return 0.0

    depths_px = defects[:, 0, 3].astype(float) / 256.0
    area = cv2.contourArea(contour)
    threshold = 0.1 * np.sqrt(max(area, 1.0))
    deep = int(np.sum(depths_px > threshold))
    return float(min(1.0, deep / 5.0))


# ── spheroid SA helper (used for hollow inner-surface correction) ─────────────

def _spheroid_sa_nm(a_nm: float, b_nm: float) -> float:
    """Analytical SA of a prolate or oblate spheroid with semi-axes a and b (b=b=minor)."""
    if a_nm <= 0 or b_nm <= 0:
        return 0.0
    if abs(a_nm - b_nm) < 0.01 * max(a_nm, b_nm):
        r = (a_nm + b_nm) / 2.0
        return 4.0 * np.pi * r * r
    if a_nm >= b_nm:   # prolate
        e = np.sqrt(1.0 - (b_nm / a_nm) ** 2)
        return float(2.0 * np.pi * b_nm * b_nm * (1.0 + (a_nm / (b_nm * e)) * np.arcsin(e)))
    else:              # oblate
        e = np.sqrt(1.0 - (a_nm / b_nm) ** 2)
        return float(2.0 * np.pi * b_nm * b_nm * (1.0 + ((1.0 - e * e) / e) * np.arctanh(e)))


# ── hollow detection ──────────────────────────────────────────────────────────

def _detect_hollow(mask_u8: np.ndarray, raw_image: Optional[np.ndarray] = None) -> tuple[bool, float]:
    """
    Detect hollow or shell-like particles.

    Without intensity data: checks for donut-shaped mask (interior holes).
    With intensity data: looks for a brighter centre relative to the edge
    (TEM bright-field convention where hollow interior appears bright).

    Returns (is_hollow, shell_thickness_px).
    """
    from scipy import ndimage

    if raw_image is None:
        filled = ndimage.binary_fill_holes(mask_u8 > 0)
        interior = filled.astype(float) - (mask_u8 > 0).astype(float)
        interior_frac = np.sum(interior) / max(float(np.sum(filled)), 1.0)
        if interior_frac > 0.1:
            eroded = mask_u8 > 0
            for t in range(1, 100):
                eroded = ndimage.binary_erosion(eroded)
                if not np.any(eroded):
                    return True, float(t)
            return True, 0.0
        return False, 0.0

    ys, xs = np.where(mask_u8 > 0)
    if len(ys) < 10:
        return False, 0.0

    cy, cx = ys.mean(), xs.mean()
    r_all = np.sqrt((ys - cy) ** 2 + (xs - cx) ** 2)
    r_max = r_all.max()
    if r_max < 5:
        return False, 0.0

    n_bins = 20
    r_edges = np.linspace(0, r_max, n_bins + 1)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        in_ring = (r_all >= r_edges[i]) & (r_all < r_edges[i + 1])
        if in_ring.any():
            profile[i] = float(raw_image[ys[in_ring], xs[in_ring]].mean())

    center_mean = profile[: n_bins // 4].mean()
    edge_mean = profile[n_bins // 2 :].mean()

    if center_mean > edge_mean * 1.3:
        shell_px = 0.0
        for i in range(n_bins - 1, 0, -1):
            if profile[i] < edge_mean * 0.9:
                shell_px = r_max * (1.0 - i / n_bins)
                break
        return True, shell_px

    return False, 0.0


# ── SEM surface roughness ─────────────────────────────────────────────────────

def _sem_roughness_index(contour) -> float:
    """std(r) / mean(r) of the radial contour profile (0 = perfectly smooth)."""
    if contour is None or len(contour) < 20:
        return 0.0
    pts = contour[:, 0, :].astype(float)
    cx, cy = pts.mean(axis=0)
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    mean_r = r.mean()
    if mean_r < 1e-6:
        return 0.0
    return float(r.std() / mean_r)


# ── estimation methods ────────────────────────────────────────────────────────

def _estimate_ellipsoid(
    contour, pixel_size_nm: float
) -> tuple[float, float, float]:
    """
    Fit an ellipse and compute prolate or oblate spheroid surface area.

    Returns (a_nm, b_nm, SA_nm2).
    """
    import cv2

    if contour is None or len(contour) < 5:
        return 0.0, 0.0, 0.0

    try:
        _, (ma_px, mb_px), _ = cv2.fitEllipse(contour)
    except Exception:
        return 0.0, 0.0, 0.0

    a_nm = max(ma_px, mb_px) / 2.0 * pixel_size_nm  # semi-major
    b_nm = min(ma_px, mb_px) / 2.0 * pixel_size_nm  # semi-minor

    if a_nm <= 0 or b_nm <= 0:
        return a_nm, b_nm, 0.0

    if abs(a_nm - b_nm) < 1e-9:
        sa = 4.0 * np.pi * a_nm ** 2
    elif a_nm > b_nm:
        # prolate spheroid: SA = 2πb²(1 + (a / (b·e)) · arcsin(e))
        e = np.sqrt(max(0.0, 1.0 - (b_nm / a_nm) ** 2))
        if e < 1e-9:
            sa = 4.0 * np.pi * a_nm ** 2
        else:
            sa = 2.0 * np.pi * b_nm ** 2 * (1.0 + (a_nm / (b_nm * e)) * np.arcsin(e))
    else:
        # oblate spheroid: SA = 2πa²(1 + ((1 - e²)/e) · arctanh(e))
        e = np.sqrt(max(0.0, 1.0 - (a_nm / b_nm) ** 2))
        if e < 1e-9:
            sa = 4.0 * np.pi * b_nm ** 2
        else:
            sa = 2.0 * np.pi * b_nm ** 2 * (1.0 + ((1.0 - e ** 2) / e) * np.arctanh(e))

    return a_nm, b_nm, float(sa)


def _estimate_cauchy(
    contour, pixel_size_nm: float
) -> tuple[float, float, float]:
    """
    Cauchy-Crofton: SA ≈ k × A_projected, where k is corrected for aspect ratio.

    For a sphere k = 4; elongated particles get a slightly larger k.
    Returns (a_nm, b_nm, SA_nm2).
    """
    import cv2

    if contour is None:
        return 0.0, 0.0, 0.0

    area_px = cv2.contourArea(contour)
    area_nm2 = area_px * pixel_size_nm ** 2

    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(min(w, h), 1.0)
    # k grows slowly with aspect ratio; empirical power-law fit
    k = 4.0 + 0.5 * (aspect - 1.0) ** 0.7

    a_nm = max(w, h) / 2.0 * pixel_size_nm
    b_nm = min(w, h) / 2.0 * pixel_size_nm

    return a_nm, b_nm, float(k * area_nm2)


def _estimate_fourier(
    contour, pixel_size_nm: float
) -> tuple[float, float, float, float]:
    """
    Decompose the radial contour signal with FFT.

    DC component → mean radius → approximate sphere.
    1st harmonic → ellipticity → prolate spheroid correction.
    Higher harmonics → surface roughness multiplier.

    Returns (a_nm, b_nm, SA_nm2, roughness_index).
    """
    if contour is None or len(contour) < 20:
        return 0.0, 0.0, 0.0, 0.0

    pts = contour[:, 0, :].astype(float)
    cx, cy = pts.mean(axis=0)
    r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

    if r.mean() < 1e-6:
        return 0.0, 0.0, 0.0, 0.0

    F = np.fft.rfft(r)
    amplitudes = np.abs(F) / len(r)

    R0 = amplitudes[0]                                        # mean radius
    R1 = float(amplitudes[1]) if len(amplitudes) > 1 else 0.0  # ellipticity

    a_px = R0 + R1
    b_px = max(R0 - R1, 1e-6)
    a_nm = a_px * pixel_size_nm
    b_nm = b_px * pixel_size_nm

    # base spheroid SA
    if abs(a_nm - b_nm) < 1e-9:
        sa_base = 4.0 * np.pi * a_nm ** 2
    elif a_nm > b_nm:
        e = np.sqrt(max(0.0, 1.0 - (b_nm / a_nm) ** 2))
        if e < 1e-9:
            sa_base = 4.0 * np.pi * a_nm ** 2
        else:
            sa_base = 2.0 * np.pi * b_nm ** 2 * (1.0 + (a_nm / (b_nm * e)) * np.arcsin(e))
    else:
        sa_base = 4.0 * np.pi * a_nm ** 2

    # roughness contribution: power in harmonics n >= 2
    high_freq_power = float(np.sum(amplitudes[2:] ** 2)) if len(amplitudes) > 2 else 0.0
    roughness_index = float(np.sqrt(high_freq_power) / max(amplitudes[0], 1e-6))
    sa = sa_base * (1.0 + 2.0 * roughness_index)

    return a_nm, b_nm, float(sa), roughness_index


def _estimate_fourier_spiky(
    contour, pixel_size_nm: float
) -> tuple[float, float, float, float]:
    """
    Fourier base + cone-modelled spikes for fractal-boundary particles.

    Spike tips are detected as local curvature maxima on the contour.
    Each spike is modelled as a cone: SA_lateral = π·r·L.
    Total spike count is estimated via N_total ≈ N_visible × (4πR² / perimeter).

    Returns (a_nm, b_nm, SA_nm2, roughness_index).
    """
    import cv2
    from scipy.signal import find_peaks

    a_nm, b_nm, sa_base, roughness = _estimate_fourier(contour, pixel_size_nm)
    if contour is None or len(contour) < 20:
        return a_nm, b_nm, sa_base, roughness

    pts = contour[:, 0, :].astype(float)
    n = len(pts)

    # discrete curvature at each contour point
    curvatures = np.zeros(n)
    for i in range(n):
        p0 = pts[(i - 2) % n]
        p1 = pts[i]
        p2 = pts[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        mag = (np.linalg.norm(v1) * np.linalg.norm(v2)) ** 1.5
        curvatures[i] = cross / max(mag, 1e-9)

    threshold = float(np.percentile(curvatures, 80))
    peaks, _ = find_peaks(curvatures, height=threshold, distance=5)

    if len(peaks) == 0:
        return a_nm, b_nm, sa_base, roughness

    cx, cy = pts.mean(axis=0)
    r_all = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    r_mean = r_all.mean()

    total_visible_spike_sa = 0.0
    for pk in peaks:
        # find left/right extent of this spike
        lo, hi = pk, pk
        level = curvatures[pk] * 0.3
        while lo > 0 and curvatures[lo] > level:
            lo -= 1
        while hi < n - 1 and curvatures[hi] > level:
            hi += 1

        w_px = float(np.linalg.norm(pts[lo] - pts[hi]))
        h_px = float(max(0.0, r_all[pk] - r_mean))

        w_nm = w_px * pixel_size_nm
        h_nm = h_px * pixel_size_nm

        if w_nm > 0 and h_nm > 0:
            r_cone = w_nm / 2.0
            slant = np.sqrt(h_nm ** 2 + r_cone ** 2)
            total_visible_spike_sa += np.pi * r_cone * slant

    # scale visible spike SA to full sphere surface
    r_mean_nm = r_mean * pixel_size_nm
    perimeter_nm = cv2.arcLength(contour, True) * pixel_size_nm
    if perimeter_nm > 0 and r_mean_nm > 0:
        sphere_sa = 4.0 * np.pi * r_mean_nm ** 2
        scale = sphere_sa / perimeter_nm
    else:
        scale = float(len(peaks))

    sa_total = sa_base + total_visible_spike_sa * scale
    return a_nm, b_nm, float(sa_total), roughness


# ── capsule (spherocylinder) ──────────────────────────────────────────────────

def _estimate_capsule(
    contour, pixel_size_nm: float
) -> tuple[float, float, float]:
    """
    Fit a minimal-area bounding rectangle; model particle as a spherocylinder.

    SA = 2πr(2r + h)  where r = cylinder radius, h = cylinder body length.
    Reduces to sphere SA = 4πr² when h = 0.

    Returns (a_nm, b_nm, SA_nm2).
    """
    import cv2

    if contour is None or len(contour) < 5:
        return 0.0, 0.0, 0.0

    _, (w_px, h_px), _ = cv2.minAreaRect(contour)
    L_px = max(w_px, h_px)
    r_px = min(w_px, h_px) / 2.0
    h_cyl_px = max(0.0, L_px - 2.0 * r_px)

    r_nm = r_px * pixel_size_nm
    h_nm = h_cyl_px * pixel_size_nm
    a_nm = (L_px / 2.0) * pixel_size_nm
    b_nm = r_nm

    sa = 2.0 * np.pi * r_nm * (2.0 * r_nm + h_nm)
    return a_nm, b_nm, float(sa)


# ── perimeter-based isoperimetric ─────────────────────────────────────────────

def _estimate_perimeter_cauchy(
    contour, pixel_size_nm: float, circularity: float = 1.0
) -> tuple[float, float, float]:
    """
    Isoperimetric SA estimate from the 2D projected perimeter.

    For an isotropic random projection of a 3D body, the projected perimeter P
    and SA are related by the isoperimetric inequality:
        SA = P² / (π · circularity_2d)
    which recovers SA = P²/π for a sphere (circularity = 1) and gives larger
    values for elongated or rough projections — capturing surface complexity
    that area-based Cauchy misses.

    Returns (a_nm, b_nm, SA_nm2).
    """
    import cv2

    if contour is None:
        return 0.0, 0.0, 0.0

    perim_nm = cv2.arcLength(contour, True) * pixel_size_nm
    circ = max(0.05, min(1.0, circularity))
    sa = perim_nm ** 2 / (np.pi * circ)

    x, y, w, h = cv2.boundingRect(contour)
    a_nm = max(w, h) / 2.0 * pixel_size_nm
    b_nm = min(w, h) / 2.0 * pixel_size_nm
    return a_nm, b_nm, float(sa)


# ── Cauchy-Crofton random test line perimeter ─────────────────────────────────

def _monte_carlo_perimeter(mask_u8: np.ndarray, n_probes: int = 2000) -> float:
    """
    Estimate 2D contour perimeter via Cauchy-Crofton random test lines.

    Cast n_probes random lines parameterised by (θ, t) with θ ~ Uniform[0, π)
    and t ~ Uniform[-diag/2, diag/2].  Count boundary crossings (mask
    transitions) along each rasterised line.

    Cauchy-Crofton formula:
        P = (π · diag) / (2 · N) × Σ n_i

    Handles rough/fractal boundaries more accurately than polygon
    approximation because it samples the boundary at sub-pixel spacing.

    Returns perimeter in pixels.
    """
    rng = np.random.default_rng(42)
    h, w = mask_u8.shape
    m = (mask_u8 > 0).astype(np.uint8)
    diag = float(np.sqrt(h ** 2 + w ** 2))
    cx, cy = w / 2.0, h / 2.0

    total_crossings = 0
    angles = rng.uniform(0.0, np.pi, n_probes)
    offsets = rng.uniform(-diag / 2.0, diag / 2.0, n_probes)

    n_steps = int(diag) + 2

    for theta, t in zip(angles, offsets):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        s_vals = np.linspace(-diag / 2.0, diag / 2.0, n_steps)
        xs = np.round(cx + s_vals * cos_t - t * sin_t).astype(int)
        ys = np.round(cy + s_vals * sin_t + t * cos_t).astype(int)

        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs_v, ys_v = xs[valid], ys[valid]
        if len(xs_v) < 2:
            continue

        vals = m[ys_v, xs_v]
        total_crossings += int(np.sum(np.abs(np.diff(vals.astype(np.int8)))))

    if n_probes == 0:
        return 0.0
    return float(np.pi * diag * total_crossings / (2.0 * n_probes))


# ── Richardson plot multiscale perimeter ──────────────────────────────────────

def _richardson_perimeter(
    contour, pixel_size_nm: float
) -> tuple[float, float]:
    """
    Measure contour perimeter at multiple step sizes and fit a Richardson plot.

    At each step size ε (px), subsample the contour and compute the polygon
    perimeter.  Fit: log P(ε) = (1 - D_f) · log ε + const.

    Returns (perimeter_at_pixel_scale_nm, fractal_dim).
    perimeter_at_pixel_scale_nm is the extrapolated perimeter at ε = 1 px.
    """
    import cv2

    if contour is None or len(contour) < 20:
        perim_px = cv2.arcLength(contour, True) if contour is not None else 0.0
        return float(perim_px * pixel_size_nm), 1.0

    pts = contour[:, 0, :].astype(float)
    n = len(pts)
    scales, perimeters = [], []

    step = 1
    while step <= max(2, n // 4):
        idx = np.arange(0, n, step)
        if len(idx) < 3:
            break
        sub = pts[idx]
        diffs = np.diff(sub, axis=0)
        seg_len = np.sqrt((diffs ** 2).sum(axis=1)).sum()
        close_len = float(np.sqrt(((sub[-1] - sub[0]) ** 2).sum()))
        scales.append(float(step))
        perimeters.append(seg_len + close_len)
        step *= 2

    if len(scales) < 3:
        perim_px = float(cv2.arcLength(contour, True))
        return float(perim_px * pixel_size_nm), 1.0

    log_s = np.log(np.array(scales, dtype=float))
    log_p = np.log(np.maximum(np.array(perimeters, dtype=float), 1e-9))
    coeffs = np.polyfit(log_s, log_p, 1)

    fractal_dim = float(np.clip(1.0 - coeffs[0], 1.0, 2.0))
    P_at_1 = float(np.exp(np.polyval(coeffs, 0.0)))   # extrapolate to log(ε)=0 → ε=1
    return float(P_at_1 * pixel_size_nm), fractal_dim


# ── gliding-box lacunarity ─────────────────────────────────────────────────────

def _lacunarity(mask_u8: np.ndarray, r: int = 10) -> float:
    """
    Gliding-box lacunarity at scale r.

    Slide a square box of side r across the binary mask; record the mass S
    (count of filled pixels) in each box position.

    Λ(r) = <S²> / <S>² − 1

    Returns 0 for a uniform mask (no texture), higher values for clustered
    or heterogeneous textures.  A lacunarity of 0 means the filled pixels
    are perfectly uniformly distributed.
    """
    h, w = mask_u8.shape
    r = max(2, min(r, h // 2, w // 2))
    if h < r or w < r:
        return 0.0

    m = (mask_u8 > 0).astype(np.float64)
    cs = np.cumsum(np.cumsum(m, axis=0), axis=1)
    cs_pad = np.pad(cs, ((1, 0), (1, 0)), mode="constant")

    i_max = h - r + 1
    j_max = w - r + 1
    if i_max <= 0 or j_max <= 0:
        return 0.0

    sums = (
        cs_pad[r : r + i_max, r : r + j_max]
        - cs_pad[0:i_max, r : r + j_max]
        - cs_pad[r : r + i_max, 0:j_max]
        + cs_pad[0:i_max, 0:j_max]
    )

    s = sums.ravel()
    z1 = float(s.mean())
    if z1 < 1e-9:
        return 0.0
    z2 = float((s ** 2).mean())
    return float(z2 / (z1 ** 2) - 1.0)


# ── 3D volume methods ─────────────────────────────────────────────────────────

def _marching_cubes_sa(
    volume: np.ndarray,
    voxel_size_nm: float,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute surface area of a 3D object via marching cubes isosurface extraction.

    Fits a triangulated mesh to the iso-surface of the volume and sums
    triangle areas.  When voxel_size_nm is the calibrated voxel spacing, the
    result is in nm².

    Requires: scikit-image (skimage.measure.marching_cubes).
    """
    from skimage.measure import marching_cubes

    if threshold is None:
        threshold = float(volume.min() + volume.max()) / 2.0

    verts, faces, _, _ = marching_cubes(
        volume,
        level=threshold,
        spacing=(voxel_size_nm, voxel_size_nm, voxel_size_nm),
    )
    if len(faces) == 0:
        return 0.0

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.sqrt((cross ** 2).sum(axis=1))
    return float(areas.sum())


def _poisson_sa(
    volume: np.ndarray,
    voxel_size_nm: float,
    threshold: Optional[float] = None,
) -> float:
    """
    Compute surface area via Poisson surface reconstruction.

    Extracts the binary surface shell as a point cloud with outward normals
    derived from the density gradient, then fits a smooth watertight mesh via
    screened Poisson reconstruction (Kazhdan & Hoppe 2013).

    More robust than marching cubes for low-SNR cryo-EM volumes because the
    Poisson solver is insensitive to noise in individual surface normals.

    Requires: open3d, scipy.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for Poisson surface reconstruction. "
            "Install with: pip install open3d"
        )
    from scipy import ndimage

    if threshold is None:
        threshold = float(volume.min() + volume.max()) / 2.0

    binary = volume > threshold
    if not binary.any():
        return 0.0

    # surface shell: foreground voxels adjacent to background
    eroded = ndimage.binary_erosion(binary)
    surface = binary & ~eroded

    coords = np.argwhere(surface).astype(float) * voxel_size_nm  # (N, 3) in nm
    if len(coords) < 10:
        return 0.0

    # outward normals from density gradient at surface voxels
    gz, gy, gx = np.gradient(volume.astype(np.float64))
    normals = np.stack([gz[surface], gy[surface], gx[surface]], axis=1)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms < 1e-9] = 1.0
    normals = normals / norms

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    return float(mesh.get_surface_area())


def estimate_surface_area_3d(
    volume: np.ndarray,
    voxel_size_nm: float,
    method: str = "marching_cubes",
    threshold: Optional[float] = None,
    particle_id: int = 0,
) -> "SurfaceAreaResult":
    """
    Estimate 3D surface area directly from a 3D binary or density volume.

    This is the ground-truth tier for cryo-ET / tomographic data — no
    projection assumptions are needed when the full 3D density is available.

    Parameters
    ----------
    volume : 3D ndarray
        Tomographic reconstruction or segmentation volume.  For binary masks
        pass a 0/1 array; for density maps the midpoint threshold is used.
    voxel_size_nm : isotropic voxel spacing in nm/voxel
    method : 'marching_cubes' | 'poisson'
        marching_cubes -- triangulated isosurface (scikit-image required)
        poisson        -- smooth watertight mesh (open3d + scipy required;
                          more robust for noisy volumes)
    threshold : float or None
        Iso-surface level.  None = (min + max) / 2.
    particle_id : int

    Returns
    -------
    SurfaceAreaResult
        SA_nm2 is the volume-derived estimate.  2D diagnostics (perimeter_nm,
        lacunarity, richardson_fractal_dim) are computed from the z-middle
        slice for cross-method comparability.
    """
    if volume.ndim != 3:
        raise ValueError(f"volume must be 3D; got shape {volume.shape}")

    result = SurfaceAreaResult(particle_id=particle_id)
    result.method_used = method

    nz, ny, nx = volume.shape

    # 2D diagnostics from the z-middle slice for comparability
    thr = threshold if threshold is not None else float(volume.min() + volume.max()) / 2.0
    mid_slice = (volume[nz // 2] > thr).astype(np.uint8)
    if mid_slice.sum() > 4:
        try:
            circularity, convexity, fractal_dim, cnt, _, aspect_ratio = _shape_metrics(mid_slice)
            result.perimeter_nm, result.richardson_fractal_dim = _richardson_perimeter(
                cnt, voxel_size_nm
            )
            lac_r = max(2, min(10, mid_slice.shape[0] // 3, mid_slice.shape[1] // 3))
            result.lacunarity = _lacunarity(mid_slice, r=lac_r)
            result.a_nm = max(nx, ny) / 2.0 * voxel_size_nm
            result.b_nm = min(nx, ny) / 2.0 * voxel_size_nm
        except Exception:
            pass

    if method == "marching_cubes":
        result.SA_nm2 = _marching_cubes_sa(volume, voxel_size_nm, threshold)
    elif method == "poisson":
        result.SA_nm2 = _poisson_sa(volume, voxel_size_nm, threshold)
    else:
        raise ValueError(
            f"Unknown 3D method {method!r}. Choose from 'marching_cubes', 'poisson'."
        )

    return result


# ── main single-particle entry point ─────────────────────────────────────────

def estimate_surface_area(
    mask: np.ndarray,
    pixel_size_nm: float,
    pixel_size_uncertainty_nm: float = 0.0,
    method: str = "auto",
    device: str = "auto",
    raw_image: Optional[np.ndarray] = None,
    particle_id: int = 0,
) -> SurfaceAreaResult:
    """
    Estimate 3D surface area of a particle from its 2D binary mask.

    Parameters
    ----------
    mask : 2D ndarray, binary (True/1 = particle region)
    pixel_size_nm : physical size of one pixel in nm/px
    pixel_size_uncertainty_nm : absolute uncertainty in pixel_size_nm
    method : 'auto' | 'ellipsoid' | 'cauchy' | 'fourier' | 'fourier_spiky'
    device : 'auto' | 'cpu' | 'cuda' | 'cuda:N'
        Passed through to batch processing; single-mask estimation always
        runs on CPU-backed NumPy/OpenCV.
    raw_image : optional grayscale float array (same H×W as mask) for
        hollow-particle detection via radial intensity profile
    particle_id : integer label written to the result

    Returns
    -------
    SurfaceAreaResult
    """
    result = SurfaceAreaResult(particle_id=particle_id)

    mask_u8 = (mask > 0).astype(np.uint8)
    if mask_u8.sum() < 4:
        result.flagged = True
        result.flag_reason = "mask_too_small"
        return result

    # shape metrics
    circularity, convexity, fractal_dim, cnt, _, aspect_ratio = _shape_metrics(mask_u8)

    # occlusion
    coverage = _coverage_score(mask_u8, cnt)
    result.coverage_score = coverage
    if coverage < 0.6:
        cnt = _complete_occluded_contour(cnt, mask_u8)
        result.flagged = True
        result.flag_reason = "partially_occluded"

    # aggregate detection
    agg = _aggregate_score(cnt)
    result.aggregate_score = agg
    if agg > 0.5:
        result.flagged = True
        result.flag_reason = ";".join(filter(None, [result.flag_reason, "aggregate"]))

    # hollow detection
    is_hollow, shell_t_px = _detect_hollow(mask_u8, raw_image)
    result.is_hollow = is_hollow
    result.shell_thickness_estimate_nm = shell_t_px * pixel_size_nm

    # method selection
    if method == "auto":
        if fractal_dim > _FRACTAL_DIM_THRESHOLD:
            method = "fourier_spiky"
        elif aspect_ratio > 2.5 and convexity > 0.85:
            method = "capsule"
        elif circularity > 0.75 and convexity > 0.85:
            method = "ellipsoid"
        elif convexity > 0.85:
            method = "cauchy"
        else:
            method = "fourier"

    result.method_used = method

    if method == "ellipsoid":
        a, b, sa = _estimate_ellipsoid(cnt, pixel_size_nm)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = _sem_roughness_index(cnt)

    elif method == "cauchy":
        a, b, sa = _estimate_cauchy(cnt, pixel_size_nm)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = _sem_roughness_index(cnt)

    elif method == "fourier":
        a, b, sa, roughness = _estimate_fourier(cnt, pixel_size_nm)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = roughness

    elif method == "fourier_spiky":
        a, b, sa, roughness = _estimate_fourier_spiky(cnt, pixel_size_nm)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = roughness

    elif method == "capsule":
        a, b, sa = _estimate_capsule(cnt, pixel_size_nm)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = _sem_roughness_index(cnt)

    elif method == "perimeter":
        a, b, sa = _estimate_perimeter_cauchy(cnt, pixel_size_nm, circularity)
        result.a_nm, result.b_nm, result.SA_nm2 = a, b, sa
        result.sem_roughness_index = _sem_roughness_index(cnt)

    elif method == "monte_carlo":
        perim_px = _monte_carlo_perimeter(mask_u8)
        perim_nm = perim_px * pixel_size_nm
        circ = max(0.05, min(1.0, circularity))
        sa = perim_nm ** 2 / (np.pi * circ)
        import cv2
        x, y, w_b, h_b = cv2.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)
        result.a_nm = max(w_b, h_b) / 2.0 * pixel_size_nm
        result.b_nm = min(w_b, h_b) / 2.0 * pixel_size_nm
        result.SA_nm2 = float(sa)
        result.sem_roughness_index = _sem_roughness_index(cnt)

    elif method == "richardson":
        perim_nm, rich_fd = _richardson_perimeter(cnt, pixel_size_nm)
        circ = max(0.05, min(1.0, circularity))
        sa = perim_nm ** 2 / (np.pi * circ)
        import cv2
        x, y, w_b, h_b = cv2.boundingRect(cnt) if cnt is not None else (0, 0, 0, 0)
        result.a_nm = max(w_b, h_b) / 2.0 * pixel_size_nm
        result.b_nm = min(w_b, h_b) / 2.0 * pixel_size_nm
        result.SA_nm2 = float(sa)
        result.sem_roughness_index = _sem_roughness_index(cnt)

    else:
        raise ValueError(
            f"Unknown method {method!r}. "
            "Choose from 'auto', 'ellipsoid', 'cauchy', 'fourier', 'fourier_spiky', "
            "'capsule', 'perimeter', 'monte_carlo', 'richardson'."
        )

    # hollow correction: total SA = outer SA + inner SA
    if result.is_hollow and result.a_nm > 0:
        result.SA_outer_nm2 = result.SA_nm2
        shell_t = result.shell_thickness_estimate_nm
        if shell_t > 0:
            a_inner = result.a_nm - shell_t
            b_inner = result.b_nm - shell_t
            if a_inner > 0 and b_inner > 0:
                result.SA_inner_nm2 = _spheroid_sa_nm(a_inner, b_inner)
                result.SA_nm2 = result.SA_outer_nm2 + result.SA_inner_nm2

    # uncertainty propagation: SA_unc = SA × 2 × (pixel_unc / pixel_size)
    if pixel_size_uncertainty_nm > 0.0 and pixel_size_nm > 0.0:
        result.SA_nm2_uncertainty = (
            result.SA_nm2 * 2.0 * (pixel_size_uncertainty_nm / pixel_size_nm)
        )

    # ── always-computed diagnostics ───────────────────────────────────────────
    result.perimeter_nm, result.richardson_fractal_dim = _richardson_perimeter(
        cnt, pixel_size_nm
    )
    lac_r = max(2, min(10, mask_u8.shape[0] // 3, mask_u8.shape[1] // 3))
    result.lacunarity = _lacunarity(mask_u8, r=lac_r)

    return result


# ── batch worker (top-level for pickling) ────────────────────────────────────

def _batch_worker(args: tuple) -> list[dict]:
    """
    Process a chunk of masks in a subprocess.

    args = (masks_list, particle_ids, pixel_size_nm, pixel_size_uncertainty_nm, gpu_id)

    gpu_id >= 0  → set CUDA_VISIBLE_DEVICES for this process
    gpu_id  < 0  → CPU only
    """
    masks, particle_ids, pixel_size_nm, pixel_size_uncertainty_nm, gpu_id = args

    if gpu_id >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        device_label = f"cuda:{gpu_id}"
    else:
        device_label = "cpu"

    logger.debug("Worker processing %d masks on %s", len(masks), device_label)

    results = []
    for mask, pid in zip(masks, particle_ids):
        r = estimate_surface_area(
            mask,
            pixel_size_nm,
            pixel_size_uncertainty_nm,
            particle_id=pid,
            device=device_label,
        )
        results.append(asdict(r))
    return results


# ── batch entry point ─────────────────────────────────────────────────────────

def batch_surface_area(
    masks: Sequence[np.ndarray],
    pixel_size_nm: float,
    pixel_size_uncertainty_nm: float = 0.0,
    device: str = "auto",
    n_gpus: Optional[int] = None,
) -> "pandas.DataFrame":
    """
    Estimate surface area for a list of binary particle masks in parallel.

    Parameters
    ----------
    masks : sequence of 2D binary ndarrays
    pixel_size_nm : nm/px (same for all masks)
    pixel_size_uncertainty_nm : absolute uncertainty in pixel_size_nm
    device : 'auto' | 'cpu' | 'cuda' | 'cuda:N'
        'auto' uses all available CUDA GPUs; falls back to CPU workers.
    n_gpus : limit the number of GPUs used even when more are available

    Returns
    -------
    pandas.DataFrame with columns:
        particle_id, a_nm, b_nm, SA_nm2, SA_nm2_uncertainty, coverage_score,
        method_used, flagged, flag_reason, is_hollow,
        shell_thickness_estimate_nm, aggregate_score, sem_roughness_index
    """
    import pandas as pd
    from concurrent.futures import ProcessPoolExecutor, as_completed

    masks = list(masks)
    n = len(masks)
    if n == 0:
        return pd.DataFrame(columns=list(asdict(SurfaceAreaResult()).keys()))

    # resolve worker pool
    resolved = _resolve_device(device)
    if "cuda" in resolved:
        gpu_ids = _available_gpu_ids(n_gpus)
    else:
        gpu_ids = []

    if gpu_ids:
        n_workers = len(gpu_ids)
        logger.info("batch_surface_area: distributing %d masks across %d GPU(s)", n, n_workers)
    else:
        n_workers = os.cpu_count() or 1
        gpu_ids = [-1] * n_workers
        logger.info("batch_surface_area: distributing %d masks across %d CPU worker(s)", n, n_workers)

    particle_ids = list(range(n))

    # split into chunks, one per worker
    chunk_size = max(1, (n + n_workers - 1) // n_workers)
    chunks = []
    for w, gid in zip(range(n_workers), (gpu_ids * ((n_workers // len(gpu_ids)) + 1))[:n_workers]):
        lo = w * chunk_size
        hi = min(lo + chunk_size, n)
        if lo >= n:
            break
        chunks.append((
            masks[lo:hi],
            particle_ids[lo:hi],
            pixel_size_nm,
            pixel_size_uncertainty_nm,
            gid,
        ))

    all_rows: list[dict] = []

    if n_workers == 1 or n < 8:
        # small batch: run in-process to avoid spawn overhead
        for chunk in chunks:
            all_rows.extend(_batch_worker(chunk))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_batch_worker, chunk): i for i, chunk in enumerate(chunks)}
            for future in as_completed(futures):
                try:
                    all_rows.extend(future.result())
                except Exception as exc:
                    logger.error("Worker chunk failed: %s", exc)

    df = pd.DataFrame(all_rows)
    # ensure column order matches spec
    ordered_cols = [
        "particle_id", "a_nm", "b_nm", "SA_nm2", "SA_nm2_uncertainty",
        "coverage_score", "method_used", "flagged", "flag_reason",
        "is_hollow", "shell_thickness_estimate_nm", "aggregate_score",
        "sem_roughness_index",
    ]
    for col in ordered_cols:
        if col not in df.columns:
            df[col] = None
    return df[ordered_cols].sort_values("particle_id").reset_index(drop=True)


# ── GPU diagnostics ───────────────────────────────────────────────────────────

def gpu_stats() -> None:
    """Print available GPUs, total memory, and current allocation."""
    try:
        import torch
    except ImportError:
        print("torch not installed. Install with: pip install torch")
        return

    if not torch.cuda.is_available():
        print("No CUDA GPUs detected. Running on CPU.")
        return

    n = torch.cuda.device_count()
    print(f"{n} CUDA GPU(s) available:")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / 1024 ** 3
        alloc_gb = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached_gb = torch.cuda.memory_reserved(i) / 1024 ** 3
        print(
            f"  [{i}] {props.name}: "
            f"{total_gb:.1f} GB total | "
            f"{alloc_gb:.2f} GB allocated | "
            f"{cached_gb:.2f} GB cached"
        )
        try:
            print(torch.cuda.memory_summary(device=i, abbreviated=True))
        except Exception:
            pass


# ── public shape-metric helpers ───────────────────────────────────────────────

def compute_circularity(mask: np.ndarray) -> float:
    """
    Compute circularity = 4π·area / perimeter² for a binary mask.

    Returns a value in [0, 1] where 1.0 = perfect circle.  Values below ~0.75
    indicate elongated or irregular particles.
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1e-6:
        return 0.0
    return float(min(1.0, 4.0 * np.pi * area / perimeter ** 2))


def compute_convexity(mask: np.ndarray) -> float:
    """
    Compute convexity (solidity) = mask area / convex hull area.

    Returns a value in [0, 1] where 1.0 = perfectly convex.  Values below ~0.85
    indicate concave indentations or lobular shapes.
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    if hull_area <= 0:
        return 0.0
    return float(min(1.0, area / hull_area))


def compute_fractal_dimension(contour: np.ndarray) -> float:
    """
    Estimate the fractal dimension of a contour via box-counting.

    Parameters
    ----------
    contour : OpenCV contour array with shape (N, 1, 2), as returned by
              cv2.findContours().  Use fit_ellipse_to_mask() or a direct
              cv2.findContours() call to obtain the contour first.

    Returns
    -------
    Fractal dimension in [1.0, 2.0].  Smooth curves → ~1.0;
    highly fractal/spiky boundaries → approaching 2.0.  Values above 1.15
    trigger the fourier_spiky estimation tier.
    """
    import cv2

    if contour is None or len(contour) < 10:
        return 1.0
    # determine canvas bounds
    pts = contour[:, 0, :]
    h = int(pts[:, 1].max()) + 2
    w = int(pts[:, 0].max()) + 2
    canvas = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(canvas, [contour], -1, 1, 1)
    return _box_counting_fractal(canvas, contour)


def fit_ellipse_to_mask(mask: np.ndarray) -> Optional[dict]:
    """
    Fit a minimum-area ellipse to the largest contour in a binary mask.

    Returns
    -------
    dict with keys:
        center_x, center_y : centroid of the fitted ellipse (pixels)
        a_px               : semi-major axis length (pixels)
        b_px               : semi-minor axis length (pixels)
        angle_deg          : rotation angle in degrees (OpenCV convention)
    or None if the contour has fewer than 5 points.
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    try:
        (cx, cy), (ma, mb), angle = cv2.fitEllipse(cnt)
        return {
            "center_x": float(cx),
            "center_y": float(cy),
            "a_px": float(max(ma, mb) / 2.0),
            "b_px": float(min(ma, mb) / 2.0),
            "angle_deg": float(angle),
        }
    except Exception:
        return None


def contour_to_radial_signal(contour: np.ndarray, n_points: int = 256) -> np.ndarray:
    """
    Sample the radial distance r(θ) of a contour at uniformly-spaced angles.

    The contour centroid is used as the origin.  Angular sampling uses linear
    interpolation on the sorted (angle, radius) pairs with periodic wrap-around.

    Parameters
    ----------
    contour  : OpenCV contour array (N, 1, 2)
    n_points : number of angular samples (default 256)

    Returns
    -------
    1D float64 ndarray of length n_points, each value = radius in pixels
    at the corresponding angle in [-π, π).
    """
    from scipy.interpolate import interp1d

    if contour is None or len(contour) < 3:
        return np.zeros(n_points)

    pts = contour[:, 0, :].astype(float)
    cx, cy = pts.mean(axis=0)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    angles = np.arctan2(dy, dx)
    radii = np.sqrt(dx ** 2 + dy ** 2)

    order = np.argsort(angles)
    angles_s = angles[order]
    radii_s = radii[order]

    # wrap-around padding for periodic interpolation
    pad = 20
    a_ext = np.concatenate([angles_s[-pad:] - 2 * np.pi, angles_s, angles_s[:pad] + 2 * np.pi])
    r_ext = np.concatenate([radii_s[-pad:], radii_s, radii_s[:pad]])

    target = np.linspace(-np.pi, np.pi, n_points, endpoint=False)
    f = interp1d(a_ext, r_ext, kind="linear", bounds_error=False, fill_value=radii_s.mean())
    return f(target).astype(float)


def detect_spikes(contour: np.ndarray) -> dict:
    """
    Detect spike-like protrusions on a particle contour via local curvature maxima.

    Each curvature peak is modelled as a spike tip; its base width and height
    are estimated from the surrounding contour geometry.

    Parameters
    ----------
    contour : OpenCV contour array (N, 1, 2)

    Returns
    -------
    dict with keys:
        spike_count     : int   — number of detected spikes
        mean_length_px  : float — mean tip-to-base height in pixels
        mean_width_px   : float — mean base width in pixels
        spike_indices   : list[int] — contour indices of spike tips
    """
    from scipy.signal import find_peaks

    if contour is None or len(contour) < 20:
        return {"spike_count": 0, "mean_length_px": 0.0, "mean_width_px": 0.0, "spike_indices": []}

    pts = contour[:, 0, :].astype(float)
    n = len(pts)
    cx, cy = pts.mean(axis=0)
    r_all = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
    r_mean = r_all.mean()

    curvatures = np.zeros(n)
    for i in range(n):
        p0 = pts[(i - 2) % n]
        p1 = pts[i]
        p2 = pts[(i + 2) % n]
        v1, v2 = p1 - p0, p2 - p1
        cross = abs(v1[0] * v2[1] - v1[1] * v2[0])
        mag = (np.linalg.norm(v1) * np.linalg.norm(v2)) ** 1.5
        curvatures[i] = cross / max(mag, 1e-9)

    threshold = float(np.percentile(curvatures, 80))
    peaks, _ = find_peaks(curvatures, height=threshold, distance=5)

    lengths, widths = [], []
    for pk in peaks:
        lo, hi = int(pk), int(pk)
        level = curvatures[pk] * 0.3
        while lo > 0 and curvatures[lo] > level:
            lo -= 1
        while hi < n - 1 and curvatures[hi] > level:
            hi += 1
        widths.append(float(np.linalg.norm(pts[lo] - pts[hi])))
        lengths.append(float(max(0.0, r_all[pk] - r_mean)))

    return {
        "spike_count": int(len(peaks)),
        "mean_length_px": float(np.mean(lengths)) if lengths else 0.0,
        "mean_width_px": float(np.mean(widths)) if widths else 0.0,
        "spike_indices": peaks.tolist(),
    }


# ── public SA formula helpers ─────────────────────────────────────────────────

def ellipsoid_sa(a: float, b: float) -> float:
    """
    Compute the analytical surface area of a prolate or oblate spheroid.

    Parameters
    ----------
    a : semi-major axis (any unit; result is in that unit squared)
    b : semi-minor axis (same unit as a)

    Returns
    -------
    Surface area as a float.

    Notes
    -----
    Uses the exact formula for a spheroid of revolution:

      Prolate (a > b):  SA = 2πb²(1 + (a / (b·e)) · arcsin(e)),  e = sqrt(1 - b²/a²)
      Oblate  (b > a):  SA = 2πb²(1 + ((1 - e²)/e) · arctanh(e)), e = sqrt(1 - a²/b²)
      Sphere  (a == b): SA = 4πa²
    """
    if a <= 0 or b <= 0:
        return 0.0
    if abs(a - b) < 1e-12 * max(a, b):
        return 4.0 * np.pi * a ** 2
    if a > b:
        e = np.sqrt(max(0.0, 1.0 - (b / a) ** 2))
        if e < 1e-12:
            return 4.0 * np.pi * a ** 2
        return float(2.0 * np.pi * b ** 2 * (1.0 + (a / (b * e)) * np.arcsin(e)))
    else:
        e = np.sqrt(max(0.0, 1.0 - (a / b) ** 2))
        if e < 1e-12:
            return 4.0 * np.pi * b ** 2
        return float(2.0 * np.pi * b ** 2 * (1.0 + ((1.0 - e ** 2) / e) * np.arctanh(e)))


def cauchy_sa(mask: np.ndarray, pixel_size_nm: float) -> float:
    """
    Estimate surface area from a binary mask using the Cauchy-Crofton theorem.

    SA ≈ k × A_projected, where k is corrected for the bounding-box aspect ratio.

    Parameters
    ----------
    mask : 2D binary ndarray
    pixel_size_nm : nm/px

    Returns
    -------
    Estimated surface area in nm².
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, _, sa = _estimate_cauchy(cnt, pixel_size_nm)
    return sa


def fourier_sa(mask: np.ndarray, pixel_size_nm: float) -> float:
    """
    Estimate surface area from a binary mask via Fourier radial decomposition.

    Parameters
    ----------
    mask : 2D binary ndarray
    pixel_size_nm : nm/px

    Returns
    -------
    Estimated surface area in nm².
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, _, sa, _ = _estimate_fourier(cnt, pixel_size_nm)
    return sa


def fourier_spiky_sa(mask: np.ndarray, pixel_size_nm: float) -> float:
    """
    Estimate surface area from a binary mask using the Fourier + spike-cone model.

    Suitable for particles with fractal or highly irregular boundaries (e.g.
    spiky nanoparticles, dendritic structures).

    Parameters
    ----------
    mask : 2D binary ndarray
    pixel_size_nm : nm/px

    Returns
    -------
    Estimated surface area in nm².
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    _, _, sa, _ = _estimate_fourier_spiky(cnt, pixel_size_nm)
    return sa


def detect_hollow(
    mask: np.ndarray,
    image_crop: Optional[np.ndarray] = None,
) -> dict:
    """
    Detect hollow or shell-like particles in a binary mask.

    Without intensity data, the function checks for donut-shaped masks
    (interior holes not covered by the mask).  With intensity data, it
    looks for a bright centre relative to the particle edge, which is the
    expected pattern in TEM bright-field images of hollow particles (empty
    lumen = fewer electrons absorbed = higher transmitted intensity).

    Parameters
    ----------
    mask       : 2D binary ndarray (H × W)
    image_crop : optional grayscale float array co-registered with mask;
                 same H × W.  Must be the un-normalised or normalised image,
                 not a contrast-stretched display copy.

    Returns
    -------
    dict with keys:
        is_hollow              : bool
        shell_thickness_px     : float — estimated shell thickness in pixels
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    is_hollow, shell_t = _detect_hollow(mask_u8, image_crop)
    return {"is_hollow": is_hollow, "shell_thickness_px": float(shell_t)}


def detect_aggregate(mask: np.ndarray) -> dict:
    """
    Detect whether a mask likely represents an aggregate of fused particles.

    Uses the number and depth of convexity defects in the contour.  Large,
    deeply-indented defects suggest that multiple particles have merged.

    Parameters
    ----------
    mask : 2D binary ndarray

    Returns
    -------
    dict with keys:
        is_aggregate    : bool   — True when aggregate_score > 0.5
        aggregate_score : float  — 0 (clean single particle) to 1 (likely aggregate)
    """
    import cv2

    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return {"is_aggregate": False, "aggregate_score": 0.0}
    cnt = max(contours, key=cv2.contourArea)
    score = _aggregate_score(cnt)
    return {"is_aggregate": bool(score > 0.5), "aggregate_score": float(score)}


# ── population-level statistics ───────────────────────────────────────────────

def compute_specific_surface_area(
    results_df: "pd.DataFrame",
    sample_volume_nm3: float,
    include_flagged: bool = False,
) -> dict:
    """
    Compute specific surface area (SSA) for a particle population.

    SSA is defined as the total particle surface area divided by the sample
    volume:  SSA = Σ SA_i / V_sample  [units: nm⁻¹].

    Parameters
    ----------
    results_df : DataFrame from batch_surface_area() or a filtered subset.
    sample_volume_nm3 : total sample volume in nm³.

        For a TEM lamella:  field_of_view_area_nm2 × section_thickness_nm.
        For an SEM image:   field_of_view_area_nm2 × estimated_depth_nm.

    include_flagged : if False (default), rows where flagged=True are excluded
        from the sum and statistics.  Flagged particles include partially
        occluded, aggregated, or undersized masks which yield less reliable SA.

    Returns
    -------
    dict with keys:
        specific_sa_nm_inv     : SSA in nm⁻¹ (= total SA / sample volume)
        total_sa_nm2           : sum of included particle SA values (nm²)
        n_particles            : total rows in results_df
        n_included             : particles contributing to SSA
        n_excluded_flagged     : particles excluded due to flag
        mean_sa_nm2            : mean per-particle SA (nm²)
        median_sa_nm2          : median per-particle SA (nm²)
        std_sa_nm2             : standard deviation of per-particle SA (nm²)
        percentiles            : dict mapping percentile (5,10,25,50,75,90,95)
                                 to SA value (nm²)

    Notes on TEM vs SEM geometry
    ----------------------------
    TEM projections are geometrically faithful silhouettes: the transmitted
    electron beam integrates through the full particle thickness, so the 2D
    outline is a true projection of the 3D shape.  SA estimates from TEM
    masks are generally more accurate for smooth or convex particles
    (ellipsoid and cauchy tiers) and for particles that are fully within the
    field of view.

    SEM images show the visible surface face; particles at the substrate edge
    are partially visible, which tends to underestimate SA.  However, the
    sem_roughness_index column (std(r)/mean(r) of the radial contour profile,
    reflecting boundary texture variance) is a meaningful supplementary
    feature specific to SEM data: higher values correlate with rougher or
    more textured particle surfaces, and values above ~0.05 suggest that the
    fourier_spiky estimation tier should be verified or applied manually.
    """
    import pandas as pd

    n_total = len(results_df)
    df = results_df[results_df["SA_nm2"] > 0].copy()

    n_excluded = 0
    if not include_flagged and "flagged" in df.columns:
        n_before = len(df)
        df = df[~df["flagged"]]
        n_excluded = n_before - len(df)

    n_included = len(df)
    empty = {
        "specific_sa_nm_inv": 0.0,
        "total_sa_nm2": 0.0,
        "n_particles": n_total,
        "n_included": 0,
        "n_excluded_flagged": n_excluded,
        "mean_sa_nm2": float("nan"),
        "median_sa_nm2": float("nan"),
        "std_sa_nm2": float("nan"),
        "percentiles": {},
    }

    if n_included == 0 or sample_volume_nm3 <= 0.0:
        return empty

    sa = df["SA_nm2"].values
    total_sa = float(sa.sum())

    return {
        "specific_sa_nm_inv": total_sa / sample_volume_nm3,
        "total_sa_nm2": total_sa,
        "n_particles": n_total,
        "n_included": n_included,
        "n_excluded_flagged": n_excluded,
        "mean_sa_nm2": float(sa.mean()),
        "median_sa_nm2": float(np.median(sa)),
        "std_sa_nm2": float(sa.std()),
        "percentiles": {p: float(np.percentile(sa, p)) for p in [5, 10, 25, 50, 75, 90, 95]},
    }


# ── diagnostic overlay figure ─────────────────────────────────────────────────

#: Color per estimation method for diagnostic overlays and plots.
METHOD_COLORS: dict[str, str] = {
    "ellipsoid": "#4878D0",
    "cauchy": "#6ACC65",
    "fourier": "#EE854A",
    "fourier_spiky": "#D65F5F",
    "unknown": "#999999",
}


def plot_sa_diagnostics(
    image: np.ndarray,
    masks: list,
    results_df: "pd.DataFrame",
    output_path: Optional[str] = None,
) -> "matplotlib.figure.Figure":
    """
    Overlay per-particle SA diagnostics onto the source image.

    Each particle mask is drawn as a semi-transparent filled polygon with
    a coloured border indicating the estimation method used:

      ellipsoid    = blue   (#4878D0)
      cauchy       = green  (#6ACC65)
      fourier      = orange (#EE854A)
      fourier_spiky = red   (#D65F5F)

    A compact text label showing the method abbreviation and SA value in nm²
    is placed at the particle centroid.  Additional markers are added for
    special cases:

      hollow      — cyan ring at centroid
      aggregate   — yellow diamond at centroid (aggregate_score > 0.5)
      flagged     — white 'x' above the mask

    Parameters
    ----------
    image      : 2D grayscale float or uint8 array (H × W).  Displayed in
                 grey scale as the background.
    masks      : list of 2D binary arrays, one per particle.  Must be in the
                 same pixel coordinate system as *image*.  Indices correspond
                 to rows in results_df (0-based).
    results_df : DataFrame from batch_surface_area() aligned with masks.
    output_path : if given, the figure is saved as a PNG at 150 DPI.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba
    import cv2

    METHOD_ABBR = {
        "ellipsoid": "E",
        "cauchy": "C",
        "fourier": "F",
        "fourier_spiky": "FS",
    }

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    img_f = image.astype(float)
    lo, hi = img_f.min(), img_f.max()
    img_disp = (img_f - lo) / max(hi - lo, 1e-9)
    ax.imshow(img_disp, cmap="gray", origin="upper", interpolation="nearest")

    for i, mask in enumerate(masks):
        row = results_df.iloc[i] if results_df is not None and i < len(results_df) else None
        method = str(row["method_used"]) if row is not None else "unknown"
        color = METHOD_COLORS.get(method, "#999999")

        mask_u8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        pts = cnt[:, 0, :]
        if len(pts) < 3:
            continue

        poly = mpatches.Polygon(
            pts, closed=True,
            facecolor=to_rgba(color, 0.15),
            edgecolor=color,
            linewidth=1.2,
            zorder=2,
        )
        ax.add_patch(poly)

        cx_f, cy_f = float(pts[:, 0].mean()), float(pts[:, 1].mean())

        if row is not None:
            sa_val = float(row["SA_nm2"])
            abbr = METHOD_ABBR.get(method, "?")
            ax.text(
                cx_f, cy_f,
                f"{abbr}\n{sa_val:.0f}",
                ha="center", va="center",
                fontsize=5, color=color, fontweight="bold",
                zorder=4,
            )
            y_top = float(pts[:, 1].min()) - 5.0
            if bool(row.get("is_hollow", False)):
                ax.plot(cx_f, cy_f, "o", ms=11, mfc="none", mec="cyan", mew=1.5, zorder=5)
            if float(row.get("aggregate_score", 0.0)) > 0.5:
                ax.plot(cx_f, cy_f, "D", ms=8, mfc="none", mec="#FFD700", mew=1.5, zorder=5)
            if bool(row.get("flagged", False)):
                ax.plot(cx_f, y_top, "x", ms=7, mec="white", mew=1.5, zorder=5)

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=c, label=m)
        for m, c in METHOD_COLORS.items()
        if m != "unknown"
    ] + [
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="none", markeredgecolor="cyan", ms=9, label="hollow"),
        plt.Line2D([0], [0], marker="D", color="none",
                   markerfacecolor="none", markeredgecolor="#FFD700", ms=8, label="aggregate"),
        plt.Line2D([0], [0], marker="x", color="white", ms=8, label="flagged"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.75, facecolor="#222222", labelcolor="white")
    ax.axis("off")
    ax.set_title("SA Diagnostics", fontsize=9, color="white", pad=4)

    if output_path is not None:
        fig.savefig(str(output_path), dpi=150, bbox_inches="tight", facecolor="black")

    return fig
