"""Atomic-column detection for atomic-resolution (S)TEM images, including
moire-patterned lattices.

Atoms in a moire image are bright peaks whose *brightness varies strongly* across
the field (the moire beating modulates intensity), so a global threshold misses
atoms in the dim regions. The approach here is intensity-robust:

  1. Band-pass (difference-of-Gaussians): subtract the slow moire/background
     envelope while keeping the atom-scale features.
  2. Local contrast normalization: divide by a local RMS so dim-region atoms and
     bright-region atoms are put on the same footing.
  3. Local-maxima peak detection with a minimum atom spacing.
  4. Sub-pixel refinement by intensity-weighted centroid in a small window.

Returns peak coordinates as an (n, 2) array of (row, col) = (y, x) in pixels.

Pure NumPy/SciPy/scikit-image — no JAX, no GPU required.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

try:
    from skimage.feature import peak_local_max
except Exception as _e:  # pragma: no cover
    peak_local_max = None


def _normalize(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)
    if img.ndim == 3:
        img = img.mean(axis=-1)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def bandpass(img: np.ndarray, atom_sigma: float, bg_sigma: float) -> np.ndarray:
    """Difference-of-Gaussians band-pass. ``atom_sigma`` smooths at the atom scale;
    ``bg_sigma`` estimates the slow moire/background envelope to subtract."""
    fine = ndi.gaussian_filter(img, max(0.3, atom_sigma))
    coarse = ndi.gaussian_filter(img, bg_sigma)
    return fine - coarse


def local_contrast_normalize(resp: np.ndarray, sigma: float) -> np.ndarray:
    """Divide the response by a local RMS so peaks in dim and bright regions are
    comparable (this is what makes moire-modulated atoms detectable uniformly)."""
    r = np.clip(resp, 0.0, None)
    local_mean = ndi.gaussian_filter(r, sigma)
    local_sq = ndi.gaussian_filter(r * r, sigma)
    local_std = np.sqrt(np.clip(local_sq - local_mean ** 2, 0.0, None))
    return r / (local_std + 1e-6)


def detect_atoms(
    image: np.ndarray,
    lattice_spacing_px: float | None = None,
    min_distance_px: float | None = None,
    atom_sigma_px: float | None = None,
    threshold: float = 3.0,
    subpixel: bool = True,
    bright_atoms: bool = True,
    fill_gaps: bool = False,
) -> np.ndarray:
    """Detect atomic columns in an atomic-resolution (S)TEM image.

    Parameters
    ----------
    image : 2D array (or HxWx3; averaged to gray).
    lattice_spacing_px : approximate nearest-neighbour atom spacing in pixels.
        If given, the atom scale and minimum separation are derived from it and
        the other size parameters can be left as None. Estimated from the image
        autocorrelation if omitted.
    min_distance_px : minimum separation between detected peaks (default
        ~0.6 x lattice spacing).
    atom_sigma_px : Gaussian sigma of an atomic column (default ~lattice/5).
    threshold : peaks must exceed this many local-RMS units (default 3.0). Lower
        finds more (and more false) peaks.
    subpixel : refine each peak by intensity-weighted centroid.
    bright_atoms : True for bright atoms on dark background (HAADF/ADF STEM);
        set False for dark atoms on bright background (e.g. some BF/TEM).

    Returns
    -------
    (n, 2) float array of (row, col) atom-column coordinates in pixels.
    """
    if peak_local_max is None:
        raise ImportError("scikit-image is required for atom detection.")

    img = _normalize(image)
    if not bright_atoms:
        img = 1.0 - img

    if lattice_spacing_px is None:
        lattice_spacing_px = estimate_lattice_spacing(img)
    a = float(lattice_spacing_px)

    if atom_sigma_px is None:
        atom_sigma_px = max(1.0, a / 5.0)
    if min_distance_px is None:
        # 0.45x spacing (not 0.6x): moire beating compresses the lattice locally,
        # so a coarser minimum would suppress real close-packed columns. Safe on
        # clean lattices (atoms are >> this apart) — verified no precision loss.
        min_distance_px = max(2.0, 0.45 * a)

    # 1-2. band-pass + local contrast normalization
    resp = bandpass(img, atom_sigma=atom_sigma_px * 0.7, bg_sigma=a * 1.5)
    norm = local_contrast_normalize(resp, sigma=a * 1.5)

    # 3. local maxima with minimum spacing
    coords = peak_local_max(
        norm,
        min_distance=int(round(min_distance_px)),
        threshold_abs=float(threshold),
        exclude_border=False,
    ).astype(np.float64)

    # 4. optional lattice-guided fill-in of dim-region dropouts
    if fill_gaps and len(coords) > 10:
        coords, _ = refine_lattice_fill(
            image, coords, lattice_spacing_px=a, atom_sigma_px=atom_sigma_px,
            bright_atoms=bright_atoms)

    # 5. sub-pixel refinement (intensity-weighted centroid on the band-passed peak)
    if subpixel and len(coords):
        coords = _refine_centroid(np.clip(resp, 0.0, None), coords, rad=max(1, int(round(atom_sigma_px))))

    return coords


def refine_lattice_fill(image: np.ndarray, coords: np.ndarray,
                        lattice_spacing_px: float | None = None,
                        atom_sigma_px: float | None = None,
                        bright_atoms: bool = True,
                        rel_threshold: float = 0.25,
                        max_iter: int = 4) -> tuple[np.ndarray, int]:
    """Recover atoms missed by the first pass using the lattice periodicity.

    For every detected atom, the local nearest-neighbour vectors predict where its
    neighbours should sit. Any predicted site that is (a) currently empty and (b)
    has a local band-pass peak above ``rel_threshold`` x the typical atom response
    is added — so dim-region dropouts are filled in at their expected positions,
    without lowering the bar everywhere (which would add noise peaks). Iterates so
    newly-added atoms seed further fill-in. Returns (coords, n_added)."""
    img = _normalize(image)
    if not bright_atoms:
        img = 1.0 - img
    if lattice_spacing_px is None:
        lattice_spacing_px = estimate_lattice_spacing(img)
    a = float(lattice_spacing_px)
    if atom_sigma_px is None:
        atom_sigma_px = max(1.0, a / 5.0)
    norm = local_contrast_normalize(bandpass(img, atom_sigma_px * 0.7, a * 1.5), a * 1.5)
    H, W = norm.shape
    pts = [tuple(map(float, c)) for c in np.asarray(coords, float)]
    rad = int(max(2, round(a * 0.35)))
    added_total = 0
    for _ in range(max_iter):
        arr = np.asarray(pts)
        tree = cKDTree(arr)
        strong = np.median([norm[int(round(y)), int(round(x))] for y, x in arr
                            if 0 <= int(round(y)) < H and 0 <= int(round(x)) < W])
        thr = rel_threshold * strong
        # predict ALL 6 hexagonal neighbour sites from each atom's local lattice
        # orientation + spacing (so directions with a MISSING neighbour — the gaps —
        # are predicted too, which the observed NN vectors can't do).
        d, idx = tree.query(arr, k=7)
        preds = []
        for i in range(len(arr)):
            v = arr[idx[i, 1:]] - arr[i]
            v = v[np.linalg.norm(v, axis=1) < 1.4 * a]
            if len(v) < 2:
                continue
            ang = np.angle(np.mean(np.exp(6j * np.arctan2(v[:, 0], v[:, 1])))) / 6.0
            sp = float(np.median(np.linalg.norm(v, axis=1)))
            th = ang + np.deg2rad(np.arange(0, 360, 60))
            preds.append(arr[i] + np.stack([sp * np.sin(th), sp * np.cos(th)], axis=1))
        preds = np.concatenate(preds, axis=0) if preds else np.empty((0, 2))
        added = 0
        for c in preds:
            y0, x0 = int(round(c[0])), int(round(c[1]))
            if not (rad <= y0 < H - rad and rad <= x0 < W - rad):
                continue
            if tree.query(c)[0] <= 0.6 * a:            # site already occupied
                continue
            patch = norm[y0 - rad:y0 + rad + 1, x0 - rad:x0 + rad + 1]
            j = np.unravel_index(int(np.argmax(patch)), patch.shape)
            py, px = y0 - rad + j[0], x0 - rad + j[1]
            if norm[py, px] >= thr and tree.query((py, px))[0] > 0.6 * a:
                pts.append((float(py), float(px)))
                added += 1
                tree = cKDTree(np.asarray(pts))        # keep tree current within the pass
        added_total += added
        if added == 0:
            break
    return np.asarray(pts), added_total


def _refine_centroid(field: np.ndarray, coords: np.ndarray, rad: int) -> np.ndarray:
    H, W = field.shape
    out = coords.copy()
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    for i, (r, c) in enumerate(coords):
        r0, c0 = int(round(r)), int(round(c))
        y1, y2 = max(0, r0 - rad), min(H, r0 + rad + 1)
        x1, x2 = max(0, c0 - rad), min(W, c0 + rad + 1)
        patch = field[y1:y2, x1:x2]
        s = patch.sum()
        if s <= 0:
            continue
        gy, gx = np.mgrid[y1:y2, x1:x2]
        out[i, 0] = (patch * gy).sum() / s
        out[i, 1] = (patch * gx).sum() / s
    return out


def estimate_lattice_spacing(img: np.ndarray, rmin: int = 3, rmax: int = 300) -> float:
    """Estimate the lattice spacing (px) from the dominant lattice frequency in the
    power spectrum: the strongest non-DC radial peak sits at N/spacing, so
    ``spacing = N / r_peak``. This is far more reliable than the autocorrelation
    first-ring for noisy, moire-modulated lattices."""
    f = np.asarray(img, np.float32)
    if f.ndim == 3:
        f = f.mean(axis=-1)
    f = f - float(np.mean(f))
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    H, W = P.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(int)
    rhi = int(min(rmax, cy - 1, cx - 1))
    if rhi <= rmin + 4:
        return 8.0
    prof = np.array([P[rad == r].sum() for r in range(rmin, rhi)])
    r_peak = int(np.argmax(prof)) + rmin
    if r_peak <= 0:
        return 8.0
    return float(((H + W) / 2.0) / r_peak)
