"""Statistics for detected atomic columns in atomic-resolution (S)TEM images.

Takes an (n, 2) array of atom coordinates (row, col = y, x, in pixels) and a pixel
size, and computes the quantities that matter for lattice/moire analysis:

  - nearest-neighbour distance distribution (-> lattice constant, local strain)
  - per-atom local lattice spacing
  - bond-orientational order psi6 (hexagonal order; highlights the moire)
  - Voronoi coordination number (defect / stacking-region detection)
  - per-atom strain tensor (exx, eyy, exy) and local rotation, from the local
    deformation gradient fitted to each atom's neighbour shell vs an averaged
    reference lattice
  - moire super-period and twist angle (from the atomic spacing + moire period)

Pure NumPy/SciPy. Returns plain dicts/arrays so it is GUI- and file-friendly.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

try:
    from scipy.spatial import Voronoi
except Exception:  # pragma: no cover
    Voronoi = None


def nearest_neighbour(coords: np.ndarray, k: int = 6):
    """Return (nn_dist_per_atom, nn_vectors) using the k nearest neighbours.

    nn_dist_per_atom : median NN distance for each atom (robust local spacing).
    nn_vectors       : list of (m,2) neighbour offset vectors per atom.
    """
    coords = np.asarray(coords, float)
    tree = cKDTree(coords)
    d, idx = tree.query(coords, k=k + 1)          # +1 = self
    d, idx = d[:, 1:], idx[:, 1:]
    nn_med = np.median(d, axis=1)
    vecs = [coords[idx[i]] - coords[i] for i in range(len(coords))]
    return nn_med, vecs, d, idx


def bond_orientational_order(coords: np.ndarray, k: int = 6) -> np.ndarray:
    """Per-atom hexagonal order parameter psi6 = |<exp(6 i theta)>| over neighbours.
    1 = perfect hexagonal; drops at defects, domain walls, and moire cores."""
    _, vecs, _, _ = nearest_neighbour(coords, k=k)
    out = np.zeros(len(coords))
    for i, v in enumerate(vecs):
        if len(v) == 0:
            continue
        ang = np.arctan2(v[:, 0], v[:, 1])
        out[i] = np.abs(np.mean(np.exp(6j * ang)))
    return out


def voronoi_coordination(coords: np.ndarray) -> np.ndarray:
    """Voronoi coordination number per atom (number of Voronoi neighbours).
    Interior hexagonal atoms -> 6; defects deviate. Border atoms -> np.nan."""
    n = len(coords)
    coord_num = np.full(n, np.nan)
    if Voronoi is None or n < 5:
        return coord_num
    vor = Voronoi(np.asarray(coords, float)[:, ::-1])   # (x,y) for Voronoi
    counts = np.zeros(n, int)
    finite = np.ones(n, bool)
    for (a, b), ridge in zip(vor.ridge_points, vor.ridge_vertices):
        counts[a] += 1
        counts[b] += 1
        if -1 in ridge:
            finite[a] = finite[b] = False
    coord_num[:] = counts
    coord_num[~finite] = np.nan
    return coord_num


def _reference_hex(spacing: float, angle0: float) -> np.ndarray:
    """Ideal hexagonal neighbour offsets (6 dirs) at `spacing`, oriented by angle0."""
    ang = angle0 + np.deg2rad(np.arange(0, 360, 60))
    return np.stack([spacing * np.sin(ang), spacing * np.cos(ang)], axis=1)  # (row,col)


def _smooth_over_neighbours(arr: np.ndarray, idx: np.ndarray, iters: int = 2) -> np.ndarray:
    """Median-smooth a per-atom scalar over each atom's neighbour list, NaN-aware."""
    out = arr.astype(float).copy()
    for _ in range(iters):
        nxt = out.copy()
        for i in range(len(out)):
            if np.isnan(out[i]):
                continue
            vals = out[idx[i]]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                nxt[i] = np.median(np.append(vals, out[i]))
        out = nxt
    return out


def strain_field(coords: np.ndarray, k: int = 6, smooth_iters: int = 2):
    """Robust per-atom lattice distortion for warped/moire lattices.

    Rather than referencing one global orientation (which spuriously reads the
    moire's local rotation as huge shear), this uses a **per-atom local
    orientation** and the global relaxed spacing:

      - ``dilatation``  : areal strain (local_spacing - a0)/a0        [fractional]
      - ``rotation_deg``: local lattice rotation vs the global mean    [degrees]
      - ``exx,eyy,exy`` : deviatoric strain from the neighbour fit against a
                          *locally-oriented* reference (shape distortion only)

    All fields are median-smoothed over neighbours and set to NaN for atoms with
    an incomplete neighbour shell (a missed neighbour would otherwise fake strain).
    """
    coords = np.asarray(coords, float)
    n = len(coords)
    nn_med, vecs, d, idx = nearest_neighbour(coords, k=k)
    a0 = float(np.median(nn_med))
    # per-atom local orientation via the 6-fold complex phasor (correct circular
    # statistic for a hexagonal lattice: mean(exp(6i.theta)) -> angle/6).
    phasor = np.array([np.mean(np.exp(6j * np.arctan2(v[:, 0], v[:, 1])))
                       if len(v) else 0 + 0j for v in vecs])
    for _ in range(2):  # smooth the phasor over neighbours
        phasor = np.array([np.mean(np.append(phasor[idx[i]], phasor[i])) for i in range(n)])
    local_ang = np.angle(phasor) / 6.0            # radians, in [-pi/6, pi/6]
    global_ang = float(np.angle(np.mean(phasor)) / 6.0)

    exx = np.full(n, np.nan); eyy = np.full(n, np.nan)
    exy = np.full(n, np.nan); rot = np.full(n, np.nan); dil = np.full(n, np.nan)
    for i, v in enumerate(vecs):
        # reject incomplete shells: too few neighbours or an anomalously far one
        if len(v) < 5 or d[i].max() > 1.6 * a0:
            continue
        dil[i] = (nn_med[i] - a0) / a0
        # difference of two lattice orientations, wrapped into +/-30 deg (60 deg cell)
        rot[i] = _wrap_deg(local_ang[i] - global_ang, 60.0)
        # deviatoric distortion: fit F against a locally-oriented, unit-spacing ref
        ref = _reference_hex(a0, local_ang[i])
        _, j = cKDTree(ref).query(v)
        R = ref[j]
        RtR = R.T @ R
        if np.linalg.det(RtR) < 1e-6:
            continue
        F = (v.T @ R) @ np.linalg.inv(RtR)
        E = 0.5 * (F + F.T) - np.eye(2)
        exx[i], eyy[i], exy[i] = E[0, 0], E[1, 1], E[0, 1]

    for arr in (dil, rot, exx, eyy, exy):
        arr[:] = _smooth_over_neighbours(arr, idx, iters=smooth_iters)
    return {"dilatation": dil, "rotation_deg": rot,
            "exx": exx, "eyy": eyy, "exy": exy,
            "spacing_px": a0, "orientation_rad": global_ang}


def _wrap_deg(rad: float, cell_deg: float) -> float:
    """Wrap a small angle difference (radians) into +/- half a symmetry cell (deg)."""
    d = np.rad2deg(rad)
    half = cell_deg / 2.0
    return ((d + half) % cell_deg) - half


def moire_period_twist(image: np.ndarray, atomic_spacing_px: float,
                       pixel_size_nm: float | None = None) -> dict:
    """Estimate the moire super-period (px) from the low-frequency satellites in
    the power spectrum, and the twist angle from theta ~ a / lambda_moire."""
    f = np.asarray(image, np.float32)
    if f.ndim == 3:
        f = f.mean(axis=-1)
    f = f - float(np.mean(f))
    P = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    H, W = P.shape; cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    rad = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    # look for the strongest peak at low radius (moire), excluding the DC core
    lo, hi = 3, max(6, int(0.5 * ((H + W) / 2) / atomic_spacing_px))
    ring = (rad >= lo) & (rad <= hi)
    if not ring.any():
        return {"moire_period_px": None, "twist_deg": None}
    # radial profile within the low-frequency band
    rr = rad[ring].astype(int); pp = P[ring]
    prof = np.array([pp[rr == r].sum() if (rr == r).any() else 0.0 for r in range(lo, hi + 1)])
    r_peak = int(np.argmax(prof)) + lo
    lam = ((H + W) / 2.0) / max(r_peak, 1)
    theta = np.rad2deg(atomic_spacing_px / lam) if lam > 0 else None
    out = {"moire_period_px": float(lam), "twist_deg": float(theta) if theta else None}
    if pixel_size_nm:
        out["moire_period_nm"] = float(lam * pixel_size_nm)
    return out


def save_stats_figure(coords: np.ndarray, out_path: str,
                      pixel_size_nm: float | None = None,
                      image: np.ndarray | None = None) -> str:
    """Render the 6-panel atom-statistics maps to a PNG (off-screen, no pyplot —
    safe to call from a Qt GUI thread). Returns the path written."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    coords = np.asarray(coords, float)
    nn_med, _, d, idx = nearest_neighbour(coords)
    psi6 = bond_orientational_order(coords)
    cn = voronoi_coordination(coords)
    st = strain_field(coords)
    unit = pixel_size_nm * 10 if pixel_size_nm else 1.0
    ulabel = "A" if pixel_size_nm else "px"

    fig = Figure(figsize=(18, 11)); FigureCanvasAgg(fig)
    ax = fig.subplots(2, 3)

    ax[0, 0].hist(d[:, 0] * unit, bins=80, color="#3b7dd8")
    ax[0, 0].set_title(f"NN distance ({ulabel})"); ax[0, 0].set_xlabel(ulabel)

    def _scatter(a, c, title, cmap, vmin=None, vmax=None):
        s = a.scatter(coords[:, 1], coords[:, 0], c=c, s=5, cmap=cmap, vmin=vmin, vmax=vmax)
        a.set_title(title); a.invert_yaxis(); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([]); fig.colorbar(s, ax=a, shrink=0.7)

    _scatter(ax[0, 1], psi6, "hexagonal order psi6", "viridis", 0.4, 1.0)
    _scatter(ax[0, 2], cn, "Voronoi coordination", "coolwarm", 4, 8)
    _scatter(ax[1, 0], nn_med * unit, f"local lattice spacing ({ulabel})", "magma")
    _scatter(ax[1, 1], st["rotation_deg"], "local rotation (deg)", "RdBu", -3, 3)
    _scatter(ax[1, 2], st["dilatation"] * 100, "areal strain (%)", "PuOr", -6, 6)

    fig.tight_layout()
    fig.savefig(out_path, dpi=90)
    return out_path


def summarize(coords: np.ndarray, pixel_size_nm: float | None = None,
              image: np.ndarray | None = None) -> dict:
    """One-call summary of the atom statistics for a status panel / CSV header."""
    coords = np.asarray(coords, float)
    out: dict = {"n_atoms": int(len(coords))}
    if len(coords) < 5:
        return out
    nn_med, _, d, _ = nearest_neighbour(coords)
    nn1 = d[:, 0]                                       # first-neighbour distance
    out["nn_distance_px_mean"] = float(np.mean(nn1))
    out["nn_distance_px_std"] = float(np.std(nn1))
    out["lattice_spacing_px"] = float(np.median(nn_med))
    psi6 = bond_orientational_order(coords)
    out["psi6_mean"] = float(np.nanmean(psi6))
    cn = voronoi_coordination(coords)
    out["coord_number_median"] = float(np.nanmedian(cn))
    # de-artifact the defect fraction: only over atoms with a complete neighbour
    # shell (an anomalously large NN distance = a missed neighbour, not a defect).
    a0 = float(np.median(nn_med))
    well = (~np.isnan(cn)) & (nn_med > 0.8 * a0) & (nn_med < 1.25 * a0)
    out["defect_fraction"] = (float(np.mean((cn[well] != 6))) if well.any() else float("nan"))
    st = strain_field(coords)
    out["dilatation_std_pct"] = float(np.nanstd(st["dilatation"]) * 100)
    out["rotation_deg_std"] = float(np.nanstd(st["rotation_deg"]))
    shear = np.sqrt(((st["exx"] - st["eyy"]) / 2.0) ** 2 + st["exy"] ** 2)
    out["shear_std_pct"] = float(np.nanstd(shear) * 100)
    if pixel_size_nm:
        out["nn_distance_A_mean"] = float(np.mean(nn1) * pixel_size_nm * 10)
        out["lattice_spacing_A"] = float(np.median(nn_med) * pixel_size_nm * 10)
    if image is not None:
        out.update(moire_period_twist(image, float(np.median(nn_med)), pixel_size_nm))
    return out
