"""Spatial point-pattern statistics for detected features.

Pure numpy/scipy (no scikit-learn dependency). Operates on feature centroids in
nanometres. Covers:
  - nearest-neighbour distance + Clark-Evans clustering index (univariate)
  - DBSCAN proximity clustering
  - kernel density (hotspot) estimation
  - per-feature local crowding
  - Ripley's K / L (multi-scale clustering)
  - cross-label association (bivariate nearest-neighbour + Ripley's K)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


# ── centroid extraction ────────────────────────────────────────────────────────

def annotation_centroid(ann) -> tuple[float, float] | None:
    """(x, y) centroid in image pixels for a feature annotation, or None."""
    t = getattr(ann, "type", None)
    if t == "roi":
        verts = getattr(ann, "vertices", None)
        if not verts or len(verts) < 3:
            if verts:
                arr = np.asarray(verts, dtype=float)
                return float(arr[:, 0].mean()), float(arr[:, 1].mean())
            return None
        arr = np.asarray(verts, dtype=float)
        x, y = arr[:, 0], arr[:, 1]
        xr, yr = np.roll(x, -1), np.roll(y, -1)
        cross = x * yr - xr * y
        a = cross.sum() / 2.0
        if abs(a) < 1e-9:                      # degenerate polygon → vertex mean
            return float(x.mean()), float(y.mean())
        cx = ((x + xr) * cross).sum() / (6.0 * a)
        cy = ((y + yr) * cross).sum() / (6.0 * a)
        return float(cx), float(cy)
    if t == "circle":
        return float(ann.cx), float(ann.cy)
    if t == "rectangle":
        return (float(ann.x0 + ann.x1) / 2.0, float(ann.y0 + ann.y1) / 2.0)
    return None


def feature_label(ann) -> str:
    lbl = getattr(ann, "label", "") or ""
    lbl = lbl.strip()
    return lbl if lbl else "Unlabelled"


def extract_points(annotations, px_nm: float = 1.0,
                   labels: set[str] | None = None) -> dict[str, np.ndarray]:
    """Return {label: (N, 2) array of centroids in nm} for matching features."""
    out: dict[str, list] = {}
    for ann in annotations:
        if getattr(ann, "type", None) not in ("roi", "circle", "rectangle"):
            continue
        lbl = feature_label(ann)
        if labels is not None and lbl not in labels:
            continue
        c = annotation_centroid(ann)
        if c is None:
            continue
        out.setdefault(lbl, []).append((c[0] * px_nm, c[1] * px_nm))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


# ── univariate nearest-neighbour + Clark-Evans ──────────────────────────────────

@dataclass
class NNDResult:
    n: int = 0
    mean_nnd_nm: float = 0.0
    median_nnd_nm: float = 0.0
    expected_nnd_nm: float = 0.0
    clark_evans_R: float = 0.0
    z_score: float = 0.0
    p_value: float = 1.0
    p_montecarlo: float | None = None
    edge_corrected: bool = False
    verdict: str = ""
    nnd_nm: np.ndarray = field(default_factory=lambda: np.empty(0))


def _norm_sf2(z: float) -> float:
    """Two-sided normal tail probability."""
    return math.erfc(abs(z) / math.sqrt(2.0))


def _mean_nnd(points: np.ndarray) -> float:
    from scipy.spatial import cKDTree
    if len(points) < 2:
        return 0.0
    d, _ = cKDTree(points).query(points, k=2)
    return float(d[:, 1].mean())


def nearest_neighbour(points: np.ndarray, area_nm2: float,
                      width_nm: float | None = None, height_nm: float | None = None,
                      n_mc: int = 0, seed: int = 0) -> NNDResult:
    """Nearest-neighbour distances + Clark-Evans clustering index.

    If width/height are given, the expected NND uses Donnelly's edge correction
    (sparse patterns near the border no longer read as falsely "regular").
    If n_mc > 0, also computes a Monte-Carlo p-value by simulating n_mc complete
    spatial randomness (CSR) patterns in the same field — robust to edge effects.
    """
    n = len(points)
    if n < 2 or area_nm2 <= 0:
        return NNDResult(n=n, verdict="Too few points for nearest-neighbour analysis.")
    nnd = _nnd_array(points)
    mean_obs = float(nnd.mean())
    density = n / area_nm2
    expected = 0.5 / math.sqrt(density)
    edge_corrected = False
    if width_nm and height_nm:
        # Donnelly (1978) edge correction for a rectangular study region
        perim = 2.0 * (width_nm + height_nm)
        expected = (0.5 * math.sqrt(area_nm2 / n)
                    + (0.0514 + 0.041 / math.sqrt(n)) * perim / n)
        edge_corrected = True
    R = mean_obs / expected if expected > 0 else 0.0
    se = 0.26136 / math.sqrt(n * density)
    z = (mean_obs - expected) / se if se > 0 else 0.0
    p_analytic = _norm_sf2(z)

    p_mc = None
    if n_mc > 0 and width_nm and height_nm:
        rng = np.random.default_rng(seed)
        sims = np.array([_mean_nnd(_csr(n, width_nm, height_nm, rng)) for _ in range(n_mc)])
        # two-sided empirical p: fraction of sims at least as extreme as observed
        n_le = int((sims <= mean_obs).sum())
        n_ge = int((sims >= mean_obs).sum())
        p_mc = 2.0 * min(n_le, n_ge) / (n_mc + 1)
        p_mc = min(p_mc, 1.0)

    p = p_mc if p_mc is not None else p_analytic
    tag = "Monte-Carlo" if p_mc is not None else "analytic"
    if p < 0.05 and R < 1:
        verdict = f"Clustered (R={R:.2f}, p={p:.1e}, {tag})"
    elif p < 0.05 and R > 1:
        verdict = f"Regularly/evenly spaced (R={R:.2f}, p={p:.1e}, {tag})"
    else:
        verdict = f"Random / no significant pattern (R={R:.2f}, p={p:.2f}, {tag})"
    return NNDResult(
        n=n, mean_nnd_nm=mean_obs, median_nnd_nm=float(np.median(nnd)),
        expected_nnd_nm=expected, clark_evans_R=R, z_score=z, p_value=p_analytic,
        p_montecarlo=p_mc, edge_corrected=edge_corrected, verdict=verdict, nnd_nm=nnd,
    )


def _nnd_array(points: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    d, _ = cKDTree(points).query(points, k=2)
    return d[:, 1]


def _csr(n: int, width_nm: float, height_nm: float, rng) -> np.ndarray:
    """n points of complete spatial randomness in a [0,w]×[0,h] rectangle."""
    return np.column_stack([rng.uniform(0, width_nm, n), rng.uniform(0, height_nm, n)])


# ── DBSCAN clustering (scipy cKDTree) ────────────────────────────────────────────

@dataclass
class ClusterResult:
    labels: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))  # -1 = noise
    n_clusters: int = 0
    n_noise: int = 0
    cluster_sizes: list[int] = field(default_factory=list)


def dbscan(points: np.ndarray, eps_nm: float, min_samples: int = 3) -> ClusterResult:
    """Density-based clustering. labels[i] = cluster id (-1 = isolated/noise)."""
    n = len(points)
    if n == 0:
        return ClusterResult()
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    neighbours = tree.query_ball_point(points, eps_nm)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster = 0
    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbours[i]) < min_samples:
            continue                                  # provisional noise
        labels[i] = cluster
        seeds = list(neighbours[i])
        k = 0
        while k < len(seeds):
            j = seeds[k]; k += 1
            if labels[j] == -1:
                labels[j] = cluster                   # border point
            if not visited[j]:
                visited[j] = True
                if len(neighbours[j]) >= min_samples:  # core → expand
                    seeds.extend(neighbours[j])
        cluster += 1
    sizes = [int((labels == c).sum()) for c in range(cluster)]
    return ClusterResult(labels=labels, n_clusters=cluster,
                         n_noise=int((labels == -1).sum()), cluster_sizes=sizes)


# ── kernel-density hotspot ───────────────────────────────────────────────────────

def kde_grid(points: np.ndarray, width_nm: float, height_nm: float,
             n_grid: int = 200, bandwidth_nm: float | None = None):
    """Gaussian KDE evaluated on a grid over [0,width]×[0,height]. Returns (density, extent)."""
    if len(points) < 2:
        return None, None
    from scipy.stats import gaussian_kde
    xy = points.T
    try:
        kde = gaussian_kde(xy)
        if bandwidth_nm is not None and bandwidth_nm > 0:
            # set isotropic bandwidth in data units
            std = xy.std(axis=1).mean()
            if std > 0:
                kde.set_bandwidth(bandwidth_nm / std)
    except np.linalg.LinAlgError:
        return None, None
    gx = np.linspace(0, width_nm, n_grid)
    gy = np.linspace(0, height_nm, n_grid)
    mx, my = np.meshgrid(gx, gy)
    dens = kde(np.vstack([mx.ravel(), my.ravel()])).reshape(my.shape)
    return dens, (0, width_nm, height_nm, 0)   # extent matches image y-down


# ── per-feature local crowding ───────────────────────────────────────────────────

def local_density(points: np.ndarray, radius_nm: float) -> np.ndarray:
    """Number of OTHER features within radius_nm of each feature."""
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=int)
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    counts = tree.query_ball_point(points, radius_nm, return_length=True)
    return np.asarray(counts, dtype=int) - 1   # exclude self


# ── Ripley's K / L ───────────────────────────────────────────────────────────────

def _ripleys_l_values(points: np.ndarray, area_nm2: float, radii_nm: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    n = len(points)
    tree = cKDTree(points)
    lam = n / area_nm2
    out = np.empty(len(radii_nm))
    for i, r in enumerate(radii_nm):
        counts = tree.query_ball_point(points, r, return_length=True).sum() - n  # exclude self
        K = counts / (n * lam) if lam > 0 else 0.0
        L = math.sqrt(K / math.pi) if K > 0 else 0.0
        out[i] = L - r
    return out


def ripleys_l(points: np.ndarray, area_nm2: float, radii_nm: np.ndarray,
              width_nm: float | None = None, height_nm: float | None = None,
              n_mc: int = 0, seed: int = 0):
    """Ripley's L(r) − r (>0 = clustering at scale r).

    Returns (L, lo, hi): lo/hi are the Monte-Carlo CSR confidence envelope
    (2.5/97.5 percentile across n_mc simulations) or None when n_mc == 0.
    Observed L outside [lo, hi] is significant clustering/dispersion at that scale.
    """
    n = len(points)
    if n < 2 or area_nm2 <= 0:
        return None, None, None
    L = _ripleys_l_values(points, area_nm2, radii_nm)
    lo = hi = None
    if n_mc > 0 and width_nm and height_nm:
        rng = np.random.default_rng(seed)
        sims = np.array([
            _ripleys_l_values(_csr(n, width_nm, height_nm, rng), area_nm2, radii_nm)
            for _ in range(n_mc)
        ])
        lo = np.percentile(sims, 2.5, axis=0)
        hi = np.percentile(sims, 97.5, axis=0)
    return L, lo, hi


# ── cross-label (bivariate) association ──────────────────────────────────────────

@dataclass
class CrossResult:
    label_a: str = ""
    label_b: str = ""
    n_a: int = 0
    n_b: int = 0
    mean_cross_nnd_nm: float = 0.0
    expected_nnd_nm: float = 0.0
    association_R: float = 0.0
    z_score: float = 0.0
    p_value: float = 1.0
    p_montecarlo: float | None = None
    verdict: str = ""
    cross_nnd_nm: np.ndarray = field(default_factory=lambda: np.empty(0))


def cross_nearest_neighbour(points_a: np.ndarray, points_b: np.ndarray,
                            area_nm2: float, label_a: str = "A",
                            label_b: str = "B",
                            width_nm: float | None = None, height_nm: float | None = None,
                            n_mc: int = 0, seed: int = 0) -> CrossResult:
    """For each A feature, distance to the nearest B feature; association vs random.

    association_R < 1 ⇒ A sits closer to B than chance (associated/co-located);
    R > 1 ⇒ A avoids B (segregated). Uses B's density as the null.
    """
    na, nb = len(points_a), len(points_b)
    if na < 1 or nb < 1 or area_nm2 <= 0:
        return CrossResult(label_a=label_a, label_b=label_b, n_a=na, n_b=nb,
                           verdict="Too few features for cross-label analysis.")
    from scipy.spatial import cKDTree
    tree_b = cKDTree(points_b)
    d, _ = tree_b.query(points_a, k=1)
    mean_obs = float(np.mean(d))
    density_b = nb / area_nm2
    expected = 0.5 / math.sqrt(density_b)
    R = mean_obs / expected if expected > 0 else 0.0
    se = 0.26136 / math.sqrt(na * density_b)
    z = (mean_obs - expected) / se if se > 0 else 0.0
    p_analytic = _norm_sf2(z)

    p_mc = None
    if n_mc > 0 and width_nm and height_nm:
        rng = np.random.default_rng(seed)
        sims = np.empty(n_mc)
        for k in range(n_mc):
            # null: B independently placed; keep A fixed, randomise B in the field
            tb = cKDTree(_csr(nb, width_nm, height_nm, rng))
            sims[k] = float(tb.query(points_a, k=1)[0].mean())
        n_le = int((sims <= mean_obs).sum()); n_ge = int((sims >= mean_obs).sum())
        p_mc = min(2.0 * min(n_le, n_ge) / (n_mc + 1), 1.0)

    p = p_mc if p_mc is not None else p_analytic
    tag = "Monte-Carlo" if p_mc is not None else "analytic"
    if p < 0.05 and R < 1:
        verdict = f"{label_a} associated with / near {label_b} (R={R:.2f}, p={p:.1e}, {tag})"
    elif p < 0.05 and R > 1:
        verdict = f"{label_a} avoids / segregated from {label_b} (R={R:.2f}, p={p:.1e}, {tag})"
    else:
        verdict = f"{label_a} & {label_b} independently placed (R={R:.2f}, p={p:.2f}, {tag})"
    return CrossResult(
        label_a=label_a, label_b=label_b, n_a=na, n_b=nb,
        mean_cross_nnd_nm=mean_obs, expected_nnd_nm=expected, association_R=R,
        z_score=z, p_value=p_analytic, p_montecarlo=p_mc, verdict=verdict, cross_nnd_nm=d,
    )
