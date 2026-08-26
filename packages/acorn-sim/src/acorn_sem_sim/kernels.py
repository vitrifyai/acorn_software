"""Interaction-volume kernels: the bridge from electron trajectories to images.

Tracing electrons per pixel would be hopeless -- a 1024^2 image at 10^4
trajectories each is 10^10 trajectories. The standard escape is that for a
laterally homogeneous material at a fixed beam energy and tilt, the lateral
distribution of escaping electrons is a *fixed radial kernel*. Trace once,
convolve forever.

So each (material, energy, tilt) yields:

    K_SE    where secondaries that escape were generated, radially
    K_BSE   where backscattered electrons cross back out, radially
    delta   secondary yield
    eta     backscatter yield

K_SE is the interesting one. It is not a Gaussian and not close to one: at 5 keV
in silicon its median radius is 0.40 nm while its 95th percentile is 209 nm. That
is the SE1/SE2 split -- a sub-nanometre core from the incident probe sitting on a
pedestal hundreds of nanometres wide contributed by backscatters re-crossing the
surface. Every single-Gaussian MTF model collapses those two into one width and
so cannot reproduce SEM resolution behaviour at all.

Kernels are cached on disk because tracing 10^5 electrons takes seconds and the
same handful of (material, energy, tilt) combinations recur constantly.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .materials import Material
from .transport import trace

# Tilts at which yields are precomputed; intermediate angles are interpolated.
# The grid is denser near grazing incidence because both yields climb steeply
# there -- that steepness is what makes topographic contrast visible at all.
TILT_GRID_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 70.0, 80.0)

_CACHE_VERSION = 3


def cache_dir() -> Path:
    """Where kernels live. Honours XDG_CACHE_HOME."""
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    d = Path(base) / "acorn" / "sem_kernels"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class RadialKernel:
    """An azimuthally symmetric PSF, stored as enclosed weight vs radius.

    The primary representation is the CDF -- `cum[i]` is the fraction of escaping
    electrons landing within `r_nm[i]` -- not the areal density. That choice is
    load-bearing. The density of a near-point-like source diverges as 1/r^2 at
    the origin, so reconstructing enclosed weight by integrating a binned
    density puts the error exactly where the SE1 core lives: doing it that way
    reported the silicon core at 1.7 nm when the trajectories say 0.40 nm.
    Accumulating the histogram directly is exact.

    Radius, not pixels, keeps one cached trace valid at every magnification.
    """

    r_nm: np.ndarray
    cum:  np.ndarray

    def enclosed(self) -> np.ndarray:
        """Fraction of total weight inside each radius in `r_nm`. Exact."""
        return self.cum

    @property
    def density(self) -> np.ndarray:
        """Areal density per nm^2, derived. For inspection and plotting only."""
        area = np.pi * self.r_nm ** 2
        d = np.zeros_like(self.cum)
        da = np.diff(area)
        d[1:] = np.where(da > 0, np.diff(self.cum) / np.maximum(da, 1e-30), 0.0)
        d[0] = d[1] if d.size > 1 else 0.0
        return d

    def radius_containing(self, frac: float) -> float:
        """Radius enclosing `frac` of the total weight."""
        enc = self.enclosed()
        if enc[-1] <= 0:
            return 0.0
        return float(np.interp(frac, enc, self.r_nm))

    def to_pixels(self, pixel_size_nm: float, max_radius_nm: float | None = None) -> np.ndarray:
        """Rasterise onto a square pixel grid, normalised to sum 1.

        Built from the enclosed-weight function rather than by sampling the
        density: pixels are grouped into annuli one pixel wide, each annulus is
        given exactly the weight the CDF says lies between its inner and outer
        radius, and that weight is shared equally among the (equal-area) pixels
        in it. Weight is conserved by construction, the singular core lands
        wholly in the centre pixel, and the centre fraction rises with pixel
        size the way it must.
        """
        if pixel_size_nm <= 0:
            raise ValueError("pixel_size_nm must be positive")
        r_max = float(max_radius_nm if max_radius_nm is not None
                      else self.radius_containing(0.99))
        half = max(int(np.ceil(max(r_max, pixel_size_nm) / pixel_size_nm)), 1)

        n = 2 * half + 1
        ax = (np.arange(n) - half) * pixel_size_nm
        yy, xx = np.meshgrid(ax, ax, indexing="ij")
        rr = np.sqrt(xx ** 2 + yy ** 2)

        enc = self.enclosed()
        idx = np.floor(rr / pixel_size_nm + 0.5).astype(np.int64)
        r_out = (idx + 0.5) * pixel_size_nm
        r_in = np.maximum((idx - 0.5) * pixel_size_nm, 0.0)
        w_out = np.interp(r_out, self.r_nm, enc, left=0.0, right=1.0)
        w_in = np.interp(r_in, self.r_nm, enc, left=0.0, right=1.0)

        counts = np.bincount(idx.ravel())
        k = np.where(counts[idx] > 0, (w_out - w_in) / np.maximum(counts[idx], 1), 0.0)

        total = k.sum()
        return (k / total).astype(np.float32) if total > 0 else k.astype(np.float32)


@dataclass
class SEMKernels:
    """Everything the imaging model needs for one material at one beam energy."""

    material_name: str
    E0_kev:        float
    se:            RadialKernel
    bse:           RadialKernel
    tilt_deg:      np.ndarray      # the tilt grid
    delta:         np.ndarray      # SE yield at each tilt
    eta:           np.ndarray      # BSE yield at each tilt
    n_electrons:   int

    def delta_at(self, tilt_deg):
        """SE yield at arbitrary tilt, interpolated on the precomputed grid.

        Note what this is *not*: it is not an imposed 1/cos(theta) secant law.
        The tilt dependence was measured from the transport, so it saturates at
        grazing incidence the way real yields do instead of diverging.
        """
        return np.interp(np.abs(tilt_deg), self.tilt_deg, self.delta)

    def eta_at(self, tilt_deg):
        return np.interp(np.abs(tilt_deg), self.tilt_deg, self.eta)


def _radial_kernel(radii_nm: np.ndarray, weights: np.ndarray | None,
                   n_bins: int = 400) -> RadialKernel:
    """Weighted exit radii -> areal density vs radius, on a log radial grid.

    Log spacing is not cosmetic: the SE distribution spans sub-nanometre to
    sub-micrometre in one kernel, and linear bins would put essentially all
    resolution in the irrelevant outer decade.
    """
    if radii_nm.size == 0:
        return RadialKernel(np.array([0.0, 1.0]), np.array([0.0, 0.0]))

    w = np.ones_like(radii_nm) if weights is None else weights
    r_hi = max(float(np.percentile(radii_nm, 99.9)), 1e-2)
    lo = max(r_hi * 1e-5, 1e-4)
    edges = np.concatenate([[0.0], np.geomspace(lo, r_hi, n_bins)])

    hist, _ = np.histogram(radii_nm, bins=edges, weights=w)
    # Accumulate the histogram itself. Enclosed weight at each bin EDGE is then
    # exact up to the binning, with no integration of a divergent density.
    cum = np.concatenate([[0.0], np.cumsum(hist)])
    total = cum[-1]
    if total > 0:
        cum = cum / total
    return RadialKernel(edges.astype(np.float64), cum.astype(np.float64))


def _cache_key(material: Material, E0_kev: float, n_electrons: int, seed: int) -> str:
    raw = (f"v{_CACHE_VERSION}|{material.name}|{material.Z}|{material.A}|"
           f"{material.rho}|{material.epsilon_ev}|{material.se_escape_nm}|"
           f"{E0_kev}|{n_electrons}|{seed}|{TILT_GRID_DEG}")
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def compute(material: Material, E0_kev: float = 5.0,
            n_electrons: int = 60_000, seed: int = 0,
            use_cache: bool = True) -> SEMKernels:
    """Trace electrons and reduce them to kernels + tilt-dependent yields.

    Costs one Monte Carlo run per tilt on `TILT_GRID_DEG`, so roughly seven
    times the single-run cost -- a few tens of seconds at the default electron
    count. Cached on disk thereafter.
    """
    key = _cache_key(material, E0_kev, n_electrons, seed)
    path = cache_dir() / f"{key}.npz"
    if use_cache and path.exists():
        try:
            z = np.load(path)
            return SEMKernels(
                material_name=str(z["material_name"]),
                E0_kev=float(z["E0_kev"]),
                se=RadialKernel(z["se_r"], z["se_cum"]),
                bse=RadialKernel(z["bse_r"], z["bse_cum"]),
                tilt_deg=z["tilt_deg"], delta=z["delta"], eta=z["eta"],
                n_electrons=int(z["n_electrons"]),
            )
        except Exception:      # deliberate; see below
            # A corrupt, truncated or stale cache entry must never break a
            # simulation: fall through and recompute. There is nothing to log
            # to and nothing the caller could do differently.
            pass

    deltas, etas = [], []
    se_ref = bse_ref = None
    for i, tilt in enumerate(TILT_GRID_DEG):
        r = trace(material, E0_kev=E0_kev, n_electrons=n_electrons,
                  tilt_deg=tilt, seed=seed + i)
        deltas.append(r.delta)
        etas.append(r.eta)
        if i == 0:                      # kernels taken at normal incidence
            se_ref = _radial_kernel(r.se_r_nm, r.se_weight)
            bse_ref = _radial_kernel(r.bse_r_nm, None)

    out = SEMKernels(material.name, float(E0_kev), se_ref, bse_ref,
                     np.array(TILT_GRID_DEG, dtype=float),
                     np.array(deltas, dtype=float), np.array(etas, dtype=float),
                     int(n_electrons))

    if use_cache:
        try:
            np.savez_compressed(
                path, material_name=out.material_name, E0_kev=out.E0_kev,
                se_r=out.se.r_nm, se_cum=out.se.cum,
                bse_r=out.bse.r_nm, bse_cum=out.bse.cum,
                tilt_deg=out.tilt_deg, delta=out.delta, eta=out.eta,
                n_electrons=out.n_electrons)
        except OSError:
            pass        # an unwritable cache directory must not break simulation
    return out
