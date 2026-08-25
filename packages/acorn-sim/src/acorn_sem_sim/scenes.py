"""Specimen phantoms for SEM simulation.

Each builder returns a `Scene`: a material label map, the material names it
indexes, and a height field. The label map is the segmentation ground truth --
which is the whole reason for simulating rather than collecting.

Scenes are deliberately chosen to separate the two contrast mechanisms an SEM
image mixes together, because a benchmark that cannot tell them apart cannot say
what a segmentation model is actually keying on:

    composition only   flat surface, differing Z          -> Z contrast alone
    topography only    one material, varying height       -> shading alone
    both               the realistic and harder case
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scene:
    """A specimen and the truth about it."""

    material_index: np.ndarray
    material_names: list[str]
    height_nm: np.ndarray
    pixel_size_nm: float
    description: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.material_index.shape


def _smooth_noise(shape, rng, sigma_px, amplitude=1.0):
    from scipy.ndimage import gaussian_filter
    n = rng.standard_normal(shape).astype(np.float32)
    n = gaussian_filter(n, sigma=max(0.5, float(sigma_px)))
    s = float(n.std())
    return (n / s * amplitude) if s > 0 else n


def _disc(shape, cy, cx, r):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r


def nanoparticles_on_substrate(shape=(512, 512), pixel_size_nm=2.0, n_particles=40,
                               diameter_nm_mean=40.0, diameter_nm_sd=12.0,
                               particle="gold", substrate="carbon",
                               relief=True, seed=0) -> Scene:
    """Discrete heavy particles on a light support -- the classic Z-contrast case.

    Particles sit *on* the substrate, so they carry real topography as well as a
    composition difference. Set `relief=False` to flatten them and isolate pure
    Z contrast, which is the control condition for asking whether a model is
    keying on composition or on edges.
    """
    rng = np.random.default_rng(seed)
    idx = np.zeros(shape, dtype=np.int32)
    h = np.zeros(shape, dtype=np.float32)
    placed = []

    for _ in range(int(n_particles) * 8):
        if len(placed) >= n_particles:
            break
        d_nm = float(rng.normal(diameter_nm_mean, diameter_nm_sd))
        r_px = 0.5 * d_nm / pixel_size_nm
        if r_px < 1.0:
            continue
        cy = rng.uniform(r_px, shape[0] - r_px)
        cx = rng.uniform(r_px, shape[1] - r_px)
        if any((cy - py) ** 2 + (cx - px) ** 2 < (r_px + pr) ** 2 for py, px, pr in placed):
            continue
        placed.append((cy, cx, r_px))
        m = _disc(shape, cy, cx, r_px)
        idx[m] = 1
        if relief:
            yy, xx = np.ogrid[:shape[0], :shape[1]]
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            cap = np.sqrt(np.clip(r_px ** 2 - rr ** 2, 0, None)) * pixel_size_nm
            h = np.maximum(h, cap.astype(np.float32))

    if relief:
        h = h + _smooth_noise(shape, rng, 8, 1.5)      # slight substrate roughness

    return Scene(idx, [substrate, particle], h, pixel_size_nm,
                 f"{len(placed)} {particle} particles on {substrate}",
                 {"n_particles": len(placed), "relief": bool(relief),
                  "diameter_nm_mean": diameter_nm_mean})


def two_phase_grains(shape=(512, 512), pixel_size_nm=5.0, n_grains=24,
                     phase_a="iron", phase_b="copper", seed=0) -> Scene:
    """A polished two-phase alloy: flat, so all contrast is compositional.

    Built as a Voronoi tessellation with phases assigned per grain. Because the
    surface is flat this isolates Z contrast completely -- useful for showing
    that a BSE detector separates the phases where an SE detector barely does.
    """
    rng = np.random.default_rng(seed)
    seeds_y = rng.uniform(0, shape[0], n_grains)
    seeds_x = rng.uniform(0, shape[1], n_grains)
    phase = rng.integers(0, 2, n_grains)

    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    d2 = ((yy[None] - seeds_y[:, None, None]) ** 2
          + (xx[None] - seeds_x[:, None, None]) ** 2)
    nearest = np.argmin(d2, axis=0)
    idx = phase[nearest].astype(np.int32)

    return Scene(idx, [phase_a, phase_b], np.zeros(shape, np.float32), pixel_size_nm,
                 f"{n_grains}-grain {phase_a}/{phase_b} alloy, polished flat",
                 {"n_grains": int(n_grains), "flat": True})


def porous_surface(shape=(512, 512), pixel_size_nm=4.0, porosity=0.25,
                   pore_scale_px=14.0, matrix="alumina", seed=0) -> Scene:
    """Open pores in a ceramic matrix. Topography and composition together.

    Pores are vacuum, so they emit nothing -- they read as genuinely black
    rather than merely dark, which is what makes pore segmentation look easy in
    SEM and is worth having as an easy end of a difficulty range.
    """
    rng = np.random.default_rng(seed)
    fieldn = _smooth_noise(shape, rng, pore_scale_px)
    thresh = np.quantile(fieldn, porosity)
    pores = fieldn < thresh

    idx = np.zeros(shape, dtype=np.int32)
    idx[pores] = 1                                   # 1 = vacuum (a pore)
    depth_nm = pore_scale_px * pixel_size_nm
    h = np.where(pores, -depth_nm, 0.0).astype(np.float32)
    from scipy.ndimage import gaussian_filter
    h = gaussian_filter(h, 1.5)

    return Scene(idx, [matrix, "vacuum"], h, pixel_size_nm,
                 f"porous {matrix}, {porosity:.0%} porosity",
                 {"porosity": float(pores.mean())})


def biological_surface(shape=(512, 512), pixel_size_nm=8.0, n_cells=14, seed=0) -> Scene:
    """Resin-embedded biological material: almost no Z contrast, all topography.

    The hard case, and the honest one for life-science SEM. Cells and resin
    differ by well under one unit of mean atomic number, so a BSE detector sees
    essentially nothing and the image lives or dies on surface relief. Any
    method that looked good on gold-on-carbon should be re-checked here.
    """
    rng = np.random.default_rng(seed)
    idx = np.zeros(shape, dtype=np.int32)
    h = _smooth_noise(shape, rng, 24, 40.0)

    for _ in range(n_cells):
        r = rng.uniform(0.05, 0.12) * min(shape)
        cy = rng.uniform(r, shape[0] - r)
        cx = rng.uniform(r, shape[1] - r)
        m = _disc(shape, cy, cx, r)
        idx[m] = 1
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        bulge = np.sqrt(np.clip(r ** 2 - rr ** 2, 0, None)) * pixel_size_nm * 0.35
        h = h + bulge.astype(np.float32)

    return Scene(idx, ["resin", "biology"], h.astype(np.float32), pixel_size_nm,
                 f"{n_cells} cells in resin -- topographic contrast only",
                 {"n_cells": int(n_cells), "z_contrast": "negligible"})


def fib_cross_section(shape=(512, 512), pixel_size_nm=5.0, n_layers=4, seed=0) -> Scene:
    """A FIB-milled cross-section: Pt cap over stacked layers.

    Mirrors what the FIB-SEM plugin produces geometrically, but imaged through
    the transport model, so the Pt cap gets Pt's compact interaction volume and
    the layers below get their own.
    """
    rng = np.random.default_rng(seed)
    idx = np.zeros(shape, dtype=np.int32)
    names = ["silicon", "platinum", "silica", "copper", "carbon"][:max(2, n_layers + 1)]

    cap = int(shape[0] * 0.18)
    idx[:cap] = 1                                       # Pt protective cap
    rest = shape[0] - cap
    bounds = np.sort(rng.uniform(0, 1, max(0, len(names) - 2)))
    edges = [cap] + [cap + int(b * rest) for b in bounds] + [shape[0]]
    for li in range(len(edges) - 1):
        lab = 2 + li if (2 + li) < len(names) else 0
        idx[edges[li]:edges[li + 1]] = lab

    h = _smooth_noise(shape, rng, 3, 8.0)               # residual curtaining relief

    return Scene(idx.astype(np.int32), names, h.astype(np.float32), pixel_size_nm,
                 f"FIB cross-section, Pt cap over {len(names) - 1} layers",
                 {"layers": names})


BUILDERS = {
    "nanoparticles": nanoparticles_on_substrate,
    "grains":        two_phase_grains,
    "porous":        porous_surface,
    "biological":    biological_surface,
    "cross_section": fib_cross_section,
}

# Plain-language labels for the UI and for CLU to match against.
SCENE_LABELS = {
    "nanoparticles": "Nanoparticles on a substrate",
    "grains":        "Two-phase alloy grains (flat)",
    "porous":        "Porous ceramic",
    "biological":    "Cells in resin (topography only)",
    "cross_section": "FIB cross-section with Pt cap",
}


def build(kind: str, **kwargs) -> Scene:
    """Build a scene by name, with an error that lists the alternatives."""
    try:
        builder = BUILDERS[kind]
    except KeyError:
        raise KeyError(f"unknown scene {kind!r}. Available: "
                       + ", ".join(sorted(BUILDERS))) from None
    import inspect
    allowed = set(inspect.signature(builder).parameters)
    return builder(**{k: v for k, v in kwargs.items() if k in allowed})
