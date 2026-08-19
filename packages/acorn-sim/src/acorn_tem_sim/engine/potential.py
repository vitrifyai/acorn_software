"""Specimen -> projected electrostatic potential (the physics input to imaging).

Electrons see the specimen's Coulomb potential, not its 'density'. For a thin
specimen the relevant quantity is the PROJECTED potential V_proj(x,y) = integral
of V along the beam, in V*Angstrom; the exit-wave phase is phi = sigma * V_proj.

v1 specimen: solid PLGA nanoparticles embedded in a vitreous-ice slab.

Contrast comes from the DIFFERENCE between particle and ice mean-inner-potential
(MIP): a uniform ice slab is a constant phase offset (invisible), so we build
V_proj relative to ice and only the (V_plga - V_ice) excess of each particle,
plus the ice's structural noise, carry signal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

# Mean inner potentials (volts). Ice/protein are literature values; PLGA is
# estimated from a C/H/O polymer at rho ~ 1.34 g/cm^3 (protein-like).
MIP_VITREOUS_ICE = 4.5
MIP_PLGA = 8.0


MIP_CRYSTALLINE_ICE = 5.4    # hexagonal/cubic ice: denser than amorphous ice


@dataclass
class Specimen:
    kind: str = "plga"
    ice_thickness_nm: float = 40.0
    n_particles: int = 40
    diameter_nm_mean: float = 30.0
    diameter_nm_sd: float = 8.0
    plga_mip_v: float = MIP_PLGA
    ice_mip_v: float = MIP_VITREOUS_ICE
    # RMS ice structural-noise potential, per sqrt(nm) of ice (tuning knob).
    solvent_noise: float = 6.0
    solvent_corr_a: float = 3.5          # ice noise correlation length (Angstrom)
    allow_overlap: bool = False
    # Crystalline-ice contamination: faceted surface crystals (compose with any
    # scene, or set kind='contamination' for a pure contamination field).
    n_contam: int = 0
    contam_thickness_nm: float = 80.0    # crystals are thick -> strong dark contrast
    contam_mip_v: float = MIP_CRYSTALLINE_ICE
    seed: int = 0


@dataclass
class PotentialResult:
    v_proj: np.ndarray      # projected potential relative to ice (V*Angstrom)
    label: np.ndarray       # per-pixel projected particle thickness (Angstrom) -> GT
    specimen: Specimen


def _sphere_chord(shape, cy, cx, R_px):
    """Projected chord length map (in pixels) through a sphere of radius R_px."""
    h, w = shape
    y0, y1 = max(0, int(cy - R_px - 1)), min(h, int(cy + R_px + 2))
    x0, x1 = max(0, int(cx - R_px - 1)), min(w, int(cx + R_px + 2))
    if y0 >= y1 or x0 >= x1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    r2 = (yy - cy) ** 2 + (xx - cx) ** 2
    inside = r2 < R_px ** 2
    chord = np.zeros_like(r2)
    chord[inside] = 2.0 * np.sqrt(R_px ** 2 - r2[inside])   # in pixels
    return (y0, y1, x0, x1), chord


def nm_to_px(nm, px):
    return nm * 10.0 / px


def _fill_polygon(V, verts_px, val):
    """Fill a polygon (Nx2 px vertices) into V, adding `val` inside it."""
    from matplotlib.path import Path as MplPath
    h, w = V.shape
    xs, ys = verts_px[:, 0], verts_px[:, 1]
    x0, x1 = max(0, int(np.floor(xs.min()))), min(w, int(np.ceil(xs.max())) + 1)
    y0, y1 = max(0, int(np.floor(ys.min()))), min(h, int(np.ceil(ys.max())) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    pts = np.stack([xx.ravel(), yy.ravel()], axis=1)
    mask = MplPath(verts_px).contains_points(pts).reshape(y1 - y0, x1 - x0)
    V[y0:y1, x0:x1] += val * mask


def _draw_contamination(v_proj, shape, px, specimen, rng):
    """Faceted crystalline-ice crystals: hexagonal plates + a few needles.

    Crystals are thick and denser than amorphous ice, so they read as sharp,
    dark, high-contrast angular shapes -- the classic ice-contamination look."""
    h, w = shape
    excess = specimen.contam_mip_v - specimen.ice_mip_v
    for _ in range(specimen.n_contam):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        R = rng.uniform(nm_to_px(60, px), nm_to_px(400, px))
        phi = rng.uniform(0, np.pi)
        t_a = max(5.0, rng.lognormal(np.log(specimen.contam_thickness_nm), 0.4)) * 10.0
        val = excess * t_a
        if rng.random() < 0.25:                          # elongated needle/shard
            sx, sy = rng.uniform(2.5, 5.0), rng.uniform(0.4, 0.8)
        else:                                            # hexagonal plate
            sx, sy = 1.0, rng.uniform(0.8, 1.0)
        ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
        j = 1 + rng.uniform(-0.12, 0.12, 6)
        vx = cx + R * j * (sx * np.cos(ang) * np.cos(phi) - sy * np.sin(ang) * np.sin(phi))
        vy = cy + R * j * (sx * np.cos(ang) * np.sin(phi) + sy * np.sin(ang) * np.cos(phi))
        _fill_polygon(v_proj, np.stack([vx, vy], axis=1), val)


def make_potential(shape, pixel_size_a, specimen: Specimen) -> PotentialResult:
    rng = np.random.default_rng(specimen.seed)
    h, w = shape
    v_proj = np.zeros(shape, dtype=np.float32)
    label = np.zeros(shape, dtype=np.float32)

    if specimen.kind not in ("plga", "contamination"):
        raise ValueError(f"kind must be 'plga' or 'contamination', got {specimen.kind!r}")

    n_particles = 0 if specimen.kind == "contamination" else specimen.n_particles
    d_excess = specimen.plga_mip_v - specimen.ice_mip_v     # V, contrast source
    placed = []
    tries = 0
    while len(placed) < n_particles and tries < max(1, n_particles) * 50:
        tries += 1
        d_nm = rng.normal(specimen.diameter_nm_mean, specimen.diameter_nm_sd)
        if d_nm <= 2:
            continue
        R_px = (d_nm * 10.0 / 2.0) / pixel_size_a           # nm->A->px
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        if not specimen.allow_overlap and any(
                np.hypot(cy - py, cx - px) < (R_px + pr) for py, px, pr in placed):
            continue
        res = _sphere_chord(shape, cy, cx, R_px)
        if res is None:
            continue
        (y0, y1, x0, x1), chord_px = res
        chord_a = chord_px * pixel_size_a                    # chord length in Angstrom
        v_proj[y0:y1, x0:x1] += (d_excess * chord_a).astype(np.float32)
        label[y0:y1, x0:x1] = np.maximum(label[y0:y1, x0:x1], chord_a)
        placed.append((cy, cx, R_px))

    # Crystalline-ice contamination (composes with particles, or stands alone).
    if specimen.n_contam > 0:
        v_before = v_proj.copy()
        _draw_contamination(v_proj, shape, pixel_size_a, specimen, rng)
        label = np.maximum(label, (v_proj - v_before) > 0)   # contamination mask

    # Vitreous-ice structural noise: random potential fluctuation, correlated at
    # ~molecular scale, scaling with sqrt(thickness). This is the dominant
    # background in real low-dose cryo images (and it IS CTF-modulated, so we add
    # it to the potential -- before the CTF -- not to the final picture).
    if specimen.solvent_noise > 0:
        white = rng.standard_normal(shape).astype(np.float32)
        corr_px = max(0.5, specimen.solvent_corr_a / pixel_size_a)
        field_ = gaussian_filter(white, sigma=corr_px)
        field_ /= (field_.std() + 1e-8)
        amp = specimen.solvent_noise * np.sqrt(specimen.ice_thickness_nm) * pixel_size_a
        v_proj = v_proj + amp * field_

    v_proj -= v_proj.mean()      # drop the constant ice offset (no contrast)
    return PotentialResult(v_proj=v_proj, label=label, specimen=specimen)
