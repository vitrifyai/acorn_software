"""Bacteria-in-media specimen, driven by a species library, for multislice.

You name an organism; the library supplies its morphology (Gram type, shape,
size, appendages) and this module builds a 3-D cell -- envelope shells,
crowded cytoplasm (ribosomes), a ribosome-excluding nucleoid, and appendages --
then yields per-z-slab projected potentials for the multislice forward model.

Whole cells are often 300 nm - 1 um thick, beyond comfortable TEM phase
contrast: expect low contrast for thick cells (which is exactly why they get
FIB-milled -- the eventual lamella hand-off). Small/thin cells image well.

Potentials are mean-inner-potentials in volts, relative to vitreous ice.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .potential import MIP_VITREOUS_ICE
from .multislice import (multislice_exit_wave, coherent_ctf, image_from_wave)
from .optics import _tukey2d
from .setup import sigma_rad_per_VA

# Mean inner potentials (V). Rough, chosen for realistic *relative* contrast.
MIP = {
    "media": 4.8, "membrane": 8.5, "pg": 7.5, "periplasm": 5.5,
    "cytoplasm": 6.0, "ribosome": 10.0, "nucleoid": 5.6, "appendage": 7.5,
}

# Species library: morphology only; the physics is filled from MIP above.
SPECIES = {
    "e_coli":         dict(gram="neg", shape="rod",    length_nm=2000, width_nm=500, flagella=4, pili=True),
    "p_aeruginosa":   dict(gram="neg", shape="rod",    length_nm=2500, width_nm=600, flagella=1, pili=True),
    "caulobacter":    dict(gram="neg", shape="vibrio", length_nm=2000, width_nm=500, flagella=1, pili=False),
    "vibrio_cholerae":dict(gram="neg", shape="vibrio", length_nm=2500, width_nm=500, flagella=1, pili=False),
    "b_subtilis":     dict(gram="pos", shape="rod",    length_nm=4000, width_nm=900, flagella=6, pili=False),
    "s_aureus":       dict(gram="pos", shape="coccus", diameter_nm=1000,            flagella=0, pili=False),
    "streptococcus":  dict(gram="pos", shape="coccus", diameter_nm=900,             flagella=0, pili=False),
    "mycoplasma":     dict(gram="none",shape="coccus", diameter_nm=300,             flagella=0, pili=False),
}

# Envelope shells from the outer surface inward: (name, thickness_nm).
ENVELOPE = {
    "neg":  [("membrane", 0.6), ("periplasm", 1.3), ("membrane", 0.6)],  # OM / peri+PG / IM
    "pos":  [("pg", 3.0), ("membrane", 0.7)],                            # thick wall / membrane
    "none": [("membrane", 0.7)],                                         # mycoplasma: membrane only
}


@dataclass
class Bacterium:
    species: str = "e_coli"
    angle_deg: float = 15.0          # in-plane orientation of the cell long axis
    media_margin_nm: float = 25.0    # ice/media around the cell in z
    n_ribosomes: int = 350           # capped for speed (real cells have ~10^4)
    ribosome_diam_nm: float = 22.0
    nucleoid_frac: float = 0.45      # nucleoid ellipsoid size as fraction of cell
    solvent_noise: float = 5.0
    solvent_corr_a: float = 3.5
    seed: int = 0
    overrides: dict = field(default_factory=dict)   # override any species field

    def morph(self):
        if self.species not in SPECIES:
            raise ValueError(f"unknown species {self.species!r}; known: {list(SPECIES)}")
        return {**SPECIES[self.species], **self.overrides}

    def thickness_nm(self):
        m = self.morph()
        w = m.get("width_nm") or m.get("diameter_nm")
        return w + 2 * self.media_margin_nm


def _seg_distance_2d(shape, A, B):
    """Distance from every pixel (in px) to the 2-D segment A->B (px)."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ax, ay = A; bx, by = B
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy + 1e-8
    t = np.clip(((xx - ax) * dx + (yy - ay) * dy) / L2, 0.0, 1.0)
    px_, py_ = ax + t * dx, ay + t * dy
    return np.sqrt((xx - px_) ** 2 + (yy - py_) ** 2)


def bacterium_slabs(shape, px, dz_a, bact: Bacterium):
    """Yield per-z-slab projected potential (V*Å) through a cell in media."""
    rng = np.random.default_rng(bact.seed)
    h, w = shape
    m = bact.morph()
    ice = MIP_VITREOUS_ICE

    # Cell geometry (Å) -> px. Long axis through the image centre at angle.
    width_a = (m.get("width_nm") or m.get("diameter_nm")) * 10.0
    Rcell = 0.5 * width_a
    length_a = (m.get("length_nm", 0.0)) * 10.0
    half = max(0.0, 0.5 * (length_a - width_a))          # half length of the cylinder axis
    th = np.deg2rad(bact.angle_deg)
    ux, uy = np.cos(th), np.sin(th)
    cx_px, cy_px = w / 2.0, h / 2.0
    A = (cx_px - half / px * ux, cy_px - half / px * uy)
    B = (cx_px + half / px * ux, cy_px + half / px * uy)
    d2d_px = _seg_distance_2d(shape, A, B)               # precompute once
    d2d_a = d2d_px * px

    shells = ENVELOPE[m["gram"]]
    env_total_a = sum(t for _, t in shells) * 10.0

    # z geometry
    thick_a = bact.thickness_nm() * 10.0
    z0 = thick_a / 2.0
    nz = max(1, int(np.ceil(thick_a / dz_a)))
    Rribo = bact.ribosome_diam_nm * 10.0 / 2.0

    # Nucleoid ellipsoid (cell-frame): central, ribosome-excluded.
    nf = bact.nucleoid_frac
    nuc_rz = nf * Rcell

    # Place ribosomes in 3-D cytoplasm, outside the nucleoid.
    ribos = []
    tries = 0
    inner_a = Rcell - env_total_a - Rribo
    while len(ribos) < bact.n_ribosomes and tries < bact.n_ribosomes * 40:
        tries += 1
        rx = rng.uniform(0, w); ry = rng.uniform(0, h); rz = rng.uniform(0, thick_a)
        # distance of this point to the cell surface
        d2 = _point_seg_dist(rx, ry, A, B) * px
        d3 = np.sqrt(d2 ** 2 + (rz - z0) ** 2) - Rcell
        if d3 < -(env_total_a + Rribo):                  # safely inside cytoplasm
            # exclude nucleoid core (ellipsoid ~ inner region)
            if (d3 / (inner_a + 1e-6)) ** 2 + ((rz - z0) / nuc_rz) ** 2 > 1.0 or nf <= 0:
                ribos.append((rx, ry, rz, Rribo))

    corr_px = max(0.5, bact.solvent_corr_a / px)
    # Per-slab ice-potential RMS (V*Å): ~sqrt(dz) random along depth, and
    # independent of lateral pixel size (else it explodes at coarse px over
    # many slabs and scrambles the wave).
    amp = bact.solvent_noise * np.sqrt(dz_a / 10.0)

    for iz in range(nz):
        z = (iz + 0.5) * dz_a
        d3d_a = np.sqrt(d2d_a ** 2 + (z - z0) ** 2) - Rcell   # signed dist to surface
        dd = -d3d_a                                            # depth inside (Å)

        mip = np.full(shape, MIP["media"], dtype=np.float32)   # default: media
        inside = dd >= 0
        mip[inside] = MIP["cytoplasm"]
        acc = 0.0
        for name, t_nm in shells:                              # envelope shells
            t_a = t_nm * 10.0
            shell = inside & (dd >= acc) & (dd < acc + t_a)
            mip[shell] = MIP[name]
            acc += t_a

        # Nucleoid: central ellipsoid region -> nucleoid MIP (replaces cytoplasm).
        if nf > 0:
            ell = ((d3d_a / (inner_a + 1e-6)) ** 2 + ((z - z0) / nuc_rz) ** 2) < 1.0
            mip[ell & (dd >= env_total_a)] = MIP["nucleoid"]

        V = (mip - ice) * dz_a

        # Ribosomes intersecting this slab (additive excess over cytoplasm).
        for (rx, ry, rz, R) in ribos:
            if abs(z - rz) < R:
                rho_px = np.sqrt(R ** 2 - (z - rz) ** 2) / px
                _disk_add(V, ry, rx, rho_px, (MIP["ribosome"] - MIP["cytoplasm"]) * dz_a)

        # Flagella: thin filaments from a pole, only near the cell mid-plane.
        if m.get("flagella") and abs(z - z0) < 0.15 * width_a:
            _add_flagella(V, shape, px, B, ux, uy, m["flagella"], length_a, dz_a, rng)

        if bact.solvent_noise > 0:
            nfld = gaussian_filter(rng.standard_normal(shape).astype(np.float32), corr_px)
            V += amp * (nfld / (nfld.std() + 1e-8))
        yield V


def _point_seg_dist(x, y, A, B):
    ax, ay = A; bx, by = B
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy + 1e-8
    t = min(1.0, max(0.0, ((x - ax) * dx + (y - ay) * dy) / L2))
    return np.hypot(x - (ax + t * dx), y - (ay + t * dy))


def _disk_add(out, cy, cx, r_px, val):
    h, w = out.shape
    y0, y1 = max(0, int(cy - r_px - 1)), min(h, int(cy + r_px + 2))
    x0, x1 = max(0, int(cx - r_px - 1)), min(w, int(cx + r_px + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] += val * ((yy - cy) ** 2 + (xx - cx) ** 2 < r_px ** 2)


def _add_flagella(V, shape, px, pole_px, ux, uy, n, length_a, dz_a, rng):
    """A few thin sinusoidal filaments trailing from one pole (schematic)."""
    h, w = shape
    L = 0.6 * max(length_a, 1000.0) / px               # filament length (px)
    val = (MIP["appendage"] - MIP["media"]) * dz_a
    perp = (-uy, ux)
    for _ in range(int(n)):
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(4, 9)
        for s in np.linspace(0, L, int(L)):
            wig = amp * np.sin(0.05 * s + phase)
            x = pole_px[0] + ux * s + perp[0] * wig
            y = pole_px[1] + uy * s + perp[1] * wig
            if 0 <= int(y) < h and 0 <= int(x) < w:
                V[int(y), int(x)] += val


def simulate_bacterium(cfg, bact: Bacterium | None = None, defocus_um=-4.0,
                       dz_a=40.0, bfactor=60.0, apodize=0.1):
    """Full multislice forward model for a bacterium in media (noiseless image)."""
    bact = bact or Bacterium()
    n = int(cfg["image_size_px"]); shape = (n, n); px = cfg["pixel_size_a"]
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))
    nz = max(1, int(np.ceil(bact.thickness_nm() * 10.0 / dz_a)))

    psi = multislice_exit_wave(shape, px, dz_a,
                               bacterium_slabs(shape, px, dz_a, bact),
                               lam, sigma, q=cfg["amplitude_contrast"])
    if apodize:
        # Taper toward the BORDER (media) value, not vacuum: the cell sits in
        # media that fills the frame, so tapering to 1 would paint a bright
        # artificial rim. Border-mean taper is seamless.
        b = int(max(2, apodize * min(shape)))
        edge = np.concatenate([psi[:b].ravel(), psi[-b:].ravel(),
                               psi[:, :b].ravel(), psi[:, -b:].ravel()])
        edge_val = edge.mean()
        psi = edge_val + (psi - edge_val) * _tukey2d(shape, apodize)
    ideal = image_from_wave(psi, coherent_ctf(shape, px, cfg, defocus_um, bfactor))
    ideal = ideal / (ideal.mean() + 1e-8)
    from .multislice import apply_inelastic
    ideal, dose_scale = apply_inelastic(ideal, bact.thickness_nm(), cfg)
    return ideal, {"species": bact.species, "thickness_nm": bact.thickness_nm(),
                   "n_slabs": nz, "defocus_um": defocus_um, "dose_scale": dose_scale}
