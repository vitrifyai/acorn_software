"""Unified composable scene: put nanoparticles + contamination + a cell into ONE
specimen, on a single z-grid, imaged in one multislice pass.

Each component contributes its EXCESS projected potential over the vitreous-ice
background per z-slab (0 outside itself), so components simply sum. The scene
adds the shared ice/buffer structural noise once. This replaces the separate
per-specimen code paths with one builder:

    scene = Scene(ice_thickness_nm=120, components=[
        Cell("e_coli"),
        Nanoparticles(n=25, diameter_nm_mean=25),
        Contamination(n=3),
    ])
    ideal, info = simulate_scene(cfg, scene, defocus_um=-3)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from .potential import (MIP_VITREOUS_ICE, MIP_PLGA, MIP_CRYSTALLINE_ICE,
                        _fill_polygon, nm_to_px)
from .bacteria import (MIP as CELL_MIP, SPECIES, ENVELOPE, Bacterium,
                       _seg_distance_2d, _point_seg_dist, _disk_add, _add_flagella)
from .multislice import (multislice_exit_wave, coherent_ctf, image_from_wave,
                         apply_inelastic)
from .optics import _tukey2d
from .setup import sigma_rad_per_VA

ICE = MIP_VITREOUS_ICE


# ---------------------------------------------------------------------------
# Components — each: prepare(shape, px, total_a, rng); add_slab(V, shape, px, z, dz)
# ---------------------------------------------------------------------------
@dataclass
class Nanoparticles:
    n: int = 20
    diameter_nm_mean: float = 25.0
    diameter_nm_sd: float = 6.0
    mip_v: float = MIP_PLGA
    z_extent_nm: float = 0.0            # 0 -> fill the ice thickness

    def prepare(self, shape, px, total_a, rng):
        h, w = shape
        self._sph = []
        zext = self.z_extent_nm * 10.0 or total_a
        z_lo = (total_a - zext) / 2
        for _ in range(self.n):
            d = rng.normal(self.diameter_nm_mean, self.diameter_nm_sd)
            if d <= 2:
                continue
            R = min(d * 10.0 / 2.0, 0.45 * zext)     # clamp so it fits the slab thickness
            if zext - 2 * R <= 0:
                continue
            self._sph.append((rng.uniform(0, h), rng.uniform(0, w),
                              rng.uniform(z_lo + R, z_lo + zext - R), R))

    def add_slab(self, V, shape, px, z, dz):
        exc = (self.mip_v - ICE) * dz
        for cy, cx, cz, R in self._sph:
            if abs(z - cz) < R:
                _disk_add(V, cy, cx, np.sqrt(R ** 2 - (z - cz) ** 2) / px, exc)

    def footprints(self, px):
        """Ground-truth polygons (label, [(x,y) px]) — one circle per particle."""
        out = []
        ang = np.linspace(0, 2 * np.pi, 20, endpoint=False)
        for cy, cx, cz, R in self._sph:
            r = R / px
            out.append(("nanoparticle",
                        [(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in ang]))
        return out


@dataclass
class Contamination:
    n: int = 3
    thickness_nm: float = 80.0
    mip_v: float = MIP_CRYSTALLINE_ICE

    def prepare(self, shape, px, total_a, rng):
        h, w = shape
        self._crys = []
        for _ in range(self.n):
            cy, cx = rng.uniform(0, h), rng.uniform(0, w)
            R = rng.uniform(nm_to_px(60, px), nm_to_px(350, px))
            phi = rng.uniform(0, np.pi)
            t_a = max(50.0, rng.lognormal(np.log(self.thickness_nm), 0.4)) * 10.0
            needle = rng.random() < 0.25
            sx = rng.uniform(2.5, 5.0) if needle else 1.0
            sy = rng.uniform(0.4, 0.8) if needle else rng.uniform(0.8, 1.0)
            ang = np.linspace(0, 2 * np.pi, 6, endpoint=False)
            j = 1 + rng.uniform(-0.12, 0.12, 6)
            vx = cx + R * j * (sx * np.cos(ang) * np.cos(phi) - sy * np.sin(ang) * np.sin(phi))
            vy = cy + R * j * (sx * np.cos(ang) * np.sin(phi) + sy * np.sin(ang) * np.cos(phi))
            verts = np.stack([vx, vy], axis=1)
            self._crys.append((verts, 0.0, t_a))          # sits at top surface (z 0..t)

    def add_slab(self, V, shape, px, z, dz):
        exc = (self.mip_v - ICE) * dz
        for verts, z0, t_a in self._crys:
            if z0 <= z <= z0 + t_a:
                _fill_polygon(V, verts, exc)

    def footprints(self, px):
        """Ground-truth polygons — the crystal footprint vertices (already px)."""
        return [("ice_contamination", [(float(x), float(y)) for x, y in verts])
                for verts, _z0, _t in self._crys]


@dataclass
class Cell:
    species: str = "e_coli"
    angle_deg: float = 15.0
    n_ribosomes: int = 300
    ribosome_diam_nm: float = 22.0
    nucleoid_frac: float = 0.45

    def morph(self):
        return SPECIES[self.species]

    def z_extent_nm(self):
        m = self.morph()
        return (m.get("width_nm") or m.get("diameter_nm"))

    def prepare(self, shape, px, total_a, rng):
        h, w = shape
        m = self.morph()
        self._m = m
        width_a = (m.get("width_nm") or m.get("diameter_nm")) * 10.0
        self.Rcell = 0.5 * width_a
        length_a = m.get("length_nm", 0.0) * 10.0
        half = max(0.0, 0.5 * (length_a - width_a))
        th = np.deg2rad(self.angle_deg)
        self.ux, self.uy = np.cos(th), np.sin(th)
        self.A = (w / 2 - half / px * self.ux, h / 2 - half / px * self.uy)
        self.B = (w / 2 + half / px * self.ux, h / 2 + half / px * self.uy)
        self.d2d_a = _seg_distance_2d(shape, self.A, self.B) * px
        self.shells = ENVELOPE[m["gram"]]
        self.env_total_a = sum(t for _, t in self.shells) * 10.0
        self.z0 = total_a / 2.0
        self.length_a, self.width_a = length_a, width_a
        Rribo = self.ribosome_diam_nm * 10.0 / 2.0
        self.inner_a = self.Rcell - self.env_total_a - Rribo
        nf = self.nucleoid_frac
        self.nuc_rz = nf * self.Rcell
        self._ribos = []; tries = 0
        while len(self._ribos) < self.n_ribosomes and tries < self.n_ribosomes * 40:
            tries += 1
            rx, ry, rz = rng.uniform(0, w), rng.uniform(0, h), rng.uniform(0, total_a)
            d2 = _point_seg_dist(rx, ry, self.A, self.B) * px
            d3 = np.sqrt(d2 ** 2 + (rz - self.z0) ** 2) - self.Rcell
            if d3 < -(self.env_total_a + Rribo):
                if nf <= 0 or (d3 / (self.inner_a + 1e-6)) ** 2 + ((rz - self.z0) / self.nuc_rz) ** 2 > 1.0:
                    self._ribos.append((rx, ry, rz, Rribo))
        self._rng = rng

    def add_slab(self, V, shape, px, z, dz):
        d3d = np.sqrt(self.d2d_a ** 2 + (z - self.z0) ** 2) - self.Rcell
        dd = -d3d
        inside = dd >= 0
        if not inside.any():
            return
        mip = np.zeros(shape, dtype=np.float32)            # 0 = background (ice)
        mip[inside] = CELL_MIP["cytoplasm"] - ICE
        acc = 0.0
        for name, t_nm in self.shells:
            t_a = t_nm * 10.0
            shell = inside & (dd >= acc) & (dd < acc + t_a)
            mip[shell] = CELL_MIP[name] - ICE
            acc += t_a
        if self.nucleoid_frac > 0:
            ell = ((d3d / (self.inner_a + 1e-6)) ** 2 + ((z - self.z0) / self.nuc_rz) ** 2) < 1.0
            mip[ell & (dd >= self.env_total_a)] = CELL_MIP["nucleoid"] - ICE
        V += mip * dz
        for rx, ry, rz, R in self._ribos:
            if abs(z - rz) < R:
                _disk_add(V, ry, rx, np.sqrt(R ** 2 - (z - rz) ** 2) / px,
                          (CELL_MIP["ribosome"] - CELL_MIP["cytoplasm"]) * dz)
        if self._m.get("flagella") and abs(z - self.z0) < 0.15 * self.width_a:
            _add_flagella(V, shape, px, self.B, self.ux, self.uy,
                          self._m["flagella"], self.length_a, dz, self._rng)

    def footprints(self, px):
        """Ground-truth polygon — the projected capsule outline of the cell."""
        r = self.Rcell / px
        ux, uy = self.ux, self.uy
        pxperp = (-uy, ux)
        pts = []
        # semicircle cap at B (pole), sweeping through +u
        for a in np.linspace(-np.pi / 2, np.pi / 2, 24):
            d = (ux * np.cos(a) + pxperp[0] * np.sin(a), uy * np.cos(a) + pxperp[1] * np.sin(a))
            pts.append((self.B[0] + r * d[0], self.B[1] + r * d[1]))
        # semicircle cap at A (other pole), sweeping through -u
        for a in np.linspace(np.pi / 2, 3 * np.pi / 2, 24):
            d = (ux * np.cos(a) + pxperp[0] * np.sin(a), uy * np.cos(a) + pxperp[1] * np.sin(a))
            pts.append((self.A[0] + r * d[0], self.A[1] + r * d[1]))
        return [(self.species, [(float(x), float(y)) for x, y in pts])]


# ---------------------------------------------------------------------------
# Scene
# ---------------------------------------------------------------------------
@dataclass
class Scene:
    ice_thickness_nm: float = 80.0
    components: list = field(default_factory=list)
    solvent_noise: float = 5.0
    solvent_corr_a: float = 3.5
    seed: int = 0

    def thickness_nm(self):
        t = self.ice_thickness_nm
        for c in self.components:
            if isinstance(c, Cell):
                t = max(t, c.z_extent_nm() + 30)
        return t


def scene_annotations(shape, px, scene: Scene):
    """Perfect ground-truth footprints for the scene as (label, [(x,y) px]).

    Replays component placement with the same seed/order as scene_slabs, so the
    polygons exactly match the imaged objects — free, flawless training labels."""
    rng = np.random.default_rng(scene.seed)
    total_a = scene.thickness_nm() * 10.0
    out = []
    for c in scene.components:
        c.prepare(shape, px, total_a, rng)
        if hasattr(c, "footprints"):
            out += c.footprints(px)
    return out


def scene_slabs(shape, px, dz_a, scene: Scene):
    rng = np.random.default_rng(scene.seed)
    total_a = scene.thickness_nm() * 10.0
    for c in scene.components:
        c.prepare(shape, px, total_a, rng)
    nz = max(1, int(np.ceil(total_a / dz_a)))
    corr_px = max(0.5, scene.solvent_corr_a / px)
    amp = scene.solvent_noise * np.sqrt(dz_a / 10.0)
    for iz in range(nz):
        z = (iz + 0.5) * dz_a
        V = np.zeros(shape, dtype=np.float32)
        for c in scene.components:
            c.add_slab(V, shape, px, z, dz_a)
        if scene.solvent_noise > 0:
            nf = gaussian_filter(rng.standard_normal(shape).astype(np.float32), corr_px)
            V += amp * (nf / (nf.std() + 1e-8))
        yield V


def simulate_scene(cfg, scene: Scene, defocus_um=-3.0, dz_a=40.0, bfactor=40.0,
                   apodize=0.1, inelastic=True):
    """One multislice pass over a composed scene -> noiseless image + info."""
    n = int(cfg["image_size_px"]); shape = (n, n); px = cfg["pixel_size_a"]
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))
    total_nm = scene.thickness_nm()
    nz = max(1, int(np.ceil(total_nm * 10.0 / dz_a)))

    psi = multislice_exit_wave(shape, px, dz_a, scene_slabs(shape, px, dz_a, scene),
                               lam, sigma, q=cfg["amplitude_contrast"])
    b = int(max(2, apodize * n))
    edge = np.concatenate([psi[:b].ravel(), psi[-b:].ravel(),
                           psi[:, :b].ravel(), psi[:, -b:].ravel()])
    psi = edge.mean() + (psi - edge.mean()) * _tukey2d(shape, apodize)
    ideal = image_from_wave(psi, coherent_ctf(shape, px, cfg, defocus_um, bfactor))
    ideal = ideal / (ideal.mean() + 1e-8)
    dose_scale = 1.0
    if inelastic:
        ideal, dose_scale = apply_inelastic(ideal, total_nm, cfg)
    return ideal, {"thickness_nm": total_nm, "n_slabs": nz,
                   "components": [type(c).__name__ for c in scene.components],
                   "defocus_um": defocus_um, "dose_scale": dose_scale}
