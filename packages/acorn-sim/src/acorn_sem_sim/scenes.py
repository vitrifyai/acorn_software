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
    """A specimen and the truth about it.

    `material_names` is physics -- what each label is made of, and what the
    transport model needs. `label_names` is what those regions should be CALLED
    in an annotation, which is not always the same thing: a pore is made of
    vacuum, but nobody annotates "vacuum", they annotate "pore".

    `background_labels` names the labels that are context rather than objects.
    A support or embedding matrix filling the frame is not a useful annotation
    and would swamp any real object count. It is a tuple, not a single label,
    and it can be empty -- in a two-phase alloy or a layer stack every region is
    an object, and excluding one of them would silently halve the ground truth.
    """

    material_index: np.ndarray
    material_names: list[str]
    height_nm: np.ndarray
    pixel_size_nm: float
    description: str = ""
    label_names: list[str] | None = None
    background_labels: tuple[int, ...] = (0,)
    meta: dict = field(default_factory=dict)

    def annotation_name(self, label: int) -> str:
        """What to call `label` in an annotation."""
        names = self.label_names or self.material_names
        return names[int(label)]

    def is_background(self, label: int) -> bool:
        return int(label) in self.background_labels


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
                 meta={"n_particles_placed": len(placed), "relief": bool(relief),
                       "diameter_nm_mean": diameter_nm_mean,
                       "n_particles_requested": int(n_particles)})


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

    # No background: in a two-phase alloy both phases are the objects of
    # interest, and treating either as context would export half the truth.
    return Scene(idx, [phase_a, phase_b], np.zeros(shape, np.float32), pixel_size_nm,
                 f"{n_grains}-grain {phase_a}/{phase_b} alloy, polished flat",
                 background_labels=(),
                 meta={"n_grains": int(n_grains), "flat": True})


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

    # The pore is what gets segmented in a porous sample. Its material is
    # vacuum, but its annotation label is "pore" -- naming it after the material
    # would have it filtered out as empty space and the scene would export no
    # ground truth at all.
    return Scene(idx, [matrix, "vacuum"], h, pixel_size_nm,
                 f"porous {matrix}, {porosity:.0%} porosity",
                 label_names=[matrix, "pore"],
                 meta={"porosity_achieved": float(pores.mean()),
                       "porosity_requested": float(porosity)})


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
                 meta={"n_cells": int(n_cells), "z_contrast": "negligible"})


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

    # Every layer is a region someone would segment, including the substrate.
    return Scene(idx.astype(np.int32), names, h.astype(np.float32), pixel_size_nm,
                 f"FIB cross-section, Pt cap over {len(names) - 1} layers",
                 background_labels=(),
                 meta={"layers": names})



def _ellipse_mask(shape, cy, cx, a_px, b_px, angle_rad):
    """Rotated-ellipse footprint. Spores are ovoid, not round, and orientation
    on the substrate is arbitrary -- a circular stand-in would make every
    detection task easier than it is."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    dy, dx = yy - cy, xx - cx
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    u = (dx * ca + dy * sa) / max(a_px, 1e-6)
    v = (-dx * sa + dy * ca) / max(b_px, 1e-6)
    return u ** 2 + v ** 2, (u ** 2 + v ** 2) <= 1.0


def bacterial_spores(shape=(512, 512), pixel_size_nm=8.0, n_spores=25,
                     length_nm=1200.0, width_nm=800.0, size_spread=0.15,
                     coating_nm=10.0, coating="gold", substrate="silicon",
                     clustering=0.35, seed=0) -> Scene:
    """Bacterial spores on a substrate, as surface SEM actually sees them.

    Spores are ovoid -- Bacillus subtilis is roughly 1.2 x 0.8 um -- they lie at
    arbitrary orientations, they aggregate, and they sit ON the substrate rather
    than in it, so they carry strong topographic relief. All four matter to a
    detection task and none of them is captured by discs on a flat field.

    The sputter coating is the physically decisive detail. Biological specimens
    are routinely coated with 5-20 nm of gold or platinum for conductivity, and
    at ordinary beam energies the secondary-electron escape depth in gold is
    around a nanometre. Essentially every secondary therefore comes from the
    coating, not from the spore: a coated spore images as gold-shaped-like-a-
    spore, with gold's compact interaction volume rather than biology's diffuse
    one. Uncoated, the same object is low-Z against a low-Z background and far
    harder to find.

    That is modelled by giving a coated spore the coating's material outright.
    Exact for secondaries; approximate for backscatters, which sample deeper and
    would partly see the biology beneath. `coating_nm = 0` leaves the spore
    uncoated.
    """
    rng = np.random.default_rng(seed)
    idx = np.zeros(shape, dtype=np.int32)
    h = np.zeros(shape, dtype=np.float32)

    coated = float(coating_nm) > 0
    spore_material = coating if coated else "biology"
    names = [substrate, spore_material]

    a_px = 0.5 * float(length_nm) / pixel_size_nm      # semi-major, pixels
    b_px = 0.5 * float(width_nm) / pixel_size_nm       # semi-minor

    # A spore larger than the field cannot be placed at all, and asking for one
    # is usually a pixel-size mistake rather than an intention. Scale to fit and
    # record that it happened, rather than raising or returning an empty frame
    # that looks like a specimen with no spores on it.
    fit = min(shape) / 2.4
    clamped = a_px > fit
    if clamped:
        shrink = fit / a_px
        a_px, b_px = a_px * shrink, b_px * shrink
        length_nm, width_nm = length_nm * shrink, width_nm * shrink

    placed = []

    # Aggregation: spores are dispensed as suspensions and dry into clumps, so
    # positions are drawn near existing ones rather than uniformly.
    for _ in range(int(n_spores) * 12):
        if len(placed) >= n_spores:
            break
        scale = 1.0 + rng.normal(0.0, float(size_spread))
        if scale < 0.4:
            continue
        a, b = a_px * scale, b_px * scale
        if a < 2 or b < 1:
            continue
        reach = a + 2
        # Guard the draw itself. Clamping the nominal size is not enough because
        # the spread can push an individual spore back over the limit, and
        # uniform() with high < low raises rather than returning nothing.
        if 2 * reach >= min(shape):
            continue

        if placed and rng.random() < float(clustering):
            py, px_, pa, _, _ = placed[rng.integers(len(placed))]
            gap = (pa + a) * rng.uniform(0.9, 1.6)
            phi = rng.uniform(0, 2 * np.pi)
            cy = py + gap * np.sin(phi)
            cx = px_ + gap * np.cos(phi)
            if not (reach <= cy < shape[0] - reach and reach <= cx < shape[1] - reach):
                continue
        else:
            cy = rng.uniform(reach, shape[0] - reach)
            cx = rng.uniform(reach, shape[1] - reach)

        angle = rng.uniform(0, np.pi)
        # Touching is realistic; heavy overlap is not, so reject only deep ones.
        if any((cy - py) ** 2 + (cx - px_) ** 2 < (0.75 * (a + pa)) ** 2
               for py, px_, pa, _, _ in placed):
            continue

        r2, mask = _ellipse_mask(shape, cy, cx, a, b, angle)
        idx[mask] = 1
        # Ellipsoid cap: height falls to zero at the rim, so the relief is the
        # shape of the object rather than a plateau with a cliff edge.
        cap = np.zeros(shape, np.float32)
        cap[mask] = (np.sqrt(np.clip(1.0 - r2[mask], 0.0, None))
                     * (0.5 * float(width_nm) * scale)).astype(np.float32)
        h = np.maximum(h, cap)
        placed.append((cy, cx, a, b, angle))

    # Substrate roughness, well below spore height so it does not compete.
    h = h + _smooth_noise(shape, rng, 6, float(width_nm) * 0.01)

    return Scene(idx, names, h.astype(np.float32), pixel_size_nm,
                 f"{len(placed)} spores on {substrate}"
                 + (f", {coating_nm:g} nm {coating} coated" if coated else ", uncoated"),
                 meta={"n_spores_placed": len(placed),
                       "n_spores_requested": int(n_spores),
                       "size_clamped_to_field": bool(clamped),
                       "length_nm": float(length_nm), "width_nm": float(width_nm),
                       "coating_nm": float(coating_nm) if coated else 0.0,
                       "coating": coating if coated else None,
                       "clustering": float(clustering)})


BUILDERS = {
    "nanoparticles": nanoparticles_on_substrate,
    "spores":        bacterial_spores,
    "grains":        two_phase_grains,
    "porous":        porous_surface,
    "biological":    biological_surface,
    "cross_section": fib_cross_section,
}

# Plain-language labels for the UI and for CLU to match against.
SCENE_LABELS = {
    "nanoparticles": "Nanoparticles on a substrate",
    "spores":        "Bacterial spores on a substrate",
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
