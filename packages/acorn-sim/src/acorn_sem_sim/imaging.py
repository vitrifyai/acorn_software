"""SEM image formation from interaction-volume kernels.

The forward model, in one line per stage:

    per-pixel yields   delta(material, local tilt), eta(material, local tilt)
    spatial spreading  convolve each material's yield map with ITS OWN kernel
    detector           mix SE and BSE, apply directional shading
    acquisition        Poisson counting, read noise, scan-line jitter

The per-material convolution is the part worth defending. A single kernel for
the whole field would be wrong wherever materials differ in atomic number,
because the interaction volume is a property of the material, not of the
instrument: at 20 keV a gold inclusion emits its backscatters within ~250 nm of
the beam while the carbon around it spreads them over ~2800 nm. Convolving both
with one kernel erases the effect that makes a heavy phase look sharp and a
light matrix look soft. Cost is one FFT pair per material, which is nothing.

The residual approximation is that each material's kernel is computed for a
laterally homogeneous half-space of that material, so a beam landing within one
interaction volume of a phase boundary is treated as though it were deep inside
one phase. That breaks down for inclusions smaller than their own interaction
volume -- and it is why `simulate` reports `boundary_fraction`, so you can tell
when you are in that regime instead of guessing.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kernels as _kernels
from .materials import Material, get


@dataclass
class Beam:
    """Column settings."""

    E0_kev:         float = 5.0
    pixel_size_nm:  float = 5.0
    electrons_per_px: float = 300.0     # dwell current x time, in electrons
    probe_nm:       float = 1.0         # finite probe size, added in quadrature


@dataclass
class Detector:
    """Which escaping electrons are collected, and from what direction.

    kind
        "ETD"    Everhart-Thornley: SE-dominated, but a biased cage also pulls
                 in a real BSE fraction. The workhorse topographic detector.
        "TLD"    Through-lens / in-lens: the immersion field preferentially
                 collects SE1 from near the beam, so it is sharper and sees
                 less of the SE2 pedestal.
        "BSE"    Annular solid-state: backscattered electrons only, giving
                 compositional (Z) contrast with little topographic shading.
    """

    kind:          str = "ETD"
    bse_mix:       float = 0.15    # BSE fraction reaching an SE detector
    elevation_deg: float = 25.0
    azimuth_deg:   float = 0.0
    asymmetry:     float = 0.30    # directional shading strength, 0..1
    read_noise_e:  float = 3.0
    scan_jitter_px: float = 0.0


@dataclass
class SEMImage:
    """A simulated micrograph and the truth that produced it."""

    image:          np.ndarray        # detector signal, arbitrary units
    signal:         np.ndarray        # noise-free signal, same units
    material_index: np.ndarray        # ground-truth label map
    material_names: list[str]
    se:             np.ndarray        # SE component before detector mixing
    bse:            np.ndarray        # BSE component
    tilt_deg:       np.ndarray
    meta:           dict = field(default_factory=dict)


def _surface_tilt_deg(height_nm: np.ndarray, pixel_size_nm: float) -> np.ndarray:
    """Local tilt of the surface away from the beam axis, in degrees."""
    gy, gx = np.gradient(height_nm.astype(np.float64), pixel_size_nm)
    return np.degrees(np.arctan(np.sqrt(gx ** 2 + gy ** 2)))


def _shading(height_nm: np.ndarray, pixel_size_nm: float, det: Detector) -> np.ndarray:
    """Directional term: surfaces facing the detector look brighter.

    This is geometry, not transport -- the Monte Carlo gives how many electrons
    escape, not which ones reach a detector sitting off to one side. Kept
    explicitly separate from the yields for that reason.
    """
    if det.asymmetry <= 0:
        return np.ones_like(height_nm, dtype=np.float64)
    gy, gx = np.gradient(height_nm.astype(np.float64), pixel_size_nm)
    nz = 1.0 / np.sqrt(1.0 + gx ** 2 + gy ** 2)
    nx, ny = -gx * nz, -gy * nz

    el = np.radians(det.elevation_deg)
    az = np.radians(det.azimuth_deg)
    dx, dy, dz = np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)

    cos_nd = np.clip(nx * dx + ny * dy + nz * dz, 0.0, None)
    return 1.0 + det.asymmetry * cos_nd


def _fftconvolve_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Convolve with edge-extended padding, not zero padding.

    Zero padding asserts that the specimen stops at the frame edge and vacuum
    begins -- which is almost never what is meant, and it darkens a band as wide
    as the interaction volume all the way around the image. On a genuinely flat,
    single-material field that artefact was still 0.8% of the signal well inside
    the frame, because at 5 keV the silicon interaction volume is a sizeable
    fraction of a 512 nm field. Extending the edge says instead that the
    specimen continues beyond the frame, which is the usual truth.
    """
    from scipy.signal import fftconvolve
    ph, pw = kernel.shape[0] // 2, kernel.shape[1] // 2
    ph = min(ph, max(image.shape[0] - 1, 0))
    pw = min(pw, max(image.shape[1] - 1, 0))
    if ph == 0 and pw == 0:
        return fftconvolve(image, kernel, mode="same")
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode="edge")
    out = fftconvolve(padded, kernel, mode="same")
    return out[ph:ph + image.shape[0], pw:pw + image.shape[1]]


def _fit_kernel(k2d: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Crop a kernel that is larger than the image, reporting what was lost.

    A kernel wider than the field cannot be applied honestly -- the missing
    pedestal is signal that in reality would have come from outside the frame.
    Cropping and renormalising is the pragmatic choice; the returned fraction
    says how much was discarded so the caller can report it rather than have it
    vanish silently.
    """
    kh, kw = k2d.shape
    mh, mw = shape
    max_h = 2 * (mh // 2) + 1
    max_w = 2 * (mw // 2) + 1
    if kh <= max_h and kw <= max_w:
        return k2d, 0.0
    ch, cw = kh // 2, kw // 2
    hh, hw = min(kh, max_h) // 2, min(kw, max_w) // 2
    crop = k2d[ch - hh:ch + hh + 1, cw - hw:cw + hw + 1]
    kept = float(crop.sum())
    if kept <= 0:
        return k2d, 0.0
    return (crop / kept).astype(np.float32), 1.0 - kept


def simulate(material_index: np.ndarray,
             material_names: list[str] | list[Material],
             height_nm: np.ndarray | None = None,
             beam: Beam | None = None,
             detector: Detector | None = None,
             seed: int = 0,
             n_electrons: int = 60_000,
             use_cache: bool = True) -> SEMImage:
    """Simulate an SEM micrograph of a labelled specimen.

    material_index
        Integer map indexing into `material_names`. This doubles as the
        segmentation ground truth -- the point of the whole exercise.
    height_nm
        Surface topography. None means a flat surface, which isolates
        compositional contrast from topographic contrast.
    """
    beam = beam or Beam()
    detector = detector or Detector()
    material_index = np.asarray(material_index)
    if material_index.ndim != 2:
        raise ValueError("material_index must be a 2-D label map")

    mats: list[Material] = [get(m) if isinstance(m, str) else m for m in material_names]
    names = [m.name for m in mats]
    present = np.unique(material_index)
    bad = [int(i) for i in present if i < 0 or i >= len(mats)]
    if bad:
        raise ValueError(f"material_index contains labels with no material: {bad}")

    h = (np.zeros(material_index.shape, dtype=np.float64) if height_nm is None
         else np.asarray(height_nm, dtype=np.float64))
    if h.shape != material_index.shape:
        raise ValueError("height_nm must match material_index in shape")

    tilt = _surface_tilt_deg(h, beam.pixel_size_nm)

    se_total = np.zeros(material_index.shape, dtype=np.float64)
    bse_total = np.zeros(material_index.shape, dtype=np.float64)
    lost = 0.0

    for idx in present:
        mat = mats[int(idx)]
        mask = (material_index == idx)
        if mat.rho <= 0:                       # vacuum emits nothing
            continue

        K = _kernels.compute(mat, beam.E0_kev, n_electrons=n_electrons,
                             seed=seed, use_cache=use_cache)

        # Yields respond to the LOCAL tilt, so a rough surface of one material
        # still shows topography. Interpolated on the transport-derived grid,
        # not from an imposed secant law.
        d_map = np.where(mask, K.delta_at(tilt), 0.0)
        e_map = np.where(mask, K.eta_at(tilt), 0.0)

        k_se, l1 = _fit_kernel(K.se.to_pixels(beam.pixel_size_nm), material_index.shape)
        k_bse, l2 = _fit_kernel(K.bse.to_pixels(beam.pixel_size_nm), material_index.shape)
        lost = max(lost, l1, l2)

        se_total += _fftconvolve_same(d_map, k_se)
        bse_total += _fftconvolve_same(e_map, k_bse)

    # Finite probe size, on top of the interaction volume.
    if beam.probe_nm > 0:
        from scipy.ndimage import gaussian_filter
        sigma = (beam.probe_nm / 2.355) / beam.pixel_size_nm
        if sigma > 0.05:
            se_total = gaussian_filter(se_total, sigma)
            bse_total = gaussian_filter(bse_total, sigma)

    kind = detector.kind.upper()
    if kind == "BSE":
        signal = bse_total.copy()
        shade = np.ones_like(signal)
    else:
        # A TLD's immersion field favours SE1 over the SE2 pedestal and admits
        # far less BSE than a chamber-mounted ETD.
        mix = 0.02 if kind == "TLD" else detector.bse_mix
        signal = se_total + mix * bse_total
        shade = _shading(h, beam.pixel_size_nm, detector)
    signal = signal * shade

    rng = np.random.default_rng(seed)
    image = _acquire(signal, beam, detector, rng)

    return SEMImage(
        image=image.astype(np.float32),
        signal=signal.astype(np.float32),
        material_index=material_index,
        material_names=names,
        se=se_total.astype(np.float32),
        bse=bse_total.astype(np.float32),
        tilt_deg=tilt.astype(np.float32),
        meta={
            "E0_kev": beam.E0_kev,
            "pixel_size_nm": beam.pixel_size_nm,
            "electrons_per_px": beam.electrons_per_px,
            "detector": detector.kind,
            "materials": [names[int(i)] for i in present],
            "kernel_weight_lost": lost,
            "boundary_fraction": _boundary_fraction(material_index),
        },
    )


def _boundary_fraction(material_index: np.ndarray) -> float:
    """Fraction of pixels adjacent to a different material.

    Reported because the per-material kernel approximation is at its worst
    exactly there. A high value means the interaction volumes of neighbouring
    phases overlap substantially and the image should be read as indicative
    rather than quantitative.
    """
    if material_index.size == 0:
        return 0.0
    diff = np.zeros(material_index.shape, dtype=bool)
    diff[:-1, :] |= material_index[:-1, :] != material_index[1:, :]
    diff[1:, :] |= material_index[:-1, :] != material_index[1:, :]
    diff[:, :-1] |= material_index[:, :-1] != material_index[:, 1:]
    diff[:, 1:] |= material_index[:, :-1] != material_index[:, 1:]
    return float(diff.mean())


def _acquire(signal: np.ndarray, beam: Beam, det: Detector,
             rng: np.random.Generator) -> np.ndarray:
    """Poisson counting, read noise and scan-line jitter.

    The signal is a yield -- electrons out per electron in -- so the count at a
    pixel is Poisson with mean yield x electrons_per_px. Dose therefore enters
    only through the noise, never through the contrast, which is the same
    separation the cryo-TEM detector model makes.
    """
    counts = rng.poisson(np.clip(signal * beam.electrons_per_px, 0.0, None))
    out = counts.astype(np.float64)
    if det.read_noise_e > 0:
        out += rng.normal(0.0, det.read_noise_e, size=out.shape)
    if det.scan_jitter_px > 0:
        shifts = rng.normal(0.0, det.scan_jitter_px, size=out.shape[0])
        for i, s in enumerate(shifts):
            out[i] = np.roll(out[i], round(s))
    return out
