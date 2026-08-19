"""Synthetic FIB-SEM surface-image simulator.

The acquisition order is:

    phantom truth -> FIB milling artifacts -> detector noise

All spatial extents are in nm; pixel_size_nm is the single source of scale.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter, gaussian_filter1d


@dataclass(frozen=True)
class ImagingConfig:
    shape: tuple[int, int] = (512, 640)
    pixel_size_nm: float = 5.0
    electrons_per_pixel: float = 400.0
    mtf_sigma_px: float = 0.7
    read_noise_e: float = 3.0
    scan_line_jitter: float = 0.4
    seed: int = 0


@dataclass(frozen=True)
class MillingConfig:
    mill_axis: str = "y"
    curtain_strength: float = 0.25
    curtain_edge_gain: float = 2.0
    mill_gradient: float = 0.15
    redeposition: float = 0.06
    charging: float = 0.0
    lamella_thickness_nm: float = 200.0


@dataclass(frozen=True)
class SceneConfig:
    liftout: bool = False
    trench: bool = True
    pt_cap: bool = True
    needle: bool = False


@dataclass(frozen=True)
class SimConfig:
    sample: str = "bio"
    imaging: ImagingConfig = field(default_factory=ImagingConfig)
    milling: MillingConfig = field(default_factory=MillingConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)


@dataclass(frozen=True)
class SimResult:
    truth: np.ndarray
    edges: np.ndarray
    milled: np.ndarray
    image: np.ndarray
    config: SimConfig


def _lowfreq(shape: tuple[int, int], rng: np.random.Generator, scale_px: float, amp: float = 1.0) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    f = gaussian_filter(n, sigma=max(0.5, float(scale_px)))
    f -= f.min()
    f /= f.max() + 1e-8
    return amp * f


def _edges(structure: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(structure.astype(np.float32))
    e = np.hypot(gy, gx)
    return e / (e.max() + 1e-8)


def _draw_membrane(canvas: np.ndarray, pts, thickness_px: float, contrast: float) -> np.ndarray:
    h, w = canvas.shape
    pts = np.asarray(pts, dtype=np.float32)
    t = np.linspace(0, 1, len(pts))
    tt = np.linspace(0, 1, 2000)
    ys = np.interp(tt, t, pts[:, 0])
    xs = np.interp(tt, t, pts[:, 1])
    line = np.zeros_like(canvas)
    line[np.clip(ys.astype(int), 0, h - 1), np.clip(xs.astype(int), 0, w - 1)] = 1.0
    dist = distance_transform_edt(1 - line)
    leaflet = np.exp(-((dist - thickness_px) ** 2) / (2 * (thickness_px * 0.5) ** 2))
    core = np.exp(-(dist**2) / (2 * (thickness_px * 0.6) ** 2))
    canvas += contrast * (0.6 * core - leaflet)
    return canvas


def bio_phantom(shape: tuple[int, int], pixel_size_nm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape

    def nm(value: float) -> float:
        return value / pixel_size_nm

    ys = 0.42 + 0.06 * _lowfreq(shape, rng, nm(300)) + 0.03 * _lowfreq(shape, rng, nm(40))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx, ry, rx = h * 0.35, w * 0.68, h * 0.42, w * 0.34
    nuc = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) < 1.0
    ys += 0.10 * gaussian_filter(nuc.astype(np.float32), sigma=nm(60))

    for _ in range(rng.integers(12, 22)):
        r = nm(rng.uniform(30, 120))
        oy, ox = rng.uniform(0, h), rng.uniform(0, w)
        blob = (((yy - oy) / r) ** 2 + ((xx - ox) / r) ** 2) < 1.0
        ys += rng.uniform(-0.12, 0.14) * gaussian_filter(blob.astype(np.float32), sigma=nm(10))

    ys = np.clip(ys, 0, 1).astype(np.float32)
    return ys, _edges(ys)


def material_phantom(shape: tuple[int, int], pixel_size_nm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape

    def nm(value: float) -> float:
        return value / pixel_size_nm

    n_grains = 60
    seeds = np.stack([rng.uniform(0, h, n_grains), rng.uniform(0, w, n_grains)], axis=1)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = (yy[..., None] - seeds[:, 0]) ** 2 + (xx[..., None] - seeds[:, 1]) ** 2
    labels = np.argmin(d, axis=-1)
    ys = rng.uniform(0.35, 0.75, n_grains).astype(np.float32)[labels]
    ys += 0.12 * (_lowfreq(shape, rng, nm(400)) > 0.6)

    for _ in range(rng.integers(8, 16)):
        r = nm(rng.uniform(20, 80))
        oy, ox = rng.uniform(0, h), rng.uniform(0, w)
        pore = (((yy - oy) / r) ** 2 + ((xx - ox) / r) ** 2) < 1.0
        ys -= 0.35 * gaussian_filter(pore.astype(np.float32), sigma=nm(6))

    for _ in range(rng.integers(20, 40)):
        r = nm(rng.uniform(6, 20))
        oy, ox = rng.uniform(0, h), rng.uniform(0, w)
        ys += 0.25 * np.exp(-(((yy - oy) ** 2 + (xx - ox) ** 2) / (2 * r**2)))

    ys = np.clip(ys, 0, 1).astype(np.float32)
    return ys, _edges(ys)


def make_phantom(sample: str, shape: tuple[int, int], pixel_size_nm: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if sample == "bio":
        return bio_phantom(shape, pixel_size_nm, rng)
    if sample == "material":
        return material_phantom(shape, pixel_size_nm, rng)
    raise ValueError(f"unknown sample type: {sample!r}")


def _axis(mill_axis: str) -> tuple[int, int]:
    return (0, 1) if mill_axis == "y" else (1, 0)


def curtaining(signal: np.ndarray, edge_map: np.ndarray, cfg: MillingConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.curtain_strength <= 0:
        return signal
    along, across = _axis(cfg.mill_axis)
    n = signal.shape[across]
    white = rng.standard_normal(n)
    freq = np.fft.rfftfreq(n, d=1.0)
    freq[0] = freq[1]
    prof = np.fft.irfft(np.fft.rfft(white) / freq, n=n)
    prof = (prof - prof.mean()) / (prof.std() + 1e-8)
    prof = gaussian_filter1d(prof, sigma=1.0)
    stripes = np.broadcast_to(prof.reshape((-1, 1) if across == 0 else (1, -1)), signal.shape).copy()
    upstream = np.cumsum(edge_map, axis=along)
    upstream /= upstream.max() + 1e-8
    gate = 1.0 + cfg.curtain_edge_gain * upstream
    return signal * (1.0 + cfg.curtain_strength * stripes * gate)


def mill_gradient(signal: np.ndarray, cfg: MillingConfig) -> np.ndarray:
    if cfg.mill_gradient == 0:
        return signal
    along, _ = _axis(cfg.mill_axis)
    ramp = np.linspace(-0.5, 0.5, signal.shape[along]).astype(np.float32)
    ramp = ramp.reshape((-1, 1) if along == 0 else (1, -1))
    return signal * (1.0 + cfg.mill_gradient * ramp)


def redeposition(signal: np.ndarray, cfg: MillingConfig) -> np.ndarray:
    if cfg.redeposition <= 0:
        return signal
    along, _ = _axis(cfg.mill_axis)
    return signal + cfg.redeposition * gaussian_filter1d(signal, 8.0, axis=along, mode="nearest")


def charging(signal: np.ndarray, cfg: MillingConfig, rng: np.random.Generator) -> np.ndarray:
    if cfg.charging <= 0:
        return signal
    bias = gaussian_filter(rng.standard_normal(signal.shape).astype(np.float32), 25.0)
    bias -= bias.min()
    bias /= bias.max() + 1e-8
    return signal + cfg.charging * bias


def apply_milling(signal: np.ndarray, edge_map: np.ndarray, cfg: MillingConfig, rng: np.random.Generator) -> np.ndarray:
    s = mill_gradient(signal, cfg)
    s = redeposition(s, cfg)
    s = curtaining(s, edge_map, cfg, rng)
    s = charging(s, cfg, rng)
    return np.clip(s, 0, None)


def liftout_scene(signal: np.ndarray, scene: SceneConfig, rng: np.random.Generator) -> np.ndarray:
    h, w = signal.shape
    out = signal.copy()
    yy, xx = np.mgrid[0:h, 0:w]
    band = (w * 0.32, w * 0.68)
    if scene.trench:
        out[xx < band[0]] *= 0.15
        out[xx > band[1]] *= 0.15
        for edge in band:
            mask = np.abs(xx - edge) < 3
            out[mask] = np.maximum(out[mask], 0.8)
    if scene.pt_cap:
        cap = (yy < h * 0.12) & (xx > band[0] - 8) & (xx < band[1] + 8)
        out[cap] = 0.9 + 0.05 * rng.standard_normal(out[cap].shape)
    if scene.needle:
        line = (xx - w * 0.95) * 0.4 + yy
        probe = (line > 0) & (line < 30) & (yy < h * 0.5)
        out[probe] = 0.75
    return np.clip(out, 0, None)


def apply_detector(signal: np.ndarray, imaging: ImagingConfig, rng: np.random.Generator) -> np.ndarray:
    s = signal / (signal.max() + 1e-8)
    counts = rng.poisson(np.clip(s * imaging.electrons_per_pixel, 0, None)).astype(np.float32)
    if imaging.mtf_sigma_px > 0:
        counts = gaussian_filter(counts, sigma=imaging.mtf_sigma_px)
    if imaging.read_noise_e > 0:
        counts += rng.normal(0, imaging.read_noise_e, counts.shape).astype(np.float32)
    if imaging.scan_line_jitter > 0:
        jit = 1.0 + imaging.scan_line_jitter / 100.0 * rng.standard_normal(counts.shape[0])
        counts *= jit[:, None]
    counts = np.clip(counts, 0, None)
    return (counts / (counts.max() + 1e-8)).astype(np.float32)


def simulate(cfg: SimConfig | None = None) -> SimResult:
    cfg = cfg or SimConfig()
    rng = np.random.default_rng(cfg.imaging.seed)
    truth, edges = make_phantom(cfg.sample, cfg.imaging.shape, cfg.imaging.pixel_size_nm, rng)
    milled = apply_milling(truth, edges, cfg.milling, rng)
    if cfg.scene.liftout:
        milled = liftout_scene(milled, cfg.scene, rng)
    image = apply_detector(milled, cfg.imaging, rng)
    return SimResult(truth=truth, edges=edges, milled=milled, image=image, config=cfg)
