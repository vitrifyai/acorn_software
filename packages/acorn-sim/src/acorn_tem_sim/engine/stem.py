"""4D-STEM forward engine + virtual detectors, with native per-position dose.

A convergent probe is scanned over the specimen; at each scan position we
multislice the probe through the specimen and record the far-field diffraction
pattern (CBED). The result is a 4-D datacube (scan_y, scan_x, k_y, k_x) from
which every STEM modality is derived:

    BF / ADF / HAADF : integrate CBED over disk / annulus / high-angle annulus
    DPC / CoM        : first moment (centre of mass) of each CBED
    iDPC             : integrate the CoM vector field -> phase-like image
    ptychography     : iterative phase retrieval from the whole cube (stem_ptycho)

Dose is applied per probe position (total e/A^2 * scan-step area -> electrons
per probe -> Poisson on the CBED), so the datacube is dose-realistic -- the
control PySlice lacks and the thing that matters for beam-sensitive proteins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .multislice import _fresnel, _antialias_mask
from .setup import sigma_rad_per_VA


def _kgrid(shape, px):
    """kx, ky (2-D, cycles/Å) and radial k."""
    ky = np.fft.fftfreq(shape[0], d=px)
    kx = np.fft.fftfreq(shape[1], d=px)
    kxx, kyy = np.meshgrid(kx, ky)
    return kxx.astype(np.float32), kyy.astype(np.float32), np.hypot(kxx, kyy).astype(np.float32)


def make_probe(shape, px, cfg, conv_mrad, defocus_um=0.0, pos_a=(0.0, 0.0)):
    """Convergent STEM probe at real-space position pos_a (Å), unit-normalised.

    Aperture (convergence semi-angle) + aberrations define A(k)=aper*exp(-iχ);
    a Fourier shift places the probe; the real-space probe is IFFT(A)."""
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2          # Å
    cs_a = cfg["cs_mm"] * 1e7
    df_a = -defocus_um * 1e4
    kx, ky, k = _kgrid(shape, px)
    k2 = k ** 2
    k_ap = (conv_mrad * 1e-3) / lam                        # aperture cutoff (1/Å)
    aperture = (k <= k_ap).astype(np.complex64)
    chi = np.pi * lam * df_a * k2 - 0.5 * np.pi * cs_a * lam ** 3 * k2 ** 2
    x0, y0 = pos_a
    shift = np.exp(-2j * np.pi * (kx * x0 + ky * y0))
    A = aperture * np.exp(-1j * chi) * shift
    probe = np.fft.ifft2(A).astype(np.complex64)
    probe /= np.sqrt((np.abs(probe) ** 2).sum()) + 1e-12   # unit incident intensity
    return probe


@dataclass
class FourDSTEM:
    cube: np.ndarray            # (scan_y, scan_x, k_y, k_x) diffraction intensities
    scan_pos_a: np.ndarray      # (scan_y, scan_x, 2) probe positions in Å
    scan_step_a: float
    conv_mrad: float
    px: float
    cfg: dict
    dosed: bool


def stem_4d(cfg, slabs, conv_mrad=20.0, defocus_um=0.0, dz_a=40.0,
            scan_shape=(40, 40), scan_step_a=None, center_a=None,
            dose_e_per_a2=None, seed=0, use_gpu=None):
    """Scan a probe over the specimen and record CBEDs.

    `slabs` is an iterable of per-slab projected potentials (V*Å) — the same
    thing our multislice specimens yield. Transmission functions are built once
    and reused for every probe position (the specimen doesn't move). The scan +
    per-probe multislice run on the GPU when CuPy is available."""
    from .backend import get_xp, to_cpu
    xp, on_gpu = get_xp(use_gpu)
    n = int(cfg["image_size_px"]); shape = (n, n); px = cfg["pixel_size_a"]
    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    q = cfg["amplitude_contrast"]

    # Transmission functions + propagator, built once and moved to the device.
    trans = [xp.asarray(np.exp(1j * sigma * V - q * sigma * np.abs(V)).astype(np.complex64))
             for V in slabs]
    P = xp.asarray((_fresnel(shape, px, dz_a, lam) * _antialias_mask(shape, px)).astype(np.complex64))

    sy, sx = scan_shape
    if scan_step_a is None:
        scan_step_a = 0.6 * n * px / max(sx, sy)           # cover ~60% of the field
    if center_a is None:
        center_a = (n * px / 2.0, n * px / 2.0)
    xs = center_a[0] + (np.arange(sx) - (sx - 1) / 2) * scan_step_a
    ys = center_a[1] + (np.arange(sy) - (sy - 1) / 2) * scan_step_a
    pos = np.stack(np.meshgrid(xs, ys), axis=-1)           # (sy,sx,2) -> (x,y)

    e_per_probe = dose_e_per_a2 * scan_step_a ** 2 if dose_e_per_a2 is not None else None

    # Un-shifted probe spectrum A0(k) and kx/ky on the device; probes at each
    # scan position are A0 * Fourier-shift. Positions are processed in BATCHES so
    # the GPU runs one big batched FFT instead of thousands of tiny ones.
    A0, kxg, kyg = _probe_spectrum(shape, px, cfg, conv_mrad, defocus_um, xp)
    posx = pos[..., 0].ravel(); posy = pos[..., 1].ravel()
    npos = posx.size
    chunk = max(1, min(npos, 4_000_000 // (n * n)))    # cap device memory
    cube = np.empty((npos, n, n), dtype=np.float32)

    for s in range(0, npos, chunk):
        px0 = xp.asarray(posx[s:s + chunk]); py0 = xp.asarray(posy[s:s + chunk])
        shift = xp.exp(-2j * np.pi * (kxg[None] * px0[:, None, None]
                                      + kyg[None] * py0[:, None, None]))
        psi = xp.fft.ifft2(A0[None] * shift, axes=(-2, -1)).astype(xp.complex64)
        psi /= xp.sqrt((xp.abs(psi) ** 2).sum(axis=(-2, -1), keepdims=True)) + 1e-12
        for t in trans:                                # transmit + propagate (batched)
            psi = xp.fft.ifft2(xp.fft.fft2(psi * t[None], axes=(-2, -1)) * P[None], axes=(-2, -1))
        cbed = xp.abs(xp.fft.fftshift(xp.fft.fft2(psi, axes=(-2, -1)), axes=(-2, -1))) ** 2
        cbed /= cbed.sum(axis=(-2, -1), keepdims=True) + 1e-12
        if e_per_probe is not None:
            cbed = xp.random.poisson(cbed * e_per_probe).astype(xp.float32)
        cube[s:s + chunk] = to_cpu(cbed)

    cube = cube.reshape(sy, sx, n, n)
    return FourDSTEM(cube=cube, scan_pos_a=pos, scan_step_a=scan_step_a,
                     conv_mrad=conv_mrad, px=px, cfg=cfg, dosed=e_per_probe is not None)


def _probe_spectrum(shape, px, cfg, conv_mrad, defocus_um, xp):
    """Un-shifted probe spectrum A0(k)=aperture*exp(-iχ) + kx,ky grids (device)."""
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    cs_a = cfg["cs_mm"] * 1e7
    df_a = -defocus_um * 1e4
    kx, ky, k = _kgrid(shape, px)
    k2 = k ** 2
    k_ap = (conv_mrad * 1e-3) / lam
    chi = np.pi * lam * df_a * k2 - 0.5 * np.pi * cs_a * lam ** 3 * k2 ** 2
    A0 = ((k <= k_ap).astype(np.complex64)) * np.exp(-1j * chi)
    return xp.asarray(A0), xp.asarray(kx), xp.asarray(ky)


# ---------------------------------------------------------------------------
# Virtual detectors
# ---------------------------------------------------------------------------
def _detector_k(fourd):
    n = fourd.cube.shape[-1]
    _, _, k = _kgrid((n, n), fourd.px)
    return np.fft.fftshift(k)                               # match fftshifted CBED


def virtual_detectors(fourd: FourDSTEM, bf_mrad=None,
                      adf_inner_mrad=None, adf_outer_mrad=None,
                      haadf_inner_mrad=None):
    """Return dict of BF/ADF/HAADF images + DPC (CoM) + iDPC."""
    lam = fourd.cfg["_derived"]["wavelength_pm"] * 1e-2
    k = _detector_k(fourd)
    mrad = k * lam * 1e3                                    # scattering angle (mrad)
    cube = fourd.cube
    tot = cube.sum(axis=(-2, -1)) + 1e-12

    conv = fourd.conv_mrad
    bf_mrad = bf_mrad or conv                               # BF ~ inside the disk
    adf_inner_mrad = adf_inner_mrad or conv * 1.2
    adf_outer_mrad = adf_outer_mrad or conv * 4
    haadf_inner_mrad = haadf_inner_mrad or conv * 3

    def integ(mask):
        return (cube * mask).sum(axis=(-2, -1))

    out = {
        "BF": integ(mrad <= bf_mrad),
        "ADF": integ((mrad > adf_inner_mrad) & (mrad <= adf_outer_mrad)),
        "HAADF": integ(mrad > haadf_inner_mrad),
    }
    # DPC / centre-of-mass (first moment of each CBED, in mrad).
    n = cube.shape[-1]
    _, _, _ = None, None, None
    ky = np.fft.fftshift(np.fft.fftfreq(n, d=fourd.px))
    kx = np.fft.fftshift(np.fft.fftfreq(n, d=fourd.px))
    KX, KY = np.meshgrid(kx, ky)
    comx = (cube * KX).sum(axis=(-2, -1)) / tot * lam * 1e3
    comy = (cube * KY).sum(axis=(-2, -1)) / tot * lam * 1e3
    out["DPCx"], out["DPCy"] = comx, comy
    out["CoM_mag"] = np.hypot(comx, comy)
    out["iDPC"] = _integrate_com(comx, comy)
    return out


def _integrate_com(comx, comy):
    """iDPC: integrate the CoM vector field via Fourier inversion of divergence."""
    sy, sx = comx.shape
    ky = np.fft.fftfreq(sy); kx = np.fft.fftfreq(sx)
    KX, KY = np.meshgrid(kx, ky)
    denom = (KX ** 2 + KY ** 2)
    denom[0, 0] = 1.0
    div = np.fft.fft2(comx) * (1j * KX) + np.fft.fft2(comy) * (1j * KY)
    phase = np.real(np.fft.ifft2(div / (-(2 * np.pi) ** 2 * denom + 0j)))
    phase[0, 0] = 0
    return phase - phase.mean()
