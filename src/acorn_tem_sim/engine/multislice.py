"""Multislice exit-wave propagation for THICK specimens.

The single-slice model (phase-object: psi = exp(i*sigma*V_proj)) assumes the
whole specimen sits in one plane. For thick specimens (thick ice, whole cells)
that is wrong: the electron wave keeps diffracting as it traverses depth, and
different depths focus differently. Multislice handles this by cutting the
volume into thin slabs and alternating, slab by slab:

    transmit :  psi <- psi * exp(i*sigma*V_slab)        (thin phase grating)
    propagate:  psi <- IFFT( FFT(psi) * exp(-i*pi*lambda*dz*k^2) )   (Fresnel)

After the last slab, the objective lens transfer H(k) = Env(k)*exp(-i*chi(k))
is applied to the complex exit wave and the image is |wave|^2. Amplitude
contrast enters as a small absorptive (imaginary) potential.

This is the standard forward model (Cowley-Moodie); it reduces to the
single-slice result when the specimen is thin, which we verify.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .optics import _freq_grid, _tukey2d
from .potential import Specimen


def inelastic_mfp_a(cfg, override_nm=None):
    """Inelastic mean free path (Å) for amorphous ice/protein, voltage-scaled.
    ~320 nm at 300 kV; shorter at lower voltage."""
    if override_nm:
        return override_nm * 10.0
    kv = float(cfg["voltage_kv"])
    return 3200.0 * (kv / 300.0) ** 0.8


def apply_inelastic(ideal, thickness_nm, cfg, mfp_nm=None):
    """Thickness-dependent contrast/SNR loss from inelastic scattering.

    Elastic fraction f = exp(-T/λ_in). With an energy filter (zero-loss) we image
    only the elastic electrons -> same contrast but the effective dose drops by f
    (more shot noise). Without a filter, inelastic electrons form a ~featureless
    background that dilutes contrast to f·I + (1-f)·<I>. This is the missing
    thick-specimen physics: it makes a thin lamella genuinely beat a thick cell.

    Returns (ideal_adjusted, dose_scale)."""
    lam_in = inelastic_mfp_a(cfg, mfp_nm)
    T = thickness_nm * 10.0
    f_el = float(np.exp(-T / lam_in))
    if cfg.get("energy_filter_ev", 0):                 # zero-loss filtered
        return ideal, f_el                             # fewer electrons -> dose*f
    bg = float(ideal.mean())                           # inelastic background ~ flat
    return (f_el * ideal + (1.0 - f_el) * bg).astype(np.float32), 1.0


def _fresnel(shape, px, dz_a, lam_a):
    k = _freq_grid(shape, px)                       # cycles/Å
    return np.exp(-1j * np.pi * lam_a * dz_a * k ** 2).astype(np.complex64)


def _disk_add(out, cy, cx, r_px, val):
    h, w = out.shape
    y0, y1 = max(0, int(cy - r_px - 1)), min(h, int(cy + r_px + 2))
    x0, x1 = max(0, int(cx - r_px - 1)), min(w, int(cx + r_px + 2))
    if y0 >= y1 or x0 >= x1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    out[y0:y1, x0:x1] += val * ((yy - cy) ** 2 + (xx - cx) ** 2 < r_px ** 2)


def plga_slabs(shape, px, dz_a, specimen: Specimen):
    """Yield each slab's projected potential (V*Å) top-to-bottom through the ice.

    Spheres are placed in 3-D within the ice slab; each slab gets the cross-
    sectional disks of the spheres that intersect it, plus per-slab solvent
    noise (so ice noise is genuinely 3-D, not a single projected sheet)."""
    rng = np.random.default_rng(specimen.seed)
    h, w = shape
    T_a = specimen.ice_thickness_nm * 10.0
    d_excess = specimen.plga_mip_v - specimen.ice_mip_v

    spheres, placed = [], 0
    while placed < specimen.n_particles and len(spheres) < specimen.n_particles * 50:
        d_nm = rng.normal(specimen.diameter_nm_mean, specimen.diameter_nm_sd)
        if d_nm <= 2:
            continue
        R_a = d_nm * 10.0 / 2.0
        if 2 * R_a > T_a:                           # too big for this slab
            R_a = 0.45 * T_a
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        cz = rng.uniform(R_a, max(R_a + 1, T_a - R_a))
        spheres.append((cy, cx, R_a, cz)); placed += 1

    nz = max(1, int(np.ceil(T_a / dz_a)))
    corr_px = max(0.5, specimen.solvent_corr_a / px)
    # per-slab noise amplitude so the summed variance matches the thin model
    amp = specimen.solvent_noise * np.sqrt(dz_a / 10.0) * px
    for iz in range(nz):
        z = (iz + 0.5) * dz_a
        V = np.zeros(shape, dtype=np.float32)
        for (cy, cx, R_a, cz) in spheres:
            if abs(z - cz) < R_a:
                rho_px = np.sqrt(R_a ** 2 - (z - cz) ** 2) / px
                _disk_add(V, cy, cx, rho_px, d_excess * dz_a)
        if specimen.solvent_noise > 0:
            from scipy.ndimage import gaussian_filter
            nf = gaussian_filter(rng.standard_normal(shape).astype(np.float32), corr_px)
            V += amp * (nf / (nf.std() + 1e-8))
        yield V


def _antialias_mask(shape, px):
    """2/3-rule band-limiting aperture: zero the outer third of frequencies so
    the Fresnel propagator can't alias high-k content back over many slabs."""
    k = _freq_grid(shape, px)
    k_ny = 0.5 / px                                 # Nyquist (cycles/Å)
    return (k <= (2.0 / 3.0) * k_ny).astype(np.float32)


def multislice_exit_wave(shape, px, dz_a, slabs, lam_a, sigma, q=0.0, use_gpu=None):
    """Propagate a plane wave through the slabs; return the complex exit wave.

    Runs the per-slab FFT loop on the GPU when CuPy is available (use_gpu=None
    auto-selects; True forces GPU, False forces CPU). Specimen slabs are built on
    the CPU and streamed to the device."""
    from .backend import get_xp, to_cpu
    xp, on_gpu = get_xp(use_gpu)
    P = (_fresnel(shape, px, dz_a, lam_a) * _antialias_mask(shape, px)).astype(np.complex64)
    P = xp.asarray(P)
    psi = xp.ones(shape, dtype=xp.complex64)
    mu = q * sigma                                  # absorptive part -> amplitude contrast
    for V in slabs:
        Vx = xp.asarray(V)
        psi *= xp.exp(1j * sigma * Vx - mu * xp.abs(Vx))
        psi = xp.fft.ifft2(xp.fft.fft2(psi) * P).astype(xp.complex64)
    return to_cpu(psi)


def coherent_ctf(shape, px, cfg, defocus_um, bfactor=40.0, dose_damage=True):
    """Complex wave transfer function Env(k)*exp(-i*chi(k)) for the exit wave."""
    from .optics import dose_damage_envelope
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    cs_a = cfg["cs_mm"] * 1e7
    df_a = -defocus_um * 1e4
    k = _freq_grid(shape, px); k2 = k ** 2
    chi = np.pi * lam * df_a * k2 - 0.5 * np.pi * cs_a * lam ** 3 * k2 ** 2

    cc_a = cfg["cc_mm"] * 1e7
    d_spread = cc_a * (cfg["energy_spread_ev"] / (float(cfg["voltage_kv"]) * 1e3))
    e_t = np.exp(-0.5 * (np.pi * lam * d_spread * k2) ** 2)
    alpha = cfg["convergence_mrad"] * 1e-3
    grad = cs_a * lam ** 3 * k2 * k - df_a * lam * k
    e_s = np.exp(-(np.pi * alpha / lam) ** 2 * grad ** 2)
    e_b = np.exp(-bfactor * k2 / 4.0)
    e_dmg = dose_damage_envelope(shape, px, cfg) if dose_damage else 1.0
    return (e_t * e_s * e_b * e_dmg * np.exp(-1j * chi)).astype(np.complex64)


def image_from_wave(psi, H):
    """Objective transfer + intensity: I = |IFFT(FFT(psi) * H)|^2."""
    img = np.fft.ifft2(np.fft.fft2(psi) * H)
    return (np.abs(img) ** 2).astype(np.float32)


@dataclass
class MultisliceResult:
    ideal: np.ndarray
    exit_amp: np.ndarray
    defocus_um: float
    dz_a: float
    n_slabs: int
    dose_scale: float = 1.0


def simulate_multislice(cfg, specimen=None, defocus_um=-1.5, dz_a=40.0,
                        bfactor=40.0, apodize=0.1, inelastic=True):
    """Thick-specimen forward model up to the noiseless image (feed to detector)."""
    from .setup import sigma_rad_per_VA
    n = int(cfg["image_size_px"]); shape = (n, n); px = cfg["pixel_size_a"]
    specimen = specimen or Specimen()
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))
    T_a = specimen.ice_thickness_nm * 10.0
    n_slabs = max(1, int(np.ceil(T_a / dz_a)))

    psi = multislice_exit_wave(shape, px, dz_a,
                               plga_slabs(shape, px, dz_a, specimen),
                               lam, sigma, q=cfg["amplitude_contrast"])
    if apodize:                                    # taper exit-wave deviation from vacuum
        w = _tukey2d(shape, apodize)
        psi = 1.0 + (psi - 1.0) * w
    H = coherent_ctf(shape, px, cfg, defocus_um, bfactor=bfactor)
    ideal = image_from_wave(psi, H)
    ideal = ideal / (ideal.mean() + 1e-8)          # normalise mean ~1 for the detector
    dose_scale = 1.0
    if inelastic:
        ideal, dose_scale = apply_inelastic(ideal, specimen.ice_thickness_nm, cfg)
    r = MultisliceResult(ideal=ideal, exit_amp=np.abs(psi).astype(np.float32),
                         defocus_um=defocus_um, dz_a=dz_a, n_slabs=n_slabs)
    r.dose_scale = dose_scale
    return r
