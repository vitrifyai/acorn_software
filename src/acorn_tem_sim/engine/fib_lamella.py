"""FIB-lamella hand-off: mill a thick specimen, then image it by cryo-TEM.

This is the join between the two simulators. A whole cell (hundreds of nm to
~1 um) is too thick for useful TEM phase contrast, so in practice it is
cryo-FIB-milled into a ~150-250 nm lamella and *that* is imaged. Here we:

    1. take a thick specimen's 3-D slab stack (e.g. a Bacterium),
    2. MILL: keep only the slabs within a lamella window along the beam (z),
       so the beam traverses ~lamella_nm instead of the whole cell,
    3. add the FIB signatures the fibsim side taught us -- vertical CURTAINING
       (multiplicative stripes through the lamella) and a thin amorphised
       DAMAGE layer at each milled face (Ga+ implantation),
    4. run the multislice forward model on the thinned stack.

The payoff: the milled lamella has far higher contrast than the intact cell,
and reveals the interior cross-section -- exactly why cryo-FIB + cryo-ET exists.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .multislice import multislice_exit_wave, coherent_ctf, image_from_wave
from .optics import _tukey2d
from .setup import sigma_rad_per_VA
from .bacteria import Bacterium, bacterium_slabs


def _curtain_profile(width, rng):
    """1-D pink-noise vertical-stripe profile across the lamella width."""
    white = rng.standard_normal(width)
    freq = np.fft.rfftfreq(width, d=1.0); freq[0] = freq[1]
    prof = np.fft.irfft(np.fft.rfft(white) / freq, n=width)
    prof = (prof - prof.mean()) / (prof.std() + 1e-8)
    return gaussian_filter1d(prof, 1.0)


def mill_slabs(slab_iter, shape, px, dz_a, total_thick_a, lamella_nm,
               curtain=0.15, damage_nm=15.0, rng=None):
    """Wrap a slab generator: keep the lamella window, add curtaining + damage."""
    rng = rng or np.random.default_rng(0)
    lam_a = lamella_nm * 10.0
    z_lo = 0.5 * total_thick_a - 0.5 * lam_a
    z_hi = z_lo + lam_a
    dmg_a = damage_nm * 10.0
    stripes = np.broadcast_to(_curtain_profile(shape[1], rng)[None, :], shape).copy()
    corr_px = max(0.5, 3.5 / px)

    for iz, V in enumerate(slab_iter):
        z = (iz + 0.5) * dz_a
        if z < z_lo or z > z_hi:
            continue                                     # milled away
        Vm = V * (1.0 + curtain * stripes)               # curtaining
        # Amorphised damage layer near each milled face: extra ice-like noise.
        d_face = min(z - z_lo, z_hi - z)
        if d_face < dmg_a:
            nf = gaussian_filter1d(rng.standard_normal(shape).astype(np.float32),
                                   corr_px, axis=1)
            Vm = Vm + (dz_a * 0.8) * (nf / (nf.std() + 1e-8))
        yield Vm


def simulate_fib_lamella(cfg, bact: Bacterium | None = None, lamella_nm=200.0,
                         curtain=0.15, damage_nm=15.0, defocus_um=-3.0,
                         dz_a=40.0, bfactor=60.0, apodize=0.1, seed=0):
    """Mill a bacterium to a lamella and image it with multislice."""
    bact = bact or Bacterium()
    rng = np.random.default_rng(seed)
    n = int(cfg["image_size_px"]); shape = (n, n); px = cfg["pixel_size_a"]
    lam = cfg["_derived"]["wavelength_pm"] * 1e-2
    sigma = sigma_rad_per_VA(float(cfg["voltage_kv"]))
    total_a = bact.thickness_nm() * 10.0
    kept = max(1, int(round(min(lamella_nm * 10.0, total_a) / dz_a)))

    slabs = mill_slabs(bacterium_slabs(shape, px, dz_a, bact), shape, px, dz_a,
                       total_a, lamella_nm, curtain, damage_nm, rng)
    psi = multislice_exit_wave(shape, px, dz_a, slabs, lam, sigma,
                               q=cfg["amplitude_contrast"])
    b = int(max(2, apodize * min(shape)))
    edge = np.concatenate([psi[:b].ravel(), psi[-b:].ravel(),
                           psi[:, :b].ravel(), psi[:, -b:].ravel()])
    psi = edge.mean() + (psi - edge.mean()) * _tukey2d(shape, apodize)

    ideal = image_from_wave(psi, coherent_ctf(shape, px, cfg, defocus_um, bfactor))
    ideal = ideal / (ideal.mean() + 1e-8)
    # Inelastic loss is set by the LAMELLA thickness (not the whole cell) — the
    # milled slab is what the beam traverses. This is the lamella SNR payoff.
    from .multislice import apply_inelastic
    eff_thick_nm = min(lamella_nm, bact.thickness_nm())
    ideal, dose_scale = apply_inelastic(ideal, eff_thick_nm, cfg)
    return ideal, {"species": bact.species, "lamella_nm": lamella_nm,
                   "kept_slabs": kept, "cell_thickness_nm": bact.thickness_nm(),
                   "defocus_um": defocus_um, "dose_scale": dose_scale}
