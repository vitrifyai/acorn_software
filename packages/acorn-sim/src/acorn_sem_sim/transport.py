"""Monte Carlo electron transport in a bulk solid — the first-principles core.

This is the piece the FIB-SEM simulator did not have. Where the cryo-TEM engine
computes an exit wave and a CTF, SEM has no coherent transmitted wave at all:
the beam is a focused probe that scatters to rest inside the specimen, and the
image is built from whatever escapes back through the surface. There is no
transfer function to oscillate, no Thon rings, no contrast reversal — the
resolution limit is the *interaction volume*, and the only honest way to get it
is to trace electrons.

Model (the classic Joy formulation, as used by CASINO and WinXRay):

    elastic scattering   screened Rutherford cross-section, single-scattering
    energy loss          Joy-Luo modified Bethe continuous slowing down
    termination          electron leaves through z<0, or drops below E_cut

Validated in `tests/test_sem_sim.py` against published backscatter yields at
20 keV (C 0.06, Si 0.17, Cu 0.31, Ag 0.42, Au 0.49) and against the
Kanaya-Okayama range.

Known limitation, stated because it matters for Pt caps and Au markers:
screened Rutherford overestimates eta for heavy elements — gold comes out
around 12 % high. Mott cross-section tables are the standard fix and would drop
in at `_elastic_mfp_cm` without disturbing anything else.

Everything is vectorised over electrons: one Python loop iteration advances the
whole surviving population by one scattering step, so 10^5 trajectories cost
seconds rather than minutes. That matters because the kernels this feeds are
precomputed per (material, energy, tilt) and cached, never evaluated per pixel.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .materials import Material

# Electrons slower than this are considered absorbed. 50 eV is also the
# conventional dividing line between a "backscattered" and a "secondary"
# electron, which is why it appears twice below.
E_CUT_KEV = 0.05


@dataclass
class TransportResult:
    """What came back out of the surface, and where."""

    eta:          float          # backscatter yield: BSE out / electrons in
    bse_r_nm:     np.ndarray     # lateral exit radius of each BSE
    bse_E_kev:    np.ndarray     # exit energy of each BSE
    se_r_nm:      np.ndarray     # lateral origin of each escaping SE packet
    se_weight:    np.ndarray     # escaping SE yield contributed by that packet
    delta:        float          # secondary yield: SE out / electrons in
    n_electrons:  int


def _mean_ionisation_kev(Z: float) -> float:
    """Berger-Seltzer mean ionisation potential J, keV."""
    return (9.76 * Z + 58.5 * Z ** -0.19) * 1e-3


def _elastic_mfp_cm(E_kev, Z: float, A: float, rho: float):
    """Screened Rutherford total cross-section -> elastic mean free path, cm."""
    alpha = 3.4e-3 * Z ** 0.67 / E_kev
    sigma = (5.21e-21 * (Z ** 2 / E_kev ** 2)
             * (4.0 * np.pi / (alpha * (1.0 + alpha)))
             * ((E_kev + 511.0) / (E_kev + 1022.0)) ** 2)      # cm^2/atom
    n_atoms = 6.022e23 * rho / A                                # atoms/cm^3
    return 1.0 / (n_atoms * sigma)


def _dEds_kev_per_cm(E_kev, Z: float, A: float, rho: float):
    """Joy-Luo modified Bethe stopping power, keV/cm. Always negative.

    The unmodified Bethe expression diverges below the mean ionisation
    potential; the Joy-Luo substitution E -> E + k*J keeps it finite and is
    accurate to a few hundred eV, which is where SE generation lives.
    """
    J = _mean_ionisation_kev(Z)
    E_eff = np.maximum(E_kev + 0.85 * J, 1.05 * J)      # keep log argument > 1
    return -78500.0 * rho * (Z / (A * E_kev)) * np.log(1.166 * E_eff / J)


def _scatter(cx, cy, cz, cos_t, phi):
    """Rotate direction cosines by polar angle cos_t and azimuth phi.

    The degenerate case must be tested on sin^2 = 1 - cz^2 *before* clamping it
    away from zero. Testing the clamped denominator instead lets a beam that
    starts exactly along +z scatter only along +-z: it never acquires lateral
    motion, every backscattered electron then exits at radius zero, and the
    interaction volume silently collapses to a line.
    """
    st = np.sqrt(np.maximum(0.0, 1.0 - cos_t ** 2))
    sp, cp = np.sin(phi), np.cos(phi)

    sin2 = 1.0 - cz ** 2
    on_axis = sin2 < 1e-12
    denom = np.sqrt(np.maximum(sin2, 1e-12))

    nx = cx * cos_t + st * (cx * cz * cp - cy * sp) / denom
    ny = cy * cos_t + st * (cy * cz * cp + cx * sp) / denom
    nz = cz * cos_t - denom * st * cp

    nx = np.where(on_axis, st * cp, nx)
    ny = np.where(on_axis, st * sp, ny)
    nz = np.where(on_axis, np.sign(cz) * cos_t, nz)

    n = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)
    return nx / n, ny / n, nz / n


def trace(material: Material,
          E0_kev: float = 20.0,
          n_electrons: int = 20_000,
          tilt_deg: float = 0.0,
          seed: int = 0,
          max_steps: int = 4000) -> TransportResult:
    """Fire `n_electrons` into a bulk half-space (solid occupies z > 0).

    The beam enters at the origin. `tilt_deg` tilts the *surface* relative to
    the beam, which is what a rough specimen presents to a fixed column: it
    shortens the path to the surface and so raises both yields. This is the
    physical origin of the secant law that topographic contrast rests on, and
    here it emerges rather than being imposed.
    """
    if material.rho <= 0:                       # vacuum: nothing comes back
        empty = np.array([], dtype=np.float64)
        return TransportResult(0.0, empty, empty, empty, empty, 0.0, n_electrons)

    Z, A, rho = material.Z, material.A, material.rho
    rng = np.random.default_rng(seed)
    n = int(n_electrons)

    x = np.zeros(n); y = np.zeros(n); z = np.zeros(n)
    theta = np.radians(tilt_deg)
    cx = np.full(n, np.sin(theta)); cy = np.zeros(n); cz = np.full(n, np.cos(theta))
    E = np.full(n, float(E0_kev))
    alive = np.ones(n, dtype=bool)

    bse_r: list[np.ndarray] = []
    bse_E: list[np.ndarray] = []
    se_r:  list[np.ndarray] = []
    se_w:  list[np.ndarray] = []

    lam_esc_cm = material.se_escape_nm * 1e-7
    eps_kev = material.epsilon_ev * 1e-3

    for _ in range(max_steps):
        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        Ei = E[idx]
        lam = _elastic_mfp_cm(Ei, Z, A, rho)
        s = -lam * np.log(rng.random(idx.size))

        x0, y0, z0 = x[idx], y[idx], z[idx]
        x1 = x0 + s * cx[idx]
        y1 = y0 + s * cy[idx]
        z1 = z0 + s * cz[idx]

        # --- secondary electrons generated along this step -------------------
        # SEs are produced in proportion to energy deposited and escape with
        # probability exp(-z/lambda_SE). The escape layer is nanometres deep
        # while a step is tens of nanometres, so evaluating the exponential at
        # the step midpoint is not a small error -- it smears the whole escape
        # layer across the step and overstates delta by an order of magnitude.
        # Integrate along the segment instead:
        #
        #   N_SE = (|dE/ds| / eps) * INT_0^s exp(-(z0 + t*cz)/lambda) dt
        #
        # which is analytic, and reduces to s*exp(-z0/lambda) as cz -> 0.
        dEds = -_dEds_kev_per_cm(Ei, Z, A, rho)          # keV/cm, positive
        czi = cz[idx]
        z_start = np.maximum(z0, 0.0)

        # Integrate only over the part of the step still inside the solid. An
        # electron heading up (cz < 0) leaves at t = z0 / -cz; carrying the
        # integral past that point makes exp(-z/lambda) grow without bound and
        # delta diverges.
        with np.errstate(divide="ignore", invalid="ignore"):
            t_surface = np.where(czi < 0, z_start / np.maximum(-czi, 1e-30), np.inf)
        s_eff = np.minimum(s, np.maximum(t_surface, 0.0))
        z_end = np.maximum(z_start + s_eff * czi, 0.0)

        # INT exp(-z/lam) dt = (exp(-z0/lam) - exp(-z_end/lam)) * lam / cz.
        # Written this way both exponentials are bounded by 1, so nothing
        # overflows however deep the electron is.
        e0 = np.exp(-z_start / lam_esc_cm)
        e1 = np.exp(-z_end / lam_esc_cm)
        flat = np.abs(czi) < 1e-9
        path_integral = np.where(
            flat,
            s_eff * e0,
            (e0 - e1) * lam_esc_cm / np.where(flat, 1.0, czi),
        )
        path_integral = np.maximum(path_integral, 0.0)

        n_se = (dEds / eps_kev) * path_integral
        keep = n_se > 1e-6
        if keep.any():
            r_mid = np.sqrt((0.5 * (x0 + x1)) ** 2 + (0.5 * (y0 + y1)) ** 2)
            se_r.append(r_mid[keep] * 1e7)
            se_w.append(n_se[keep])

        # --- electrons crossing back through the surface ---------------------
        exited = z1 < 0.0
        if exited.any():
            e_idx = idx[exited]
            span = z0[exited] - z1[exited]
            f = np.where(span > 0, z0[exited] / np.where(span > 0, span, 1.0), 0.0)
            xe = x0[exited] + f * (x1[exited] - x0[exited])
            ye = y0[exited] + f * (y1[exited] - y0[exited])
            Ee = E[e_idx]
            is_bse = Ee > E_CUT_KEV
            if is_bse.any():
                bse_r.append(np.sqrt(xe[is_bse] ** 2 + ye[is_bse] ** 2) * 1e7)
                bse_E.append(Ee[is_bse])
            alive[e_idx] = False

        # --- electrons that stayed in: lose energy, scatter -------------------
        stay = idx[~exited]
        if stay.size:
            ss = s[~exited]
            x[stay] = x1[~exited]; y[stay] = y1[~exited]; z[stay] = z1[~exited]
            E[stay] = E[stay] + _dEds_kev_per_cm(E[stay], Z, A, rho) * ss

            spent = E[stay] < E_CUT_KEV
            alive[stay[spent]] = False
            live = stay[~spent]
            if live.size:
                a = 3.4e-3 * Z ** 0.67 / E[live]
                R = rng.random(live.size)
                cos_t = 1.0 - 2.0 * a * R / (1.0 + a - R)
                phi = 2.0 * np.pi * rng.random(live.size)
                cx[live], cy[live], cz[live] = _scatter(
                    cx[live], cy[live], cz[live], cos_t, phi)

    def _cat(parts):
        return np.concatenate(parts) if parts else np.array([], dtype=np.float64)

    bse_r_a, bse_E_a = _cat(bse_r), _cat(bse_E)
    se_r_a, se_w_a = _cat(se_r), _cat(se_w)

    return TransportResult(
        eta=bse_r_a.size / n,
        bse_r_nm=bse_r_a,
        bse_E_kev=bse_E_a,
        se_r_nm=se_r_a,
        se_weight=se_w_a,
        delta=float(se_w_a.sum()) / n if se_w_a.size else 0.0,
        n_electrons=n,
    )
