"""Validation for the SEM Monte Carlo transport model.

These are physics regression tests, not unit tests. They assert that the model
reproduces measured quantities from the literature, because that is the only
thing that makes a "first-principles" claim mean anything. If a refactor breaks
the physics, these fail rather than the code silently producing plausible
pictures.

References for the target values: Reimer, *Scanning Electron Microscopy*;
Joy's backscatter database; Kanaya & Okayama (1972) for the range formula.
"""
from __future__ import annotations

import numpy as np
import pytest

from acorn_sem_sim import materials as M
from acorn_sem_sim import transport as T

# Backscatter yield at 20 keV, normal incidence. These are PREDICTED by the
# model -- there is no parameter that could be tuned to hit them.
ETA_20KEV = {
    "carbon":  0.06,
    "silicon": 0.17,
    "copper":  0.31,
    "silver":  0.42,
    "gold":    0.49,
}

# Secondary yield at 20 keV. These are reproduced BY CONSTRUCTION via the
# calibrated epsilon in materials.py, so this test guards the calibration
# rather than validating the physics.
DELTA_20KEV = {
    "carbon":  0.05,
    "silicon": 0.10,
    "copper":  0.13,
    "gold":    0.20,
}

N = 20_000


@pytest.mark.parametrize("name,eta_ref", sorted(ETA_20KEV.items()))
def test_backscatter_yield_matches_published(name, eta_ref):
    """eta within 15% of published, except gold.

    Screened Rutherford is known to overestimate backscattering for heavy
    elements; gold is allowed 20%. Mott cross-sections would tighten this and
    are the documented upgrade path.
    """
    r = T.trace(M.get(name), E0_kev=20.0, n_electrons=N, seed=1)
    tol = 0.20 if name == "gold" else 0.15
    assert r.eta == pytest.approx(eta_ref, rel=tol), (
        f"{name}: eta={r.eta:.3f}, published={eta_ref:.3f}"
    )


def test_backscatter_yield_increases_with_atomic_number():
    """The monotonic eta(Z) relation is what all BSE compositional contrast rests on."""
    etas = [T.trace(M.get(n), 20.0, n_electrons=8000, seed=2).eta
            for n in ("carbon", "silicon", "copper", "silver", "gold")]
    assert etas == sorted(etas), f"eta not monotonic in Z: {etas}"


@pytest.mark.parametrize("name,delta_ref", sorted(DELTA_20KEV.items()))
def test_secondary_yield_matches_calibration(name, delta_ref):
    r = T.trace(M.get(name), E0_kev=20.0, n_electrons=N, seed=1)
    assert r.delta == pytest.approx(delta_ref, rel=0.25), (
        f"{name}: delta={r.delta:.3f}, target={delta_ref:.3f}"
    )


def test_secondary_yield_peaks_at_low_energy():
    """delta(E0) must rise to a maximum near 0.5-1 keV and fall away.

    This shape is a genuine prediction: it comes out of the competition between
    more energy deposited at higher E0 and that energy being deposited deeper
    than the escape depth. Getting it for free is the strongest evidence the
    transport is behaving.
    """
    energies = np.array([0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    deltas = np.array([T.trace(M.get("silicon"), float(e), n_electrons=6000, seed=3).delta
                       for e in energies])
    peak = energies[int(np.argmax(deltas))]
    assert peak <= 1.0, f"delta peaked at {peak} keV, expected <= 1 keV"
    assert deltas[-1] < deltas[int(np.argmax(deltas))] / 3.0, (
        f"delta did not fall away by 20 keV: {deltas}"
    )


@pytest.mark.parametrize("name", ["carbon", "silicon", "copper", "gold"])
def test_interaction_volume_within_kanaya_okayama_range(name):
    """95th-percentile BSE exit radius must sit inside the KO range.

    KO is the maximum penetration depth, so the lateral spread of escaping
    backscatters has to be comfortably smaller. A model that violates this is
    diffusing electrons too far.
    """
    mat = M.get(name)
    r = T.trace(mat, 20.0, n_electrons=N, seed=1)
    assert r.bse_r_nm.size > 100
    r95 = float(np.percentile(r.bse_r_nm, 95))
    ko = M.kanaya_okayama_nm(mat, 20.0)
    assert 0.0 < r95 < ko, f"{name}: r95={r95:.0f} nm vs KO range {ko:.0f} nm"


def test_interaction_volume_shrinks_with_atomic_number():
    """Heavy elements backscatter closer to the entry point.

    This is the physical reason BSE imaging of a heavy phase is sharper than of
    a light matrix, and it is exactly the effect the old Gaussian-MTF FIB-SEM
    model could not represent.
    """
    r95 = []
    for n in ("carbon", "silicon", "copper", "gold"):
        r = T.trace(M.get(n), 20.0, n_electrons=8000, seed=4)
        r95.append(float(np.percentile(r.bse_r_nm, 95)))
    assert r95 == sorted(r95, reverse=True), f"r95 not decreasing with Z: {r95}"


def test_interaction_volume_grows_with_beam_energy():
    """Lower kV is the standard way to get surface-sensitive SEM. Model must agree."""
    r95 = []
    for e in (2.0, 5.0, 20.0):
        r = T.trace(M.get("silicon"), e, n_electrons=8000, seed=5)
        r95.append(float(np.percentile(r.bse_r_nm, 95)))
    assert r95 == sorted(r95), f"interaction volume not growing with E0: {r95}"


def test_tilting_the_surface_raises_both_yields():
    """The secant law, emerging rather than imposed.

    Topographic contrast in SEM exists because a tilted surface puts more of the
    interaction volume within escape range. `sem_physics.py` applies this as an
    analytic 1/cos(theta); here it should fall out of the transport itself.
    """
    flat = T.trace(M.get("silicon"), 5.0, n_electrons=12000, tilt_deg=0.0, seed=6)
    tilt = T.trace(M.get("silicon"), 5.0, n_electrons=12000, tilt_deg=60.0, seed=6)
    assert tilt.delta > flat.delta, (flat.delta, tilt.delta)
    assert tilt.eta > flat.eta, (flat.eta, tilt.eta)


def test_vacuum_returns_nothing():
    r = T.trace(M.get("vacuum"), 20.0, n_electrons=500, seed=0)
    assert r.eta == 0.0 and r.delta == 0.0 and r.bse_r_nm.size == 0


def test_yields_are_finite_across_the_useful_energy_range():
    """Guards the two overflow bugs this model had while it was being written:
    a degenerate on-axis rotation, and integrating SE escape past the surface."""
    for name in ("biology", "silicon", "gold"):
        for e in (0.5, 1.0, 5.0, 30.0):
            r = T.trace(M.get(name), e, n_electrons=3000, seed=7)
            assert np.isfinite(r.delta) and 0.0 <= r.delta < 50.0, (name, e, r.delta)
            assert np.isfinite(r.eta) and 0.0 <= r.eta <= 1.0, (name, e, r.eta)


def test_backscatters_land_off_axis():
    """Every BSE exiting at radius exactly zero means the direction rotation has
    collapsed to the beam axis -- the bug this model shipped with initially."""
    r = T.trace(M.get("copper"), 20.0, n_electrons=8000, seed=8)
    assert float(np.median(r.bse_r_nm)) > 1.0


def test_mix_weight_averages_and_normalises():
    half = M.mix({"gold": 1.0, "carbon": 1.0})
    assert half.Z == pytest.approx((79.0 + 6.0) / 2)
    # unnormalised weights must give the same answer as normalised ones
    same = M.mix({"gold": 5.0, "carbon": 5.0})
    assert same.Z == pytest.approx(half.Z)


def test_unknown_material_names_what_exists():
    with pytest.raises(KeyError, match="silicon"):
        M.get("unobtainium")
