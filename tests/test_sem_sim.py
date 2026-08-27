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

import json

import numpy as np
import pytest
from acorn_sem_sim import imaging as IM
from acorn_sem_sim import kernels as K
from acorn_sem_sim import materials as M
from acorn_sem_sim import scenes as SC
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


# ---------------------------------------------------------------------------
# Kernels
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def si_kernels():
    return K.compute(M.get("silicon"), 5.0, n_electrons=30_000, seed=0, use_cache=False)


def test_kernel_cdf_reproduces_trajectory_quantiles(si_kernels):
    """The kernel is a lossy summary of the trajectories; it must not be a
    distorting one. Reconstructing enclosed weight by integrating a binned
    areal density put the silicon SE core at 1.7 nm when the trajectories say
    0.40 nm, because the density diverges at the origin. Accumulating the
    histogram directly is exact, and this test pins that down."""
    r = T.trace(M.get("silicon"), 5.0, n_electrons=30_000, seed=0)
    order = np.argsort(r.se_r_nm)
    cw = np.cumsum(r.se_weight[order]) / r.se_weight.sum()
    for frac in (0.5, 0.9, 0.95):
        direct = r.se_r_nm[order][np.searchsorted(cw, frac)]
        via_kernel = si_kernels.se.radius_containing(frac)
        assert via_kernel == pytest.approx(direct, rel=0.10), (
            f"r{int(frac * 100)}: kernel {via_kernel:.2f} nm vs trajectories {direct:.2f} nm"
        )


@pytest.mark.parametrize("px", [0.5, 1.0, 5.0, 20.0, 50.0])
def test_kernel_normalised(si_kernels, px):
    assert si_kernels.se.to_pixels(px).sum() == pytest.approx(1.0, rel=1e-5)
    assert si_kernels.bse.to_pixels(px).sum() == pytest.approx(1.0, rel=1e-5)


def test_kernel_centre_weight_rises_with_pixel_size(si_kernels):
    """A coarser pixel must capture more of the core, never less. Violating this
    is the signature of under-resolving the sub-nanometre SE1 core."""
    fracs = []
    for px in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        a = si_kernels.se.to_pixels(px)
        fracs.append(float(a[a.shape[0] // 2, a.shape[1] // 2]))
    assert fracs == sorted(fracs), f"centre fraction not monotonic: {fracs}"


def test_se_kernel_is_far_more_peaked_than_bse(si_kernels):
    """The SE1/SE2 split: a sub-nanometre core on a pedestal hundreds of nm
    wide. This is precisely what a single Gaussian MTF cannot represent, and it
    is the reason this module exists."""
    se50 = si_kernels.se.radius_containing(0.5)
    bse50 = si_kernels.bse.radius_containing(0.5)
    se95 = si_kernels.se.radius_containing(0.95)
    assert se50 < bse50 / 20, f"SE r50={se50:.2f} vs BSE r50={bse50:.1f}"
    assert se95 > 20 * se50, f"SE kernel has no broad pedestal: r50={se50}, r95={se95}"


def test_bse_kernel_narrows_with_atomic_number():
    r50 = []
    for name in ("carbon", "silicon", "copper", "gold"):
        k = K.compute(M.get(name), 10.0, n_electrons=12_000, seed=0, use_cache=False)
        r50.append(k.bse.radius_containing(0.5))
    assert r50 == sorted(r50, reverse=True), f"BSE kernel not narrowing with Z: {r50}"


def test_yields_rise_with_tilt_but_stay_below_secant_law(si_kernels):
    """Tilt dependence comes from transport, so unlike an imposed 1/cos(theta)
    it saturates near grazing incidence instead of diverging."""
    d0 = si_kernels.delta_at(0.0)
    d60 = si_kernels.delta_at(60.0)
    d80 = si_kernels.delta_at(80.0)
    assert d0 < d60 < d80
    assert d80 / d0 < 1.0 / np.cos(np.radians(80.0)), "yield exceeded the secant law"


def test_kernel_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    mat = M.get("copper")
    a = K.compute(mat, 5.0, n_electrons=4000, seed=0, use_cache=True)
    b = K.compute(mat, 5.0, n_electrons=4000, seed=0, use_cache=True)
    assert np.allclose(a.se.cum, b.se.cum)
    assert np.allclose(a.delta, b.delta)
    assert len(list((tmp_path / "acorn" / "sem_kernels").glob("*.npz"))) == 1


# ---------------------------------------------------------------------------
# Image formation
# ---------------------------------------------------------------------------

def _discs(n=192, r=18, centres=((60, 60), (130, 120))):
    idx = np.zeros((n, n), dtype=int)
    yy, xx = np.mgrid[0:n, 0:n]
    for cy, cx in centres:
        idx[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = 1
    return idx


@pytest.fixture(scope="module")
def au_in_carbon():
    idx = _discs()
    return idx, IM.simulate(idx, ["carbon", "gold"],
                            beam=IM.Beam(5.0, 4.0, 400.0),
                            n_electrons=12_000, seed=1)


def test_simulate_returns_ground_truth_unchanged(au_in_carbon):
    idx, res = au_in_carbon
    assert res.image.shape == idx.shape
    assert np.array_equal(res.material_index, idx)
    assert res.material_names == ["carbon", "gold"]
    assert 0.0 <= res.meta["boundary_fraction"] <= 1.0


def test_backscatter_gives_stronger_compositional_contrast_than_secondaries(au_in_carbon):
    """BSE imaging is the Z-contrast mode; that must fall out of the model."""
    idx, res = au_in_carbon
    au, c = idx == 1, idx == 0
    se_ratio = res.se[au].mean() / res.se[c].mean()
    bse_ratio = res.bse[au].mean() / res.bse[c].mean()
    assert bse_ratio > se_ratio > 1.0, (se_ratio, bse_ratio)


def test_per_material_kernels_differ_from_one_shared_kernel():
    """The claim the design rests on.

    Gold's interaction volume is far smaller than carbon's, so convolving both
    phases with a single kernel measurably changes the image. If this test ever
    passes trivially, the per-material convolution has stopped doing anything
    and the extra FFTs are waste.
    """
    idx = _discs()
    beam = IM.Beam(20.0, 4.0, 400.0)
    proper = IM.simulate(idx, ["carbon", "gold"], beam=beam,
                         n_electrons=12_000, seed=2)

    # Same scene, but force both phases through carbon's kernel by relabelling
    # gold as a material with carbon's transport and gold's yields.
    kc = K.compute(M.get("carbon"), beam.E0_kev, n_electrons=12_000, seed=2,
                   use_cache=False)
    kau = K.compute(M.get("gold"), beam.E0_kev, n_electrons=12_000, seed=2,
                    use_cache=False)
    yield_map = np.where(idx == 1, kau.eta_at(0.0), kc.eta_at(0.0))
    from scipy.signal import fftconvolve
    shared = fftconvolve(yield_map, kc.bse.to_pixels(beam.pixel_size_nm), mode="same")

    edge = np.abs(np.gradient(proper.bse)[0]).max()
    edge_shared = np.abs(np.gradient(shared)[0]).max()
    assert edge > 1.5 * edge_shared, (
        f"per-material kernels made no difference: {edge:.4g} vs {edge_shared:.4g}"
    )


def test_bse_detector_is_insensitive_to_topography():
    """An annular BSE detector gives composition, not shading; an ETD gives both."""
    from scipy.ndimage import gaussian_filter
    n = 160
    idx = np.zeros((n, n), dtype=int)
    rng = np.random.default_rng(0)
    h = gaussian_filter(rng.normal(0, 1, (n, n)), 8) * 400.0

    beam = IM.Beam(5.0, 4.0, 2000.0)
    etd = IM.simulate(idx, ["silicon"], height_nm=h, beam=beam,
                      detector=IM.Detector("ETD", asymmetry=0.6),
                      n_electrons=8000, seed=3)
    bse = IM.simulate(idx, ["silicon"], height_nm=h, beam=beam,
                      detector=IM.Detector("BSE"), n_electrons=8000, seed=3)
    # correlation of signal with the surface slope facing the detector
    _, gx = np.gradient(h, beam.pixel_size_nm)
    slope = gx.ravel()
    c_etd = abs(np.corrcoef(etd.signal.ravel(), slope)[0, 1])
    c_bse = abs(np.corrcoef(bse.signal.ravel(), slope)[0, 1])
    assert c_etd > c_bse, (c_etd, c_bse)


def test_more_electrons_per_pixel_improves_snr():
    idx = _discs()
    out = {}
    for dose in (50.0, 5000.0):
        r = IM.simulate(idx, ["carbon", "gold"], beam=IM.Beam(5.0, 4.0, dose),
                        n_electrons=8000, seed=4)
        au, c = idx == 1, idx == 0
        out[dose] = abs(r.image[au].mean() - r.image[c].mean()) / r.image[c].std()
    assert out[5000.0] > out[50.0] * 3


def test_vacuum_region_emits_nothing():
    idx = np.zeros((64, 64), dtype=int)
    idx[:32] = 1
    r = IM.simulate(idx, ["vacuum", "silicon"], beam=IM.Beam(5.0, 8.0, 500.0),
                    n_electrons=6000, seed=5)
    assert r.se[:16].mean() > r.se[48:].mean() * 5


def test_topography_alone_creates_contrast():
    """Flat single-material specimen must be featureless; tilting it must not be."""
    from scipy.ndimage import gaussian_filter
    n = 128
    idx = np.zeros((n, n), dtype=int)
    beam = IM.Beam(5.0, 4.0, 4000.0)
    flat = IM.simulate(idx, ["silicon"], beam=beam, n_electrons=8000, seed=6)
    rng = np.random.default_rng(1)
    h = gaussian_filter(rng.normal(0, 1, (n, n)), 6) * 500.0
    rough = IM.simulate(idx, ["silicon"], height_nm=h, beam=beam,
                        n_electrons=8000, seed=6)
    inner = (slice(20, -20), slice(20, -20))
    assert rough.signal[inner].std() > 10 * flat.signal[inner].std()


def test_rejects_labels_with_no_material():
    idx = np.array([[0, 5]], dtype=int)
    with pytest.raises(ValueError, match="no material"):
        IM.simulate(idx, ["silicon"])


def test_rejects_mismatched_height_map():
    with pytest.raises(ValueError, match="shape"):
        IM.simulate(np.zeros((8, 8), dtype=int), ["silicon"],
                    height_nm=np.zeros((4, 4)))


# ---------------------------------------------------------------------------
# Scenes, export, and CLU parameter handling
# ---------------------------------------------------------------------------




@pytest.mark.parametrize("kind", sorted(SC.BUILDERS))
def test_every_scene_builds_consistently(kind):
    s = SC.build(kind, shape=(96, 96), pixel_size_nm=6.0, seed=0)
    assert s.material_index.shape == (96, 96)
    assert s.height_nm.shape == (96, 96)
    assert s.material_index.min() >= 0
    assert s.material_index.max() < len(s.material_names)
    for name in s.material_names:
        M.get(name)                       # every name must resolve to a material
    assert s.description


def test_every_scene_is_listed_for_the_ui():
    """A scene the panel and CLU cannot name is a scene nobody can reach."""
    assert set(SC.BUILDERS) == set(SC.SCENE_LABELS)


def test_flat_scenes_really_are_flat():
    """The grains scene exists to isolate composition; relief would defeat it."""
    s = SC.build("grains", shape=(64, 64), pixel_size_nm=8.0, seed=0)
    assert float(np.abs(s.height_nm).max()) == 0.0


def test_relief_flag_controls_topography():
    on = SC.build("nanoparticles", shape=(128, 128), pixel_size_nm=4.0,
                  n_particles=8, relief=True, seed=1)
    off = SC.build("nanoparticles", shape=(128, 128), pixel_size_nm=4.0,
                   n_particles=8, relief=False, seed=1)
    assert float(np.abs(on.height_nm).max()) > 0
    assert float(np.abs(off.height_nm).max()) == 0.0
    assert np.array_equal(on.material_index, off.material_index)


def test_unknown_scene_lists_alternatives():
    with pytest.raises(KeyError, match="nanoparticles"):
        SC.build("banana")


def test_dataset_export_writes_images_truth_and_annotations(tmp_path):
    from acorn_sem_sim.io import generate_sem_dataset
    paths = generate_sem_dataset(tmp_path, 2, {
        "scene": "nanoparticles", "E0_kev": 5.0, "pixel_size_nm": 4.0,
        "image_size_px": 128, "n_particles": 10, "diameter_nm_mean": 40.0,
        "n_electrons": 4000,
    }, seed=0)
    assert len(paths) == 2 and all(p.exists() for p in paths)

    meta = json.loads((tmp_path / "metadata.json").read_text())
    assert len(meta["images"]) == 2
    assert all(i["annotations"] > 0 for i in meta["images"]), \
        "ground truth produced no annotations -- the point of simulating is lost"

    for name in ("_truth.tif", "_se.tif", "_bse.tif"):
        assert (tmp_path / "layers" / f"semsim_00000{name}").exists()

    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    assert ann["annotations"]
    first = ann["annotations"][0]
    assert first["label"] == "gold" and len(first["polygon"]) >= 3
    assert first["accepted"] is True


def test_export_does_not_annotate_background_or_vacuum(tmp_path):
    """A substrate filling the frame is not a useful annotation, and pores are
    holes rather than objects. Exporting either would swamp the real count."""
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": "porous", "E0_kev": 5.0, "pixel_size_nm": 8.0,
        "image_size_px": 128, "n_electrons": 4000,
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    labels = {a["label"] for a in ann["annotations"]}
    assert "vacuum" not in labels and "alumina" not in labels


# -- CLU parameter normalisation (pure logic, no Qt) -------------------------

def test_clu_accepts_the_synonyms_a_model_will_reach_for():
    from acorn_sem_sim.plugin import _params_from_clu
    p = _params_from_clu({"kv": 12, "sample": "alloy", "signal": "backscatter",
                          "pixel_nm": 3, "images": 4, "electrons_per_pixel": 250})
    assert p["E0_kev"] == 12.0
    assert p["scene"] == "grains"
    assert p["detector"] == "BSE"
    assert p["pixel_size_nm"] == 3.0
    assert p["count"] == 4
    assert p["electrons_per_px"] == 250.0


@pytest.mark.parametrize("text,expected", [
    ("in-lens", "TLD"), ("inlens", "TLD"), ("Z contrast", "ETD"),
    ("bse", "BSE"), ("backscattered", "BSE"), ("", "ETD"), (None, "ETD"),
])
def test_detector_synonyms(text, expected):
    from acorn_sem_sim.plugin import _detector_from_text
    assert _detector_from_text(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("cells", "biological"), ("resin", "biological"), ("pores", "porous"),
    ("lamella", "cross_section"), ("polished", "grains"), ("anything else", "nanoparticles"),
])
def test_scene_synonyms(text, expected):
    from acorn_sem_sim.plugin import _scene_from_text
    assert _scene_from_text(text) == expected


def test_clu_defaults_are_complete_enough_to_run(tmp_path):
    """An empty CLU call must still produce a runnable parameter set."""
    from acorn_sem_sim.io import generate_sem_dataset
    from acorn_sem_sim.plugin import _params_from_clu
    p = _params_from_clu({})
    p.update({"image_size_px": 96, "n_electrons": 3000, "count": 1})
    paths = generate_sem_dataset(tmp_path, 1, p, seed=0)
    assert len(paths) == 1 and paths[0].exists()


def test_field_of_view_warning_fires_only_when_it_should():
    from acorn_sem_sim.plugin import _field_of_view_warning
    ok = _field_of_view_warning({"E0_kev": 2.0, "pixel_size_nm": 4.0,
                                 "image_size_px": 512, "substrate": "gold"})
    bad = _field_of_view_warning({"E0_kev": 30.0, "pixel_size_nm": 1.0,
                                  "image_size_px": 128, "substrate": "carbon"})
    assert ok == ""
    assert "interaction volume" in bad and "lower the kV" in bad


# ---------------------------------------------------------------------------
# Regressions found by reading the code back
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", sorted(SC.BUILDERS))
def test_every_scene_exports_some_ground_truth(tmp_path, kind):
    """No scene may produce an empty truth file.

    The porous scene did exactly that. Its pores are the objects anyone would
    segment, they are made of vacuum, and the exporter skipped regions by
    MATERIAL name -- so the one scene whose objects are empty space silently
    shipped zero annotations while still looking like it had worked.
    """
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": kind, "image_size_px": 128, "pixel_size_nm": 8.0,
        "n_electrons": 3000, "n_particles": 10,
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    assert ann["annotations"], f"scene {kind!r} exported no ground truth"


def test_pores_are_annotated_as_pores_not_as_vacuum(tmp_path):
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": "porous", "image_size_px": 128, "pixel_size_nm": 8.0,
        "n_electrons": 3000,
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    labels = {a["label"] for a in ann["annotations"]}
    assert labels == {"pore"}, labels


def test_both_alloy_phases_are_annotated(tmp_path):
    """Treating either phase as background would export half the truth."""
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": "grains", "image_size_px": 128, "pixel_size_nm": 8.0,
        "n_electrons": 3000, "phase_a": "iron", "phase_b": "copper",
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    assert {a["label"] for a in ann["annotations"]} == {"iron", "copper"}


def test_substrate_is_not_annotated(tmp_path):
    """The support fills the frame; annotating it would swamp the object count."""
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": "nanoparticles", "image_size_px": 128, "pixel_size_nm": 8.0,
        "n_electrons": 3000, "n_particles": 10,
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    assert {a["label"] for a in ann["annotations"]} == {"gold"}


def test_scene_parameters_reach_the_dataset_record(tmp_path):
    """What a scene actually built, not just what was asked for.

    Requested and achieved differ -- particles are dropped when they will not
    fit without overlapping -- and a dataset record that only stores the request
    misdescribes its own contents.
    """
    from acorn_sem_sim.io import generate_sem_dataset
    generate_sem_dataset(tmp_path, 1, {
        "scene": "nanoparticles", "image_size_px": 128, "pixel_size_nm": 4.0,
        "n_electrons": 3000, "n_particles": 200, "diameter_nm_mean": 60.0,
    }, seed=0)
    rec = json.loads((tmp_path / "metadata.json").read_text())["images"][0]
    sp = rec["scene_params"]
    assert sp["n_particles_requested"] == 200
    assert 0 < sp["n_particles_placed"] < 200, "crowding was not recorded"
    assert rec["pixel_size_nm"] == 4.0


def test_shared_plugin_helpers_have_one_definition():
    """These were copied verbatim into three plugins, which is how three copies
    become three behaviours."""
    import acorn_sim_common as common
    from acorn_fib_sim import plugin as fib
    from acorn_sem_sim import plugin as sem
    from acorn_tem_sim import plugin as tem
    for mod in (tem, fib, sem):
        assert mod.fresh_run_dir is common.fresh_run_dir
        assert mod.open_paths_in_acorn is common.open_paths_in_acorn
        assert mod.as_bool is common.as_bool


@pytest.mark.parametrize("value,expected", [
    (True, True), (False, False), ("false", False), ("no", False),
    ("off", False), ("0", False), ("", False), ("true", True), ("yes", True),
])
def test_as_bool_handles_the_strings_a_model_sends(value, expected):
    """Plain bool("false") is True, which silently inverts the setting."""
    from acorn_sim_common import as_bool
    assert as_bool(value) is expected


def test_fresh_run_dir_never_reuses_a_path(tmp_path):
    from acorn_sim_common import fresh_run_dir
    seen = set()
    for _ in range(5):
        d = fresh_run_dir(tmp_path, "run")
        assert d not in seen
        d.mkdir(parents=True)
        seen.add(d)


def test_all_public_submodules_are_reachable():
    """`import acorn_sem_sim; acorn_sem_sim.scenes` must work. The lazy __getattr__
    only serves names in __all__, and two real submodules were missing from it."""
    import importlib

    import acorn_sem_sim as pkg
    for name in ("transport", "kernels", "imaging", "materials", "scenes", "io"):
        assert getattr(pkg, name) is importlib.import_module(f"acorn_sem_sim.{name}")


def test_bare_import_stays_headless():
    """The deferred imports exist so headless use never pays for Qt or scipy."""
    import subprocess
    import sys
    code = ("import sys, acorn_sem_sim; "
            "print('PyQt6' in sys.modules, 'scipy' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "False False", out.stdout


def test_plugin_can_actually_build_its_panel():
    """Construct the panel the way the plugin does.

    Removing a signal from the panel while the plugin still connected to it made
    every test here pass and the dock vanish from the running application. The
    unit tests build the panel directly and never exercised that connection.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from acorn_sem_sim.plugin import SemSimulationPlugin
    from PyQt6.QtWidgets import QApplication, QWidget

    app = QApplication.instance() or QApplication([])

    class _Context:
        """Minimal stand-in for AcornContext."""

        class _Signal:
            def connect(self, _slot):
                pass

        action_requested = _Signal()

        def set_status(self, *_a, **_k):
            pass

    plugin = SemSimulationPlugin(_Context())
    panel = plugin.create_panel()
    try:
        assert isinstance(panel, QWidget)
        assert panel.params()["scene"]        # the plugin reads this back
    finally:
        panel.deleteLater()
        app.processEvents()


# ---------------------------------------------------------------------------
# Bacterial spores
# ---------------------------------------------------------------------------

def test_spores_are_ovoid_and_not_all_aligned():
    """Discs on a flat field would make every detection task easier than it is.

    Spores are ovoid and lie at arbitrary orientations, so the footprint has to
    be an ellipse and the orientations must vary.
    """
    from skimage.measure import label, regionprops

    s = SC.build("spores", shape=(512, 512), pixel_size_nm=16.0, n_spores=20,
                 length_nm=1200.0, width_nm=800.0, clustering=0.0, seed=5)
    props = [p for p in regionprops(label(s.material_index == 1)) if p.area > 200]
    assert len(props) >= 8, len(props)

    elongation = [p.axis_major_length / max(p.axis_minor_length, 1e-6) for p in props]
    assert np.median(elongation) > 1.2, np.median(elongation)

    orientations = np.array([p.orientation for p in props])
    assert orientations.std() > 0.3, "spores are all aligned"


def test_spore_size_follows_the_requested_dimensions():
    from skimage.measure import label, regionprops

    px = 16.0
    s = SC.build("spores", shape=(512, 512), pixel_size_nm=px, n_spores=14,
                 length_nm=1600.0, width_nm=700.0, size_spread=0.0,
                 clustering=0.0, seed=2)
    props = [p for p in regionprops(label(s.material_index == 1)) if p.area > 200]
    major_nm = np.median([p.axis_major_length for p in props]) * px
    minor_nm = np.median([p.axis_minor_length for p in props]) * px
    assert major_nm == pytest.approx(1600.0, rel=0.15), major_nm
    assert minor_nm == pytest.approx(700.0, rel=0.20), minor_nm


def test_spores_stand_proud_of_the_substrate():
    """They sit ON the surface, so relief is real and roughly the short axis."""
    s = SC.build("spores", shape=(384, 384), pixel_size_nm=16.0, n_spores=12,
                 width_nm=800.0, seed=1)
    on = s.material_index == 1
    assert s.height_nm[on].max() == pytest.approx(400.0, rel=0.3)
    assert s.height_nm[on].mean() > s.height_nm[~on].mean() * 5


@pytest.mark.parametrize("coating,expected", [
    (10.0, "gold"), (25.0, "gold"), (0.0, "biology"),
])
def test_a_coated_spore_images_as_its_coating(coating, expected):
    """The physically decisive detail.

    At ordinary beam energies the secondary escape depth in gold is about a
    nanometre, so essentially every secondary from a coated spore comes from the
    coating rather than from the biology beneath it.
    """
    s = SC.build("spores", shape=(256, 256), pixel_size_nm=16.0, n_spores=6,
                 coating_nm=coating, coating="gold", seed=0)
    assert s.material_names[1] == expected
    assert s.meta["coating_nm"] == coating


def test_coating_turns_an_invisible_specimen_into_an_obvious_one():
    """Coated and uncoated are two different detection problems, and the model
    should say so rather than the difference being an assertion."""
    kw = dict(shape=(256, 256), pixel_size_nm=16.0, n_spores=10,
              substrate="silicon", seed=3)
    beam = IM.Beam(E0_kev=5.0, pixel_size_nm=16.0, electrons_per_px=600.0)

    ratios = {}
    for tag, coating in (("coated", 10.0), ("uncoated", 0.0)):
        s = SC.build("spores", coating_nm=coating, **kw)
        r = IM.simulate(s.material_index, s.material_names, height_nm=s.height_nm,
                        beam=beam, n_electrons=8000, seed=3)
        spore, substrate = s.material_index == 1, s.material_index == 0
        ratios[tag] = float(r.signal[spore].mean() / r.signal[substrate].mean())

    assert ratios["coated"] > 1.8, ratios
    # uncoated biology on silicon is close to indistinguishable by signal alone
    assert 0.7 < ratios["uncoated"] < 1.3, ratios
    assert ratios["coated"] > ratios["uncoated"] * 1.5, ratios


def test_uncoated_spores_are_found_by_their_rims_not_their_bodies():
    """The steep sides have a much higher secondary yield than the flat tops, so
    an uncoated spore reads as an outline. That emerges from the tilt dependence
    rather than being drawn in."""
    s = SC.build("spores", shape=(256, 256), pixel_size_nm=16.0, n_spores=8,
                 coating_nm=0.0, substrate="silicon", seed=4)
    r = IM.simulate(s.material_index, s.material_names, height_nm=s.height_nm,
                    beam=IM.Beam(E0_kev=2.0, pixel_size_nm=16.0,
                                 electrons_per_px=600.0),
                    n_electrons=8000, seed=4)

    from scipy.ndimage import binary_erosion
    body = s.material_index == 1
    core = binary_erosion(body, iterations=4)
    rim = body & ~core
    assert rim.sum() > 50 and core.sum() > 50
    assert r.signal[rim].mean() > r.signal[core].mean(), "no rim brightening"


def test_clustering_changes_how_spores_are_distributed():
    """Spores dry into clumps rather than scattering evenly.

    Measured as nearest-neighbour spacing between centroids. Counting connected
    regions does not work: the overlap rejection stops neighbours merging, so a
    clumped field and a spread one have much the same component count.
    """
    from scipy.spatial.distance import pdist, squareform
    from skimage.measure import label, regionprops

    def median_nn(clustering):
        s = SC.build("spores", shape=(512, 512), pixel_size_nm=16.0, n_spores=22,
                     clustering=clustering, seed=7)
        cents = np.array([p.centroid for p in regionprops(label(s.material_index == 1))
                          if p.area > 200])
        d = squareform(pdist(cents))
        np.fill_diagonal(d, np.inf)
        return float(np.median(d.min(axis=1)))

    assert median_nn(0.95) < median_nn(0.0)


def test_spores_export_ground_truth_under_their_own_name(tmp_path):
    from acorn_sem_sim.io import generate_sem_dataset

    generate_sem_dataset(tmp_path, 1, {
        "scene": "spores", "image_size_px": 256, "pixel_size_nm": 16.0,
        "n_spores": 10, "n_electrons": 4000, "coating_nm": 10.0,
    }, seed=0)
    ann = json.loads((tmp_path / "images" / "semsim_00000.annotations.json").read_text())
    assert ann["annotations"]
    assert {a["label"] for a in ann["annotations"]} == {"gold"}


def test_spore_scene_records_what_it_actually_placed(tmp_path):
    from acorn_sem_sim.io import generate_sem_dataset

    generate_sem_dataset(tmp_path, 1, {
        "scene": "spores", "image_size_px": 256, "pixel_size_nm": 16.0,
        "n_spores": 200, "n_electrons": 4000,
    }, seed=0)
    rec = json.loads((tmp_path / "metadata.json").read_text())["images"][0]
    sp = rec["scene_params"]
    assert sp["n_spores_requested"] == 200
    assert 0 < sp["n_spores_placed"] < 200, "crowding was not recorded"
