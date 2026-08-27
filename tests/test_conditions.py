"""Nothing may alter a result silently, and nothing may alter it wrongly.

Two guarantees, both driven from the single declaration in acorn.core.conditions:

  1. Anything in force is NAMED. A setting that changes a result and does not
     appear in the summary is the failure this whole module exists to prevent.

  2. Anything claiming to affect a MEASUREMENT has an invariant proving a known
     object still measures correctly under it. A condition declared without that
     evidence fails here, so adding a setting means demonstrating it does not
     corrupt a number rather than asserting it.

The second is the one with teeth: it turns "we checked binning" into "every
measurement-affecting setting is checked, and the build knows which".
"""
from __future__ import annotations

import numpy as np
import pytest

from acorn.core.conditions import (
    CONDITIONS,
    ActiveCondition,
    Affects,
    MeasurementClaim,
    evaluate,
    explain,
    get,
    summarise,
)

# --- the declaration itself --------------------------------------------------

def test_condition_keys_are_unique():
    keys = [c.key for c in CONDITIONS]
    assert len(keys) == len(set(keys))


def test_every_condition_says_what_it_affects_and_why():
    for c in CONDITIONS:
        assert c.affects, f"{c.key} declares no effect"
        assert all(isinstance(a, Affects) for a in c.affects), c.key
        assert len(c.note) > 40, f"{c.key} has no usable explanation"
        assert c.label and not c.label.endswith("."), c.key


def test_an_unknown_condition_names_the_declared_ones():
    with pytest.raises(KeyError, match="bin_factor"):
        get("nonsense")


# --- nothing in force silently ----------------------------------------------

def test_defaults_report_nothing_in_force():
    assert evaluate() == []
    assert summarise([]) == ""
    assert "Nothing is altering results" in explain([])


@pytest.mark.parametrize("kwargs,expected_key", [
    ({"bin_factor": 4}, "bin_factor"),
    ({"pixel_size_override": 0.5}, "pixel_size_override"),
    ({"denoise_method": "nlmeans", "denoise_strength": 0.5}, "denoise"),
    ({"crop_region": (0, 0, 10, 10)}, "crop_region"),
    ({"exclude_zones": [(0, 0, 5, 5)]}, "exclude_zones"),
    ({"contrast_method": "bandpass"}, "contrast"),
    ({"charging": 1.0}, "charging"),
])
def test_each_setting_is_reported_when_it_is_on(kwargs, expected_key):
    active = evaluate(**kwargs)
    assert [a.condition.key for a in active] == [expected_key]
    assert summarise(active), "in force but not named"
    assert active[0].detail, "named but with no value"


@pytest.mark.parametrize("kwargs", [
    {"bin_factor": 1},
    {"pixel_size_override": None},
    {"pixel_size_override": 0.0},
    {"denoise_method": "none"},
    {"denoise_method": ""},
    {"crop_region": None},
    {"exclude_zones": []},
    {"contrast_method": "percentile"},
    {"charging": 0.0},
])
def test_a_setting_at_its_default_is_not_reported(kwargs):
    """Reporting defaults as "in force" is how a warning becomes noise, and then
    the one that matters gets ignored with the rest."""
    assert evaluate(**kwargs) == []


# --- not crying wolf --------------------------------------------------------

def test_a_calibration_matching_the_file_is_not_called_a_manual_override():
    """A sidecar restoring the header's own pixel size is not an override.

    Reported as one, the indicator lit on every previously-annotated image and
    the word "manual" was simply false.
    """
    assert evaluate(pixel_size_override=0.2081, header_pixel_size=0.2081) == []


def test_a_pixel_size_that_disagrees_with_the_file_is_reported_with_both():
    active = evaluate(pixel_size_override=0.3, header_pixel_size=0.2081)
    assert len(active) == 1
    detail = active[0].detail
    assert "0.3" in detail and "0.2081" in detail, detail
    assert "file says" in detail


def test_a_pixel_size_with_no_header_to_compare_against_is_still_reported():
    """No calibration in the file means the measurement rests on a value from
    somewhere else, which is worth knowing."""
    assert [a.condition.key for a in evaluate(pixel_size_override=0.5)] == \
        ["pixel_size_override"]


def test_the_default_contrast_for_this_file_type_is_not_reported():
    """Electron-microscopy formats default to bandpass. Flagging that would
    light the indicator on every image ever opened, and an indicator that is
    always lit is one nobody reads."""
    assert evaluate(contrast_method="bandpass", default_contrast="bandpass") == []
    assert evaluate(contrast_method="percentile", default_contrast="percentile") == []


def test_a_contrast_method_away_from_this_file_type_default_is_reported():
    active = evaluate(contrast_method="fourier", default_contrast="bandpass")
    assert [a.condition.key for a in active] == ["contrast"]


def test_measurement_altering_conditions_are_listed_first():
    active = evaluate(bin_factor=4, contrast_method="bandpass",
                      pixel_size_override=0.5)
    keys = [a.condition.key for a in active]
    assert keys[0] == "pixel_size_override", keys
    assert keys[-1] == "contrast", keys


def test_the_summary_leads_with_measurements_when_numbers_are_affected():
    assert summarise(evaluate(pixel_size_override=0.5)).startswith(
        "Affecting measurements")
    assert summarise(evaluate(contrast_method="bandpass")).startswith("In force")


def test_the_explanation_covers_every_active_condition():
    active = evaluate(bin_factor=4, denoise_method="tv", denoise_strength=0.3)
    text = explain(active)
    for a in active:
        assert a.condition.label in text
        assert a.condition.note[:40] in text


# --- evidence that measurement-affecting settings do not corrupt numbers -----

def _disc(size=512, radius_px=60):
    yy, xx = np.mgrid[:size, :size]
    return (((yy - size // 2) ** 2 + (xx - size // 2) ** 2)
            < radius_px ** 2).astype(np.float32)


def _measure_diameter_nm(mask, px_nm):
    area_px = float((mask > 0.5).sum())
    return 2.0 * np.sqrt(area_px / np.pi) * px_nm


def test_binning_does_not_change_a_measured_diameter():
    """Evidence for CONDITIONS['bin_factor']."""
    from acorn.core.binning import VALID_FACTORS, bin_image

    px_nm, radius_px = 0.25, 60
    truth = 2 * radius_px * px_nm
    for factor in VALID_FACTORS:
        r = bin_image(_disc(radius_px=radius_px), factor, px_nm)
        got = _measure_diameter_nm(r.data, r.pixel_size_nm)
        assert got == pytest.approx(truth, rel=0.03), (factor, got, truth)


def test_a_manual_pixel_size_is_the_only_thing_that_scales_a_measurement():
    """Evidence for CONDITIONS['pixel_size_override'].

    An override must scale a measurement exactly and linearly -- that is its
    entire job. Anything else changing the number alongside it would be a bug.
    """
    mask = _disc(radius_px=60)
    base = _measure_diameter_nm(mask, 0.25)
    for factor in (0.5, 2.0, 10.0):
        assert _measure_diameter_nm(mask, 0.25 * factor) == pytest.approx(
            base * factor, rel=1e-9)


def test_denoising_biases_a_measured_size_and_that_is_why_it_is_declared():
    """Evidence for CONDITIONS['denoise'].

    Not an invariant -- denoising genuinely does move boundaries, which is
    exactly why it must be reported rather than silently applied. This pins the
    direction of the bias so the warning stays true.
    """
    from acorn.core.denoise import DenoiseParams, apply_denoise

    rng = np.random.default_rng(0)
    mask = _disc(radius_px=60)
    noisy = mask + rng.normal(0, 0.35, mask.shape).astype(np.float32)

    clean = apply_denoise(noisy, DenoiseParams(method="gaussian", strength=0.8),
                          pixel_size_nm=0.25)
    assert not np.allclose(clean, noisy), "denoiser did nothing"

    truth_px = float((mask > 0.5).sum())
    denoised_px = float((clean > 0.5).sum())
    assert denoised_px != pytest.approx(truth_px, rel=1e-6), (
        "denoising left the area untouched, so the declared caveat is wrong")


# --- the guarantee that keeps this honest -----------------------------------

_EVIDENCE = {
    "bin_factor": "test_binning_does_not_change_a_measured_diameter",
    "pixel_size_override": "test_a_manual_pixel_size_is_the_only_thing_that_scales_a_measurement",
    "denoise": "test_denoising_biases_a_measured_size_and_that_is_why_it_is_declared",
    "charging": "test_charging_moves_a_measured_size_and_an_object_count",
}


def test_every_condition_with_a_measurement_claim_has_evidence_in_this_file():
    """A claim about numbers obliges you to demonstrate it.

    Both directions need proving. ALTERS must show what it does to a
    measurement; INVARIANT must show that it does nothing to one -- and that is
    the easier claim to get wrong, because a broken invariant still produces
    plausible numbers. A new condition fails here until its test is written and
    named below.
    """
    import pathlib

    source = pathlib.Path(__file__).read_text()
    for c in CONDITIONS:
        if not c.needs_evidence:
            continue
        name = _EVIDENCE.get(c.key)
        assert name, (
            f"condition {c.key!r} claims '{c.claim.value}' about measurements "
            f"but has no evidence test; add one and register it in _EVIDENCE")
        assert f"def {name}(" in source, f"{name} is registered but not defined"


def test_evidence_registry_has_no_stale_entries():
    """The reverse: an entry left behind after a condition was removed."""
    declared = {c.key for c in CONDITIONS if c.needs_evidence}
    assert set(_EVIDENCE) <= declared, sorted(set(_EVIDENCE) - declared)


def test_binning_claims_invariance_rather_than_no_relationship():
    """The claim itself is worth pinning. Downgrading binning to UNRELATED would
    quietly remove the obligation to prove a measurement survives it."""
    assert get("bin_factor").claim is MeasurementClaim.INVARIANT
    assert get("pixel_size_override").claim is MeasurementClaim.ALTERS
    assert get("contrast").claim is MeasurementClaim.UNRELATED


def test_active_condition_reports_whether_it_touches_numbers():
    assert ActiveCondition(get("pixel_size_override"), "x").alters_numbers
    assert not ActiveCondition(get("contrast"), "x").alters_numbers


# --- the status-bar pixel size must say where its number came from -----------

def test_the_pixel_size_button_names_the_binning_that_produced_it():
    """The number was always right; the attribution was not.

    Under 4x binning the button read "0.8324 nm/px (header)" while the header
    actually says 0.2081. Anyone checking provenance would have been told the
    file said something it does not.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from acorn.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    try:
        window = MainWindow()
        window._img_idx = 0

        window._bin_factor = 1
        window._update_px_btn(0.2081, from_header=True)
        native = window._px_btn.text()
        assert "0.2081" in native and "header" in native
        assert "binned" not in native

        window._bin_factor = 4
        window._update_px_btn(0.8324, from_header=True)
        binned = window._px_btn.text()
        assert "0.8324" in binned
        assert "4x binned" in binned, binned
        # and the file's own grid stays recoverable
        assert "0.2081" in window._px_btn.toolTip()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_an_uncalibrated_image_says_so_rather_than_showing_a_number():
    """Measurements are meaningless until it is set, so this must not look
    like a valid calibration."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication

    from acorn.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    try:
        window = MainWindow()
        window._img_idx = 0
        window._px_overrides.clear()
        window._bin_factor = 1
        window._update_px_btn(1.0, from_header=False)
        assert "not set" in window._px_btn.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_charging_moves_a_measured_size_and_an_object_count():
    """Evidence for the ALTERS claim on simulated charging.

    Worth stating precisely, because the effect is not the one you would guess.
    Flaring does not merely bloat objects -- it fills the dark ridged interiors
    of an uncoated spore, which an intensity threshold would otherwise split
    into fragments. So the object count moves TOWARD the truth while the areas
    inflate. A detector tuned against charged simulations can therefore look
    better than it is, which is precisely why this is declared rather than left
    for someone to discover in their results.
    """
    from acorn_sem_sim import scenes as SC
    from acorn_sem_sim.imaging import Beam, Detector, simulate
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    scene = SC.build("spores", shape=(384, 384), pixel_size_nm=20.0, n_spores=30,
                     coating_nm=0.0, substrate="resin", clustering=0.4, seed=11)
    kw = dict(material_names=scene.material_names, height_nm=scene.height_nm,
              beam=Beam(E0_kev=2.0, pixel_size_nm=20.0, electrons_per_px=800),
              n_electrons=8000, seed=11)

    def measure(charging):
        img = simulate(scene.material_index,
                       detector=Detector("TLD", charging=charging), **kw).signal
        lab, n = ndimage.label(img > threshold_otsu(img))
        areas = np.array(ndimage.sum(np.ones_like(lab), lab, range(1, n + 1)))
        areas = areas[areas > 30]
        return len(areas), float(areas.mean())

    n_clean, area_clean = measure(0.0)
    n_charged, area_charged = measure(1.2)

    assert area_charged > area_clean * 1.4, (
        f"charging barely moved measured area ({area_clean:.0f} -> "
        f"{area_charged:.0f} px); the ALTERS claim would be overstated")
    assert n_charged != n_clean, (
        f"charging left the object count at {n_clean}; it is declared as "
        "affecting detection, so it must actually affect it")
