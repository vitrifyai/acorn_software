"""
Denoising for low-dose EM.

Two things must hold: a method that cannot run must say so rather than quietly
returning the image unchanged, and whatever is applied must be recorded, because
ACORN denoises before handing an image to a detector and before writing training
tiles.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from acorn.core import denoise as D
from acorn.core.contrast import ContrastParams, apply_contrast


def _noisy(seed=0):
    yy, xx = np.mgrid[0:256, 0:256]
    clean = 0.5 - 0.35 * sum(
        np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 8.0 ** 2)))
        for cy, cx in [(60, 60), (60, 190), (190, 60), (190, 190), (128, 128)])
    return (clean + np.random.RandomState(seed).normal(0, 0.12, clean.shape)).astype("float32")


def _snr(a):
    bg = a[5:50, 5:50]
    return abs(a[118:138, 118:138].mean() - bg.mean()) / (bg.std() + 1e-9)


# ── the catalogue ─────────────────────────────────────────────────────────────

def test_every_method_declares_what_it_needs():
    for m in D.METHODS:
        assert m.key and m.label and m.note
        assert callable(m.fn)


def test_availability_reflects_installed_packages():
    import importlib.util
    for m in D.METHODS:
        expected = all(importlib.util.find_spec(r) is not None for r in m.requires)
        assert D.is_available(m.key) is expected, f"{m.key} misreports availability"


def test_an_unavailable_method_is_named_with_what_to_install(monkeypatch):
    """
    Driven rather than left to the environment: once everything is installed this
    path stops being exercised exactly when it still needs to work, because the
    people who hit it are the ones with a fresh install.
    """
    monkeypatch.setattr(D, "is_available", lambda key: False)
    for m in D.METHODS:
        if not m.requires:
            continue
        note = D.missing_note(m.key)
        assert m.label in note, f"{m.key}: the message does not name the method"
        for r in m.requires:
            assert r in note, f"{m.key}: the message does not name {r}"


def test_an_unavailable_method_returns_the_image_untouched(monkeypatch):
    monkeypatch.setattr(D, "is_available", lambda key: False)
    a = _noisy()
    assert np.array_equal(D.apply_denoise(a, D.DenoiseParams(method="wavelet")), a)


def test_wavelet_and_nlmeans_declare_pywavelets():
    """
    Both reach estimate_sigma, which needs PyWavelets. Undeclared, they reported
    themselves available and silently returned the input — the failure mode this
    module exists to avoid.
    """
    for key in ("wavelet", "nlmeans"):
        m = next(m for m in D.METHODS if m.key == key)
        assert "pywt" in m.requires


# ── behaviour ─────────────────────────────────────────────────────────────────

def test_none_is_the_default_and_changes_nothing():
    a = _noisy()
    assert np.array_equal(D.apply_denoise(a, D.DenoiseParams()), a)


@pytest.mark.parametrize("method", ["gaussian", "median", "butterworth", "tv",
                                    "wavelet", "nlmeans"])
def test_each_method_actually_improves_signal_to_noise(method):
    if not D.is_available(method):
        pytest.skip(f"{method} not installed")
    a = _noisy()
    out = D.apply_denoise(a, D.DenoiseParams(method=method), pixel_size_nm=0.6)
    assert out.shape == a.shape
    assert _snr(out) > _snr(a), f"{method} did not improve SNR"


def test_a_failing_method_returns_the_image_rather_than_raising():
    """A denoiser is an aid; it must not be able to stop an image being shown."""
    a = _noisy()
    assert np.array_equal(D.apply_denoise(a, D.DenoiseParams(method="no-such-method")), a)
    broken = D.DenoiseParams(method="butterworth", resolution_a=-5.0)
    assert D.apply_denoise(a, broken, pixel_size_nm=0.6).shape == a.shape


def test_the_resolution_cutoff_uses_the_pixel_size():
    """Same setting, different calibration, different result — or it is decorative."""
    a = _noisy()
    p = D.DenoiseParams(method="butterworth", resolution_a=20.0)
    fine = D.apply_denoise(a, p, pixel_size_nm=0.2)
    coarse = D.apply_denoise(a, p, pixel_size_nm=2.0)
    assert not np.allclose(fine, coarse), "pixel size had no effect on the cutoff"


# ── it has to travel with the image ───────────────────────────────────────────

def test_denoising_rides_with_the_contrast_settings():
    a = _noisy()
    plain = apply_contrast(a, ContrastParams(method="percentile"))
    cleaned = apply_contrast(
        a, ContrastParams(method="percentile", denoise_method="gaussian"),
        pixel_size_nm=0.6)
    assert _snr(cleaned) > _snr(plain)


def test_what_was_applied_can_be_recorded():
    assert D.DenoiseParams().describe() == "none"
    assert "wavelet" in D.DenoiseParams(method="wavelet", strength=0.7).describe()
    d = D.DenoiseParams(method="butterworth", resolution_a=12.0).describe()
    assert "12.0 A" in d and "butterworth" in d


def test_contrast_params_carry_the_denoise_fields():
    """Detectors and the training exporter both read ContrastParams."""
    p = ContrastParams(method="percentile", denoise_method="wavelet")
    for field in ("denoise_method", "denoise_strength", "denoise_resolution_a"):
        assert hasattr(p, field)
    assert p.denoise_method == "wavelet"


# ── the optional heavyweights ─────────────────────────────────────────────────

@pytest.mark.parametrize("method", ["bm3d", "topaz"])
def test_optional_denoisers_run_when_installed(method):
    """
    Both were written against an API I had not run. Topaz in particular needs the
    model NAME rather than a loaded module — Denoise.__init__ compares with
    `type(model) == torch.nn.Module`, false for any real subclass, and the error
    path then crashes concatenating the model into a string.
    """
    if not D.is_available(method):
        pytest.skip(f"{method} not installed")
    a = _noisy()
    out = D.apply_denoise(a, D.DenoiseParams(method=method))
    assert not np.array_equal(out, a), f"{method} silently returned the input"
    assert out.shape == a.shape
    assert _snr(out) > _snr(a), f"{method} did not improve SNR"


def test_topaz_reuses_its_loaded_model():
    """Loading costs seconds; the panel re-denoises on every parameter change."""
    if not D.is_available("topaz"):
        pytest.skip("topaz not installed")
    import time
    a = _noisy()
    D.apply_denoise(a, D.DenoiseParams(method="topaz"))    # warm
    t0 = time.time()
    D.apply_denoise(a, D.DenoiseParams(method="topaz"))
    assert time.time() - t0 < 5.0, "the model appears to be reloading each call"
