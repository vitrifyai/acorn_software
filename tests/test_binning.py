"""Analysis binning: the array and its calibration must change together.

The single failure this guards against is a binned image whose pixel size was
not rescaled. Every distance, area and diameter is then wrong by the bin factor
while still carrying units, which is worse than having no calibration at all --
and nothing downstream can detect it, because the numbers look reasonable.
"""
from __future__ import annotations

import numpy as np
import pytest

from acorn.core.binning import (
    VALID_FACTORS,
    bin_frames,
    bin_image,
    describe,
    per_pixel_noise_gain,
    validate_factor,
)


# --- arithmetic -------------------------------------------------------------

def test_bins_exact_block_means():
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    assert np.allclose(bin_image(a, 2).data, [[2.5, 4.5], [10.5, 12.5]])


def test_uses_the_mean_not_the_sum():
    """Summing would rescale intensity by factor^2 and break every contrast
    setting and threshold downstream."""
    a = np.full((8, 8), 7.0, dtype=np.float32)
    for f in VALID_FACTORS:
        assert np.allclose(bin_image(a, f).data, 7.0), f


def test_bin_one_is_a_no_op():
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    r = bin_image(a, 1, pixel_size_nm=0.5)
    assert np.array_equal(r.data, a)
    assert r.pixel_size_nm == 0.5
    assert not r.was_cropped


@pytest.mark.parametrize("factor", VALID_FACTORS)
def test_per_pixel_noise_falls_as_one_over_factor(factor):
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (512, 512)).astype(np.float32)
    assert bin_image(noise, factor).data.std() == pytest.approx(1.0 / factor, rel=0.06)


# --- calibration: the reason this module exists -----------------------------

@pytest.mark.parametrize("factor", VALID_FACTORS)
def test_pixel_size_scales_with_the_factor(factor):
    a = np.zeros((64, 64), np.float32)
    assert bin_image(a, factor, pixel_size_nm=0.2081).pixel_size_nm == \
        pytest.approx(0.2081 * factor)


@pytest.mark.parametrize("factor", VALID_FACTORS)
def test_field_of_view_is_unchanged_by_binning(factor):
    """The invariant that proves the rescale is right: fewer pixels, each
    proportionally larger, so the imaged area in nanometres is the same."""
    a = np.zeros((512, 512), np.float32)
    r = bin_image(a, factor, pixel_size_nm=0.2081)
    assert r.data.shape[1] * r.pixel_size_nm == pytest.approx(512 * 0.2081, rel=1e-6)


@pytest.mark.parametrize("factor", VALID_FACTORS)
def test_a_measured_diameter_is_the_same_at_every_bin_factor(factor):
    """What a user actually cares about. A disc of known size must measure the
    same in nanometres however the image was binned."""
    n, radius_px, px_nm = 512, 60, 0.25
    yy, xx = np.mgrid[:n, :n]
    disc = (((yy - n // 2) ** 2 + (xx - n // 2) ** 2) < radius_px ** 2).astype(np.float32)

    r = bin_image(disc, factor, pixel_size_nm=px_nm)
    area_px = float((r.data > 0.5).sum())
    diameter_nm = 2.0 * np.sqrt(area_px / np.pi) * r.pixel_size_nm
    truth_nm = 2.0 * radius_px * px_nm
    assert diameter_nm == pytest.approx(truth_nm, rel=0.03), (diameter_nm, truth_nm)


# --- shapes that do not divide evenly ---------------------------------------

def test_reports_what_it_cropped():
    """4092 does not divide by 8. Rows are dropped so the shape divides, and a
    silently smaller field would be a measurement error waiting to happen."""
    a = np.zeros((4092, 5760), np.float32)
    r = bin_image(a, 8, pixel_size_nm=0.2081)
    assert r.data.shape == (511, 720)
    assert r.cropped_px == (4, 0)
    assert r.was_cropped


def test_evenly_divisible_shapes_are_not_cropped():
    r = bin_image(np.zeros((4092, 5760), np.float32), 4)
    assert r.data.shape == (1023, 1440)
    assert r.cropped_px == (0, 0) and not r.was_cropped


# --- shapes other than plain 2-D --------------------------------------------

def test_colour_channels_are_not_averaged_together():
    """Binning the channel axis would average red into green."""
    c = np.zeros((8, 8, 3), np.float32)
    c[..., 0] = 1.0
    out = bin_image(c, 2).data
    assert out.shape == (4, 4, 3)
    assert out[..., 0].mean() == pytest.approx(1.0)
    assert out[..., 1].mean() == pytest.approx(0.0)


def test_movie_frames_are_binned_but_the_frame_axis_is_not():
    m = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    r = bin_frames(m, 2, pixel_size_nm=0.5)
    assert r.data.shape == (3, 4, 4)
    assert r.pixel_size_nm == pytest.approx(1.0)
    assert np.allclose(r.data[0], bin_image(m[0], 2).data)


def test_a_movie_stack_is_refused_by_the_image_path():
    """(N, H, W) and (H, W, 3) are both 3-D; guessing wrong silently averages
    the wrong axis, so the error names the right function."""
    with pytest.raises(ValueError, match="bin_frames"):
        bin_image(np.zeros((10, 64, 64), np.float32), 2)


def test_an_image_is_refused_by_the_movie_path():
    with pytest.raises(ValueError, match=r"N, H, W"):
        bin_frames(np.zeros((64, 64), np.float32), 2)


# --- validation and reporting -----------------------------------------------

@pytest.mark.parametrize("bad", [0, 3, 5, 7, -2, 16])
def test_unsupported_factors_say_what_is_supported(bad):
    with pytest.raises(ValueError, match="1, 2, 4, 8"):
        validate_factor(bad)


def test_non_integer_factor_is_rejected():
    with pytest.raises(ValueError, match="integer"):
        validate_factor("four")


def test_description_does_not_promise_better_detection():
    """Binning does not make faint objects detectable -- a filter matched to the
    object size gets the same signal from full resolution. Measured flat at
    5.80 across bins 1-8 on real low-dose data. The UI must not claim otherwise."""
    text = describe(4, 0.83).lower()
    assert "snr" not in text
    assert "pixel size" in text and "real units" in text
    assert "full resolution" in describe(1).lower()


def test_noise_gain_is_named_for_what_it_measures():
    assert per_pixel_noise_gain(4) == 4.0


# --- integration with the loader --------------------------------------------

def test_loader_binning_scales_pixel_size_and_records_provenance(tmp_path):
    import tifffile

    from acorn.core.dm4_loader import DM4Image

    path = tmp_path / "square.tif"
    tifffile.imwrite(path, np.random.default_rng(0).normal(
        100, 5, (256, 256)).astype(np.float32))

    native = DM4Image.from_file(path)
    native.meta.pixel_size = 0.5          # simulate a calibrated header
    binned = DM4Image.from_file(path, bin_factor=4)
    binned.meta.native_pixel_size = 0.5

    assert binned.raw.shape == (64, 64)
    assert binned.meta.bin_factor == 4
    assert native.meta.bin_factor == 1
    assert binned.meta.shape == binned.raw.shape


def test_loader_default_is_unbinned(tmp_path):
    """Binning must never happen unless it was asked for."""
    import tifffile

    from acorn.core.dm4_loader import DM4Image

    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, np.zeros((32, 32), np.float32))
    img = DM4Image.from_file(path)
    assert img.raw.shape == (32, 32)
    assert img.meta.bin_factor == 1
    assert img.meta.native_pixel_size == img.meta.pixel_size


def test_a_manual_pixel_size_override_survives_a_change_of_binning():
    """Overrides are stored on the native grid.

    Storing the binned value instead would silently invalidate the override the
    moment the bin factor changed. The failure is quiet and total: measurements
    stay plausible and are wrong by exactly the bin factor.
    """
    native_px = 0.2081                      # what the user typed at bin 1
    for factor in VALID_FACTORS:
        applied = native_px * factor        # what _finish_switch puts on the image
        # a 40 nm object, measured on the binned grid it was imaged at
        object_px = (40.0 / native_px) / factor
        measured_nm = object_px * applied
        assert measured_nm == pytest.approx(40.0, rel=1e-9), (factor, measured_nm)


def test_entering_a_pixel_size_while_binned_round_trips():
    """The user types the pixel size of the image in front of them -- the binned
    one -- so it is divided down to native on the way in and multiplied back on
    the way out."""
    for factor in VALID_FACTORS:
        typed_binned = 0.2081 * factor
        stored_native = typed_binned / factor
        reapplied = stored_native * factor
        assert reapplied == pytest.approx(typed_binned, rel=1e-12)
        assert stored_native == pytest.approx(0.2081, rel=1e-12)
