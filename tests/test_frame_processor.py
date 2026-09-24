import numpy as np
from scipy.ndimage import shift as nd_shift

from acorn.core.frame_processor import (
    align_frames,
    motion_correct_frames,
    motion_corrected_dose_series,
)


def _translated_stack():
    rng = np.random.default_rng(7)
    reference = rng.normal(size=(96, 96)).astype(np.float32)
    reference[28:52, 35:61] += 4.0
    applied = np.array([[0.0, 0.0], [1.5, -2.0], [-2.0, 1.0], [3.0, 2.5]])
    frames = np.stack([nd_shift(reference, s, order=3) for s in applied])
    return reference, frames.astype(np.float32), applied


def test_align_frames_returns_corrected_stack_and_shifts():
    reference, frames, applied = _translated_stack()
    aligned, shifts = align_frames(frames)

    assert aligned.shape == frames.shape
    assert shifts.shape == applied.shape
    assert np.allclose(shifts[1:], -applied[1:], atol=0.25)
    corrected_error = np.mean((aligned[1:] - reference) ** 2)
    uncorrected_error = np.mean((frames[1:] - reference) ** 2)
    assert corrected_error < 0.1 * uncorrected_error


def test_existing_motion_correct_frames_api_is_preserved():
    _, frames, _ = _translated_stack()
    averaged, shifts = motion_correct_frames(frames)

    assert averaged.shape == frames.shape[1:]
    assert shifts.shape == (len(frames), 2)


def test_motion_corrected_dose_series_uses_aligned_frames():
    reference, frames, _ = _translated_stack()
    averages, ranges, shifts, aligned = motion_corrected_dose_series(frames, 2)

    assert ranges == [(0, 2), (2, 4)]
    assert len(averages) == 2
    assert aligned.shape == frames.shape
    assert shifts.shape == (len(frames), 2)
    assert np.mean((averages[1] - reference) ** 2) < 0.1
