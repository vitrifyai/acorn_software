"""What the machine has, versus what this process can currently see.

CUDA_VISIBLE_DEVICES is process-global and one-way: once the driver initialises
the visible set is fixed, and restoring the variable gives nothing back. So one
subsystem narrowing it narrows it for every later consumer, and anything sizing
a workload from the reported count is told the machine is smaller than it is.

That happened. CryoBLOB pins to one GPU so JAX does not create a context on all
sixteen, and that pin reached the batch surface-area worker pool: run CryoBLOB
first and sixteen workers became one, silently, decided by button order.
"""
from __future__ import annotations

import pytest

from acorn.core import gpu


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """Each test gets a clean record; the module state is process-wide."""
    monkeypatch.setattr(gpu, "_original_visible", None, raising=False)
    monkeypatch.setattr(gpu, "_recorded", False, raising=False)
    yield


def test_counting_a_device_list():
    assert gpu._count_from_env("0,1,2,3") == 4
    assert gpu._count_from_env("0") == 1
    assert gpu._count_from_env("") == 0
    assert gpu._count_from_env(None) == 0


def test_remembering_is_idempotent_so_the_first_pin_wins(monkeypatch):
    """A second pin must not overwrite the record with an already-narrowed
    value -- that would launder the narrowing into the 'original'."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    gpu.remember_visible_devices()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    gpu.remember_visible_devices()
    assert gpu._original_visible == "0,1,2,3"


def test_physical_count_survives_a_pin(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    gpu.pin_visible_devices([0])
    assert monkeypatch is not None
    assert gpu._count_from_env(gpu._original_visible) == 8
    assert gpu.physical_device_count() == 8


def test_pinning_narrows_the_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    gpu.pin_visible_devices([2])
    import os
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"


def test_pin_report_is_silent_when_nothing_is_pinned():
    assert gpu.pin_report() == ""


def test_pin_report_names_the_narrowing(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
    gpu.pin_visible_devices([0])
    monkeypatch.setattr(gpu, "visible_device_count", lambda: 1)
    report = gpu.pin_report()
    assert "1 of 4" in report and "sized from the physical count" in report


def test_worker_pool_is_sized_from_physical_not_visible(monkeypatch):
    """The regression itself: a pinned process must still spread work over
    every GPU the machine has, because each worker sets its own visibility."""
    pytest.importorskip("torch")
    from acorn.analysis import surface_area

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    gpu.remember_visible_devices()
    gpu.pin_visible_devices([0])                       # as CryoBLOB does
    monkeypatch.setattr(gpu, "visible_device_count", lambda: 1)

    assert len(surface_area._available_gpu_ids()) == 8


def test_worker_pool_respects_an_explicit_cap(monkeypatch):
    pytest.importorskip("torch")
    from acorn.analysis import surface_area

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    gpu.remember_visible_devices()
    assert len(surface_area._available_gpu_ids(n_gpus=3)) == 3


def test_no_gpu_means_cpu_rather_than_a_crash(monkeypatch):
    from acorn.analysis import surface_area

    monkeypatch.setattr(gpu, "physical_device_count", lambda: 0)
    assert surface_area._available_gpu_ids() == []
