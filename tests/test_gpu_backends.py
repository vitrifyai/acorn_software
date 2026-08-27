"""GPU paths: available, correct, and equivalent to the CPU ones.

A GPU backend that runs but computes something slightly different is worse than
one that does not run, because nothing complains. These check agreement first
and speed second.

Skipped cleanly on a machine without a GPU.
"""
from __future__ import annotations

import numpy as np
import pytest

cupy = pytest.importorskip("cupy")


def _has_gpu():
    from acorn_tem_sim.engine.backend import gpu_available
    return gpu_available()


pytestmark = pytest.mark.skipif(not _has_gpu(), reason="no working CuPy/CUDA device")


def _reset_backend():
    """Clear the cached probe so a forced choice is honoured."""
    import acorn_tem_sim.engine.backend as backend
    backend._CUPY = None
    return backend


def test_backend_selection_honours_an_explicit_choice():
    backend = _reset_backend()
    assert backend.get_xp(False)[0].__name__ == "numpy"
    _reset_backend()
    xp, on_gpu = backend.get_xp(True)
    assert on_gpu and xp.__name__ == "cupy"


def test_arrays_come_back_to_the_cpu():
    backend = _reset_backend()
    xp, _ = backend.get_xp(True)
    out = backend.to_cpu(xp.arange(5, dtype=xp.float32))
    assert isinstance(out, np.ndarray)
    assert out.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def _simulate(use_gpu, size=256):
    backend = _reset_backend()
    xp, on_gpu = backend.get_xp(use_gpu)
    backend._CUPY = xp if on_gpu else False

    from acorn_tem_sim.engine.scene import Nanoparticles, Scene, simulate_scene
    from acorn_tem_sim.engine.setup import resolve

    cfg = resolve({"pixel_size_a": 4.0, "image_size_px": size,
                   "total_dose_e_per_a2": 40.0, "microscope": "krios",
                   "detector_model": "K3", "defocus_min_um": -2.0,
                   "defocus_max_um": -2.0})
    cfg["use_gpu"] = bool(use_gpu)
    scene = Scene(ice_thickness_nm=60.0,
                  components=[Nanoparticles(n=20, diameter_nm_mean=28.0,
                                            diameter_nm_sd=5.0)])
    ideal, _ = simulate_scene(cfg, scene, defocus_um=-2.0, dz_a=40.0)
    return np.asarray(backend.to_cpu(ideal), dtype=np.float64)


def test_multislice_agrees_between_gpu_and_cpu():
    """The result must not depend on where it was computed."""
    cpu, gpu = _simulate(False), _simulate(True)
    assert cpu.shape == gpu.shape
    rel = np.abs(cpu - gpu).max() / max(np.abs(cpu).max(), 1e-12)
    assert rel < 1e-4, f"GPU and CPU multislice differ by {rel:.2e}"


def _stem(use_gpu, scan=12, n=96):
    backend = _reset_backend()
    from acorn_tem_sim.engine.scene import Nanoparticles, Scene, scene_slabs
    from acorn_tem_sim.engine.setup import resolve
    from acorn_tem_sim.engine.stem import stem_4d, virtual_detectors

    cfg = resolve({"pixel_size_a": 1.0, "image_size_px": n,
                   "total_dose_e_per_a2": 40.0, "microscope": "krios",
                   "detector_model": "K3"})
    scene = Scene(ice_thickness_nm=20.0,
                  components=[Nanoparticles(n=6, diameter_nm_mean=6.0,
                                            diameter_nm_sd=1.0)])
    slabs = list(scene_slabs((n, n), 1.0, 40.0, scene))
    fourd = stem_4d(cfg, slabs, conv_mrad=20.0, scan_shape=(scan, scan),
                    dz_a=40.0, seed=0, use_gpu=use_gpu)
    return {k: np.asarray(backend.to_cpu(v), dtype=np.float64)
            for k, v in virtual_detectors(fourd).items()}


def test_every_4dstem_channel_agrees_between_gpu_and_cpu():
    cpu, gpu = _stem(False), _stem(True)
    assert set(cpu) == set(gpu)
    for channel in sorted(cpu):
        scale = max(np.abs(cpu[channel]).max(), 1e-12)
        rel = np.abs(cpu[channel] - gpu[channel]).max() / scale
        assert rel < 1e-4, f"{channel} differs by {rel:.2e}"


def test_cryoblob_jax_reaches_the_gpu():
    """CryoBLOB patches the published package at import; the patch has to hold."""
    pytest.importorskip("jax")
    pytest.importorskip("cryoblob")
    from acorn_cryoblob.thread import _install_cryoblob_gpu_backend

    _install_cryoblob_gpu_backend()
    import jax
    assert jax.default_backend() == "gpu"


def test_cryoblob_detects_on_a_full_size_micrograph():
    """4092 x 5760 is the size that crashes the published cryoblob; the patch
    exists so this works."""
    pytest.importorskip("jax")
    pytest.importorskip("cryoblob")
    from acorn_cryoblob.thread import _detect_blobs_jax, _install_cryoblob_gpu_backend

    _install_cryoblob_gpu_backend()
    rng = np.random.default_rng(0)
    image = rng.normal(100, 5, (2048, 2048)).astype(np.float32)
    yy, xx = np.mgrid[:2048, :2048]
    for cy, cx in ((500, 600), (1200, 1500), (1700, 400)):
        image[((yy - cy) ** 2 + (xx - cx) ** 2) < 40 ** 2] -= 30

    rows = _detect_blobs_jax(image, blob_downscale=4.0, min_sigma=3.0,
                             max_sigma=20.0, threshold_rel=0.1, max_detections=50)
    assert rows is not None and len(rows) > 0
