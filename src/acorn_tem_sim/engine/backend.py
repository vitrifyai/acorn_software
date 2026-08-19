"""Array-backend selection: CuPy (GPU) when available, else NumPy.

The FFT-heavy inner loops (multislice propagation, 4D-STEM scan) are written
against a returned array module `xp` so they run on the GPU transparently. Only
the hot loops move to the GPU; specimen geometry is built on the CPU and each
slab is transferred as needed (the FFTs dominate for large grids / many slabs).
"""
from __future__ import annotations

_CUPY = None


def _try_cupy():
    global _CUPY
    if _CUPY is not None:
        return _CUPY or None
    try:
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        # Importing + seeing a device is not enough: on some installs the CUDA
        # runtime clashes (e.g. cupy-cuda12x next to a CUDA-13 toolchain pulled
        # in by torch) and kernels fail to *compile* at first use. Force a tiny
        # kernel + FFT here so a broken GPU falls back to CPU instead of
        # crashing deep inside a simulation.
        float((cp.arange(4, dtype=cp.float32) ** 2).sum())
        cp.fft.fft(cp.arange(4, dtype=cp.complex64))
        _CUPY = cp
        return cp
    except Exception:
        _CUPY = False
        return None


def get_xp(use_gpu=None):
    """Return (array_module, on_gpu). use_gpu: True force GPU, False force CPU,
    None auto (GPU if CuPy+device available)."""
    if use_gpu is False:
        import numpy as np
        return np, False
    cp = _try_cupy()
    if cp is not None:
        return cp, True
    if use_gpu is True:
        raise RuntimeError("GPU requested but CuPy/CUDA is unavailable")
    import numpy as np
    return np, False


def to_cpu(a):
    """Move an array back to NumPy if it is a CuPy array."""
    cp = _CUPY if _CUPY else None
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return a


def gpu_available():
    return _try_cupy() is not None
