"""Focused FIB-SEM surface simulation plugin.

Imports are deferred, for the same reason as the cryo-TEM plugin: eagerly
importing `simulator` pulled scipy.ndimage the moment the plugin was touched,
charging ~0.23 s to the first switch into the Simulate workspace before anything
had been asked of the simulator. Making only the TEM plugin lazy just moved that
cost here.
"""
from __future__ import annotations

__all__ = [
    "ImagingConfig",
    "MillingConfig",
    "SceneConfig",
    "SimConfig",
    "SimResult",
    "simulate",
]

_SIMULATOR = "acorn_fib_sim.simulator"


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(_SIMULATOR), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
