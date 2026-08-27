"""Focused cryo-TEM simulation plugin.

Imports are deferred. Eagerly importing `simulator` here pulled scipy.ndimage
into the process the moment the plugin was touched -- 0.24 s that ACORN does not
otherwise pay, charged to the first switch into the Simulate workspace, before
anything had been asked of the simulator.
"""
from __future__ import annotations

__all__ = ["Micrograph", "Specimen", "resolve", "simulate_micrograph"]

_EXPORTS = {
    "Micrograph":         "acorn_tem_sim.simulator",
    "Specimen":           "acorn_tem_sim.simulator",
    "resolve":            "acorn_tem_sim.simulator",
    "simulate_micrograph": "acorn_tem_sim.simulator",
}


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module), name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
