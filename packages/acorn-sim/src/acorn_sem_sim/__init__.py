"""Physics-based SEM image simulation for ACORN.

Unlike the cryo-TEM engine there is no coherent transmitted wave here, so there
is no CTF: the resolution limit is the interaction volume, and it is obtained by
tracing electrons. See `transport` for the Monte Carlo and `materials` for what
is measured versus what is calibrated.

Imports are deferred so that merely loading the plugin does not pull scipy in.
"""
from __future__ import annotations

__all__ = ["imaging", "kernels", "materials", "transport"]


def __getattr__(name):
    if name in __all__:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
