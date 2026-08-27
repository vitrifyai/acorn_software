"""What GPUs this process could use, as distinct from what it can see right now.

CUDA_VISIBLE_DEVICES is process-global and one-way. Once the CUDA driver has
initialised, the visible set is fixed: restoring the variable afterwards does
not give a library back the devices it was started without. So a subsystem that
narrows it for its own reasons narrows it for everything, permanently, and
every later consumer is told the machine has less hardware than it does.

That is not hypothetical. CryoBLOB pins the process to one GPU so JAX does not
create a context on all sixteen -- which is worth doing, it wastes about 5 GB
across a shared DGX. But the pin also reached the batch surface-area analysis,
whose worker count comes from the device count: running CryoBLOB first silently
dropped it from sixteen workers to one, and nothing said so. Run the analysis
first and it used all sixteen. Same session, different answer, decided by the
order the buttons were pressed.

This module keeps the two questions apart:

    visible_device_count()   what CUDA will hand this process now
    physical_device_count()  what the machine actually has

Anything sizing a workload -- how many workers, how many shards -- wants the
second. Anything placing a specific computation wants the first.
"""
from __future__ import annotations

import os
import subprocess

_ENV = "CUDA_VISIBLE_DEVICES"

# The visible set as it stood before anything narrowed it. None means it was
# never recorded, which is itself informative: nothing has pinned anything.
_original_visible: str | None = None
_recorded = False


def remember_visible_devices() -> None:
    """Snapshot CUDA_VISIBLE_DEVICES before a caller narrows it.

    Idempotent, so the first pin wins and a second one cannot overwrite the
    record with an already-narrowed value.
    """
    global _original_visible, _recorded
    if _recorded:
        return
    _original_visible = os.environ.get(_ENV)
    _recorded = True


def pin_visible_devices(device_ids) -> None:
    """Narrow this process to `device_ids`, remembering what it was first.

    Only takes effect before the CUDA driver initialises; afterwards it is a
    no-op as far as already-loaded libraries are concerned, which is exactly why
    the original has to be recorded rather than assumed recoverable.
    """
    remember_visible_devices()
    os.environ[_ENV] = ",".join(str(int(d)) for d in device_ids)


def visible_device_count() -> int:
    """How many GPUs CUDA will hand this process as it currently stands."""
    try:
        import torch
        return int(torch.cuda.device_count())
    except Exception:
        return _count_from_env(os.environ.get(_ENV))


def physical_device_count() -> int:
    """How many GPUs the machine has, ignoring any narrowing done since start.

    Use this to size a workload. The count CUDA reports may have been reduced by
    an unrelated subsystem, and sizing a worker pool from that number is how
    sixteen GPUs quietly become one.
    """
    if _recorded and _original_visible is not None:
        n = _count_from_env(_original_visible)
        if n:
            return n

    # Nothing has pinned anything, or the original was unset (meaning "all").
    # Ask the driver directly rather than a framework, since a framework may
    # already be constrained.
    n = _count_from_smi()
    if n:
        return n
    return visible_device_count()


def _count_from_env(value: str | None) -> int:
    if not value:
        return 0
    return len([p for p in value.split(",") if p.strip() != ""])


def _count_from_smi() -> int:
    """Physical GPU count from nvidia-smi, which CUDA_VISIBLE_DEVICES does not
    filter. Returns 0 when there is no driver or the call fails."""
    try:
        out = subprocess.run(["nvidia-smi", "--list-gpus"],
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return 0
    if out.returncode != 0:
        return 0
    return len([line for line in out.stdout.splitlines() if line.strip()])


def pin_report() -> str:
    """One line describing any narrowing in force, for logs and diagnostics."""
    if not _recorded:
        return ""
    now = visible_device_count()
    physical = physical_device_count()
    if now >= physical:
        return ""
    return (f"This process is pinned to {now} of {physical} GPUs "
            f"({_ENV}={os.environ.get(_ENV)!r}). Workloads are sized from the "
            f"physical count.")
