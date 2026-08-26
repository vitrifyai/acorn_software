"""Helpers shared by the TEM, SEM and FIB simulation plugins.

These three plugins do the same three things around their very different
physics: pick a fresh output directory, hand the results to the viewer, and
coerce whatever CLU passed for a boolean. Those helpers had been copied
verbatim into all three, which is how three copies drift into three behaviours.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PyQt6.QtCore import QTimer

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext

__all__ = ["as_bool", "fresh_run_dir", "open_paths_in_acorn"]


def as_bool(value: Any) -> bool:
    """Coerce a CLU-supplied value to a boolean.

    A model may send a real boolean or the string "false"; plain bool() reads
    the latter as True, which silently inverts the setting.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)


def fresh_run_dir(base_dir: Path, prefix: str) -> Path:
    """A timestamped directory under `base_dir` that does not already exist.

    Generation runs are never allowed to overwrite each other: a second run with
    the same settings is usually a deliberate repeat, and silently replacing the
    first would destroy data the user still wanted.
    """
    base_dir = Path(base_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = base_dir / f"{prefix}_{stamp}"
    counter = 1
    while run_dir.exists():
        run_dir = base_dir / f"{prefix}_{stamp}_{counter}"
        counter += 1
    return run_dir


def open_paths_in_acorn(context: "AcornContext", paths: list[str]) -> tuple[int, str]:
    """Open generated images in the viewer. Returns (count, message).

    Returns a reason rather than raising when the window is unavailable -- a
    dataset that generated successfully but could not be displayed is a
    reporting problem, not a failed run, and the caller should say so without
    the run looking like it failed.

    The wording is asserted by the TEM and FIB tests. That is deliberate: the
    three copies this replaces had already drifted apart in exactly this string.
    """
    real_paths = [Path(p) for p in paths if p and Path(p).exists()]
    if not real_paths:
        return 0, "Generated images, but no output TIFF files were found to open."
    window_getter = getattr(context, "_w", None)
    if window_getter is None:
        return 0, "Generated images, but the Acorn window is unavailable for auto-open."
    window = window_getter()
    if window is None or not hasattr(window, "open_files"):
        return 0, "Generated images, but the Acorn viewer is unavailable for auto-open."
    # Deferred: opening files re-enters the event loop, and the generation
    # thread's finished handler is still on the stack at this point.
    QTimer.singleShot(0, lambda p=real_paths: window.open_files(p))
    return len(real_paths), f"Opening {len(real_paths)} generated image(s)."
