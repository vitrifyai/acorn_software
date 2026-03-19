"""Headless image export — no Qt imports, uses matplotlib Agg backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from acorn.core.dm4_loader import DM4Image
from acorn.core.contrast import ContrastParams
from acorn.core.annotations import AnnotationStore, ScalebarAnnotation
from acorn.render.canvas import CryoCanvas
from acorn.render.scalebar import nice_scalebar_nm


def export_image(
    dm4img: DM4Image,
    output_path: str | Path,
    params: Optional[ContrastParams] = None,
    annotations: Optional[AnnotationStore] = None,
    dpi: int = 300,
    fmt: Optional[str] = None,
    add_scalebar: bool = True,
    scalebar_color: str = "#FFFFFF",
    figsize: tuple[float, float] = (8, 8),
) -> Path:
    """
    Export a DM4 image to file with contrast applied and optional annotations.

    Parameters
    ----------
    dm4img        : loaded DM4Image
    output_path   : destination file path (extension → format)
    params        : contrast parameters (defaults to bandpass if None)
    annotations   : annotation store to render (empty if None)
    dpi           : resolution (300–600 for publication)
    fmt           : matplotlib format override ("png", "svg", "tiff", etc.)
    add_scalebar  : automatically add a calibrated scale bar
    scalebar_color: scale bar and label colour
    figsize       : figure size in inches

    Returns
    -------
    Resolved Path of the saved file.
    """
    import matplotlib
    matplotlib.use("Agg")   # ensure headless — must call before importing pyplot

    if params is None:
        params = ContrastParams()

    canvas = CryoCanvas(figsize=figsize)
    canvas.load_image(dm4img, params)

    if annotations is not None:
        canvas.store.replace_all(list(annotations))

    if add_scalebar and not _has_scalebar(canvas.store):
        canvas.add_default_scalebar(color=scalebar_color)

    out = canvas.save(output_path, dpi=dpi, fmt=fmt)
    canvas.close()
    return out


def _has_scalebar(store: AnnotationStore) -> bool:
    return any(getattr(a, "type", "") == "scalebar" for a in store)
