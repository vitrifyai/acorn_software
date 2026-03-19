"""Export labelled ROI masks for segmentation workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from acorn.core.annotations import AnnotationStore


def export_masks(
    store: "AnnotationStore",
    image_shape: tuple[int, int],
    output_path: str | Path,
) -> dict:
    """
    Export ROI annotations as a labelled mask PNG + JSON manifest.

    The mask is a uint8 PNG the same size as the image:
      - 0  = unlabelled
      - 1+ = region index (in draw order)

    The JSON maps each index to its label name and vertices.

    Parameters
    ----------
    store        : annotation store containing ROIAnnotations
    image_shape  : (height, width) of the source image
    output_path  : stem path — saves <stem>_mask.png and <stem>_labels.json

    Returns
    -------
    dict with keys "mask_path", "json_path", "n_regions"
    """
    from skimage.draw import polygon as skpoly
    from PIL import Image

    out = Path(output_path)
    h, w = image_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    rois = [a for a in store if getattr(a, "type", None) == "roi"]
    manifest = {}

    for idx, roi in enumerate(rois, start=1):
        if len(roi.vertices) < 3:
            continue
        verts = np.array(roi.vertices)
        rr, cc = skpoly(verts[:, 1], verts[:, 0], shape=(h, w))
        mask[rr, cc] = idx
        manifest[idx] = {
            "label": roi.label or f"region_{idx}",
            "color": roi.color,
            "vertices": [[float(x), float(y)] for x, y in roi.vertices],
            "area_nm2": roi.area_nm2,
        }

    mask_path = out.parent / f"{out.stem}_mask.png"
    json_path = out.parent / f"{out.stem}_labels.json"

    Image.fromarray(mask).save(str(mask_path))
    json_path.write_text(json.dumps(manifest, indent=2))

    return {
        "mask_path": str(mask_path),
        "json_path": str(json_path),
        "n_regions": len(rois),
    }
