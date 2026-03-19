"""
RELION STAR file parser and particle-pick importer.

STAR files (used by RELION, cryoSPARC, etc.) store particle coordinate tables.
This module parses them and converts particle picks into ROI annotations for
the ACORN annotation store.

Supported columns (case-insensitive)
-------------------------------------
_rlnCoordinateX, _rlnCoordinateY   — particle centre in pixels
_rlnClassNumber                     — optional class assignment
_rlnMagnification, _rlnDetectorPixelSize — optional pixel calibration

Usage
-----
from acorn.core.star_loader import load_star_picks, picks_to_roi_annotations

picks = load_star_picks("particles.star")
print(picks[0])  # {"x": 1024.5, "y": 2048.3, "class": 1}

from acorn.core.annotations import AnnotationStore
store = AnnotationStore()
picks_to_roi_annotations(picks, store, radius_px=50, label="Foreground")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


# ── parser ────────────────────────────────────────────────────────────────────

def _parse_star_blocks(text: str) -> list[dict]:
    """
    Parse all data blocks in a STAR file.
    Returns a list of dicts, one per block:
        { "name": str, "columns": [str...], "rows": [[str...]] }
    """
    blocks: list[dict] = []
    current: Optional[dict] = None
    in_loop = False
    columns: list[str] = []
    rows: list[list[str]] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("data_"):
            # Flush previous block
            if current is not None:
                current["columns"] = columns
                current["rows"] = rows
                blocks.append(current)
            current = {"name": line[5:], "columns": [], "rows": []}
            in_loop = False
            columns = []
            rows = []
            continue

        if line == "loop_":
            in_loop = True
            columns = []
            rows = []
            continue

        if in_loop:
            if line.startswith("_"):
                # Strip trailing comment / numbering like "#1"
                col_name = re.split(r"\s+", line)[0]
                columns.append(col_name)
            else:
                # Data row — may have fewer tokens if trailing columns are absent
                tokens = line.split()
                if tokens:
                    rows.append(tokens)

    # Flush last block
    if current is not None:
        current["columns"] = columns
        current["rows"] = rows
        blocks.append(current)

    return blocks


def load_star_picks(
    filepath: str | Path,
    x_col: str = "_rlnCoordinateX",
    y_col: str = "_rlnCoordinateY",
    class_col: str = "_rlnClassNumber",
) -> list[dict]:
    """
    Parse a STAR file and return a list of particle pick dicts.

    Each dict has:
        x      : float  — pixel X coordinate
        y      : float  — pixel Y coordinate
        class  : int    — class number (1 if not present)
        source : str    — source file path

    Parameters
    ----------
    filepath   : path to .star file
    x_col      : STAR column name for X coordinate
    y_col      : STAR column name for Y coordinate
    class_col  : STAR column name for class number (optional)
    """
    path = Path(filepath)
    text = path.read_text(encoding="utf-8", errors="replace")
    blocks = _parse_star_blocks(text)

    picks: list[dict] = []
    for block in blocks:
        cols = [c.lower() for c in block["columns"]]
        x_col_lower     = x_col.lower()
        y_col_lower     = y_col.lower()
        class_col_lower = class_col.lower()

        if x_col_lower not in cols or y_col_lower not in cols:
            continue  # this block has no coordinate columns

        xi = cols.index(x_col_lower)
        yi = cols.index(y_col_lower)
        ci = cols.index(class_col_lower) if class_col_lower in cols else None

        for row in block["rows"]:
            try:
                x = float(row[xi])
                y = float(row[yi])
            except (IndexError, ValueError):
                continue
            class_num = 1
            if ci is not None:
                try:
                    class_num = int(row[ci])
                except (IndexError, ValueError):
                    pass
            picks.append({"x": x, "y": y, "class": class_num, "source": str(path)})

    return picks


# ── annotation converter ──────────────────────────────────────────────────────

def picks_to_roi_annotations(
    picks: list[dict],
    store,          # AnnotationStore — avoid circular import
    radius_px: float = 50.0,
    label: str = "Foreground",
    color: str = "#00FF88",
) -> int:
    """
    Convert particle picks to circular ROI polygon annotations and add them
    to the given AnnotationStore.

    Each pick becomes a regular polygon (32-point circle) of radius ``radius_px``
    centred at (pick["x"], pick["y"]).

    Returns the number of ROIs added.
    """
    import math
    from acorn.core.annotations import ROIAnnotation

    n_sides = 32
    angles  = [2 * math.pi * i / n_sides for i in range(n_sides)]
    n_added = 0

    for pick in picks:
        cx, cy = pick["x"], pick["y"]
        vertices = [
            (cx + radius_px * math.cos(a), cy + radius_px * math.sin(a))
            for a in angles
        ]
        roi = ROIAnnotation(
            vertices  = vertices,
            area_nm2  = 0.0,
            stats     = {},
            color     = color,
            linewidth = 1.5,
            label     = label,
        )
        store.add(roi)
        n_added += 1

    return n_added
