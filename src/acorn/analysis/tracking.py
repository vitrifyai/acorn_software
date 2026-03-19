"""
ACORN Particle Tracking Module  (tracking.py)
==============================================
Link and track particle annotations across image sequences (time series or
z-stacks).

Algorithm
---------
1. Extract centroids from each frame's AnnotationStore (ROI, circle,
   rectangle, freehand annotations).
2. Frame-to-frame linking via the Hungarian algorithm (optimal minimum-cost
   assignment; scipy.optimize.linear_sum_assignment) with a configurable
   maximum displacement threshold.
3. Gap closing: a track that disappears for up to *max_gap* frames can be
   re-linked if a candidate centroid appears within *max_displacement_nm*.
4. Tracks shorter than *min_frames* are discarded as spurious detections.

Output
------
pandas.DataFrame with columns:
  track_id, frame, ann_idx, x_px, y_px, x_nm, y_nm, area_nm2,
  displacement_nm, cumulative_displacement_nm, annotation_type

Requires: numpy, scipy, pandas
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── centroid extraction ───────────────────────────────────────────────────────

def _centroid_of_annotation(ann) -> Optional[tuple[float, float]]:
    """Return (x_px, y_px) centroid for any supported annotation type."""
    t = getattr(ann, "type", None)

    if t == "circle":
        return float(ann.cx), float(ann.cy)

    if t == "rectangle":
        return (
            float(ann.x0 + ann.x1) / 2.0,
            float(ann.y0 + ann.y1) / 2.0,
        )

    if t == "roi":
        verts = ann.vertices
        if not verts:
            return None
        xs = [v[0] for v in verts]
        ys = [v[1] for v in verts]
        return float(np.mean(xs)), float(np.mean(ys))

    if t in ("arrow", "line", "distance"):
        p1 = getattr(ann, "p1", None)
        p2 = getattr(ann, "p2", None)
        if p1 and p2:
            return (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0

    return None


def _area_of_annotation(ann) -> float:
    """Return area_nm2 if stored on the annotation, else 0.0."""
    return float(getattr(ann, "area_nm2", 0.0))


def extract_centroids(store, pixel_size_nm: float = 1.0) -> list[dict]:
    """
    Extract particle centroids from an AnnotationStore.

    Returns a list of dicts with keys:
      ann_idx, x_px, y_px, x_nm, y_nm, area_nm2, annotation_type
    Only annotations that have a computable centroid are included.
    """
    rows = []
    for i, ann in enumerate(store):
        c = _centroid_of_annotation(ann)
        if c is None:
            continue
        x_px, y_px = c
        rows.append({
            "ann_idx": i,
            "x_px": x_px,
            "y_px": y_px,
            "x_nm": x_px * pixel_size_nm,
            "y_nm": y_px * pixel_size_nm,
            "area_nm2": _area_of_annotation(ann),
            "annotation_type": ann.type,
        })
    return rows


# ── frame-to-frame linker ─────────────────────────────────────────────────────

def _link_two_frames(
    cents_a: list[dict],
    cents_b: list[dict],
    max_displacement_px: float,
) -> list[tuple[int, int]]:
    """
    Return a list of (index_in_a, index_in_b) optimal assignments.
    Pairs whose distance exceeds max_displacement_px are excluded.

    Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
    """
    from scipy.optimize import linear_sum_assignment

    if not cents_a or not cents_b:
        return []

    xa = np.array([[c["x_px"], c["y_px"]] for c in cents_a])
    xb = np.array([[c["x_px"], c["y_px"]] for c in cents_b])

    # cost matrix: Euclidean distances
    diff = xa[:, np.newaxis, :] - xb[np.newaxis, :, :]   # (Na, Nb, 2)
    dist = np.sqrt((diff ** 2).sum(axis=2))               # (Na, Nb)

    # set cost to a large number for pairs that exceed the threshold
    inf_cost = max_displacement_px * 10.0
    cost = np.where(dist <= max_displacement_px, dist, inf_cost)

    row_ind, col_ind = linear_sum_assignment(cost)

    pairs = []
    for r, c in zip(row_ind, col_ind):
        if dist[r, c] <= max_displacement_px:
            pairs.append((int(r), int(c)))
    return pairs


# ── main tracking entry point ─────────────────────────────────────────────────

def track_annotations(
    stores: Sequence,
    pixel_size_nm: float = 1.0,
    max_displacement_nm: float = 500.0,
    min_frames: int = 2,
    max_gap: int = 1,
) -> "pandas.DataFrame":
    """
    Track particles across a sequence of AnnotationStores.

    Parameters
    ----------
    stores : sequence of AnnotationStore (one per frame / time point)
    pixel_size_nm : nm/px (same for all frames)
    max_displacement_nm : maximum allowed frame-to-frame displacement (nm)
    min_frames : discard tracks shorter than this many frames
    max_gap : number of frames a particle may disappear and still be re-linked

    Returns
    -------
    pandas.DataFrame with columns:
      track_id, frame, ann_idx, x_px, y_px, x_nm, y_nm, area_nm2,
      displacement_nm, cumulative_displacement_nm, annotation_type
    """
    import pandas as pd

    if not stores:
        return pd.DataFrame()

    max_disp_px = max_displacement_nm / max(pixel_size_nm, 1e-9)

    # Step 1: extract centroids per frame
    per_frame: list[list[dict]] = [
        extract_centroids(s, pixel_size_nm) for s in stores
    ]

    n_frames = len(per_frame)

    # Step 2: active tracks — dict mapping track_id -> last seen frame + last centroid index
    # Each track is a list of (frame, centroid_dict)
    tracks: list[list[tuple[int, dict]]] = []
    # active_tracks: list of (track_idx, last_frame, last_centroid_dict)
    active: list[tuple[int, int, dict]] = []

    def _start_track(frame: int, cent: dict) -> None:
        track_idx = len(tracks)
        tracks.append([(frame, cent)])
        active.append((track_idx, frame, cent))

    def _extend_track(track_idx: int, frame: int, cent: dict) -> None:
        tracks[track_idx].append((frame, cent))
        # update active entry
        for i, (tid, _, _) in enumerate(active):
            if tid == track_idx:
                active[i] = (track_idx, frame, cent)
                return

    # seed with frame 0
    for c in per_frame[0]:
        _start_track(0, c)

    # Step 3: link frames
    for frame in range(1, n_frames):
        cents_b = per_frame[frame]

        # collect active tracks that are within the gap window
        linkable = [
            (tid, last_f, last_c)
            for (tid, last_f, last_c) in active
            if frame - last_f <= max_gap + 1
        ]

        if not linkable or not cents_b:
            # no candidates — start new tracks for all detections
            for c in cents_b:
                _start_track(frame, c)
            # remove tracks that have been inactive too long
            active[:] = [
                (tid, lf, lc) for (tid, lf, lc) in active
                if frame - lf <= max_gap
            ]
            continue

        cents_a = [lc for (_, _, lc) in linkable]
        pairs = _link_two_frames(cents_a, cents_b, max_disp_px)

        linked_b = set()
        for ia, ib in pairs:
            tid = linkable[ia][0]
            _extend_track(tid, frame, cents_b[ib])
            linked_b.add(ib)

        # unlinked detections start new tracks
        for ib, c in enumerate(cents_b):
            if ib not in linked_b:
                _start_track(frame, c)

        # retire tracks inactive beyond max_gap
        active[:] = [
            (tid, lf, lc) for (tid, lf, lc) in active
            if frame - lf <= max_gap
        ]

    # Step 4: build output DataFrame
    rows = []
    for track_id, events in enumerate(tracks):
        if len(events) < min_frames:
            continue
        prev_x, prev_y = None, None
        cum_disp = 0.0
        for frame, cent in events:
            x_nm = cent["x_nm"]
            y_nm = cent["y_nm"]
            if prev_x is not None:
                disp = float(np.sqrt((x_nm - prev_x) ** 2 + (y_nm - prev_y) ** 2))
            else:
                disp = 0.0
            cum_disp += disp
            rows.append({
                "track_id": track_id,
                "frame": frame,
                "ann_idx": cent["ann_idx"],
                "x_px": cent["x_px"],
                "y_px": cent["y_px"],
                "x_nm": x_nm,
                "y_nm": y_nm,
                "area_nm2": cent["area_nm2"],
                "annotation_type": cent["annotation_type"],
                "displacement_nm": disp,
                "cumulative_displacement_nm": cum_disp,
            })
            prev_x, prev_y = x_nm, y_nm

    if not rows:
        return pd.DataFrame(columns=[
            "track_id", "frame", "ann_idx", "x_px", "y_px",
            "x_nm", "y_nm", "area_nm2", "annotation_type",
            "displacement_nm", "cumulative_displacement_nm",
        ])

    df = pd.DataFrame(rows)
    return df.sort_values(["track_id", "frame"]).reset_index(drop=True)


# ── per-track statistics ───────────────────────────────────────────────────────

def track_statistics(df: "pandas.DataFrame") -> "pandas.DataFrame":
    """
    Summarise each track.

    Returns a DataFrame with one row per track_id and columns:
      track_id, n_frames, total_displacement_nm, mean_step_nm,
      max_step_nm, net_displacement_nm, mean_area_nm2, first_frame, last_frame
    """
    import pandas as pd

    if df.empty:
        return pd.DataFrame()

    rows = []
    for tid, grp in df.groupby("track_id"):
        grp = grp.sort_values("frame")
        steps = grp["displacement_nm"].values[1:]  # skip the zero first step
        x0, y0 = grp["x_nm"].iloc[0], grp["y_nm"].iloc[0]
        xn, yn = grp["x_nm"].iloc[-1], grp["y_nm"].iloc[-1]
        net = float(np.sqrt((xn - x0) ** 2 + (yn - y0) ** 2))
        rows.append({
            "track_id": tid,
            "n_frames": len(grp),
            "first_frame": int(grp["frame"].min()),
            "last_frame": int(grp["frame"].max()),
            "total_displacement_nm": float(grp["cumulative_displacement_nm"].max()),
            "mean_step_nm": float(steps.mean()) if len(steps) > 0 else 0.0,
            "max_step_nm": float(steps.max()) if len(steps) > 0 else 0.0,
            "net_displacement_nm": net,
            "mean_area_nm2": float(grp["area_nm2"].mean()),
        })

    return pd.DataFrame(rows).sort_values("track_id").reset_index(drop=True)
