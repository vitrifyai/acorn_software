"""Annotation data model — typed dataclasses + observable store with undo.

Every annotation carries a ``provenance`` block (see ``acorn.core.provenance``).
The store is the single choke point that stamps provenance on ``add`` and logs
mutations, so predictors, plugins, CLU, and manual drawing are all captured
without per-caller changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, Union

from acorn.core.provenance import (
    Provenance, provenance_from_dict, on_add, on_update, on_remove,
)


def _prov() -> Provenance:
    return Provenance()


# ── Annotation primitives ─────────────────────────────────────────────────────

@dataclass
class ArrowAnnotation:
    type: str = field(default="arrow", init=False)
    p1: tuple[float, float] = (0.0, 0.0)   # tail (x, y) in image pixels
    p2: tuple[float, float] = (0.0, 0.0)   # head
    color: str = "#FFFF00"
    linewidth: float = 2.0
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class LineAnnotation:
    type: str = field(default="line", init=False)
    p1: tuple[float, float] = (0.0, 0.0)
    p2: tuple[float, float] = (0.0, 0.0)
    color: str = "#FFFF00"
    linewidth: float = 2.0
    linestyle: str = "-"
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class CircleAnnotation:
    type: str = field(default="circle", init=False)
    cx: float = 0.0
    cy: float = 0.0
    r: float = 10.0
    color: str = "#FFFF00"
    linewidth: float = 2.0
    linestyle: str = "-"
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class RectangleAnnotation:
    type: str = field(default="rectangle", init=False)
    x0: float = 0.0
    y0: float = 0.0
    x1: float = 0.0
    y1: float = 0.0
    color: str = "#FFFF00"
    linewidth: float = 2.0
    linestyle: str = "-"
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class TextAnnotation:
    type: str = field(default="text", init=False)
    x: float = 0.0
    y: float = 0.0
    label: str = "Label"
    color: str = "#FFFF00"
    fontsize: int = 12
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class ScalebarAnnotation:
    type: str = field(default="scalebar", init=False)
    nm: float = 100.0         # physical length in nm
    x_frac: float = 0.03      # left edge as fraction of image width
    y_frac: float = 0.93      # top edge as fraction of image height
    color: str = "#FFFFFF"
    linewidth: float = 2.0
    fontsize: int = 12
    provenance: Provenance = field(default_factory=_prov)


# ── Measurement overlays (output from MeasurementEngine) ──────────────────────

@dataclass
class DistanceMeasurement:
    type: str = field(default="distance", init=False)
    p1: tuple[float, float] = (0.0, 0.0)
    p2: tuple[float, float] = (0.0, 0.0)
    distance_nm: float = 0.0
    distance_px: float = 0.0
    calibrated: bool = True
    color: str = "#00FF88"
    linewidth: float = 1.5
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class AngleMeasurement:
    type: str = field(default="angle", init=False)
    p1: tuple[float, float] = (0.0, 0.0)       # ray 1 endpoint
    vertex: tuple[float, float] = (0.0, 0.0)   # vertex
    p2: tuple[float, float] = (0.0, 0.0)       # ray 2 endpoint
    angle_deg: float = 0.0
    color: str = "#00FF88"
    linewidth: float = 1.5
    provenance: Provenance = field(default_factory=_prov)


@dataclass
class ROIAnnotation:
    type: str = field(default="roi", init=False)
    vertices: list = field(default_factory=list)    # list of (x, y) in image px
    area_nm2: float = 0.0
    stats: dict = field(default_factory=dict)       # mean, std, min, max
    color: str = "#E8833A"
    linewidth: float = 1.5
    label: str = ""                                 # user-assigned region label
    provenance: Provenance = field(default_factory=_prov)


AnyAnnotation = Union[
    ArrowAnnotation, LineAnnotation, CircleAnnotation, RectangleAnnotation,
    TextAnnotation, ScalebarAnnotation, DistanceMeasurement,
    AngleMeasurement, ROIAnnotation,
]

_TYPE_MAP: dict[str, type] = {
    "arrow":      ArrowAnnotation,
    "line":       LineAnnotation,
    "circle":     CircleAnnotation,
    "rectangle":  RectangleAnnotation,
    "text":       TextAnnotation,
    "scalebar":   ScalebarAnnotation,
    "distance":   DistanceMeasurement,
    "angle":      AngleMeasurement,
    "roi":        ROIAnnotation,
}


# ── Observable annotation store ───────────────────────────────────────────────

class AnnotationStore:
    """
    Observable list of annotations. Supports undo and JSON serialisation.

    Register callbacks with ``on_change()``; they are called with the
    current list whenever the store is mutated. ``add``/``update``/``remove``
    are the provenance choke point (stamp on add, log every mutation).
    """

    def __init__(self) -> None:
        self._items: list[AnyAnnotation] = []
        self._callbacks: list[Callable[[list[AnyAnnotation]], None]] = []

    # ── mutation ──────────────────────────────────────────────────────────────

    def add(self, annotation: AnyAnnotation) -> None:
        on_add(annotation)          # stamp provenance + log annotation_add
        self._items.append(annotation)
        self._notify()

    def update(self, annotation: AnyAnnotation, event: str = "annotation_edit") -> None:
        """Record an in-place edit of an already-added annotation (bumps
        modified_at / modification_count and logs the event)."""
        on_update(annotation, event)
        self._notify()

    def undo(self) -> Optional[AnyAnnotation]:
        if self._items:
            removed = self._items.pop()
            on_remove(removed)
            self._notify()
            return removed
        return None

    def remove(self, annotation: AnyAnnotation) -> bool:
        """Remove a specific annotation by identity. Returns True if found."""
        for i, item in enumerate(self._items):
            if item is annotation:
                on_remove(item)
                self._items.pop(i)
                self._notify()
                return True
        return False

    def clear(self) -> None:
        self._items.clear()
        self._notify()

    def replace_all(self, items: list[AnyAnnotation]) -> None:
        """Replace the entire store contents (e.g. after loading a session).
        Does NOT stamp — loaded annotations keep their existing provenance."""
        self._items = list(items)
        self._notify()

    # ── observation ───────────────────────────────────────────────────────────

    def on_change(self, callback: Callable[[list[AnyAnnotation]], None]) -> None:
        """Register a callback invoked whenever the store changes."""
        self._callbacks.append(callback)

    def _notify(self) -> None:
        for cb in self._callbacks:
            cb(list(self._items))

    # ── container protocol ────────────────────────────────────────────────────

    def __iter__(self):
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> AnyAnnotation:
        return self._items[idx]

    # ── serialisation ─────────────────────────────────────────────────────────

    def to_json(self) -> str:
        return json.dumps([asdict(a) for a in self._items], indent=2)

    @classmethod
    def from_json(cls, data: str) -> "AnnotationStore":
        import dataclasses
        store = cls()
        for d in json.loads(data):
            if not isinstance(d, dict):
                continue
            ann_type = d.pop("type", None)
            klass = _TYPE_MAP.get(ann_type)
            if klass is None:
                continue
            # Keep only fields this dataclass accepts, so an unknown/forward-
            # incompatible key doesn't crash the whole load.
            valid = {f.name for f in dataclasses.fields(klass) if f.init}
            kwargs = {k: v for k, v in d.items() if k in valid}
            # Point fields are (x, y) tuples
            for pf in ("p1", "p2", "vertex"):
                if isinstance(kwargs.get(pf), list):
                    kwargs[pf] = tuple(kwargs[pf])
            # ROI vertices is a list of (x, y) points
            if isinstance(kwargs.get("vertices"), list):
                kwargs["vertices"] = [
                    tuple(v) if isinstance(v, (list, tuple)) else v
                    for v in kwargs["vertices"]
                ]
            # Provenance block: rebuild from dict; v3 sidecars have none, so the
            # dataclass default (origin=unknown, created_at=None) applies — which
            # is exactly the "fill unknown rather than fail" behaviour we want.
            if isinstance(kwargs.get("provenance"), dict):
                kwargs["provenance"] = provenance_from_dict(kwargs["provenance"])
            # One malformed record must not discard the rest.
            try:
                store._items.append(klass(**kwargs))
            except (TypeError, ValueError):
                continue
        return store
