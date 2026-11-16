"""
Per-slice annotations for a volume, and conversion to / from a label volume.

ACORN stores annotations as polygon ROIs (ROIAnnotation) in an AnnotationStore
per 2D image.  For a tomogram we keep one list of annotations per z slice,
persist them in a single sidecar next to the volume
(``<dir>/.<stem>.acorn3d.json``) and rasterise them into a (Z, Y, X) integer
label volume for export and rendering.

Class ids are assigned from the label strings in a stable, case-insensitive
order that the caller can pin with ``class_order``.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from skimage.draw import polygon as _polygon

from acorn.core.annotations import AnnotationStore, ROIAnnotation, _TYPE_MAP  # type: ignore


SIDECAR_VERSION = 1


def sidecar_path_for_volume(volume_path: str | Path) -> Path:
    p = Path(volume_path)
    return p.with_name(f".{p.stem}.acorn3d.json")


class VolumeAnnotations:
    """Annotations per z slice for one volume."""

    def __init__(self, n_slices: int, volume_path: Optional[Path] = None) -> None:
        self.n_slices = int(n_slices)
        self.volume_path = Path(volume_path) if volume_path else None
        self._per_z: dict[int, list] = {}
        self.pixel_size_nm: Optional[float] = None
        self.voxel_depth_nm: Optional[float] = None
        self.prompts: list[dict] = []      # record of prompts used (text, z, confidence, ...)

    # ── access ──────────────────────────────────────────────────────────────
    def get(self, z: int) -> list:
        return list(self._per_z.get(int(z), []))

    def set(self, z: int, anns: Iterable) -> None:
        items = [a for a in anns]
        if items:
            self._per_z[int(z)] = items
        else:
            self._per_z.pop(int(z), None)

    def clear_z(self, z: int) -> None:
        self._per_z.pop(int(z), None)

    def clear_all(self) -> None:
        self._per_z.clear()

    def slices_with_annotations(self) -> list[int]:
        return sorted(self._per_z)

    def labels(self) -> list[str]:
        seen: dict[str, str] = {}
        for anns in self._per_z.values():
            for a in anns:
                lab = (getattr(a, "label", "") or "Unlabelled")
                seen.setdefault(lab.lower(), lab)
        return [seen[k] for k in sorted(seen)]

    def count(self) -> int:
        return sum(len(v) for v in self._per_z.values())

    # ── persistence ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "version": SIDECAR_VERSION,
            "n_slices": self.n_slices,
            "pixel_size_nm": self.pixel_size_nm,
            "voxel_depth_nm": self.voxel_depth_nm,
            "prompts": self.prompts,
            "slices": {str(z): [asdict(a) for a in anns] for z, anns in sorted(self._per_z.items())},
        }

    @classmethod
    def from_dict(cls, d: dict, volume_path: Optional[Path] = None) -> "VolumeAnnotations":
        va = cls(d.get("n_slices", 0), volume_path)
        va.pixel_size_nm = d.get("pixel_size_nm")
        va.voxel_depth_nm = d.get("voxel_depth_nm")
        va.prompts = list(d.get("prompts", []))
        for z, items in d.get("slices", {}).items():
            anns = []
            for item in items:
                item = dict(item)
                t = item.pop("type", "roi")
                klass = _TYPE_MAP.get(t)
                if klass is None:
                    continue
                allowed = {f for f in klass.__dataclass_fields__ if klass.__dataclass_fields__[f].init}
                prov = item.pop("provenance", None)
                kwargs = {k: v for k, v in item.items() if k in allowed}
                try:
                    ann = klass(**kwargs)
                except TypeError:
                    continue
                if prov is not None and hasattr(ann, "provenance"):
                    try:
                        from acorn.core.provenance import Provenance  # type: ignore
                        ann.provenance = Provenance(**prov) if isinstance(prov, dict) else prov
                    except Exception:
                        pass
                anns.append(ann)
            va.set(int(z), anns)
        return va

    def save(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else sidecar_path_for_volume(self.volume_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1))
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, volume_path: Path, n_slices: int) -> "VolumeAnnotations":
        path = sidecar_path_for_volume(volume_path)
        if path.exists():
            try:
                va = cls.from_dict(json.loads(path.read_text()), Path(volume_path))
                va.n_slices = n_slices
                return va
            except (json.JSONDecodeError, OSError):
                pass
        return cls(n_slices, Path(volume_path))

    # ── rasterisation ───────────────────────────────────────────────────────
    def class_ids(self, class_order: Optional[list[str]] = None) -> dict[str, int]:
        labels = self.labels()
        if class_order:
            ordered = [l for l in class_order] + [l for l in labels if l.lower() not in {c.lower() for c in class_order}]
        else:
            ordered = labels
        return {lab.lower(): i + 1 for i, lab in enumerate(ordered)}

    def to_label_volume(self, shape_yx: tuple[int, int], class_order: Optional[list[str]] = None,
                        dtype=np.uint8) -> tuple[np.ndarray, dict[int, str]]:
        """Rasterise polygons into a (Z, Y, X) label volume. Later ROIs overwrite earlier ones."""
        ids = self.class_ids(class_order)
        names = {}
        for lab in ([*class_order] if class_order else []) + self.labels():
            names.setdefault(ids[lab.lower()], lab)
        ny, nx = shape_yx
        vol = np.zeros((self.n_slices, ny, nx), dtype)
        for z, anns in self._per_z.items():
            if not (0 <= z < self.n_slices):
                continue
            for a in anns:
                if not isinstance(a, ROIAnnotation) or len(a.vertices) < 3:
                    continue
                cid = ids[(a.label or "Unlabelled").lower()]
                xs = np.array([v[0] for v in a.vertices]); ys = np.array([v[1] for v in a.vertices])
                rr, cc = _polygon(ys, xs, shape=(ny, nx))
                vol[z, rr, cc] = cid
        return vol, names


def masks_to_rois(masks: Iterable[np.ndarray], label: str, color: str, pixel_size_nm: float,
                  tolerance: float = 1.0, min_area_px: int = 20) -> list[ROIAnnotation]:
    """Boolean masks -> polygon ROIs (largest contour of each mask, as ACORN does)."""
    from skimage import measure
    rois = []
    for m in masks:
        if m.sum() < min_area_px:
            continue
        contours = measure.find_contours(np.pad(m.astype(np.float32), 1), 0.5)
        if not contours:
            continue
        c = max(contours, key=len) - 1.0
        c = measure.approximate_polygon(c, tolerance)
        if len(c) < 3:
            continue
        verts = [(float(x), float(y)) for y, x in c]
        area_nm2 = float(m.sum()) * pixel_size_nm ** 2
        rois.append(ROIAnnotation(vertices=verts, area_nm2=area_nm2, label=label, color=color))
    return rois


def write_label_mrc(vol: np.ndarray, path: Path, voxel_nm: tuple[float, float, float]) -> Path:
    import mrcfile
    path = Path(path)
    with mrcfile.new(str(path), overwrite=True) as m:
        m.set_data(vol.astype(np.int8 if vol.max() < 128 else np.int16))
        vz, vy, vx = voxel_nm
        m.voxel_size = (vx * 10, vy * 10, vz * 10)
        m.update_header_from_data(); m.update_header_stats()
    return path
