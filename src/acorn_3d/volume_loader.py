"""
VolumeImage — lazy multi-slice loader for MRC and multi-page TIFF files.

Exposes the same .raw / .pixel_size / .shape / .filepath / .filename / .meta
interface as DM4Image so CryoCanvas.load_image() works unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class VolumeMeta:
    pixel_size: float = 1.0
    pixel_size_from_header: bool = False
    voxel_depth_nm: float = 1.0   # nm per z-slice
    shape: tuple = field(default_factory=tuple)
    filepath: Path = field(default_factory=Path)
    filename: str = ""
    mag: Optional[float] = None
    voltage_kV: Optional[float] = None
    raw_dtype: str = "float32"


class VolumeImage:
    """
    Lazy-loading wrapper for 3D MRC / multi-page TIFF volumes.

    DM4Image compatibility shim: exposes .raw (the current 2D slice),
    .pixel_size, .shape, .filepath, .filename, .meta so the existing
    canvas pipeline works unchanged with any slice.
    """

    LAZY_THRESHOLD_BYTES = 2 * 1024 ** 3  # 2 GB — load fully below this

    MRC_EXTS  = {".mrc", ".mrcs"}
    TIFF_EXTS = {".tif", ".tiff"}

    def __init__(self) -> None:
        self._path: Optional[Path] = None
        self._data: Optional[np.ndarray] = None   # (Z, H, W) float32 if fully loaded
        self._shape_3d: tuple[int, int, int] = (0, 0, 0)
        self._current_z: int = 0
        self.meta: VolumeMeta = VolumeMeta()

    @classmethod
    def from_file(cls, filepath: str | Path) -> "VolumeImage":
        obj = cls()
        p = Path(filepath)
        obj._path = p
        ext = p.suffix.lower()
        if ext in cls.MRC_EXTS:
            obj._probe_mrc(p)
        elif ext in cls.TIFF_EXTS:
            obj._probe_tiff(p)
        else:
            raise ValueError(f"acorn_3d handles MRC and TIFF volumes only, got {ext!r}")
        return obj

    @classmethod
    def is_volume_file(cls, filepath: str | Path) -> bool:
        """Return True if the file is a recognised multi-slice format."""
        p = Path(filepath)
        ext = p.suffix.lower()
        if ext not in (cls.MRC_EXTS | cls.TIFF_EXTS):
            return False
        if ext in cls.MRC_EXTS:
            try:
                import mrcfile
                with mrcfile.mmap(str(p), mode="r") as mrc:
                    return mrc.data.ndim == 3 and mrc.data.shape[0] > 1
            except Exception:
                return False
        else:
            try:
                import tifffile
                with tifffile.TiffFile(str(p)) as tf:
                    return len(tf.pages) > 1
            except Exception:
                return False

    def _probe_mrc(self, filepath: Path) -> None:
        import mrcfile
        with mrcfile.mmap(str(filepath), mode="r") as mrc:
            data = mrc.data
            if data.ndim == 2:
                self._shape_3d = (1, data.shape[0], data.shape[1])
            else:
                self._shape_3d = tuple(data.shape)
            nbytes = data.nbytes
            if nbytes < self.LAZY_THRESHOLD_BYTES:
                self._data = np.array(data, dtype=np.float32)
            voxel = mrc.voxel_size
            if hasattr(voxel, "x") and float(voxel.x) > 0:
                self.meta.pixel_size = float(voxel.x) * 0.1  # Angstrom -> nm
                self.meta.pixel_size_from_header = True
                self.meta.voxel_depth_nm = float(voxel.z) * 0.1 if float(voxel.z) > 0 else self.meta.pixel_size
        self.meta.filepath = filepath
        self.meta.filename = filepath.stem
        self.meta.shape    = self._shape_3d[1:]  # (H, W) for 2D compatibility

    def _probe_tiff(self, filepath: Path) -> None:
        import tifffile
        with tifffile.TiffFile(str(filepath)) as tf:
            n_pages = len(tf.pages)
            page_shape = tf.pages[0].shape
            self._shape_3d = (n_pages,) + tuple(page_shape[:2])
            nbytes = tf.pages[0].size * n_pages * 4
            if nbytes < self.LAZY_THRESHOLD_BYTES:
                arr = tifffile.imread(str(filepath))
                if arr.ndim == 2:
                    arr = arr[np.newaxis]
                elif arr.ndim > 3:
                    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])
                self._data = arr.astype(np.float32)
            # Try to read pixel size from ImageJ / OME metadata
            try:
                tags = tf.pages[0].tags
                xres = tags.get("XResolution")
                unit = tags.get("ResolutionUnit")
                if xres and unit:
                    xres_val = xres.value
                    unit_val = unit.value
                    if isinstance(xres_val, tuple):
                        px_per_unit = xres_val[0] / xres_val[1] if xres_val[1] != 0 else float(xres_val[0])
                    else:
                        px_per_unit = float(xres_val)
                    if unit_val == 2 and px_per_unit > 0:    # inch
                        self.meta.pixel_size = 25.4e6 / px_per_unit
                        self.meta.pixel_size_from_header = True
                    elif unit_val == 3 and px_per_unit > 0:  # cm
                        self.meta.pixel_size = 1e7 / px_per_unit
                        self.meta.pixel_size_from_header = True
            except Exception:
                pass
        self.meta.filepath = filepath
        self.meta.filename = filepath.stem
        self.meta.shape    = self._shape_3d[1:]

    # ── navigation ────────────────────────────────────────────────────────────

    @property
    def n_slices(self) -> int:
        return self._shape_3d[0] if self._shape_3d[0] > 0 else 1

    @property
    def current_z(self) -> int:
        return self._current_z

    def set_slice(self, z: int) -> None:
        self._current_z = max(0, min(z, self.n_slices - 1))

    # ── DM4Image compatibility shim ───────────────────────────────────────────

    @property
    def raw(self) -> np.ndarray:
        """Return current 2D slice as float32."""
        if self._data is not None:
            return self._data[self._current_z]
        ext = self._path.suffix.lower()
        if ext in self.MRC_EXTS:
            import mrcfile
            with mrcfile.mmap(str(self._path), mode="r") as mrc:
                if mrc.data.ndim == 2:
                    return mrc.data.astype(np.float32)
                return np.array(mrc.data[self._current_z], dtype=np.float32)
        else:
            import tifffile
            with tifffile.TiffFile(str(self._path)) as tf:
                if len(tf.pages) == 1:
                    return tf.pages[0].asarray().astype(np.float32)
                return tf.pages[self._current_z].asarray().astype(np.float32)

    def projection(self, method: str = "max", z_from: int = 0, z_to: Optional[int] = None) -> np.ndarray:
        """Return a 2D projection along Z."""
        z_to = z_to if z_to is not None else self.n_slices
        z_from = max(0, z_from)
        z_to   = min(self.n_slices, z_to)
        slices = []
        for z in range(z_from, z_to):
            self.set_slice(z)
            slices.append(self.raw)
        self.set_slice(self._current_z)  # restore
        stack = np.stack(slices, axis=0)
        if method == "max":
            return stack.max(axis=0)
        elif method == "min":
            return stack.min(axis=0)
        else:  # mean
            return stack.mean(axis=0)

    @property
    def pixel_size(self) -> float:
        return self.meta.pixel_size

    @property
    def shape(self) -> tuple:
        return self.meta.shape

    @property
    def filepath(self) -> Path:
        return self.meta.filepath

    @property
    def filename(self) -> str:
        return self.meta.filename

    @property
    def mag(self):
        return self.meta.mag

    @property
    def voltage_kV(self):
        return self.meta.voltage_kV
