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


def open_mrc_memmap(path: "str | Path"):
    """
    Memory-map an MRC file's data without mrcfile.mmap, which overflows a 32-bit
    offset on files over 2 GB with current numpy.  Returns (array[Z,H,W], voxel_size_A_xyz).
    """
    import mrcfile
    with mrcfile.open(str(path), permissive=True, header_only=True) as mrc:
        h = mrc.header
        nx, ny, nz = int(h.nx), int(h.ny), int(h.nz)
        mode = int(h.mode)
        offset = 1024 + int(h.nsymbt)
        vox = (float(mrc.voxel_size.x), float(mrc.voxel_size.y), float(mrc.voxel_size.z))
    dtype = {0: np.int8, 1: np.int16, 2: np.float32, 6: np.uint16, 12: np.float16, 101: np.uint8}.get(mode)
    if dtype is None:
        raise ValueError(f"unsupported MRC mode {mode}")
    shape = (nz, ny, nx) if nz > 1 else (ny, nx)
    return np.memmap(str(path), dtype=dtype, mode="r", offset=offset, shape=shape), vox


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
        self._mm = None                            # np.memmap for large MRC files
        self._shape_3d: tuple[int, int, int] = (0, 0, 0)
        self._current_z: int = 0
        self.slab: int = 1          # number of slices averaged around the current z for display
        self.smooth_px: float = 0.0 # Gaussian sigma (px) applied to the displayed slice
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
                data, _ = open_mrc_memmap(p)
                return data.ndim == 3 and data.shape[0] > 1
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
        data, voxel = open_mrc_memmap(filepath)
        if data.ndim == 2:
            self._shape_3d = (1, data.shape[0], data.shape[1])
        else:
            self._shape_3d = tuple(data.shape)
        nbytes = data.size * 4
        if nbytes < self.LAZY_THRESHOLD_BYTES:
            self._data = np.array(data, dtype=np.float32)
        else:
            self._mm = data          # keep the memmap for per-slice reads
        if voxel[0] > 0:
            self.meta.pixel_size = voxel[0] * 0.1  # Angstrom -> nm
            self.meta.pixel_size_from_header = True
            self.meta.voxel_depth_nm = voxel[2] * 0.1 if voxel[2] > 0 else self.meta.pixel_size
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

    def _read_slice(self, z: int) -> np.ndarray:
        z = max(0, min(int(z), self.n_slices - 1))
        if self._data is not None:
            return np.asarray(self._data[z], dtype=np.float32)
        ext = self._path.suffix.lower()
        if ext in self.MRC_EXTS:
            mm = getattr(self, "_mm", None)
            if mm is None:
                mm, _ = open_mrc_memmap(self._path)
                self._mm = mm
            return np.array(mm if mm.ndim == 2 else mm[z], dtype=np.float32)
        import tifffile
        with tifffile.TiffFile(str(self._path)) as tf:
            page = tf.pages[0] if len(tf.pages) == 1 else tf.pages[z]
            return page.asarray().astype(np.float32)

    @property
    def raw(self) -> np.ndarray:
        """The displayed 2D image: the current slice, or the mean of `slab` slices around it, optionally smoothed."""
        k = max(1, int(self.slab))
        if k > 1 or self.smooth_px > 0:
            lo = max(0, self._current_z - k // 2); hi = min(self.n_slices, self._current_z + k // 2 + 1)
            img = np.mean([self._read_slice(z) for z in range(lo, hi)], axis=0, dtype=np.float32)
            if self.smooth_px > 0:
                from scipy.ndimage import gaussian_filter
                img = gaussian_filter(img, self.smooth_px)
            return img
        if self._data is not None:
            return self._data[self._current_z]
        ext = self._path.suffix.lower()
        if ext in self.MRC_EXTS:
            mm = getattr(self, "_mm", None)
            if mm is None:
                mm, _ = open_mrc_memmap(self._path)
                self._mm = mm
            if mm.ndim == 2:
                return np.array(mm, dtype=np.float32)
            return np.array(mm[self._current_z], dtype=np.float32)
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

    # ── DM4Image API the canvas and panels probe ──────────────────────────────
    @property
    def is_color(self) -> bool:
        return False

    @property
    def is_movie(self) -> bool:
        return False          # z is navigated by the 3D panel, not the movie controls

    @property
    def n_frames(self) -> int:
        return 0

    @property
    def frames(self):
        return None

    def get_frame(self, i: int) -> np.ndarray:
        self.set_slice(i)
        return self.raw

    def summary(self) -> str:
        z, h, w = self._shape_3d
        return f"{self.filename}: volume {z} x {h} x {w}, {self.meta.pixel_size:.3f} nm/px"

    def apply_binning(self, *a, **k):
        raise NotImplementedError("bin the volume on disk; the viewer shows it as stored")

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
