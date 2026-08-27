"""
Multi-format image loader for cryo-EM data.

Supported formats
-----------------
  .dm4            — Gatan Digital Micrograph (ncempy)
  .tif / .tiff    — TIFF including multi-page (tifffile)
  .mrc / .mrcs    — MRC2014 cryo-EM standard (mrcfile)
  .emd            — HDF5-based Thermo Fisher / Velox or NCEM EMD
  .h5 / .hdf5     — Generic HDF5 image datasets
  .png / .jpg /
  .jpeg           — Standard 8/16-bit images (Pillow)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ── supported extensions ──────────────────────────────────────────────────────

DM4_EXTS   = {".dm4"}
TIFF_EXTS  = {".tif", ".tiff"}
MRC_EXTS   = {".mrc", ".mrcs"}
EMD_EXTS   = {".emd"}
HDF5_EXTS  = {".h5", ".hdf5"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
STAR_EXTS  = {".star"}
# Fluorescence / multichannel light-microscopy formats — read via bioio
# (pure-Python plugins: bioio-czi, bioio-nd2, bioio-lif; no Java required).
FLUO_EXTS  = {".czi", ".nd2", ".lif", ".ims", ".oib", ".oif", ".lsm"}

ALL_EXTS       = DM4_EXTS | TIFF_EXTS | MRC_EXTS | EMD_EXTS | HDF5_EXTS | IMAGE_EXTS | FLUO_EXTS
IMAGE_ONLY_EXTS = ALL_EXTS  # alias — STAR excluded from image scanning

# Formats that are electron-microscopy data — bandpass contrast is the right default
EM_EXTS = DM4_EXTS | MRC_EXTS | EMD_EXTS   # {".dm4", ".mrc", ".mrcs", ".emd"}
DEFAULT_EM_CONTRAST = "bandpass"  # used by _switch_to and contrast-panel init


# ── metadata dataclass ────────────────────────────────────────────────────────

@dataclass
class DM4Metadata:
    pixel_size: float = 1.0      # nm/px — AFTER binning, so measurements stay calibrated
    pixel_unit: str = "nm"
    pixel_size_from_header: bool = False  # True when read from file header
    bin_factor: int = 1          # analysis binning applied at load; 1 = native
    native_pixel_size: float = 1.0        # nm/px as stored in the file
    binning_cropped_px: tuple = (0, 0)    # rows, cols dropped so the shape divided
    mag: Optional[float] = None
    voltage_kV: Optional[float] = None
    shape: tuple = field(default_factory=tuple)
    raw_dtype: str = "float32"
    filepath: Path = field(default_factory=Path)
    filename: str = ""
    all_tags: dict = field(default_factory=dict)


# ── main image class ──────────────────────────────────────────────────────────

class DM4Image:
    """
    Container for a loaded cryo-EM image with calibrated metadata.

    Supports DM4, TIFF, MRC, PNG, and JPEG inputs.

    Usage
    -----
    img = DM4Image.from_file("sample.dm4")
    img = DM4Image.from_file("sample.mrc")
    img = DM4Image.from_file("sample.tif")
    """

    UNIT_TO_NM: dict[str, float] = {
        "nm": 1.0, "nanometer": 1.0, "nanometre": 1.0,
        "um": 1e3, "µm": 1e3, "micron": 1e3, "micrometer": 1e3,
        "pm": 1e-3, "picometer": 1e-3,
        "å": 0.1, "angstrom": 0.1, "a": 0.1,
        "m": 1e9,
    }

    def __init__(self) -> None:
        self.raw: Optional[np.ndarray] = None
        self._frames: Optional[np.ndarray] = None  # (n, h, w) float32 — set when file is a movie
        self.meta: DM4Metadata = DM4Metadata()

    # ── movie properties ──────────────────────────────────────────────────────

    @property
    def is_color(self) -> bool:
        """True when raw is an (H, W, 3) RGB array (not grayscale, not a movie frame)."""
        return self.raw is not None and self.raw.ndim == 3 and self.raw.shape[-1] == 3

    @property
    def is_movie(self) -> bool:
        return self._frames is not None

    @property
    def n_frames(self) -> int:
        return int(self._frames.shape[0]) if self._frames is not None else 0

    @property
    def frames(self) -> Optional[np.ndarray]:
        return self._frames

    def get_frame(self, i: int) -> np.ndarray:
        if self._frames is None:
            raise ValueError("Not a movie — no individual frames stored.")
        return self._frames[i % self._frames.shape[0]].astype(np.float32)

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def pixel_size(self) -> float:
        return self.meta.pixel_size

    @property
    def shape(self) -> tuple:
        return self.meta.shape

    @property
    def filename(self) -> str:
        return self.meta.filename

    @property
    def filepath(self) -> Path:
        return self.meta.filepath

    @property
    def mag(self) -> Optional[float]:
        return self.meta.mag

    @property
    def voltage_kV(self) -> Optional[float]:
        return self.meta.voltage_kV

    # ── factory methods ───────────────────────────────────────────────────────

    def apply_binning(self, factor: int) -> None:
        """Bin the image (and every movie frame) and rescale the pixel size.

        The two must happen together. Binning without rescaling leaves every
        distance, area and diameter wrong by the bin factor while still carrying
        units, which is worse than having no calibration at all -- so this is the
        only supported way to bin a loaded image, and it is a method on the
        object that owns both the array and its calibration.
        """
        from acorn.core.binning import bin_frames, bin_image, validate_factor

        factor = validate_factor(factor)
        if factor == 1 or self.raw is None:
            return

        native = self.meta.pixel_size
        result = bin_image(self.raw, factor, native)
        self.raw = result.data

        if self._frames is not None:
            self._frames = bin_frames(self._frames, factor, native).data

        self.meta.native_pixel_size = native
        self.meta.pixel_size = result.pixel_size_nm
        self.meta.bin_factor = factor
        self.meta.binning_cropped_px = result.cropped_px
        self.meta.shape = self.raw.shape

    @classmethod
    def from_file(cls, filepath: str | Path, bin_factor: int = 1) -> "DM4Image":
        """Load any supported format. Dispatches by file extension.

        `bin_factor` bins for ANALYSIS, not for display: the returned array is
        what detectors and measurements will see, and `meta.pixel_size` is
        already scaled to match.
        """
        p = Path(str(filepath).strip().strip('"').strip("'"))
        ext = p.suffix.lower()
        obj = cls()
        if ext in DM4_EXTS:
            obj._load_dm4(p)
        elif ext in TIFF_EXTS:
            obj._load_tiff(p)
        elif ext in MRC_EXTS:
            obj._load_mrc(p)
        elif ext in EMD_EXTS | HDF5_EXTS:
            obj._load_hdf5(p)
        elif ext in IMAGE_EXTS:
            obj._load_image(p)
        else:
            raise ValueError(
                f"Unsupported file format: {ext!r}\n"
                f"Supported: {sorted(ALL_EXTS)}"
            )
        obj.meta.native_pixel_size = obj.meta.pixel_size
        obj.apply_binning(bin_factor)
        return obj

    @classmethod
    def open(cls, filepath: str | Path, bin_factor: int = 1) -> "DM4Image":
        """Alias for from_file; use with `with` statement."""
        return cls.from_file(filepath, bin_factor=bin_factor)

    def __enter__(self) -> "DM4Image":
        return self

    def __exit__(self, *_) -> None:
        pass

    # ── format loaders ────────────────────────────────────────────────────────

    def _load_dm4(self, filepath: Path) -> None:
        import ncempy.io as nio
        dm = nio.dm.fileDM(str(filepath))
        dm.parseHeader()
        dataset = dm.getDataset(0)
        data = dataset["data"]
        if data.ndim == 3 and data.shape[0] > 1:
            self._frames = data.astype(np.float32)
            self.raw = self._frames.mean(axis=0)
        elif data.ndim == 3:
            self.raw = data[0].astype(np.float32)   # single-frame stack → 2D, not a movie
        elif data.ndim > 3:
            while data.ndim > 2:
                data = data[0]
            self.raw = data.astype(np.float32)
        else:
            self.raw = data.astype(np.float32)
        self.meta.shape    = self.raw.shape
        self.meta.filepath = filepath
        self.meta.filename = filepath.stem
        self.meta.raw_dtype = str(self.raw.dtype)
        self.meta.all_tags  = dm.allTags

        has_ps = "pixelSize" in dataset
        ps_raw = dataset.get("pixelSize", [1.0, 1.0])
        pu_raw = dataset.get("pixelUnit", ["nm", "nm"])
        ps = float(ps_raw[0] if isinstance(ps_raw, (list, tuple, np.ndarray)) else ps_raw)
        pu = str(pu_raw[0] if isinstance(pu_raw, (list, tuple)) else pu_raw)
        pu_clean = pu.strip().lower().replace("\x00", "").replace(" ", "")
        ps_nm = ps * self.UNIT_TO_NM.get(pu_clean, 1.0)
        self.meta.pixel_unit = "nm"
        # Only flag as header-calibrated when the tag was actually present and valid,
        # so downstream measurements/export can tell a real 1.0 nm/px from "unknown".
        if has_ps and ps_nm > 0:
            self.meta.pixel_size = ps_nm
            self.meta.pixel_size_from_header = True
        else:
            self.meta.pixel_size = 1.0
            self.meta.pixel_size_from_header = False

        for _, v in self._deep_search(dm.allTags, "magnification"):
            try:
                self.meta.mag = float(v); break
            except (TypeError, ValueError):
                pass
        for _, v in self._deep_search(dm.allTags, "voltage"):
            try:
                val = float(v)
                self.meta.voltage_kV = val / 1000 if val > 1000 else val
                break
            except (TypeError, ValueError):
                pass
        try:
            dm.fid.close()
        except Exception:
            pass

    def _load_tiff(self, filepath: Path) -> None:
        import tifffile
        data = tifffile.imread(str(filepath))
        if data.ndim == 3:
            if data.shape[-1] in (3, 4):
                # Color image (H, W, 3/4) — preserve RGB
                rgb = data[..., :3].astype(np.float32)
                mx = float(rgb.max())
                self.raw = rgb / mx if mx > 0 else rgb
            elif data.shape[-1] in (1, 2):
                # Single/dual channel — convert to grayscale
                self.raw = data[..., 0].astype(np.float32)
            elif data.shape[0] > 1:
                # Multi-frame stack (n_frames, H, W)
                self._frames = data.astype(np.float32)
                self.raw = self._frames.mean(axis=0)
            else:
                # Single-frame stack → 2D image, not a movie
                self.raw = data[0].astype(np.float32)
        elif data.ndim > 3:
            while data.ndim > 2:
                data = data[0]
            self.raw = data.astype(np.float32)
        else:
            self.raw = data.astype(np.float32)
        self.meta.shape    = self.raw.shape
        self.meta.filepath = filepath
        self.meta.filename = filepath.stem
        self.meta.raw_dtype = str(data.dtype)

        # Try to read pixel size from ImageJ / OME-TIFF metadata
        try:
            with tifffile.TiffFile(str(filepath)) as tf:
                pages = tf.pages
                if pages:
                    page = pages[0]
                    tags  = {t.name: t.value for t in page.tags.values()}
                    xres  = tags.get("XResolution")
                    unit  = tags.get("ResolutionUnit", 1)
                    if xres and xres[0] != 0:
                        px_per_unit = xres[0] / xres[1] if isinstance(xres, tuple) else float(xres)
                        # TIFF units: 1=no absolute, 2=inch, 3=cm
                        if unit == 2:   # px/inch → nm/px
                            self.meta.pixel_size = 25.4e6 / px_per_unit
                            self.meta.pixel_size_from_header = True
                        elif unit == 3: # px/cm → nm/px
                            self.meta.pixel_size = 1e7 / px_per_unit
                            self.meta.pixel_size_from_header = True
                # ImageJ metadata override — spacing is in ij["unit"], not nm.
                # Convert by unit; treat pixel/unknown units as uncalibrated.
                ij = getattr(tf, "imagej_metadata", None) or {}
                if "spacing" in ij:
                    ij_unit = (str(ij.get("unit", "nm")).strip().lower()
                               .replace("\x00", "").replace(" ", ""))
                    factor = self.UNIT_TO_NM.get(ij_unit)
                    try:
                        spacing = float(ij["spacing"])
                    except (TypeError, ValueError):
                        spacing = 0.0
                    if factor is not None and spacing > 0:
                        self.meta.pixel_size = spacing * factor
                        self.meta.pixel_size_from_header = True
        except Exception:
            pass

    def _load_mrc(self, filepath: Path) -> None:
        try:
            import mrcfile
        except ImportError as exc:
            raise ImportError(
                "mrcfile is required to open MRC files.\n"
                "Install it with: pip install mrcfile"
            ) from exc

        with mrcfile.open(str(filepath), mode="r", permissive=True) as mrc:
            data = mrc.data.copy()
            if data.ndim == 3 and data.shape[0] > 1:
                self._frames = data.astype(np.float32)
                self.raw = self._frames.mean(axis=0)
            elif data.ndim == 3:
                self.raw = data[0].astype(np.float32)   # single-frame stack → 2D, not a movie
            elif data.ndim > 3:
                while data.ndim > 2:
                    data = data[0]
                self.raw = data.astype(np.float32)
            else:
                self.raw = data.astype(np.float32)
            self.meta.shape    = self.raw.shape
            self.meta.filepath = filepath
            self.meta.filename = filepath.stem
            self.meta.raw_dtype = str(data.dtype)

            # Pixel spacing in Ångströms → convert to nm
            voxel = mrc.voxel_size
            if voxel.x > 0:
                self.meta.pixel_size = float(voxel.x) * 0.1   # Å → nm
                self.meta.pixel_size_from_header = True
            else:
                self.meta.pixel_size = 1.0

    def _load_hdf5(self, filepath: Path) -> None:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required to open EMD/HDF5 files.\n"
                "Install it with: pip install h5py"
            ) from exc

        with h5py.File(str(filepath), "r") as h5:
            candidates = _hdf5_image_candidates(h5)
            if not candidates:
                raise ValueError(f"No numeric 2D/stack image dataset found in {filepath}")
            dataset_path = candidates[0][1]
            ds = h5[dataset_path]
            data = ds[()]
            image, frames = _coerce_hdf5_image(data)
            self.raw = image
            self._frames = frames
            self.meta.shape = self.raw.shape
            self.meta.filepath = filepath
            self.meta.filename = filepath.stem
            self.meta.raw_dtype = str(getattr(data, "dtype", self.raw.dtype))
            self.meta.all_tags = {
                "hdf5_dataset": dataset_path,
                "hdf5_attrs": _hdf5_attrs_to_dict(ds.attrs),
                "hdf5_root_attrs": _hdf5_attrs_to_dict(h5.attrs),
            }
            px_nm = _hdf5_pixel_size_nm(h5, ds)
            if px_nm is not None and px_nm > 0:
                self.meta.pixel_size = px_nm
                self.meta.pixel_size_from_header = True
            else:
                self.meta.pixel_size = 1.0
                self.meta.pixel_size_from_header = False

    def _load_image(self, filepath: Path) -> None:
        from PIL import Image as PILImage
        img = PILImage.open(str(filepath))
        if img.mode in ("RGB", "RGBA"):
            # Preserve color: store as (H, W, 3) float32 in [0, 1]
            rgb = img.convert("RGB")
            self.raw = np.array(rgb, dtype=np.float32) / 255.0
        elif img.mode not in ("L", "I", "F"):
            img = img.convert("L")
            self.raw = np.array(img, dtype=np.float32)
        else:
            self.raw = np.array(img, dtype=np.float32)
        self.meta.shape    = self.raw.shape
        self.meta.filepath = filepath
        self.meta.filename = filepath.stem
        self.meta.raw_dtype = str(self.raw.dtype)
        self.meta.pixel_size = 1.0   # no calibration available in PNG/JPG

    # ── utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _deep_search(
        d: object, key: str, results: Optional[list] = None
    ) -> list[tuple]:
        if results is None:
            results = []
        if isinstance(d, dict):
            for k, v in d.items():
                if key.lower() in str(k).lower():
                    results.append((k, v))
                DM4Image._deep_search(v, key, results)
        elif isinstance(d, (list, tuple)):
            for item in d:
                DM4Image._deep_search(item, key, results)
        return results

    def summary(self) -> str:
        lines = [
            "─" * 55,
            f" File      : {self.meta.filename}",
            f" Shape     : {self.meta.shape}",
            f" Pixel size: {self.meta.pixel_size:.6f} nm/px",
        ]
        if self.meta.mag:
            lines.append(f" Mag       : {int(self.meta.mag):,}x")
        if self.meta.voltage_kV:
            lines.append(f" Voltage   : {self.meta.voltage_kV} kV")
        lines += [
            f" Dtype     : {self.meta.raw_dtype}",
            f" Range     : {self.raw.min():.2f} -> {self.raw.max():.2f}",
            "─" * 55,
        ]
        return "\n".join(lines)


def _decode_hdf5_value(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _decode_hdf5_value(value.item())
        return [_decode_hdf5_value(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _hdf5_attrs_to_dict(attrs) -> dict:
    out = {}
    for key, value in attrs.items():
        try:
            out[str(key)] = _decode_hdf5_value(value)
        except Exception:
            out[str(key)] = repr(value)
    return out


def _hdf5_image_candidates(h5) -> list[tuple[float, str]]:
    import h5py

    candidates: list[tuple[float, str]] = []

    def visit(name: str, obj) -> None:
        if not isinstance(obj, h5py.Dataset):
            return
        if obj.dtype.kind not in "uifcb":
            return
        if obj.ndim < 2:
            return
        shape = tuple(int(s) for s in obj.shape)
        non_single = [s for s in shape if s > 1]
        if len(non_single) < 2:
            return
        largest = sorted(non_single)[-2:]
        if min(largest) < 8:
            return
        path = "/" + name
        lower = path.lower()
        score = float(np.prod(largest))
        if path.endswith("/Data") or path.endswith("/data"):
            score += 1_000_000
        if "image" in lower:
            score += 750_000
        if "data/image" in lower or "image/data" in lower:
            score += 750_000
        if "spectrum" in lower or "metadata" in lower or "preview" in lower:
            score -= 500_000
        if obj.ndim == 2:
            score += 250_000
        candidates.append((score, path))

    h5.visititems(visit)
    return sorted(candidates, key=lambda item: item[0], reverse=True)


def _coerce_hdf5_image(data) -> tuple[np.ndarray, Optional[np.ndarray]]:
    arr = np.asarray(data)
    if arr.dtype.kind == "c":
        arr = np.abs(arr)
    arr = np.squeeze(arr)
    if arr.ndim < 2:
        raise ValueError("HDF5 dataset is not image-like after squeezing.")
    if arr.ndim == 2:
        return arr.astype(np.float32), None
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3].astype(np.float32)
        mx = float(rgb.max())
        return (rgb / mx if mx > 0 else rgb), None
    if arr.ndim == 3 and arr.shape[0] > 1:
        frames = arr.astype(np.float32)
        return frames.mean(axis=0), frames

    # Generic HDF5/Velox fallback: preserve the two largest axes as image Y/X
    # and average over navigation/channel axes.
    image_axes = tuple(sorted(np.argsort(arr.shape)[-2:]))
    nav_axes = tuple(i for i in range(arr.ndim) if i not in image_axes)
    moved = np.moveaxis(arr, image_axes, (-2, -1))
    if nav_axes:
        moved = moved.reshape((-1, moved.shape[-2], moved.shape[-1]))
        frames = moved.astype(np.float32)
        return frames.mean(axis=0), frames if frames.shape[0] > 1 else None
    return moved.astype(np.float32), None


def _flatten_hdf5_metadata(h5, ds) -> list[tuple[str, object]]:
    values: list[tuple[str, object]] = []

    def add_attrs(prefix: str, attrs) -> None:
        for key, value in attrs.items():
            values.append((f"{prefix}/{key}".lower(), _decode_hdf5_value(value)))

    add_attrs("root", h5.attrs)
    add_attrs(ds.name, ds.attrs)

    def visit(name: str, obj) -> None:
        add_attrs(name, getattr(obj, "attrs", {}))
        try:
            if getattr(obj, "shape", None) == () and getattr(obj, "dtype", None) is not None:
                if obj.dtype.kind in "SUOf":
                    values.append((name.lower(), _decode_hdf5_value(obj[()])))
        except Exception:
            pass

    h5.visititems(visit)
    return values


def _unit_to_nm(unit: str) -> Optional[float]:
    clean = str(unit).strip().lower().replace("\x00", "").replace(" ", "")
    if clean in DM4Image.UNIT_TO_NM:
        return DM4Image.UNIT_TO_NM[clean]
    if clean in {"meter", "metre"}:
        return 1e9
    if clean in {"angstroms", "å"}:
        return 0.1
    if clean in {"1/m", "m^-1", "pixels", "pixel", "px"}:
        return None
    return None


def _numbers_from_value(value) -> list[float]:
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        nums = []
        for item in value:
            nums.extend(_numbers_from_value(item))
        return nums
    if isinstance(value, str):
        import re

        return [float(x) for x in re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value)]
    return []


def _hdf5_pixel_size_nm(h5, ds) -> Optional[float]:
    metadata = _flatten_hdf5_metadata(h5, ds)
    likely_units = {}
    for key, value in metadata:
        if "unit" in key:
            factor = _unit_to_nm(str(value))
            if factor is not None:
                likely_units[key] = factor

    for key, value in metadata:
        k = key.lower()
        if not any(token in k for token in ("pixelsize", "pixel_size", "pixel size", "scale", "spacing", "calibration")):
            continue
        if any(skip in k for skip in ("offset", "origin", "units", "unit")):
            continue
        nums = [n for n in _numbers_from_value(value) if n > 0]
        if not nums:
            continue
        raw = min(nums)
        unit_factor = None
        for unit_key, factor in likely_units.items():
            if unit_key.rsplit("/", 1)[0] == key.rsplit("/", 1)[0]:
                unit_factor = factor
                break
        if unit_factor is None:
            if raw < 1e-6:
                unit_factor = 1e9  # meters
            elif raw < 0.5:
                unit_factor = 1.0  # often nm in Velox scalar metadata
            else:
                unit_factor = 1.0
        px_nm = raw * unit_factor
        if 0 < px_nm < 1e6:
            return float(px_nm)
    return None


def write_hdf5_image(path: str | Path, image: np.ndarray, *, dataset: str = "data", pixel_size_nm: float | None = None) -> Path:
    """Write an image/stack to a simple HDF5 file Acorn can read back."""
    import h5py

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(out), "w") as h5:
        ds = h5.create_dataset(dataset, data=np.asarray(image), compression="gzip")
        if pixel_size_nm is not None and pixel_size_nm > 0:
            ds.attrs["pixel_size"] = float(pixel_size_nm)
            ds.attrs["pixel_unit"] = "nm"
        h5.attrs["creator"] = "ACORN"
    return out


# ── folder scanning ───────────────────────────────────────────────────────────

def scan_folder(
    folder: str | Path,
    extensions: set[str] | None = None,
) -> list[Path]:
    """
    Return sorted list of supported image files in a folder.

    Parameters
    ----------
    folder     : directory to search (non-recursive)
    extensions : set of lowercase extensions to include, e.g. {'.dm4', '.mrc'}.
                 Defaults to ALL_EXTS (all supported formats).
    """
    exts = extensions if extensions is not None else ALL_EXTS
    paths: list[Path] = []
    try:
        entries = list(Path(folder).iterdir())
    except OSError:
        return []
    for p in entries:
        try:
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
        except OSError:
            continue
    return sorted(paths)
