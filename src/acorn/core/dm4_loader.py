"""
Multi-format image loader for cryo-EM data.

Supported formats
-----------------
  .dm4            — Gatan Digital Micrograph (ncempy)
  .tif / .tiff    — TIFF including multi-page (tifffile)
  .mrc / .mrcs    — MRC2014 cryo-EM standard (mrcfile)
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
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
STAR_EXTS  = {".star"}

ALL_EXTS       = DM4_EXTS | TIFF_EXTS | MRC_EXTS | IMAGE_EXTS
IMAGE_ONLY_EXTS = ALL_EXTS  # alias — STAR excluded from image scanning

# Formats that are electron-microscopy data — bandpass contrast is the right default
EM_EXTS = DM4_EXTS | MRC_EXTS   # {".dm4", ".mrc", ".mrcs"}
DEFAULT_EM_CONTRAST = "bandpass"  # used by _switch_to and contrast-panel init


# ── metadata dataclass ────────────────────────────────────────────────────────

@dataclass
class DM4Metadata:
    pixel_size: float = 1.0      # nm/px
    pixel_unit: str = "nm"
    pixel_size_from_header: bool = False  # True when read from file header
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

    @classmethod
    def from_file(cls, filepath: str | Path) -> "DM4Image":
        """Load any supported format. Dispatches by file extension."""
        p = Path(str(filepath).strip().strip('"').strip("'"))
        ext = p.suffix.lower()
        obj = cls()
        if ext in DM4_EXTS:
            obj._load_dm4(p)
        elif ext in TIFF_EXTS:
            obj._load_tiff(p)
        elif ext in MRC_EXTS:
            obj._load_mrc(p)
        elif ext in IMAGE_EXTS:
            obj._load_image(p)
        else:
            raise ValueError(
                f"Unsupported file format: {ext!r}\n"
                f"Supported: {sorted(ALL_EXTS)}"
            )
        return obj

    @classmethod
    def open(cls, filepath: str | Path) -> "DM4Image":
        """Alias for from_file; use with `with` statement."""
        return cls.from_file(filepath)

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
        if data.ndim == 3:
            self._frames = data.astype(np.float32)
            self.raw = self._frames.mean(axis=0)
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

        ps_raw = dataset.get("pixelSize", [1.0, 1.0])
        pu_raw = dataset.get("pixelUnit", ["nm", "nm"])
        ps = float(ps_raw[0] if isinstance(ps_raw, (list, tuple, np.ndarray)) else ps_raw)
        pu = str(pu_raw[0] if isinstance(pu_raw, (list, tuple)) else pu_raw)
        pu_clean = pu.strip().lower().replace("\x00", "").replace(" ", "")
        self.meta.pixel_size = ps * self.UNIT_TO_NM.get(pu_clean, 1.0)
        self.meta.pixel_unit = "nm"
        self.meta.pixel_size_from_header = True

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
            else:
                # Multi-frame stack (n_frames, H, W)
                self._frames = data.astype(np.float32)
                self.raw = self._frames.mean(axis=0)
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
                # ImageJ metadata override
                ij = getattr(tf, "imagej_metadata", None) or {}
                if "spacing" in ij:
                    self.meta.pixel_size = float(ij["spacing"])
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
            if data.ndim == 3:
                self._frames = data.astype(np.float32)
                self.raw = self._frames.mean(axis=0)
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
