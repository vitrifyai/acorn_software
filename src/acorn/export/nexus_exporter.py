"""
NeXus HDF5 exporter for ACORN measurement and annotation data.

Produces files compatible with NeXus readers (h5web, nexpy, pynxtools) and
CNMS/ORNL database ingestion workflows.  Uses h5py only — no extra dependencies.

File structure
--------------
/ [NXroot]
└── entry/ [NXentry]
    ├── instrument/ [NXinstrument]
    ├── sample/     [NXsample]
    ├── images/     [NXcollection]  — one NXentry per source image
    │   └── <stem>/ [NXentry]
    │       ├── data/         [NXdata]       image array + calibrated axes
    │       └── annotations/  [NXcollection] ROI polygons, distances, labels
    └── measurements/ [NXdata]  — particle measurement table
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


# NeXus version string written to every file root
_NEXUS_VERSION = "4.3.0"
_ACORN_CREATOR = "ACORN"


def export_nexus(
    output_path: Path | str,
    image_paths: list[Path],
    ann_states: dict,            # {idx: list[AnyAnnotation]}
    px_overrides: dict,          # {idx: float nm/px}
    image_cache: dict | None = None,   # {idx: DM4Image} — used for image data + meta
    measurements_df=None,        # pandas DataFrame or None
    include_images: bool = True,
    title: str = "ACORN export",
    sample_name: str = "",
    instrument_name: str = "",
) -> Path:
    """
    Write a NeXus-compatible HDF5 file.

    Parameters
    ----------
    output_path : Path
        Destination file (.nxs or .h5).
    image_paths : list[Path]
        All image paths loaded in ACORN.
    ann_states : dict
        Per-image annotation lists keyed by image index.
    px_overrides : dict
        Manual pixel-size overrides keyed by image index (nm/px).
    image_cache : dict or None
        Loaded DM4Image objects; if present their pixel arrays and metadata
        are included.  Images not in the cache are referenced by path only.
    measurements_df : DataFrame or None
        Particle measurement table (from run_particle_analysis /
        export_measurements).
    include_images : bool
        If True, embed the image pixel arrays.  Set False for large datasets
        where you only want the measurements and annotations.
    title : str
        Written to entry/title.
    sample_name : str
        Written to entry/sample/name.
    instrument_name : str
        Written to entry/instrument/name.

    Returns
    -------
    Path
        The written file path.
    """
    import h5py

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with h5py.File(output_path, "w") as f:

        # ── root attributes ───────────────────────────────────────────────────
        f.attrs["NX_class"]       = "NXroot"
        f.attrs["NXS_version"]    = _NEXUS_VERSION
        f.attrs["file_name"]      = str(output_path.name)
        f.attrs["file_time"]      = now
        f.attrs["creator"]        = _ACORN_CREATOR
        f.attrs["HDF5_Version"]   = h5py.version.hdf5_version

        # ── /entry ───────────────────────────────────────────────────────────
        entry = f.create_group("entry")
        entry.attrs["NX_class"]   = "NXentry"
        entry.attrs["definition"] = "NXem_acorn"

        _ds(entry, "title",      title)
        _ds(entry, "start_time", now)

        # ── instrument ───────────────────────────────────────────────────────
        instr = entry.create_group("instrument")
        instr.attrs["NX_class"] = "NXinstrument"
        if instrument_name:
            _ds(instr, "name", instrument_name)

        # Pull voltage/mag from first cached image if available
        if image_cache:
            for img in image_cache.values():
                if img is not None:
                    if getattr(img, "voltage_kV", None):
                        _ds(instr, "voltage_kV", float(img.voltage_kV),
                            units="kV")
                    if getattr(img, "mag", None):
                        _ds(instr, "magnification", float(img.mag))
                    break

        # ── sample ───────────────────────────────────────────────────────────
        sample = entry.create_group("sample")
        sample.attrs["NX_class"] = "NXsample"
        _ds(sample, "name", sample_name or "unknown")

        # ── images ───────────────────────────────────────────────────────────
        images_grp = entry.create_group("images")
        images_grp.attrs["NX_class"] = "NXcollection"

        for idx, img_path in enumerate(image_paths):
            stem = _safe_name(img_path.stem)
            img_entry = images_grp.create_group(stem)
            img_entry.attrs["NX_class"]   = "NXentry"
            img_entry.attrs["source_file"] = str(img_path)

            # Pixel size (override > cached > 1.0)
            px_nm = float(px_overrides.get(idx, 0.0))
            if px_nm <= 0 and image_cache and idx in image_cache:
                loaded = image_cache[idx]
                if loaded and loaded.pixel_size > 0:
                    px_nm = float(loaded.pixel_size)
            if px_nm <= 0:
                px_nm = 1.0
            _ds(img_entry, "pixel_size_nm", px_nm, units="nm", long_name="Calibrated pixel size")

            # Instrument meta from this image
            if image_cache and idx in image_cache:
                loaded = image_cache[idx]
                if loaded:
                    if getattr(loaded, "voltage_kV", None):
                        _ds(img_entry, "voltage_kV", float(loaded.voltage_kV), units="kV")
                    if getattr(loaded, "mag", None):
                        _ds(img_entry, "magnification", float(loaded.mag))

            # Image data + calibrated axes
            if include_images and image_cache and idx in image_cache:
                loaded = image_cache.get(idx)
                if loaded is not None and loaded.raw is not None:
                    raw = loaded.raw.astype(np.float32)
                    h, w = raw.shape[:2]
                    x_ax = np.arange(w, dtype=np.float64) * px_nm   # nm
                    y_ax = np.arange(h, dtype=np.float64) * px_nm

                    data_grp = img_entry.create_group("data")
                    data_grp.attrs["NX_class"] = "NXdata"
                    data_grp.attrs["signal"]   = "image"
                    data_grp.attrs["axes"]     = ["y", "x"]

                    ds_img = data_grp.create_dataset(
                        "image", data=raw,
                        compression="gzip", compression_opts=4,
                    )
                    ds_img.attrs["long_name"] = img_path.name
                    ds_img.attrs["units"]     = "counts"

                    ds_x = data_grp.create_dataset("x", data=x_ax)
                    ds_x.attrs["units"]     = "nm"
                    ds_x.attrs["long_name"] = "x position"
                    ds_x.attrs["axis"]      = 1

                    ds_y = data_grp.create_dataset("y", data=y_ax)
                    ds_y.attrs["units"]     = "nm"
                    ds_y.attrs["long_name"] = "y position"
                    ds_y.attrs["axis"]      = 0

            # Annotations
            anns = ann_states.get(idx) or []
            if anns:
                ann_grp = img_entry.create_group("annotations")
                ann_grp.attrs["NX_class"] = "NXcollection"
                ann_grp.attrs["count"]    = len(anns)
                for ai, ann in enumerate(anns):
                    ag = ann_grp.create_group(str(ai))
                    ag.attrs["NX_class"] = "NXobject"
                    _ds(ag, "type",  getattr(ann, "type",  ""))
                    _ds(ag, "label", getattr(ann, "label", "") or getattr(ann, "text", "") or "")
                    area = getattr(ann, "area_nm2", None)
                    if area is not None:
                        _ds(ag, "area_nm2", float(area), units="nm^2")
                    verts = getattr(ann, "vertices", None)
                    if verts:
                        verts_arr = np.array(verts, dtype=np.float32)
                        dv = ag.create_dataset("vertices_px", data=verts_arr)
                        dv.attrs["units"] = "px"
                    dist = getattr(ann, "distance_nm", None)
                    if dist is not None:
                        _ds(ag, "distance_nm", float(dist), units="nm")

        # ── measurements table ────────────────────────────────────────────────
        if measurements_df is not None and not measurements_df.empty:
            meas_grp = entry.create_group("measurements")
            meas_grp.attrs["NX_class"] = "NXdata"
            meas_grp.attrs["signal"]   = "ecd_nm" if "ecd_nm" in measurements_df.columns else list(measurements_df.columns)[0]
            meas_grp.attrs["axes"]     = ["index"]

            n = len(measurements_df)
            _ds(meas_grp, "index", np.arange(n, dtype=np.int32))

            _col_units = {
                "ecd_nm":        "nm",
                "feret_nm":      "nm",
                "area_nm2":      "nm^2",
                "perimeter_nm":  "nm",
                "bbox_w_nm":     "nm",
                "bbox_h_nm":     "nm",
                "distance_nm":   "nm",
                "pixel_size_nm": "nm",
                "circularity":   "",
                "aspect_ratio":  "",
            }
            for col in measurements_df.columns:
                vals = measurements_df[col]
                try:
                    arr = vals.to_numpy(dtype=np.float64, na_value=np.nan)
                    ds  = meas_grp.create_dataset(col, data=arr)
                except (ValueError, TypeError):
                    # String column
                    arr = vals.fillna("").astype(str).to_numpy()
                    dt  = h5py.string_dtype()
                    ds  = meas_grp.create_dataset(col, data=arr.astype(object), dtype=dt)
                if col in _col_units:
                    ds.attrs["units"] = _col_units[col]
                ds.attrs["long_name"] = col

    return output_path


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ds(group, name: str, value, units: str = "", long_name: str = "") -> None:
    """Create a scalar or string dataset in *group*."""
    import h5py
    if isinstance(value, str):
        ds = group.create_dataset(name, data=value, dtype=h5py.string_dtype())
    elif isinstance(value, (int, float, np.floating, np.integer)):
        ds = group.create_dataset(name, data=np.array(value))
    else:
        ds = group.create_dataset(name, data=value)
    if units:
        ds.attrs["units"] = units
    if long_name:
        ds.attrs["long_name"] = long_name


def _safe_name(s: str) -> str:
    """Turn an arbitrary string into a valid HDF5 group name."""
    import re
    s = re.sub(r"[^\w\-.]", "_", s)
    if s and s[0].isdigit():
        s = "img_" + s
    return s or "image"
