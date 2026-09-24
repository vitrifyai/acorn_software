"""Headless classical pickers for measurement-bias pilot studies."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage
from skimage.feature import blob_log, peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import remove_small_holes, remove_small_objects
from skimage.segmentation import watershed


@dataclass(frozen=True)
class DatasetItem:
    stem: str
    split: str
    image_path: Path
    truth_path: Path
    pixel_size_nm: float
    image_shape_px: tuple[int, int]
    truth_diameters_nm: tuple[float, ...]


def run_picker(
    dataset_dir: Path,
    output_dir: Path,
    *,
    method: str,
    class_name: str,
    polarity: str = "auto",
    min_diameter_nm: float | None = None,
    max_diameter_nm: float | None = None,
    threshold_rel: float = 0.08,
    max_detections: int = 300,
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    items = _dataset_items(dataset_dir, splits=splits)
    written: list[Path] = []
    for item in items:
        image = _read_image(item.image_path)
        lo_nm, hi_nm = _diameter_window(item, min_diameter_nm, max_diameter_nm)
        if method == "cryoblob-log":
            predictions = _predict_cryoblob_log(
                image_path=item.image_path,
                pixel_size_nm=item.pixel_size_nm,
                min_diameter_nm=lo_nm,
                max_diameter_nm=hi_nm,
                threshold_rel=threshold_rel,
                max_detections=max_detections,
                class_name=class_name,
                polarity=polarity,
            )
        elif method == "log":
            predictions = _predict_log(
                image,
                item.pixel_size_nm,
                min_diameter_nm=lo_nm,
                max_diameter_nm=hi_nm,
                threshold_rel=threshold_rel,
                max_detections=max_detections,
                class_name=class_name,
                polarity=polarity,
            )
        elif method == "threshold-watershed":
            predictions = _predict_threshold_watershed(
                image,
                item.pixel_size_nm,
                min_diameter_nm=lo_nm,
                max_diameter_nm=hi_nm,
                max_detections=max_detections,
                class_name=class_name,
                polarity=polarity,
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        payload = {
            "stem": item.stem,
            "method": method,
            "class_name": class_name,
            "pixel_size_nm": item.pixel_size_nm,
            "source_image": str(item.image_path),
            "predictions": predictions,
        }
        out_path = output_dir / item.split / f"{item.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(out_path)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "method": method,
                "class_name": class_name,
                "source_dataset": str(dataset_dir),
                "n_images": len(written),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return written


def _dataset_items(dataset_dir: Path, *, splits: tuple[str, ...] | None = None) -> list[DatasetItem]:
    truth_paths = sorted((dataset_dir / "truth").glob("*/*.json"))
    if not truth_paths:
        truth_paths = sorted((dataset_dir / "truth").glob("*.json"))
    split_filter = set(splits) if splits is not None else None
    items: list[DatasetItem] = []
    for truth_path in truth_paths:
        payload = json.loads(truth_path.read_text(encoding="utf-8"))
        stem = str(payload.get("stem") or truth_path.stem)
        split = str(payload.get("split") or truth_path.parent.name)
        if split_filter is not None and split not in split_filter:
            continue
        image_path = _image_path(dataset_dir, split, stem)
        pixel_size_nm = float(payload["pixel_size_nm"])
        shape = payload.get("image_shape_px")
        if not shape:
            raise KeyError(f"{truth_path} lacks image_shape_px")
        objects = payload.get("objects", [])
        diameters = tuple(_truth_diameter(obj) for obj in objects)
        items.append(
            DatasetItem(
                stem=stem,
                split=split,
                image_path=image_path,
                truth_path=truth_path,
                pixel_size_nm=pixel_size_nm,
                image_shape_px=(int(shape[0]), int(shape[1])),
                truth_diameters_nm=diameters,
            )
        )
    return items


def _image_path(dataset_dir: Path, split: str, stem: str) -> Path:
    for suffix in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        path = dataset_dir / "images" / split / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"no image found for {stem!r} in {dataset_dir / 'images' / split}")


def _read_image(path: Path) -> np.ndarray:
    arr = tifffile.imread(path)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.float32)
    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _truth_diameter(obj: dict) -> float:
    for key in ("diameter_nm", "ecd_nm", "d_outer_leaflet_nm", "major_nm"):
        if key in obj:
            return float(obj[key])
    raise KeyError("truth object has no recognized diameter field")


def _diameter_window(
    item: DatasetItem,
    min_diameter_nm: float | None,
    max_diameter_nm: float | None,
) -> tuple[float, float]:
    values = np.asarray(item.truth_diameters_nm, dtype=float)
    if values.size:
        inferred_lo = max(item.pixel_size_nm * 3.0, float(np.percentile(values, 5)) * 0.55)
        inferred_hi = max(inferred_lo + item.pixel_size_nm, float(np.percentile(values, 95)) * 1.55)
    else:
        inferred_lo = item.pixel_size_nm * 3.0
        inferred_hi = item.pixel_size_nm * 80.0
    return (
        float(min_diameter_nm) if min_diameter_nm is not None else inferred_lo,
        float(max_diameter_nm) if max_diameter_nm is not None else inferred_hi,
    )


def _predict_cryoblob_log(
    *,
    image_path: Path,
    pixel_size_nm: float,
    min_diameter_nm: float,
    max_diameter_nm: float,
    threshold_rel: float,
    max_detections: int,
    class_name: str,
    polarity: str,
) -> list[dict]:
    from acorn_cryoblob.thread import _process_single_file

    min_sigma_px, max_sigma_px = _diameter_nm_to_sigma_px(min_diameter_nm, max_diameter_nm, pixel_size_nm)
    rows = _process_single_file(
        str(image_path),
        detection_mode="log",
        use_watershed=False,
        contrast_polarity="auto" if polarity == "auto" else polarity,
        pixel_size_nm=pixel_size_nm,
        run_mode="final",
        blob_downscale=1.0,
        min_sigma=min_sigma_px,
        max_sigma=max_sigma_px,
        blob_step=1.0,
        threshold_rel=threshold_rel,
        max_detections=max_detections,
        refine_sizes=True,
        size_scale=1.0,
        ridge_threshold=0.006,
        ridge_scales=20,
        min_marker_distance=4.0,
        use_ridge_detection=False,
        stream_large_files=False,
        exponential=False,
        logarizer=False,
        gblur=2,
        background=0,
        apply_filter=0,
        cache_results=False,
    )
    predictions = []
    for i, row in enumerate(rows):
        predictions.append(
            {
                "id": i + 1,
                "class_name": class_name,
                "cx_px": float(row["Center X (nm)"]) / pixel_size_nm,
                "cy_px": float(row["Center Y (nm)"]) / pixel_size_nm,
                "diameter_nm": float(row["Size (nm)"]),
                "score": None,
                "source": "cryoblob-log",
            }
        )
    return predictions


def _predict_log(
    image: np.ndarray,
    pixel_size_nm: float,
    *,
    min_diameter_nm: float,
    max_diameter_nm: float,
    threshold_rel: float,
    max_detections: int,
    class_name: str,
    polarity: str,
) -> list[dict]:
    min_sigma_px, max_sigma_px = _diameter_nm_to_sigma_px(min_diameter_nm, max_diameter_nm, pixel_size_nm)
    candidates = []
    for use_polarity in _polarities(polarity):
        work = _polarity_image(image, use_polarity)
        blobs = blob_log(
            gaussian(work, sigma=1.0, preserve_range=True),
            min_sigma=min_sigma_px,
            max_sigma=max_sigma_px,
            num_sigma=max(5, int(math.ceil(max_sigma_px - min_sigma_px)) + 1),
            threshold=0.0,
            threshold_rel=threshold_rel,
            overlap=0.45,
            exclude_border=False,
        )
        predictions = []
        for i, (cy, cx, sigma) in enumerate(blobs[:max_detections]):
            predictions.append(
                {
                    "id": i + 1,
                    "class_name": class_name,
                    "cx_px": float(cx),
                    "cy_px": float(cy),
                    "diameter_nm": float(2.0 * math.sqrt(2.0) * sigma * pixel_size_nm),
                    "score": None,
                    "source": f"log-{use_polarity}",
                }
            )
        candidates.append(predictions)
    return max(candidates, key=len) if candidates else []


def _predict_threshold_watershed(
    image: np.ndarray,
    pixel_size_nm: float,
    *,
    min_diameter_nm: float,
    max_diameter_nm: float,
    max_detections: int,
    class_name: str,
    polarity: str,
) -> list[dict]:
    min_area_px = math.pi * (min_diameter_nm / (2.0 * pixel_size_nm)) ** 2
    max_area_px = math.pi * (max_diameter_nm / (2.0 * pixel_size_nm)) ** 2
    candidates = []
    for use_polarity in _polarities(polarity):
        work = gaussian(_polarity_image(image, use_polarity), sigma=1.0, preserve_range=True)
        try:
            thresh = threshold_otsu(work)
        except ValueError:
            candidates.append([])
            continue
        mask = work > thresh
        mask = remove_small_objects(mask, max_size=max(6, int(min_area_px * 0.12)))
        mask = remove_small_holes(mask, max_size=max(8, int(min_area_px * 0.20)))
        distance = ndimage.distance_transform_edt(mask)
        min_distance = max(2, int((min_diameter_nm / pixel_size_nm) * 0.20))
        peaks = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=mask,
            exclude_border=False,
        )
        markers = np.zeros(mask.shape, dtype=np.int32)
        for i, (y, x) in enumerate(peaks, start=1):
            markers[int(y), int(x)] = i
        if markers.max() == 0:
            markers, _ = ndimage.label(mask)
        labels = watershed(-distance, markers, mask=mask)
        predictions = []
        for prop in regionprops(labels):
            if prop.area < min_area_px * 0.20 or prop.area > max_area_px * 2.25:
                continue
            cy, cx = prop.centroid
            diameter_nm = 2.0 * math.sqrt(float(prop.area) / math.pi) * pixel_size_nm
            predictions.append(
                {
                    "id": len(predictions) + 1,
                    "class_name": class_name,
                    "cx_px": float(cx),
                    "cy_px": float(cy),
                    "diameter_nm": float(diameter_nm),
                    "score": None,
                    "source": f"threshold-watershed-{use_polarity}",
                }
            )
            if len(predictions) >= max_detections:
                break
        candidates.append(predictions)
    return max(candidates, key=len) if candidates else []


def _diameter_nm_to_sigma_px(
    min_diameter_nm: float,
    max_diameter_nm: float,
    pixel_size_nm: float,
) -> tuple[float, float]:
    min_sigma = max(0.8, min_diameter_nm / (2.0 * math.sqrt(2.0) * pixel_size_nm))
    max_sigma = max(min_sigma + 0.1, max_diameter_nm / (2.0 * math.sqrt(2.0) * pixel_size_nm))
    return min_sigma, max_sigma


def _polarities(polarity: str) -> tuple[str, ...]:
    if polarity == "auto":
        return ("dark", "light")
    if polarity not in {"dark", "light"}:
        raise ValueError("polarity must be auto, dark, or light")
    return (polarity,)


def _polarity_image(image: np.ndarray, polarity: str) -> np.ndarray:
    if polarity == "dark":
        return 1.0 - image
    return image


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run classical object pickers on an exact-truth simulator dataset.")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--method", choices=("cryoblob-log", "log", "threshold-watershed"), required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--polarity", choices=("auto", "dark", "light"), default="auto")
    parser.add_argument("--min-diameter-nm", type=float, default=None)
    parser.add_argument("--max-diameter-nm", type=float, default=None)
    parser.add_argument("--threshold-rel", type=float, default=0.08)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument("--splits", nargs="+", default=None)
    args = parser.parse_args(argv)

    paths = run_picker(
        args.dataset,
        args.output,
        method=args.method,
        class_name=args.class_name,
        polarity=args.polarity,
        min_diameter_nm=args.min_diameter_nm,
        max_diameter_nm=args.max_diameter_nm,
        threshold_rel=args.threshold_rel,
        max_detections=args.max_detections,
        splits=None if args.splits is None else tuple(args.splits),
    )
    print(f"Wrote {len(paths)} {args.method} prediction files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
