"""
SAM-compatible training dataset exporter.

Incrementally builds a dataset from annotated cryo-EM images.
Each ``add_image`` call tiles the image, optionally applies 8-orientation
augmentation, and appends everything to the cumulative COCO manifest and
HDF5 tile store.

Output layout
-------------
<dataset_dir>/
  dataset.h5             single HDF5 file containing all tile images, masks, prompts
  annotations.json       cumulative COCO manifest (RLE + polygon)
  dataset_info.json      pixel sizes, norm params, source files

HDF5 structure
--------------
  /images/{image_id}     uint8 (H, W) grayscale tile, gzip compressed
      attrs: file_name, width, height, pixel_size_nm, tile_offset (list),
             aug, source_image_id, contrast_params (JSON str)
  /masks/{image_id}/{ann_id}  uint8 (H, W) binary 0/255 mask, gzip compressed
      attrs: label, category_id
  /prompts/{image_id}    scalar JSON string with SAM prompt data

annotations.json schema
-----------------------
  info / images / annotations / categories  (standard COCO)
  images[*].file_name   = "hdf5:{image_id}" — key into dataset.h5 /images group
  images extra fields:  pixel_size_nm, tile_offset [y0, x0], source_image_id, aug
  annotations extra:    label, neg_prompts [[x,y], ...]
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_AUG_WORKERS = min(8, (os.cpu_count() or 4))


def _atomic_write_text(path: Path, text: str) -> None:
    """Write to a temp file in the same dir then os.replace() — never leaves the
    accumulating manifest truncated if a write is interrupted (e.g. NAS drop)."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)

if TYPE_CHECKING:
    from acorn.core.dm4_loader import DM4Image
    from acorn.core.annotations import AnnotationStore
    from acorn.core.contrast import ContrastParams


# ── training configuration ────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """All options that control how one image is turned into training samples."""
    tile_size: int | None = 1024    # px side-length; None = whole image as one tile
    tile_overlap: float = 0.25      # fraction overlap between neighbouring tiles
    augment: bool = True            # generate all 8 rigid orientations per tile
    n_neg_prompts: int = 3          # random negative SAM points per instance
    skip_empty_tiles: bool = True   # discard tiles that contain no mask pixels
    encode_rle: bool = True         # store RLE in COCO (alongside polygon)


# ── default COCO categories ───────────────────────────────────────────────────

_DEFAULT_CATEGORIES = [
    {"id": 1, "name": "Foreground"},
    {"id": 2, "name": "Background"},
    {"id": 3, "name": "Ignore"},
]

_AUG_SUFFIXES = [
    "orig",
    "rot90", "rot180", "rot270",
    "fliplr",
    "fliplr_rot90", "fliplr_rot180", "fliplr_rot270",
]


# ── low-level helpers ─────────────────────────────────────────────────────────

def _cat_id_for(label: str, categories: list[dict]) -> int:
    label = label.strip() or "Unlabelled"
    for c in categories:
        if c["name"].lower() == label.lower():
            return c["id"]
    new_id = max(c["id"] for c in categories) + 1
    categories.append({"id": new_id, "name": label})
    return new_id


def _mask_to_rle(mask: np.ndarray) -> dict:
    """COCO uncompressed RLE from a binary mask (Fortran/column-major order)."""
    h, w = mask.shape
    flat = (mask > 0).flatten(order="F").view(np.uint8)
    if flat.size == 0:
        return {"counts": [0], "size": [h, w]}
    counts: list[int] = []
    current = 0
    run = 0
    for v in flat:
        if v == current:
            run += 1
        else:
            counts.append(run)
            run = 1
            current = v
    counts.append(run)
    if flat[0] > 0:          # COCO always starts with the 0-run length
        counts = [0] + counts
    return {"counts": counts, "size": [h, w]}


def _mask_to_polygon(mask: np.ndarray) -> list:
    """Largest external contour of a binary mask as [[x, y], ...] in mask-local px.

    Derived from the (already-augmented) mask so the polygon always matches the
    mask/RLE for every orientation — unlike transforming the original vertices.
    """
    try:
        import cv2
        m8 = (mask > 0).astype(np.uint8)
        contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        c = max(contours, key=cv2.contourArea)
        eps = 0.01 * cv2.arcLength(c, True)
        pts = cv2.approxPolyDP(c, eps, True).reshape(-1, 2)
        if len(pts) < 3:
            return []
        return [[int(x), int(y)] for x, y in pts]
    except Exception:
        return []


def _polygon_to_bbox(vertices: list) -> list[float]:
    """[x, y, w, h] from polygon vertex list."""
    xs = [v[0] for v in vertices]
    ys = [v[1] for v in vertices]
    x, y = min(xs), min(ys)
    return [x, y, max(xs) - x, max(ys) - y]


def _polygon_area(vertices: list) -> float:
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _mask_centroid(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return [w / 2.0, h / 2.0]
    return [float(xs.mean()), float(ys.mean())]


def _mask_bbox(mask: np.ndarray) -> list[float]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()),
            float(xs.max() - xs.min()), float(ys.max() - ys.min())]


def _compute_aug_instance_data(
    aug_img: np.ndarray,
    aug_masks: list[np.ndarray],
    aug_suffix: str,
    active: list[int],
    roi_meta: list[dict],
    config: "TrainingConfig",
    rng_seed: int,
    y0: int,
    x0: int,
) -> dict:
    """
    Pre-compute all CPU-intensive instance stats for one augmented tile.
    Pure numpy — no HDF5 writes, fully thread-safe.
    Returns a dict ready for sequential HDF5 writing.
    """
    th, tw = aug_img.shape[:2]
    rng = np.random.default_rng(rng_seed)
    aug_masks_active = [aug_masks[i] for i in active]
    instances: list[dict] = []

    for orig_idx in active:
        m    = aug_masks[orig_idx]
        meta = roi_meta[orig_idx]
        cx, cy  = _mask_centroid(m)
        bbox    = _mask_bbox(m)
        rle     = _mask_to_rle(m) if config.encode_rle else None
        neg_pts = _negative_prompts(aug_masks_active, config.n_neg_prompts, rng, th, tw)
        # Polygon from the augmented mask so it matches the mask/RLE for every
        # orientation (transforming the original vertices would misalign flips/rots).
        verts_local = _mask_to_polygon(m)
        instances.append({
            "label":       meta["label"],
            "mask":        m,
            "cx":          cx,
            "cy":          cy,
            "bbox":        bbox,
            "rle":         rle,
            "neg_pts":     neg_pts,
            "verts_local": verts_local,
        })

    return {
        "aug_img":   aug_img,
        "aug_suffix": aug_suffix,
        "th": th, "tw": tw,
        "y0": y0, "x0": x0,
        "instances": instances,
    }


def _negative_prompts(
    all_masks: list[np.ndarray],
    n: int,
    rng: np.random.Generator,
    h: int,
    w: int,
) -> list[list[float]]:
    """Sample n random points that fall outside every instance mask."""
    if n == 0:
        return []
    combined = np.zeros((h, w), dtype=bool)
    for m in all_masks:
        combined |= (m > 0)
    outside_yx = np.argwhere(~combined)
    if len(outside_yx) == 0:
        return []
    chosen = outside_yx[rng.choice(len(outside_yx), size=min(n, len(outside_yx)), replace=False)]
    return [[float(c), float(r)] for r, c in chosen]


# ── tiling ────────────────────────────────────────────────────────────────────

def _extract_tiles(
    img8: np.ndarray,
    inst_masks: list[np.ndarray],
    tile_size: int,
    overlap: float,
) -> list[dict]:
    """
    Slice a (H, W) image and per-instance masks into overlapping square tiles.
    If the image is smaller than tile_size in either dimension, returns one
    tile (the full image, zero-padded to tile_size).
    """
    h, w = img8.shape[:2]

    # Pad if image is smaller than tile_size
    if h < tile_size or w < tile_size:
        ph = max(0, tile_size - h)
        pw = max(0, tile_size - w)
        img8 = np.pad(img8, ((0, ph), (0, pw)), mode="reflect")
        inst_masks = [np.pad(m, ((0, ph), (0, pw)), mode="constant") for m in inst_masks]
        h, w = img8.shape[:2]

    stride = max(1, int(tile_size * (1 - overlap)))
    tiles: list[dict] = []
    tile_idx = 0
    y0 = 0
    while True:
        y0 = min(y0, h - tile_size)
        x0 = 0
        while True:
            x0 = min(x0, w - tile_size)
            tiles.append({
                "img":   img8[y0:y0 + tile_size, x0:x0 + tile_size].copy(),
                "masks": [m[y0:y0 + tile_size, x0:x0 + tile_size].copy() for m in inst_masks],
                "y0": y0, "x0": x0,
                "tile_idx": tile_idx,
            })
            tile_idx += 1
            if x0 + tile_size >= w:
                break
            x0 += stride
        if y0 + tile_size >= h:
            break
        y0 += stride

    return tiles


# ── augmentation ──────────────────────────────────────────────────────────────

def _augment_tile(img: np.ndarray, masks: list[np.ndarray]) -> list[tuple]:
    """
    Return 8 (or 1 if no augmentation) rigid-body orientations.
    Each entry: (aug_img, aug_masks, suffix_str)
    """
    results = []
    for flip in [False, True]:
        base_img = np.fliplr(img) if flip else img
        base_masks = [np.fliplr(m) if flip else m for m in masks]
        for k in range(4):
            aug_img   = np.rot90(base_img, k=k)
            aug_masks = [np.rot90(m, k=k) for m in base_masks]
            flip_str  = "fliplr_" if flip else ""
            rot_str   = "" if k == 0 else f"rot{k * 90}"
            suffix    = (flip_str + rot_str).strip("_") or "orig"
            results.append((aug_img, aug_masks, suffix))
    return results


# ── COCO / dataset I/O ────────────────────────────────────────────────────────

def _load_coco(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {
        "info":        {"description": "ACORN training dataset", "version": "1.0"},
        "images":      [],
        "annotations": [],
        "categories":  list(_DEFAULT_CATEGORIES),
    }


def _load_info(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {"source_images": []}


# ── main API ──────────────────────────────────────────────────────────────────

def add_image(
    dataset_dir: str | Path,
    dm4img: "DM4Image",
    store: "AnnotationStore",
    params: "ContrastParams",
    config: TrainingConfig | None = None,
    progress_callback=None,
) -> dict:
    """
    Tile, augment, and export one annotated image into the training dataset.

    All tile images, instance masks, and SAM prompts are written into a single
    HDF5 file (dataset.h5) — one file write per add_image call, NAS-friendly.

    Returns a summary dict:
        source_stem, n_tiles, n_augmented, n_instances_total, n_skipped_tiles
    """
    import h5py
    from skimage.draw import polygon as skpoly
    from acorn.core.contrast import apply_contrast

    if config is None:
        config = TrainingConfig()

    rng = np.random.default_rng()

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    h5_path   = dataset_dir / "dataset.h5"
    ann_path  = dataset_dir / "annotations.json"
    info_path = dataset_dir / "dataset_info.json"
    coco = _load_coco(ann_path)
    info = _load_info(info_path)
    categories = coco["categories"]

    # ── source image id (groups all tiles/augs from this raw file) ────────────
    source_image_id = (max((s.get("source_id", 0) for s in info["source_images"]), default=0)) + 1

    h, w = dm4img.shape[:2]

    # ── normalize to 8-bit ────────────────────────────────────────────────────
    norm = apply_contrast(dm4img.raw, params)
    img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)

    # ── rasterise all ROI masks ───────────────────────────────────────────────
    rois = [a for a in store if getattr(a, "type", None) == "roi"]
    inst_masks: list[np.ndarray] = []
    roi_meta: list[dict] = []

    for roi in rois:
        if len(roi.vertices) < 3:
            continue
        verts  = np.array(roi.vertices)
        binary = np.zeros((h, w), dtype=np.uint8)
        rr, cc = skpoly(verts[:, 1], verts[:, 0], shape=(h, w))
        binary[rr, cc] = 255
        inst_masks.append(binary)
        roi_meta.append({
            "label":    roi.label.strip() if roi.label and roi.label.strip() else "Unlabelled",
            "vertices": [[float(x), float(y)] for x, y in roi.vertices],
        })

    # ── tile ──────────────────────────────────────────────────────────────────
    if config.tile_size is not None:
        tiles = _extract_tiles(img8, inst_masks, config.tile_size, config.tile_overlap)
    else:
        tiles = [{"img": img8, "masks": list(inst_masks), "y0": 0, "x0": 0, "tile_idx": 0}]

    n_skipped = 0
    n_written  = 0
    n_inst_total = 0

    next_img_id = (max((im["id"] for im in coco["images"]), default=0)) + 1
    next_ann_id = (max((a["id"] for a in coco["annotations"]), default=0)) + 1

    n_tiles_total = len(tiles)

    # ── open HDF5 once for the entire add_image call ──────────────────────────
    with h5py.File(h5_path, "a") as h5:
        images_grp  = h5.require_group("images")
        masks_grp   = h5.require_group("masks")
        prompts_grp = h5.require_group("prompts")

        for tile_num, tile in enumerate(tiles, start=1):
            if progress_callback is not None:
                progress_callback(tile_num, n_tiles_total)
            tile_img   = tile["img"]
            tile_masks = tile["masks"]
            y0, x0     = tile["y0"], tile["x0"]
            t_idx      = tile["tile_idx"]

            # Identify which instances have any pixels in this tile
            active = [i for i, m in enumerate(tile_masks) if m.max() > 0]

            if config.skip_empty_tiles and len(active) == 0:
                n_skipped += 1
                continue

            # Generate augmented versions (or just the original)
            if config.augment:
                versions = _augment_tile(tile_img, tile_masks)
            else:
                versions = [(tile_img, tile_masks, "orig")]

            # ── Pre-compute all instance stats in parallel (pure numpy) ───────
            # Each future handles one augmented variant; h5 writes stay sequential.
            base_seed = int(rng.integers(0, 2**31))

            def _submit(idx_aug):
                aug_img_v, aug_masks_v, aug_suffix_v = versions[idx_aug]
                return _compute_aug_instance_data(
                    aug_img_v, aug_masks_v, aug_suffix_v,
                    active, roi_meta, config,
                    rng_seed=base_seed ^ idx_aug,
                    y0=y0, x0=x0,
                )

            n_workers = min(_AUG_WORKERS, len(versions))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                computed_versions = list(pool.map(_submit, range(len(versions))))

            # ── Sequential HDF5 writes using pre-computed data ────────────────
            for cv in computed_versions:
                aug_img    = cv["aug_img"]
                aug_suffix = cv["aug_suffix"]
                th, tw     = cv["th"], cv["tw"]
                instances  = cv["instances"]
                img_key    = str(next_img_id)

                # write tile image to HDF5
                ds = images_grp.create_dataset(
                    img_key, data=aug_img,
                    compression="gzip", compression_opts=4, shuffle=True,
                )
                ds.attrs["width"]           = tw
                ds.attrs["height"]          = th
                ds.attrs["pixel_size_nm"]   = dm4img.pixel_size
                ds.attrs["tile_offset"]     = [y0, x0]
                ds.attrs["aug"]             = aug_suffix
                ds.attrs["source_image_id"] = source_image_id
                ds.attrs["contrast_params"] = json.dumps(asdict(params))

                # COCO image record
                coco["images"].append({
                    "id":              next_img_id,
                    "file_name":       f"hdf5:{next_img_id}",
                    "width":           tw,
                    "height":          th,
                    "pixel_size_nm":   dm4img.pixel_size,
                    "tile_offset":     [y0, x0],
                    "aug":             aug_suffix,
                    "source_image_id": source_image_id,
                    "contrast_params": asdict(params),
                })

                # per-instance masks and COCO annotations
                img_masks_grp    = masks_grp.require_group(img_key)
                prompt_instances: list[dict] = []

                for local_id, inst in enumerate(instances, start=1):
                    label   = inst["label"]
                    m       = inst["mask"]
                    cat_id  = _cat_id_for(label, categories)
                    ann_key = str(next_ann_id)

                    mds = img_masks_grp.create_dataset(
                        ann_key, data=m,
                        compression="gzip", compression_opts=6,
                    )
                    mds.attrs["label"]       = label
                    mds.attrs["category_id"] = cat_id

                    bbox    = inst["bbox"]
                    cx, cy  = inst["cx"], inst["cy"]
                    neg_pts = inst["neg_pts"]
                    verts_local = inst["verts_local"]

                    ann: dict = {
                        "id":          next_ann_id,
                        "image_id":    next_img_id,
                        "category_id": cat_id,
                        "bbox":        [round(v, 2) for v in bbox],
                        "area":        float(int((m > 0).sum())),
                        "iscrowd":     0,
                        "label":       label,
                    }
                    if len(verts_local) >= 3:
                        ann["segmentation"] = [[c for xy in verts_local for c in (float(xy[0]), float(xy[1]))]]
                    else:
                        ann["segmentation"] = []
                    if inst["rle"] is not None:
                        ann["segmentation_rle"] = inst["rle"]

                    coco["annotations"].append(ann)

                    prompt_instances.append({
                        "id":           local_id,
                        "label":        label,
                        "category_id":  cat_id,
                        "mask_key":     f"masks/{img_key}/{ann_key}",
                        "point_prompt": [round(cx, 2), round(cy, 2)],
                        "bbox_prompt":  [round(v, 2) for v in bbox],
                        "neg_prompts":  [[round(p[0], 2), round(p[1], 2)] for p in neg_pts],
                    })

                    next_ann_id  += 1
                    n_inst_total += 1

                # write prompts
                prompt_data = json.dumps({
                    "image_key":     f"images/{img_key}",
                    "pixel_size_nm": dm4img.pixel_size,
                    "instances":     prompt_instances,
                })
                prompts_grp.create_dataset(img_key, data=prompt_data)

                next_img_id += 1
                n_written   += 1

    # ── dataset_info.json ─────────────────────────────────────────────────────
    info["storage_format"]  = "hdf5"
    info["normalization"]   = asdict(params)
    info["training_config"] = asdict(config)
    info["source_images"].append({
        "source_id":     source_image_id,
        "source_file":   str(dm4img.filepath),
        "pixel_size_nm": dm4img.pixel_size,
        "shape":         list(dm4img.shape),
        "n_rois":        len(roi_meta),
        "n_tiles":       len(tiles),
        "n_written":     n_written,
        "n_skipped":     n_skipped,
    })
    _atomic_write_text(info_path, json.dumps(info, indent=2))
    _atomic_write_text(ann_path, json.dumps(coco, indent=2))

    return {
        "source_stem":       dm4img.filename,
        "n_tiles":           len(tiles),
        "n_augmented":       n_written,
        "n_instances_total": n_inst_total,
        "n_skipped_tiles":   n_skipped,
    }
