"""
Dataset finalizer: session-aware train/val/test split + statistics.

Usage
-----
from acorn.export.dataset_finalizer import finalize_dataset

result = finalize_dataset(
    "/path/to/training_data",
    val_frac=0.1,
    test_frac=0.1,
    seed=42,
)
print(result["stats_str"])

Outputs
-------
<dataset_dir>/
  splits/
    train.json      COCO file containing only train-split images + annotations
    val.json
    test.json
    split_map.json  mapping: image_id -> split name
  dataset_stats.json
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict | list | None:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return None
    return None


def _coco_subset(coco: dict, image_ids: set) -> dict:
    """Return a new COCO dict keeping only the given image_ids."""
    images      = [im for im in coco["images"] if im["id"] in image_ids]
    annotations = [a  for a  in coco["annotations"] if a["image_id"] in image_ids]
    return {
        "info":        coco.get("info", {}),
        "images":      images,
        "annotations": annotations,
        "categories":  coco.get("categories", []),
    }


# ── main function ─────────────────────────────────────────────────────────────

def finalize_dataset(
    dataset_dir: str | Path,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = 42,
    explicit_splits: dict[str, str] | None = None,
) -> dict:
    """
    Create train/val/test splits that keep all tiles/augmentations from the
    same **source image** in the same split (prevents data leakage).

    Parameters
    ----------
    dataset_dir     : root of the training dataset produced by training_exporter
    val_frac        : fraction of source images held out for validation
                      (applied only to images not covered by explicit_splits)
    test_frac       : fraction held out for test (0 = no test split)
    seed            : random seed for reproducible splits
    explicit_splits : optional dict mapping source file path (or stem) to
                      "Train" | "Validation" | "Test" | "Exclude".
                      Explicitly assigned images always use that split;
                      any remaining images are randomly assigned by fraction.
                      "Exclude" removes the image from all splits entirely.

    Returns
    -------
    dict with keys:
        split_counts   : { "train": n_images, "val": n, "test": n }
        stats          : full stats dict (also written to dataset_stats.json)
        stats_str      : human-readable summary string
    """
    dataset_dir = Path(dataset_dir)
    splits_dir  = dataset_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    ann_path  = dataset_dir / "annotations.json"
    info_path = dataset_dir / "dataset_info.json"

    coco = _load_json(ann_path)
    info = _load_json(info_path)

    if coco is None:
        raise FileNotFoundError(f"annotations.json not found in {dataset_dir}")

    images      = coco["images"]
    annotations = coco["annotations"]
    categories  = {c["id"]: c["name"] for c in coco.get("categories", [])}

    # ── group COCO image IDs by source_image_id ───────────────────────────────
    source_to_ids: dict[int, list[int]] = defaultdict(list)
    for im in images:
        sid = im.get("source_image_id", im["id"])
        source_to_ids[sid].append(im["id"])

    source_ids = sorted(source_to_ids.keys())
    n_total    = len(source_ids)
    rng = random.Random(seed)
    rng.shuffle(source_ids)

    # ── resolve explicit assignments against dataset_info.json ────────────────
    # Normalise GUI labels ("Validation") to internal keys ("val")
    _label_map = {"train": "train", "Train": "train",
                  "val": "val", "Validation": "val",
                  "test": "test", "Test": "test",
                  "exclude": "exclude", "Exclude": "exclude"}

    explicit_by_source: dict[int, str] = {}  # source_image_id -> split
    if explicit_splits and info:
        # Build lookup: stem and full path -> split
        stem_to_split = {Path(k).stem: _label_map.get(v, "train")
                         for k, v in explicit_splits.items()}
        path_to_split = {k: _label_map.get(v, "train")
                         for k, v in explicit_splits.items()}
        for src in info.get("source_images", []):
            src_file = src.get("source_file", "")
            split = (path_to_split.get(src_file)
                     or stem_to_split.get(Path(src_file).stem))
            if split:
                explicit_by_source[src["source_id"]] = split

    # ── separate explicitly assigned from those left to random split ──────────
    assigned_ids   = []   # (source_id, split) already decided
    unassigned_ids = []   # source_ids to be randomly split

    for sid in source_ids:
        spl = explicit_by_source.get(sid)
        if spl:
            assigned_ids.append((sid, spl))
        else:
            unassigned_ids.append(sid)

    # Random split for unassigned images
    n_u     = len(unassigned_ids)
    n_val   = max(1, round(n_u * val_frac))  if val_frac  > 0 and n_u > 1 else 0
    n_test  = max(1, round(n_u * test_frac)) if test_frac > 0 and n_u > 1 else 0
    n_train = n_u - n_val - n_test

    if n_train < 0:
        n_val  = max(0, n_u - 1)
        n_test = 0
        n_train = n_u - n_val

    # Totals across explicit + random assignments (for stats display)
    n_train_sources = n_train + sum(1 for _, s in assigned_ids if s == "train")
    n_val_sources   = n_val   + sum(1 for _, s in assigned_ids if s == "val")
    n_test_sources  = n_test  + sum(1 for _, s in assigned_ids if s == "test")

    # Check we have at least some train images overall
    n_explicit_train = sum(1 for _, s in assigned_ids if s == "train")
    if n_train + n_explicit_train < 1 and len(source_ids) > 0:
        raise ValueError(
            "No images assigned to Train. Adjust split fractions or assignments."
        )

    # Build split_map: COCO image_id -> split name
    split_map: dict[int, str] = {}

    for sid, spl in assigned_ids:
        if spl == "exclude":
            continue  # excluded images don't appear in any split
        for img_id in source_to_ids[sid]:
            split_map[img_id] = spl

    for i, sid in enumerate(unassigned_ids):
        if i < n_test:
            spl = "test"
        elif i < n_test + n_val:
            spl = "val"
        else:
            spl = "train"
        for img_id in source_to_ids[sid]:
            split_map[img_id] = spl

    # ── write per-split COCO files ────────────────────────────────────────────
    for split_name in ("train", "val", "test"):
        ids_in_split = {img_id for img_id, s in split_map.items() if s == split_name}
        if not ids_in_split:
            continue
        subset = _coco_subset(coco, ids_in_split)
        (splits_dir / f"{split_name}.json").write_text(json.dumps(subset, indent=2))

    (splits_dir / "split_map.json").write_text(
        json.dumps({str(k): v for k, v in split_map.items()}, indent=2)
    )

    # ── compute statistics ────────────────────────────────────────────────────
    ann_by_image: dict[int, list] = defaultdict(list)
    for a in annotations:
        ann_by_image[a["image_id"]].append(a)

    cat_counts: dict[str, int] = defaultdict(int)
    areas: list[float] = []
    inst_per_image: list[int] = []
    images_with_zero = 0

    for im in images:
        anns = ann_by_image[im["id"]]
        n_inst = len(anns)
        inst_per_image.append(n_inst)
        if n_inst == 0:
            images_with_zero += 1
        for a in anns:
            cat_name = categories.get(a["category_id"], "unknown")
            cat_counts[cat_name] += 1
            if "area" in a:
                areas.append(float(a["area"]))

    split_counts = {
        "train": sum(1 for s in split_map.values() if s == "train"),
        "val":   sum(1 for s in split_map.values() if s == "val"),
        "test":  sum(1 for s in split_map.values() if s == "test"),
    }

    stats = {
        "n_source_images":     n_total,
        "n_tiles_augmented":   len(images),
        "n_annotations":       len(annotations),
        "n_categories":        len(categories),
        "categories":          dict(cat_counts),
        "images_with_zero_annotations": images_with_zero,
        "instances_per_image": {
            "mean":   round(float(sum(inst_per_image)) / max(1, len(inst_per_image)), 2),
            "min":    min(inst_per_image) if inst_per_image else 0,
            "max":    max(inst_per_image) if inst_per_image else 0,
        },
        "mask_area_px": {
            "mean": round(float(sum(areas)) / max(1, len(areas)), 1) if areas else 0,
            "min":  round(min(areas), 1) if areas else 0,
            "max":  round(max(areas), 1) if areas else 0,
        },
        "split_counts": split_counts,
        "split_seed":   seed,
    }

    (dataset_dir / "dataset_stats.json").write_text(json.dumps(stats, indent=2))

    # ── human-readable summary ────────────────────────────────────────────────
    lines = [
        f"Dataset: {dataset_dir.name}",
        f"  Source images : {n_total}",
        f"  Tiles + augs  : {len(images)}",
        f"  Annotations   : {len(annotations)}",
        "  Category counts:",
    ]
    for cat, cnt in sorted(cat_counts.items()):
        lines.append(f"    {cat:<20s} {cnt}")
    lines += [
        f"  Instances/image : mean={stats['instances_per_image']['mean']}  "
        f"min={stats['instances_per_image']['min']}  "
        f"max={stats['instances_per_image']['max']}",
        f"  Empty tiles     : {images_with_zero}",
        f"  Split (source images): "
        f"train={n_train_sources}  val={n_val_sources}  test={n_test_sources}",
        f"  Split (tiles):   "
        f"train={split_counts['train']}  val={split_counts['val']}  test={split_counts['test']}",
    ]
    stats_str = "\n".join(lines)

    return {
        "split_counts": split_counts,
        "stats":        stats,
        "stats_str":    stats_str,
    }
