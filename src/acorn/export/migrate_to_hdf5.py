"""
Migrate a legacy ACORN dataset (loose PNG files) to HDF5 format.

Usage
-----
Python:
    from acorn.export.migrate_to_hdf5 import migrate_dataset
    migrate_dataset("/path/to/training_data", delete_originals=False)

CLI:
    acorn migrate-to-hdf5 --dataset-dir /path/to/training_data
    acorn migrate-to-hdf5 --dataset-dir /path/to/training_data --delete-originals
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path


def migrate_dataset(
    dataset_dir: str | Path,
    delete_originals: bool = False,
    progress_callback=None,
) -> dict:
    """
    Convert a legacy PNG-based ACORN dataset to HDF5 format in-place.

    Reads images/ and masks/ PNG files, writes them into dataset.h5, then
    updates annotations.json to use "hdf5:{image_id}" file_name keys.

    Parameters
    ----------
    dataset_dir       : root of the training dataset
    delete_originals  : if True, remove images/, masks/, prompts/ after migration
    progress_callback : optional callable(message: str) for progress updates

    Returns
    -------
    dict with keys: n_images, n_masks, n_skipped
    """
    import h5py
    import numpy as np
    from PIL import Image as PILImage

    dataset_dir = Path(dataset_dir)
    ann_path  = dataset_dir / "annotations.json"
    h5_path   = dataset_dir / "dataset.h5"

    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found in {dataset_dir}")

    if h5_path.exists():
        raise FileExistsError(
            f"dataset.h5 already exists in {dataset_dir}. "
            "Delete it manually if you want to re-migrate."
        )

    coco = json.loads(ann_path.read_text())
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # Build ann index: image_id → [ann, ...]
    ann_by_image: dict[int, list] = {}
    for ann in annotations:
        ann_by_image.setdefault(ann["image_id"], []).append(ann)

    n_images  = 0
    n_masks   = 0
    n_skipped = 0

    with h5py.File(h5_path, "w") as h5:
        images_grp  = h5.require_group("images")
        masks_grp   = h5.require_group("masks")
        prompts_grp = h5.require_group("prompts")

        for idx, im in enumerate(images):
            img_id  = im["id"]
            img_key = str(img_id)

            if progress_callback:
                progress_callback(f"Migrating image {idx + 1}/{len(images)}…")

            # ── tile image ────────────────────────────────────────────────────
            img_path = dataset_dir / im["file_name"]
            if not img_path.exists():
                n_skipped += 1
                continue

            arr = np.array(PILImage.open(str(img_path)).convert("L"))
            ds = images_grp.create_dataset(
                img_key, data=arr,
                compression="gzip", compression_opts=4, shuffle=True,
            )
            ds.attrs["width"]          = im.get("width", arr.shape[1])
            ds.attrs["height"]         = im.get("height", arr.shape[0])
            ds.attrs["pixel_size_nm"]  = im.get("pixel_size_nm", 1.0)
            ds.attrs["tile_offset"]    = im.get("tile_offset", [0, 0])
            ds.attrs["aug"]            = im.get("aug", "orig")
            ds.attrs["source_image_id"] = im.get("source_image_id", 0)
            ds.attrs["contrast_params"] = json.dumps(im.get("contrast_params", {}))

            # Update file_name in COCO record
            im["file_name"] = f"hdf5:{img_id}"
            n_images += 1

            # ── instance masks ────────────────────────────────────────────────
            img_masks_grp = masks_grp.require_group(img_key)
            for ann in ann_by_image.get(img_id, []):
                ann_id = ann["id"]
                # Legacy mask path: masks/{entry_stem}/{local_id:03d}_{label}.png
                # We reconstruct the stem from the original file_name
                # The entry_stem is the PNG filename without extension
                old_stem = Path(img_path).stem   # e.g. "0001_stem_t0000_orig"
                label    = ann.get("label", "Unlabelled").replace(" ", "_")
                # Find the mask file — try common naming patterns
                mask_file = None
                inst_dir = dataset_dir / "masks" / old_stem
                if inst_dir.exists():
                    # Find any PNG whose name contains the ann_id or label
                    candidates = list(inst_dir.glob("*.png"))
                    # Match by position: annotation index within this image
                    img_anns = ann_by_image.get(img_id, [])
                    local_id = next(
                        (i + 1 for i, a in enumerate(img_anns) if a["id"] == ann_id),
                        None,
                    )
                    if local_id is not None:
                        for c in candidates:
                            if c.name.startswith(f"{local_id:03d}_"):
                                mask_file = c
                                break

                if mask_file and mask_file.exists():
                    marr = np.array(PILImage.open(str(mask_file)).convert("L"))
                    mds = img_masks_grp.create_dataset(
                        str(ann_id), data=marr,
                        compression="gzip", compression_opts=6,
                    )
                    mds.attrs["label"]       = ann.get("label", "Unlabelled")
                    mds.attrs["category_id"] = ann.get("category_id", 1)
                    n_masks += 1

            # ── prompts ───────────────────────────────────────────────────────
            prompt_file = dataset_dir / "prompts" / f"{old_stem}.json"
            if prompt_file.exists():
                prompt_data = prompt_file.read_text()
                prompts_grp.create_dataset(img_key, data=prompt_data)

    # Write updated annotations.json
    ann_path.write_text(json.dumps(coco, indent=2))

    if delete_originals:
        for d in ("images", "masks", "prompts"):
            p = dataset_dir / d
            if p.exists():
                shutil.rmtree(p)

    return {"n_images": n_images, "n_masks": n_masks, "n_skipped": n_skipped}
