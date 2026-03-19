"""
HuggingFace Hub dataset pusher.

Converts the local ACORN COCO training dataset to a HuggingFace
datasets.Dataset and pushes it to the Hub.  Requires:
    pip install datasets huggingface_hub

Dataset structure on the Hub
-----------------------------
Each row corresponds to one tile/augmentation entry:
    image         : PIL Image (from images/)
    image_id      : int
    file_name     : str
    width, height : int
    pixel_size_nm : float
    tile_offset_y, tile_offset_x : int
    aug           : str
    source_image_id : int
    contrast_params : dict (JSON string)
    annotations   : list[dict]  -- COCO annotation records for this image
    n_instances   : int

Usage
-----
from acorn.export.hub_exporter import push_to_hub

push_to_hub(
    dataset_dir = "/data/training_data",
    repo_id     = "myorg/cryoem-particles",
    token       = "hf_...",      # or set HF_TOKEN env var
    private     = True,
    split       = "train",       # upload split file if finalized, else all images
)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_json(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return None


# ── main function ─────────────────────────────────────────────────────────────

def push_to_hub(
    dataset_dir: str | Path,
    repo_id: str,
    token: Optional[str] = None,
    private: bool = True,
    split: str = "train",
    max_shard_size: str = "500MB",
) -> str:
    """
    Push the local training dataset to the HuggingFace Hub.

    If ``splits/<split>.json`` exists (i.e., finalize_dataset has been run),
    only images belonging to that split are uploaded.  Otherwise all images
    in annotations.json are uploaded as the requested split name.

    Parameters
    ----------
    dataset_dir    : root of the training dataset
    repo_id        : Hub repository, e.g. "myorg/cryoem-particles"
    token          : HuggingFace write token (falls back to HF_TOKEN env var)
    private        : create a private repository (default True)
    split          : split name to upload ("train", "val", "test", or "all")
    max_shard_size : maximum Parquet shard size passed to push_to_hub

    Returns
    -------
    URL of the repository on the Hub.
    """
    try:
        import datasets as hf_datasets
    except ImportError as exc:
        raise ImportError(
            "datasets is required to push to the HuggingFace Hub.\n"
            "Install it with:  pip install datasets huggingface_hub"
        ) from exc

    dataset_dir = Path(dataset_dir)
    ann_path    = dataset_dir / "annotations.json"

    coco = _load_json(ann_path)
    if coco is None:
        raise FileNotFoundError(f"annotations.json not found in {dataset_dir}")

    # Determine which image IDs to include
    split_path = dataset_dir / "splits" / f"{split}.json"
    if split != "all" and split_path.exists():
        split_coco  = _load_json(split_path)
        valid_ids   = {im["id"] for im in split_coco["images"]}
        images_list = [im for im in coco["images"] if im["id"] in valid_ids]
    else:
        images_list = coco["images"]
        valid_ids   = {im["id"] for im in images_list}

    # Build annotation index
    ann_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        ann_by_image.setdefault(a["image_id"], []).append(a)

    # Build rows
    from PIL import Image as PILImage

    use_hdf5 = (dataset_dir / "dataset.h5").exists()
    h5 = None
    if use_hdf5:
        import h5py
        h5 = h5py.File(dataset_dir / "dataset.h5", "r")

    rows: list[dict] = []
    missing = 0
    try:
        for im in images_list:
            if use_hdf5 and h5 is not None:
                img_id = im["id"]
                key = f"images/{img_id}"
                if key not in h5:
                    missing += 1
                    continue
                arr = h5[key][()]
                pil_img = PILImage.fromarray(arr).convert("RGB")
            else:
                img_path = dataset_dir / im["file_name"]
                if not img_path.exists():
                    missing += 1
                    continue
                pil_img = PILImage.open(str(img_path)).convert("RGB")
            tile_offset = im.get("tile_offset", [0, 0])
            rows.append({
                "image":           pil_img,
                "image_id":        im["id"],
                "file_name":       im["file_name"],
                "width":           im["width"],
                "height":          im["height"],
                "pixel_size_nm":   float(im.get("pixel_size_nm", 0.0)),
                "tile_offset_y":   int(tile_offset[0]),
                "tile_offset_x":   int(tile_offset[1]),
                "aug":             im.get("aug", "orig"),
                "source_image_id": int(im.get("source_image_id", 0)),
                "contrast_params": json.dumps(im.get("contrast_params", {})),
                "annotations":     json.dumps(ann_by_image.get(im["id"], [])),
                "n_instances":     len(ann_by_image.get(im["id"], [])),
            })
    finally:
        if h5 is not None:
            h5.close()

    if not rows:
        raise ValueError(
            f"No image files found for split '{split}' in {dataset_dir}. "
            f"Check that images/ exists and annotations.json is correct."
        )

    # Build HuggingFace Dataset
    import datasets as hf_datasets

    features = hf_datasets.Features({
        "image":           hf_datasets.Image(),
        "image_id":        hf_datasets.Value("int32"),
        "file_name":       hf_datasets.Value("string"),
        "width":           hf_datasets.Value("int32"),
        "height":          hf_datasets.Value("int32"),
        "pixel_size_nm":   hf_datasets.Value("float32"),
        "tile_offset_y":   hf_datasets.Value("int32"),
        "tile_offset_x":   hf_datasets.Value("int32"),
        "aug":             hf_datasets.Value("string"),
        "source_image_id": hf_datasets.Value("int32"),
        "contrast_params": hf_datasets.Value("string"),
        "annotations":     hf_datasets.Value("string"),
        "n_instances":     hf_datasets.Value("int32"),
    })

    # Separate image bytes from other columns (datasets.Image() needs bytes/path)
    # Use from_list then cast
    ds = hf_datasets.Dataset.from_list(rows, features=features)

    ds.push_to_hub(
        repo_id        = repo_id,
        split          = split,
        token          = token,
        private        = private,
        max_shard_size = max_shard_size,
    )

    repo_url = f"https://huggingface.co/datasets/{repo_id}"
    if missing > 0:
        print(f"Warning: {missing} image file(s) were missing and skipped.")
    print(f"Pushed {len(rows)} entries to {repo_url}")
    return repo_url
