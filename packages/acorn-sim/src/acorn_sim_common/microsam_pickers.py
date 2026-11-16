"""microSAM picker for exact-truth measurement-bias studies.

Wraps the ``micro_sam.train`` and ``micro_sam.automatic_segmentation`` command
line tools, with the configuration used for the simulation-trained comparator
in the vocabulary study: ViT-B, instance decoder, percentile normalisation,
automatic instance segmentation (AIS) at inference. Training labels are
instance maps rasterized from the analytic truth, so no annotation enters.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.measure import regionprops

from acorn_sim_common.classical_pickers import _dataset_items
from acorn_sim_common.unet_pickers import _instance_record, rasterize_truth_instances


def prepare_microsam_dataset(dataset_dir: Path, output_dir: Path, *, target_field: str = "diameter_nm") -> Path:
    """Write ``images/<split>/*.tif`` and matching ``labels/<split>/*.tif``."""

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    for item in _dataset_items(dataset_dir):
        payload = json.loads(item.truth_path.read_text(encoding="utf-8"))
        img_dir = output_dir / "images" / item.split
        lab_dir = output_dir / "labels" / item.split
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        image = tifffile.imread(item.image_path)
        if image.ndim == 3:
            image = image[..., 0]
        tifffile.imwrite(img_dir / f"{item.stem}.tif", image)
        tifffile.imwrite(lab_dir / f"{item.stem}.tif", rasterize_truth_instances(payload, target_field=target_field))
    return output_dir


def train_microsam(
    dataset_dir: Path,
    output_path: Path,
    *,
    target_field: str = "diameter_nm",
    epochs: int = 25,
    model_type: str = "vit_b",
    device: str = "cuda",
    patch_size: int | None = None,
    work_dir: Path | None = None,
    min_instances: int = 0,
) -> Path:
    """Fine-tune microSAM on the train split, validating on val.

    ``min_instances`` drops images with fewer labelled objects than that: the
    torch_em sampler cannot draw a patch from an image with no objects, which
    sparse real fields cut into tiles often are.
    """

    output_path = Path(output_path)
    work_dir = Path(work_dir) if work_dir is not None else output_path.parent / f"{output_path.stem}_work"
    prepared = prepare_microsam_dataset(dataset_dir, work_dir / "data", target_field=target_field)

    def usable(split: str) -> list[Path]:
        paths = sorted((prepared / "images" / split).glob("*.tif"))
        if min_instances <= 0:
            return paths
        return [p for p in paths if len(np.unique(tifffile.imread(prepared / "labels" / split / p.name))) - 1 >= min_instances]

    train_images = usable("train")
    val_images = usable("val")
    if not train_images or not val_images:
        raise ValueError("microSAM training needs both train and val splits")
    if patch_size is None:
        h, w = tifffile.imread(train_images[0]).shape[:2]
        patch_size = min(512, h, w)
    cmd = [
        _tool("micro_sam.train"),
        "--images", *map(str, train_images),
        "--labels", *[str(prepared / "labels" / "train" / p.name) for p in train_images],
        "--val_images", *map(str, val_images),
        "--val_labels", *[str(prepared / "labels" / "val" / p.name) for p in val_images],
        "--configuration", "V100",
        "--model_type", model_type,
        "--segmentation_decoder", "instances",
        "--device", device,
        "--patch_shape", str(patch_size), str(patch_size),
        "--batch_size", "1",
        "--n_epochs", str(int(epochs)),
        "--num_workers", "4",
        "--preprocess", "normalize_percentile",
        "--save_root", str(work_dir / "runs"),
        "--trained_model_name", output_path.stem,
        "--output_path", str(output_path),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with (work_dir / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    if not output_path.exists():
        raise FileNotFoundError(f"micro_sam.train did not write {output_path}")
    return output_path


def predict_microsam(
    dataset_dir: Path,
    checkpoint: Path,
    output_dir: Path,
    *,
    class_name: str,
    model_type: str = "vit_b",
    device: str = "cuda",
    splits: tuple[str, ...] | None = ("test",),
) -> list[Path]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    items = _dataset_items(dataset_dir, splits=splits)
    written: list[Path] = []
    for split in sorted({item.split for item in items}):
        split_items = [item for item in items if item.split == split]
        stage = output_dir / "_ais_input" / split
        masks_dir = output_dir / "_ais_masks" / split
        stage.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)
        for item in split_items:
            image = tifffile.imread(item.image_path)
            tifffile.imwrite(stage / f"{item.stem}.tif", image[..., 0] if image.ndim == 3 else image)
        cmd = [
            _tool("micro_sam.automatic_segmentation"),
            "--input_path", str(stage),
            "--pattern", "*.tif",
            "--output_path", str(masks_dir),
            "--model_type", model_type,
            "--checkpoint", str(checkpoint),
            "--mode", "ais",
            "--device", device,
        ]
        with (output_dir / f"ais_{split}.log").open("w", encoding="utf-8") as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
        for item in split_items:
            instances = tifffile.imread(_mask_path(masks_dir, item.stem))
            predictions = [
                {**_instance_record(instances == region.label, region, item.pixel_size_nm, class_name), "source": "microsam-ais"}
                for region in regionprops(np.asarray(instances, dtype=np.int64))
            ]
            payload = {
                "stem": item.stem,
                "method": "microsam-ais",
                "class_name": class_name,
                "pixel_size_nm": item.pixel_size_nm,
                "source_image": str(item.image_path),
                "checkpoint": str(checkpoint),
                "predictions": predictions,
            }
            out_path = output_dir / split / f"{item.stem}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written.append(out_path)
    shutil.rmtree(output_dir / "_ais_input", ignore_errors=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"method": "microsam-ais", "class_name": class_name, "source_dataset": str(dataset_dir), "n_images": len(written)}, indent=2),
        encoding="utf-8",
    )
    return written


def _mask_path(masks_dir: Path, stem: str) -> Path:
    for candidate in (masks_dir / f"{stem}.tif", masks_dir / f"{stem}.tiff"):
        if candidate.exists():
            return candidate
    hits = sorted(masks_dir.rglob(f"{stem}*.tif*"))
    if not hits:
        raise FileNotFoundError(f"no microSAM output for {stem} under {masks_dir}")
    return hits[0]


def _tool(name: str) -> str:
    beside = Path(sys.executable).parent / name
    if beside.exists():
        return str(beside)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(f"{name} is not installed in this environment")
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or run microSAM on an exact-truth simulator dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path, help="Checkpoint path (.pt).")
    tr.add_argument("--target-field", default="diameter_nm")
    tr.add_argument("--epochs", type=int, default=25)
    tr.add_argument("--model-type", default="vit_b")
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--patch-size", type=int, default=None)
    tr.add_argument("--min-instances", type=int, default=0, help="Skip images with fewer labelled objects.")

    pr = sub.add_parser("predict")
    pr.add_argument("--dataset", required=True, type=Path)
    pr.add_argument("--checkpoint", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    pr.add_argument("--class-name", required=True)
    pr.add_argument("--model-type", default="vit_b")
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--splits", nargs="+", default=["test"])

    args = parser.parse_args(argv)
    if args.cmd == "train":
        print(
            train_microsam(
                args.dataset, args.output, target_field=args.target_field, epochs=args.epochs,
                model_type=args.model_type, device=args.device, patch_size=args.patch_size,
                min_instances=args.min_instances,
            )
        )
        return 0
    paths = predict_microsam(
        args.dataset, args.checkpoint, args.output, class_name=args.class_name,
        model_type=args.model_type, device=args.device, splits=tuple(args.splits),
    )
    print(f"Wrote {len(paths)} microSAM prediction files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
