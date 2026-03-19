"""
acorn CLI entry point.

Subcommands
-----------
view          FILE [FILE...]    Open the PyQt6 GUI with one or more files
export        FILE              Headless export to PNG/TIFF/SVG/etc.
batch         --input-dir DIR   Batch convert a folder of images
train-export  --input-dir DIR   Build an AI training dataset from a folder
                                (tiles + augmentation, no GUI annotation)
finalize      --dataset-dir DIR Create train/val/test splits + statistics

Usage examples
--------------
# GUI
acorn view sample.dm4
acorn view /path/to/*.mrc

# Headless single export
acorn export sample.dm4 --output out.png --dpi 300
acorn export sample.mrc --contrast bandpass --bp-low 20 --format svg

# Batch headless image conversion
acorn batch --input-dir /data/dm4s --output-dir /data/pngs --format tiff --workers 8

# Build training dataset (image tiles only, no annotation masks)
acorn train-export --input-dir /data/raw --dataset-dir /data/train \\
    --tile-size 1024 --overlap 0.25 --augment --workers 8

# Finalize dataset: session-aware splits + statistics
acorn finalize --dataset-dir /data/train --val 0.1 --test 0.1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── helpers ───────────────────────────────────────────────────────────────────

def _build_contrast_params(args) -> "ContrastParams":
    """Build a ContrastParams from parsed CLI args (headless path)."""
    from acorn.core.contrast import ContrastParams
    return ContrastParams(
        method=args.contrast,
        low_pct=args.pct_low,
        high_pct=args.pct_high,
        n_sigma=args.sigma,
        clip_limit=args.clahe_clip,
        bp_low_sigma=args.bp_low,
        bp_high_sigma=args.bp_high,
        gamma=args.gamma,
        colormap=args.colormap,
    )


def _add_contrast_args(p: argparse.ArgumentParser) -> None:
    """Attach shared contrast flags to a subparser."""
    g = p.add_argument_group("contrast")
    g.add_argument(
        "--contrast", default="bandpass",
        choices=["percentile", "sigma", "adaptive", "bandpass"],
        help="Normalisation method (default: bandpass — best for low-dose cryo-EM)",
    )
    g.add_argument("--pct-low",    type=float, default=0.5,   metavar="PCT", help="Low percentile clip (percentile method)")
    g.add_argument("--pct-high",   type=float, default=99.5,  metavar="PCT", help="High percentile clip (percentile method)")
    g.add_argument("--sigma",      type=float, default=3.0,   metavar="N",   help="Sigma width (sigma method)")
    g.add_argument("--clahe-clip", type=float, default=0.03,  metavar="C",   help="CLAHE clip limit (adaptive method)")
    g.add_argument("--bp-low",     type=float, default=20.0,  metavar="PX",  help="Background subtraction Gaussian radius in px (bandpass)")
    g.add_argument("--bp-high",    type=float, default=1.0,   metavar="PX",  help="Noise smoothing Gaussian radius in px (bandpass)")
    g.add_argument("--gamma",      type=float, default=1.0,   metavar="G",   help="Gamma correction (1 = linear)")
    g.add_argument("--colormap",   default="gray",             metavar="CM",  help="Matplotlib colormap name (default: gray)")


# ── subcommand handlers ───────────────────────────────────────────────────────

def cmd_view(args) -> None:
    """Launch the PyQt6 GUI."""
    print("acorn: cmd_view reached", flush=True)
    # IMOD and other cryo-EM tools inject old X11/xcb libraries into
    # LD_LIBRARY_PATH which conflict with Qt. Strip them before Qt loads.
    import os
    ldp = os.environ.get("LD_LIBRARY_PATH", "")
    if ldp:
        cleaned = ":".join(
            p for p in ldp.split(":") if "IMOD" not in p and "imod" not in p
        )
        os.environ["LD_LIBRARY_PATH"] = cleaned

    print("acorn: importing GUI module", flush=True)
    try:
        from acorn.gui.main_window import launch
    except ImportError as e:
        print(f"ERROR: GUI dependencies not installed ({e})\n"
              "Install them with:  uv pip install 'acorn[gui]'",
              file=sys.stderr, flush=True)
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"ERROR importing GUI: {e}", file=sys.stderr, flush=True)
        traceback.print_exc()
        sys.exit(1)

    print("acorn: calling launch()", flush=True)
    files = [str(Path(f)) for f in getattr(args, "files", []) if Path(f).is_file()]
    sys.exit(launch(files=files))


def cmd_export(args) -> None:
    """Headless single-file export."""
    import matplotlib
    matplotlib.use("Agg")

    from acorn.core.dm4_loader import DM4Image
    from acorn.export.image_exporter import export_image

    src = Path(args.file)
    if not src.is_file():
        print(f"File not found: {src}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        out = Path(args.output)
    else:
        fmt = args.format or "png"
        ext_map = {"jpeg": "jpg", "tiff": "tif"}
        ext = ext_map.get(fmt, fmt)
        out = src.with_suffix(f".{ext}")

    fmt = args.format or (out.suffix.lstrip(".") if out.suffix else "png")
    params = _build_contrast_params(args)

    print(f"Loading {src.name} …")
    img = DM4Image.from_file(src)
    print(img.summary())

    print(f"Exporting → {out}  (DPI={args.dpi}, contrast={params.method})")
    result = export_image(
        img, out,
        params=params,
        dpi=args.dpi,
        fmt=fmt,
        add_scalebar=not args.no_scalebar,
        scalebar_color=args.scalebar_color,
    )
    print(f"Saved: {result}")


def cmd_train_export(args) -> None:
    """
    Headless training dataset builder.

    Tiles and augments every image in --input-dir, writing normalized PNGs
    (no annotation masks — use the GUI for annotated export).  Useful for
    building large self-supervised pre-training corpora from raw cryo-EM data.
    """
    import concurrent.futures
    from acorn.core.dm4_loader import DM4Image, scan_folder
    from acorn.core.contrast import apply_contrast
    from acorn.export.training_exporter import TrainingConfig, _extract_tiles, _augment_tile
    from PIL import Image as PILImage
    import numpy as np

    params = _build_contrast_params(args)
    config = TrainingConfig(
        tile_size        = args.tile_size if args.tile_size > 0 else None,
        tile_overlap     = args.overlap,
        augment          = args.augment,
        n_neg_prompts    = 0,          # no ROI masks in headless mode
        skip_empty_tiles = False,      # keep all tiles (no masks to check)
        encode_rle       = False,
    )

    in_dir  = args.input_dir
    ds_dir  = args.dataset_dir
    images_dir = __import__("pathlib").Path(ds_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    files = scan_folder(in_dir)
    if not files:
        print(f"No supported image files found in {in_dir}", flush=True)
        return

    print(f"Found {len(files)} file(s) in {in_dir}")
    print(f"Tile size: {config.tile_size}  Overlap: {config.tile_overlap}  Augment: {config.augment}")

    n_written = 0

    def _process(src):
        try:
            img = DM4Image.from_file(src)
            h, w = img.shape[:2]
            norm = apply_contrast(img.raw, params)
            img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)

            if config.tile_size is not None:
                tiles = _extract_tiles(img8, [], config.tile_size, config.tile_overlap)
            else:
                tiles = [{"img": img8, "masks": [], "y0": 0, "x0": 0, "tile_idx": 0}]

            written = 0
            for tile in tiles:
                versions = _augment_tile(tile["img"], []) if config.augment else [(tile["img"], [], "orig")]
                for aug_img, _, suffix in versions:
                    fname = f"{src.stem}_t{tile['tile_idx']:04d}_{suffix}.png"
                    PILImage.fromarray(aug_img, mode="L").save(str(images_dir / fname))
                    written += 1
            return written, None
        except Exception as exc:
            return 0, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process, f): f for f in files}
        for i, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            count, err = fut.result()
            src = futures[fut]
            n_written += count
            status = f"{count} tiles" if err is None else f"FAILED: {err}"
            print(f"  [{i:>{len(str(len(files)))}}/{len(files)}] {src.name}  {status}", flush=True)

    print(f"\nDone. {n_written} image tiles written to {images_dir}")


def cmd_finalize(args) -> None:
    """Finalize a training dataset: session-aware splits + statistics."""
    from acorn.export.dataset_finalizer import finalize_dataset
    print(f"Finalizing dataset: {args.dataset_dir}")
    result = finalize_dataset(args.dataset_dir, val_frac=args.val, test_frac=args.test)
    print(result["stats_str"])
    sc = result["split_counts"]
    print(f"\nSplit files written to {args.dataset_dir}/splits/")
    print(f"  train.json : {sc['train']} tiles")
    print(f"  val.json   : {sc['val']} tiles")
    if sc["test"] > 0:
        print(f"  test.json  : {sc['test']} tiles")


def cmd_push_hub(args) -> None:
    """Push training dataset to HuggingFace Hub."""
    from acorn.export.hub_exporter import push_to_hub
    print(f"Pushing dataset: {args.dataset_dir}  ->  {args.repo_id}  (split={args.split})")
    url = push_to_hub(
        dataset_dir    = args.dataset_dir,
        repo_id        = args.repo_id,
        token          = args.token,
        private        = not args.public,
        split          = args.split,
        max_shard_size = args.shard_size,
    )
    print(f"Dataset available at: {url}")


def cmd_migrate_hdf5(args) -> None:
    from acorn.export.migrate_to_hdf5 import migrate_dataset
    result = migrate_dataset(
        args.dataset_dir,
        delete_originals=args.delete_originals,
        progress_callback=print,
    )
    print(f"Done. {result['n_images']} images, {result['n_masks']} masks migrated. "
          f"{result['n_skipped']} skipped.")


def cmd_batch(args) -> None:
    """Headless batch conversion."""
    import matplotlib
    matplotlib.use("Agg")

    from acorn.export.batch import batch_export

    params = _build_contrast_params(args)

    print(f"Batch: {args.input_dir}  →  {args.output_dir}")
    print(f"Format: {args.format.upper()}  DPI: {args.dpi}  Method: {params.method}  Workers: {args.workers}")

    def progress(n_done, n_total, name):
        print(f"  [{n_done:>{len(str(n_total))}}/{n_total}] {name}")

    result = batch_export(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        params=params,
        fmt=args.format,
        dpi=args.dpi,
        add_scalebar=not args.no_scalebar,
        scalebar_color=args.scalebar_color,
        workers=args.workers,
        progress_cb=progress,
    )
    print()
    print(result.summary())
    if result.failed:
        sys.exit(1)


# ── argument parser ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acorn",
        description="DM4 cryo-EM image viewer, annotator, and headless exporter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version="acorn 0.1.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    # ── view ──────────────────────────────────────────────────────────────────
    p_view = sub.add_parser("view", help="Open GUI viewer")
    p_view.add_argument("files", nargs="*", metavar="FILE", help="DM4 file(s) to open")
    p_view.set_defaults(func=cmd_view)

    # ── export ────────────────────────────────────────────────────────────────
    p_exp = sub.add_parser("export", help="Headless single-file export")
    p_exp.add_argument("file", metavar="FILE", help="Input DM4 file")
    p_exp.add_argument("-o", "--output", metavar="PATH", help="Output file path (auto-named if omitted)")
    p_exp.add_argument("-f", "--format", metavar="FMT",
                       choices=["png", "tiff", "jpeg", "svg", "eps", "pdf"],
                       help="Output format (inferred from --output extension if omitted)")
    p_exp.add_argument("--dpi", type=int, default=300, help="Export DPI (default: 300)")
    p_exp.add_argument("--no-scalebar", action="store_true", help="Suppress automatic scale bar")
    p_exp.add_argument("--scalebar-color", default="#FFFFFF", metavar="COLOR", help="Scale bar colour (default: #FFFFFF)")
    _add_contrast_args(p_exp)
    p_exp.set_defaults(func=cmd_export)

    # ── batch ─────────────────────────────────────────────────────────────────
    p_bat = sub.add_parser("batch", help="Headless batch folder conversion")
    p_bat.add_argument("--input-dir",  required=True, metavar="DIR", help="Folder containing image files")
    p_bat.add_argument("--output-dir", required=True, metavar="DIR", help="Destination folder")
    p_bat.add_argument("-f", "--format", default="tiff", metavar="FMT",
                       choices=["png", "tiff", "jpeg", "svg", "eps", "pdf"],
                       help="Output format (default: tiff)")
    p_bat.add_argument("--dpi",     type=int, default=300, help="Export DPI (default: 300)")
    p_bat.add_argument("--workers", type=int, default=4,   help="Parallel worker threads (default: 4)")
    p_bat.add_argument("--no-scalebar", action="store_true")
    p_bat.add_argument("--scalebar-color", default="#FFFFFF", metavar="COLOR")
    _add_contrast_args(p_bat)
    p_bat.set_defaults(func=cmd_batch)

    # ── train-export ──────────────────────────────────────────────────────────
    p_tr = sub.add_parser(
        "train-export",
        help="Build an AI training dataset from a folder (image tiles, no annotation masks)"
    )
    p_tr.add_argument("--input-dir",   required=True, metavar="DIR", help="Folder of images to process")
    p_tr.add_argument("--dataset-dir", required=True, metavar="DIR", help="Output training dataset folder")
    p_tr.add_argument("--tile-size", type=int, default=1024, metavar="PX",
                      help="Tile side length in pixels; 0 = full image (default: 1024)")
    p_tr.add_argument("--overlap", type=float, default=0.25, metavar="FRAC",
                      help="Fractional overlap between tiles, e.g. 0.25 = 25%% (default: 0.25)")
    p_tr.add_argument("--augment", action="store_true", default=True,
                      help="Generate all 8 rigid orientations per tile (default: on)")
    p_tr.add_argument("--no-augment", dest="augment", action="store_false",
                      help="Disable augmentation")
    p_tr.add_argument("--workers", type=int, default=4, help="Parallel worker threads (default: 4)")
    _add_contrast_args(p_tr)
    p_tr.set_defaults(func=cmd_train_export)

    # ── finalize ──────────────────────────────────────────────────────────────
    p_fin = sub.add_parser(
        "finalize",
        help="Session-aware train/val/test split + dataset statistics"
    )
    p_fin.add_argument("--dataset-dir", required=True, metavar="DIR",
                       help="Root of training dataset (must contain annotations.json)")
    p_fin.add_argument("--val",  type=float, default=0.1, metavar="FRAC",
                       help="Fraction of source images held out for validation (default: 0.1)")
    p_fin.add_argument("--test", type=float, default=0.1, metavar="FRAC",
                       help="Fraction held out for testing; 0 = no test split (default: 0.1)")
    p_fin.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p_fin.set_defaults(func=cmd_finalize)

    # ── push-to-hub ───────────────────────────────────────────────────────────
    p_hub = sub.add_parser(
        "push-to-hub",
        help="Push training dataset to the HuggingFace Hub (requires: pip install datasets)"
    )
    p_hub.add_argument("--dataset-dir", required=True, metavar="DIR",
                       help="Root of training dataset (must contain annotations.json)")
    p_hub.add_argument("--repo-id", required=True, metavar="REPO",
                       help="HuggingFace repository, e.g. myorg/cryoem-particles")
    p_hub.add_argument("--token", default=None, metavar="TOKEN",
                       help="HuggingFace write token (falls back to HF_TOKEN env var)")
    p_hub.add_argument("--public", action="store_true",
                       help="Create a public repository (default: private)")
    p_hub.add_argument("--split", default="train", metavar="SPLIT",
                       help="Split name to upload: train, val, test, or all (default: train)")
    p_hub.add_argument("--shard-size", default="500MB", metavar="SIZE",
                       help="Maximum Parquet shard size (default: 500MB)")
    p_hub.set_defaults(func=cmd_push_hub)

    # ── migrate-to-hdf5 ───────────────────────────────────────────────────────
    p_mig = sub.add_parser(
        "migrate-to-hdf5",
        help="Convert a legacy PNG-based dataset to HDF5 format"
    )
    p_mig.add_argument("--dataset-dir", required=True, metavar="DIR",
                       help="Root of training dataset (must contain annotations.json)")
    p_mig.add_argument("--delete-originals", action="store_true",
                       help="Remove images/, masks/, prompts/ after migration")
    p_mig.set_defaults(func=cmd_migrate_hdf5)

    return parser


def main() -> None:
    print("acorn: starting", flush=True)
    parser = build_parser()
    args = parser.parse_args()
    if not hasattr(args, "func"):
        print("acorn: no subcommand, launching GUI", flush=True)
        from types import SimpleNamespace
        args = SimpleNamespace(func=cmd_view, files=[])
    args.func(args)


if __name__ == "__main__":
    main()
