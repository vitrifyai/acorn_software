"""Export blinded simulated micrographs for operator annotation.

Operators annotate simulated micrographs exactly as they would acquired ones.
Because every true diameter is known, their outlines can then be scored like
any other method, which measures the bias of a hand-drawn reference directly.

Micrographs are renamed to opaque identifiers and shuffled so that neither file
names nor order reveal the specimen condition. Truth and the micrograph key are
written to a separate ``_key`` folder that must not be sent to annotators.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from acorn_sim_common.classical_pickers import _dataset_items


def export_annotation_package(
    datasets: dict[str, Path],
    output_dir: Path,
    *,
    fields_per_specimen: int = 20,
    seed: int = 0,
    splits: tuple[str, ...] = ("test",),
) -> Path:
    """Write ``micrographs/<specimen>/F###.tif`` plus ``_key/`` with truth.

    Plain ``.tif`` names, not ``.ome.tif``: ACORN names its sidecar after the
    file, and the stem must match the truth file for scoring.
    """

    output_dir = Path(output_dir)
    rng = np.random.default_rng(seed)
    key_rows: list[dict] = []
    for specimen, dataset_dir in datasets.items():
        items = _dataset_items(Path(dataset_dir), splits=splits)
        if len(items) < fields_per_specimen:
            raise ValueError(f"{specimen}: only {len(items)} micrographs available, {fields_per_specimen} requested")
        chosen = [items[int(i)] for i in rng.choice(len(items), size=fields_per_specimen, replace=False)]
        field_dir = output_dir / "micrographs" / specimen
        truth_dir = output_dir / "_key" / "truth" / specimen
        field_dir.mkdir(parents=True, exist_ok=True)
        truth_dir.mkdir(parents=True, exist_ok=True)
        for n, item in enumerate(chosen, start=1):
            field_id = f"F{n:03d}"
            image = tifffile.imread(item.image_path)
            if image.ndim == 3:
                image = image[..., 0]
            # OME PhysicalSizeX is what ACORN reads for calibration.
            tifffile.imwrite(
                field_dir / f"{field_id}.tif",
                image,
                ome=True,
                metadata={
                    "axes": "YX",
                    "PhysicalSizeX": float(item.pixel_size_nm),
                    "PhysicalSizeXUnit": "nm",
                    "PhysicalSizeY": float(item.pixel_size_nm),
                    "PhysicalSizeYUnit": "nm",
                },
            )
            payload = json.loads(item.truth_path.read_text(encoding="utf-8"))
            payload["stem"] = field_id
            (truth_dir / f"{field_id}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            key_rows.append(
                {
                    "specimen": specimen,
                    "field_id": field_id,
                    "source_dataset": str(dataset_dir),
                    "source_stem": item.stem,
                    "source_split": item.split,
                    "pixel_size_nm": item.pixel_size_nm,
                    "n_truth_objects": len(payload.get("objects", [])),
                    "condition": json.dumps(payload.get("condition", {}), sort_keys=True),
                }
            )
    key_path = output_dir / "_key" / "field_key.csv"
    with key_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(key_rows[0]))
        writer.writeheader()
        writer.writerows(key_rows)
    (output_dir / "_key" / "README.txt").write_text(
        "Do not send this folder to annotators. It holds the exact truth for every micrograph.\n"
        "Score returned annotations with:\n"
        "  python -m acorn_sim_common.measurement_bias --truth-dir _key/truth/<specimen> \\\n"
        "      --predictions <returned folder>/micrographs/<specimen> --output scored/<specimen> --method operator\n",
        encoding="utf-8",
    )
    return output_dir


def write_handout(output_dir: Path) -> Path:
    """Copy-ready folder for annotators: the micrographs and nothing else."""

    output_dir = Path(output_dir)
    handout = output_dir / "handout"
    if handout.exists():
        shutil.rmtree(handout)
    shutil.copytree(output_dir / "micrographs", handout / "micrographs")
    return handout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export blinded simulated fields for operator annotation.")
    parser.add_argument("--dataset", action="append", required=True, metavar="SPECIMEN=DIR",
                        help="Specimen name and simulator dataset directory; repeat per specimen.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fields-per-specimen", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--splits", nargs="+", default=["test"])
    args = parser.parse_args(argv)
    datasets = dict(spec.split("=", 1) for spec in args.dataset)
    out = export_annotation_package(
        {k: Path(v) for k, v in datasets.items()}, args.output,
        fields_per_specimen=args.fields_per_specimen, seed=args.seed, splits=tuple(args.splits),
    )
    write_handout(out)
    print(f"Wrote annotation package to {out}; send only {out / 'handout'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
