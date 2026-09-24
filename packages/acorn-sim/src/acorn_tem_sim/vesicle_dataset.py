"""Dataset generator for lipid-vesicle cryo-TEM measurement benchmarks.

The base TEM simulator already records vesicle geometry before rasterisation.
This module packages that primitive as a first-class benchmark export: images,
YOLO segmentation labels, exact per-object geometry truth, and manifests.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile
import yaml

from acorn_tem_sim.io import uint8_image
from acorn_tem_sim.simulator import (
    Micrograph,
    Specimen,
    apply_ctf,
    ctf,
    expose,
    make_potential,
    resolve,
    sigma_rad_per_va,
    to_display,
)


DENSITY_PARTICLE_RANGES = {
    "sparse": (8, 18),
    "medium": (25, 45),
    "dense": (55, 90),
}

CONTRAST_SETTINGS = {
    "low": {"dose_e_per_a2": 20.0, "solvent_noise": 8.0},
    "high": {"dose_e_per_a2": 50.0, "solvent_noise": 4.0},
}


@dataclass(frozen=True)
class VesicleCondition:
    outer_diameter_nm: float
    pixel_size_a: float
    density: str
    contrast: str


@dataclass(frozen=True)
class VesicleSimulation:
    micrograph: Micrograph
    objects: tuple[dict, ...]


def default_conditions(
    *,
    outer_diameters_nm: Iterable[float] = (30.0, 60.0),
    pixel_sizes_a: Iterable[float] = (10.0, 20.0),
    densities: Iterable[str] = ("sparse", "medium", "dense"),
    contrasts: Iterable[str] = ("low", "high"),
) -> tuple[VesicleCondition, ...]:
    """Return the factorial pilot design for the vesicle benchmark."""

    return tuple(
        VesicleCondition(float(d), float(px), density, contrast)
        for d, px, density, contrast in product(
            outer_diameters_nm,
            pixel_sizes_a,
            densities,
            contrasts,
        )
    )


def simulate_vesicle_micrograph(
    cfg: dict,
    specimen: Specimen,
    *,
    seed: int = 0,
    defocus_um: float | None = None,
    bfactor: float = 40.0,
) -> VesicleSimulation:
    """Simulate a vesicle micrograph while preserving exact object geometry."""

    rng = np.random.default_rng(seed)
    n = int(cfg["image_size_px"])
    shape = (n, n)
    px = float(cfg["pixel_size_a"])
    df = _pick_defocus(cfg, rng, defocus_um)
    if cfg.get("phase_plate"):
        df = df if abs(df) > 0.05 else -0.05

    pot = make_potential(shape, px, specimen)
    sigma = sigma_rad_per_va(float(cfg["voltage_kv"]))
    ideal = apply_ctf(sigma * pot.v_proj, ctf(shape, px, cfg, df, bfactor=bfactor))
    counts = expose(ideal, cfg, rng)
    micrograph = Micrograph(
        image=to_display(counts),
        counts=counts,
        ideal=ideal,
        potential=pot.v_proj,
        label=pot.label,
        defocus_um=df,
        config=cfg,
    )
    return VesicleSimulation(micrograph=micrograph, objects=tuple(pot.objects))


def generate_vesicle_dataset(
    output_dir: Path,
    count: int,
    *,
    seed: int = 0,
    image_size_px: int = 640,
    membrane_thickness_nm: float = 4.0,
    outer_diameters_nm: Iterable[float] = (30.0, 60.0),
    pixel_sizes_a: Iterable[float] = (10.0, 20.0),
    densities: Iterable[str] = ("sparse", "medium", "dense"),
    contrasts: Iterable[str] = ("low", "high"),
    splits: tuple[str, ...] = ("train", "val", "test"),
    split_fractions: tuple[float, ...] = (0.7, 0.15, 0.15),
) -> list[Path]:
    """Generate a vesicle benchmark dataset on disk.

    Labels use class 0 for the outer vesicle boundary. The truth sidecar stores
    all three diameter conventions so measurement bias can be scored against
    outer leaflet, bilayer midplane, or inner leaflet definitions.
    """

    output_dir = Path(output_dir)
    if count < 1:
        raise ValueError("count must be positive")
    if len(splits) != len(split_fractions):
        raise ValueError("splits and split_fractions must have the same length")

    conditions = default_conditions(
        outer_diameters_nm=outer_diameters_nm,
        pixel_sizes_a=pixel_sizes_a,
        densities=densities,
        contrasts=contrasts,
    )
    if not conditions:
        raise ValueError("at least one vesicle condition is required")
    _validate_conditions(conditions)

    rng = np.random.default_rng(seed)
    split_counts = _split_counts(count, splits, split_fractions)
    written: list[Path] = []
    manifest: list[dict] = []

    for split in splits:
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "truth" / split).mkdir(parents=True, exist_ok=True)

    index = 0
    for split in splits:
        split_manifest: list[dict] = []
        for local_index in range(split_counts[split]):
            condition = conditions[int(rng.integers(0, len(conditions)))]
            image_seed = seed + index * 2 + 1
            specimen_seed = seed + index * 2 + 2
            cfg = _config_for_condition(condition, image_size_px)
            specimen = _specimen_for_condition(
                condition,
                rng,
                seed=specimen_seed,
                membrane_thickness_nm=membrane_thickness_nm,
            )
            sim = simulate_vesicle_micrograph(cfg, specimen, seed=image_seed)
            stem = f"vesicle_{split}_{local_index:05d}"
            image_path = output_dir / "images" / split / f"{stem}.tif"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            truth_path = output_dir / "truth" / split / f"{stem}.json"

            tifffile.imwrite(image_path, uint8_image(sim.micrograph.image))
            objects = _truth_objects(sim.objects, cfg)
            label_path.write_text(
                _yolo_segmentation(
                    objects,
                    image_shape_px=sim.micrograph.image.shape,
                    pixel_size_nm=cfg["pixel_size_a"] / 10.0,
                ),
                encoding="utf-8",
            )
            truth = _truth_payload(
                stem=stem,
                split=split,
                condition=condition,
                cfg=cfg,
                specimen=specimen,
                micrograph=sim.micrograph,
                objects=objects,
            )
            truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

            item = {
                "id": index + 1,
                "split": split,
                "stem": stem,
                "image": str(image_path.relative_to(output_dir)),
                "labels": str(label_path.relative_to(output_dir)),
                "truth": str(truth_path.relative_to(output_dir)),
                "n_vesicles": len(objects),
                "outer_diameter_nm": condition.outer_diameter_nm,
                "pixel_size_a": condition.pixel_size_a,
                "pixel_size_nm": condition.pixel_size_a / 10.0,
                "density": condition.density,
                "contrast": condition.contrast,
                "defocus_um": sim.micrograph.defocus_um,
            }
            split_manifest.append(item)
            manifest.append(item)
            written.append(image_path)
            index += 1
        (output_dir / f"manifest_{split}.json").write_text(
            json.dumps({"items": split_manifest}, indent=2),
            encoding="utf-8",
        )

    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generator": "acorn_tem_sim.vesicle_dataset",
                "class_names": {"0": "vesicle"},
                "geometry_truth": [
                    "d_outer_leaflet_nm",
                    "d_bilayer_mid_nm",
                    "d_inner_leaflet_nm",
                ],
                "items": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_data_yaml(output_dir, splits)
    return written


def _config_for_condition(condition: VesicleCondition, image_size_px: int) -> dict:
    contrast = CONTRAST_SETTINGS[condition.contrast]
    return resolve(
        {
            "simulation_path": "fast",
            "image_size_px": int(image_size_px),
            "pixel_size_a": float(condition.pixel_size_a),
            "total_dose_e_per_a2": contrast["dose_e_per_a2"],
            "defocus_min_um": -1.0,
            "defocus_max_um": -2.5,
        }
    )


def _validate_conditions(conditions: Iterable[VesicleCondition]) -> None:
    for condition in conditions:
        if condition.density not in DENSITY_PARTICLE_RANGES:
            known = ", ".join(sorted(DENSITY_PARTICLE_RANGES))
            raise ValueError(f"unknown density {condition.density!r}; expected one of {known}")
        if condition.contrast not in CONTRAST_SETTINGS:
            known = ", ".join(sorted(CONTRAST_SETTINGS))
            raise ValueError(f"unknown contrast {condition.contrast!r}; expected one of {known}")


def _specimen_for_condition(
    condition: VesicleCondition,
    rng: np.random.Generator,
    *,
    seed: int,
    membrane_thickness_nm: float,
) -> Specimen:
    lo, hi = DENSITY_PARTICLE_RANGES[condition.density]
    contrast = CONTRAST_SETTINGS[condition.contrast]
    n_particles = int(rng.integers(lo, hi + 1))
    return Specimen(
        kind="lipid_single",
        ice_thickness_nm=40.0,
        n_particles=n_particles,
        diameter_nm_mean=float(condition.outer_diameter_nm),
        diameter_nm_sd=max(1.0, float(condition.outer_diameter_nm) * 0.08),
        membrane_thickness_nm=float(membrane_thickness_nm),
        solvent_noise=contrast["solvent_noise"],
        solvent_corr_a=3.5,
        allow_overlap=False,
        seed=int(seed),
    )


def _truth_payload(
    *,
    stem: str,
    split: str,
    condition: VesicleCondition,
    cfg: dict,
    specimen: Specimen,
    micrograph: Micrograph,
    objects: list[dict],
) -> dict:
    return {
        "version": 1,
        "stem": stem,
        "split": split,
        "specimen_kind": specimen.kind,
        "image_shape_px": list(micrograph.image.shape),
        "pixel_size_a": cfg["pixel_size_a"],
        "pixel_size_nm": cfg["pixel_size_a"] / 10.0,
        "defocus_um": micrograph.defocus_um,
        "dose_e_per_a2": cfg["total_dose_e_per_a2"],
        "geometry_targets_nm": [
            "d_outer_leaflet_nm",
            "d_bilayer_mid_nm",
            "d_inner_leaflet_nm",
        ],
        "condition": {
            "outer_diameter_nm": condition.outer_diameter_nm,
            "pixel_size_a": condition.pixel_size_a,
            "density": condition.density,
            "contrast": condition.contrast,
        },
        "edge_conventions": {
            "outer_leaflet": "outer membrane edge",
            "bilayer_midplane": "midpoint through the membrane thickness",
            "inner_leaflet": "inner membrane edge",
        },
        "objects": objects,
    }


def _truth_objects(objects: tuple[dict, ...], cfg: dict) -> list[dict]:
    h = w = int(cfg["image_size_px"])
    px_nm = float(cfg["pixel_size_a"]) / 10.0
    truth: list[dict] = []
    for i, obj in enumerate(objects):
        outer_d_nm = float(obj["d_outer_leaflet_nm"])
        radius_px = outer_d_nm / (2.0 * px_nm)
        cy = float(obj["cy_px"])
        cx = float(obj["cx_px"])
        bbox = [
            max(0.0, cx - radius_px),
            max(0.0, cy - radius_px),
            min(float(w), cx + radius_px),
            min(float(h), cy + radius_px),
        ]
        fully_in_frame = (
            cx - radius_px >= 0
            and cy - radius_px >= 0
            and cx + radius_px <= w
            and cy + radius_px <= h
        )
        truth.append(
            {
                "id": i + 1,
                "class_name": "vesicle",
                "cy_px": cy,
                "cx_px": cx,
                "bbox_xyxy_px": bbox,
                "fully_in_frame": fully_in_frame,
                "d_outer_leaflet_nm": outer_d_nm,
                "d_bilayer_mid_nm": float(obj["d_bilayer_mid_nm"]),
                "d_inner_leaflet_nm": float(obj["d_inner_leaflet_nm"]),
                "membrane_thickness_nm": float(obj["membrane_thickness_nm"]),
                "n_lamellae": int(obj["n_lamellae"]),
                "ecd_nm": float(obj["ecd_nm"]),
            }
        )
    return truth


def _yolo_segmentation(
    objects: list[dict],
    *,
    image_shape_px: tuple[int, int],
    pixel_size_nm: float,
    vertices: int = 48,
) -> str:
    lines = []
    for obj in objects:
        polygon = _circle_polygon(
            cx=float(obj["cx_px"]),
            cy=float(obj["cy_px"]),
            diameter_nm=float(obj["d_outer_leaflet_nm"]),
            pixel_size_nm=pixel_size_nm,
            image_shape_px=image_shape_px,
            vertices=vertices,
        )
        lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon))
    return "\n".join(lines) + ("\n" if lines else "")


def _circle_polygon(
    *,
    cx: float,
    cy: float,
    diameter_nm: float,
    pixel_size_nm: float,
    image_shape_px: tuple[int, int],
    vertices: int,
) -> list[tuple[float, float]]:
    h, w = image_shape_px
    radius_px = diameter_nm / (2.0 * pixel_size_nm)
    theta = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    xs = np.clip(cx + radius_px * np.cos(theta), 0.0, float(w - 1)) / float(w)
    ys = np.clip(cy + radius_px * np.sin(theta), 0.0, float(h - 1)) / float(h)
    return list(zip(xs.tolist(), ys.tolist()))


def _split_counts(count: int, splits: tuple[str, ...], fractions: tuple[float, ...]) -> dict[str, int]:
    weights = np.asarray(fractions, dtype=float)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError("split fractions must be non-negative and sum to a positive value")
    weights = weights / float(weights.sum())
    raw = weights * count
    counts = np.floor(raw).astype(int)
    remainder = count - int(counts.sum())
    order = np.argsort(-(raw - counts))
    for idx in order[:remainder]:
        counts[int(idx)] += 1
    return {split: int(n) for split, n in zip(splits, counts)}


def _pick_defocus(cfg: dict, rng: np.random.Generator, defocus_um: float | None) -> float:
    if defocus_um is not None:
        return float(defocus_um)
    lo, hi = cfg["defocus_min_um"], cfg["defocus_max_um"]
    return float(rng.uniform(min(lo, hi), max(lo, hi)))


def _write_data_yaml(output_dir: Path, splits: tuple[str, ...]) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train" if "train" in splits else None,
        "val": "images/val" if "val" in splits else None,
        "test": "images/test" if "test" in splits else None,
        "nc": 1,
        "names": {0: "vesicle"},
        "truth_dir": "truth",
        "geometry_targets_nm": [
            "d_outer_leaflet_nm",
            "d_bilayer_mid_nm",
            "d_inner_leaflet_nm",
        ],
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump({k: v for k, v in payload.items() if v is not None}, sort_keys=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a lipid-vesicle cryo-TEM benchmark dataset.")
    parser.add_argument("--output", required=True, type=Path, help="Output dataset directory.")
    parser.add_argument("--count", type=int, default=30, help="Total images to generate.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--image-size", type=int, default=640, help="Square image size in pixels.")
    parser.add_argument("--membrane-thickness-nm", type=float, default=4.0)
    parser.add_argument("--outer-diameters-nm", type=float, nargs="+", default=(30.0, 60.0))
    parser.add_argument("--pixel-sizes-a", type=float, nargs="+", default=(10.0, 20.0))
    parser.add_argument("--densities", nargs="+", default=("sparse", "medium", "dense"))
    parser.add_argument("--contrasts", nargs="+", default=("low", "high"))
    args = parser.parse_args(argv)

    paths = generate_vesicle_dataset(
        args.output,
        args.count,
        seed=args.seed,
        image_size_px=args.image_size,
        membrane_thickness_nm=args.membrane_thickness_nm,
        outer_diameters_nm=args.outer_diameters_nm,
        pixel_sizes_a=args.pixel_sizes_a,
        densities=args.densities,
        contrasts=args.contrasts,
    )
    print(f"Wrote {len(paths)} vesicle images plus labels, truth, and manifests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
