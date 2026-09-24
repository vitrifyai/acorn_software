"""Dataset generator for bacterial-spore SEM measurement benchmarks."""

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
from scipy import ndimage
from skimage import measure

from acorn_sem_sim import imaging as I
from acorn_sem_sim import scenes as SC
from acorn_sem_sim.io import uint8_image


SIZE_CLASSES_NM = {
    "small": (800.0, 450.0),
    "medium": (1200.0, 700.0),
    "large": (2000.0, 1200.0),
}

CLUSTERING_LEVELS = {
    "isolated": 0.15,
    "touching": 0.50,
    "clumped": 0.85,
}

PIXEL_SIZES_NM = (4.0, 9.6, 19.3)


@dataclass(frozen=True)
class SporeCondition:
    size_class: str
    clustering: str
    pixel_size_nm: float
    debris: bool


def default_conditions(
    *,
    size_classes: Iterable[str] = ("small", "medium", "large"),
    clusterings: Iterable[str] = ("isolated", "touching", "clumped"),
    pixel_sizes_nm: Iterable[float] = PIXEL_SIZES_NM,
    debris: Iterable[bool] = (False, True),
) -> tuple[SporeCondition, ...]:
    return tuple(
        SporeCondition(size_class, clustering, float(px), bool(has_debris))
        for size_class, clustering, px, has_debris in product(size_classes, clusterings, pixel_sizes_nm, debris)
    )


def generate_spore_dataset(
    output_dir: Path,
    count: int,
    *,
    seed: int = 0,
    image_size_px: int = 640,
    conditions: Iterable[SporeCondition] | None = None,
    splits: tuple[str, ...] = ("train", "val", "test"),
    split_fractions: tuple[float, ...] = (0.7, 0.15, 0.15),
) -> list[Path]:
    output_dir = Path(output_dir)
    if count < 1:
        raise ValueError("count must be positive")
    if len(splits) != len(split_fractions):
        raise ValueError("splits and split_fractions must have the same length")

    condition_list = tuple(conditions or default_conditions())
    if not condition_list:
        raise ValueError("at least one spore condition is required")
    _validate_conditions(condition_list)

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
            condition = condition_list[int(rng.integers(0, len(condition_list)))]
            params = _params_for_condition(condition, rng, image_size_px)
            scene_seed = seed + index * 2 + 1
            image_seed = seed + index * 2 + 2
            scene = SC.build("spores", seed=scene_seed, **params["scene"])
            result = I.simulate(
                scene.material_index,
                scene.material_names,
                height_nm=scene.height_nm,
                beam=params["beam"],
                detector=params["detector"],
                n_electrons=params["n_electrons"],
                seed=image_seed,
            )
            stem = f"spore_{split}_{local_index:05d}"
            image_path = output_dir / "images" / split / f"{stem}.tif"
            label_path = output_dir / "labels" / split / f"{stem}.txt"
            truth_path = output_dir / "truth" / split / f"{stem}.json"

            tifffile.imwrite(image_path, uint8_image(result.image))
            label_path.write_text(_yolo_labels(scene), encoding="utf-8")
            objects = _truth_objects(scene)
            truth = _truth_payload(
                stem=stem,
                split=split,
                condition=condition,
                scene=scene,
                result=result,
                objects=objects,
                params=params,
            )
            truth_path.write_text(json.dumps(truth, indent=2), encoding="utf-8")

            item = {
                "id": index + 1,
                "split": split,
                "stem": stem,
                "image": str(image_path.relative_to(output_dir)),
                "labels": str(label_path.relative_to(output_dir)),
                "truth": str(truth_path.relative_to(output_dir)),
                "n_spores": len(objects),
                "size_class": condition.size_class,
                "length_nm": SIZE_CLASSES_NM[condition.size_class][0],
                "width_nm": SIZE_CLASSES_NM[condition.size_class][1],
                "pixel_size_nm": condition.pixel_size_nm,
                "clustering": condition.clustering,
                "debris": condition.debris,
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
                "generator": "acorn_sem_sim.spore_dataset",
                "class_names": {"0": "spore", "1": "debris"},
                "geometry_truth": ["major_nm", "minor_nm", "ecd_nm", "area_nm2"],
                "items": manifest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_data_yaml(output_dir, splits)
    return written


def _params_for_condition(
    condition: SporeCondition,
    rng: np.random.Generator,
    image_size_px: int,
) -> dict:
    length_nm, width_nm = SIZE_CLASSES_NM[condition.size_class]
    fov_um = image_size_px * condition.pixel_size_nm / 1000.0
    density = float(rng.uniform(0.28, 0.85))
    n_spores = int(np.clip(round(density * fov_um * fov_um), 4, 120))
    coated = bool(rng.random() < 0.45)
    beam = I.Beam(
        E0_kev=float(rng.uniform(3.0, 6.0)) if coated else float(rng.uniform(0.8, 1.5)),
        pixel_size_nm=float(condition.pixel_size_nm),
        electrons_per_px=float(rng.choice([300, 500, 800, 1200])),
        probe_nm=1.0,
    )
    detector = I.Detector(
        "TLD",
        charging=0.0 if coated else float(rng.choice([0.0, 0.0, 0.4, 0.8])),
        shadowing=float(rng.uniform(0.25, 0.7)),
        azimuth_deg=float(rng.uniform(0, 360)),
        elevation_deg=float(rng.uniform(15, 40)),
    )
    scene = {
        "shape": (int(image_size_px), int(image_size_px)),
        "pixel_size_nm": float(condition.pixel_size_nm),
        "n_spores": n_spores,
        "clustering": CLUSTERING_LEVELS[condition.clustering],
        "length_nm": float(length_nm),
        "width_nm": float(width_nm),
        "coating_nm": float(rng.uniform(8, 16)) if coated else 0.0,
        "coating": "gold",
        "substrate": str(rng.choice(["carbon", "biology", "resin"])),
        "substrate_texture_nm": float(rng.uniform(120, 380)),
        "surface_texture_nm": float(rng.uniform(35, 70)),
        "debris_clumps": int(rng.integers(3, 14)) if condition.debris else 0,
        "debris_grain_nm": float(rng.uniform(45, 110)),
        "max_tilt_deg": float(rng.uniform(0, 45)),
        "stacking": float(rng.uniform(0.4, 1.0)),
    }
    return {"scene": scene, "beam": beam, "detector": detector, "n_electrons": 9000}


def _truth_payload(
    *,
    stem: str,
    split: str,
    condition: SporeCondition,
    scene: SC.Scene,
    result: I.SEMImage,
    objects: list[dict],
    params: dict,
) -> dict:
    return {
        "version": 1,
        "stem": stem,
        "split": split,
        "specimen_kind": "spore",
        "image_shape_px": list(result.image.shape),
        "pixel_size_nm": scene.pixel_size_nm,
        "geometry_targets_nm": ["major_nm", "minor_nm", "ecd_nm"],
        "condition": {
            "size_class": condition.size_class,
            "length_nm": SIZE_CLASSES_NM[condition.size_class][0],
            "width_nm": SIZE_CLASSES_NM[condition.size_class][1],
            "pixel_size_nm": condition.pixel_size_nm,
            "clustering": condition.clustering,
            "debris": condition.debris,
        },
        "beam": vars(params["beam"]),
        "detector": vars(params["detector"]),
        "scene_meta": {k: v for k, v in scene.meta.items() if k != "objects"},
        "objects": objects,
    }


def _truth_objects(scene: SC.Scene) -> list[dict]:
    truth: list[dict] = []
    for i, obj in enumerate(scene.meta.get("objects", [])):
        cy = float(obj["cy_px"])
        cx = float(obj["cx_px"])
        major_px = float(obj["major_nm"]) / (2.0 * scene.pixel_size_nm)
        minor_px = float(obj["minor_nm"]) / (2.0 * scene.pixel_size_nm)
        reach = max(major_px, minor_px)
        h, w = scene.material_index.shape
        fully_in_frame = (
            cx - reach >= 0
            and cy - reach >= 0
            and cx + reach <= w
            and cy + reach <= h
        )
        truth.append(
            {
                "id": i + 1,
                "class_name": "spore",
                "cy_px": cy,
                "cx_px": cx,
                "fully_in_frame": fully_in_frame,
                "major_nm": float(obj["major_nm"]),
                "minor_nm": float(obj["minor_nm"]),
                "ecd_nm": float(obj["ecd_nm"]),
                "area_nm2": float(obj["area_nm2"]),
                "angle_rad": float(obj["angle_rad"]),
            }
        )
    return truth


def _yolo_labels(scene: SC.Scene) -> str:
    names = scene.label_names or scene.material_names
    lines: list[str] = []
    for obj in scene.meta.get("objects", []):
        poly = _ellipse_polygon(obj, scene.pixel_size_nm, scene.material_index.shape)
        lines.append("0 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))

    for label_index, name in enumerate(names):
        if name != "debris":
            continue
        lab, _n = ndimage.label(scene.material_index == label_index)
        for poly in _polys_from_labels(lab):
            lines.append("1 " + " ".join(f"{x:.6f} {y:.6f}" for x, y in poly))
    return "\n".join(lines) + ("\n" if lines else "")


def _ellipse_polygon(
    obj: dict,
    pixel_size_nm: float,
    image_shape_px: tuple[int, int],
    vertices: int = 48,
) -> list[tuple[float, float]]:
    h, w = image_shape_px
    cx = float(obj["cx_px"])
    cy = float(obj["cy_px"])
    a = float(obj["major_nm"]) / (2.0 * pixel_size_nm)
    b = float(obj["minor_nm"]) / (2.0 * pixel_size_nm)
    angle = float(obj["angle_rad"])
    theta = np.linspace(0.0, 2.0 * np.pi, vertices, endpoint=False)
    xs = cx + a * np.cos(theta) * np.cos(angle) - b * np.sin(theta) * np.sin(angle)
    ys = cy + a * np.cos(theta) * np.sin(angle) + b * np.sin(theta) * np.cos(angle)
    xs = np.clip(xs, 0.0, float(w - 1)) / float(w)
    ys = np.clip(ys, 0.0, float(h - 1)) / float(h)
    return list(zip(xs.tolist(), ys.tolist()))


def _polys_from_labels(lab: np.ndarray, min_px: int = 60) -> list[list[tuple[float, float]]]:
    out = []
    h, w = lab.shape
    for i in range(1, int(lab.max()) + 1):
        mask = lab == i
        if int(mask.sum()) < min_px:
            continue
        contours = measure.find_contours(mask.astype(float), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        if len(contour) > 60:
            contour = contour[:: max(1, len(contour) // 60)]
        if len(contour) < 6:
            continue
        out.append([(float(x) / w, float(y) / h) for y, x in contour])
    return out


def _validate_conditions(conditions: Iterable[SporeCondition]) -> None:
    for condition in conditions:
        if condition.size_class not in SIZE_CLASSES_NM:
            known = ", ".join(sorted(SIZE_CLASSES_NM))
            raise ValueError(f"unknown size class {condition.size_class!r}; expected one of {known}")
        if condition.clustering not in CLUSTERING_LEVELS:
            known = ", ".join(sorted(CLUSTERING_LEVELS))
            raise ValueError(f"unknown clustering {condition.clustering!r}; expected one of {known}")


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


def _write_data_yaml(output_dir: Path, splits: tuple[str, ...]) -> None:
    payload = {
        "path": str(output_dir),
        "train": "images/train" if "train" in splits else None,
        "val": "images/val" if "val" in splits else None,
        "test": "images/test" if "test" in splits else None,
        "nc": 2,
        "names": {0: "spore", 1: "debris"},
        "truth_dir": "truth",
        "geometry_targets_nm": ["major_nm", "minor_nm", "ecd_nm"],
    }
    (output_dir / "data.yaml").write_text(
        yaml.safe_dump({k: v for k, v in payload.items() if v is not None}, sort_keys=False),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a bacterial-spore SEM benchmark dataset.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=640)
    args = parser.parse_args(argv)

    paths = generate_spore_dataset(
        args.output,
        args.count,
        seed=args.seed,
        image_size_px=args.image_size,
    )
    print(f"Wrote {len(paths)} spore images plus labels, truth, and manifests to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
