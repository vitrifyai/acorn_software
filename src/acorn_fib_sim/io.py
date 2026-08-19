"""Disk export helpers for FIB simulation output."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import tifffile

from acorn_fib_sim.simulator import SimConfig, SimResult, simulate


def uint8_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    arr = arr - float(arr.min())
    arr = arr / (float(arr.max()) + 1e-8)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def generate_fib_dataset(output_dir: Path, count: int, config: SimConfig, *, save_layers: bool = True) -> list[Path]:
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    layer_dir = output_dir / "layers"
    image_dir.mkdir(parents=True, exist_ok=True)
    if save_layers:
        layer_dir.mkdir(parents=True, exist_ok=True)

    image_paths: list[Path] = []
    metadata = []
    for i in range(count):
        cfg = SimConfig(
            sample=config.sample,
            imaging=type(config.imaging)(
                **{
                    **asdict(config.imaging),
                    "seed": int(config.imaging.seed) + i,
                }
            ),
            milling=config.milling,
            scene=config.scene,
        )
        result = simulate(cfg)
        stem = f"fibsim_{i:05d}"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(image_path, uint8_image(result.image))
        image_paths.append(image_path)

        layers = {}
        if save_layers:
            layers = _write_layers(layer_dir, stem, result)
        metadata.append(
            {
                "id": i + 1,
                "file_name": image_path.name,
                "sample": cfg.sample,
                "height": int(result.image.shape[0]),
                "width": int(result.image.shape[1]),
                "pixel_size_nm": cfg.imaging.pixel_size_nm,
                "layers": {name: path.name for name, path in layers.items()},
                "config": _jsonable_config(cfg),
            }
        )

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return image_paths


def _write_layers(layer_dir: Path, stem: str, result: SimResult) -> dict[str, Path]:
    layers = {
        "truth": result.truth,
        "edges": result.edges,
        "milled": result.milled / (float(result.milled.max()) + 1e-8),
        "artifact_delta": np.abs(result.milled / (float(result.milled.max()) + 1e-8) - result.truth),
    }
    paths = {}
    for name, arr in layers.items():
        path = layer_dir / f"{stem}_{name}.tif"
        tifffile.imwrite(path, uint8_image(arr))
        paths[name] = path
    return paths


def _jsonable_config(config: SimConfig) -> dict:
    return {
        "sample": config.sample,
        "imaging": asdict(config.imaging),
        "milling": asdict(config.milling),
        "scene": asdict(config.scene),
    }
