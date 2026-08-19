"""Disk export helpers for cryo-TEM simulation output."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import numpy as np
import tifffile

from acorn_tem_sim.simulator import Micrograph, Specimen, resolve, simulate_micrograph


def uint8_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32, copy=False)
    arr = arr - float(arr.min())
    arr = arr / (float(arr.max()) + 1e-8)
    return (np.clip(arr, 0, 1) * 255).astype(np.uint8)


def generate_tem_dataset(
    output_dir: Path,
    count: int,
    cfg: dict,
    specimen: Specimen,
    *,
    save_layers: bool = True,
    seed: int = 0,
) -> list[Path]:
    output_dir = Path(output_dir)
    if str(cfg.get("simulation_path", "fast")).lower() == "custom":
        return _generate_custom_dataset(output_dir, count, cfg, specimen, save_layers=save_layers, seed=seed)
    image_dir = output_dir / "images"
    layer_dir = output_dir / "layers"
    image_dir.mkdir(parents=True, exist_ok=True)
    if save_layers:
        layer_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    metadata = []
    for i in range(count):
        spec = Specimen(
            kind=specimen.kind,
            ice_thickness_nm=specimen.ice_thickness_nm,
            n_particles=specimen.n_particles,
            diameter_nm_mean=specimen.diameter_nm_mean,
            diameter_nm_sd=specimen.diameter_nm_sd,
            plga_mip_v=specimen.plga_mip_v,
            ice_mip_v=specimen.ice_mip_v,
            lipid_mip_v=specimen.lipid_mip_v,
            protein_mip_v=specimen.protein_mip_v,
            bacteria_mip_v=specimen.bacteria_mip_v,
            membrane_thickness_nm=specimen.membrane_thickness_nm,
            oligomer_count=specimen.oligomer_count,
            pdb_path=specimen.pdb_path,
            solvent_noise=specimen.solvent_noise,
            solvent_corr_a=specimen.solvent_corr_a,
            allow_overlap=specimen.allow_overlap,
            seed=specimen.seed + i,
        )
        micrograph = simulate_micrograph(cfg, spec, seed=seed + i)
        stem = f"temsim_{i:05d}"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(image_path, uint8_image(micrograph.image))
        image_paths.append(image_path)
        layers = _write_layers(layer_dir, stem, micrograph) if save_layers else {}
        metadata.append(
            {
                "id": i + 1,
                "file_name": image_path.name,
                "defocus_um": micrograph.defocus_um,
                "pixel_size_a": cfg["pixel_size_a"],
                "voltage_kv": cfg["voltage_kv"],
                "dose_e_per_a2": cfg["total_dose_e_per_a2"],
                "detector_model": cfg["detector_model"],
                "layers": {name: path.name for name, path in layers.items()},
            }
        )
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"config": _jsonable_cfg(cfg), "specimen": specimen.__dict__, "items": metadata}, fh, indent=2)
    return image_paths


def _generate_custom_dataset(
    output_dir: Path,
    count: int,
    cfg: dict,
    specimen: Specimen,
    *,
    save_layers: bool,
    seed: int,
) -> list[Path]:
    script_path = Path(str(cfg.get("custom_script_path", ""))).expanduser()
    if not script_path.exists():
        raise FileNotFoundError("Choose a Custom Python script before using the custom TEM path.")
    spec = importlib.util.spec_from_file_location("acorn_tem_custom_recipe", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load custom TEM script: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "generate_dataset"):
        paths = module.generate_dataset(
            output_dir=output_dir,
            count=count,
            cfg=cfg,
            specimen=specimen,
            save_layers=save_layers,
            seed=seed,
        )
        return [Path(p) for p in paths]
    if not hasattr(module, "generate_image"):
        raise AttributeError("Custom TEM script must define generate_dataset(...) or generate_image(...).")

    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    metadata = []
    for i in range(count):
        arr = module.generate_image(cfg=cfg, specimen=specimen, seed=seed + i, index=i)
        image = np.asarray(arr, dtype=np.float32)
        stem = f"temsim_{i:05d}"
        path = image_dir / f"{stem}.tif"
        tifffile.imwrite(path, uint8_image(image))
        paths.append(path)
        metadata.append({"id": i + 1, "file_name": path.name, "custom_script": str(script_path)})
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"config": _jsonable_cfg(cfg), "specimen": specimen.__dict__, "items": metadata}, fh, indent=2)
    return paths


def _write_layers(layer_dir: Path, stem: str, micrograph: Micrograph) -> dict[str, Path]:
    layers = {
        "potential": micrograph.potential,
        "ideal": micrograph.ideal,
        "counts": micrograph.counts,
        "label": micrograph.label,
    }
    paths = {}
    for name, arr in layers.items():
        path = layer_dir / f"{stem}_{name}.tif"
        tifffile.imwrite(path, uint8_image(arr))
        paths[name] = path
    return paths


def _jsonable_cfg(cfg: dict) -> dict:
    return {k: v for k, v in cfg.items() if k != "_derived"} | {"_derived": cfg.get("_derived", {})}


def default_config(**answers) -> dict:
    return resolve(answers=answers)
