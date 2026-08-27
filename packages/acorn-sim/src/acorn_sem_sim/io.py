"""Write SEM simulation output to disk, ground truth included.

Every image ships with the label map that produced it, plus an ACORN-format
annotation sidecar so simulated data drops straight into the same workflow as
real data. The whole point of simulating is that the truth is exact; leaving it
on the floor would waste the exercise.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from . import scenes as _scenes
from .imaging import Beam, Detector, SEMImage, simulate
from .scenes import Scene


def uint8_image(image: np.ndarray) -> np.ndarray:
    """Percentile-stretch to 8-bit for viewing, matching the FIB/TEM exporters."""
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr, [0.1, 99.9])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _polygons_for_label(mask: np.ndarray, min_area_px: int = 12) -> list[list[list[float]]]:
    """Trace each connected region of `mask` to a polygon in ACORN's format."""
    try:
        from scipy.ndimage import find_objects
        from scipy.ndimage import label as cc_label
        from skimage.measure import find_contours
    except ImportError:                       # geometry is optional, images are not
        return []

    lab, _n = cc_label(mask)
    out: list[list[list[float]]] = []
    for i, sl in enumerate(find_objects(lab), start=1):
        if sl is None:
            continue
        sub = (lab[sl] == i)
        if sub.sum() < min_area_px:
            continue
        padded = np.pad(sub.astype(float), 1)
        contours = find_contours(padded, 0.5)
        if not contours:
            continue
        c = max(contours, key=len)
        y0, x0 = sl[0].start - 1, sl[1].start - 1
        pts = [[float(x + x0), float(y + y0)] for y, x in c[::2]]
        if len(pts) >= 3:
            out.append(pts)
    return out


def write_annotations(path: Path, result: SEMImage, scene: Scene | None = None,
                      min_area_px: int = 12) -> int:
    """Write an ACORN annotation sidecar from the ground-truth label map.

    The scene decides what counts as an object and what it is called. Filtering
    on the material name instead -- skipping "vacuum", say -- looks reasonable
    and is wrong: in a porous specimen the pores ARE the objects, they are made
    of vacuum, and that rule silently exported nothing at all.
    """
    names = (scene.label_names or scene.material_names) if scene else result.material_names
    background = scene.background_labels if scene else (0,)

    annotations = []
    for value in np.unique(result.material_index):
        label = int(value)
        if label in background:
            continue
        name = names[label] if label < len(names) else str(label)
        for poly in _polygons_for_label(result.material_index == value, min_area_px):
            annotations.append({"label": name, "polygon": poly, "source": "simulation",
                                "accepted": True})
    path.write_text(json.dumps({"annotations": annotations}, indent=2))
    return len(annotations)


def generate_sem_dataset(output_dir: Path, count: int, params: dict, *,
                         save_layers: bool = True, seed: int = 0) -> list[Path]:
    """Simulate `count` SEM images with ground truth into `output_dir`.

    Each image gets a fresh scene seed so the set is varied, while the beam,
    detector and material choices stay fixed -- which is what makes the set a
    controlled experiment rather than an assortment.
    """
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    if save_layers:
        layer_dir.mkdir(parents=True, exist_ok=True)

    beam = Beam(
        E0_kev=float(params.get("E0_kev", 5.0)),
        pixel_size_nm=float(params.get("pixel_size_nm", 4.0)),
        electrons_per_px=float(params.get("electrons_per_px", 500.0)),
        probe_nm=float(params.get("probe_nm", 1.0)),
    )
    detector = Detector(
        kind=str(params.get("detector", "ETD")),
        bse_mix=float(params.get("bse_mix", 0.15)),
        elevation_deg=float(params.get("elevation_deg", 25.0)),
        azimuth_deg=float(params.get("azimuth_deg", 0.0)),
        asymmetry=float(params.get("asymmetry", 0.30)),
        read_noise_e=float(params.get("read_noise_e", 3.0)),
        scan_jitter_px=float(params.get("scan_jitter_px", 0.0)),
    )

    size = int(params.get("image_size_px", 512))
    kind = str(params.get("scene", "nanoparticles"))
    n_electrons = int(params.get("n_electrons", 40_000))

    paths: list[Path] = []
    metadata = []
    for i in range(int(count)):
        scene = _scenes.build(
            kind,
            shape=(size, size),
            pixel_size_nm=beam.pixel_size_nm,
            seed=seed + i,
            n_particles=int(params.get("n_particles", 40)),
            diameter_nm_mean=float(params.get("diameter_nm_mean", 40.0)),
            diameter_nm_sd=float(params.get("diameter_nm_sd", 12.0)),
            particle=str(params.get("particle", "gold")),
            substrate=str(params.get("substrate", "carbon")),
            relief=bool(params.get("relief", True)),
            n_grains=int(params.get("n_grains", 24)),
            phase_a=str(params.get("phase_a", "iron")),
            phase_b=str(params.get("phase_b", "copper")),
            porosity=float(params.get("porosity", 0.25)),
            matrix=str(params.get("matrix", "alumina")),
            n_cells=int(params.get("n_cells", 14)),
            n_layers=int(params.get("n_layers", 4)),
            n_spores=int(params.get("n_spores", 25)),
            length_nm=float(params.get("length_nm", 1200.0)),
            width_nm=float(params.get("width_nm", 800.0)),
            size_spread=float(params.get("size_spread", 0.15)),
            coating_nm=float(params.get("coating_nm", 10.0)),
            coating=str(params.get("coating", "gold")),
            clustering=float(params.get("clustering", 0.35)),
        )

        result = simulate(scene.material_index, scene.material_names,
                          height_nm=scene.height_nm, beam=beam, detector=detector,
                          seed=seed + i, n_electrons=n_electrons)

        stem = f"semsim_{i:05d}"
        image_path = image_dir / f"{stem}.tif"
        tifffile.imwrite(image_path, uint8_image(result.image))
        paths.append(image_path)

        n_ann = write_annotations(image_dir / f"{stem}.annotations.json",
                                  result, scene)

        if save_layers:
            tifffile.imwrite(layer_dir / f"{stem}_truth.tif",
                             result.material_index.astype(np.uint8))
            tifffile.imwrite(layer_dir / f"{stem}_se.tif", result.se.astype(np.float32))
            tifffile.imwrite(layer_dir / f"{stem}_bse.tif", result.bse.astype(np.float32))
            tifffile.imwrite(layer_dir / f"{stem}_tilt.tif", result.tilt_deg.astype(np.float32))

        metadata.append({
            "id": i + 1,
            "file_name": image_path.name,
            "scene": kind,
            "description": scene.description,
            "annotations": n_ann,
            "pixel_size_nm": scene.pixel_size_nm,
            "width": int(result.image.shape[1]),
            "height": int(result.image.shape[0]),
            "scene_params": scene.meta,
            **result.meta,
        })

    (output_dir / "metadata.json").write_text(json.dumps({
        "generator": "acorn_sem_sim",
        "model": "Monte Carlo electron transport (screened Rutherford + Joy-Luo Bethe)",
        "beam": vars(beam),
        "detector": vars(detector),
        "images": metadata,
    }, indent=2, default=str))
    return paths
