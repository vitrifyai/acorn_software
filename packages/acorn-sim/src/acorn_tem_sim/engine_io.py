"""Engine-backed generation for ACORN: advanced cryo-TEM + 4D-STEM.

Uses the vendored physics engine (`acorn_tem_sim.engine`, the full cryotem) to
add capabilities the legacy `io.generate_tem_dataset` path does not have:
detector-specific DQE/MTF/NPS, microscope presets, multislice, inelastic loss,
dose-dependent radiation damage, composable scenes (nanoparticles + crystalline
ice contamination + a bacterial cell), GPU acceleration, and full 4D-STEM
(BF/ADF/HAADF/DPC/iDPC). Output matches the ACORN dataset format (uint8 TIFFs +
metadata.json) so results open in the viewer with calibration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from acorn_tem_sim import engine as E
from acorn_tem_sim.engine.detector import expose, to_display
from acorn_tem_sim.engine.scene import (Scene, Nanoparticles, Contamination, Cell,
                                        simulate_scene, scene_slabs, scene_annotations)
from acorn_tem_sim.engine.stem import stem_4d, virtual_detectors
from acorn_tem_sim.io import uint8_image


def _annotation_color(label: str) -> str:
    """Same colour a hand-drawn annotation of this label would get in ACORN."""
    try:
        from acorn.render.palette import color_for_label
        return color_for_label(label)
    except Exception:      # acorn core unavailable (engine used standalone)
        return "#E8833A"

def _write_acorn_sidecar(image_path: Path, footprints, pixel_size_nm: float, shape):
    """Write the ACORN .<stem>.acorn.json sidecar: perfect ground-truth ROI
    polygons (for YOLO/UNet training) + pixel-size calibration."""
    h, w = shape
    anns = []
    for label, verts in footprints:
        xs = [x for x, _ in verts]; ys = [y for _, y in verts]
        if max(xs) < 0 or min(xs) > w or max(ys) < 0 or min(ys) > h:
            continue                                    # object entirely off-frame
        vv = [[float(min(max(x, 0), w - 1)), float(min(max(y, 0), h - 1))] for x, y in verts]
        anns.append({"type": "roi", "vertices": vv, "area_nm2": 0.0, "stats": {},
                     "color": _annotation_color(label), "linewidth": 1.5, "label": label})
    side = Path(image_path).parent / f".{Path(image_path).stem}.acorn.json"
    side.write_text(json.dumps({"version": 3, "annotations": anns,
                                "pixel_size_nm": float(pixel_size_nm),
                                "exclude_zone": None, "crop_region": None}))
    return len(anns)


def _scan_footprints(footprints, fd):
    """Map scene footprints (specimen-pixel coords) into the 4D-STEM scan-image
    frame. Virtual-detector images are sampled on the scan grid, so a specimen
    position `p` (Å) lands at scan pixel `(p - scan_origin) / scan_step`."""
    x0 = float(fd.scan_pos_a[0, 0, 0]); y0 = float(fd.scan_pos_a[0, 0, 1])
    step = float(fd.scan_step_a); px = float(fd.px)
    out = []
    for label, verts in footprints:
        vv = [[(vx * px - x0) / step, (vy * px - y0) / step] for vx, vy in verts]
        out.append((label, vv))
    return out


def _build_scene(params: dict, seed: int) -> Scene:
    """Compose a Scene from CLU-style params (any combination of components)."""
    comps = []
    kind = str(params.get("specimen_kind", "plga")).lower()
    if kind in ("bacteria", "cell", "bacterium"):
        comps.append(Cell(species=str(params.get("species", "e_coli")),
                          angle_deg=float(params.get("angle_deg", 15.0)),
                          n_ribosomes=int(params.get("n_ribosomes", 300))))
    if kind in ("plga", "nanoparticles") or params.get("add_nanoparticles"):
        comps.append(Nanoparticles(n=int(params.get("n_particles", 30)),
                                    diameter_nm_mean=float(params.get("diameter_nm_mean", 30.0)),
                                    diameter_nm_sd=float(params.get("diameter_nm_sd", 8.0))))
    if kind == "contamination" or params.get("add_contamination"):
        comps.append(Contamination(n=int(params.get("n_contam", 3)),
                                   thickness_nm=float(params.get("contam_thickness_nm", 80.0))))
    if not comps:                                   # default: PLGA nanoparticles
        comps.append(Nanoparticles(n=int(params.get("n_particles", 30))))
    return Scene(ice_thickness_nm=float(params.get("ice_thickness_nm", 60.0)),
                 components=comps, solvent_noise=float(params.get("solvent_noise", 5.0)),
                 seed=seed)


def _cfg_from_params(params: dict) -> dict:
    return E.resolve(answers={
        "microscope": str(params.get("microscope", "krios")),
        "detector_model": str(params.get("detector_model", "K3")),
        "pixel_size_a": float(params.get("pixel_size_a", 2.0)),
        "image_size_px": int(params.get("image_size_px", 512)),
        "voltage_kv": str(params.get("voltage_kv", "300")),
        # cfg validator floors dose at 0.5; a 0/None here means "noise-free" and
        # is honored separately where the dose is actually applied.
        "total_dose_e_per_a2": max(0.5, float(params.get("total_dose_e_per_a2") or 40.0)),
        "defocus_min_um": float(params.get("defocus_min_um", -1.0)),
        "defocus_max_um": float(params.get("defocus_max_um", -3.0)),
        "energy_filter_ev": float(params.get("energy_filter_ev", 0.0)),
    })


def generate_tem_advanced(output_dir: Path, count: int, params: dict, *,
                          seed: int = 1, save_layers: bool = True) -> list[Path]:
    """Advanced multislice TEM dataset over a composed scene (GPU when available)."""
    output_dir = Path(output_dir)
    img_dir = output_dir / "images"; img_dir.mkdir(parents=True, exist_ok=True)
    layer_dir = output_dir / "layers"
    if save_layers:
        layer_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_params(params)
    dz = float(params.get("slice_thickness_a", 40.0))
    df_lo, df_hi = cfg["defocus_min_um"], cfg["defocus_max_um"]

    paths, meta = [], []
    for i in range(count):
        rng = np.random.default_rng(seed + i)
        scene = _build_scene(params, seed=seed + i)
        df = float(rng.uniform(min(df_lo, df_hi), max(df_lo, df_hi)))
        ideal, info = simulate_scene(cfg, scene, defocus_um=df, dz_a=dz)
        img = to_display(expose(ideal, cfg, rng, dose_scale=info["dose_scale"]))
        stem = f"temsim_{i:05d}"
        p = img_dir / f"{stem}.tif"
        tifffile.imwrite(p, uint8_image(img)); paths.append(p)
        if save_layers:
            tifffile.imwrite(layer_dir / f"{stem}_ideal.tif", uint8_image(ideal))
        # Ground-truth annotations + calibration sidecar (opens labeled in ACORN).
        n_ann = _write_acorn_sidecar(p, scene_annotations((cfg["image_size_px"],) * 2,
                                     cfg["pixel_size_a"], scene),
                                     cfg["pixel_size_a"] / 10.0, (cfg["image_size_px"],) * 2)
        meta.append({"id": i + 1, "file_name": p.name, "defocus_um": round(df, 3),
                     "n_annotations": n_ann,
                     "pixel_size_a": cfg["pixel_size_a"], "detector_model": cfg["detector_model"],
                     "microscope": cfg["_derived"]["microscope"],
                     "components": info["components"], "thickness_nm": info["thickness_nm"],
                     "dose_scale": round(info["dose_scale"], 3)})
    _write_meta(output_dir, cfg, params, meta)
    return paths


def generate_4dstem(output_dir: Path, params: dict, *, seed: int = 1) -> list[Path]:
    """4D-STEM: scan a probe, record the datacube, save all virtual-detector images."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_params(params)
    n = cfg["image_size_px"]; px = cfg["pixel_size_a"]
    dz = float(params.get("slice_thickness_a", 40.0))
    scene = _build_scene(params, seed=seed)
    slabs = list(scene_slabs((n, n), px, dz, scene))
    scan = int(params.get("scan_size", 64))
    fd = stem_4d(cfg, slabs, conv_mrad=float(params.get("conv_mrad", 20.0)),
                 defocus_um=float(params.get("defocus_um", -0.5)),
                 dz_a=dz, scan_shape=(scan, scan),
                 dose_e_per_a2=(float(params["total_dose_e_per_a2"])
                                if params.get("total_dose_e_per_a2") else None),
                 seed=seed, use_gpu=params.get("gpu", None))
    det = virtual_detectors(fd)
    # Ground-truth footprints, mapped from specimen pixels into the scan frame.
    sy, sx = fd.cube.shape[:2]
    scan_foot = _scan_footprints(scene_annotations((n, n), px, scene), fd)
    scan_px_nm = float(fd.scan_step_a) / 10.0              # scan step is the image pixel size
    want = params.get("detectors") or ["BF", "ADF", "HAADF", "CoM_mag", "iDPC"]
    paths = []
    for name in want:
        if name not in det:
            continue
        p = output_dir / f"4dstem_{name}.tif"
        tifffile.imwrite(p, uint8_image(det[name])); paths.append(p)
        _write_acorn_sidecar(p, scan_foot, scan_px_nm, (sy, sx))
    np.save(output_dir / "datacube.npy", fd.cube)          # raw 4D for external tools
    _write_meta(output_dir, cfg, params,
                [{"detector": name, "file_name": f"4dstem_{name}.tif"} for name in want if name in det],
                extra={"modality": "4D-STEM", "scan": [scan, scan],
                       "conv_mrad": params.get("conv_mrad", 20.0),
                       "cube_shape": list(fd.cube.shape)})
    return paths


def _write_meta(output_dir, cfg, params, items, extra=None):
    payload = {"config": {k: v for k, v in cfg.items() if k != "_derived"},
               "_derived": cfg.get("_derived", {}), "params": params, "items": items}
    if extra:
        payload.update(extra)
    with open(Path(output_dir) / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
