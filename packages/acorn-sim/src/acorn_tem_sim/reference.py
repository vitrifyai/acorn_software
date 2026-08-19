#!/usr/bin/env python3
"""
Level-1 reference router: point at a real image, say 'fib' or 'cryoem', pass the
parameters you know, and get N fresh simulated images at the same settings.

Flow:
    1. load the reference -> canvas size (+ header params for cryo MRC/DM4)
    2. merge:  header  <  your params        (you override anything)
    3. route:  cryoem -> cryotem ,  fib -> fibsim
    4. generate N images with new seeds (new specimen realisations, same style)
    5. write PNGs (+ MRC for cryo), ground-truth labels, a montage, and a manifest

Level 1 replicates the acquisition SETTINGS; it does not measure the reference
(that is Level 2 / calibration). New images share the reference's parameters and
canvas size, not its exact scene.

Usage:
    python reference_router.py REF.mrc  --modality cryoem --n 6 \
        --param pixel_size_a=1.5 --param total_dose_e_per_a2=40 --param n_particles=40
    python reference_router.py REF.png  --modality fib --n 6 \
        --param sample=material --param pixel_size_nm=5 --param curtain_strength=0.2
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

# cryo-TEM
from acorn_tem_sim.engine import resolve, Specimen, simulate_micrograph, probe_metadata
from acorn_tem_sim.engine.setup import FIELDS as CRYO_FIELDS
# FIB-SEM
from acorn_fib_sim.simulator import SimConfig, ImagingConfig, MillingConfig, SceneConfig, simulate
# Level-2 calibration
from acorn_tem_sim.engine.characterize import characterize_cryo, characterize_fib


# --------------------------------------------------------------------------
# Reference loading (shape + any trustworthy header params)
# --------------------------------------------------------------------------
def load_reference(path):
    """Return (gray_array_or_None, ProbedMeta-or-None, (rows, cols))."""
    path = Path(path)
    ext = path.suffix.lower()
    arr, meta = None, None
    if ext in (".mrc", ".mrcs", ".map", ".rec", ".ali", ".st", ".dm4", ".dm3"):
        meta = probe_metadata(path)
        if ext.startswith(".mrc") or ext in (".map", ".rec", ".ali", ".st"):
            import mrcfile
            with mrcfile.open(path, permissive=True) as m:
                arr = None if m.data is None else np.asarray(m.data, dtype=np.float32)
        else:
            from ncempy.io import dm
            with dm.fileDM(str(path)) as f:
                arr = np.asarray(f.getDataset(0)["data"], dtype=np.float32)
    elif ext in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
        from PIL import Image
        arr = np.asarray(Image.open(path).convert("F"), dtype=np.float32)
    else:
        raise ValueError(f"unsupported reference format: {ext}")

    if arr is not None and arr.ndim > 2:      # take first 2-D plane if a stack
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    shape = tuple(arr.shape[-2:]) if arr is not None else (
        meta.shape if meta and meta.shape else (512, 512))
    return arr, meta, shape


# --------------------------------------------------------------------------
# Param routing helpers
# --------------------------------------------------------------------------
def _to_native(obj):
    """Recursively convert numpy scalars/arrays to native types for YAML."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _coerce(v):
    """Turn a CLI string into bool/int/float where possible."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    for cast in (int, float):
        try:
            return cast(s)
        except ValueError:
            pass
    return s


def _dataclass_fields(cls):
    return {f.name for f in dataclasses.fields(cls)}


# --------------------------------------------------------------------------
# Cryo-EM branch
# --------------------------------------------------------------------------
_CRYO_SETUP_KEYS = {f.name for group in CRYO_FIELDS.values() for f in group}
_SPEC_KEYS = _dataclass_fields(Specimen) - {"seed"}


def _run_cryoem(shape, meta, params, n, defocus_um, out, bfactor=40.0):
    rows, cols = shape
    setup_answers = {k: params[k] for k in params if k in _CRYO_SETUP_KEYS}
    spec_params = {k: params[k] for k in params if k in _SPEC_KEYS}
    # Match the reference canvas unless the user set image_size_px explicitly.
    setup_answers.setdefault("image_size_px", int(min(rows, cols)))

    unknown = set(params) - _CRYO_SETUP_KEYS - _SPEC_KEYS
    if unknown:
        print(f"  ! ignoring unknown cryoem params: {sorted(unknown)}")

    cfg = resolve(answers=setup_answers, meta=meta)
    results = []
    for seed in range(n):
        spec = Specimen(seed=seed, **spec_params)
        m = simulate_micrograph(cfg, spec, defocus_um=defocus_um, bfactor=bfactor, seed=seed)
        _save_image(out / f"sim_{seed:03d}.png", m.image)
        _save_mrc(out / f"sim_{seed:03d}.mrc", m.counts, cfg["pixel_size_a"])
        np.save(out / f"label_{seed:03d}.npy", m.label)
        results.append({"seed": seed, "defocus_um": round(m.defocus_um, 3)})
    return cfg, {"setup": setup_answers, "specimen": spec_params, "per_image": results}


# --------------------------------------------------------------------------
# FIB branch
# --------------------------------------------------------------------------
_IMG_KEYS = _dataclass_fields(ImagingConfig) - {"shape", "seed"}
_MILL_KEYS = _dataclass_fields(MillingConfig)
_SCENE_KEYS = _dataclass_fields(SceneConfig)


def _run_fib(shape, params, n, out):
    sample = params.get("sample", "bio")
    img_p = {k: params[k] for k in params if k in _IMG_KEYS}
    mill_p = {k: params[k] for k in params if k in _MILL_KEYS}
    scene_p = {k: params[k] for k in params if k in _SCENE_KEYS}
    unknown = set(params) - _IMG_KEYS - _MILL_KEYS - _SCENE_KEYS - {"sample"}
    if unknown:
        print(f"  ! ignoring unknown fib params: {sorted(unknown)}")

    results = []
    for seed in range(n):
        cfg = SimConfig(
            sample=sample,
            imaging=ImagingConfig(shape=shape, seed=seed, **img_p),
            milling=MillingConfig(**mill_p),
            scene=SceneConfig(**scene_p),
        )
        r = simulate(cfg)
        _save_image(out / f"sim_{seed:03d}.png", r.image)
        np.save(out / f"label_{seed:03d}.npy", r.truth)
        results.append({"seed": seed})
    resolved = {"sample": sample, "imaging": img_p, "milling": mill_p, "scene": scene_p}
    return resolved, {"resolved": resolved, "per_image": results}


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def _save_image(path, img01):
    import matplotlib.image as mpimg
    mpimg.imsave(path, np.clip(img01, 0, 1), cmap="gray", vmin=0, vmax=1)


def _save_mrc(path, data, pixel_size_a):
    import mrcfile
    with mrcfile.new(path, overwrite=True) as m:
        m.set_data(np.asarray(data, dtype=np.float32))
        m.voxel_size = float(pixel_size_a)


def _montage(out, n, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.set_xticks([]); ax.set_yticks([])
        if i < n:
            ax.imshow(mpimg.imread(out / f"sim_{i:03d}.png"))
            ax.set_title(f"sim_{i:03d}", fontsize=8)
        else:
            ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110, bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def _match_bfactor(cfg, spec_params, target_b, defocus_um, seed=0, iters=3, tol=25.0):
    """Tune the applied microscope B so a generated image's spectral falloff
    matches the reference's `target_b`. Uses secant iteration (the falloff is
    monotonic in applied B). Returns (b_apply, achieved_b)."""
    from acorn_tem_sim.engine.characterize import radial_power, fit_spectral_decay

    def measure(b_apply):
        spec = Specimen(seed=seed, **spec_params)
        m = simulate_micrograph(cfg, spec, defocus_um=defocus_um, bfactor=b_apply, seed=seed)
        kc, Pr = radial_power(m.counts, cfg["pixel_size_a"])
        return fit_spectral_decay(kc, Pr)[0]

    b0, y0 = 0.0, measure(0.0)
    b1, y1 = 400.0, measure(400.0)
    for _ in range(iters):
        if abs(y1 - target_b) < tol or y1 == y0:
            break
        b2 = float(np.clip(b1 + (target_b - y1) * (b1 - b0) / (y1 - y0), 0.0, 2000.0))
        b0, y0, b1, y1 = b1, y1, b2, measure(b2)
    return max(0.0, b1), y1


def _calibrate(arr, modality, meta, shape, params):
    """Measure the reference and fold results into params (user values win).
    Returns (params, measured_dict, bfactor_override)."""
    if arr is None:
        print("  ! calibrate requested but reference has no pixel data — skipping")
        return params, None, 40.0
    bfactor = 40.0
    if modality == "cryoem":
        base_answers = {k: params[k] for k in params if k in _CRYO_SETUP_KEYS}
        base_answers.setdefault("image_size_px", int(min(shape)))
        base_cfg = resolve(answers=base_answers, meta=meta)
        meas = characterize_cryo(arr, base_cfg)
        measured = {}
        if "defocus_min_um" not in params and "defocus_max_um" not in params:
            measured["defocus_min_um"] = meas["defocus_um"]
            measured["defocus_max_um"] = meas["defocus_um"]
        # Closed-loop: tune applied B so the generated falloff matches the ref's.
        if meas.get("bfactor_reliable"):
            spec_params = {k: params[k] for k in params if k in _SPEC_KEYS}
            bfactor, achieved = _match_bfactor(base_cfg, spec_params, meas["bfactor"],
                                               meas["defocus_um"])
            meas["bfactor_applied"] = round(bfactor, 1)
            meas["bfactor_achieved"] = round(achieved, 1)
            b_note = (f"ref B_spec={meas['bfactor']} → applied {bfactor:.0f} "
                      f"(achieved {achieved:.0f}, r²={meas['bfactor_r2']})")
        else:
            bfactor = 40.0
            b_note = f"unreliable (r²={meas['bfactor_r2']})→40"
        print(f"  calibrated (cryo): defocus={meas['defocus_um']} µm "
              f"(score {meas['fit_score']}); B: {b_note}; SNR~{meas['snr_proxy']}")
    else:
        meas = characterize_fib(arr)
        measured = {"mill_axis": meas["mill_axis"],
                    "curtain_strength": meas["curtain_strength"],
                    "electrons_per_pixel": meas["electrons_per_pixel"]}
        print(f"  calibrated (fib): mill_axis={meas['mill_axis']}, "
              f"curtain_strength={meas['curtain_strength']} "
              f"(index {meas['curtain_index']}), dose~{meas['electrons_per_pixel']} e/px")
    meas.pop("_spectrum", None)
    return {**measured, **params}, meas, bfactor      # user params override measured


def simulate_from_reference(image_path, modality, params=None, n=10,
                            defocus_um=None, out_dir="sim_out", calibrate=False):
    params = {k: _coerce(v) for k, v in (params or {}).items()}
    modality = modality.lower()
    if modality not in ("cryoem", "fib"):
        raise ValueError("modality must be 'cryoem' or 'fib'")

    arr, meta, shape = load_reference(image_path)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    print(f"Reference {image_path}: canvas {shape}"
          + (f", header pixel_size={meta.pixel_size_a}" if meta and meta.pixel_size_a else ""))
    if meta:
        for w in meta.warnings:
            print(f"  ! {w}")

    measured, bfactor = None, 40.0
    if calibrate:
        params, measured, bfactor = _calibrate(arr, modality, meta, shape, params)

    if modality == "cryoem":
        resolved, manifest = _run_cryoem(shape, meta, params, n, defocus_um, out, bfactor)
        px = resolved["pixel_size_a"]
    else:
        resolved, manifest = _run_fib(shape, params, n, out)
        px = None

    manifest = {"modality": modality, "reference": str(image_path),
                "canvas": list(shape), "n": n, "calibrated": bool(calibrate),
                "measured": measured, **manifest}
    import yaml
    (out / "manifest.yaml").write_text(yaml.safe_dump(_to_native(manifest), sort_keys=False))
    _montage(out, n, out / "montage.png")
    print(f"Wrote {n} images + labels + manifest.yaml + montage.png to {out}/")
    return out


def _cli():
    ap = argparse.ArgumentParser(description="Level-1 reference-matched simulator")
    ap.add_argument("image", help="reference image (mrc/dm4/tif/png)")
    ap.add_argument("--modality", required=True, choices=["cryoem", "fib"])
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--defocus_um", type=float, default=None,
                    help="cryo: fix defocus (else sampled from config range)")
    ap.add_argument("--param", action="append", default=[],
                    metavar="KEY=VALUE", help="repeatable simulator parameter")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure the reference (Level 2) and match its statistics")
    ap.add_argument("--out_dir", default="sim_out")
    args = ap.parse_args()
    params = dict(p.split("=", 1) for p in args.param)
    simulate_from_reference(args.image, args.modality, params, n=args.n,
                            defocus_um=args.defocus_um, out_dir=args.out_dir,
                            calibrate=args.calibrate)


if __name__ == "__main__":
    _cli()
