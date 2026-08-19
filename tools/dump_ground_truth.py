#!/usr/bin/env python
"""Dump ground-truth particle centers + radii for legacy TEM-sim run folders.

Reproduces the generator's _place_solid_particles deterministically (same seed
order) so the output exactly matches the placed spheres in each image. Writes
<image>_ground_truth.csv next to each image.

Usage:
    python tools/dump_ground_truth.py <run_folder>
    python tools/dump_ground_truth.py <parent_folder_of_runs>   # processes all
"""
import csv, json, sys
from pathlib import Path
import numpy as np
from acorn_tem_sim.simulator import _sphere_chord, Specimen


def dump_run(run: Path) -> bool:
    meta = run / "metadata.json"
    if not meta.exists():
        return False
    m = json.loads(meta.read_text())
    cfg, sp = m.get("config", {}), m.get("specimen", {})
    if sp.get("kind", "").lower() not in {"plga", "solid", "solid_particles"}:
        print(f"{run.name}: skipped (specimen kind={sp.get('kind')!r} — solid-sphere GT only)")
        return False
    px_a = float(cfg["pixel_size_a"]); px_nm = px_a / 10.0
    H = W = int(cfg["image_size_px"])
    n = int(sp["n_particles"]); mean = float(sp["diameter_nm_mean"]); sd = float(sp["diameter_nm_sd"])
    allow_overlap = bool(sp.get("allow_overlap", Specimen().allow_overlap))
    base_seed = int(sp.get("seed", 1))

    def placed_for(seed):
        rng = np.random.default_rng(seed); placed = []; tries = 0
        while len(placed) < n and tries < max(1, n * 50):
            tries += 1
            d_nm = rng.normal(mean, sd)
            if d_nm <= 2: continue
            r_px = (d_nm * 10.0 / 2.0) / px_a
            cy = rng.uniform(0, H); cx = rng.uniform(0, W)
            if not allow_overlap and any(np.hypot(cy - py, cx - qx) < (r_px + pr) for py, qx, pr in placed):
                continue
            if _sphere_chord((H, W), cy, cx, r_px) is None: continue
            placed.append((cy, cx, r_px))
        return placed

    imgdir = run / "images"
    for i, item in enumerate(m["items"]):
        stem = Path(item["file_name"]).stem
        placed = placed_for(base_seed + i)
        out = imgdir / f"{stem}_ground_truth.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "center_x_px", "center_y_px", "radius_px",
                        "center_x_nm", "center_y_nm", "radius_nm", "diameter_nm"])
            for j, (cy, cx, r_px) in enumerate(placed):
                w.writerow([j, round(cx, 2), round(cy, 2), round(r_px, 3),
                            round(cx * px_nm, 2), round(cy * px_nm, 2),
                            round(r_px * px_nm, 3), round(2 * r_px * px_nm, 3)])
        print(f"  {stem}: seed {base_seed + i} -> {len(placed)} spheres -> {out.name}")
    return True


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    root = Path(sys.argv[1]).expanduser()
    runs = [root] if (root / "metadata.json").exists() else sorted(
        p for p in root.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    if not runs:
        print(f"No run folders (with metadata.json) found under {root}"); sys.exit(1)
    done = 0
    for run in runs:
        print(f"{run.name}:")
        done += dump_run(run)
    print(f"\nDone: ground truth written for {done} run(s).")


if __name__ == "__main__":
    main()
