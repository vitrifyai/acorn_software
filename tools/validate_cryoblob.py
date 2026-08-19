"""CryoBlob validation on one synthetic run folder (adapted to ACORN run layout).

Scores CryoBlob detections vs ground truth (from dump_ground_truth.py CSVs) and
compares against a scikit-image blob_log baseline. Precision/recall/F1,
localization error, size error, merge/split, IoU, Dice; saves a results table
and per-image overlay figures.

Layout handled here:
  RUN_DIR/                      (an ACORN sim run's images/ folder)
    temsim_0000N.tif
    temsim_0000N_ground_truth.csv          (center_x_px, center_y_px, radius_px)
    temsim_0000N_cryoblob.csv              (per-image) OR cryoblob_results.csv (combined)
"""
import os, glob, sys
import numpy as np
import pandas as pd
import tifffile
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.feature import blob_log
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

RUN_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images"
MICROGRAPH_SUFFIX = ".tif"
PARTICLES_ARE_DARK = True
TOL_FACTOR = 0.5
SKIMAGE_THRESHOLD = 0.02
DOWNSAMPLE = 0.5
SAVE_OVERLAYS = True
OUT_DIR = os.path.join(RUN_DIR, "scoring_output")


def find_stems(run_dir):
    stems = [os.path.basename(p).replace("_ground_truth.csv", "")
             for p in glob.glob(os.path.join(run_dir, "*_ground_truth.csv"))]
    return sorted(set(stems))


def load_micrograph(run_dir, stem):
    return tifffile.imread(os.path.join(run_dir, stem + MICROGRAPH_SUFFIX)).astype(float)


def load_gt(run_dir, stem):
    g = pd.read_csv(os.path.join(run_dir, stem + "_ground_truth.csv"))
    return g[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)


def load_all_cryoblob(run_dir):
    """Concat CryoBlob CSV(s). If CRYOBLOB_CSV (argv[2]) is set, use only that;
    otherwise every top-level *_cryoblob.csv (+ cryoblob_results.csv) in run_dir.
    Skips the *_outputs/ subfolders' *_particles.csv."""
    if len(sys.argv) > 2:
        files = [sys.argv[2]]
    else:
        files = sorted(glob.glob(os.path.join(run_dir, "*_cryoblob.csv"))) + \
                sorted(glob.glob(os.path.join(run_dir, "cryoblob_results.csv")))
    frames = [pd.read_csv(f) for f in files if os.path.isfile(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cryoblob_dets(df, stem):
    if df.empty:
        return np.empty((0, 3))
    base = df["File Location"].map(lambda s: os.path.basename(str(s)))
    # Prefer rows for the raw micrograph (temsim_0000N.tif). Fall back to the
    # annotated PNG only if there are no raw-tif rows (old runs pointed there).
    sub = df[base == stem + MICROGRAPH_SUFFIX]
    if len(sub) == 0:
        sub = df[base == stem + "_annotated.png"]
    if len(sub) == 0:
        return np.empty((0, 3))
    # Guard against bogus embedded pixel sizes (some derived PNGs carry garbage).
    sub = sub[(sub["Pixel Size X (nm/px)"] > 0) & (sub["Pixel Size X (nm/px)"] < 100)]
    if len(sub) == 0:
        return np.empty((0, 3))
    px = sub["Pixel Size X (nm/px)"].to_numpy()
    x = sub["Center X (nm)"].to_numpy() / px
    y = sub["Center Y (nm)"].to_numpy() / px
    r = sub["Radius (nm)"].to_numpy() / px
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(r)
    return np.column_stack([x[ok], y[ok], r[ok]])


def skimage_dets(img, gt_radii_px):
    im = (img - img.min()) / (np.ptp(img) + 1e-9)
    if PARTICLES_ARE_DARK:
        im = 1.0 - im
    sm = gaussian(rescale(im, DOWNSAMPLE, anti_aliasing=True), sigma=2)
    rmin, rmax = gt_radii_px.min(), gt_radii_px.max()
    smin = max(2.0, (rmin / np.sqrt(2)) * DOWNSAMPLE)
    smax = max(smin + 1, (rmax / np.sqrt(2)) * DOWNSAMPLE)
    b = blob_log(sm, min_sigma=smin, max_sigma=smax, num_sigma=8, threshold=SKIMAGE_THRESHOLD)
    if len(b) == 0:
        return np.empty((0, 3))
    s = 1.0 / DOWNSAMPLE
    return np.column_stack([b[:, 1] * s, b[:, 0] * s, b[:, 2] * np.sqrt(2) * s])


def match(det, gt, tol_factor):
    if len(det) == 0:
        return 0, 0, len(gt), [], [], 0, 0
    if len(gt) == 0:
        return 0, len(det), 0, [], [], 0, 0
    D = np.linalg.norm(det[:, None, :2] - gt[None, :, :2], axis=2)
    tol = tol_factor * gt[:, 2][None, :]
    allowed = D <= tol
    splits = int((allowed.sum(0) > 1).sum())
    merges = int((allowed.sum(1) > 1).sum())
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    ud, ug, loc, sz = set(), set(), [], []
    for di, gi in order:
        if di in ud or gi in ug or D[di, gi] > tol[0, gi]:
            continue
        ud.add(di); ug.add(gi)
        loc.append(D[di, gi]); sz.append(abs(det[di, 2] - gt[gi, 2]))
    tp = len(ud)
    return tp, len(det) - tp, len(gt) - len(ug), loc, sz, merges, splits


def mask_from_circles(arr, shape):
    m = np.zeros(shape, bool); H, W = shape
    for x, y, r in arr:
        x0, x1 = max(0, int(x - r - 1)), min(W, int(x + r + 2))
        y0, y1 = max(0, int(y - r - 1)), min(H, int(y + r + 2))
        yy, xx = np.ogrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 <= r * r
    return m


def iou_dice(a, b):
    inter = np.logical_and(a, b).sum()
    return inter / max(1, np.logical_or(a, b).sum()), 2 * inter / max(1, a.sum() + b.sum())


def save_overlay(img, gt, det, title, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img, cmap="gray")
    for x, y, r in gt:
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="yellow", lw=1.2))
    for x, y, r in det:
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="lime", lw=1.0, ls="--"))
    ax.set_title(title + "   (yellow = ground truth, green dashed = detections)")
    ax.axis("off"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    stems = find_stems(RUN_DIR)
    if not stems:
        raise SystemExit(f"No *_ground_truth.csv in {RUN_DIR}")
    print("RUN_DIR:", RUN_DIR)
    print("Scoring stems:", stems)
    cb = load_all_cryoblob(RUN_DIR)

    methods = [("skimage blob_log", False), ("CryoBLOB", True)]
    rows = []
    for label, is_cb in methods:
        T = F = N = MG = SP = 0
        L, Z, IO, DI = [], [], [], []
        for stem in stems:
            img = load_micrograph(RUN_DIR, stem)
            gt = load_gt(RUN_DIR, stem)
            det = cryoblob_dets(cb, stem) if is_cb else skimage_dets(img, gt[:, 2])
            tp, fp, fn, loc, sz, mg, sp = match(det, gt, TOL_FACTOR)
            gm = mask_from_circles(gt, img.shape)
            dm = mask_from_circles(det, img.shape)
            iou, dice = iou_dice(gm, dm)
            T += tp; F += fp; N += fn; MG += mg; SP += sp
            L += loc; Z += sz; IO.append(iou); DI.append(dice)
            print(f"  [{label}] {stem}: det={len(det)} gt={len(gt)} TP={tp} FP={fp} FN={fn} IoU={iou:.2f}")
            if SAVE_OVERLAYS:
                safe = label.replace(" ", "_")
                save_overlay(img, gt, det, f"{stem}  {label}",
                             os.path.join(OUT_DIR, f"{stem}_{safe}.png"))
        P = T / (T + F) if T + F else 0
        R = T / (T + N) if T + N else 0
        Fp = 2 * P * R / (P + R) if P + R else 0
        rows.append(dict(method=label, precision=round(P, 3), recall=round(R, 3),
                         f1=round(Fp, 3), loc_err_px=round(np.mean(L), 1) if L else None,
                         size_err_px=round(np.mean(Z), 1) if Z else None,
                         merges=MG, splits=SP,
                         mean_IoU=round(np.mean(IO), 3), mean_Dice=round(np.mean(DI), 3),
                         TP=T, FP=F, FN=N))
    res = pd.DataFrame(rows)
    print("\n" + res.to_string(index=False))
    out_csv = os.path.join(OUT_DIR, "results_table.csv")
    res.to_csv(out_csv, index=False)
    print(f"\nSaved table to {out_csv}")
    if SAVE_OVERLAYS:
        print(f"Saved overlays to {OUT_DIR}")


if __name__ == "__main__":
    main()
