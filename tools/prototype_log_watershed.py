"""Prototype + validate a LoG+watershed particle picker.

LoG (high precision) recovers well-separated particles; a distance-transform
watershed recovers particles inside overlapping clusters that LoG misses. We KEEP
all LoG detections and ADD only watershed detections that LoG didn't already find.
Scored vs simulator ground truth, compared to LoG-only, on both datasets.
"""
import os, glob, sys
os.environ["JAX_PLATFORM_NAME"] = "gpu"
sys.path.insert(0, "/home/vnw/acorn-cryoblob-plugin/src")
import numpy as np, pandas as pd, tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.transform import rescale
from acorn_cryoblob.thread import _detect_blobs_jax

DS = 4               # downscale for both LoG and watershed
TOL_FACTOR = 0.5
RUNS = {
    "first_dataset (non-overlap)":
        "/nas-158/syn_data/first_dataset/tem_run_20260720_205052_483427/images",
    "second_dataset (overlap)":
        "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images",
}

# LoG params matching the validated run (sigma in downscaled px)
LOG = dict(blob_downscale=float(DS), min_sigma=8.0, max_sigma=47.0,
           threshold_rel=0.15, max_detections=200)


def log_dets(arr):
    """LoG detections in FULL-RES pixels [x, y, r]."""
    rows = np.asarray(_detect_blobs_jax(arr, **LOG))  # [y, x, r] downscaled px
    if len(rows) == 0:
        return np.empty((0, 3))
    return np.column_stack([rows[:, 1] * DS, rows[:, 0] * DS, rows[:, 2] * DS])


def watershed_dets(arr, min_r_full=40.0, max_r_full=700.0):
    """Distance-transform watershed detections in FULL-RES pixels [x, y, r]."""
    small = rescale(arr.astype(np.float32), 1.0 / DS, anti_aliasing=True)
    inv = (small.max() - small)                       # particles are dark -> bright
    inv = (inv - inv.min()) / (np.ptp(inv) + 1e-9)
    sm = gaussian(inv, sigma=2)
    try:
        thr = threshold_otsu(sm)
    except Exception:
        thr = sm.mean() + 0.5 * sm.std()
    mask = sm > thr
    mask = ndi.binary_fill_holes(mask)
    if mask.sum() == 0:
        return np.empty((0, 3))
    dist = ndi.distance_transform_edt(mask)
    min_r_ds = max(3.0, min_r_full / DS)
    peaks = peak_local_max(dist, min_distance=int(min_r_ds), labels=mask,
                           exclude_border=False)
    if len(peaks) == 0:
        return np.empty((0, 3))
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    labels = watershed(-dist, markers, mask=mask)
    out = []
    for rp in ndi.find_objects(labels):
        pass
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lab)
        if len(ys) < 4:
            continue
        area = len(ys)
        r_ds = (area / np.pi) ** 0.5
        r_full = r_ds * DS
        if not (min_r_full <= r_full <= max_r_full):
            continue
        cy, cx = ys.mean(), xs.mean()
        out.append([cx * DS, cy * DS, r_full])
    return np.array(out) if out else np.empty((0, 3))


def merge(log, wat, tol_factor=0.6):
    """Keep all LoG; add watershed dets not near an existing LoG det."""
    if len(log) == 0:
        return wat
    if len(wat) == 0:
        return log
    keep = []
    for w in wat:
        d = np.linalg.norm(log[:, :2] - w[:2], axis=1)
        # reject if within tol*radius of an existing LoG detection
        if np.all(d > tol_factor * np.maximum(w[2], log[:, 2])):
            keep.append(w)
    return np.vstack([log, np.array(keep)]) if keep else log


def match(det, gt):
    if len(det) == 0:
        return 0, 0, len(gt), []
    D = np.linalg.norm(det[:, None, :2] - gt[None, :, :2], axis=2)
    tol = TOL_FACTOR * gt[:, 2][None, :]
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    ud, ug, loc = set(), set(), []
    for di, gi in order:
        if di in ud or gi in ug or D[di, gi] > tol[0, gi]:
            continue
        ud.add(di); ug.add(gi); loc.append(D[di, gi])
    return len(ud), len(det) - len(ud), len(gt) - len(ug), loc


def load_gt(run, stem):
    g = pd.read_csv(os.path.join(run, stem + "_ground_truth.csv"))
    return g[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)


def score(run, method):
    stems = sorted(os.path.basename(p).replace("_ground_truth.csv", "")
                   for p in glob.glob(os.path.join(run, "*_ground_truth.csv")))
    T = F = N = 0; L = []
    for stem in stems:
        arr = tifffile.imread(os.path.join(run, stem + ".tif")).astype(np.float32)
        det = method(arr)
        gt = load_gt(run, stem)
        tp, fp, fn, loc = match(det, gt)
        T += tp; F += fp; N += fn; L += loc
    P = T / (T + F) if T + F else 0
    R = T / (T + N) if T + N else 0
    Fp = 2 * P * R / (P + R) if P + R else 0
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(Fp, 3),
                loc_err_px=round(np.mean(L), 1) if L else None, TP=T, FP=F, FN=N)


for run_name, run in RUNS.items():
    print(f"\n########## {run_name} ##########")
    rows = []
    r = score(run, log_dets); r["method"] = "LoG only"; rows.append(r)
    r = score(run, lambda a: merge(log_dets(a), watershed_dets(a))); r["method"] = "LoG + watershed"; rows.append(r)
    df = pd.DataFrame(rows)[["method", "precision", "recall", "f1", "loc_err_px", "TP", "FP", "FN"]]
    print(df.to_string(index=False))
