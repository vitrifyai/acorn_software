"""Baseline comparison for the paper: CryoBlob vs classical reference-free blob
pickers, scored against simulator ground truth on both datasets.

Baselines:
  - scikit-image blob_log : the LoG algorithm family (== RELION LoG autopicker core)
  - scikit-image blob_dog : Difference-of-Gaussians (== DoGpicker, Voss 2009)
CryoBlob LoG / LoG+Watershed numbers are loaded from the saved paper detections so
they match the manuscript tables exactly.
"""
import os, glob
import numpy as np, pandas as pd, tifffile
from skimage.transform import rescale
from skimage.filters import gaussian
from skimage.feature import blob_log, blob_dog

TOL = 0.5
DOWNSAMPLE = 0.25       # heavier downsample: CPU blob_log/blob_dog on 4096 are slow
THRESH = 0.02
RUNS = {
    "first_dataset (separated)": "/nas-158/syn_data/first_dataset/tem_run_20260720_205052_483427/images",
    "second_dataset (overlap)":  "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images",
}

def load_gt(run, stem):
    g = pd.read_csv(os.path.join(run, stem + "_ground_truth.csv"))
    return g[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)

def prep(img):
    im = (img - img.min()) / (np.ptp(img) + 1e-9)
    im = 1.0 - im                                  # particles dark -> bright
    return gaussian(rescale(im, DOWNSAMPLE, anti_aliasing=True), 2)

def skimage_dets(img, gt_r, fn):
    sm = prep(img); s = 1.0 / DOWNSAMPLE
    rmin, rmax = gt_r.min(), gt_r.max()
    smin = max(2.0, (rmin / np.sqrt(2)) * DOWNSAMPLE)
    smax = max(smin + 1, (rmax / np.sqrt(2)) * DOWNSAMPLE)
    b = fn(sm, min_sigma=smin, max_sigma=smax, threshold=THRESH)
    if len(b) == 0: return np.empty((0, 3))
    return np.column_stack([b[:, 1] * s, b[:, 0] * s, b[:, 2] * np.sqrt(2) * s])

def cryoblob_saved(run, stem, variant):
    p = os.path.join(run, "cryoblob_paper_output", f"{stem}_{variant}.csv")
    if not os.path.isfile(p): return np.empty((0, 3))
    d = pd.read_csv(p)
    return d[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)

def match(det, gt):
    if len(det) == 0: return 0, 0, len(gt), []
    D = np.linalg.norm(det[:, None, :2] - gt[None, :, :2], axis=2)
    tol = TOL * gt[:, 2][None, :]
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    ud, ug, loc = set(), set(), []
    for di, gi in order:
        if di in ud or gi in ug or D[di, gi] > tol[0, gi]: continue
        ud.add(di); ug.add(gi); loc.append(D[di, gi])
    return len(ud), len(det) - len(ud), len(gt) - len(ug), loc

METHODS = [
    ("CryoBlob LoG",            lambda run, stem, img, gt: cryoblob_saved(run, stem, "LoG")),
    ("CryoBlob LoG+Watershed",  lambda run, stem, img, gt: cryoblob_saved(run, stem, "LoG_Watershed")),
    ("blob_log (RELION-LoG family)", lambda run, stem, img, gt: skimage_dets(img, gt[:, 2], blob_log)),
    ("blob_dog (DoGpicker)",    lambda run, stem, img, gt: skimage_dets(img, gt[:, 2], blob_dog)),
]

rows = []
for name, run in RUNS.items():
    stems = sorted(os.path.basename(p).replace("_ground_truth.csv", "")
                   for p in glob.glob(os.path.join(run, "*_ground_truth.csv")))
    for mname, fn in METHODS:
        T = F = N = 0; L = []
        for stem in stems:
            img = tifffile.imread(os.path.join(run, stem + ".tif")).astype(float)
            gt = load_gt(run, stem)
            det = fn(run, stem, img, gt)
            tp, fp, fn_, loc = match(det, gt)
            T += tp; F += fp; N += fn_; L += loc
        P = T/(T+F) if T+F else 0; R = T/(T+N) if T+N else 0; Fp = 2*P*R/(P+R) if P+R else 0
        rows.append(dict(dataset=name, method=mname, precision=round(P, 3), recall=round(R, 3),
                         f1=round(Fp, 3), loc_err_nm=round(np.mean(L)*0.5, 1) if L else None,
                         TP=T, FP=F, FN=N))
        print(f"{name:26s} {mname:32s} P={P:.3f} R={R:.3f} F1={Fp:.3f}", flush=True)

res = pd.DataFrame(rows)
print("\n" + res.to_string(index=False))
res.to_csv("/home/vnw/cryoblob_paper/tables/baseline_comparison.csv", index=False)
print("\nSaved -> /home/vnw/cryoblob_paper/tables/baseline_comparison.csv")
