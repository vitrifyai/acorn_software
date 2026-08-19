"""Paper run: fixed standalone cryoblob (FFT-LoG) on first + second datasets, with
(a) exclude-border cleanup and (b) a LoG+watershed variant that recovers
overlapping particles. Saves detection CSVs, overlays, and a results table.

Methods scored per image vs simulator ground truth:
  - cryoblob LoG            : preprocessing + FFT blob_list_log, border-filtered
  - cryoblob LoG+Watershed  : LoG + distance-transform watershed (keeps all LoG,
                              adds only watershed detections LoG missed)
"""
import os, glob
os.environ["JAXTYPING_DISABLE"] = "1"
import numpy as np, pandas as pd, tifffile
import jax.numpy as jnp
import cryoblob
from cryoblob.types import MRC_Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu, gaussian
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.transform import rescale
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DS = 4
BORDER = 5
TOL = 0.5
CFG = dict(min_blob_size=8, max_blob_size=110, blob_step=6, downscale=DS, std_threshold=6)
RUNS = {
    "first_dataset (non-overlap)": "/nas-158/syn_data/first_dataset/tem_run_20260720_205052_483427/images",
    "second_dataset (overlap)":    "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images",
}

def _mrc(d):
    d = jnp.asarray(d, dtype=jnp.float32)
    return MRC_Image(image_data=d, voxel_size=jnp.array([1.0, 1.0, 1.0]), origin=jnp.zeros(3),
                     data_min=float(jnp.min(d)), data_max=float(jnp.max(d)),
                     data_mean=float(jnp.mean(d)), mode=jnp.asarray(2, dtype=jnp.int32))

def _border_filter(det, h, w):
    if len(det) == 0: return det
    x, y = det[:, 0], det[:, 1]
    keep = (x >= BORDER) & (y >= BORDER) & (x <= w - BORDER) & (y <= h - BORDER)
    return det[keep]

def log_dets(arr):
    pre = cryoblob.preprocessing(image_orig=_mrc(arr).image_data, return_params=False,
                                 exponential=True, logarizer=False, gblur=2.0, background=0, apply_filter=0)
    b = np.asarray(cryoblob.blob_list_log(_mrc(pre), **CFG))
    if b.size == 0: return np.empty((0, 3))
    det = np.column_stack([b[:, 1], b[:, 0], b[:, 2] * DS * np.sqrt(2.0)])  # [X,Y,r]
    return _border_filter(det, *arr.shape)

def watershed_dets(arr, min_r=40.0, max_r=800.0):
    small = rescale(arr.astype(np.float32), 1.0 / DS, anti_aliasing=True)
    inv = small.max() - small
    inv = (inv - inv.min()) / (np.ptp(inv) + 1e-9)
    sm = gaussian(inv, sigma=2)
    try: thr = threshold_otsu(sm)
    except Exception: thr = sm.mean() + 0.5 * sm.std()
    mask = ndi.binary_fill_holes(sm > thr)
    if mask.sum() == 0: return np.empty((0, 3))
    dist = ndi.distance_transform_edt(mask)
    peaks = peak_local_max(dist, min_distance=int(max(3, (min_r / DS))), labels=mask, exclude_border=False)
    if len(peaks) == 0: return np.empty((0, 3))
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (yy, xx) in enumerate(peaks, 1): markers[yy, xx] = i
    labels = watershed(-dist, markers, mask=mask)
    out = []
    for lab in range(1, labels.max() + 1):
        ys, xs = np.where(labels == lab)
        if len(ys) < 4: continue
        r = (len(ys) / np.pi) ** 0.5 * DS
        if not (min_r <= r <= max_r): continue
        out.append([xs.mean() * DS, ys.mean() * DS, r])
    return _border_filter(np.array(out) if out else np.empty((0, 3)), *arr.shape)

def merge(log, wat, tol=0.6):
    if len(log) == 0: return wat
    if len(wat) == 0: return log
    keep = [w for w in wat if np.all(np.linalg.norm(log[:, :2] - w[:2], axis=1) > tol * np.maximum(w[2], log[:, 2]))]
    return np.vstack([log, np.array(keep)]) if keep else log

def load_gt(run, stem):
    g = pd.read_csv(os.path.join(run, stem + "_ground_truth.csv"))
    return g[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)

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

def overlay(arr, gt, det, title, path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr, cmap="gray")
    for x, y, r in gt: ax.add_patch(Circle((x, y), r, fill=False, edgecolor="yellow", lw=1.0))
    for x, y, r in det: ax.add_patch(Circle((x, y), max(r, 10), fill=False, edgecolor="lime", lw=1.2, ls="--"))
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def agg(T, F, N, L):
    P = T/(T+F) if T+F else 0; R = T/(T+N) if T+N else 0; Fp = 2*P*R/(P+R) if P+R else 0
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(Fp, 3),
                loc_err_nm=round(np.mean(L)*0.5, 1) if L else None, TP=T, FP=F, FN=N)

rows = []
for name, run in RUNS.items():
    out = os.path.join(run, "cryoblob_paper_output"); os.makedirs(out, exist_ok=True)
    stems = sorted(os.path.basename(p).replace("_ground_truth.csv", "")
                   for p in glob.glob(os.path.join(run, "*_ground_truth.csv")))
    print(f"\n########## {name} ##########")
    acc = {"LoG": [0, 0, 0, []], "LoG+Watershed": [0, 0, 0, []]}
    for stem in stems:
        arr = tifffile.imread(os.path.join(run, stem + ".tif")).astype(np.float32)
        gt = load_gt(run, stem)
        ld = log_dets(arr)
        wd = watershed_dets(arr)
        md = merge(ld, wd)
        for label, det in [("LoG", ld), ("LoG+Watershed", md)]:
            tp, fp, fn, loc = match(det, gt)
            a = acc[label]; a[0] += tp; a[1] += fp; a[2] += fn; a[3] += loc
            safe = label.replace("+", "_")
            pd.DataFrame(det, columns=["center_x_px", "center_y_px", "radius_px"]).to_csv(
                os.path.join(out, f"{stem}_{safe}.csv"), index=False)
            overlay(arr, gt, det, f"{name} {stem} — cryoblob {label}", os.path.join(out, f"{stem}_{safe}.png"))
        print(f"  {stem}: LoG det={len(ld)}  LoG+WS det={len(md)}  gt={len(gt)}")
    for label, a in acc.items():
        r = agg(*a); r["dataset"] = name; r["method"] = f"cryoblob {label}"; rows.append(r)
        print(f"  {label:14s}: P={r['precision']} R={r['recall']} F1={r['f1']} loc={r['loc_err_nm']}nm "
              f"TP={r['TP']} FP={r['FP']} FN={r['FN']}")
    print(f"  outputs -> {out}")

res = pd.DataFrame(rows)[["dataset", "method", "precision", "recall", "f1", "loc_err_nm", "TP", "FP", "FN"]]
print("\n================ RESULTS TABLE ================")
print(res.to_string(index=False))
res.to_csv("/home/vnw/cryoblob_paper_results_table.csv", index=False)
print("\nSaved table -> /home/vnw/cryoblob_paper_results_table.csv")
