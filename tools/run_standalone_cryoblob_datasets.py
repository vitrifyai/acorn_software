"""Run the (fixed) standalone cryoblob on the first + second datasets.

For every micrograph: run cryoblob preprocessing + FFT-LoG blob_list_log, save a
detection CSV and an overlay PNG (yellow = ground truth, green = cryoblob), and
score vs ground truth. Writes into <dataset>/images/cryoblob_standalone_output/.
"""
import os, glob
os.environ["JAXTYPING_DISABLE"] = "1"
import numpy as np, pandas as pd, tifffile
import jax.numpy as jnp
import cryoblob
from cryoblob.types import MRC_Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

DS = 4
CFG = dict(min_blob_size=8, max_blob_size=110, blob_step=6, downscale=DS, std_threshold=6)
TOL = 0.5
RUNS = {
    "first_dataset": "/nas-158/syn_data/first_dataset/tem_run_20260720_205052_483427/images",
    "second_dataset": "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images",
}

def mrc(d):
    d = jnp.asarray(d, dtype=jnp.float32)
    return MRC_Image(image_data=d, voxel_size=jnp.array([1.0, 1.0, 1.0]), origin=jnp.zeros(3),
                     data_min=float(jnp.min(d)), data_max=float(jnp.max(d)),
                     data_mean=float(jnp.mean(d)), mode=jnp.asarray(2, dtype=jnp.int32))

def detect(arr):
    pre = cryoblob.preprocessing(image_orig=mrc(arr).image_data, return_params=False,
                                 exponential=True, logarizer=False, gblur=2.0, background=0, apply_filter=0)
    blobs = np.asarray(cryoblob.blob_list_log(mrc(pre), **CFG))  # [Y, X, size]
    if blobs.size == 0:
        return np.empty((0, 3))
    # Y,X already full-res; size is downscaled sigma -> full-res radius
    return np.column_stack([blobs[:, 1], blobs[:, 0], blobs[:, 2] * DS * np.sqrt(2.0)])

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
    for x, y, r in gt:
        ax.add_patch(Circle((x, y), r, fill=False, edgecolor="yellow", lw=1.0))
    for x, y, r in det:
        ax.add_patch(Circle((x, y), max(r, 8), fill=False, edgecolor="lime", lw=1.0, ls="--"))
    ax.set_title(title + "  (yellow=truth, green=cryoblob)"); ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

for name, run in RUNS.items():
    out = os.path.join(run, "cryoblob_standalone_output"); os.makedirs(out, exist_ok=True)
    stems = sorted(os.path.basename(p).replace("_ground_truth.csv", "")
                   for p in glob.glob(os.path.join(run, "*_ground_truth.csv")))
    print(f"\n########## STANDALONE CRYOBLOB — {name} ##########")
    T = F = N = 0; L = []
    for stem in stems:
        arr = tifffile.imread(os.path.join(run, stem + ".tif")).astype(np.float32)
        det = detect(arr)
        gt = load_gt(run, stem)
        tp, fp, fn, loc = match(det, gt)
        T += tp; F += fp; N += fn; L += loc
        pd.DataFrame(det, columns=["center_x_px", "center_y_px", "radius_px"]).to_csv(
            os.path.join(out, f"{stem}_cryoblob.csv"), index=False)
        overlay(arr, gt, det, f"{name} {stem}", os.path.join(out, f"{stem}_overlay.png"))
        print(f"  {stem}: detected={len(det)} gt={len(gt)} TP={tp} FP={fp} FN={fn}")
    P = T/(T+F) if T+F else 0; R = T/(T+N) if T+N else 0; Fp = 2*P*R/(P+R) if P+R else 0
    print(f"  TOTAL: precision={P:.3f} recall={R:.3f} f1={Fp:.3f} loc_err={np.mean(L):.1f}px "
          f"TP={T} FP={F} FN={N}")
    print(f"  outputs -> {out}")
