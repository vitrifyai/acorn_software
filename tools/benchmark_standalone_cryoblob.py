"""Benchmark STANDALONE cryoblob (unpatched, as published) vs simulator ground truth.

Uses the real published entry points on real MRC files:
  cryoblob.load_mrc -> cryoblob.process_single_file   (baseline)
  cryoblob.load_mrc -> preprocessing -> blob_list_log  (tuned sigma sweep)
MRC voxel_size is written as 1.0 so all coords come back in full-res pixels and
score directly against the pixel ground truth. Reports precision/recall/F1/loc.
"""
import os, glob
# The published cryoblob has inconsistent jaxtyping annotations (e.g. preprocessing
# declares gblur:int but forwards it to apply_gaussian_blur:float) so its own
# defaults fail runtime type-checking. Disable jaxtyping runtime checks to run the
# actual numerical algorithm as intended.
os.environ["JAXTYPING_DISABLE"] = "1"
import numpy as np, pandas as pd, tifffile
import jax.numpy as jnp
import cryoblob
from cryoblob.types import MRC_Image

TOL_FACTOR = 0.5
RUNS = {
    "first_dataset (non-overlap, 20/img)":
        "/nas-158/syn_data/first_dataset/tem_run_20260720_205052_483427/images",
    "second_dataset (overlap, 30/img)":
        "/nas-158/syn_data/second_dataset/tem_run_20260721_045252_787851/images",
}
def _mrc(d):
    """Construct MRC_Image directly (bypasses cryoblob.make_mrc_image, which has a
    lax.cond validation bug that crashes on 4096x4096 input). voxel_size=1 -> px."""
    d = jnp.asarray(d, dtype=jnp.float32)
    return MRC_Image(image_data=d, voxel_size=jnp.array([1.0, 1.0, 1.0]),
                     origin=jnp.zeros(3), data_min=float(jnp.min(d)),
                     data_max=float(jnp.max(d)), data_mean=float(jnp.mean(d)),
                     mode=jnp.asarray(2, dtype=jnp.int32))

def match(det_xyr, gt, tol_factor=TOL_FACTOR):
    if len(det_xyr) == 0: return 0, 0, len(gt), []
    if len(gt) == 0: return 0, len(det_xyr), 0, []
    D = np.linalg.norm(det_xyr[:, None, :2] - gt[None, :, :2], axis=2)
    tol = tol_factor * gt[:, 2][None, :]
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    ud, ug, loc = set(), set(), []
    for di, gi in order:
        if di in ud or gi in ug or D[di, gi] > tol[0, gi]: continue
        ud.add(di); ug.add(gi); loc.append(D[di, gi])
    tp = len(ud)
    return tp, len(det_xyr) - tp, len(gt) - len(ug), loc

def load_gt(run, stem):
    g = pd.read_csv(os.path.join(run, stem + "_ground_truth.csv"))
    return g[["center_x_px", "center_y_px", "radius_px"]].to_numpy(float)

def blobs_to_xyr(blobs, ds=1):
    """cryoblob returns [Y, X, size]. Y,X are already full-res (blob_list_log
    applies downscale internally); size is a downscaled sigma -> full-res
    radius = size*ds*sqrt(2). Returns [X, Y, radius]. Centers drive scoring."""
    b = np.asarray(blobs)
    if b.size == 0:
        return np.empty((0, 3))
    return np.column_stack([b[:, 1], b[:, 0], b[:, 2] * ds * np.sqrt(2.0)])

# published preprocessing defaults (make_preprocessing_config defaults)
PP = cryoblob.make_preprocessing_config()

def _preprocess(arr):
    mrc = _mrc(arr)
    # published make_preprocessing_config defaults, as plain Python values
    pre = cryoblob.preprocessing(image_orig=mrc.image_data, return_params=False,
                                 exponential=True, logarizer=False,
                                 gblur=2.0, background=0, apply_filter=0)
    return _mrc(pre)

DS = 4  # FFT-LoG removes the memory ceiling, so full downscale-4 resolution.
        # higher downscale keeps the working image small enough to fit.

def detect_baseline(arr):
    # process_single_file's exact detection: default preprocessing + blob_list_log defaults
    blobs = cryoblob.blob_list_log(_preprocess(arr), downscale=DS)
    return blobs_to_xyr(blobs, DS)

def detect_tuned(arr, std_threshold):
    blobs = cryoblob.blob_list_log(_preprocess(arr), min_blob_size=8, max_blob_size=110,
                                   blob_step=6, downscale=DS, std_threshold=std_threshold)
    return blobs_to_xyr(blobs, DS)

def score(run, detector):
    stems = sorted(os.path.basename(p).replace("_ground_truth.csv", "")
                   for p in glob.glob(os.path.join(run, "*_ground_truth.csv")))
    T = F = N = 0; L = []
    for stem in stems:
        arr = tifffile.imread(os.path.join(run, stem + ".tif")).astype(np.float32)
        det = detector(arr)
        gt = load_gt(run, stem)
        tp, fp, fn, loc = match(det, gt)
        T += tp; F += fp; N += fn; L += loc
    P = T / (T + F) if T + F else 0
    R = T / (T + N) if T + N else 0
    Fp = 2 * P * R / (P + R) if P + R else 0
    return dict(precision=round(P, 3), recall=round(R, 3), f1=round(Fp, 3),
                loc_err_px=round(np.mean(L), 1) if L else None, TP=T, FP=F, FN=N)

for run_name, run in RUNS.items():
    print(f"\n########## STANDALONE CRYOBLOB — {run_name} ##########")
    rows = []
    r = score(run, detect_baseline); r["config"] = "baseline (default 5-20, std6)"; rows.append(r)
    for t in (6, 4, 3):
        r = score(run, lambda m, t=t: detect_tuned(m, t)); r["config"] = f"tuned 8-110 std{t}"; rows.append(r)
    df = pd.DataFrame(rows)[["config", "precision", "recall", "f1", "loc_err_px", "TP", "FP", "FN"]]
    print(df.to_string(index=False))
