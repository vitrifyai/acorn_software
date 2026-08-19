"""Robustness grid for STANDALONE cryoblob (Reviewer 2 Q6): quantitative
detection metrics across morphology, SNR, size, and overlap, on simulator data
with known ground truth.

Morphologies: round (plga), elongated (bacteria capsules), hollow/amorphous
(lipid vesicle). Ground truth is derived uniformly from the rendered label mask
(connected components -> per-object centroid + equivalent radius). cryoblob is run
with a FIXED config (no per-image tuning) so the grid measures robustness, not
tuning. LoG mode for round/hollow; ridge mode for elongated.

Metrics per condition (3 seeds each): precision, recall, F1, localization error
(nm), mask IoU & Dice.
"""
import os, time
os.environ["JAXTYPING_DISABLE"] = "1"
import numpy as np, pandas as pd
import jax.numpy as jnp
from scipy import ndimage as ndi
import cryoblob
from cryoblob.types import MRC_Image
from acorn_tem_sim import simulator as S

NPX = 2048
PX_A = 5.0           # 0.5 nm/px (matches the validated regime)
PX_NM = PX_A / 10.0
ICE_NM = 300.0       # thick ice -> particle signal >> sqrt(noise); realistic cryo regime
DS = 4
SEEDS = [1, 2, 3]
TOL = 0.5
CFG_TEM = S.resolve(preset="krios-k3"); CFG_TEM["image_size_px"] = NPX; CFG_TEM["pixel_size_a"] = PX_A
# FIXED cryoblob LoG config (no per-condition tuning -> the grid measures robustness)
LOG = dict(min_blob_size=8, max_blob_size=80, blob_step=4, downscale=DS, std_threshold=6)

def _mrc(d):
    d = jnp.asarray(d, dtype=jnp.float32)
    return MRC_Image(image_data=d, voxel_size=jnp.array([1.0, 1.0, 1.0]), origin=jnp.zeros(3),
                     data_min=float(jnp.min(d)), data_max=float(jnp.max(d)),
                     data_mean=float(jnp.mean(d)), mode=jnp.asarray(2, dtype=jnp.int32))

def gt_from_label(label):
    """Per-object ground truth from the rendered mask: centroid + equiv radius (px)."""
    lbl, n = ndi.label(np.asarray(label) > 0)
    gt = []
    for i in range(1, n + 1):
        ys, xs = np.where(lbl == i)
        if len(ys) < 4: continue
        gt.append([xs.mean(), ys.mean(), (len(ys) / np.pi) ** 0.5])
    return np.array(gt) if gt else np.empty((0, 3)), (np.asarray(label) > 0)

def log_detect(arr):
    pre = cryoblob.preprocessing(image_orig=_mrc(arr).image_data, return_params=False,
                                 exponential=True, logarizer=False, gblur=2.0, background=0, apply_filter=0)
    b = np.asarray(cryoblob.blob_list_log(_mrc(pre), **LOG))
    if b.size == 0: return np.empty((0, 3))
    return np.column_stack([b[:, 1], b[:, 0], b[:, 2] * DS * np.sqrt(2.0)])

def ridge_detect(arr):
    """Elongated: use cryoblob's enhanced/ridge path via blob_list_log on the ridge
    response is not exposed; approximate elongated detection by LoG on the capsule
    bodies (centroid recall) — documented limitation for elongated objects."""
    return log_detect(arr)

def det_mask(det, shape):
    m = np.zeros(shape, bool)
    if len(det) == 0: return m
    H, W = shape
    yy, xx = np.ogrid[:H, :W]
    for x, y, r in det:
        r = max(r, 3)
        y0, y1 = max(0, int(y - r - 1)), min(H, int(y + r + 2))
        x0, x1 = max(0, int(x - r - 1)), min(W, int(x + r + 2))
        m[y0:y1, x0:x1] |= (xx[:, x0:x1] - x) ** 2 + (yy[y0:y1] - y) ** 2 <= r * r
    return m

def match(det, gt):
    if len(det) == 0: return 0, 0, len(gt), []
    if len(gt) == 0: return 0, len(det), 0, []
    D = np.linalg.norm(det[:, None, :2] - gt[None, :, :2], axis=2)
    tol = TOL * gt[:, 2][None, :]
    order = np.dstack(np.unravel_index(np.argsort(D, axis=None), D.shape))[0]
    ud, ug, loc = set(), set(), []
    for di, gi in order:
        if di in ud or gi in ug or D[di, gi] > tol[0, gi]: continue
        ud.add(di); ug.add(gi); loc.append(D[di, gi])
    return len(ud), len(det) - len(ud), len(gt) - len(ug), loc

def score_condition(name, kind, dia, noise, overlap, npart, detector):
    T = F = N = 0; L = []; IoU = []; Dice = []
    for sd in SEEDS:
        sp = S.Specimen(kind=kind, n_particles=npart, diameter_nm_mean=dia, diameter_nm_sd=dia * 0.25,
                        solvent_noise=noise, ice_thickness_nm=ICE_NM, allow_overlap=overlap, seed=sd)
        mg = S.simulate_micrograph(CFG_TEM, specimen=sp, seed=sd)
        arr = np.asarray(mg.image, dtype=np.float32)
        gt, gmask = gt_from_label(mg.label)
        det = detector(arr)
        tp, fp, fn, loc = match(det, gt)
        T += tp; F += fp; N += fn; L += loc
        dm = det_mask(det, arr.shape)
        inter = np.logical_and(gmask, dm).sum()
        IoU.append(inter / max(1, np.logical_or(gmask, dm).sum()))
        Dice.append(2 * inter / max(1, gmask.sum() + dm.sum()))
    P = T / (T + F) if T + F else 0; R = T / (T + N) if T + N else 0
    Fp = 2 * P * R / (P + R) if P + R else 0
    return dict(condition=name, morphology=kind, dia_nm=dia, noise=noise, overlap=overlap,
                precision=round(P, 3), recall=round(R, 3), f1=round(Fp, 3),
                loc_err_nm=round(np.mean(L) * PX_NM, 1) if L else None,
                IoU=round(np.mean(IoU), 3), Dice=round(np.mean(Dice), 3), TP=T, FP=F, FN=N)

# --- grid: one-factor-at-a-time around a baseline + morphology set ---
COND = [
    # (name, kind, dia_nm, noise, overlap, npart, detector)
    ("baseline round",       "plga", 90, 6,  False, 30, log_detect),
    ("SNR: low noise",       "plga", 90, 3,  False, 30, log_detect),
    ("SNR: high noise",      "plga", 90, 12, False, 30, log_detect),
    ("SNR: v.high noise",    "plga", 90, 20, False, 30, log_detect),
    ("size: small 50nm",     "plga", 50, 6,  False, 30, log_detect),
    ("size: large 130nm",    "plga", 130, 6, False, 30, log_detect),
    ("overlap: dense",       "plga", 90, 6,  True,  50, log_detect),
    ("overlap: v.dense",     "plga", 90, 6,  True,  80, log_detect),
    ("morphology: hollow",   "vesicle", 90, 6, False, 25, log_detect),
    ("morphology: elongated","bacteria", 60, 6, False, 20, ridge_detect),
]

rows = []
for name, kind, dia, noise, overlap, npart, det in COND:
    t = time.time()
    r = score_condition(name, kind, dia, noise, overlap, npart, det)
    print(f"{name:24s} P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
          f"IoU={r['IoU']:.3f} loc={r['loc_err_nm']}nm  ({time.time()-t:.0f}s)", flush=True)
    rows.append(r)

res = pd.DataFrame(rows)
print("\n================ ROBUSTNESS GRID ================")
print(res.to_string(index=False))
res.to_csv("/home/vnw/cryoblob_robustness_grid.csv", index=False)
print("\nSaved -> /home/vnw/cryoblob_robustness_grid.csv")
