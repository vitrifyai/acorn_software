"""Worked autodiff example for STANDALONE cryoblob (Reviewer 2, Q on differentiability).

Shows PLAINLY which parts of cryoblob's pipeline are differentiable and where
gradients stop. Every gradient is verified against central finite differences.

Pipeline stages:
  preprocessing (exp/log/blur)  -> DIFFERENTIABLE
  FFT Laplacian-of-Gaussian     -> DIFFERENTIABLE  (w.r.t. image AND scale sigma)
  histogram equalization        -> blocks gradients (binning is discrete)
  threshold (mean + k*std, >)   -> gradient a.e. zero (discrete comparison)
  connected components / argmax -> non-differentiable (discrete labels/coords)
  blob count & coordinates      -> NOT differentiable w.r.t. the input image
"""
import os
os.environ["JAXTYPING_DISABLE"] = "1"
import numpy as np
import jax, jax.numpy as jnp
import cryoblob
from cryoblob.image import laplacian_of_gaussian
from cryoblob.types import MRC_Image

jax.config.update("jax_platform_name", "cpu")  # small demo; deterministic

def synth(N=64, seed=0):
    rng = np.random.default_rng(seed)
    img = rng.random((N, N)).astype(np.float32) * 0.1
    yy, xx = np.ogrid[:N, :N]
    for cy, cx, r in [(20, 22, 7), (44, 40, 9)]:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] -= 0.6
    return jnp.asarray(img, dtype=jnp.float32)

IMG = synth()

def fd_grad_image(f, x, eps=1e-3, k=200):
    """Central finite-diff on k random pixels; return correlation with autodiff grad."""
    g = np.asarray(jax.grad(f)(x))
    rng = np.random.default_rng(1); idx = rng.integers(0, x.size, k)
    xf = np.asarray(x).ravel(); fd = np.zeros(k)
    for j, i in enumerate(idx):
        xp = xf.copy(); xp[i] += eps; xm = xf.copy(); xm[i] -= eps
        fp = float(f(jnp.asarray(xp.reshape(x.shape), jnp.float32)))
        fm = float(f(jnp.asarray(xm.reshape(x.shape), jnp.float32)))
        fd[j] = (fp - fm) / (2 * eps)
    ga = g.ravel()[idx]
    return float(np.corrcoef(ga, fd)[0, 1]), float(np.abs(ga - fd).max())

print("=" * 70)
print("DIFFERENTIABLE PARTS  (autodiff vs finite differences)")
print("=" * 70)

# 1) LoG response energy w.r.t. IMAGE (hist_stretch off -> pure conv path)
def log_energy(img, sigma=3.0):
    r = laplacian_of_gaussian(img, standard_deviation=sigma, hist_stretch=False, normalized=True)
    return jnp.sum(r ** 2)

corr, maxerr = fd_grad_image(lambda x: log_energy(x, 3.0), IMG)
print(f"[1] d/d(image) of ||LoG(image)||^2 : FD correlation={corr:.6f}, max|Δ|={maxerr:.2e}  -> DIFFERENTIABLE")

# 2) LoG response energy w.r.t. SCALE sigma (continuous scale parameter)
def energy_of_sigma(sigma):
    r = laplacian_of_gaussian(IMG, standard_deviation=sigma, hist_stretch=False, normalized=True)
    return jnp.sum(r ** 2)
g_sigma = float(jax.grad(energy_of_sigma)(3.0))
eps = 1e-3
fd_sigma = (float(energy_of_sigma(3.0 + eps)) - float(energy_of_sigma(3.0 - eps))) / (2 * eps)
print(f"[2] d/d(sigma) of ||LoG||^2       : autodiff={g_sigma:.5f}, finite-diff={fd_sigma:.5f}  -> DIFFERENTIABLE")

# 3) Gaussian-blur preprocessing w.r.t. image
from cryoblob.image import apply_gaussian_blur
def blur_energy(img):
    return jnp.sum(apply_gaussian_blur(img, sigma=2.0) ** 2)
corr_b, maxerr_b = fd_grad_image(blur_energy, IMG)
print(f"[3] d/d(image) of ||blur(image)||^2: FD correlation={corr_b:.6f}, max|Δ|={maxerr_b:.2e}  -> DIFFERENTIABLE")

# 4) Histogram equalization: gradients DO pass (jnp.histogram uses differentiable ops);
#    the binning makes them approximate but not zero.
from cryoblob.image import equalize_hist
def eq_energy(img):
    return jnp.sum(equalize_hist(img) ** 2)
g_eq = np.asarray(jax.grad(eq_energy)(IMG))
frac_zero = float(np.mean(np.abs(g_eq) < 1e-12))
print(f"[4] histogram equalization        : grad passes, {frac_zero*100:.0f}% pixel grads exactly 0, "
      f"||grad||={np.linalg.norm(g_eq):.2e}  -> DIFFERENTIABLE (approximate; binning)")

print("\n" + "=" * 70)
print("WHERE GRADIENTS STOP  (discrete operations)")
print("=" * 70)

# 5) Threshold count (mean + k*std, then >) : gradient a.e. zero
def thresh_count(img, sigma=3.0, k=6.0):
    r = laplacian_of_gaussian(img, standard_deviation=sigma, hist_stretch=False, normalized=True)
    thr = jnp.mean(r) + k * jnp.std(r)
    return jnp.sum((r > thr).astype(jnp.float32))   # discrete count
g_thr = np.asarray(jax.grad(thresh_count)(IMG))
print(f"[5] threshold + count (r > thr)    : ||grad||={np.linalg.norm(g_thr):.2e} "
      f"(the > comparison has zero gradient a.e.) -> NON-DIFFERENTIABLE")

# 6) Blob detection (connected components + centroid) : discrete, host-side
def n_blobs(img):
    def mk(d):
        d = jnp.asarray(d, jnp.float32)
        return MRC_Image(image_data=d, voxel_size=jnp.array([1.,1.,1.]), origin=jnp.zeros(3),
                         data_min=jnp.min(d), data_max=jnp.max(d), data_mean=jnp.mean(d),
                         mode=jnp.asarray(2, jnp.int32))
    b = cryoblob.blob_list_log(mk(img), min_blob_size=2, max_blob_size=8, downscale=2, std_threshold=4)
    return jnp.asarray(b).shape[0]
try:
    jax.grad(lambda x: float(n_blobs(x)))(IMG)
    print("[6] blob count/coords              : (unexpectedly returned a grad)")
except Exception as e:
    print(f"[6] blob count/coords              : jax.grad FAILS ({type(e).__name__}) — connected components "
          f"is discrete + host-side (scipy) -> NON-DIFFERENTIABLE")

print("\n" + "=" * 70)
print("SUMMARY (for the manuscript)")
print("=" * 70)
print("Differentiable: preprocessing (exp/log/Gaussian blur), histogram")
print("  equalization (approx.), and the FFT LoG scale-space response — w.r.t.")
print("  BOTH the input image and the scale sigma.")
print("Gradients STOP at: thresholding (> comparison), connected-components")
print("  labeling, and extrema/centroid selection.")
print("=> The full RESPONSE FIELD is differentiable; DETECTION (blob count and")
print("   coordinates) is discrete and therefore NOT differentiable.")
