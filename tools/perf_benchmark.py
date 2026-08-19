"""Performance benchmark for standalone cryoblob (Reviewer 2 #5).

Reports, for the current JAX platform (gpu or cpu, from JAX_PLATFORM_NAME):
  cold time (first call, includes JIT compile) vs warm time (median of repeats),
  across image size and scale count, plus peak device memory and full env/version
  provenance. Timing uses block_until_ready for accurate device timing.

Run twice:  JAX_PLATFORM_NAME=gpu ... ;  JAX_PLATFORM_NAME=cpu ...
"""
import os, time, platform, importlib.metadata as im
os.environ.setdefault("JAXTYPING_DISABLE", "1")
import numpy as np
import jax, jax.numpy as jnp
import cryoblob
from cryoblob.types import MRC_Image

PLATFORM = jax.default_backend()

def versions():
    v = {p: im.version(p) for p in ("cryoblob", "jax", "jaxlib", "jaxtyping",
                                    "numpy", "scipy", "scikit-image") if _has(p)}
    return v
def _has(p):
    try: im.version(p); return True
    except Exception: return False

def gpu_name():
    try:
        d = jax.devices()[0]
        return getattr(d, "device_kind", str(d))
    except Exception:
        return "?"

def peak_mem_mb():
    try:
        st = jax.devices()[0].memory_stats()
        return round(st.get("peak_bytes_in_use", 0) / 1e6, 1)
    except Exception:
        return None

def mk(d):
    d = jnp.asarray(d, jnp.float32)
    return MRC_Image(image_data=d, voxel_size=jnp.array([1., 1., 1.]), origin=jnp.zeros(3),
                     data_min=jnp.min(d), data_max=jnp.max(d), data_mean=jnp.mean(d),
                     mode=jnp.asarray(2, jnp.int32))

def detect(mrc, max_sigma, step):
    return cryoblob.blob_list_log(mrc, min_blob_size=8, max_blob_size=max_sigma,
                                  blob_step=step, downscale=4, std_threshold=6)

print(f"platform={PLATFORM}  device={gpu_name()}  cpu={platform.processor() or platform.machine()}")
print("versions:", versions())
print(f"{'size':>6} {'scales':>7} {'cold_s':>8} {'warm_s':>8} {'peakMB':>8}")

REPEAT = 5
for N in (1024, 2048, 4096):
    rng = np.random.default_rng(0)
    arr = rng.random((N, N)).astype(np.float32) * 0.2
    yy, xx = np.ogrid[:N, :N]
    for _ in range(20):
        cy, cx, r = rng.uniform(0, N), rng.uniform(0, N), rng.uniform(N/40, N/15)
        arr[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] -= 0.5
    pre = cryoblob.preprocessing(image_orig=mk(arr).image_data, return_params=False,
                                 exponential=True, logarizer=False, gblur=2.0, background=0, apply_filter=0)
    pmrc = mk(np.asarray(pre))
    for max_sigma, step in ((80, 4), (110, 4)):
        nscales = int((max_sigma - 8) / step)
        t0 = time.time()
        r = detect(pmrc, max_sigma, step); jax.block_until_ready(r)          # cold (compile)
        cold = time.time() - t0
        ts = []
        for _ in range(REPEAT):
            t0 = time.time()
            r = detect(pmrc, max_sigma, step); jax.block_until_ready(r)
            ts.append(time.time() - t0)
        warm = float(np.median(ts))
        print(f"{N:>6} {nscales:>7} {cold:>8.2f} {warm:>8.3f} {str(peak_mem_mb()):>8}", flush=True)
