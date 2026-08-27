"""Background worker for CryoBLOB detection."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

DEFAULT_COLUMNS = [
    "File Location",
    "Particle ID",
    "Detection Mode",
    "Detection Type",
    "Center Y (nm)",
    "Center X (nm)",
    "Radius (nm)",
    "Size (nm)",
    "Raw CryoBLOB Size (nm)",
    "Size Measurement",
    "Ridge Start Y (nm)",
    "Ridge Start X (nm)",
    "Ridge End Y (nm)",
    "Ridge End X (nm)",
    "Ridge Length (nm)",
    "Ridge Path (nm)",
    "Pixel Size Y (nm/px)",
    "Pixel Size X (nm/px)",
    "Pixel Size Source",
]
SUPPORTED_EXTS = {".dm4", ".mrc", ".mrcs", ".tif", ".tiff", ".png", ".jpg", ".jpeg"}
_JAX_RESPONSE_STACK_AND_MASK = None
_SOURCE_IMAGE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_PREPROCESS_CACHE: OrderedDict[tuple, object] = OrderedDict()
_DETECTION_IMAGE_CACHE: OrderedDict[tuple, object] = OrderedDict()
_RIDGE_RESPONSE_CACHE: OrderedDict[tuple, tuple] = OrderedDict()
_CACHE_LIMIT = 16


def _configure_jax_memory() -> None:
    """Keep JAX from grabbing most of GPU memory when ACORN starts a run."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    # Keep CryoBLOB's JAX context on a single GPU unless the user explicitly
    # pins a different set. This avoids lighting up all 16 DGX GPUs for one run.
    # This JAX build honors CUDA_VISIBLE_DEVICES; JAX_CUDA_VISIBLE_DEVICES alone
    # was not sufficient on the DGX.
    # Record what the process could see BEFORE narrowing it. The pin is
    # process-global and one-way -- once CUDA initialises, restoring the
    # variable gives nothing back -- so anything sizing a workload later has to
    # be able to ask what the machine actually has. Without this, running
    # CryoBLOB silently cut the batch surface-area analysis from sixteen GPU
    # workers to one.
    from acorn.core.gpu import remember_visible_devices
    remember_visible_devices()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("JAX_CUDA_VISIBLE_DEVICES", os.environ["CUDA_VISIBLE_DEVICES"])
    flags = os.environ.get("XLA_FLAGS", "")
    flag = "--xla_gpu_strict_conv_algorithm_picker=false"
    if flag not in flags:
        os.environ["XLA_FLAGS"] = f"{flags} {flag}".strip()


def _cache_get(cache: OrderedDict, key):
    value = cache.get(key)
    if value is not None:
        cache.move_to_end(key)
    return value


def _cache_set(cache: OrderedDict, key, value):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_LIMIT:
        cache.popitem(last=False)


def _file_signature(file_path: str) -> tuple[str, int, int]:
    stat = Path(file_path).stat()
    return (str(Path(file_path).resolve()), int(stat.st_mtime_ns), int(stat.st_size))


def _ratio_to_float(value) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    numerator = getattr(value, "numerator", None)
    denominator = getattr(value, "denominator", None)
    if numerator is not None and denominator:
        return float(numerator) / float(denominator)
    return float(value)


def _unit_to_nm(unit: str | None) -> float | None:
    if not unit:
        return None
    clean = unit.strip().lower().replace("µ", "u").replace("μ", "u")
    clean = clean.replace(" ", "")
    units = {
        "nm": 1.0,
        "nanometer": 1.0,
        "nanometers": 1.0,
        "nanometre": 1.0,
        "nanometres": 1.0,
        "um": 1000.0,
        "micron": 1000.0,
        "microns": 1000.0,
        "micrometer": 1000.0,
        "micrometers": 1000.0,
        "micrometre": 1000.0,
        "micrometres": 1000.0,
        "angstrom": 0.1,
        "angstroms": 0.1,
        "a": 0.1,
        "å": 0.1,
        "pm": 0.001,
        "m": 1e9,
        "meter": 1e9,
        "metre": 1e9,
    }
    return units.get(clean)


def _tiff_pixel_size_nm(file_path: str) -> tuple[float | None, float | None, str | None]:
    import re
    import xml.etree.ElementTree as ET

    import tifffile

    with tifffile.TiffFile(file_path) as tf:
        ome_metadata = getattr(tf, "ome_metadata", None)
        if ome_metadata:
            try:
                root = ET.fromstring(ome_metadata)
                pixels = next((el for el in root.iter() if el.tag.endswith("Pixels")), None)
                if pixels is not None:
                    sx = pixels.attrib.get("PhysicalSizeX")
                    sy = pixels.attrib.get("PhysicalSizeY")
                    ux = _unit_to_nm(pixels.attrib.get("PhysicalSizeXUnit")) or 1000.0
                    uy = _unit_to_nm(pixels.attrib.get("PhysicalSizeYUnit")) or 1000.0
                    if sx and sy:
                        return float(sy) * uy, float(sx) * ux, "OME-TIFF header"
            except Exception:
                pass

        ij = getattr(tf, "imagej_metadata", None) or {}
        unit_nm = _unit_to_nm(str(ij.get("unit", "")))
        if unit_nm is not None:
            x_spacing = ij.get("pixel_width") or ij.get("PixelWidth")
            y_spacing = ij.get("pixel_height") or ij.get("PixelHeight")
            if x_spacing and y_spacing:
                return float(y_spacing) * unit_nm, float(x_spacing) * unit_nm, "ImageJ TIFF header"

        if tf.pages:
            page = tf.pages[0]
            tags = {t.name: t.value for t in page.tags.values()}
            xres = tags.get("XResolution")
            yres = tags.get("YResolution")
            unit = tags.get("ResolutionUnit", 1)
            try:
                unit_value = int(unit)
            except Exception:
                unit_value = int(getattr(unit, "value", 1) or 1)
            if xres and yres:
                x_per_unit = _ratio_to_float(xres)
                y_per_unit = _ratio_to_float(yres)
                if x_per_unit > 0 and y_per_unit > 0:
                    if unit_value == 2:
                        return 25.4e6 / y_per_unit, 25.4e6 / x_per_unit, "TIFF resolution header"
                    if unit_value == 3:
                        return 1e7 / y_per_unit, 1e7 / x_per_unit, "TIFF resolution header"

            description = str(tags.get("ImageDescription", ""))
            match_unit = re.search(r"unit=([^\n\r]+)", description)
            match_spacing = re.search(r"spacing=([0-9.eE+-]+)", description)
            unit_nm = _unit_to_nm(match_unit.group(1)) if match_unit else None
            if unit_nm is not None and match_spacing:
                spacing = float(match_spacing.group(1)) * unit_nm
                return spacing, spacing, "ImageJ TIFF description"

    return None, None, None


def _coerce_to_grayscale_2d(data):
    import numpy as np

    arr = np.asarray(data)
    while arr.ndim > 2:
        if arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32, copy=False)
            arr = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            break
        if arr.shape[0] in (3, 4) and arr.ndim >= 3 and arr.shape[-1] > 4:
            rgb = np.moveaxis(arr[:3], 0, -1).astype(np.float32, copy=False)
            arr = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
            break
        arr = arr[arr.shape[0] // 2]
    return np.asarray(arr, dtype=np.float32)


def _pil_pixel_size_nm(image) -> tuple[float | None, float | None, str | None]:
    dpi = image.info.get("dpi")
    if dpi and len(dpi) >= 2 and dpi[0] and dpi[1]:
        return 25.4e6 / float(dpi[1]), 25.4e6 / float(dpi[0]), "image DPI header"

    jfif_density = image.info.get("jfif_density")
    jfif_unit = image.info.get("jfif_unit", 0)
    if jfif_density and len(jfif_density) >= 2 and jfif_density[0] and jfif_density[1]:
        if jfif_unit == 1:
            return (
                25.4e6 / float(jfif_density[1]),
                25.4e6 / float(jfif_density[0]),
                "JFIF density header",
            )
        if jfif_unit == 2:
            return (
                1e7 / float(jfif_density[1]),
                1e7 / float(jfif_density[0]),
                "JFIF density header",
            )

    return None, None, None


def _load_source_image(file_path: str, pixel_size_nm: float, *, use_cache: bool,
                       bin_factor: int = 1):
    import mrcfile
    import tifffile
    from PIL import Image

    from acorn.core.dm4_loader import DM4Image

    # The bin factor belongs in the key: the same file at a different binning is
    # a different image, and serving the cached one would ignore the setting.
    cache_key = (_file_signature(file_path), round(float(pixel_size_nm), 6),
                 int(bin_factor))
    if use_cache:
        cached = _cache_get(_SOURCE_IMAGE_CACHE, cache_key)
        if cached is not None:
            return cached

    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".dm4":
        dm4img = DM4Image.from_file(file_path)
        x_nm_per_px = float(dm4img.pixel_size if dm4img.pixel_size > 0 else pixel_size_nm)
        y_nm_per_px = x_nm_per_px
        pixel_size_source = (
            "DM4 header" if getattr(dm4img.meta, "pixel_size_from_header", False) else "manual pixel size"
        )
        im_np = _coerce_to_grayscale_2d(dm4img.raw)
    elif ext in {".mrc", ".mrcs"}:
        with mrcfile.mmap(file_path, mode="r") as data_mrc:
            data = data_mrc.data
            x_nm_per_px = float(data_mrc.voxel_size.x) * 0.1
            y_nm_per_px = float(data_mrc.voxel_size.y) * 0.1
            pixel_size_source = "MRC voxel size header"
            im_np = _coerce_to_grayscale_2d(data)
    elif ext in {".tif", ".tiff"}:
        data = tifffile.imread(file_path)
        y_header, x_header, source = _tiff_pixel_size_nm(file_path)
        x_nm_per_px = float(x_header if x_header is not None else pixel_size_nm)
        y_nm_per_px = float(y_header if y_header is not None else pixel_size_nm)
        pixel_size_source = source or "manual pixel size"
        im_np = _coerce_to_grayscale_2d(data)
    elif ext in {".png", ".jpg", ".jpeg"}:
        image = Image.open(file_path)
        y_header, x_header, source = _pil_pixel_size_nm(image)
        if image.mode not in ("L", "I", "F"):
            image = image.convert("L")
        x_nm_per_px = float(x_header if x_header is not None else pixel_size_nm)
        y_nm_per_px = float(y_header if y_header is not None else pixel_size_nm)
        pixel_size_source = source or "manual pixel size"
        im_np = _coerce_to_grayscale_2d(image)
    else:
        raise ValueError(f"Unsupported CryoBLOB input format: {ext or '<none>'}")

    # Apply analysis binning uniformly, after the format-specific readers, so
    # every input behaves the same. CryoBLOB reads files itself rather than
    # taking the image the window already binned, so without this it would
    # quietly analyse native pixels while the operator had asked for 4x.
    if int(bin_factor) > 1:
        from acorn.core.binning import bin_image
        binned = bin_image(im_np, int(bin_factor))
        im_np = binned.data
        # Scale both axes: the pixel size is per-axis here and binning is square.
        x_nm_per_px *= binned.factor
        y_nm_per_px *= binned.factor
        pixel_size_source = f"{pixel_size_source} (x{binned.factor} binned)"

    payload = (im_np, y_nm_per_px, x_nm_per_px, pixel_size_source)
    if use_cache:
        _cache_set(_SOURCE_IMAGE_CACHE, cache_key, payload)
    return payload


def _get_preprocessed_image(
    file_path: str,
    image_np,
    *,
    use_cache: bool,
    exponential: bool,
    logarizer: bool,
    gblur: int,
    background: int,
    apply_filter: int,
):
    import numpy as np

    cache_key = (
        _file_signature(file_path),
        bool(exponential),
        bool(logarizer),
        int(gblur),
        int(background),
        int(apply_filter),
    )
    if use_cache:
        cached = _cache_get(_PREPROCESS_CACHE, cache_key)
        if cached is not None:
            return cached

    preprocessed = np.asarray(
        _preprocess_image_jax(
            image_np,
            exponential=exponential,
            logarizer=logarizer,
            gblur=gblur,
            background=background,
            apply_filter=apply_filter,
        ),
        dtype=np.float32,
    )
    if use_cache:
        _cache_set(_PREPROCESS_CACHE, cache_key, preprocessed)
    return preprocessed


def _response_stack_and_mask_fn():
    global _JAX_RESPONSE_STACK_AND_MASK
    if _JAX_RESPONSE_STACK_AND_MASK is not None:
        return _JAX_RESPONSE_STACK_AND_MASK

    import jax
    import jax.numpy as jnp

    @jax.jit
    def response_stack_and_mask(img, scale_values, rel_threshold):
        h, w = img.shape
        fy = jnp.fft.fftfreq(h).astype(jnp.float32)[:, None]
        fx = jnp.fft.fftfreq(w).astype(jnp.float32)[None, :]
        freq2 = fx**2 + fy**2
        img_fft = jnp.fft.fft2(img)

        def response_for_sigma(sigma):
            gaussian_transfer = jnp.exp(-2.0 * (jnp.pi**2) * (sigma**2) * freq2)
            neg_laplacian_transfer = 4.0 * (jnp.pi**2) * freq2
            response = jnp.fft.ifft2(
                img_fft * gaussian_transfer * neg_laplacian_transfer
            ).real
            return (sigma**2 * response).astype(jnp.float32)

        stack = jax.vmap(response_for_sigma)(scale_values)
        stack = jnp.moveaxis(stack, 0, -1)
        local_max = jax.lax.reduce_window(
            stack,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(3, 3, 3),
            window_strides=(1, 1, 1),
            padding="SAME",
        )
        threshold = rel_threshold * jnp.max(stack)
        maxima = (stack == local_max) & (stack >= threshold) & (stack > 0)
        return stack, maxima

    _JAX_RESPONSE_STACK_AND_MASK = response_stack_and_mask
    return _JAX_RESPONSE_STACK_AND_MASK


def _detect_blobs_jax(
    image_np,
    *,
    blob_downscale: float,
    min_sigma: float,
    max_sigma: float,
    threshold_rel: float,
    max_detections: int,
):
    """GPU LoG detector implemented with JAX FFTs and static-shape windows."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    image = jnp.asarray(image_np, dtype=jnp.float32)
    downscale = max(float(blob_downscale), 1.0)
    if downscale > 1.0:
        h, w = image.shape
        scaled_shape = (
            max(1, int(round(h / downscale))),
            max(1, int(round(w / downscale))),
        )
        image = jax.image.resize(image, scaled_shape, method="linear", antialias=True)

    # Cryo-EM particles are usually darker than the local background. Inverting
    # here makes particle bodies produce LoG maxima instead of membrane/edge rims.
    image = -image
    image_min = jnp.nanmin(image)
    image_max = jnp.nanmax(image)
    image_range = image_max - image_min
    image = jnp.where(image_range > 0, (image - image_min) / image_range, jnp.zeros_like(image))

    min_sigma = max(0.5, float(min_sigma))
    max_sigma = max(min_sigma, float(max_sigma))
    sigmas = jnp.asarray(np.linspace(min_sigma, max_sigma, 16), dtype=jnp.float32)

    responses, mask = _response_stack_and_mask_fn()(
        image, sigmas, jnp.asarray(max(0.001, float(threshold_rel)), dtype=jnp.float32)
    )

    coords = np.asarray(jnp.argwhere(mask))
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    responses_np = np.asarray(responses)
    sigmas_np = np.asarray(sigmas)
    validation_image = np.asarray(image)
    scores = responses_np[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(scores)[::-1]
    coords = coords[order]
    scores = scores[order]

    def candidate_quality(y: float, x: float, radius: float) -> float:
        """Reject line/ridge hits; keep round dark interiors against a local ring."""
        h, w = validation_image.shape
        radius = max(2.0, float(radius))
        outer = max(radius * 2.2, radius + 4.0)
        y0 = max(0, int(np.floor(y - outer)))
        y1 = min(h, int(np.ceil(y + outer + 1)))
        x0 = max(0, int(np.floor(x - outer)))
        x1 = min(w, int(np.ceil(x + outer + 1)))
        if y1 - y0 < 5 or x1 - x0 < 5:
            return 0.0

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy = yy - y
        dx = xx - x
        dist = np.hypot(dy, dx)
        inner = dist <= radius
        annulus = (dist >= radius * 1.25) & (dist <= radius * 2.1)
        if int(inner.sum()) < 12 or int(annulus.sum()) < 24:
            return 0.0

        crop = validation_image[y0:y1, x0:x1]
        inner_mean = float(crop[inner].mean())
        annulus_mean = float(crop[annulus].mean())
        annulus_std = float(crop[annulus].std()) + 1e-6
        contrast = inner_mean - annulus_mean
        if contrast < max(0.025, 0.55 * annulus_std):
            return 0.0

        dy_full = np.broadcast_to(dy, dist.shape)
        dx_full = np.broadcast_to(dx, dist.shape)
        angles = np.arctan2(dy_full[inner], dx_full[inner])
        values = crop[inner]
        sector_means = []
        for idx in range(8):
            lo = -np.pi + idx * (2.0 * np.pi / 8.0)
            hi = -np.pi + (idx + 1) * (2.0 * np.pi / 8.0)
            mask = (angles >= lo) & (angles < hi)
            if int(mask.sum()) >= 2:
                sector_means.append(float(values[mask].mean()))
        if len(sector_means) < 6:
            return 0.0
        sector_means_np = np.asarray(sector_means, dtype=np.float32)
        coverage = float(np.mean(sector_means_np > annulus_mean + 0.35 * contrast))
        balance = float(sector_means_np.std() / (abs(sector_means_np.mean()) + 1e-6))
        if coverage < 0.62 or balance > 0.55:
            return 0.0

        return contrast * coverage / (1.0 + balance)

    max_detections = max(1, int(max_detections))

    # Candidate cap: coords/scores are already score-sorted (best first). On noisy
    # or large images the raw LoG mask can hold hundreds of thousands of maxima;
    # nearly all are noise that fail candidate_quality(), so `selected` never fills
    # to max_detections and the early break below never fires — pinning one CPU
    # core for many minutes. The strongest blobs are at the top of the sorted list,
    # so evaluating the weak tail is wasted work. Keep a generous multiple of the
    # requested detection count.
    candidate_cap = max(3000, 40 * max_detections)
    if coords.shape[0] > candidate_cap:
        print(f"[CryoBLOB] capping candidates {coords.shape[0]} -> {candidate_cap} "
              f"(raise threshold_rel to reduce noise maxima)", flush=True)
        coords = coords[:candidate_cap]
        scores = scores[:candidate_cap]

    selected: list[tuple[float, float, float, float]] = []
    for (y_idx, x_idx, s_idx), score in zip(coords, scores, strict=False):
        sigma = float(sigmas_np[int(s_idx)])
        y = float(y_idx)
        x = float(x_idx)
        radius = sigma * np.sqrt(2.0)
        quality = candidate_quality(y, x, radius)
        if quality <= 0:
            continue
        score = float(score) * quality
        keep = True
        for prev_y, prev_x, prev_radius, _ in selected:
            distance = np.hypot(y - prev_y, x - prev_x)
            if distance < 1.15 * max(radius, prev_radius):
                keep = False
                break
        if keep:
            selected.append((y, x, radius, float(score)))
            if len(selected) >= max_detections:
                break

    rows = np.zeros((len(selected), 3), dtype=np.float32)
    for idx, (y, x, radius, _) in enumerate(selected):
        rows[idx] = (y, x, radius)
    return rows


def _preprocess_image_jax(
    image_np,
    *,
    exponential: bool,
    logarizer: bool,
    gblur: int,
    background: int,
    apply_filter: int,
):
    import jax.numpy as jnp

    image = jnp.asarray(image_np, dtype=jnp.float32)
    image_min = jnp.nanmin(image)
    image_max = jnp.nanmax(image)
    image_range = image_max - image_min
    image = jnp.where(image_range > 0, (image - image_min) / image_range, jnp.zeros_like(image))

    if exponential:
        image = jnp.exp(image)
    if logarizer:
        image = jnp.log(jnp.maximum(image, jnp.finfo(jnp.float32).eps))

    def gaussian_fft(img, sigma):
        if sigma <= 0:
            return img
        h, w = img.shape
        fy = jnp.fft.fftfreq(h).astype(jnp.float32)[:, None]
        fx = jnp.fft.fftfreq(w).astype(jnp.float32)[None, :]
        freq2 = fx**2 + fy**2
        transfer = jnp.exp(-2.0 * (jnp.pi**2) * (float(sigma) ** 2) * freq2)
        return jnp.fft.ifft2(jnp.fft.fft2(img) * transfer).real.astype(jnp.float32)

    if gblur > 0:
        image = gaussian_fft(image, float(gblur))
    if background > 0:
        image = image - gaussian_fft(image, float(background))

    # CryoBLOB's Wiener helper is not GPU-safe in this environment. The control
    # remains accepted so saved workflows do not break, but it is skipped here.
    del apply_filter
    return image


def _refine_blob_sizes_from_image(
    image_np,
    rows,
    *,
    y_nm_per_px: float,
    x_nm_per_px: float,
    enabled: bool,
    size_scale: float,
):
    """Estimate particle radius from the original image radial edge profile."""
    import numpy as np

    if rows.size == 0:
        return rows, "CryoBLOB LoG scale"

    rows = np.asarray(rows, dtype=np.float32).copy()
    rows[:, 2] *= max(0.01, float(size_scale))
    if not enabled:
        return rows, "CryoBLOB LoG scale"

    image = np.asarray(image_np, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return rows, "CryoBLOB LoG scale"
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        return rows, "CryoBLOB LoG scale"
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)

    px_nm = float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
    h, w = image.shape

    def radial_profile_means(
        dist_bins: np.ndarray,
        values: np.ndarray,
        max_bin: int,
    ) -> np.ndarray | None:
        counts = np.bincount(dist_bins, minlength=max_bin).astype(np.float32)
        valid_bins = counts > 0
        if int(valid_bins.sum()) < 8:
            return None
        weighted = np.bincount(dist_bins, weights=values, minlength=max_bin).astype(np.float32)
        means = np.empty(max_bin, dtype=np.float32)
        means.fill(np.nan)
        means[valid_bins] = weighted[valid_bins] / counts[valid_bins]
        good = np.isfinite(means)
        if int(good.sum()) < 8:
            return None
        means = np.interp(np.arange(max_bin), np.flatnonzero(good), means[good]).astype(np.float32)
        return means

    def refined_radius_nm(y_nm: float, x_nm: float, raw_radius_nm: float) -> float:
        y = y_nm / max(y_nm_per_px, 1e-9)
        x = x_nm / max(x_nm_per_px, 1e-9)
        raw_r = max(3.0, raw_radius_nm / px_nm)
        # Size-aware outward search:
        # - small particles: tighter search, avoids noisy overshoot
        # - large particles: much wider search so full-res edge can dominate
        if raw_r < 20:
            search_r = min(max(raw_r * 2.6, raw_r + 28.0), min(h, w) * 0.28)
        elif raw_r < 60:
            search_r = min(max(raw_r * 3.1, raw_r + 48.0), min(h, w) * 0.36)
        else:
            search_r = min(max(raw_r * 3.8, raw_r + 90.0), min(h, w) * 0.45)
        y0 = max(0, int(np.floor(y - search_r)))
        y1 = min(h, int(np.ceil(y + search_r + 1)))
        x0 = max(0, int(np.floor(x - search_r)))
        x1 = min(w, int(np.ceil(x + search_r + 1)))
        if y1 - y0 < 12 or x1 - x0 < 12:
            return raw_radius_nm

        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(yy - y, xx - x)
        ang = np.arctan2(yy - y, xx - x)
        crop = image[y0:y1, x0:x1]
        max_bin = int(min(search_r, dist.max()))
        if max_bin < 8:
            return raw_radius_nm

        dist_bins = np.floor(dist).astype(np.int32)
        valid_dist = (dist_bins >= 0) & (dist_bins < max_bin)
        if int(valid_dist.sum()) < 64:
            return raw_radius_nm
        flat_bins = dist_bins[valid_dist].ravel()
        flat_vals = crop[valid_dist].astype(np.float32, copy=False).ravel()
        means = radial_profile_means(flat_bins, flat_vals, max_bin)
        if means is None:
            return raw_radius_nm
        kernel = np.ones(5, dtype=np.float32) / 5.0
        smooth = np.convolve(means, kernel, mode="same")

        def crossing_from_profile(
            profile: np.ndarray,
            *,
            large_particle: bool,
        ) -> tuple[float | None, float]:
            inner_hi = max(4, int(raw_r * (0.50 if large_particle else 0.55)))
            outer_lo = min(max_bin - 2, max(inner_hi + 2, int(raw_r * (1.45 if large_particle else 1.25))))
            outer_hi = min(max_bin, max(outer_lo + (10 if large_particle else 6), int(raw_r * (3.1 if large_particle else 2.4))))
            if outer_hi - outer_lo < 4 or inner_hi < 3:
                return None, 0.0
            inner_level = float(np.median(profile[1:inner_hi]))
            outer_level = float(np.median(profile[outer_lo:outer_hi]))
            contrast = outer_level - inner_level
            if contrast <= (0.010 if large_particle else 0.015):
                return None, contrast
            threshold = inner_level + (0.40 if large_particle else 0.52) * contrast
            start = max(3, int(raw_r * 0.45))
            stop = min(max_bin - 1, int(max(raw_r * (4.0 if large_particle else 3.0), raw_r + (70.0 if large_particle else 35.0))))
            if stop <= start + 2:
                return None, contrast
            for idx in range(start, stop):
                if profile[idx] >= threshold:
                    return float(idx), contrast
            gradients = np.gradient(profile)
            return float(start + int(np.argmax(gradients[start:stop]))), contrast

        # Global fallback profile.
        large_particle = raw_r >= 60
        global_cross, global_contrast = crossing_from_profile(smooth, large_particle=large_particle)

        # Most particles do not need expensive overlap-aware refinement. Use a
        # cheap anisotropy check to decide whether sector scans are worth it.
        decision_sector_count = 8
        decision_sector_ids = np.floor(
            ((ang[valid_dist] + np.pi) / (2.0 * np.pi)) * decision_sector_count
        ).astype(np.int32)
        decision_sector_ids = np.clip(decision_sector_ids, 0, decision_sector_count - 1)
        ring_lo = max(2, int(raw_r * 0.75))
        ring_hi = min(max_bin - 1, max(ring_lo + 3, int(raw_r * 1.35)))
        ring_mask = (flat_bins >= ring_lo) & (flat_bins <= ring_hi)
        ring_values = flat_vals[ring_mask]
        ring_sector_ids = decision_sector_ids[ring_mask]
        anisotropy = 0.0
        if ring_values.size >= 24:
            ring_counts = np.bincount(ring_sector_ids, minlength=decision_sector_count).astype(np.float32)
            ring_weighted = np.bincount(
                ring_sector_ids,
                weights=ring_values,
                minlength=decision_sector_count,
            ).astype(np.float32)
            ring_good = ring_counts >= 3
            if int(ring_good.sum()) >= max(4, decision_sector_count // 2):
                ring_means = ring_weighted[ring_good] / ring_counts[ring_good]
                anisotropy = float(ring_means.std() / (abs(ring_means.mean()) + 1e-6))

        need_sector_refine = (
            global_cross is None
            or large_particle
            or (raw_r >= 28.0 and anisotropy > 0.085)
            or global_contrast < (0.020 if raw_r < 30.0 else 0.014)
        )

        sector_crossings = []
        if need_sector_refine:
            sector_count = 10 if raw_r < 45 else 12
            sector_ids = np.floor(
                ((ang[valid_dist] + np.pi) / (2.0 * np.pi)) * sector_count
            ).astype(np.int32)
            sector_ids = np.clip(sector_ids, 0, sector_count - 1)
            for sector_idx in range(sector_count):
                sector_mask = sector_ids == sector_idx
                if int(sector_mask.sum()) < max(32, max_bin // 3):
                    continue
                sector_means = radial_profile_means(
                    flat_bins[sector_mask],
                    flat_vals[sector_mask],
                    max_bin,
                )
                if sector_means is None:
                    continue
                sector_smooth = np.convolve(sector_means, kernel, mode="same")
                crossing, _ = crossing_from_profile(
                    sector_smooth,
                    large_particle=large_particle,
                )
                if crossing is not None and np.isfinite(crossing):
                    sector_crossings.append(float(crossing))

        if sector_crossings:
            sector_crossings_np = np.asarray(sector_crossings, dtype=np.float32)
            # Use a robust lower-middle percentile so overlaps do not inflate
            # the estimate, but we still cover the particle body well.
            if large_particle:
                if anisotropy < 0.075 and global_cross is not None and np.isfinite(global_cross):
                    refined_px = float(
                        max(
                            float(global_cross),
                            float(np.percentile(sector_crossings_np, 50.0)),
                        )
                    )
                else:
                    refined_px = float(np.percentile(sector_crossings_np, 40.0))
            elif raw_r >= 25:
                refined_px = float(np.percentile(sector_crossings_np, 45.0))
            else:
                refined_px = float(np.median(sector_crossings_np))
        elif global_cross is not None and np.isfinite(global_cross):
            refined_px = float(global_cross)
        else:
            return raw_radius_nm

        if large_particle:
            refined_px = float(np.clip(refined_px, raw_r * 0.70, raw_r * 3.8))
        elif raw_r >= 25:
            refined_px = float(np.clip(refined_px, raw_r * 0.70, raw_r * 2.8))
        else:
            refined_px = float(np.clip(refined_px, raw_r * 0.75, raw_r * 2.2))
        return refined_px * px_nm * max(0.01, float(size_scale))

    for idx in range(len(rows)):
        rows[idx, 2] = refined_radius_nm(
            float(rows[idx, 0]), float(rows[idx, 1]), float(rows[idx, 2])
        )
    return rows, "size-aware radial edge refinement"


def _normalize_detection_image(image_np):
    import numpy as np

    image = np.asarray(image_np, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return 1.0 - image


def _normalize_ridge_image(image_np):
    import numpy as np

    image = np.asarray(image_np, dtype=np.float32)
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def _downscale_image_np(image_np, downscale: float):
    from scipy import ndimage as ndi

    factor = max(float(downscale), 1.0)
    if factor <= 1.0:
        return image_np.astype("float32", copy=False)
    zoom = 1.0 / factor
    return ndi.zoom(image_np, zoom=zoom, order=1).astype("float32", copy=False)


def _find_hole_interior_mask(image_np, downscale: float):
    """
    Auto-detect the bright circular aperture/hole in a cryo-EM micrograph and
    return a downscaled boolean mask of its interior, eroded to exclude the
    boundary ring.  Returns None when no clear large hole is detected.
    """
    import numpy as np
    from scipy import ndimage as ndi

    small = _downscale_image_np(image_np.astype(np.float32), downscale)
    h, w = small.shape
    smoothed = ndi.gaussian_filter(small, sigma=max(1.0, float(min(h, w)) * 0.005))
    bright = smoothed > float(np.median(smoothed))
    filled = ndi.binary_fill_holes(bright)
    labeled, n = ndi.label(filled)
    if n == 0:
        return None
    sizes = ndi.sum(filled, labeled, range(1, n + 1))
    largest_id = int(np.argmax(sizes)) + 1
    hole = labeled == largest_id
    if float(hole.sum()) < 0.15 * h * w:
        return None
    # Erode inward to exclude the bright-dark boundary ring (the aperture edge)
    margin = max(3, int(round(float(min(h, w)) * 0.025)))
    interior = ndi.binary_erosion(hole, iterations=margin)
    return interior if interior.any() else None


def _get_detection_image(
    file_path: str,
    image_np,
    *,
    downscale: float,
    use_cache: bool,
):
    cache_key = (_file_signature(file_path), round(float(downscale), 4))
    if use_cache:
        cached = _cache_get(_DETECTION_IMAGE_CACHE, cache_key)
        if cached is not None:
            return cached
    detection_image = _downscale_image_np(_normalize_detection_image(image_np), downscale)
    if use_cache:
        _cache_set(_DETECTION_IMAGE_CACHE, cache_key, detection_image)
    return detection_image


def _get_ridge_response(
    file_path: str,
    image_np,
    *,
    downscale: float,
    min_sigma: float,
    max_sigma: float,
    ridge_scales: int,
    use_cache: bool,
):
    import numpy as np
    from skimage.filters import meijering, sato

    cache_key = (
        _file_signature(file_path),
        round(float(downscale), 4),
        round(float(min_sigma), 4),
        round(float(max_sigma), 4),
        int(ridge_scales),
    )
    if use_cache:
        cached = _cache_get(_RIDGE_RESPONSE_CACHE, cache_key)
        if cached is not None:
            return cached

    image = _downscale_image_np(_normalize_ridge_image(image_np), downscale)
    min_scale = max(0.5, float(min_sigma) / 2.0)
    max_scale = max(min_scale, float(max_sigma) * 2.0)
    sigmas = np.linspace(min_scale, max_scale, max(2, int(ridge_scales)), dtype=np.float32)
    inv_image = 1.0 - image
    response = np.maximum.reduce(
        [
            np.asarray(sato(image, sigmas=tuple(float(s) for s in sigmas), black_ridges=False), dtype=np.float32),
            np.asarray(sato(image, sigmas=tuple(float(s) for s in sigmas), black_ridges=True), dtype=np.float32),
            np.asarray(sato(inv_image, sigmas=tuple(float(s) for s in sigmas), black_ridges=False), dtype=np.float32),
            np.asarray(sato(inv_image, sigmas=tuple(float(s) for s in sigmas), black_ridges=True), dtype=np.float32),
            np.asarray(meijering(image, sigmas=tuple(float(s) for s in sigmas), black_ridges=False), dtype=np.float32),
            np.asarray(meijering(inv_image, sigmas=tuple(float(s) for s in sigmas), black_ridges=False), dtype=np.float32),
        ]
    )
    payload = (image, response)
    if use_cache:
        _cache_set(_RIDGE_RESPONSE_CACHE, cache_key, payload)
    return payload


def _merge_nearby_rows(rows, *, distance_factor: float):
    import numpy as np

    rows = np.asarray(rows, dtype=np.float32).reshape(-1, 3)
    if len(rows) <= 1:
        return rows

    order = np.argsort(rows[:, 2])[::-1]
    taken = np.zeros(len(rows), dtype=bool)
    merged = []
    for order_idx in order:
        if taken[order_idx]:
            continue
        base = rows[order_idx]
        cluster = [int(order_idx)]
        taken[order_idx] = True
        by, bx, br = map(float, base)
        for other_idx in order:
            if taken[other_idx]:
                continue
            oy, ox, oradius = map(float, rows[other_idx])
            dist = float(np.hypot(oy - by, ox - bx))
            if dist <= max(2.0, distance_factor * max(br, oradius)):
                taken[other_idx] = True
                cluster.append(int(other_idx))
        cluster_rows = rows[cluster]
        weights = np.maximum(cluster_rows[:, 2], 1.0)
        merged.append(
            [
                float(np.average(cluster_rows[:, 0], weights=weights)),
                float(np.average(cluster_rows[:, 1], weights=weights)),
                float(np.max(cluster_rows[:, 2])),
            ]
        )
    return np.asarray(merged, dtype=np.float32).reshape(-1, 3)


def _detect_hessian_rows_np(
    file_path: str,
    image_np,
    *,
    downscale: float,
    min_sigma: float,
    max_sigma: float,
    blob_step: float,
    std_threshold: float,
    max_detections: int,
    y_nm_per_px: float,
    x_nm_per_px: float,
    use_cache: bool,
):
    import numpy as np
    from scipy import ndimage as ndi

    image = _get_detection_image(file_path, image_np, downscale=downscale, use_cache=use_cache)
    scales = np.arange(max(0.5, float(min_sigma)), max(float(max_sigma), float(min_sigma)) + 1e-6, max(0.1, float(blob_step)), dtype=np.float32)
    if scales.size == 0:
        scales = np.asarray([max(0.5, float(min_sigma))], dtype=np.float32)

    responses = []
    for sigma in scales:
        hxx = ndi.gaussian_filter(image, sigma=float(sigma), order=(2, 0))
        hxy = ndi.gaussian_filter(image, sigma=float(sigma), order=(1, 1))
        hyy = ndi.gaussian_filter(image, sigma=float(sigma), order=(0, 2))
        det = (float(sigma) ** 4) * (hxx * hyy - hxy * hxy)
        responses.append(det.astype(np.float32, copy=False))
    stack = np.stack(responses, axis=-1)
    local_max = ndi.maximum_filter(stack, size=(3, 3, 3), mode="nearest")
    threshold = float(stack.mean() + max(0.001, float(std_threshold)) * stack.std())
    maxima = (stack == local_max) & (stack > threshold)
    labels, num = ndi.label(maxima)
    if num <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    centers = ndi.center_of_mass(stack, labels, range(1, num + 1))
    rows = []
    size_scale = float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
    for y, x, s in centers:
        if not np.isfinite(y) or not np.isfinite(x) or not np.isfinite(s):
            continue
        sigma = float(scales[int(np.clip(round(s), 0, len(scales) - 1))])
        rows.append(
            [
                float(y) * float(downscale) * y_nm_per_px,
                float(x) * float(downscale) * x_nm_per_px,
                sigma * float(downscale) * size_scale,
            ]
        )
        if len(rows) >= int(max_detections):
            break
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def _detect_ridge_rows_np(
    file_path: str,
    image_np,
    *,
    downscale: float,
    min_sigma: float,
    max_sigma: float,
    ridge_threshold: float,
    ridge_scales: int,
    max_detections: int,
    y_nm_per_px: float,
    x_nm_per_px: float,
    use_cache: bool,
    run_mode: str,
):
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.morphology import skeletonize
    from skimage.transform import probabilistic_hough_line

    def ridge_binary_to_labels(binary_mask):
        connected = ndi.binary_closing(binary_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        connected = ndi.binary_dilation(connected, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), iterations=1)
        return ndi.label(connected)

    image, response = _get_ridge_response(
        file_path,
        image_np,
        downscale=downscale,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        ridge_scales=ridge_scales,
        use_cache=use_cache,
    )
    final_mode = str(run_mode).lower() == "final"
    percentile_scale = 0.25 if final_mode else 0.16
    requested_threshold = max(0.0, float(ridge_threshold))
    dynamic_threshold = float(np.percentile(response, 99.0) * percentile_scale)
    threshold = max(requested_threshold, dynamic_threshold)
    binary = response > threshold
    labels, num = ridge_binary_to_labels(binary)
    if num <= 0:
        # If a first pass yields nothing, relax the cutoff in a few controlled
        # steps so faint filament images do not collapse to zero detections.
        relax_candidates = [
            max(dynamic_threshold * 0.75, requested_threshold * 0.75),
            max(dynamic_threshold * 0.55, requested_threshold * 0.55),
            max(dynamic_threshold * 0.35, requested_threshold * 0.35),
        ]
        for relaxed in relax_candidates:
            if relaxed <= 0:
                continue
            binary = response > float(relaxed)
            labels, num = ridge_binary_to_labels(binary)
            if num > 0:
                threshold = float(relaxed)
                break
    if num <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    rows = []
    segment_meta: list[dict[str, object]] = []
    size_scale = float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
    ridge_size = ((float(min_sigma) + float(max_sigma)) / 2.0) * float(downscale) * size_scale
    component_slices = ndi.find_objects(labels)
    h, w = image.shape
    hole_mask = _find_hole_interior_mask(image_np, downscale)
    border_margin = max(2, int(round(min(h, w) * 0.015)))
    max_major_span = max(12, int(round(min(h, w) * (0.28 if final_mode else 0.22))))
    max_minor_span = max(5, int(round(float(max_sigma) * (5.5 if final_mode else 4.5))))
    min_aspect = 1.25 if final_mode else 1.35
    max_fill_ratio = 0.62 if final_mode else 0.55
    min_linearity = 2.2 if final_mode else 3.0
    max_thickness = 4.6 if final_mode else 4.0
    prominence_sigma_scale = 0.45 if final_mode else 0.60
    min_prominence_floor = 0.004 if final_mode else 0.006

    def trace_skeleton_path(component_mask: np.ndarray) -> list[tuple[float, float]]:
        skeleton = skeletonize(component_mask)
        coords = np.column_stack(np.nonzero(skeleton)).astype(np.int32)
        if len(coords) < 2:
            coords = np.column_stack(np.nonzero(component_mask)).astype(np.int32)
            if len(coords) < 2:
                return []
        coord_set = {tuple(c.tolist()) for c in coords}
        neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cy, cx in coord_set:
            local = []
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    other = (cy + dy, cx + dx)
                    if other in coord_set:
                        local.append(other)
            neighbors[(cy, cx)] = local

        endpoints = [node for node, neigh in neighbors.items() if len(neigh) == 1]
        start = endpoints[0] if endpoints else tuple(coords[0].tolist())

        def farthest(src: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
            queue = [src]
            parents: dict[tuple[int, int], tuple[int, int] | None] = {src: None}
            dist = {src: 0.0}
            best = src
            best_dist = 0.0
            head = 0
            while head < len(queue):
                node = queue[head]
                head += 1
                for nxt in neighbors.get(node, []):
                    step = float(np.hypot(nxt[0] - node[0], nxt[1] - node[1]))
                    cand = dist[node] + step
                    if nxt not in dist or cand > dist[nxt]:
                        dist[nxt] = cand
                        parents[nxt] = node
                        queue.append(nxt)
                        if cand > best_dist:
                            best_dist = cand
                            best = nxt
            return best, parents

        end_a, _ = farthest(start)
        end_b, parents = farthest(end_a)
        path = [end_b]
        cur = end_b
        while parents.get(cur) is not None:
            cur = parents[cur]
            path.append(cur)
        path.reverse()
        if len(path) > 64:
            step = max(1, len(path) // 64)
            reduced = path[::step]
            if reduced[-1] != path[-1]:
                reduced.append(path[-1])
            path = reduced
        return [(float(py), float(px)) for py, px in path]

    preview_component_limit = max(160, int(max_detections) * 5)
    if not final_mode and num > preview_component_limit:
        component_scores = ndi.sum(response, labels, index=range(1, num + 1))
        component_scores = np.asarray(component_scores, dtype=np.float32)
        if component_scores.size:
            keep_ids = np.argsort(component_scores)[::-1][:preview_component_limit] + 1
            keep_mask = np.isin(labels, keep_ids)
            labels, num = ndi.label(keep_mask)
            component_slices = ndi.find_objects(labels)
        else:
            component_slices = []
    else:
        component_slices = ndi.find_objects(labels)

    def simple_component_meta(label_id: int, slc) -> dict[str, object] | None:
        import numpy as np

        if slc is None:
            return None
        ys, xs = slc
        y0, y1 = int(ys.start), int(ys.stop)
        x0, x1 = int(xs.start), int(xs.stop)
        if (
            y0 <= border_margin
            or x0 <= border_margin
            or y1 >= h - border_margin
            or x1 >= w - border_margin
        ):
            return None
        component = labels[slc] == label_id
        area = int(component.sum())
        if area < 4:
            return None
        coords = np.column_stack(np.nonzero(component)).astype(np.float32)
        if len(coords) < 2:
            return None
        y_span = max(1, y1 - y0)
        x_span = max(1, x1 - x0)
        major = max(y_span, x_span)
        minor = min(y_span, x_span)
        if major < 3 or major > max(18, int(round(min(h, w) * 0.32))):
            return None
        if minor > max(8, int(round(float(max_sigma) * 7.0))):
            return None
        aspect = float(major / max(1, minor))
        if aspect < 1.05:
            return None
        coords_mean = coords.mean(axis=0)
        centered = coords - coords_mean
        cov = np.cov(centered, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-6))
        principal_dir = eigvecs[:, -1].astype(np.float32)
        projections = centered @ principal_dir
        start_idx = int(np.argmin(projections))
        end_idx = int(np.argmax(projections))
        start_local = coords[start_idx] + np.array([y0, x0], dtype=np.float32)
        end_local = coords[end_idx] + np.array([y0, x0], dtype=np.float32)
        start_y = float(start_local[0]) * float(downscale) * y_nm_per_px
        start_x = float(start_local[1]) * float(downscale) * x_nm_per_px
        end_y = float(end_local[0]) * float(downscale) * y_nm_per_px
        end_x = float(end_local[1]) * float(downscale) * x_nm_per_px
        center_y = float((coords_mean[0] + y0) * float(downscale) * y_nm_per_px)
        center_x = float((coords_mean[1] + x0) * float(downscale) * x_nm_per_px)
        return {
            "row": [center_y, center_x, ridge_size],
            "meta": {
                "ridge_start_y_nm": start_y,
                "ridge_start_x_nm": start_x,
                "ridge_end_y_nm": end_y,
                "ridge_end_x_nm": end_x,
                "ridge_length_nm": float(np.hypot(end_y - start_y, end_x - start_x)),
                "ridge_path_nm": [[start_y, start_x], [end_y, end_x]],
            },
        }

    def hough_line_candidates() -> list[dict[str, object]]:
        response_finite = response[np.isfinite(response)]
        if response_finite.size == 0:
            return []
        hough_threshold = float(np.percentile(response_finite, 88.0 if final_mode else 91.0))
        if hough_threshold <= 0:
            return []
        candidate_mask = response >= hough_threshold
        candidate_mask = ndi.binary_opening(candidate_mask, structure=np.ones((2, 2), dtype=bool), iterations=1)
        candidate_mask = ndi.binary_closing(candidate_mask, structure=np.ones((3, 3), dtype=bool), iterations=1)
        min_length = max(5, int(round(float(min_sigma + max_sigma) * 0.9)))
        max_length = max(min_length + 1, int(round(min(h, w) * 0.22)))
        lines = probabilistic_hough_line(
            candidate_mask,
            threshold=max(4, min_length // 2),
            line_length=min_length,
            line_gap=max(2, int(round(float(max_sigma) * 1.5))),
        )
        candidates = []
        seen_centers: list[tuple[float, float]] = []
        for (x0, y0), (x1, y1) in lines:
            if (
                y0 <= border_margin
                or y1 <= border_margin
                or x0 <= border_margin
                or x1 <= border_margin
                or y0 >= h - border_margin
                or y1 >= h - border_margin
                or x0 >= w - border_margin
                or x1 >= w - border_margin
            ):
                continue
            length_px = float(np.hypot(y1 - y0, x1 - x0))
            if length_px < min_length or length_px > max_length:
                continue
            center_y_px = 0.5 * float(y0 + y1)
            center_x_px = 0.5 * float(x0 + x1)
            if any(np.hypot(center_y_px - py, center_x_px - px) < max(4.0, length_px * 0.35) for py, px in seen_centers):
                continue
            seen_centers.append((center_y_px, center_x_px))
            start_y = float(y0) * float(downscale) * y_nm_per_px
            start_x = float(x0) * float(downscale) * x_nm_per_px
            end_y = float(y1) * float(downscale) * y_nm_per_px
            end_x = float(x1) * float(downscale) * x_nm_per_px
            center_y = center_y_px * float(downscale) * y_nm_per_px
            center_x = center_x_px * float(downscale) * x_nm_per_px
            candidates.append(
                {
                    "row": [center_y, center_x, ridge_size],
                    "meta": {
                        "ridge_start_y_nm": start_y,
                        "ridge_start_x_nm": start_x,
                        "ridge_end_y_nm": end_y,
                        "ridge_end_x_nm": end_x,
                        "ridge_length_nm": float(np.hypot(end_y - start_y, end_x - start_x)),
                        "ridge_path_nm": [[start_y, start_x], [end_y, end_x]],
                    },
                }
            )
            if len(candidates) >= int(max_detections):
                break
        return candidates

    hough_seeded = False
    if final_mode:
        for fallback in hough_line_candidates():
            rows.append(fallback["row"])
            segment_meta.append(fallback["meta"])
            hough_seeded = True
            if len(rows) >= int(max_detections):
                break

    for label_id, slc in enumerate(component_slices, start=1):
        if len(rows) >= int(max_detections):
            break
        if slc is None:
            continue
        ys, xs = slc
        y0, y1 = int(ys.start), int(ys.stop)
        x0, x1 = int(xs.start), int(xs.stop)
        if (
            y0 <= border_margin
            or x0 <= border_margin
            or y1 >= h - border_margin
            or x1 >= w - border_margin
        ):
            continue

        component = labels[slc] == label_id
        area = int(component.sum())
        if area < 6:
            continue
        y_span = max(1, y1 - y0)
        x_span = max(1, x1 - x0)
        major = max(y_span, x_span)
        minor = min(y_span, x_span)
        if major > max_major_span or minor > max_minor_span:
            continue
        aspect = float(major / max(1, minor))
        fill_ratio = float(area / max(1, y_span * x_span))
        if aspect < min_aspect or fill_ratio > max_fill_ratio:
            continue

        coords = np.column_stack(np.nonzero(component)).astype(np.float32)
        coords_mean = coords.mean(axis=0)
        centered = coords - coords_mean
        if len(coords) < 4:
            continue
        cov = np.cov(centered, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.sort(np.maximum(eigvals, 1e-6))
        linearity = float(eigvals[-1] / eigvals[0])
        if linearity < min_linearity:
            continue
        thickness = float(area / max(major, 1))
        if thickness > max_thickness:
            continue

        local_response = response[slc]
        weights = np.where(component, np.maximum(local_response, 0.0), 0.0)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        yy, xx = np.indices(component.shape, dtype=np.float32)
        y = float((weights * (yy + y0)).sum() / weight_sum)
        x = float((weights * (xx + x0)).sum() / weight_sum)

        if hole_mask is not None:
            iy = int(np.clip(round(y), 0, hole_mask.shape[0] - 1))
            ix = int(np.clip(round(x), 0, hole_mask.shape[1] - 1))
            if not hole_mask[iy, ix]:
                continue

        support_r = max(3.0, 0.5 * float(major))
        outer_r = max(support_r + 2.0, support_r * 2.2)
        cy0 = max(0, int(np.floor(y - outer_r - 1)))
        cy1 = min(h, int(np.ceil(y + outer_r + 2)))
        cx0 = max(0, int(np.floor(x - outer_r - 1)))
        cx1 = min(w, int(np.ceil(x + outer_r + 2)))
        ryy, rxx = np.ogrid[cy0:cy1, cx0:cx1]
        rdist = np.hypot(ryy - y, rxx - x)
        inner_mask = rdist <= support_r
        annulus_mask = (rdist >= support_r * 1.3) & (rdist <= outer_r)
        if int(annulus_mask.sum()) < 12:
            continue
        local_patch = response[cy0:cy1, cx0:cx1]
        inner_mean = float(local_patch[inner_mask].mean()) if int(inner_mask.sum()) else 0.0
        annulus_mean = float(local_patch[annulus_mask].mean())
        annulus_std = float(local_patch[annulus_mask].std()) + 1e-6
        prominence = inner_mean - annulus_mean
        if prominence < max(min_prominence_floor, prominence_sigma_scale * annulus_std):
            continue

        principal_dir = np.linalg.eigh(cov)[1][:, -1].astype(np.float32)
        projections = centered @ principal_dir
        start_idx = int(np.argmin(projections))
        end_idx = int(np.argmax(projections))
        start_local = coords[start_idx] + np.array([y0, x0], dtype=np.float32)
        end_local = coords[end_idx] + np.array([y0, x0], dtype=np.float32)
        start_y = float(start_local[0]) * float(downscale) * y_nm_per_px
        start_x = float(start_local[1]) * float(downscale) * x_nm_per_px
        end_y = float(end_local[0]) * float(downscale) * y_nm_per_px
        end_x = float(end_local[1]) * float(downscale) * x_nm_per_px
        ridge_length_nm = float(
            np.hypot(end_y - start_y, end_x - start_x)
        )
        trace_mask = component
        if final_mode:
            pad = max(2, int(round(max(major, minor) * 0.25)))
            ty0 = max(0, y0 - pad)
            ty1 = min(h, y1 + pad)
            tx0 = max(0, x0 - pad)
            tx1 = min(w, x1 + pad)
            local_crop = response[ty0:ty1, tx0:tx1]
            local_thresh = max(
                float(np.percentile(local_crop, 97.5) * 0.45),
                float(local_response[component].mean() * 0.65),
            )
            refined = local_crop >= local_thresh
            local_labels, local_num = ndi.label(refined)
            if local_num > 0:
                ly = int(np.clip(round(y - ty0), 0, refined.shape[0] - 1))
                lx = int(np.clip(round(x - tx0), 0, refined.shape[1] - 1))
                label_here = int(local_labels[ly, lx])
                if label_here > 0:
                    trace_mask = local_labels == label_here
                    y0, y1, x0, x1 = ty0, ty1, tx0, tx1

        if final_mode:
            polyline_local = trace_skeleton_path(trace_mask)
            ridge_path_nm = [
                [
                    (py + y0) * float(downscale) * y_nm_per_px,
                    (px + x0) * float(downscale) * x_nm_per_px,
                ]
                for py, px in polyline_local
            ]
            if len(ridge_path_nm) < 2:
                ridge_path_nm = [[start_y, start_x], [end_y, end_x]]
        else:
            ridge_path_nm = [[start_y, start_x], [end_y, end_x]]

        rows.append(
            [
                float(y) * float(downscale) * y_nm_per_px,
                float(x) * float(downscale) * x_nm_per_px,
                ridge_size,
            ]
        )
        segment_meta.append(
            {
                "ridge_start_y_nm": start_y,
                "ridge_start_x_nm": start_x,
                "ridge_end_y_nm": end_y,
                "ridge_end_x_nm": end_x,
                "ridge_length_nm": ridge_length_nm,
                "ridge_path_nm": ridge_path_nm,
            }
        )
        if len(rows) >= int(max_detections):
            break

    if len(rows) == 0 and num > 0:
        component_scores = np.asarray(ndi.sum(response, labels, index=range(1, num + 1)), dtype=np.float32)
        if component_scores.size:
            fallback_order = np.argsort(component_scores)[::-1]
            for order_idx in fallback_order:
                label_id = int(order_idx) + 1
                if order_idx >= len(component_slices):
                    continue
                fallback = simple_component_meta(label_id, component_slices[order_idx])
                if fallback is None:
                    continue
                rows.append(fallback["row"])
                segment_meta.append(fallback["meta"])
                if len(rows) >= int(max_detections):
                    break

    if not hough_seeded and len(rows) < max(8, int(max_detections) // 8):
        for fallback in hough_line_candidates():
            rows.append(fallback["row"])
            segment_meta.append(fallback["meta"])
            if len(rows) >= int(max_detections):
                break

    if hole_mask is not None and rows:
        kept = []
        for i, row in enumerate(rows):
            cy_ds = float(row[0]) / (float(downscale) * max(float(y_nm_per_px), 1e-9))
            cx_ds = float(row[1]) / (float(downscale) * max(float(x_nm_per_px), 1e-9))
            iy = int(np.clip(round(cy_ds), 0, hole_mask.shape[0] - 1))
            ix = int(np.clip(round(cx_ds), 0, hole_mask.shape[1] - 1))
            if hole_mask[iy, ix]:
                kept.append(i)
        rows = [rows[i] for i in kept]
        segment_meta = [segment_meta[i] for i in kept]

    rows_np = np.asarray(rows, dtype=np.float32).reshape(-1, 3)
    if len(rows_np) == 0:
        return rows_np, []

    order = np.argsort(rows_np[:, 2])[::-1]
    taken = np.zeros(len(rows_np), dtype=bool)
    merged_rows = []
    merged_meta = []
    for order_idx in order:
        if taken[order_idx]:
            continue
        base = rows_np[order_idx]
        cluster = [int(order_idx)]
        taken[order_idx] = True
        by, bx, br = map(float, base)
        for other_idx in order:
            if taken[other_idx]:
                continue
            oy, ox, oradius = map(float, rows_np[other_idx])
            dist = float(np.hypot(oy - by, ox - bx))
            if dist <= max(2.0, 0.85 * max(br, oradius)):
                taken[other_idx] = True
                cluster.append(int(other_idx))
        cluster_rows = rows_np[cluster]
        weights = np.maximum(cluster_rows[:, 2], 1.0)
        merged_rows.append(
            [
                float(np.average(cluster_rows[:, 0], weights=weights)),
                float(np.average(cluster_rows[:, 1], weights=weights)),
                float(np.max(cluster_rows[:, 2])),
            ]
        )
        cluster_meta = [segment_meta[i] for i in cluster]
        longest = max(cluster_meta, key=lambda meta: float(meta["ridge_length_nm"]))
        merged_meta.append(longest)
    return np.asarray(merged_rows, dtype=np.float32).reshape(-1, 3), merged_meta


def _detect_distance_watershed_np(
    image_np,
    *,
    y_nm_per_px: float,
    x_nm_per_px: float,
    downscale: float,
    min_radius_nm: float,
    max_radius_nm: float,
):
    """Distance-transform watershed to recover overlapping particles that LoG
    misses. Returns rows [y_nm, x_nm, radius_nm]. Independent of the Hessian
    detector: thresholds the (inverted) image, distance-transforms the foreground,
    seeds one marker per distance peak, and watersheds to split touching blobs."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.filters import gaussian, threshold_otsu
    from skimage.segmentation import watershed
    from skimage.transform import rescale

    ds = max(float(downscale), 1.0)
    px_nm = float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
    arr = np.asarray(image_np, dtype=np.float32)
    small = rescale(arr, 1.0 / ds, anti_aliasing=True) if ds > 1.0 else arr
    inv = small.max() - small                       # particles are dark -> bright
    rng = np.ptp(inv)
    if rng <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    inv = (inv - inv.min()) / (rng + 1e-9)
    sm = gaussian(inv, sigma=2)
    try:
        thr = float(threshold_otsu(sm))
    except Exception:
        thr = float(sm.mean() + 0.5 * sm.std())
    mask = ndi.binary_fill_holes(sm > thr)
    if mask.sum() == 0:
        return np.zeros((0, 3), dtype=np.float32)

    dist = ndi.distance_transform_edt(mask)
    min_r_ds = max(3.0, (min_radius_nm / px_nm) / ds)
    max_r_ds = max(min_r_ds + 1.0, (max_radius_nm / px_nm) / ds)
    peaks = peak_local_max(dist, min_distance=int(min_r_ds), labels=mask,
                           exclude_border=False)
    if len(peaks) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    markers = np.zeros_like(dist, dtype=np.int32)
    for i, (y, x) in enumerate(peaks, start=1):
        markers[y, x] = i
    labels = watershed(-dist, markers, mask=mask)

    rows = []
    for lab in range(1, int(labels.max()) + 1):
        ys, xs = np.where(labels == lab)
        if len(ys) < 4:
            continue
        r_ds = (len(ys) / np.pi) ** 0.5
        if not (min_r_ds <= r_ds <= max_r_ds):
            continue
        cy = float(ys.mean()) * ds
        cx = float(xs.mean()) * ds
        rows.append([cy * y_nm_per_px, cx * x_nm_per_px, r_ds * ds * px_nm])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 3)


def _merge_watershed_into_log(log_rows, watershed_rows, *, tol_factor: float = 0.6):
    """Keep every LoG detection (high precision); add only watershed detections
    that LoG did not already find (recovers overlapping-cluster particles). All
    rows are [y_nm, x_nm, radius_nm]."""
    import numpy as np

    log_rows = np.asarray(log_rows, dtype=np.float32).reshape(-1, 3)
    watershed_rows = np.asarray(watershed_rows, dtype=np.float32).reshape(-1, 3)
    if len(log_rows) == 0:
        return watershed_rows
    if len(watershed_rows) == 0:
        return log_rows
    keep = []
    for w in watershed_rows:
        d = np.hypot(log_rows[:, 0] - w[0], log_rows[:, 1] - w[1])
        if np.all(d > tol_factor * np.maximum(w[2], log_rows[:, 2])):
            keep.append(w)
    if not keep:
        return log_rows
    return np.vstack([log_rows, np.asarray(keep, dtype=np.float32)])


def _detect_watershed_rows_np(
    hessian_rows,
    *,
    image_shape: tuple[int, int],
    downscale: float,
    min_marker_distance: float,
    y_nm_per_px: float,
    x_nm_per_px: float,
    run_mode: str,
):
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.feature import peak_local_max
    from skimage.segmentation import watershed

    rows = _merge_nearby_rows(hessian_rows, distance_factor=0.65)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    scaled_shape = (
        max(1, int(round(image_shape[0] / max(float(downscale), 1.0)))),
        max(1, int(round(image_shape[1] / max(float(downscale), 1.0)))),
    )
    mask = np.zeros(scaled_shape, dtype=bool)
    size_px_nm = float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
    yy, xx = np.ogrid[: scaled_shape[0], : scaled_shape[1]]

    for y_nm, x_nm, size_nm in rows:
        y = float(y_nm) / (max(y_nm_per_px, 1e-9) * max(float(downscale), 1.0))
        x = float(x_nm) / (max(x_nm_per_px, 1e-9) * max(float(downscale), 1.0))
        radius = max(2.0, float(size_nm) / max(size_px_nm * max(float(downscale), 1.0), 1e-9))
        mask |= ((yy - y) ** 2 + (xx - x) ** 2) <= radius**2

    if not mask.any():
        return np.zeros((0, 3), dtype=np.float32)

    final_mode = str(run_mode).lower() == "final"
    connected, n_connected = ndi.label(mask)
    if n_connected <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    labels = np.zeros_like(connected, dtype=np.int32)
    next_label = 1
    median_seed_radius = float(
        np.median(rows[:, 2]) / max(size_px_nm * max(float(downscale), 1.0), 1e-9)
    )
    component_slices = ndi.find_objects(connected)
    for comp_id, slc in enumerate(component_slices, start=1):
        if slc is None:
            continue
        component_mask = connected[slc] == comp_id
        if not component_mask.any():
            continue
        distance = ndi.distance_transform_edt(component_mask)
        local_min_distance = max(
            1,
            int(
                round(
                    max(
                        float(min_marker_distance),
                        (0.45 if final_mode else 0.65) * median_seed_radius,
                    )
                )
            ),
        )
        peak_coords = peak_local_max(
            distance,
            labels=component_mask,
            exclude_border=False,
            min_distance=local_min_distance,
        )
        if len(peak_coords) == 0:
            continue
        markers = np.zeros_like(distance, dtype=np.int32)
        for idx, (y, x) in enumerate(peak_coords, start=1):
            markers[int(y), int(x)] = idx
        local_labels = watershed(-distance, markers, mask=component_mask)
        if local_labels.max() <= 0:
            continue
        for local_id in range(1, int(local_labels.max()) + 1):
            region = local_labels == local_id
            if not region.any():
                continue
            label_view = labels[slc]
            label_view[region] = next_label
            next_label += 1

    out = []
    h, w = scaled_shape
    border_margin = max(2, int(round(min(h, w) * 0.015)))
    for label_id in range(1, int(labels.max()) + 1):
        region = labels == label_id
        if not region.any():
            continue
        ys, xs = np.where(region)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        if (
            y0 <= border_margin
            or x0 <= border_margin
            or y1 >= h - border_margin
            or x1 >= w - border_margin
        ):
            continue
        area = int(region.sum())
        if area < 12:
            continue
        y_span = max(1, y1 - y0)
        x_span = max(1, x1 - x0)
        aspect = float(max(y_span, x_span) / max(1, min(y_span, x_span)))
        fill_ratio = float(area / max(1, y_span * x_span))
        if aspect > 2.5 or fill_ratio < 0.22:
            continue
        y, x = ndi.center_of_mass(region)
        radius_px = float(np.sqrt(region.sum() / np.pi))
        out.append(
            [
                float(y) * float(downscale) * y_nm_per_px,
                float(x) * float(downscale) * x_nm_per_px,
                radius_px * float(downscale) * size_px_nm,
            ]
        )
    out_np = np.asarray(out, dtype=np.float32).reshape(-1, 3)
    return _merge_nearby_rows(out_np, distance_factor=0.8)


def _read_display_image(file_path: str):
    import mrcfile
    import tifffile
    from PIL import Image

    from acorn.core.dm4_loader import DM4Image

    path = Path(file_path)
    ext = path.suffix.lower()
    if ext == ".dm4":
        image = _coerce_to_grayscale_2d(DM4Image.from_file(file_path).raw)
    elif ext in {".mrc", ".mrcs"}:
        with mrcfile.mmap(file_path, mode="r") as data_mrc:
            data = data_mrc.data
            image = _coerce_to_grayscale_2d(data)
    elif ext in {".tif", ".tiff"}:
        image = _coerce_to_grayscale_2d(tifffile.imread(file_path))
    else:
        pil_image = Image.open(file_path)
        if pil_image.mode not in ("L", "I", "F"):
            pil_image = pil_image.convert("L")
        image = _coerce_to_grayscale_2d(pil_image)

    import numpy as np
    finite = np.isfinite(image)
    if not finite.any():
        return np.zeros(image.shape, dtype=np.uint8)
    lo, hi = np.percentile(image[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    return (np.clip((image - lo) / (hi - lo), 0.0, 1.0) * 255.0).astype(np.uint8)


def _export_run_outputs(df, output_csv: str) -> str:
    """Write annotated PNGs and publication-friendly tables beside the CSV."""
    import pandas as pd
    from PIL import Image, ImageDraw

    output_path = Path(output_csv)
    export_dir = output_path.with_suffix("")
    export_dir = export_dir.with_name(f"{export_dir.name}_outputs")
    annotated_dir = export_dir / "annotated_images"
    export_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    measurements_path = export_dir / "particle_measurements.csv"
    df.to_csv(measurements_path, index=False)

    summary_rows = []
    if df.empty:
        pd.DataFrame(
            columns=[
                "File Location",
                "Particle Count",
                "Mean Size (nm)",
                "Median Size (nm)",
                "Std Size (nm)",
                "Pixel Size Y (nm/px)",
                "Pixel Size X (nm/px)",
                "Pixel Size Source",
                "Size Measurement",
            ]
        ).to_csv(export_dir / "summary_by_file.csv", index=False)
        return str(export_dir)

    for file_path, group in df.groupby("File Location", sort=False):
        image_stem = Path(str(file_path)).stem
        per_image_csv = annotated_dir / f"{image_stem}_cryoblob_particles.csv"
        group.to_csv(per_image_csv, index=False)
        try:
            image = _read_display_image(str(file_path))
            rgb = Image.fromarray(image, mode="L").convert("RGB")
            draw = ImageDraw.Draw(rgb)
            first = group.iloc[0]
            y_px_nm = float(first.get("Pixel Size Y (nm/px)", 1.0) or 1.0)
            x_px_nm = float(first.get("Pixel Size X (nm/px)", 1.0) or 1.0)
            size_px_nm = float((y_px_nm * x_px_nm) ** 0.5)
            for _, row in group.iterrows():
                if str(row.get("Detection Type", "")) == "ridge" and row.get("Ridge Length (nm)"):
                    path_raw = row.get("Ridge Path (nm)")
                    path_pts = []
                    if path_raw:
                        try:
                            path_nm = json.loads(path_raw)
                            path_pts = [
                                (float(x_nm) / max(x_px_nm, 1e-9), float(y_nm) / max(y_px_nm, 1e-9))
                                for y_nm, x_nm in path_nm
                            ]
                        except Exception:
                            path_pts = []
                    if len(path_pts) >= 2:
                        draw.line(path_pts, fill=(0, 255, 136), width=3)
                    else:
                        x0 = float(row["Ridge Start X (nm)"]) / max(x_px_nm, 1e-9)
                        y0 = float(row["Ridge Start Y (nm)"]) / max(y_px_nm, 1e-9)
                        x1 = float(row["Ridge End X (nm)"]) / max(x_px_nm, 1e-9)
                        y1 = float(row["Ridge End Y (nm)"]) / max(y_px_nm, 1e-9)
                        draw.line([x0, y0, x1, y1], fill=(0, 255, 136), width=3)
                else:
                    x = float(row["Center X (nm)"]) / max(x_px_nm, 1e-9)
                    y = float(row["Center Y (nm)"]) / max(y_px_nm, 1e-9)
                    radius = 0.5 * float(row["Size (nm)"]) / max(size_px_nm, 1e-9)
                    box = [x - radius, y - radius, x + radius, y + radius]
                    draw.ellipse(box, outline=(0, 255, 136), width=3)
            annotated_path = annotated_dir / f"{image_stem}_cryoblob_annotated.png"
            rgb.save(annotated_path)
        except Exception:
            pass

        summary_rows.append(
            {
                "File Location": file_path,
                "Particle Count": int(len(group)),
                "Mean Size (nm)": float(group["Size (nm)"].mean()),
                "Median Size (nm)": float(group["Size (nm)"].median()),
                "Std Size (nm)": float(group["Size (nm)"].std(ddof=1)) if len(group) > 1 else 0.0,
                "Pixel Size Y (nm/px)": float(group["Pixel Size Y (nm/px)"].iloc[0]),
                "Pixel Size X (nm/px)": float(group["Pixel Size X (nm/px)"].iloc[0]),
                "Pixel Size Source": str(group["Pixel Size Source"].iloc[0]),
                "Size Measurement": str(group["Size Measurement"].iloc[0]),
            }
        )

    pd.DataFrame(summary_rows).to_csv(export_dir / "summary_by_file.csv", index=False)
    return str(export_dir)


def _install_cryoblob_gpu_backend() -> None:
    """Patch CryoBLOB runtime pieces that break in the ACORN environment."""
    import cryoblob
    import cryoblob.blobs as cryoblob_blobs
    import cryoblob.image as cryoblob_image
    import cryoblob.multi as cryoblob_multi
    import jax
    import jax.numpy as jnp
    import numpy as np

    if getattr(cryoblob_blobs.blob_list_log, "_acorn_gpu_patched", False):
        pass
    else:
        def acorn_gpu_blob_list_log(
            mrc_image,
            min_blob_size=5,
            max_blob_size=20,
            blob_step=1,
            downscale=4,
            std_threshold=0.1,
            max_detections=100,
        ):
            del blob_step
            pixel_rows = _detect_blobs_jax(
                mrc_image.image_data,
                blob_downscale=float(downscale),
                min_sigma=float(min_blob_size),
                max_sigma=float(max_blob_size),
                threshold_rel=float(std_threshold),
                max_detections=int(max_detections),
            )
            if len(pixel_rows) == 0:
                return jnp.zeros((0, 3), dtype=jnp.float32)
            voxel_size = np.asarray(mrc_image.voxel_size, dtype=np.float32)
            y_scale = float(voxel_size[1])
            x_scale = float(voxel_size[2])
            size_scale = float(np.sqrt(y_scale * x_scale))
            scaled = np.zeros_like(pixel_rows, dtype=np.float32)
            scaled[:, 0] = pixel_rows[:, 0] * float(downscale) * y_scale
            scaled[:, 1] = pixel_rows[:, 1] * float(downscale) * x_scale
            scaled[:, 2] = pixel_rows[:, 2] * float(downscale) * size_scale
            return jnp.asarray(scaled, dtype=jnp.float32)

        acorn_gpu_blob_list_log._acorn_gpu_patched = True
        cryoblob.blob_list_log = acorn_gpu_blob_list_log
        cryoblob_blobs.blob_list_log = acorn_gpu_blob_list_log

    if not getattr(cryoblob_image.image_resizer, "_acorn_gpu_patched", False):
        def acorn_image_resizer(image, downscale=4):
            image = jnp.asarray(image, dtype=jnp.float32)
            factor = max(float(downscale), 1.0)
            if factor <= 1.0:
                return image
            h, w = image.shape
            scaled_shape = (
                max(1, int(round(h / factor))),
                max(1, int(round(w / factor))),
            )
            return jax.image.resize(image, scaled_shape, method="linear", antialias=True)

        acorn_image_resizer._acorn_gpu_patched = True
        cryoblob_image.image_resizer = acorn_image_resizer
        cryoblob.image_resizer = acorn_image_resizer
        cryoblob_multi.image_resizer = acorn_image_resizer


def detect_contrast_polarity(image) -> str:
    """
    Report whether the features in *image* are darker or lighter than the field.

    CryoBLOB looks for density minima, which is right for cryo-EM: particles
    scatter electrons and come out dark on a light field. Hand the same detector
    an inverted image - a rendered mask, a STEM ADF frame, a TIFF someone already
    contrast-flipped - and it returns zero detections with no error and no way to
    tell that the data was the problem.

    Sparse features skew the intensity histogram in the direction they sit. This
    compares how far the bright tail runs above the median against how far the
    dark tail runs below it, using percentiles so a handful of hot pixels cannot
    decide the answer.

    Returns "dark" (features darker than the field - CryoBLOB's native case) or
    "light".
    """
    import numpy as np

    finite = np.asarray(image, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return "dark"
    lo, mid, hi = np.percentile(finite, [2.0, 50.0, 98.0])
    bright_tail = float(hi - mid)
    dark_tail = float(mid - lo)
    # A clear majority is needed to overturn the default, so near-symmetric
    # histograms (dense or textured fields) stay on CryoBLOB's native case.
    if bright_tail > dark_tail * 1.25:
        return "light"
    return "dark"


def _apply_contrast_polarity(image, mode: str):
    """
    Return (image_for_detection, polarity_actually_used).

    mode is "auto" (measure it), "dark" (assume CryoBLOB's native case), or
    "light" (features are brighter than the field, so invert before detecting).
    Inversion preserves the intensity range, so sigma and threshold settings
    carry over unchanged.
    """
    import numpy as np

    choice = str(mode or "auto").strip().lower()
    if choice not in {"auto", "dark", "light"}:
        choice = "auto"
    if choice == "auto":
        choice = detect_contrast_polarity(image)
    if choice == "light":
        arr = np.asarray(image)
        inverted = (arr.max() + arr.min()) - arr
        print("[CryoBLOB] features are lighter than the field - inverting before detection",
              flush=True)
        return inverted, "light"
    return image, "dark"


def _process_single_file(
    file_path: str,
    *,
    pixel_size_nm: float,
    bin_factor: int = 1,
    run_mode: str,
    detection_mode: str,
    blob_downscale: float,
    min_sigma: float,
    max_sigma: float,
    blob_step: float,
    threshold_rel: float,
    max_detections: int,
    refine_sizes: bool,
    size_scale: float,
    ridge_threshold: float,
    ridge_scales: int,
    min_marker_distance: float,
    use_ridge_detection: bool,
    use_watershed: bool,
    stream_large_files: bool,
    exponential: bool,
    logarizer: bool,
    gblur: int,
    background: int,
    apply_filter: int,
    cache_results: bool,
    contrast_polarity: str = "auto",
):
    """Run the requested CryoBLOB detector and return ACORN-friendly records."""
    _configure_jax_memory()

    import cryoblob
    import jax.numpy as jnp
    import numpy as np
    from cryoblob.types import MRC_Image

    im_np, y_nm_per_px, x_nm_per_px, pixel_size_source = _load_source_image(
        file_path,
        pixel_size_nm,
        use_cache=bool(cache_results),
        bin_factor=int(bin_factor or 1),
    )

    im_np, _polarity_used = _apply_contrast_polarity(im_np, contrast_polarity)

    _install_cryoblob_gpu_backend()
    preprocessed_image = _get_preprocessed_image(
        file_path,
        im_np,
        use_cache=bool(cache_results),
        exponential=exponential,
        logarizer=logarizer,
        gblur=gblur,
        background=background,
        apply_filter=apply_filter,
    )
    jax_preprocessed = jnp.asarray(preprocessed_image, dtype=jnp.float32)
    mrc_image = MRC_Image(
        image_data=jax_preprocessed,
        voxel_size=jnp.asarray([1.0, y_nm_per_px, x_nm_per_px], dtype=jnp.float32),
        origin=jnp.zeros(3, dtype=jnp.float32),
        data_min=jnp.min(jax_preprocessed),
        data_max=jnp.max(jax_preprocessed),
        data_mean=jnp.mean(jax_preprocessed),
        mode=jnp.asarray(2, dtype=jnp.int32),
    )
    _install_cryoblob_gpu_backend()
    records: list[dict] = []
    detector = str(detection_mode or "log").lower()
    execution_mode = str(run_mode or "preview").lower()
    preview_mode = execution_mode != "final"
    effective_blob_downscale = max(float(blob_downscale), 1.0)
    effective_max_detections = int(max_detections)
    effective_ridge_scales = int(ridge_scales)
    effective_refine_sizes = bool(refine_sizes)
    if preview_mode:
        effective_blob_downscale = max(effective_blob_downscale, 1.5 if detector == "ridge" else 2.0)
        effective_max_detections = min(effective_max_detections, 120 if detector in {"ridge", "enhanced"} else 80)
        effective_ridge_scales = min(effective_ridge_scales, 10)
        if detector != "log":
            effective_refine_sizes = False

    def finalize_rows(rows_in, detection_type: str, allow_refine: bool, extra_meta=None) -> None:
        rows = np.asarray(rows_in, dtype=np.float32).reshape(-1, 3)
        if rows.size == 0:
            return
        raw_rows = rows.copy()
        if allow_refine:
            rows, size_measurement = _refine_blob_sizes_from_image(
                im_np,
                rows,
                y_nm_per_px=y_nm_per_px,
                x_nm_per_px=x_nm_per_px,
                enabled=effective_refine_sizes,
                size_scale=float(size_scale),
            )
        else:
            rows = rows.copy()
            rows[:, 2] *= max(0.01, float(size_scale))
            size_measurement = "CryoBLOB ridge scale"

        if extra_meta is None:
            extra_meta = [None] * len(rows)
        for idx, (blob, raw_blob, meta) in enumerate(
            zip(rows, raw_rows, extra_meta, strict=False), start=len(records) + 1
        ):
            records.append(
                {
                    "File Location": file_path,
                    "Particle ID": idx,
                    "Detection Mode": detector,
                    "Detection Type": detection_type,
                    "Center Y (nm)": float(blob[0]),
                    "Center X (nm)": float(blob[1]),
                    "Radius (nm)": float(blob[2]),
                    "Size (nm)": float(blob[2]) * 2.0,
                    "Raw CryoBLOB Size (nm)": float(raw_blob[2]) * 2.0,
                    "Size Measurement": f"{size_measurement} (diameter reported)",
                    "Ridge Start Y (nm)": float(meta["ridge_start_y_nm"]) if meta else None,
                    "Ridge Start X (nm)": float(meta["ridge_start_x_nm"]) if meta else None,
                    "Ridge End Y (nm)": float(meta["ridge_end_y_nm"]) if meta else None,
                    "Ridge End X (nm)": float(meta["ridge_end_x_nm"]) if meta else None,
                    "Ridge Length (nm)": float(meta["ridge_length_nm"]) if meta else None,
                    "Ridge Path (nm)": json.dumps(meta["ridge_path_nm"]) if meta and meta.get("ridge_path_nm") else None,
                    "Pixel Size Y (nm/px)": float(y_nm_per_px),
                    "Pixel Size X (nm/px)": float(x_nm_per_px),
                    "Pixel Size Source": pixel_size_source,
                }
            )

    if detector == "log":
        rows = np.asarray(
            cryoblob.blob_list_log(
                mrc_image,
                min_blob_size=float(min_sigma),
                max_blob_size=float(max_sigma),
                downscale=effective_blob_downscale,
                std_threshold=max(0.001, float(threshold_rel)),
                max_detections=effective_max_detections,
            ),
            dtype=np.float32,
        )
        finalize_rows(rows, "blob", allow_refine=effective_refine_sizes)
    elif detector in ("log_watershed", "log+watershed"):
        # LoG (high precision) + distance-transform watershed (recovers overlapping
        # particles LoG misses). Keep all LoG detections; add only the watershed
        # detections LoG didn't already find. Unlike "enhanced" this is LoG-based,
        # not Hessian-based, so it preserves LoG's precision.
        log_rows = np.asarray(
            cryoblob.blob_list_log(
                mrc_image,
                min_blob_size=float(min_sigma),
                max_blob_size=float(max_sigma),
                downscale=effective_blob_downscale,
                std_threshold=max(0.001, float(threshold_rel)),
                max_detections=effective_max_detections,
            ),
            dtype=np.float32,
        ).reshape(-1, 3)
        # Watershed size window derived from the LoG sigma range (radius = sigma*sqrt2).
        r_lo_nm = max(1.0, float(min_sigma) * 1.414 * effective_blob_downscale * 0.6
                      * float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9))))
        r_hi_nm = float(max_sigma) * 1.414 * effective_blob_downscale * 3.0 \
            * float(np.sqrt(max(y_nm_per_px, 1e-9) * max(x_nm_per_px, 1e-9)))
        wat_rows = _detect_distance_watershed_np(
            im_np, y_nm_per_px=y_nm_per_px, x_nm_per_px=x_nm_per_px,
            downscale=effective_blob_downscale,
            min_radius_nm=r_lo_nm, max_radius_nm=r_hi_nm,
        )
        merged = _merge_watershed_into_log(log_rows, wat_rows)
        finalize_rows(merged, "log_watershed", allow_refine=effective_refine_sizes)
    elif detector == "hessian":
        rows = _detect_hessian_rows_np(
            file_path,
            im_np,
            downscale=effective_blob_downscale,
            min_sigma=float(min_sigma),
            max_sigma=float(max_sigma),
            blob_step=max(0.1, float(blob_step)),
            std_threshold=max(0.001, float(threshold_rel)),
            max_detections=effective_max_detections,
            y_nm_per_px=y_nm_per_px,
            x_nm_per_px=x_nm_per_px,
            use_cache=bool(cache_results),
        )
        finalize_rows(rows, "hessian_blob", allow_refine=effective_refine_sizes)
    elif detector == "ridge":
        rows, ridge_meta = _detect_ridge_rows_np(
            file_path,
            im_np,
            downscale=effective_blob_downscale,
            min_sigma=float(min_sigma),
            max_sigma=float(max_sigma),
            ridge_threshold=max(0.0001, float(ridge_threshold)),
            ridge_scales=max(2, effective_ridge_scales),
            max_detections=effective_max_detections,
            y_nm_per_px=y_nm_per_px,
            x_nm_per_px=x_nm_per_px,
            use_cache=bool(cache_results),
            run_mode=execution_mode,
        )
        finalize_rows(rows, "ridge", allow_refine=False, extra_meta=ridge_meta)
    elif detector == "enhanced":
        circular = _detect_hessian_rows_np(
            file_path,
            im_np,
            downscale=effective_blob_downscale,
            min_sigma=float(min_sigma),
            max_sigma=float(max_sigma),
            blob_step=max(0.1, float(blob_step)),
            std_threshold=max(0.001, float(threshold_rel)),
            max_detections=effective_max_detections,
            y_nm_per_px=y_nm_per_px,
            x_nm_per_px=x_nm_per_px,
            use_cache=bool(cache_results),
        )
        elongated = (
            _detect_ridge_rows_np(
                file_path,
                im_np,
                downscale=effective_blob_downscale,
                min_sigma=float(min_sigma),
                max_sigma=float(max_sigma),
                ridge_threshold=max(0.0001, float(ridge_threshold)),
                ridge_scales=max(2, effective_ridge_scales),
                max_detections=effective_max_detections,
                y_nm_per_px=y_nm_per_px,
                x_nm_per_px=x_nm_per_px,
                use_cache=bool(cache_results),
                run_mode=execution_mode,
            )
            if use_ridge_detection
            else (np.zeros((0, 3), dtype=np.float32), [])
        )
        watershed = (
            _detect_watershed_rows_np(
                circular,
                image_shape=im_np.shape,
                downscale=effective_blob_downscale,
                min_marker_distance=max(0.5, float(min_marker_distance)),
                y_nm_per_px=y_nm_per_px,
                x_nm_per_px=x_nm_per_px,
                run_mode=execution_mode,
            )
            if use_watershed
            else np.zeros((0, 3), dtype=np.float32)
        )
        finalize_rows(np.asarray(circular, dtype=np.float32), "hessian_blob", allow_refine=effective_refine_sizes)
        elongated_rows, elongated_meta = elongated
        finalize_rows(np.asarray(elongated_rows, dtype=np.float32), "ridge", allow_refine=False, extra_meta=elongated_meta)
        finalize_rows(np.asarray(watershed, dtype=np.float32), "watershed_blob", allow_refine=effective_refine_sizes)
        if len(records) > effective_max_detections:
            records = records[:effective_max_detections]
            for idx, record in enumerate(records, start=1):
                record["Particle ID"] = idx
    else:
        raise ValueError(f"Unsupported CryoBLOB detection mode: {detection_mode}")

    return records


class CryoBlobThread(QThread):
    """Run CryoBLOB on one or more MRC files without blocking the GUI."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, str, object)  # dataframe, output_csv, failures
    error = pyqtSignal(str)

    def __init__(
        self,
        *,
        files: list[str],
        output_csv: str,
        pixel_size_nm: float,
        run_mode: str,
        detection_mode: str,
        blob_downscale: float,
        min_sigma: float,
        max_sigma: float,
        blob_step: float,
        threshold_rel: float,
        max_detections: int,
        refine_sizes: bool,
        size_scale: float,
        ridge_threshold: float,
        ridge_scales: int,
        min_marker_distance: float,
        use_ridge_detection: bool,
        use_watershed: bool,
        stream_large_files: bool,
        exponential: bool,
        logarizer: bool,
        gblur: int,
        background: int,
        apply_filter: int,
        cache_results: bool,
        contrast_polarity: str = "auto",
        bin_factor: int = 1,
    ) -> None:
        super().__init__()
        self._files = [str(Path(p)) for p in files]
        self._output_csv = str(Path(output_csv))
        self._pixel_size_nm = float(pixel_size_nm)
        self._bin_factor = int(bin_factor or 1)
        self._run_mode = str(run_mode)
        self._detection_mode = str(detection_mode)
        self._blob_downscale = float(blob_downscale)
        self._min_sigma = float(min_sigma)
        self._max_sigma = float(max_sigma)
        self._blob_step = float(blob_step)
        self._threshold_rel = float(threshold_rel)
        self._max_detections = int(max_detections)
        self._refine_sizes = bool(refine_sizes)
        self._size_scale = float(size_scale)
        self._ridge_threshold = float(ridge_threshold)
        self._ridge_scales = int(ridge_scales)
        self._min_marker_distance = float(min_marker_distance)
        self._use_ridge_detection = bool(use_ridge_detection)
        self._use_watershed = bool(use_watershed)
        self._stream_large_files = bool(stream_large_files)
        self._exponential = bool(exponential)
        self._logarizer = bool(logarizer)
        self._gblur = int(gblur)
        self._background = int(background)
        self._apply_filter = int(apply_filter)
        self._cache_results = bool(cache_results)
        self._contrast_polarity = str(contrast_polarity or "auto")

    def run(self) -> None:
        _configure_jax_memory()

        try:
            import pandas as pd
        except ImportError:
            self.error.emit(
                "CryoBLOB is not installed in the ACORN environment.\n"
                "Install the plugin package in the same environment as ACORN."
            )
            return

        if not self._files:
            self.error.emit("No supported image files were selected for CryoBLOB processing.")
            return

        records: list[dict] = []
        failures: list[tuple[str, str]] = []
        total = len(self._files)

        for idx, file_path in enumerate(self._files, start=1):
            path = Path(file_path)
            self.progress.emit(
                int((idx - 1) * 100 / total),
                f"Processing {path.name} ({idx}/{total})",
            )
            try:
                file_records = _process_single_file(
                    str(path),
                    pixel_size_nm=self._pixel_size_nm,
                    bin_factor=getattr(self, "_bin_factor", 1),
                    run_mode=self._run_mode,
                    detection_mode=self._detection_mode,
                    blob_downscale=self._blob_downscale,
                    min_sigma=self._min_sigma,
                    max_sigma=self._max_sigma,
                    blob_step=self._blob_step,
                    threshold_rel=self._threshold_rel,
                    max_detections=self._max_detections,
                    refine_sizes=self._refine_sizes,
                    size_scale=self._size_scale,
                    ridge_threshold=self._ridge_threshold,
                    ridge_scales=self._ridge_scales,
                    min_marker_distance=self._min_marker_distance,
                    use_ridge_detection=self._use_ridge_detection,
                    use_watershed=self._use_watershed,
                    stream_large_files=self._stream_large_files,
                    exponential=self._exponential,
                    logarizer=self._logarizer,
                    gblur=self._gblur,
                    background=self._background,
                    apply_filter=self._apply_filter,
                    cache_results=self._cache_results,
                    contrast_polarity=self._contrast_polarity,
                )
                records.extend(file_records)
            except Exception as exc:
                failures.append((str(path), str(exc)))

            self.progress.emit(
                int(idx * 100 / total),
                f"Finished {path.name} ({idx}/{total})",
            )

        df = pd.DataFrame(records, columns=DEFAULT_COLUMNS)
        output_path = Path(self._output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        _export_run_outputs(df, str(output_path))
        self.finished.emit(df, str(output_path), failures)
