"""
micro-SAM (μSAM) predictor — loads fine-tuned SAM1 checkpoints from the
micro-sam project (computational-cell-analytics) without requiring the
micro_sam package itself (which has an unresolvable C++ dependency in pip
environments).

Checkpoints are downloaded on first use to ~/.cache/micro_sam/ (or the path
set by the MICROSAM_CACHEDIR environment variable).

Install:
    pip install git+https://github.com/facebookresearch/segment-anything.git

Usage:
    pred = MicroSAMPredictor(model_type="vit_b_lm")
    pred.load_model()
    pred.encode_image(img8)
    masks = pred.predict_points(img8, [(x, y)], [1])
    mask  = pred.predict_box(img8, (x0, y0, x1, y1))
    masks = pred.predict_everything(img8)
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np

# ── model registry ─────────────────────────────────────────────────────────
# Maps model_type → (download_url, SAM encoder architecture)
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    # Light microscopy fine-tuned (recommended for cryo-EM)
    "vit_b_lm": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
        "vit_b",
    ),
    "vit_l_lm": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l.pt",
        "vit_l",
    ),
    # Electron microscopy organelle fine-tuned
    "vit_b_em_organelles": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt",
        "vit_b",
    ),
    "vit_l_em_organelles": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1.2/files/vit_l.pt",
        "vit_l",
    ),
    # Generic SAM1 (Meta) — no fine-tuning
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "vit_b",
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "vit_l",
    ),
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "vit_h",
    ),
}

_DISPLAY_NAMES: dict[str, str] = {
    "vit_b_lm":           "vit_b_lm  —  light microscopy (recommended)",
    "vit_l_lm":           "vit_l_lm  —  light microscopy, large",
    "vit_b_em_organelles":"vit_b_em_organelles  —  EM organelles",
    "vit_l_em_organelles":"vit_l_em_organelles  —  EM organelles, large",
    "vit_b":              "vit_b  —  generic SAM",
    "vit_l":              "vit_l  —  generic SAM, large",
    "vit_h":              "vit_h  —  generic SAM, huge",
}


def available_models() -> list[tuple[str, str]]:
    """Return list of (model_type, display_name) pairs."""
    return [(k, _DISPLAY_NAMES[k]) for k in _MODEL_REGISTRY]


# Shared system-wide model directory (populated by install.sh / download_models.py)
_SHARED_MODELS_DIR = Path("/opt/acorn/models/micro_sam")


def cache_dir() -> Path:
    env = os.environ.get("MICROSAM_CACHEDIR")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "micro_sam"


def _checkpoint_path(model_type: str) -> Path:
    url, arch = _MODEL_REGISTRY[model_type]
    filename = url.rsplit("/", 1)[-1]
    return cache_dir() / model_type / filename


def _find_existing_checkpoint(model_type: str) -> Optional[Path]:
    """
    Locate an already-downloaded checkpoint without downloading.

    Search order:
      1. User's local cache  (~/.cache/micro_sam/<type>/<file>)
      2. Shared system cache (/opt/acorn/models/micro_sam/<type>/<file>)
    Returns None if not found.
    """
    url, _ = _MODEL_REGISTRY[model_type]
    filename = url.rsplit("/", 1)[-1]

    user_path = cache_dir() / model_type / filename
    if user_path.exists():
        return user_path

    shared_path = _SHARED_MODELS_DIR / model_type / filename
    if shared_path.exists():
        return shared_path

    return None


def _download(model_type: str, progress_cb=None) -> Path:
    """Download checkpoint if not already cached. Returns local path."""
    url, _ = _MODEL_REGISTRY[model_type]
    existing = _find_existing_checkpoint(model_type)
    if existing is not None:
        return existing
    dest = _checkpoint_path(model_type)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        def _reporthook(count, block_size, total_size):
            if progress_cb and total_size > 0:
                pct = min(100, int(count * block_size * 100 / total_size))
                progress_cb(pct)
        urllib.request.urlretrieve(url, str(tmp), reporthook=_reporthook)
        tmp.rename(dest)
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise
    return dest


# ── predictor ──────────────────────────────────────────────────────────────

class MicroSAMPredictor:
    """
    SAM1-based predictor using micro-sam fine-tuned checkpoints.

    Matches the same public API as SAMPredictor so the main window can use
    both interchangeably.

    Parameters
    ----------
    model_type:
        One of the keys in _MODEL_REGISTRY, e.g. "vit_b_lm".
    checkpoint_path:
        Path to a local .pt file.  If None, the checkpoint is downloaded
        automatically to ~/.cache/micro_sam/ on load_model().
    device:
        "cuda", "cpu", or None (auto-detected).
    """

    def __init__(
        self,
        model_type: str = "vit_b_lm",
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        if model_type not in _MODEL_REGISTRY and checkpoint_path is None:
            raise ValueError(
                f"Unknown model_type '{model_type}'. "
                f"Valid options: {list(_MODEL_REGISTRY)}"
            )
        self._model_type  = model_type
        self._checkpoint  = Path(checkpoint_path) if checkpoint_path else None
        self._device      = device
        self._predictor   = None   # segment_anything.SamPredictor
        self._cached_img_hash: Optional[int] = None
        self._cached_content_hash: Optional[str] = None   # md5 of pixel data for disk cache

    # ── public API ────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    @property
    def backend(self) -> str:
        return "usam"

    def load_model(self, progress_cb=None) -> None:
        """Load the model, downloading the checkpoint if needed."""
        import torch
        from segment_anything import sam_model_registry, SamPredictor

        device = self._resolve_device()

        if self._checkpoint is not None and Path(self._checkpoint).exists():
            ckpt_path = self._checkpoint
            arch = _MODEL_REGISTRY.get(self._model_type, (None, "vit_b"))[1]
        else:
            ckpt_path = _download(self._model_type, progress_cb=progress_cb)
            arch = _MODEL_REGISTRY[self._model_type][1]

        sam = sam_model_registry[arch](checkpoint=str(ckpt_path))
        sam.to(device=device)
        self._predictor = SamPredictor(sam)

    def encode_image(self, img8: np.ndarray, progress_cb=None) -> bool:
        """Pre-compute and cache the image embedding for the current image.

        Embeddings are persisted to disk (keyed by image content + model type)
        so reopening the same image in a future session skips the encoder pass.

        Returns True if the embedding was loaded from disk, False if recomputed.
        """
        self._ensure_loaded()
        import torch
        from acorn.core import embedding_cache as ec

        key = id(img8)
        if self._cached_img_hash == key:
            return True

        rgb        = self._to_rgb(img8)
        img_hash   = ec.content_hash(img8)
        disk_path  = ec.cache_path(img_hash, f"usam_{self._model_type}")

        if disk_path.exists():
            payload = ec.load(disk_path)
            if payload is not None:
                try:
                    self._restore_from_cache(payload)
                    self._cached_img_hash    = key
                    self._cached_content_hash = img_hash
                    return True
                except Exception:
                    pass

        if progress_cb:
            progress_cb(0)
        with torch.inference_mode():
            self._predictor.set_image(rgb)
        if progress_cb:
            progress_cb(100)

        self._cached_img_hash    = key
        self._cached_content_hash = img_hash

        try:
            ec.save(disk_path, self._build_cache_payload())
        except Exception:
            pass
        return False

    def invalidate_cache(self) -> None:
        self._cached_img_hash    = None
        self._cached_content_hash = None

    def _build_cache_payload(self) -> dict:
        p = self._predictor
        return {
            "backend":       "usam",
            "features":      p.features.cpu(),
            "original_size": list(p.original_size),
            "input_size":    list(p.input_size),
        }

    def _restore_from_cache(self, payload: dict) -> None:
        import torch
        device = next(self._predictor.model.parameters()).device
        self._predictor.features      = payload["features"].to(device)
        self._predictor.original_size = tuple(payload["original_size"])
        self._predictor.input_size    = tuple(payload["input_size"])
        self._predictor.is_image_set  = True

    def predict_points(
        self,
        img8: np.ndarray,
        points: list[tuple[float, float]],
        labels: Optional[list[int]] = None,
        multimask_output: bool = True,
    ) -> list[np.ndarray]:
        """Predict masks from point prompts. Returns masks sorted by score."""
        self._ensure_loaded()
        import torch
        self.encode_image(img8)
        coords = np.array(points, dtype=np.float32)
        lbls   = np.array(labels if labels is not None else [1] * len(points),
                           dtype=np.int32)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords     = coords,
                point_labels     = lbls,
                multimask_output = multimask_output,
            )
        order = np.argsort(scores)[::-1]
        return [masks[i].astype(bool) for i in order]

    def predict_box(
        self,
        img8: np.ndarray,
        box: tuple[float, float, float, float],
    ) -> np.ndarray:
        """Predict mask from bounding box (x0, y0, x1, y1). Returns (H,W) bool mask."""
        self._ensure_loaded()
        import torch
        self.encode_image(img8)
        box_arr = np.array(box, dtype=np.float32)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                box              = box_arr,
                multimask_output = False,
            )
        return masks[np.argmax(scores)].astype(bool)

    def predict_everything(
        self,
        img8: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 200,
    ) -> list[np.ndarray]:
        """Automatic mask generation using a point grid. Returns bool masks."""
        self._ensure_loaded()
        import torch
        from segment_anything import SamAutomaticMaskGenerator
        generator = SamAutomaticMaskGenerator(
            model                  = self._predictor.model,
            points_per_side        = points_per_side,
            pred_iou_thresh        = pred_iou_thresh,
            stability_score_thresh = stability_score_thresh,
            min_mask_region_area   = min_mask_region_area,
        )
        rgb = self._to_rgb(img8)
        with torch.inference_mode():
            results = generator.generate(rgb)
        masks = [r["segmentation"] for r in results]
        masks.sort(key=lambda m: int(m.sum()), reverse=True)
        return masks

    def mask_to_polygon(self, mask: np.ndarray, tolerance: float = 1.0) -> list[tuple]:
        """Convert binary mask to simplified polygon vertices [(x, y), ...]."""
        try:
            from skimage.measure import find_contours, approximate_polygon
            contours = find_contours(mask.astype(np.uint8), 0.5)
            if not contours:
                return []
            contour = max(contours, key=len)
            approx  = approximate_polygon(contour, tolerance=tolerance)
            return [(float(c[1]), float(c[0])) for c in approx]
        except Exception:
            return []

    # ── helpers ───────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _to_rgb(self, img8: np.ndarray) -> np.ndarray:
        if img8.ndim == 2:
            return np.stack([img8, img8, img8], axis=-1)
        if img8.ndim == 3 and img8.shape[2] == 1:
            return np.concatenate([img8, img8, img8], axis=-1)
        return img8

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
