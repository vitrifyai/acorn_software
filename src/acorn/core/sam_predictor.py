"""
SAM 3 (and SAM 2 fallback) inference wrapper for cryo-EM images.

SAM 3 API (correct usage)
--------------------------
  processor = Sam3Processor(model)
  state     = processor.set_image(rgb_hwc_uint8)
  masks, scores, _ = model.predict_inst(state, point_coords=..., point_labels=...)

SAM 2 API
---------
  predictor = SAM2ImagePredictor(model)
  predictor.set_image(rgb_hwc_uint8)
  masks, scores, _ = predictor.predict(point_coords=..., point_labels=...)

Install
-------
pip install sam3          # SAM 3 (recommended)
pip install sam2          # SAM 2 fallback

Usage
-----
from acorn.core.sam_predictor import SAMPredictor

predictor = SAMPredictor()
predictor.load_model()

masks = predictor.predict_everything(img8)
masks = predictor.predict_points(img8, [(x, y)])
mask  = predictor.predict_box(img8, (x0, y0, x1, y1))
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


_DEFAULT_SAM2_HF_REPO = "facebook/sam2-hiera-large"
_DEFAULT_SAM2_CFG     = "sam2_hiera_large.yaml"


class SAMPredictor:
    """
    SAM 3 / SAM 2 image predictor with automatic backend detection.

    Parameters
    ----------
    checkpoint_path : local .pt checkpoint file.  If None the model is
                      downloaded from HuggingFace Hub on load_model().
    model_cfg       : SAM 2 config name — ignored for SAM 3.
    device          : "cuda", "cpu", or "mps".  Auto-detected if None.
    backend         : "sam3" | "sam2" | "auto" (default).
    """

    def __init__(
        self,
        checkpoint_path: Optional[str | Path] = None,
        model_cfg: str = _DEFAULT_SAM2_CFG,
        device: Optional[str] = None,
        backend: str = "auto",
    ) -> None:
        self._checkpoint = Path(checkpoint_path) if checkpoint_path else None
        self._model_cfg  = model_cfg
        self._device     = device
        self._backend    = backend

        # SAM 3 state
        self._sam3_model     = None   # Sam3Image
        self._sam3_processor = None   # Sam3Processor

        # SAM 2 state
        self._predictor = None        # SAM2ImagePredictor

        self._active_backend: Optional[str] = None

        # Cached image embedding — set once per image, reused for all prompts
        self._cached_state    = None   # SAM 3: processor state
        self._cached_img_hash: Optional[int] = None   # id() of the last encoded array
        self._cached_content_hash: Optional[str] = None  # content md5 for disk cache

    # ── public API ────────────────────────────────────────────────────────────

    def load_model(self) -> None:
        """Load (or re-load) the segmentation model."""
        import torch
        device = self._device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

        if self._backend in ("sam3", "auto"):
            try:
                self._load_sam3(device)
                self._active_backend = "sam3"
                return
            except ImportError:
                if self._backend == "sam3":
                    raise

        self._load_sam2(device)
        self._active_backend = "sam2"

    def _load_sam3(self, device: str) -> None:
        from sam3 import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        if self._checkpoint is not None and self._checkpoint.exists():
            model = build_sam3_image_model(
                checkpoint_path=str(self._checkpoint),
                device=device,
                enable_inst_interactivity=True,
                load_from_HF=False,
            )
        else:
            model = build_sam3_image_model(
                device=device,
                enable_inst_interactivity=True,
                load_from_HF=True,
            )

        self._sam3_model     = model
        self._sam3_processor = Sam3Processor(model, device=device)

    def _load_sam2(self, device: str) -> None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if self._checkpoint is not None and self._checkpoint.exists():
            from sam2.build_sam import build_sam2
            model = build_sam2(self._model_cfg, str(self._checkpoint), device=device)
            self._predictor = SAM2ImagePredictor(model)
        else:
            try:
                self._predictor = SAM2ImagePredictor.from_pretrained(
                    _DEFAULT_SAM2_HF_REPO, device=device
                )
            except AttributeError:
                from huggingface_hub import hf_hub_download
                from sam2.build_sam import build_sam2
                ckpt = hf_hub_download(
                    repo_id  = _DEFAULT_SAM2_HF_REPO,
                    filename = "sam2_hiera_large.pt",
                )
                model = build_sam2(self._model_cfg, ckpt, device=device)
                self._predictor = SAM2ImagePredictor(model)

    @property
    def is_loaded(self) -> bool:
        if self._active_backend == "sam3":
            return self._sam3_model is not None
        return self._predictor is not None

    @property
    def backend(self) -> Optional[str]:
        return self._active_backend

    def _to_pil(self, rgb: np.ndarray):
        """Convert HWC uint8 numpy array to PIL Image (required by Sam3Processor)."""
        from PIL import Image
        return Image.fromarray(rgb.astype(np.uint8))

    def _to_rgb(self, img8: np.ndarray) -> np.ndarray:
        if img8.ndim == 2:
            return np.stack([img8, img8, img8], axis=-1)
        if img8.ndim == 3 and img8.shape[2] == 1:
            return np.concatenate([img8, img8, img8], axis=-1)
        return img8

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

    # ── image embedding cache ─────────────────────────────────────────────────

    def encode_image(self, img8: np.ndarray) -> bool:
        """Pre-compute and cache the image embedding.

        Embeddings are persisted to disk so reopening the same image in a
        future session skips the expensive ViT encoder pass.

        Returns True if loaded from disk cache, False if recomputed.
        """
        self._ensure_loaded()
        from acorn.core import embedding_cache as ec

        key = id(img8)
        if self._cached_img_hash == key and self._cached_state is not None:
            return True  # already in memory

        rgb          = self._to_rgb(img8)
        img_hash     = ec.content_hash(img8)
        model_id     = f"{self._active_backend}_{self._model_cfg}"
        disk_path    = ec.cache_path(img_hash, model_id)

        # ── try disk cache ────────────────────────────────────────────────
        if disk_path.exists():
            payload = ec.load(disk_path)
            if payload is not None:
                try:
                    self._restore_from_cache(payload)
                    self._cached_img_hash    = key
                    self._cached_content_hash = img_hash
                    return True
                except Exception:
                    pass  # corrupt/stale — fall through to recompute

        # ── compute ───────────────────────────────────────────────────────
        import torch
        if self._active_backend == "sam3":
            state = self._sam3_processor.set_image(self._to_pil(rgb))
            self._cached_state = state
        else:
            with torch.inference_mode():
                self._predictor.set_image(rgb)
            self._cached_state = True

        self._cached_img_hash    = key
        self._cached_content_hash = img_hash

        # ── persist ───────────────────────────────────────────────────────
        try:
            ec.save(disk_path, self._build_cache_payload())
        except Exception:
            pass  # non-fatal

        return False

    def _build_cache_payload(self) -> dict:
        """Serialise current embedding state to a torch-saveable dict."""
        import torch
        if self._active_backend == "sam3":
            # SAM3 state is a dict of tensors — save as-is
            state = self._cached_state
            if isinstance(state, dict):
                return {"backend": "sam3", "state": state}
            # Fallback: try torch.save directly
            return {"backend": "sam3", "state": state}
        else:
            # SAM2: save _features dict + _orig_hw
            p = self._predictor
            feats = {k: v.cpu() if hasattr(v, "cpu") else
                     [t.cpu() for t in v]
                     for k, v in p._features.items()}
            return {
                "backend":  "sam2",
                "features": feats,
                "orig_hw":  p._orig_hw,
            }

    def _restore_from_cache(self, payload: dict) -> None:
        """Restore predictor state from a cached payload dict."""
        import torch
        backend = payload.get("backend")
        if backend == "sam3":
            state = payload["state"]
            if isinstance(state, dict):
                device = next(self._sam3_model.parameters()).device
                state  = {k: v.to(device) if hasattr(v, "to") else v
                          for k, v in state.items()}
            self._cached_state = state
        else:
            # SAM2
            p      = self._predictor
            device = next(p.model.parameters()).device
            feats  = payload["features"]
            p._features = {
                k: v.to(device) if hasattr(v, "to") else
                   [t.to(device) for t in v]
                for k, v in feats.items()
            }
            p._orig_hw       = payload["orig_hw"]
            p._is_image_set  = True
            self._cached_state = True

    def invalidate_cache(self) -> None:
        """Discard cached embedding (call on image switch)."""
        self._cached_state    = None
        self._cached_img_hash = None
        self._cached_content_hash = None

    def _get_or_encode(self, img8: np.ndarray):
        """Return SAM3 state (or True for SAM2), encoding if not already cached."""
        if self._cached_img_hash != id(img8) or self._cached_state is None:
            self.encode_image(img8)
        return self._cached_state

    # ── inference ─────────────────────────────────────────────────────────────

    def predict_everything(
        self,
        img8: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 200,
    ) -> list[np.ndarray]:
        """Grid-point automatic segmentation. Returns list of (H,W) bool masks."""
        self._ensure_loaded()
        if self._active_backend == "sam2":
            return self._auto_sam2(
                img8, points_per_side, pred_iou_thresh,
                stability_score_thresh, min_mask_region_area,
            )
        return self._auto_grid_sam3(
            img8, points_per_side, pred_iou_thresh,
            stability_score_thresh, min_mask_region_area,
        )

    def _auto_sam2(
        self, img8, points_per_side, pred_iou_thresh,
        stability_score_thresh, min_mask_region_area,
    ) -> list[np.ndarray]:
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        generator = SAM2AutomaticMaskGenerator(
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

    def _auto_grid_sam3(
        self, img8, points_per_side, pred_iou_thresh,
        stability_score_thresh, min_mask_region_area,
    ) -> list[np.ndarray]:
        h, w = img8.shape[:2]
        rgb   = self._to_rgb(img8)
        state = self._sam3_processor.set_image(self._to_pil(rgb))

        xs = np.linspace(0, w - 1, points_per_side, dtype=np.float32)
        ys = np.linspace(0, h - 1, points_per_side, dtype=np.float32)

        collected_masks: list[np.ndarray] = []
        collected_scores: list[float]     = []

        import torch
        device = next(self._sam3_model.parameters()).device
        for px, py in [(float(x), float(y)) for y in ys for x in xs]:
            try:
                masks, scores, _ = self._sam3_model.predict_inst(
                    state,
                    point_coords     = torch.as_tensor([[px, py]], dtype=torch.float32, device=device),
                    point_labels     = torch.as_tensor([1],        dtype=torch.int32,   device=device),
                    multimask_output = True,
                )
                for m, s in zip(masks, scores):
                    m_bool = m.astype(bool)
                    if s < pred_iou_thresh or int(m_bool.sum()) < min_mask_region_area:
                        continue
                    collected_masks.append(m_bool)
                    collected_scores.append(float(s))
            except Exception:
                continue

        if not collected_masks:
            return []

        keep   = _mask_nms(collected_masks, collected_scores, iou_thresh=0.7)
        result = [collected_masks[i] for i in keep]
        result.sort(key=lambda m: int(m.sum()), reverse=True)
        return result

    def predict_points(
        self,
        img8: np.ndarray,
        points: list[tuple[float, float]],
        labels: Optional[list[int]] = None,
        multimask_output: bool = True,
    ) -> list[np.ndarray]:
        """Predict masks from point prompts. Returns masks sorted by score."""
        self._ensure_loaded()

        coords = np.array(points, dtype=np.float32)
        lbls   = np.array(labels if labels is not None else [1] * len(points),
                           dtype=np.int32)

        state = self._get_or_encode(img8)

        if self._active_backend == "sam3":
            import torch
            device = next(self._sam3_model.parameters()).device
            masks, scores, _ = self._sam3_model.predict_inst(
                state,
                point_coords     = torch.as_tensor(coords, dtype=torch.float32, device=device),
                point_labels     = torch.as_tensor(lbls,   dtype=torch.int32,   device=device),
                multimask_output = multimask_output,
            )
        else:
            import torch
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

        box_arr = np.array(box, dtype=np.float32)
        state = self._get_or_encode(img8)

        if self._active_backend == "sam3":
            import torch
            device = next(self._sam3_model.parameters()).device
            masks, scores, _ = self._sam3_model.predict_inst(
                state,
                box              = torch.as_tensor(box_arr, dtype=torch.float32, device=device),
                multimask_output = False,
            )
        else:
            import torch
            with torch.inference_mode():
                masks, scores, _ = self._predictor.predict(
                    box              = box_arr,
                    multimask_output = False,
                )

        return masks[np.argmax(scores)].astype(bool)

    def mask_to_polygon(self, mask: np.ndarray, tolerance: float = 1.0) -> list[tuple]:
        """Convert binary mask to simplified polygon. Returns list of (x, y) tuples."""
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int((a & b).sum())
    union = int((a | b).sum())
    return inter / union if union > 0 else 0.0


def _mask_nms(
    masks: list[np.ndarray],
    scores: list[float],
    iou_thresh: float = 0.7,
) -> list[int]:
    order      = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    keep       = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        for j in order:
            if j <= i or j in suppressed:
                continue
            if _mask_iou(masks[i], masks[j]) > iou_thresh:
                suppressed.add(j)
    return keep
