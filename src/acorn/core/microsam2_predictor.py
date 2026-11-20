"""micro-SAM 2 image segmentation backend for ACORN.

micro-SAM 2 provides microscopy-tuned SAM2 checkpoints and wraps SAM2's
image-predictor API. Imports are delayed so the viewer remains usable when the
optional ``microsam2`` install extra is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


DEFAULT_MODEL = "hvit_t_cells"

MODEL_TYPES: tuple[tuple[str, str], ...] = (
    ("hvit_t_cells", "hvit_t_cells — microscopy (fastest; LM cells/nuclei, EM neurites)"),
    ("hvit_s_cells", "hvit_s_cells — microscopy (small)"),
    ("hvit_b_cells", "hvit_b_cells — microscopy (base)"),
    ("hvit_l_cells", "hvit_l_cells — microscopy (large)"),
    ("hvit_s_em_mito", "hvit_s_em_mito — EM mitochondria"),
    ("hvit_s_em_nuclei", "hvit_s_em_nuclei — EM nuclei"),
)


def available_models() -> list[tuple[str, str]]:
    """Return registered micro-SAM 2 model IDs and UI labels."""
    return list(MODEL_TYPES)


class MicroSAM2Predictor:
    """Promptable SAM2 predictor backed by micro-SAM's v2 model registry.

    Parameters
    ----------
    model_type:
        A micro-SAM 2 model ID, such as ``hvit_t_cells`` or ``hvit_s_em_mito``.
    checkpoint_path:
        Optional local SAM2-compatible checkpoint. With no path, micro-SAM
        downloads its registered microscopy weights into its model cache.
    device:
        ``cuda``, ``cpu`` or ``mps``. Auto-detected when omitted.
    """

    def __init__(
        self,
        model_type: str = DEFAULT_MODEL,
        checkpoint_path: Optional[str | Path] = None,
        device: Optional[str] = None,
    ) -> None:
        valid = {name for name, _ in MODEL_TYPES}
        if model_type not in valid:
            raise ValueError(f"Unknown micro-SAM 2 model '{model_type}'. Choose from {sorted(valid)}.")
        self._model_type = model_type
        self._checkpoint = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self._device = device
        self._model = None
        self._predictor = None
        self._cached_img_id: Optional[int] = None

    @property
    def backend(self) -> str:
        return "microsam2"

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load_model(self, progress_cb=None) -> None:
        """Load the selected registered model or explicit local checkpoint."""
        from micro_sam.v2.util import get_sam2_image_predictor, get_sam2_model

        if self._checkpoint is not None and not self._checkpoint.is_file():
            raise FileNotFoundError(f"micro-SAM 2 checkpoint does not exist: {self._checkpoint}")
        if progress_cb:
            progress_cb(0)
        model = get_sam2_model(
            model_type=self._model_type,
            device=self._resolve_device(),
            checkpoint_path=str(self._checkpoint) if self._checkpoint else None,
            input_type="images",
        )
        self._model = model
        self._predictor = get_sam2_image_predictor(model)
        if progress_cb:
            progress_cb(100)

    def encode_image(self, img8: np.ndarray, progress_cb=None) -> bool:
        """Encode an image once per in-memory image object; return cache status."""
        self._ensure_loaded()
        if self._cached_img_id == id(img8):
            return True
        import torch

        if progress_cb:
            progress_cb(0)
        with torch.inference_mode():
            self._predictor.set_image(self._to_rgb(img8))
        self._cached_img_id = id(img8)
        if progress_cb:
            progress_cb(100)
        return False

    def invalidate_cache(self) -> None:
        """Forget the current image embedding."""
        self._cached_img_id = None
        if self._predictor is not None:
            reset = getattr(self._predictor, "reset_predictor", None)
            if reset is not None:
                reset()

    def predict_points(
        self,
        img8: np.ndarray,
        points: list[tuple[float, float]],
        labels: Optional[list[int]] = None,
        multimask_output: bool = True,
    ) -> list[np.ndarray]:
        """Predict masks from ``(x, y)`` points, sorted by confidence."""
        self._ensure_loaded()
        import torch

        self.encode_image(img8)
        coords = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        point_labels = np.asarray(labels if labels is not None else [1] * len(points), dtype=np.int32)
        if len(coords) != len(point_labels):
            raise ValueError("Each micro-SAM 2 point must have one prompt label.")
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
        return [np.asarray(masks[i]).astype(bool) for i in np.argsort(scores)[::-1]]

    def predict_box(self, img8: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
        """Predict one binary mask from an ``(x0, y0, x1, y1)`` box."""
        self._ensure_loaded()
        import torch

        self.encode_image(img8)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                box=np.asarray(box, dtype=np.float32), multimask_output=False
            )
        return np.asarray(masks[int(np.argmax(scores))]).astype(bool)

    def predict_everything(
        self,
        img8: np.ndarray,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 200,
    ) -> list[np.ndarray]:
        """Run SAM2 automatic mask generation and return exact boolean masks."""
        self._ensure_loaded()
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

        generator = SAM2AutomaticMaskGenerator(
            model=self._model,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area,
        )
        with torch.inference_mode():
            results = generator.generate(self._to_rgb(img8))
        masks = [np.asarray(result["segmentation"]).astype(bool) for result in results]
        masks.sort(key=lambda mask: int(mask.sum()), reverse=True)
        return masks

    def mask_to_polygon(self, mask: np.ndarray, tolerance: float = 1.0) -> list[tuple[float, float]]:
        """Convert a binary mask to simplified ``(x, y)`` polygon vertices."""
        try:
            from skimage.measure import approximate_polygon, find_contours

            contours = find_contours(np.asarray(mask, dtype=np.uint8), 0.5)
            if not contours:
                return []
            contour = max(contours, key=len)
            approx = approximate_polygon(contour, tolerance=tolerance)
            return [(float(column), float(row)) for row, column in approx]
        except Exception:
            return []

    @staticmethod
    def _resolve_device() -> str:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def _to_rgb(img8: np.ndarray) -> np.ndarray:
        image = np.asarray(img8)
        if image.ndim == 2:
            return np.repeat(image[..., None], 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] == 1:
            return np.repeat(image, 3, axis=-1)
        if image.ndim == 3 and image.shape[-1] in (3, 4):
            return image[..., :3]
        raise ValueError(f"micro-SAM 2 expects a 2D grayscale or RGB image, got {image.shape}.")

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("micro-SAM 2 is not loaded. Click Load Model first.")
