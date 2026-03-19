"""
YOLO object detection and segmentation wrapper for EM images.

Supports any ultralytics YOLO model — standard detection (YOLOv8/11) or
segmentation (YOLO-seg), pre-trained or custom-trained on any EM modality
(cryo-EM, STEM, TEM, EDX, tomography slices, materials science EM, etc.).

Install
-------
pip install ultralytics

Usage
-----
from acorn.core.yolo_predictor import YOLOPredictor

pred = YOLOPredictor()
pred.load_model("yolo11n.pt")           # downloads on first run
pred.load_model("/path/to/custom.pt")   # local checkpoint

detections = pred.detect(img8)          # list of detection dicts
detections = pred.detect_and_segment(img8)  # adds "mask" key for YOLO-seg

# Add results to an AnnotationStore
from acorn.core.yolo_predictor import boxes_to_roi_annotations
boxes_to_roi_annotations(detections, store)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


class YOLOPredictor:
    """
    YOLO detection and segmentation predictor.

    Parameters
    ----------
    model_path  : local .pt checkpoint or ultralytics model name
                  (e.g. "yolo11n.pt", "yolo11n-seg.pt").  May be set later
                  via load_model().
    device      : "cuda", "cpu", "mps", or None (auto-detect).
    conf_thresh : default confidence threshold.
    iou_thresh  : default NMS IoU threshold.
    """

    def __init__(
        self,
        model_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> None:
        self._model_path  = Path(model_path) if model_path else None
        self._device      = device
        self._conf_thresh = conf_thresh
        self._iou_thresh  = iou_thresh
        self._model       = None
        self._is_seg      = False   # True if model has segmentation heads

    # ── public API ────────────────────────────────────────────────────────────

    def _resolve_device(self) -> str:
        if self._device:
            return self._device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    def load_model(self, model_path: Optional[str | Path] = None) -> None:
        """
        Load a YOLO model.

        Parameters
        ----------
        model_path : .pt file path or ultralytics model name such as
                     "yolo11n.pt" (detection) or "yolo11n-seg.pt"
                     (segmentation).  Downloads automatically if not found
                     locally.  If None, uses the path supplied at
                     construction.
        """
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for YOLO inference.\n"
                "Install with:  pip install ultralytics"
            ) from exc

        if model_path is not None:
            self._model_path = Path(model_path)
        if self._model_path is None:
            raise ValueError("No model path provided to load_model().")

        self._model = YOLO(str(self._model_path))

        try:
            self._is_seg = (self._model.task == "segment")
        except Exception:
            self._is_seg = False

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def is_seg(self) -> bool:
        """True if the loaded model produces instance segmentation masks."""
        return self._is_seg

    @property
    def class_names(self) -> dict[int, str]:
        """Class index -> name mapping from the loaded model."""
        if self._model is None:
            return {}
        try:
            return dict(self._model.names)
        except Exception:
            return {}

    def detect(
        self,
        img8: np.ndarray,
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        classes: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Run object detection on an 8-bit image.

        Parameters
        ----------
        img8        : H×W or H×W×C uint8 image
        conf_thresh : confidence threshold (overrides instance default)
        iou_thresh  : NMS IoU threshold (overrides instance default)
        classes     : restrict to these class indices (None = all classes)

        Returns
        -------
        List of detection dicts, each with keys:
            box      : (x0, y0, x1, y1) pixel coordinates
            conf     : float confidence score
            cls      : integer class index
            cls_name : class name string
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        conf   = conf_thresh if conf_thresh is not None else self._conf_thresh
        iou    = iou_thresh  if iou_thresh  is not None else self._iou_thresh
        rgb    = _to_rgb(img8)
        device = self._resolve_device()

        results = self._model.predict(
            rgb, conf=conf, iou=iou, classes=classes,
            device=device, verbose=False,
        )
        return _parse_boxes(results[0], self.class_names)

    def detect_and_segment(
        self,
        img8: np.ndarray,
        conf_thresh: Optional[float] = None,
        iou_thresh: Optional[float] = None,
        classes: Optional[list[int]] = None,
    ) -> list[dict]:
        """
        Run detection with instance segmentation masks (YOLO-seg model).

        Falls back to box-only detection if the model is not a seg model.
        Returns the same dicts as detect() with an additional key:
            mask : (H, W) bool array, or absent if not available
        """
        if not self._is_seg:
            return self.detect(img8, conf_thresh, iou_thresh, classes)

        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        conf   = conf_thresh if conf_thresh is not None else self._conf_thresh
        iou    = iou_thresh  if iou_thresh  is not None else self._iou_thresh
        rgb    = _to_rgb(img8)
        device = self._resolve_device()

        results = self._model.predict(
            rgb, conf=conf, iou=iou, classes=classes,
            device=device, verbose=False,
        )
        return _parse_seg(results[0], self.class_names, img8.shape[:2])


# ── utilities exposed at module level ─────────────────────────────────────────

def boxes_to_roi_annotations(
    detections: list[dict],
    store,
    label: str = "Foreground",
    color: str = "#FFD700",
    as_rectangles: bool = False,
) -> int:
    """
    Add YOLO detection boxes as ROI annotations.

    Parameters
    ----------
    detections    : output of YOLOPredictor.detect() / detect_and_segment()
    store         : AnnotationStore
    label         : annotation label string
    color         : annotation color hex string
    as_rectangles : if True, use RectangleAnnotation instead of 4-vertex ROI

    Returns
    -------
    Number of annotations added.
    """
    from acorn.core.annotations import ROIAnnotation, RectangleAnnotation

    n = 0
    for d in detections:
        x0, y0, x1, y1 = d["box"]
        if as_rectangles:
            store.add(RectangleAnnotation(
                x0=x0, y0=y0, x1=x1, y1=y1,
                color=color, linewidth=1.5,
            ))
        else:
            vertices = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            store.add(ROIAnnotation(
                vertices=vertices, area_nm2=0.0, stats={},
                color=color, linewidth=1.5, label=label,
            ))
        n += 1
    return n


def masks_to_roi_annotations(
    detections: list[dict],
    store,
    label: str = "Foreground",
    color: str = "#FFD700",
    tolerance: float = 1.0,
) -> int:
    """
    Convert YOLO-seg instance masks to polygon ROI annotations.

    Only processes detections that have a "mask" key (from detect_and_segment()).
    Returns number of annotations added.
    """
    from acorn.core.annotations import ROIAnnotation

    try:
        from skimage.measure import find_contours, approximate_polygon
    except ImportError as exc:
        raise ImportError(
            "scikit-image is required for mask-to-polygon conversion."
        ) from exc

    n = 0
    for d in detections:
        mask = d.get("mask")
        if mask is None:
            continue
        contours = find_contours(mask.astype(np.uint8), 0.5)
        if not contours:
            continue
        contour = max(contours, key=len)
        approx  = approximate_polygon(contour, tolerance=tolerance)
        vertices = [(float(c[1]), float(c[0])) for c in approx]
        if len(vertices) < 3:
            continue
        store.add(ROIAnnotation(
            vertices=vertices, area_nm2=0.0, stats={},
            color=color, linewidth=1.5, label=label,
        ))
        n += 1
    return n


# ── private helpers ───────────────────────────────────────────────────────────

def _to_rgb(img8: np.ndarray) -> np.ndarray:
    if img8.ndim == 2:
        return np.stack([img8, img8, img8], axis=-1)
    if img8.ndim == 3 and img8.shape[2] == 1:
        return np.concatenate([img8, img8, img8], axis=-1)
    return img8


def _parse_boxes(result, class_names: dict) -> list[dict]:
    detections = []
    try:
        boxes = result.boxes
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i].cpu().numpy())
            cls  = int(boxes.cls[i].cpu().numpy())
            detections.append({
                "box":      (xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                "conf":     conf,
                "cls":      cls,
                "cls_name": class_names.get(cls, str(cls)),
            })
    except Exception:
        pass
    return detections


def _parse_seg(result, class_names: dict, img_shape: tuple) -> list[dict]:
    detections = _parse_boxes(result, class_names)
    try:
        if result.masks is not None:
            h, w = img_shape
            import cv2
            for i, d in enumerate(detections):
                raw = result.masks.data[i].cpu().numpy().astype(np.float32)
                if raw.shape != (h, w):
                    raw = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
                d["mask"] = raw > 0.5
    except Exception:
        pass
    return detections
