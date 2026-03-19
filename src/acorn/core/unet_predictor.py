"""
UNet (and UNet-family) segmentation wrapper for EM images.

Supports two backends:
  - segmentation_models_pytorch (smp): UNet, UNet++, FPN, DeepLabV3+, etc.
    pip install segmentation-models-pytorch
  - Raw PyTorch .pt: any serialized model that accepts (B, C, H, W) and
    returns (B, n_classes, H, W) logits.

Large images are handled transparently by tiled inference with configurable
overlap and linear blending of tile predictions.

Suitable for any EM modality: cryo-EM SPA, STEM, TEM, EDX/EELS maps,
tomography slices, materials science (grain boundaries, defects,
nanoparticles, 2D materials), and biological TEM.

Install
-------
pip install segmentation-models-pytorch    # for smp architectures
pip install torch                          # required for both backends

Usage
-----
from acorn.core.unet_predictor import UNetPredictor

pred = UNetPredictor(architecture="Unet", encoder="resnet34",
                     in_channels=1, n_classes=2)
pred.load_model("/path/to/checkpoint.pt")

masks     = pred.predict(img8)            # list of (H, W) bool instance masks
label_map = pred.predict_semantic(img8)   # (H, W) int class map
proba     = pred.predict_proba(img8)      # (n_classes, H, W) float
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional


# Architectures available in segmentation_models_pytorch
SMP_ARCHITECTURES = [
    "Unet", "UnetPlusPlus", "FPN", "PSPNet",
    "DeepLabV3", "DeepLabV3Plus", "PAN", "MAnet",
]

# Common encoder backbones
SMP_ENCODERS = [
    "resnet18", "resnet34", "resnet50",
    "efficientnet-b0", "efficientnet-b3",
    "mobilenet_v2",
    "mit_b0", "mit_b2",
]


class UNetPredictor:
    """
    UNet / UNet-family segmentation predictor.

    Parameters
    ----------
    architecture : smp architecture name ("Unet", "UnetPlusPlus", "FPN",
                   "DeepLabV3Plus", etc.).
    encoder      : smp encoder backbone ("resnet34", "efficientnet-b0", …).
    in_channels  : number of input channels (1 for grayscale, 3 for RGB).
    n_classes    : number of output classes (2 for binary foreground/bg).
    device       : "cuda", "cpu", "mps", or None (auto-detect).
    tile_size    : max side length for tiled inference on large images.
                   None disables tiling (uses whole image at once).
    tile_overlap : fractional overlap between adjacent tiles [0, 1).
    """

    def __init__(
        self,
        architecture: str = "Unet",
        encoder: str = "resnet34",
        in_channels: int = 1,
        n_classes: int = 2,
        device: Optional[str] = None,
        tile_size: Optional[int] = 512,
        tile_overlap: float = 0.25,
    ) -> None:
        self._architecture = architecture
        self._encoder      = encoder
        self._in_channels  = in_channels
        self._n_classes    = n_classes
        self._device_str   = device
        self._tile_size    = tile_size
        self._tile_overlap = tile_overlap
        self._model        = None
        self._device       = None   # resolved torch.device

    # ── public API ────────────────────────────────────────────────────────────

    def load_model(
        self,
        checkpoint_path: Optional[str | Path] = None,
        architecture: Optional[str] = None,
        encoder: Optional[str] = None,
        in_channels: Optional[int] = None,
        n_classes: Optional[int] = None,
    ) -> None:
        """
        Load a segmentation model from a checkpoint file.

        Tries to load as an smp architecture first (state dict), then falls
        back to torch.load() for scripted/traced modules.  If no checkpoint
        is provided a randomly-initialized smp model is built (useful for
        testing the pipeline).

        Parameters
        ----------
        checkpoint_path : .pt file containing model weights or a full
                          serialized module.  None builds a fresh model.
        architecture    : override instance architecture.
        encoder         : override instance encoder.
        in_channels     : override instance in_channels.
        n_classes       : override instance n_classes.
        """
        import torch

        if architecture is not None: self._architecture = architecture
        if encoder      is not None: self._encoder      = encoder
        if in_channels  is not None: self._in_channels  = in_channels
        if n_classes    is not None: self._n_classes    = n_classes

        self._device = torch.device(
            self._device_str or (
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        )

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)
            state = torch.load(str(checkpoint_path), map_location=self._device,
                                weights_only=False)
            if isinstance(state, dict):
                # Unwrap common checkpoint wrappers
                for key in ("model_state_dict", "state_dict", "model"):
                    if key in state:
                        state = state[key]
                        break
            if isinstance(state, dict):
                # Assume state dict — build smp model and load weights
                self._model = self._build_smp_model()
                self._model.load_state_dict(state, strict=True)
            else:
                # Full serialized module (scripted / whole-model save)
                self._model = state
        else:
            self._model = self._build_smp_model()

        self._model.to(self._device)
        self._model.eval()

    def _build_smp_model(self):
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "segmentation_models_pytorch is required for UNet architectures.\n"
                "Install with:  pip install segmentation-models-pytorch"
            ) from exc
        arch_cls = getattr(smp, self._architecture)
        return arch_cls(
            encoder_name    = self._encoder,
            encoder_weights = None,
            in_channels     = self._in_channels,
            classes         = self._n_classes,
        )

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ── inference ─────────────────────────────────────────────────────────────

    def predict(
        self,
        img8: np.ndarray,
        threshold: float = 0.5,
        foreground_class: int = 1,
        min_area: int = 50,
    ) -> list[np.ndarray]:
        """
        Predict instance masks via semantic segmentation + connected components.

        Parameters
        ----------
        img8             : H×W or H×W×C uint8 image
        threshold        : foreground probability threshold
        foreground_class : which class index to treat as foreground (ignored
                           for binary single-channel output)
        min_area         : discard connected components smaller than this (px)

        Returns
        -------
        List of (H, W) bool masks sorted by descending area.
        """
        prob = self._infer(img8)
        if prob.shape[0] == 1:
            fg = prob[0] > threshold
        else:
            fg = prob[foreground_class] > threshold
        return _connected_components_to_masks(fg, min_area=min_area)

    def predict_semantic(self, img8: np.ndarray) -> np.ndarray:
        """Return a (H, W) int class map (argmax over class dimension)."""
        return np.argmax(self._infer(img8), axis=0).astype(np.int32)

    def predict_proba(self, img8: np.ndarray) -> np.ndarray:
        """Return (n_classes, H, W) float32 probability map."""
        return self._infer(img8)

    def mask_to_polygon(
        self, mask: np.ndarray, tolerance: float = 1.0,
    ) -> list[tuple]:
        """
        Convert a binary mask to a simplified polygon.
        Returns list of (x, y) tuples.
        """
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

    # ── internal ──────────────────────────────────────────────────────────────

    def _infer(self, img8: np.ndarray) -> np.ndarray:
        """Run forward pass, returning (n_classes, H, W) float32 probabilities."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        h, w = img8.shape[:2]
        if self._tile_size is not None and (h > self._tile_size or w > self._tile_size):
            return self._tiled_infer(img8)
        return self._single_infer(img8)

    def _to_tensor(self, img8: np.ndarray):
        import torch
        arr = img8.astype(np.float32) / 255.0
        if arr.ndim == 2:
            if self._in_channels == 1:
                arr = arr[np.newaxis, np.newaxis]       # (1,1,H,W)
            else:
                arr = np.stack([arr] * self._in_channels, axis=0)[np.newaxis]
        else:
            arr = arr.transpose(2, 0, 1)[np.newaxis]   # (1,C,H,W)
            c = arr.shape[1]
            if c < self._in_channels:
                repeats = self._in_channels // c + 1
                arr = np.concatenate([arr] * repeats, axis=1)[:, :self._in_channels]
            elif c > self._in_channels:
                arr = arr[:, :self._in_channels]
        return torch.from_numpy(arr).to(self._device)

    def _single_infer(self, img8: np.ndarray) -> np.ndarray:
        import torch
        from torch.nn import functional as F

        tensor = self._to_tensor(img8)
        h, w = img8.shape[:2]

        # Pad to multiple of 32 (required by most CNN encoders)
        pad_h = (32 - h % 32) % 32
        pad_w = (32 - w % 32) % 32
        if pad_h or pad_w:
            tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.inference_mode():
            logits = self._model(tensor)

        out = logits[0, :, :h, :w].float().cpu().numpy()
        n_cls = out.shape[0]

        # Apply activation to convert logits to probabilities
        if n_cls == 1:
            from scipy.special import expit
            out = expit(out)
        else:
            from scipy.special import softmax
            out = softmax(out, axis=0)

        return out.astype(np.float32)

    def _tiled_infer(self, img8: np.ndarray) -> np.ndarray:
        """Tiled inference with linear-weight blending over tile overlaps."""
        h, w   = img8.shape[:2]
        ts     = self._tile_size
        step   = max(1, int(ts * (1.0 - self._tile_overlap)))
        n_cls  = max(self._n_classes, 1)
        acc    = np.zeros((n_cls, h, w), dtype=np.float64)
        weight = np.zeros((h, w), dtype=np.float64)

        ys = list(range(0, h - ts, step)) + [max(0, h - ts)]
        xs = list(range(0, w - ts, step)) + [max(0, w - ts)]

        for y0 in ys:
            for x0 in xs:
                y1  = min(y0 + ts, h)
                x1  = min(x0 + ts, w)
                tile = img8[y0:y1, x0:x1]
                pred = self._single_infer(tile)    # (n_cls, th, tw)
                th, tw = pred.shape[1], pred.shape[2]
                acc[:, y0:y0+th, x0:x0+tw] += pred
                weight[y0:y0+th, x0:x0+tw] += 1.0

        weight = np.maximum(weight, 1e-8)
        return (acc / weight[np.newaxis]).astype(np.float32)


# ── helpers ───────────────────────────────────────────────────────────────────

def _connected_components_to_masks(
    fg: np.ndarray,
    min_area: int = 50,
) -> list[np.ndarray]:
    """Split a binary foreground map into per-instance bool masks."""
    from scipy.ndimage import label as scipy_label
    labeled, n = scipy_label(fg)
    masks = []
    for i in range(1, n + 1):
        m = labeled == i
        if int(m.sum()) >= min_area:
            masks.append(m.astype(bool))
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    return masks
