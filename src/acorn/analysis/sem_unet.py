"""
Lightweight U-Net that predicts residual height corrections on top of
physics-based shape-from-shading.

Input:  2-channel image  [I_normalised, h_physics_normalised]  (B, 2, H, W)
Output: 1-channel delta_h correction                           (B, 1, H, W)

The network is fully convolutional and handles arbitrary spatial dimensions.
~1.5 M parameters — trains in minutes on 2 000 synthetic examples.

Training data is generated entirely synthetically via SyntheticSEMDataset,
which renders random height fields (spheres, ellipsoids, rough surfaces)
through the physics forward model and uses the result of a fast
shape-from-shading pass as the h_physics channel, so the network learns
to correct SFS artefacts rather than just denoise.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# U-Net building blocks
# ---------------------------------------------------------------------------

def _double_conv(in_ch: int, out_ch: int):
    """Two (Conv 3x3 + BN + ReLU) blocks."""
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(in_ch,  out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SEMHeightUNet:
    """
    Residual U-Net for SEM height correction.

    Wrap in a try/except at import time so that the module can be imported
    even when PyTorch is not installed — only inference/training need it.
    """

    def __new__(cls, max_correction: float = 0.3):
        try:
            import torch.nn as nn
        except ImportError as exc:
            raise ImportError("PyTorch required: pip install torch") from exc
        return _SEMHeightUNetImpl(max_correction)


class _SEMHeightUNetImpl:
    """Actual torch.nn.Module — instantiated only when torch is available."""

    def __new__(cls, max_correction: float):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, max_correction: float = 0.3) -> None:
                super().__init__()
                self.max_correction = max_correction

                self.enc1 = _double_conv(2, 32)
                self.pool1 = nn.MaxPool2d(2)
                self.enc2 = _double_conv(32, 64)
                self.pool2 = nn.MaxPool2d(2)
                self.enc3 = _double_conv(64, 128)
                self.pool3 = nn.MaxPool2d(2)

                self.bottleneck = _double_conv(128, 256)

                self.up3   = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self.dec3  = _double_conv(256, 128)
                self.up2   = nn.ConvTranspose2d(128, 64,  2, stride=2)
                self.dec2  = _double_conv(128, 64)
                self.up1   = nn.ConvTranspose2d(64,  32,  2, stride=2)
                self.dec1  = _double_conv(64,  32)

                self.head  = nn.Conv2d(32, 1, 1)

            def forward(self, x):
                import torch
                import torch.nn.functional as F
                s1 = self.enc1(x)
                s2 = self.enc2(self.pool1(s1))
                s3 = self.enc3(self.pool2(s2))
                b  = self.bottleneck(self.pool3(s3))

                d3 = self.dec3(torch.cat([self.up3(b),  s3], dim=1))
                d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
                d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

                delta = torch.tanh(self.head(d1)) * self.max_correction
                return delta

        return _Net(max_correction)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticSEMDataset:
    """
    Generates (image_2ch, delta_h) pairs from random synthetic surfaces.

    Height field types (randomly sampled):
      - sphere       : h = sqrt(max(0, R^2 - r^2))
      - ellipsoid    : h = c * sqrt(max(0, 1 - x^2/a^2 - y^2/b^2))
      - rough sphere : sphere + band-limited Gaussian noise
      - hollow       : outer sphere shell with hollowed-out centre

    Each sample is rendered through SEMForwardModel, Poisson noise is
    added, and a fast SFS pass (n_iters=100) is run to get h_physics.
    The U-Net target is delta_h = h_true - h_physics.
    """

    def __new__(
        cls,
        n_samples: int = 2000,
        image_size: int = 128,
        detector_params=None,
        noise_level: float = 0.05,
    ):
        try:
            import torch
            import torch.utils.data as tdata
        except ImportError as exc:
            raise ImportError("PyTorch required: pip install torch") from exc
        return _SyntheticSEMDatasetImpl(
            n_samples, image_size, detector_params, noise_level
        )


class _SyntheticSEMDatasetImpl:
    def __new__(cls, n_samples, image_size, detector_params, noise_level):
        import torch
        import torch.utils.data as tdata
        from acorn.analysis.sem_physics import (
            DetectorParams, render, shape_from_shading,
        )

        class _DS(tdata.Dataset):
            def __init__(self) -> None:
                self._n      = n_samples
                self._sz     = image_size
                self._params = detector_params or DetectorParams()
                self._noise  = noise_level
                self._rng    = np.random.default_rng(42)

            def __len__(self) -> int:
                return self._n

            def __getitem__(self, idx: int) -> dict:
                rng = np.random.default_rng(idx)
                sz  = self._sz
                h_true = self._gen_height(rng, sz)
                mask   = (h_true > 0).astype(np.float32)

                I_clean = render(h_true, self._params)
                # Normalise then add Poisson-like noise
                I_norm  = (I_clean - I_clean.min()) / (I_clean.max() - I_clean.min() + 1e-8)
                noise   = rng.normal(0, self._noise, I_norm.shape).astype(np.float32)
                I_noisy = np.clip(I_norm + noise, 0, None).astype(np.float32)

                # Fast physics-only SFS for the h_physics channel
                h_phys = shape_from_shading(
                    I_noisy * (self._params.eta0 + self._params.I_bg),
                    mask,
                    self._params,
                    n_iters=100,
                    lr=8e-3,
                    smoothness_weight=0.1,
                )

                # Normalise both channels to [0, 1] within mask region
                mx = float(h_true.max()) + 1e-8
                I_ch   = torch.from_numpy(I_noisy).unsqueeze(0)
                h_phys_norm = torch.from_numpy(
                    (h_phys / mx).astype(np.float32)
                ).unsqueeze(0)
                img_2ch = torch.cat([I_ch, h_phys_norm], dim=0)

                delta_h = torch.from_numpy(
                    ((h_true - h_phys) / mx).astype(np.float32)
                ).unsqueeze(0)

                return {"image": img_2ch, "target": delta_h,
                        "h_true": torch.from_numpy(h_true).unsqueeze(0),
                        "h_physics": torch.from_numpy(h_phys).unsqueeze(0)}

            def _gen_height(self, rng: np.random.Generator, sz: int) -> np.ndarray:
                cx = rng.uniform(sz * 0.35, sz * 0.65)
                cy = rng.uniform(sz * 0.35, sz * 0.65)
                shape_type = rng.integers(0, 4)

                xs = np.arange(sz, dtype=np.float32)
                ys = np.arange(sz, dtype=np.float32)
                X, Y = np.meshgrid(xs, ys)
                dx = X - cx
                dy = Y - cy

                if shape_type == 0:   # sphere
                    R = rng.uniform(sz * 0.15, sz * 0.35)
                    r2 = dx ** 2 + dy ** 2
                    h = np.sqrt(np.maximum(0.0, R ** 2 - r2)).astype(np.float32)

                elif shape_type == 1:   # ellipsoid
                    a = rng.uniform(sz * 0.15, sz * 0.35)
                    b = rng.uniform(sz * 0.15, sz * 0.35)
                    c = rng.uniform(sz * 0.10, sz * 0.30)
                    inner = 1.0 - (dx / a) ** 2 - (dy / b) ** 2
                    h = (c * np.sqrt(np.maximum(0.0, inner))).astype(np.float32)

                elif shape_type == 2:   # rough sphere
                    R  = rng.uniform(sz * 0.15, sz * 0.35)
                    r2 = dx ** 2 + dy ** 2
                    h  = np.sqrt(np.maximum(0.0, R ** 2 - r2)).astype(np.float32)
                    # Band-limited roughness
                    freq_cutoff = rng.uniform(0.05, 0.15)
                    rough = rng.standard_normal((sz, sz)).astype(np.float32)
                    from scipy.ndimage import gaussian_filter
                    rough = gaussian_filter(rough, sigma=1.0 / (2 * math.pi * freq_cutoff + 1e-8))
                    rough *= rng.uniform(0.02, 0.08) * R
                    h = np.maximum(0.0, h + rough * (h > 0)).astype(np.float32)

                else:   # hollow sphere shell
                    R_out = rng.uniform(sz * 0.20, sz * 0.35)
                    R_in  = R_out * rng.uniform(0.5, 0.75)
                    r2    = dx ** 2 + dy ** 2
                    h_out = np.sqrt(np.maximum(0.0, R_out ** 2 - r2))
                    h_in  = np.sqrt(np.maximum(0.0, R_in  ** 2 - r2))
                    h = (h_out - h_in).clip(min=0.0).astype(np.float32)

                return h

        return _DS()


# ---------------------------------------------------------------------------
# Training entry-point
# ---------------------------------------------------------------------------

def train_sem_unet(
    output_dir: str | Path,
    *,
    n_samples: int = 2000,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 3e-4,
    image_size: int = 128,
    detector_params=None,
    device: str = "auto",
    log_cb: Callable[[str], None] | None = None,
    progress_cb: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Train the SEM U-Net on synthetic data and save best_weights.pt.
    Returns the path to the saved checkpoint.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    log = log_cb or print
    dev = torch.device(
        device if device != "auto"
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    log(f"Training SEM U-Net on {dev}  ({n_samples} synthetic samples, {epochs} epochs)")

    from acorn.analysis.sem_physics import DetectorParams
    params = detector_params or DetectorParams()

    dataset = SyntheticSEMDataset(
        n_samples=n_samples,
        image_size=image_size,
        detector_params=params,
    )

    n_val = max(1, int(n_samples * 0.1))
    n_train = n_samples - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(0))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=(dev.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=0)

    model = SEMHeightUNet(max_correction=0.5).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    best_ckpt = out / "best_weights.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            img   = batch["image"].to(dev)
            tgt   = batch["target"].to(dev)
            optimizer.zero_grad()
            pred  = model(img)
            loss  = nn.functional.mse_loss(pred, tgt)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * img.size(0)

        train_loss /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                img  = batch["image"].to(dev)
                tgt  = batch["target"].to(dev)
                pred = model(img)
                val_loss += nn.functional.mse_loss(pred, tgt).item() * img.size(0)
        val_loss /= n_val

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt)

        log(f"Epoch {epoch}/{epochs}  train_loss={train_loss:.5f}  val_loss={val_loss:.5f}")
        if progress_cb:
            progress_cb(epoch, epochs)

    log(f"Training complete.  Best val_loss={best_val_loss:.5f}")
    log(f"Checkpoint saved: {best_ckpt}")
    return best_ckpt


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def apply_nn_correction(
    I_crop: np.ndarray,
    h_physics: np.ndarray,
    mask: np.ndarray,
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> np.ndarray:
    """
    Load checkpoint and apply U-Net residual correction.
    Returns h_refined = h_physics + delta_h (float32 numpy, same shape as h_physics).
    """
    import torch

    dev = torch.device(device)
    model = SEMHeightUNet(max_correction=0.5).to(dev)
    state = torch.load(str(checkpoint_path), map_location=dev)
    model.load_state_dict(state)
    model.eval()

    # Normalise inputs within mask
    mx = float(h_physics.max()) + 1e-8
    I_norm = (I_crop.astype(np.float32) - I_crop.min()) / (I_crop.max() - I_crop.min() + 1e-8)
    h_norm = (h_physics.astype(np.float32) / mx).clip(0, 1)

    inp = np.stack([I_norm, h_norm], axis=0)[None]   # (1, 2, H, W)
    inp_t = torch.from_numpy(inp).to(dev)

    with torch.no_grad():
        delta = model(inp_t).squeeze().cpu().numpy()

    h_refined = (h_physics + delta * mx * mask).astype(np.float32)
    h_refined = np.maximum(h_refined, 0.0)
    return h_refined
