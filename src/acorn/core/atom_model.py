"""Trainable atom-column finder (U-Net heatmap regression).

Leverages the packages ACORN already ships (PyTorch + segmentation-models-pytorch).
Workflow, self-labelled and human-in-the-loop:

  1. Run the classical detector (atom_detect) to seed atom positions, OR use
     atoms the user has corrected in the Annotate tab.
  2. `train_atom_model` turns those positions into a Gaussian **heatmap** target
     and trains a small U-Net (tiled + augmented from as little as one image) to
     regress it. Saves a `.pt` checkpoint.
  3. `detect_atoms_model` runs the trained U-Net over an image (tiled + stitched)
     and peak-finds its heatmap -> atom coordinates, which feed atom_stats.

Correcting the seed labels (fixing dim-region misses) and re-training is what
lifts recall beyond the classical detector.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.feature import peak_local_max
except Exception:
    peak_local_max = None


def make_heatmap(coords: np.ndarray, shape, sigma: float) -> np.ndarray:
    """Gaussian heatmap (float32, peak ~1) with a blob at each atom coordinate."""
    hm = np.zeros(shape, np.float32)
    if len(coords):
        ys = np.clip(np.round(coords[:, 0]).astype(int), 0, shape[0] - 1)
        xs = np.clip(np.round(coords[:, 1]).astype(int), 0, shape[1] - 1)
        hm[ys, xs] = 1.0
    hm = ndi.gaussian_filter(hm, sigma)
    m = hm.max()
    return (hm / m).astype(np.float32) if m > 0 else hm


def _norm01(img):
    img = np.asarray(img, np.float32)
    if img.ndim == 3:
        img = img.mean(-1)
    lo, hi = np.percentile(img, [0.5, 99.5])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1).astype(np.float32)


def _tiles(H, W, tile, stride):
    ys = list(range(0, max(1, H - tile + 1), stride))
    xs = list(range(0, max(1, W - tile + 1), stride))
    if ys[-1] != H - tile:
        ys.append(max(0, H - tile))
    if xs[-1] != W - tile:
        xs.append(max(0, W - tile))
    return [(y, x) for y in ys for x in xs]


def train_atom_model(image, coords, out_path: str, atom_sigma: float = 2.0,
                     tile: int = 256, stride: int = 128, epochs: int = 40,
                     batch: int = 8, lr: float = 1e-3, encoder: str = "resnet34",
                     device: str | None = None, log=print) -> dict:
    """Train a U-Net to regress the atom heatmap. Returns a summary dict; writes
    a checkpoint (state_dict + config) to ``out_path``."""
    import torch
    import segmentation_models_pytorch as smp

    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    img = _norm01(image)
    H, W = img.shape
    hm = make_heatmap(np.asarray(coords, float), (H, W), atom_sigma)

    # tiles + 8-fold dihedral augmentation (plenty of data from one image)
    Xs, Ys = [], []
    for (y, x) in _tiles(H, W, tile, stride):
        it, ht = img[y:y + tile, x:x + tile], hm[y:y + tile, x:x + tile]
        for f in range(4):
            ir, hr = np.rot90(it, f), np.rot90(ht, f)
            Xs.append(ir); Ys.append(hr)
            Xs.append(np.fliplr(ir)); Ys.append(np.fliplr(hr))
    X = torch.from_numpy(np.stack(Xs)[:, None].astype(np.float32))
    Y = torch.from_numpy(np.stack(Ys)[:, None].astype(np.float32))
    log(f"Atom model: {len(X)} tiles ({tile}px), training on {dev}…")

    model = smp.Unet(encoder_name=encoder, encoder_weights=None,
                     in_channels=1, classes=1, activation=None).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = torch.nn.MSELoss()
    n = len(X)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            xb, yb = X[b].to(dev), Y[b].to(dev)
            opt.zero_grad()
            pred = torch.sigmoid(model(xb))
            loss = lossf(pred, yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(b)
        if ep % 2 == 0 or ep == epochs - 1:
            log(f"Atom model: epoch {ep+1}/{epochs}  loss {tot/n:.5f}")

    torch.save({"state_dict": model.state_dict(), "encoder": encoder,
                "tile": tile, "atom_sigma": atom_sigma}, out_path)
    return {"checkpoint": out_path, "n_tiles": len(X), "epochs": epochs,
            "final_loss": tot / n, "device": dev, "n_seed_atoms": int(len(coords))}


def _load(model_path, device):
    import torch
    import segmentation_models_pytorch as smp
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model = smp.Unet(encoder_name=ck.get("encoder", "resnet34"), encoder_weights=None,
                     in_channels=1, classes=1, activation=None).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


def predict_heatmap(image, model_path: str, tile: int = 256, stride: int = 192,
                    device: str | None = None) -> np.ndarray:
    """Run the trained U-Net over an image (tiled + Hann-blended) -> heatmap."""
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = _load(model_path, dev)
    tile = int(ck.get("tile", tile))
    img = _norm01(image); H, W = img.shape
    acc = np.zeros((H, W), np.float32); wsum = np.zeros((H, W), np.float32)
    win = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    with torch.no_grad():
        for (y, x) in _tiles(H, W, tile, stride):
            t = img[y:y + tile, x:x + tile]
            xb = torch.from_numpy(t[None, None].astype(np.float32)).to(dev)
            p = torch.sigmoid(model(xb)).cpu().numpy()[0, 0]
            acc[y:y + tile, x:x + tile] += p * win
            wsum[y:y + tile, x:x + tile] += win
    return acc / np.maximum(wsum, 1e-6)


def detect_atoms_model(image, model_path: str, min_distance_px: float = 6.0,
                       threshold: float = 0.2, tile: int = 256, stride: int = 192,
                       device: str | None = None) -> np.ndarray:
    """Detect atoms with the trained model: predicted heatmap -> peak finding."""
    if peak_local_max is None:
        raise ImportError("scikit-image is required.")
    hm = predict_heatmap(image, model_path, tile=tile, stride=stride, device=device)
    coords = peak_local_max(hm, min_distance=int(round(min_distance_px)),
                            threshold_abs=float(threshold), exclude_border=False)
    return coords.astype(np.float64)
