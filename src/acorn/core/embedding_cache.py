"""
Disk-persistent image embedding cache shared by all SAM backends.

Embeddings are stored as torch .pt files keyed by a content hash of the
image + a model identifier string.  Loading from cache skips the expensive
ViT encoder pass (typically 2–8 s on a cryo-EM image).

Cache location: ~/.cache/acorn/embeddings/  (or $ACORN_EMBEDDING_CACHE)
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Optional

import numpy as np


def cache_dir() -> Path:
    env = os.environ.get("ACORN_EMBEDDING_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "acorn" / "embeddings"


def content_hash(img8: np.ndarray) -> str:
    """Fast content fingerprint — stride-sampled md5 of pixel data."""
    flat   = img8.ravel()
    stride = max(1, len(flat) // 65536)
    return hashlib.md5(flat[::stride].tobytes()).hexdigest()


def cache_path(img_hash: str, model_id: str) -> Path:
    """Return the .pt path for a given image hash + model identifier."""
    safe = model_id.replace("/", "_").replace(" ", "_")
    return cache_dir() / f"{img_hash}_{safe}.pt"


def save(path: Path, payload: dict) -> None:
    """Save a dict of tensors/arrays to disk (torch.save)."""
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        torch.save(payload, str(tmp))
        tmp.rename(path)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def load(path: Path) -> Optional[dict]:
    """Load cached payload; returns None on any error."""
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


def cache_size_mb() -> float:
    """Total size of all cached embeddings in MB."""
    d = cache_dir()
    if not d.exists():
        return 0.0
    return sum(p.stat().st_size for p in d.glob("*.pt")) / 1e6


def clear_cache() -> int:
    """Delete all cached embeddings. Returns number of files removed."""
    d = cache_dir()
    if not d.exists():
        return 0
    removed = 0
    for p in d.glob("*.pt"):
        try:
            p.unlink()
            removed += 1
        except Exception:
            pass
    return removed
