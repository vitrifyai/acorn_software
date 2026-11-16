"""U-Net semantic segmentation picker for exact-truth measurement-bias studies.

The network predicts three classes per pixel: background, object interior, and
a thin band at the object boundary. Interiors seed a watershed over the
foreground, which separates touching objects. That post-processing is part of
the method being measured, because where the watershed places a shared
boundary is itself a source of directional error.

Training labels are rasterized from the analytic truth shapes (circles, or
ellipses for spores) at a stated edge definition, so no annotation enters.
Objects that carry a ``polygon_px`` outline (hand-drawn references) are
rasterized from that outline instead, so the same trainer learns from
annotations when they are the labels.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from skimage.segmentation import watershed

from acorn_sim_common.classical_pickers import _dataset_items, _read_image, mask_outline

BACKGROUND, INTERIOR, BOUNDARY = 0, 1, 2
IGNORE = 255  # excluded from the loss


def rasterize_truth_labels(
    truth_payload: dict,
    *,
    target_field: str = "diameter_nm",
    boundary_px: float = 1.5,
) -> np.ndarray:
    """Three-class label map from analytic truth shapes.

    Pixels within ``boundary_px`` inside any object's edge, and pixels where
    two objects overlap, are labelled BOUNDARY so the network learns to keep
    neighbours apart.
    """

    h, w = (int(v) for v in truth_payload["image_shape_px"][:2])
    px = float(truth_payload["pixel_size_nm"])
    # Pixel i is centred at coordinate i, the simulator's convention.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cover = np.zeros((h, w), dtype=np.int16)
    band = np.zeros((h, w), dtype=bool)
    for obj in truth_payload.get("objects", []):
        # Signed distance in px from the edge, negative inside.
        dx, dy = xx - float(obj["cx_px"]), yy - float(obj["cy_px"])
        if obj.get("polygon_px"):
            inside = _polygon_mask(obj["polygon_px"], (h, w))
            cover += inside
            band |= inside & (ndimage.distance_transform_edt(inside) <= boundary_px)
            continue
        if obj.get("major_nm") is not None and obj.get("minor_nm") is not None:
            a = float(obj["major_nm"]) / (2.0 * px)
            b = float(obj["minor_nm"]) / (2.0 * px)
            t = float(obj.get("angle_rad") or 0.0)
            u = dx * math.cos(t) + dy * math.sin(t)
            v = -dx * math.sin(t) + dy * math.cos(t)
            rho = np.sqrt((u / a) ** 2 + (v / b) ** 2)
            signed = (rho - 1.0) * b  # minor-axis scale keeps the band thin
        else:
            r = _truth_diameter_nm(obj, target_field) / (2.0 * px)
            signed = np.hypot(dx, dy) - r
        inside = signed <= 0.0
        cover += inside
        band |= inside & (signed > -boundary_px)
    labels = np.zeros((h, w), dtype=np.uint8)
    labels[cover > 0] = INTERIOR
    labels[band | (cover > 1)] = BOUNDARY
    return labels


def ignore_unlabelled(labels: np.ndarray, truth_payload: dict, radii: float) -> np.ndarray:
    """Mark pixels farther than ``radii`` equivalent radii beyond every object as IGNORE.

    For selective annotation (only some particles marked): unmarked particles
    then never act as background, and the network learns edges, not which
    particles were chosen.
    """

    h, w = labels.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    px = float(truth_payload["pixel_size_nm"])
    known = np.zeros((h, w), dtype=bool)
    for obj in truth_payload.get("objects", []):
        r = _truth_diameter_nm(obj, "diameter_nm") / (2.0 * px)
        known |= np.hypot(xx - float(obj["cx_px"]), yy - float(obj["cy_px"])) <= (1.0 + radii) * r
    out = labels.copy()
    out[~known] = IGNORE
    return out


def rasterize_truth_instances(truth_payload: dict, *, target_field: str = "diameter_nm") -> np.ndarray:
    """Instance label map (0 = background, k = k-th object) from analytic truth."""

    h, w = (int(v) for v in truth_payload["image_shape_px"][:2])
    px = float(truth_payload["pixel_size_nm"])
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    labels = np.zeros((h, w), dtype=np.uint16)
    for k, obj in enumerate(truth_payload.get("objects", []), start=1):
        dx, dy = xx - float(obj["cx_px"]), yy - float(obj["cy_px"])
        if obj.get("polygon_px"):
            inside = _polygon_mask(obj["polygon_px"], (h, w))
        elif obj.get("major_nm") is not None and obj.get("minor_nm") is not None:
            a = float(obj["major_nm"]) / (2.0 * px)
            b = float(obj["minor_nm"]) / (2.0 * px)
            t = float(obj.get("angle_rad") or 0.0)
            u = dx * math.cos(t) + dy * math.sin(t)
            v = -dx * math.sin(t) + dy * math.cos(t)
            inside = (u / a) ** 2 + (v / b) ** 2 <= 1.0
        else:
            r = _truth_diameter_nm(obj, target_field) / (2.0 * px)
            inside = dx * dx + dy * dy <= r * r
        labels[inside] = k
    return labels


def _polygon_mask(polygon_px, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centres fall inside the outline (pixel i centred at i)."""

    from skimage.draw import polygon as draw_polygon

    pts = np.asarray(polygon_px, dtype=float)
    mask = np.zeros(shape, dtype=bool)
    rr, cc = draw_polygon(pts[:, 1], pts[:, 0], shape=shape)
    mask[rr, cc] = True
    return mask


def build_unet(in_channels: int = 1, n_classes: int = 3, base: int = 32, depth: int = 4):
    import torch
    from torch import nn

    def block(cin: int, cout: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            chans = [base * 2**i for i in range(depth + 1)]
            self.down = nn.ModuleList([block(in_channels, chans[0])])
            self.down.extend(block(chans[i], chans[i + 1]) for i in range(depth))
            self.up = nn.ModuleList(nn.ConvTranspose2d(chans[i + 1], chans[i], 2, stride=2) for i in reversed(range(depth)))
            self.merge = nn.ModuleList(block(chans[i] * 2, chans[i]) for i in reversed(range(depth)))
            self.head = nn.Conv2d(chans[0], n_classes, 1)

        def forward(self, x):
            skips = []
            for i, layer in enumerate(self.down):
                x = layer(x if i == 0 else nn.functional.max_pool2d(x, 2))
                skips.append(x)
            skips.pop()
            for up, merge in zip(self.up, self.merge):
                x = up(x)
                x = merge(torch.cat([x, skips.pop()], dim=1))
            return self.head(x)

    return UNet()


def train_unet(
    dataset_dir: Path,
    output_path: Path,
    *,
    target_field: str = "diameter_nm",
    epochs: int = 100,
    batch: int = 8,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
    base: int = 32,
    init_weights: Path | None = None,
    ignore_beyond_radii: float | None = None,
    class_weights: bool = True,
) -> Path:
    """Train on the ``train`` split, keep the checkpoint with the best ``val`` loss.

    ``init_weights`` starts from an earlier checkpoint (fine-tuning) instead of
    random initialisation. ``ignore_beyond_radii`` trains only near labelled
    objects (see ignore_unlabelled), for selectively annotated images.
    """

    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset_dir = Path(dataset_dir)
    train_x, train_y = _load_split(dataset_dir, "train", target_field, ignore_beyond_radii)
    val_x, val_y = _load_split(dataset_dir, "val", target_field, ignore_beyond_radii)
    if not train_x:
        raise ValueError(f"no training images under {dataset_dir}")

    model = build_unet(base=base).to(device)
    if init_weights is not None:
        model.load_state_dict(torch.load(init_weights, map_location=device)["state_dict"])
    # Boundary pixels are rare and decide instance separation; weight them up.
    counts = np.bincount(np.concatenate([y[y != IGNORE].ravel() for y in train_y]), minlength=3)[:3].astype(float)
    weights = torch.tensor(counts.sum() / (3.0 * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weights if class_weights else None, ignore_index=IGNORE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    best = math.inf
    history = []
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(train_x))
        train_loss = 0.0
        for start in range(0, len(order), batch):
            xb, yb = _augmented_batch([train_x[i] for i in order[start : start + batch]], [train_y[i] for i in order[start : start + batch]], rng)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(yb).to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            train_loss += float(loss) * len(xb)
        sched.step()
        val_loss = _eval_loss(model, loss_fn, val_x, val_y, device) if val_x else train_loss / len(train_x)
        history.append({"epoch": epoch, "train_loss": train_loss / len(train_x), "val_loss": val_loss})
        if val_loss < best:
            best = val_loss
            torch.save({"state_dict": model.state_dict(), "base": base, "target_field": target_field, "epoch": epoch}, output_path)
    output_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return output_path


def predict_unet(
    dataset_dir: Path,
    weights: Path,
    output_dir: Path,
    *,
    class_name: str,
    device: str = "cuda",
    min_area_px: int = 4,
    splits: tuple[str, ...] | None = ("test",),
) -> list[Path]:
    import torch

    ckpt = torch.load(weights, map_location=device, weights_only=False)
    model = build_unet(base=int(ckpt.get("base", 32))).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    written: list[Path] = []
    for item in _dataset_items(dataset_dir, splits=splits):
        image = _read_image(item.image_path)
        with torch.inference_mode():
            prob = torch.softmax(model(torch.from_numpy(_pad_to(image, 16)[None, None]).to(device)), dim=1)[0]
        prob = prob[:, : image.shape[0], : image.shape[1]].cpu().numpy()
        instances = instances_from_probabilities(prob, min_area_px=min_area_px)
        predictions = [
            _instance_record(instances == region.label, region, item.pixel_size_nm, class_name)
            for region in regionprops(instances)
        ]
        payload = {
            "stem": item.stem,
            "method": "unet-watershed",
            "class_name": class_name,
            "pixel_size_nm": item.pixel_size_nm,
            "source_image": str(item.image_path),
            "weights": str(weights),
            "predictions": predictions,
        }
        out_path = output_dir / item.split / f"{item.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(out_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps({"method": "unet-watershed", "class_name": class_name, "source_dataset": str(dataset_dir), "n_images": len(written)}, indent=2),
        encoding="utf-8",
    )
    return written


def instances_from_probabilities(
    prob: np.ndarray,
    *,
    min_area_px: int = 4,
    min_seed_fraction: float = 0.2,
) -> np.ndarray:
    """Seeded watershed: interior components seed, foreground bounds the flood.

    Within each connected foreground region, seeds smaller than
    ``min_seed_fraction`` of that region's largest seed are discarded, so
    stray interior pixels around a boundary band are flooded into the object
    they belong to rather than reported as objects of their own. Touching
    objects of comparable size keep separate seeds; at the default fraction a
    neighbour is split off down to about 0.45 of the larger one's diameter.
    """

    cls = np.argmax(prob, axis=0)
    foreground = cls != BACKGROUND
    regions, _ = ndimage.label(foreground)
    seeds, n_seeds = ndimage.label(cls == INTERIOR)
    sizes = np.bincount(seeds.ravel(), minlength=n_seeds + 1)
    sizes[0] = 0
    seed_region = np.zeros(n_seeds + 1, dtype=np.int64)
    seed_region[seeds.ravel()] = regions.ravel()
    largest = np.zeros(regions.max() + 1, dtype=np.int64)
    np.maximum.at(largest, seed_region[1:], sizes[1:])
    keep = (sizes >= min_area_px) & (sizes >= min_seed_fraction * largest[seed_region])
    keep[0] = False
    seeds[~keep[seeds]] = 0
    instances = watershed(-prob[INTERIOR], markers=seeds, mask=foreground)
    sizes = np.bincount(instances.ravel())
    instances[sizes[instances] < min_area_px] = 0
    return instances


def _instance_record(mask: np.ndarray, region, pixel_size_nm: float, class_name: str) -> dict:
    area_px = float(region.area)
    cy, cx = region.centroid
    return {
        "class_name": class_name,
        "cx_px": float(cx),
        "cy_px": float(cy),
        "diameter_nm": float(2.0 * math.sqrt(area_px / math.pi) * pixel_size_nm),
        "area_px": area_px,
        "polygon": mask_outline(mask),
        "touches_border": bool(
            region.bbox[0] == 0 or region.bbox[1] == 0 or region.bbox[2] == mask.shape[0] or region.bbox[3] == mask.shape[1]
        ),
        "source": "unet-watershed",
    }


def _load_split(
    dataset_dir: Path, split: str, target_field: str, ignore_beyond_radii: float | None = None
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    xs, ys = [], []
    for item in _dataset_items(dataset_dir, splits=(split,)):
        payload = json.loads(item.truth_path.read_text(encoding="utf-8"))
        y = rasterize_truth_labels(payload, target_field=target_field)
        if ignore_beyond_radii is not None:
            y = ignore_unlabelled(y, payload, ignore_beyond_radii)
            if not (y != IGNORE).any():
                continue  # nothing labelled in this image
        xs.append(_read_image(item.image_path))
        ys.append(y)
    return xs, ys


def _augmented_batch(xs: list[np.ndarray], ys: list[np.ndarray], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Dihedral augmentation only: it changes no object size, so it cannot
    # itself introduce a size bias into the trained model.
    bx, by = [], []
    for x, y in zip(xs, ys):
        k = int(rng.integers(4))
        x, y = np.rot90(x, k), np.rot90(y, k)
        if rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]
        bx.append(_pad_to(np.ascontiguousarray(x), 16))
        by.append(_pad_to(np.ascontiguousarray(y), 16))
    return np.stack(bx)[:, None].astype(np.float32), np.stack(by).astype(np.int64)


def _eval_loss(model, loss_fn, xs, ys, device) -> float:
    import torch

    model.eval()
    total = 0.0
    with torch.inference_mode():
        for x, y in zip(xs, ys):
            xb = torch.from_numpy(_pad_to(x, 16)[None, None]).to(device)
            yb = torch.from_numpy(_pad_to(y, 16)[None].astype(np.int64)).to(device)
            total += float(loss_fn(model(xb), yb))
    return total / len(xs)


def _pad_to(arr: np.ndarray, multiple: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ph, pw = (-h) % multiple, (-w) % multiple
    if ph == 0 and pw == 0:
        return arr
    return np.pad(arr, ((0, ph), (0, pw)), mode="reflect" if arr.dtype.kind == "f" else "edge")


def _truth_diameter_nm(obj: dict, target_field: str) -> float:
    for key in (target_field, "diameter_nm", "ecd_nm", "d_outer_leaflet_nm"):
        if obj.get(key) is not None:
            return float(obj[key])
    raise KeyError(f"truth object has no diameter field {target_field!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train or run the U-Net picker on an exact-truth simulator dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--dataset", required=True, type=Path)
    tr.add_argument("--output", required=True, type=Path, help="Checkpoint path (.pt).")
    tr.add_argument("--target-field", default="diameter_nm")
    tr.add_argument("--epochs", type=int, default=100)
    tr.add_argument("--batch", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--device", default="cuda")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--base", type=int, default=32)
    tr.add_argument("--init-weights", type=Path, default=None, help="Checkpoint to fine-tune from.")
    tr.add_argument("--no-class-weights", action="store_true",
                    help="Unweighted loss (default up-weights the rare boundary class).")
    tr.add_argument("--ignore-beyond-radii", type=float, default=None,
                    help="Leave pixels this many radii beyond every labelled object out of the loss.")

    pr = sub.add_parser("predict")
    pr.add_argument("--dataset", required=True, type=Path)
    pr.add_argument("--weights", required=True, type=Path)
    pr.add_argument("--output", required=True, type=Path)
    pr.add_argument("--class-name", required=True)
    pr.add_argument("--device", default="cuda")
    pr.add_argument("--min-area-px", type=int, default=4)
    pr.add_argument("--splits", nargs="+", default=["test"])

    args = parser.parse_args(argv)
    if args.cmd == "train":
        print(
            train_unet(
                args.dataset, args.output, target_field=args.target_field, epochs=args.epochs, batch=args.batch,
                lr=args.lr, device=args.device, seed=args.seed, base=args.base,
                init_weights=args.init_weights, ignore_beyond_radii=args.ignore_beyond_radii,
                class_weights=not args.no_class_weights,
            )
        )
        return 0
    paths = predict_unet(
        args.dataset, args.weights, args.output, class_name=args.class_name, device=args.device,
        min_area_px=args.min_area_px, splits=tuple(args.splits),
    )
    print(f"Wrote {len(paths)} U-Net prediction files to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
