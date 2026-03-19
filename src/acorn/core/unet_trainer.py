"""
UNet training backend for ACORN datasets.

Reads ACORN's COCO export directly — no format conversion needed.
Per-instance binary masks are composited into a single multi-class
label mask at load time (background=0, class1=1, class2=2, …).

Output layout (inside dataset_dir/training/unet/)
--------------------------------------------------
  best_weights.pt    state-dict of best validation-loss epoch
  last_weights.pt    state-dict of final epoch
  training_info.json arch, encoder, class list, hyperparams
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


# ── PyTorch dataset ───────────────────────────────────────────────────────────

class ACORNSegDataset:
    """PyTorch Dataset that reads an ACORN COCO split for UNet training.

    Each item is (image_tensor [1,H,W] float32, mask_tensor [H,W] int64).
    mask pixel values: 0 = background, 1..N = class indices.
    """

    def __init__(
        self,
        coco: dict,
        dataset_dir: Path,
        class_names: list[str],
        imgsz: int = 512,
        h5_path: Path | None = None,
    ) -> None:
        self.coco = coco
        self.dataset_dir = dataset_dir
        self.class_names = class_names
        self.imgsz = imgsz
        self.h5_path = h5_path
        self._h5_handle = None   # opened lazily per worker process
        self.images = coco.get("images", [])

        _skip = {"background", "ignore"}
        # cat_id → 1-based class index (0 reserved for background)
        name_to_idx: dict[str, int] = {
            n.lower(): i + 1 for i, n in enumerate(class_names)
        }
        self.cat_to_cls: dict[int, int] = {}
        next_free = [len(class_names) + 1]
        for cat in coco.get("categories", []):
            key = cat["name"].lower()
            if key in _skip:
                continue
            if key in name_to_idx:
                self.cat_to_cls[cat["id"]] = name_to_idx[key]
            else:
                # Unknown category — append dynamically
                self.cat_to_cls[cat["id"]] = next_free[0]
                name_to_idx[key] = next_free[0]
                next_free[0] += 1

        # image_id → annotation list
        self.img_anns: dict[int, list] = {}
        for ann in coco.get("annotations", []):
            self.img_anns.setdefault(ann["image_id"], []).append(ann)

    def __len__(self) -> int:
        return len(self.images)

    def _get_h5(self):
        """Open HDF5 file lazily — once per worker process (fork-safe)."""
        if self._h5_handle is None and self.h5_path is not None:
            import h5py
            self._h5_handle = h5py.File(self.h5_path, "r")
        return self._h5_handle

    def __getitem__(self, idx: int):
        import torch
        from PIL import Image as PILImage
        import torchvision.transforms.functional as TF
        from skimage.draw import polygon as skpoly

        img_rec = self.images[idx]

        file_name = img_rec.get("file_name", "")
        if self.h5_path is not None and file_name.startswith("hdf5:"):
            hdf5_key = file_name.split(":", 1)[1]
            arr = self._get_h5()[f"images/{hdf5_key}"][()]
            img = PILImage.fromarray(arr)
        else:
            img = PILImage.open(self.dataset_dir / file_name).convert("L")
        w, h = img.size

        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in self.img_anns.get(img_rec["id"], []):
            cls_idx = self.cat_to_cls.get(ann["category_id"], 0)
            if cls_idx == 0:
                continue
            seg = ann.get("segmentation", [])
            if not seg or not seg[0] or len(seg[0]) < 6:
                # Fall back to bbox fill
                bx, by, bw, bh = (int(v) for v in ann["bbox"])
                mask[by:by + bh, bx:bx + bw] = cls_idx
                continue
            flat = seg[0]
            xs = [flat[i] for i in range(0, len(flat), 2)]
            ys = [flat[i] for i in range(1, len(flat), 2)]
            rr, cc = skpoly(ys, xs, shape=(h, w))
            mask[rr, cc] = cls_idx

        # Resize image and mask
        img_r = img.resize((self.imgsz, self.imgsz), PILImage.BILINEAR)
        mask_pil = PILImage.fromarray(mask).resize(
            (self.imgsz, self.imgsz), PILImage.NEAREST
        )

        img_t = TF.to_tensor(img_r)                       # [1, H, W] float32
        mask_t = torch.from_numpy(np.array(mask_pil)).long()  # [H, W] int64
        return img_t, mask_t


# ── trainer ───────────────────────────────────────────────────────────────────

class UNetTrainer:
    """Train a UNet-family segmentation model on an ACORN dataset.

    Parameters
    ----------
    dataset_dir  : ACORN export directory (contains annotations.json, splits/).
    arch         : smp architecture name, e.g. "Unet", "UnetPlusPlus", "FPN".
    encoder      : smp encoder backbone, e.g. "resnet34", "efficientnet-b0".
    epochs       : Number of training epochs.
    batch        : Batch size.
    lr           : Initial AdamW learning rate.
    imgsz        : Input image size (square, pixels).
    devices      : List of GPU indices, e.g. [0, 1], or "cpu".
    project_dir  : Where to write output.  Defaults to
                   <dataset_dir>/training/unet/.
    log_cb       : Called with a log message string each epoch.
    progress_cb  : Called with (current_epoch, total_epochs) each epoch.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        arch: str = "Unet",
        encoder: str = "resnet34",
        epochs: int = 50,
        batch: int = 8,
        lr: float = 1e-4,
        imgsz: int = 512,
        devices: list[int] | str = "cpu",
        project_dir: str | Path | None = None,
        log_cb: Callable[[str], None] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        metrics_cb: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.arch = arch
        self.encoder = encoder
        self.epochs = epochs
        self.batch = batch
        self.lr = lr
        self.imgsz = imgsz
        self.devices = devices
        self.project_dir = (
            Path(project_dir) if project_dir
            else self.dataset_dir / "training" / "unet"
        )
        self.log_cb = log_cb or (lambda m: None)
        self.progress_cb = progress_cb or (lambda e, t: None)
        self.metrics_cb = metrics_cb or (lambda e, tl, m: None)

    def train(self) -> Path:
        """Run training and return the path to best_weights.pt."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        import segmentation_models_pytorch as smp

        splits_dir = self.dataset_dir / "splits"
        for name in ("train.json", "val.json"):
            if not (splits_dir / name).exists():
                raise FileNotFoundError(
                    f"splits/{name} not found — run Finalize Dataset first."
                )

        train_coco = json.loads((splits_dir / "train.json").read_text())
        val_coco   = json.loads((splits_dir / "val.json").read_text())

        # Determine ordered class list (exclude Background / Ignore)
        _skip = {"background", "ignore"}
        all_cats = train_coco.get("categories", [])
        class_names = [c["name"] for c in all_cats if c["name"].lower() not in _skip]
        n_classes = len(class_names) + 1   # +1 for background

        self.log_cb(
            f"Classes ({n_classes}): background + {', '.join(class_names)}"
        )

        _h5 = self.dataset_dir / "dataset.h5"
        h5_path = _h5 if _h5.exists() else None

        train_ds = ACORNSegDataset(
            train_coco, self.dataset_dir, class_names, self.imgsz, h5_path=h5_path
        )
        val_ds = ACORNSegDataset(
            val_coco, self.dataset_dir, class_names, self.imgsz, h5_path=h5_path
        )

        self.log_cb(
            f"Train: {len(train_ds)} images    Val: {len(val_ds)} images"
        )

        train_loader = DataLoader(
            train_ds, batch_size=self.batch, shuffle=True,
            num_workers=0, pin_memory=True, drop_last=len(train_ds) >= self.batch,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.batch, shuffle=False,
            num_workers=0, pin_memory=True,
        )

        # Build model
        model_cls = getattr(smp, self.arch)
        model = model_cls(
            encoder_name=self.encoder,
            encoder_weights="imagenet",
            in_channels=1,
            classes=n_classes,
        )

        # Device setup
        if isinstance(self.devices, list) and self.devices:
            primary = torch.device(f"cuda:{self.devices[0]}")
            model = model.to(primary)
            if len(self.devices) > 1:
                model = nn.DataParallel(model, device_ids=self.devices)
        else:
            primary = torch.device("cpu")
            model = model.to(primary)

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs, eta_min=self.lr * 0.01
        )
        criterion = nn.CrossEntropyLoss()

        self.project_dir.mkdir(parents=True, exist_ok=True)
        best_val_dice = -1.0
        best_weights_path = self.project_dir / "best_weights.pt"
        last_weights_path = self.project_dir / "last_weights.pt"
        metrics_csv_path  = self.project_dir / "metrics.csv"

        # CSV columns — same names as YOLO trainer for cross-model comparability
        # Per-class columns: precision_C, recall_C, f1_C, iou_C for every class
        all_class_labels = ["background"] + class_names
        per_cls_headers: list[str] = []
        for metric in ("precision", "recall", "f1", "iou"):
            for c in all_class_labels:
                per_cls_headers.append(f"{metric}_{c}")
        csv_header = (
            "epoch,train_loss,val_loss,pixel_acc,"
            "mean_precision,mean_recall,mean_f1,mean_iou,"
            + ",".join(per_cls_headers)
        )
        metrics_csv_path.write_text(csv_header + "\n")

        best_metrics: dict = {}   # filled at best epoch

        self.log_cb(
            f"Starting UNet training\n"
            f"  arch:    {self.arch} / {self.encoder}\n"
            f"  epochs:  {self.epochs}  batch: {self.batch}  lr: {self.lr}\n"
            f"  imgsz:   {self.imgsz}  device: {primary}\n"
            f"  metrics: loss, pixel_acc, Dice (per class + mean), IoU (per class + mean)\n"
            f"  metrics CSV: {metrics_csv_path}"
        )

        for epoch in range(1, self.epochs + 1):
            # ── train ──────────────────────────────────────────────────────────
            model.train()
            train_loss = 0.0
            for imgs, masks in train_loader:
                imgs  = imgs.to(primary)
                masks = masks.to(primary)
                optimizer.zero_grad()
                loss = criterion(model(imgs), masks)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * imgs.size(0)
            train_loss /= max(len(train_ds), 1)

            # ── validate ───────────────────────────────────────────────────────
            model.eval()
            val_loss  = 0.0
            correct   = 0
            total_px  = 0
            # Accumulators for per-class TP, FP, FN
            tp = torch.zeros(n_classes, dtype=torch.long)
            fp = torch.zeros(n_classes, dtype=torch.long)
            fn = torch.zeros(n_classes, dtype=torch.long)

            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs  = imgs.to(primary)
                    masks = masks.to(primary)
                    preds = model(imgs)
                    val_loss += criterion(preds, masks).item() * imgs.size(0)
                    pred_cls = preds.argmax(dim=1)   # (B, H, W)
                    correct  += (pred_cls == masks).sum().item()
                    total_px += masks.numel()
                    # Accumulate confusion stats on CPU
                    p = pred_cls.cpu()
                    t = masks.cpu()
                    for c in range(n_classes):
                        pred_c = p == c
                        true_c = t == c
                        tp[c] += (pred_c & true_c).sum()
                        fp[c] += (pred_c & ~true_c).sum()
                        fn[c] += (~pred_c & true_c).sum()

            val_loss /= max(len(val_ds), 1)
            acc = 100.0 * correct / max(total_px, 1)

            eps = 1e-6
            tp_f = tp.float()
            fp_f = fp.float()
            fn_f = fn.float()
            prec_per = (tp_f / (tp_f + fp_f + eps)).tolist()
            rec_per  = (tp_f / (tp_f + fn_f + eps)).tolist()
            f1_per   = (2 * tp_f / (2 * tp_f + fp_f + fn_f + eps)).tolist()   # = Dice
            iou_per  = (tp_f / (tp_f + fp_f + fn_f + eps)).tolist()

            # Means over foreground classes only (skip background index 0)
            fg = slice(1, None) if n_classes > 1 else slice(0, None)
            mean_prec = float(np.mean(prec_per[fg]))
            mean_rec  = float(np.mean(rec_per[fg]))
            mean_f1   = float(np.mean(f1_per[fg]))
            mean_iou  = float(np.mean(iou_per[fg]))

            # Per-class log lines (foreground only)
            col_w = max(len(n) for n in class_names) + 2
            header_line = f"  {'Class':<{col_w}}  Prec    Rec     F1      IoU"
            cls_lines = "\n".join(
                f"  {name:<{col_w}}  {prec_per[i+1]:.3f}   {rec_per[i+1]:.3f}   "
                f"{f1_per[i+1]:.3f}   {iou_per[i+1]:.3f}"
                for i, name in enumerate(class_names)
            )
            self.log_cb(
                f"Epoch {epoch}/{self.epochs}  "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"acc={acc:.1f}%  mF1={mean_f1:.3f}  mIoU={mean_iou:.3f}\n"
                f"{header_line}\n{cls_lines}"
            )
            self.progress_cb(epoch, self.epochs)
            self.metrics_cb(epoch, train_loss, mean_f1)
            scheduler.step()

            # Append row to CSV (same column order as YOLO trainer)
            per_vals: list[str] = []
            for vals in (prec_per, rec_per, f1_per, iou_per):
                per_vals.extend(f"{v:.4f}" for v in vals)
            csv_row = (
                f"{epoch},{train_loss:.4f},{val_loss:.4f},{acc:.2f},"
                f"{mean_prec:.4f},{mean_rec:.4f},{mean_f1:.4f},{mean_iou:.4f},"
                + ",".join(per_vals)
            )
            with open(metrics_csv_path, "a") as fcsv:
                fcsv.write(csv_row + "\n")

            # Save best checkpoint on mean foreground F1
            if mean_f1 > best_val_dice:
                best_val_dice = mean_f1
                m = model.module if isinstance(model, nn.DataParallel) else model
                torch.save(m.state_dict(), best_weights_path)
                best_metrics = {
                    "epoch": epoch,
                    "mean_precision": mean_prec,
                    "mean_recall":    mean_rec,
                    "mean_f1":        mean_f1,
                    "mean_iou":       mean_iou,
                    "per_class": {
                        name: {
                            "precision": prec_per[i+1],
                            "recall":    rec_per[i+1],
                            "f1":        f1_per[i+1],
                            "iou":       iou_per[i+1],
                        }
                        for i, name in enumerate(class_names)
                    },
                }

        # Save last weights
        m = model.module if isinstance(model, nn.DataParallel) else model
        torch.save(m.state_dict(), last_weights_path)

        # ── test set evaluation on best weights (held-out, for publication) ───
        test_metrics: dict = {}
        test_coco_path = splits_dir / "test.json"
        if test_coco_path.exists():
            self.log_cb("Evaluating on held-out test set...")
            test_coco = json.loads(test_coco_path.read_text())
            test_ds = ACORNSegDataset(test_coco, self.dataset_dir, class_names, self.imgsz, h5_path=h5_path)
            if len(test_ds) > 0:
                test_loader = DataLoader(
                    test_ds, batch_size=self.batch, shuffle=False,
                    num_workers=0, pin_memory=True,
                )
                # Reload best weights
                m_best = model.module if isinstance(model, nn.DataParallel) else model
                m_best.load_state_dict(torch.load(best_weights_path, map_location=primary,
                                                   weights_only=True))
                m_best.eval()
                tp_t = torch.zeros(n_classes, dtype=torch.long)
                fp_t = torch.zeros(n_classes, dtype=torch.long)
                fn_t = torch.zeros(n_classes, dtype=torch.long)
                correct_t = 0
                total_t   = 0
                with torch.no_grad():
                    for imgs, masks in test_loader:
                        imgs  = imgs.to(primary)
                        masks = masks.to(primary)
                        pred_cls = m_best(imgs).argmax(dim=1)
                        correct_t += (pred_cls == masks).sum().item()
                        total_t   += masks.numel()
                        p = pred_cls.cpu()
                        t = masks.cpu()
                        for c in range(n_classes):
                            pred_c = p == c
                            true_c = t == c
                            tp_t[c] += (pred_c & true_c).sum()
                            fp_t[c] += (pred_c & ~true_c).sum()
                            fn_t[c] += (~pred_c & true_c).sum()
                tp_f = tp_t.float(); fp_f = fp_t.float(); fn_f = fn_t.float()
                prec_t = (tp_f / (tp_f + fp_f + eps)).tolist()
                rec_t  = (tp_f / (tp_f + fn_f + eps)).tolist()
                f1_t   = (2 * tp_f / (2 * tp_f + fp_f + fn_f + eps)).tolist()
                iou_t  = (tp_f / (tp_f + fp_f + fn_f + eps)).tolist()
                acc_t  = 100.0 * correct_t / max(total_t, 1)
                test_metrics = {
                    "n_images":       len(test_ds),
                    "pixel_acc":      acc_t,
                    "mean_precision": float(np.mean(prec_t[1:])),
                    "mean_recall":    float(np.mean(rec_t[1:])),
                    "mean_f1":        float(np.mean(f1_t[1:])),
                    "mean_iou":       float(np.mean(iou_t[1:])),
                    "per_class": {
                        name: {
                            "precision": prec_t[i+1],
                            "recall":    rec_t[i+1],
                            "f1":        f1_t[i+1],
                            "iou":       iou_t[i+1],
                        }
                        for i, name in enumerate(class_names)
                    },
                }
                mf1_t = test_metrics["mean_f1"]
                miou_t = test_metrics["mean_iou"]
                self.log_cb(
                    f"Test set ({len(test_ds)} images):  "
                    f"mF1={mf1_t:.3f}  mIoU={miou_t:.3f}  acc={acc_t:.1f}%"
                )
        else:
            self.log_cb("No test split found — skipping test evaluation.")

        # ── figures ───────────────────────────────────────────────────────────
        _save_figures(
            project_dir=self.project_dir,
            metrics_csv=metrics_csv_path,
            best_metrics=best_metrics,
            test_metrics=test_metrics,
            class_names=class_names,
            log_cb=self.log_cb,
        )

        # Save metadata for the UNet panel to auto-populate
        info = {
            "model_type": "unet",
            "arch": self.arch,
            "encoder": self.encoder,
            "in_channels": 1,
            "n_classes": n_classes,
            "class_names": ["background"] + class_names,
            "epochs": self.epochs,
            "batch": self.batch,
            "lr": self.lr,
            "imgsz": self.imgsz,
            "best_weights": str(best_weights_path),
            "last_weights": str(last_weights_path),
            "best_metrics": best_metrics,
            "test_metrics": test_metrics,
        }
        (self.project_dir / "training_info.json").write_text(
            json.dumps(info, indent=2)
        )

        # ── human-readable summary report ─────────────────────────────────────
        report = _format_summary_report(
            model_label=f"UNet  {self.arch} / {self.encoder}",
            dataset_dir=self.dataset_dir,
            val_metrics=best_metrics,
            test_metrics=test_metrics,
            metrics_csv=metrics_csv_path,
        )
        report_path = self.project_dir / "training_report.txt"
        report_path.write_text(report)
        self.log_cb(f"\n{report}\nReport saved: {report_path}")

        return best_weights_path


# ── shared report formatter ───────────────────────────────────────────────────

def _format_summary_report(
    model_label: str,
    dataset_dir: Path,
    val_metrics: dict,
    test_metrics: dict,
    metrics_csv: Path,
) -> str:
    """Return a formatted plain-text training summary (val + test)."""

    def _table(metrics: dict, class_names_ordered: list[str]) -> list[str]:
        if not metrics:
            return ["  No metrics recorded."]
        col_w = max((len(n) for n in class_names_ordered), default=5) + 2
        col_w = max(col_w, len("Mean (foreground)") + 2)
        mprec = metrics.get("mean_precision", float("nan"))
        mrec  = metrics.get("mean_recall",    float("nan"))
        mf1   = metrics.get("mean_f1",        float("nan"))
        miou  = metrics.get("mean_iou",       float("nan"))
        rows = [
            f"  {'Class':<{col_w}}  {'Precision':>9}  {'Recall':>6}  {'F1':>6}  {'IoU@50':>7}",
            "  " + "-" * (col_w + 38),
        ]
        per = metrics.get("per_class", {})
        for name in class_names_ordered:
            v = per.get(name, {})
            rows.append(
                f"  {name:<{col_w}}  {v.get('precision', float('nan')):>9.3f}  "
                f"{v.get('recall', float('nan')):>6.3f}  "
                f"{v.get('f1', float('nan')):>6.3f}  "
                f"{v.get('iou', float('nan')):>7.3f}"
            )
        rows += [
            "  " + "-" * (col_w + 38),
            f"  {'Mean (foreground)':<{col_w}}  {mprec:>9.3f}  {mrec:>6.3f}  "
            f"{mf1:>6.3f}  {miou:>7.3f}",
        ]
        return rows

    class_names = list(val_metrics.get("per_class", {}).keys())

    lines = [
        "=" * 62,
        f"  Training Summary: {model_label}",
        f"  Dataset: {dataset_dir}",
        "=" * 62,
    ]

    best_ep = val_metrics.get("epoch", "?")
    lines.append(f"  Best checkpoint: epoch {best_ep}")
    lines.append("")
    lines.append("  VALIDATION SET (used for model selection)")
    lines.extend(_table(val_metrics, class_names))

    if test_metrics:
        n_test = test_metrics.get("n_images", "?")
        lines.append("")
        lines.append(f"  TEST SET — held-out, report these numbers in your paper  (n={n_test})")
        lines.extend(_table(test_metrics, class_names))
    else:
        lines += ["", "  TEST SET: not available (no test split found)"]

    lines += [
        "",
        "  NOTE: F1 = Dice coefficient.  IoU@50 = Jaccard index.",
        "  Comparable across UNet and YOLO via these two metrics.",
        f"  Full per-epoch metrics: {metrics_csv}",
        "=" * 62,
    ]
    return "\n".join(lines)


# ── figure generation ─────────────────────────────────────────────────────────

def _save_figures(
    project_dir: Path,
    metrics_csv: Path,
    best_metrics: dict,
    test_metrics: dict,
    class_names: list[str],
    log_cb: Callable[[str], None],
) -> None:
    """Save publication-quality figures to project_dir/figures/."""
    try:
        import csv as _csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
    except ImportError:
        log_cb("matplotlib not available — skipping figure generation.")
        return

    fig_dir = project_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # ── read metrics CSV ───────────────────────────────────────────────────────
    epochs, train_loss, val_loss = [], [], []
    mean_f1, mean_iou = [], []

    try:
        with open(metrics_csv, newline="") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                try:
                    epochs.append(int(row["epoch"]))
                    train_loss.append(float(row.get("train_loss", "nan")))
                    val_loss.append(float(row.get("val_loss", "nan")))
                    mean_f1.append(float(row.get("mean_f1", "nan")))
                    mean_iou.append(float(row.get("mean_iou", "nan")))
                except (ValueError, KeyError):
                    continue
    except Exception as exc:
        log_cb(f"Could not read metrics CSV for figures: {exc}")
        return

    if not epochs:
        log_cb("No epoch data found — skipping figures.")
        return

    GREY  = "#333333"
    BLUE  = "#1a6fa8"
    GREEN = "#27ae60"
    RED   = "#e74c3c"

    def _apply_style(ax, xlabel, ylabel):
        ax.set_xlabel(xlabel, color=GREY)
        ax.set_ylabel(ylabel, color=GREY)
        ax.tick_params(colors=GREY)
        for spine in ax.spines.values():
            spine.set_edgecolor("#cccccc")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(framealpha=0.9)

    # ── Figure 1: Loss curves ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, train_loss, color=BLUE,  linewidth=1.5, label="Train loss")
    ax.plot(epochs, val_loss,   color=RED,   linewidth=1.5, label="Val loss",
            linestyle="--")
    if best_metrics.get("epoch"):
        ax.axvline(best_metrics["epoch"], color=GREEN, linestyle=":", linewidth=1,
                   label=f"Best epoch ({best_metrics['epoch']})")
    _apply_style(ax, "Epoch", "Loss")
    ax.set_title("Training and Validation Loss", color=GREY)
    fig.tight_layout()
    fig.savefig(fig_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: F1 and IoU curves ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, mean_f1,  color=BLUE,  linewidth=1.5, label="Mean F1 (val)")
    ax.plot(epochs, mean_iou, color=GREEN, linewidth=1.5, label="Mean IoU (val)",
            linestyle="--")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    _apply_style(ax, "Epoch", "Score")
    ax.set_title("Validation F1 and IoU", color=GREY)
    fig.tight_layout()
    fig.savefig(fig_dir / "metric_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 3: Per-class bar chart (test if available, else val) ───────────
    metrics_for_bar = test_metrics if test_metrics else best_metrics
    bar_label = "Test set" if test_metrics else "Val set (best epoch)"
    per = metrics_for_bar.get("per_class", {})
    if per and class_names:
        bar_names = list(per.keys())
        f1_vals  = [per[n]["f1"]  for n in bar_names]
        iou_vals = [per[n]["iou"] for n in bar_names]

        x = np.arange(len(bar_names))
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(5, len(bar_names) * 1.5), 4))
        ax.bar(x - width/2, f1_vals,  width, label="F1 / Dice", color=BLUE,  alpha=0.85)
        ax.bar(x + width/2, iou_vals, width, label="IoU",        color=GREEN, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_names, rotation=20, ha="right", color=GREY)
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        _apply_style(ax, "Class", "Score")
        ax.set_title(f"Per-class F1 and IoU  ({bar_label})", color=GREY)
        fig.tight_layout()
        fig.savefig(fig_dir / "per_class_metrics.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    log_cb(f"Figures saved to {fig_dir}/\n"
           f"  loss_curves.png  |  metric_curves.png  |  per_class_metrics.png")
