"""
YOLO training backend for ACORN datasets.

Converts ACORN's COCO export to YOLO segmentation format and runs
Ultralytics YOLO training with progress callbacks.

Output layout (inside dataset_dir/training/yolo/)
--------------------------------------------------
  data/
    images/train/   images/val/
    labels/train/   labels/val/
    dataset.yaml
  run/
    weights/best.pt
    weights/last.pt
  training_info.json   class list, arch, training params
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import threading
from pathlib import Path
from typing import Callable

import numpy as np


# ── COCO → YOLO format conversion ────────────────────────────────────────────

def _is_hdf5_dataset(dataset_dir: Path) -> bool:
    """Return True if this dataset uses HDF5 storage (dataset.h5 present)."""
    return (dataset_dir / "dataset.h5").exists()


def convert_to_yolo(dataset_dir: Path, out_dir: Path) -> tuple[Path, list[str]]:
    """Convert an ACORN COCO export to YOLO segmentation format.

    For HDF5 datasets (dataset.h5 present), tile images are extracted from
    the HDF5 file into out_dir/images/.  For legacy PNG datasets the images
    are copied directly.

    Returns (yaml_path, class_names).
    Skips Background and Ignore categories — YOLO learns them implicitly.
    """
    from PIL import Image as PILImage

    ann_path = dataset_dir / "annotations.json"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotations.json not found in {dataset_dir}")

    coco = json.loads(ann_path.read_text())
    categories = coco.get("categories", [])

    _skip = {"background", "ignore"}
    valid_cats = [c for c in categories if c["name"].lower() not in _skip]
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(valid_cats)}
    class_names = [c["name"] for c in valid_cats]

    splits_dir = dataset_dir / "splits"
    if not splits_dir.exists():
        raise FileNotFoundError(
            "splits/ directory not found — run Finalize Dataset first."
        )

    use_hdf5 = _is_hdf5_dataset(dataset_dir)

    h5 = None
    if use_hdf5:
        import h5py
        h5 = h5py.File(dataset_dir / "dataset.h5", "r")

    splits_written: set[str] = set()

    try:
        for split in ("train", "val"):
            split_file = splits_dir / f"{split}.json"
            if not split_file.exists():
                continue

            split_coco = json.loads(split_file.read_text())
            if not split_coco.get("images"):
                continue

            img_dir = out_dir / "images" / split
            lbl_dir = out_dir / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            splits_written.add(split)

            # image_id → list of annotations
            img_anns: dict[int, list] = {}
            for ann in split_coco.get("annotations", []):
                img_anns.setdefault(ann["image_id"], []).append(ann)

            for img_rec in split_coco.get("images", []):
                img_id = img_rec["id"]
                w = img_rec["width"]
                h_px = img_rec["height"]

                # Write image to YOLO data dir
                dst_img = img_dir / f"{img_id}.png"
                file_name = img_rec.get("file_name", "")
                if file_name.startswith("hdf5:") and h5 is not None:
                    hdf5_key = file_name.split(":", 1)[1]
                    arr = h5[f"images/{hdf5_key}"][()]
                    PILImage.fromarray(arr).save(str(dst_img))
                elif not file_name.startswith("hdf5:"):
                    src_img = dataset_dir / file_name
                    if src_img.exists():
                        shutil.copy2(src_img, dst_img)

                lbl_path = lbl_dir / f"{img_id}.txt"
                lines: list[str] = []

                for ann in img_anns.get(img_id, []):
                    cat_id = ann["category_id"]
                    if cat_id not in cat_id_to_idx:
                        continue
                    cls_idx = cat_id_to_idx[cat_id]

                    seg = ann.get("segmentation", [])
                    if seg and seg[0] and len(seg[0]) >= 6:
                        flat = seg[0]
                    else:
                        # Fall back to bbox corners
                        bx, by, bw, bh = ann["bbox"]
                        flat = [bx, by, bx + bw, by, bx + bw, by + bh, bx, by + bh]

                    norm = [
                        flat[i] / w if i % 2 == 0 else flat[i] / h_px
                        for i in range(len(flat))
                    ]
                    lines.append(str(cls_idx) + " " + " ".join(f"{v:.6f}" for v in norm))

                lbl_path.write_text("\n".join(lines))
    finally:
        if h5 is not None:
            h5.close()

    # If no dedicated val split exists, reuse train so YOLO doesn't error.
    val_dir = "images/val" if "val" in splits_written else "images/train"

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: {val_dir}\n"
        f"nc: {len(class_names)}\n"
        f"names: {class_names}\n"
    )
    return yaml_path, class_names


# ── stdout tee ────────────────────────────────────────────────────────────────

class _TeeToLog(io.TextIOBase):
    """Write to both original stdout and a log callback (line-buffered)."""

    def __init__(self, original, log_cb):
        self._orig = original
        self._log_cb = log_cb
        self._buf = ""
        self._lock = threading.Lock()

    def write(self, s: str) -> int:
        self._orig.write(s)
        self._orig.flush()
        with self._lock:
            self._buf += s
            while "\n" in self._buf:
                line, self._buf = self._buf.split("\n", 1)
                stripped = line.strip()
                if stripped:
                    try:
                        self._log_cb(stripped)
                    except Exception:
                        pass
        return len(s)

    def flush(self):
        self._orig.flush()

    def fileno(self):
        return self._orig.fileno()


# ── trainer ───────────────────────────────────────────────────────────────────

class YOLOTrainer:
    """Train a YOLO instance-segmentation model on an ACORN dataset.

    Parameters
    ----------
    dataset_dir : ACORN export directory (contains annotations.json, splits/).
    base_model  : Ultralytics model tag, e.g. "yolo11n-seg.pt".
    epochs      : Number of training epochs.
    batch       : Batch size (-1 = auto).
    imgsz       : Input image size (square).
    devices     : List of GPU indices, e.g. [0, 1], or "cpu".
    project_dir : Where to write training output.  Defaults to
                  <dataset_dir>/training/yolo/.
    log_cb      : Called with a log message string each epoch.
    progress_cb : Called with (current_epoch, total_epochs) each epoch.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        base_model: str = "yolo11n-seg.pt",
        epochs: int = 100,
        batch: int = 8,
        imgsz: int = 640,
        devices: list[int] | str = "cpu",
        project_dir: str | Path | None = None,
        log_cb: Callable[[str], None] | None = None,
        progress_cb: Callable[[int, int], None] | None = None,
        metrics_cb: Callable[[int, float, float], None] | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.base_model = base_model
        self.epochs = epochs
        self.batch = batch
        self.imgsz = imgsz
        self.devices = devices
        self.project_dir = (
            Path(project_dir) if project_dir
            else self.dataset_dir / "training" / "yolo"
        )
        self.log_cb = log_cb or (lambda m: None)
        self.progress_cb = progress_cb or (lambda e, t: None)
        self.metrics_cb = metrics_cb or (lambda e, tl, m: None)

    def train(self) -> Path:
        """Run training and return the path to best.pt."""
        import csv
        from ultralytics import YOLO

        yolo_data_dir = self.project_dir / "data"
        self.log_cb("Converting dataset to YOLO format...")
        yaml_path, class_names = convert_to_yolo(self.dataset_dir, yolo_data_dir)
        val_line = yaml_path.read_text().splitlines()
        val_reused = any("val: images/train" in l for l in val_line)
        self.log_cb(
            f"Classes ({len(class_names)}): {', '.join(class_names)}\n"
            f"Dataset written to {yolo_data_dir}"
            + ("\nNo val split found — validation metrics will use train set." if val_reused else "")
        )

        if isinstance(self.devices, list) and self.devices:
            device_str = ",".join(str(d) for d in self.devices)
        else:
            device_str = str(self.devices) if self.devices else "cpu"

        self.project_dir.mkdir(parents=True, exist_ok=True)
        metrics_csv_path = self.project_dir / "metrics.csv"
        csv_fields = [
            "epoch", "train_loss",
            "box_map50", "box_map50_95",
            "seg_map50", "seg_map50_95",
            "precision", "recall",
        ]
        with open(metrics_csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()

        self.log_cb(
            f"Starting YOLO training\n"
            f"  Base model  : {self.base_model}\n"
            f"  Epochs      : {self.epochs}\n"
            f"  Batch size  : {self.batch}\n"
            f"  Image size  : {self.imgsz}px\n"
            f"  Device      : {device_str}\n"
            f"  Progress log: {metrics_csv_path}\n"
            f"Training will continue in the background — the GUI will remain responsive.\n"
            f"Do not close this window until training is complete."
        )

        try:
            model = YOLO(self.base_model)
        except Exception as exc:
            msg = str(exc)
            if "Download failure" in msg or "Curl return value" in msg or "ConnectionError" in msg:
                raise RuntimeError(
                    f"Could not load '{self.base_model}': download failed.\n"
                    f"Download the weights manually and use Browse in the Train tab:\n"
                    f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{self.base_model}"
                ) from exc
            raise

        epoch_counter = [0]

        def _on_epoch_end(trainer):
            epoch_counter[0] += 1
            ep = epoch_counter[0]

            loss = getattr(trainer, "loss", None)
            loss_val = float(loss) if loss is not None else float("nan")

            # Ultralytics stores metrics in trainer.metrics after validation
            m = getattr(trainer, "metrics", None)

            def _get(obj, *attrs):
                for a in attrs:
                    obj = getattr(obj, a, None)
                    if obj is None:
                        return float("nan")
                try:
                    return float(obj)
                except Exception:
                    return float("nan")

            box_map50    = _get(m, "box", "map50")
            box_map5095  = _get(m, "box", "map")
            seg_map50    = _get(m, "seg", "map50")
            seg_map5095  = _get(m, "seg", "map")
            precision    = _get(m, "box", "mp")   # mean precision
            recall       = _get(m, "box", "mr")   # mean recall

            def _fmt(v):
                return f"{v:.3f}" if v == v else "n/a"   # nan check

            self.log_cb(
                f"Epoch {ep}/{self.epochs}"
                f"  |  loss {loss_val:.4f}"
                f"  |  detection accuracy (mAP50) {_fmt(seg_map50)}"
                f"  |  precision {_fmt(precision)}"
                f"  |  recall {_fmt(recall)}"
            )
            self.progress_cb(ep, self.epochs)
            self.metrics_cb(ep, loss_val, seg_map50)

            row = {
                "epoch":        ep,
                "train_loss":   f"{loss_val:.4f}",
                "box_map50":    f"{box_map50:.4f}",
                "box_map50_95": f"{box_map5095:.4f}",
                "seg_map50":    f"{seg_map50:.4f}",
                "seg_map50_95": f"{seg_map5095:.4f}",
                "precision":    f"{precision:.4f}",
                "recall":       f"{recall:.4f}",
            }
            with open(metrics_csv_path, "a", newline="") as f:
                csv.DictWriter(f, fieldnames=csv_fields).writerow(row)

        model.add_callback("on_train_epoch_end", _on_epoch_end)

        _orig_stdout = sys.stdout
        sys.stdout = _TeeToLog(_orig_stdout, self.log_cb)
        try:
            model.train(
                data=str(yaml_path),
                epochs=self.epochs,
                batch=self.batch,
                imgsz=self.imgsz,
                device=device_str,
                project=str(self.project_dir),
                name="run",
                exist_ok=True,
                verbose=False,
            )
        finally:
            sys.stdout = _orig_stdout

        best_pt = self.project_dir / "run" / "weights" / "best.pt"

        # ── final per-class validation on best weights ─────────────────────────
        self.log_cb("Running final validation on best weights...")
        best_model = YOLO(str(best_pt))
        val_results = best_model.val(
            data=str(yaml_path),
            device=device_str,
            verbose=False,
            project=str(self.project_dir),
            name="val_best",
            exist_ok=True,
        )

        best_metrics: dict = {}
        try:
            seg = val_results.seg
            names = val_results.names   # {0: 'Spore', 1: 'Vesicle', ...}

            # per-class arrays from ultralytics (one entry per class)
            prec_arr  = seg.p.tolist()   if hasattr(seg, "p")    else []
            rec_arr   = seg.r.tolist()   if hasattr(seg, "r")    else []
            f1_arr    = seg.f1.tolist()  if hasattr(seg, "f1")   else []
            ap50_arr  = seg.ap50.tolist() if hasattr(seg, "ap50") else []

            def _safe(arr, i):
                try: return float(arr[i])
                except Exception: return float("nan")

            per_class = {}
            for i, name in names.items():
                p  = _safe(prec_arr, i)
                r  = _safe(rec_arr, i)
                f1 = _safe(f1_arr, i)
                iou = _safe(ap50_arr, i)   # mAP@50 ≈ IoU@50 for segmentation
                per_class[name] = {
                    "precision": p, "recall": r, "f1": f1, "iou": iou
                }

            fg_vals = list(per_class.values())
            def _mean(key):
                v = [x[key] for x in fg_vals if x[key] == x[key]]
                return float(np.mean(v)) if v else float("nan")

            best_metrics = {
                "mean_precision": _mean("precision"),
                "mean_recall":    _mean("recall"),
                "mean_f1":        _mean("f1"),
                "mean_iou":       _mean("iou"),
                "per_class":      per_class,
            }

            # Append final validation row to CSV
            col_w = max(len(n) for n in names.values()) + 2 if names else 10
            header_line = f"  {'Class':<{col_w}}  Prec    Rec     F1      IoU@50"
            cls_lines = "\n".join(
                f"  {name:<{col_w}}  {v['precision']:.3f}   {v['recall']:.3f}   "
                f"{v['f1']:.3f}   {v['iou']:.3f}"
                for name, v in per_class.items()
            )
            self.log_cb(
                f"Final validation (best weights):\n"
                f"{header_line}\n{cls_lines}"
            )

        except Exception as exc:
            self.log_cb(f"Could not extract per-class metrics: {exc}")

        # Save metadata
        info = {
            "model_type":   "yolo",
            "base_model":   self.base_model,
            "class_names":  class_names,
            "epochs":       self.epochs,
            "batch":        self.batch,
            "imgsz":        self.imgsz,
            "best_weights": str(best_pt),
            "best_metrics": best_metrics,
        }
        (self.project_dir / "training_info.json").write_text(
            json.dumps(info, indent=2)
        )

        # ── test set evaluation on best weights ───────────────────────────────
        test_metrics: dict = {}
        test_yaml_path = yolo_data_dir / "dataset_test.yaml"
        test_img_dir   = yolo_data_dir / "images" / "test"
        test_lbl_dir   = yolo_data_dir / "labels" / "test"

        # Build test split in YOLO format if it exists in ACORN splits
        test_coco_path = self.dataset_dir / "splits" / "test.json"
        if test_coco_path.exists():
            self.log_cb("Preparing test split for YOLO evaluation...")
            test_coco = json.loads(test_coco_path.read_text())
            test_img_dir.mkdir(parents=True, exist_ok=True)
            test_lbl_dir.mkdir(parents=True, exist_ok=True)

            # Reuse the cat_id_to_idx from the train yaml
            ann_coco = json.loads((self.dataset_dir / "annotations.json").read_text())
            _skip2 = {"background", "ignore"}
            valid_cats2 = [c for c in ann_coco.get("categories", [])
                           if c["name"].lower() not in _skip2]
            c2i = {c["id"]: i for i, c in enumerate(valid_cats2)}

            img_anns_t: dict[int, list] = {}
            for ann in test_coco.get("annotations", []):
                img_anns_t.setdefault(ann["image_id"], []).append(ann)

            for img_rec in test_coco.get("images", []):
                src = self.dataset_dir / img_rec["file_name"]
                dst = test_img_dir / src.name
                if src.exists():
                    shutil.copy2(src, dst)
                w, h = img_rec["width"], img_rec["height"]
                lbl = test_lbl_dir / (src.stem + ".txt")
                lines_t = []
                for ann in img_anns_t.get(img_rec["id"], []):
                    if ann["category_id"] not in c2i:
                        continue
                    cls_idx = c2i[ann["category_id"]]
                    seg = ann.get("segmentation", [])
                    flat = seg[0] if seg and seg[0] and len(seg[0]) >= 6 else []
                    if not flat:
                        bx, by, bw, bh = ann["bbox"]
                        flat = [bx, by, bx+bw, by, bx+bw, by+bh, bx, by+bh]
                    norm = [flat[i]/w if i%2==0 else flat[i]/h for i in range(len(flat))]
                    lines_t.append(str(cls_idx) + " " + " ".join(f"{v:.6f}" for v in norm))
                lbl.write_text("\n".join(lines_t))

            test_yaml_path.write_text(
                f"path: {yolo_data_dir.resolve()}\n"
                f"train: images/train\nval: images/val\ntest: images/test\n"
                f"nc: {len(class_names)}\nnames: {class_names}\n"
            )

            self.log_cb("Evaluating best model on held-out test set...")
            try:
                test_results = best_model.val(
                    data=str(test_yaml_path),
                    split="test",
                    device=device_str,
                    verbose=False,
                    project=str(self.project_dir),
                    name="val_test",
                    exist_ok=True,
                )
                seg_t = test_results.seg
                names_t = test_results.names
                prec_t  = seg_t.p.tolist()  if hasattr(seg_t, "p")   else []
                rec_t   = seg_t.r.tolist()  if hasattr(seg_t, "r")   else []
                f1_t    = seg_t.f1.tolist() if hasattr(seg_t, "f1")  else []
                ap50_t  = seg_t.ap50.tolist() if hasattr(seg_t, "ap50") else []

                def _s(arr, i):
                    try: return float(arr[i])
                    except Exception: return float("nan")

                per_class_t = {
                    name: {"precision": _s(prec_t,i), "recall": _s(rec_t,i),
                           "f1": _s(f1_t,i), "iou": _s(ap50_t,i)}
                    for i, name in names_t.items()
                }
                fg_t = list(per_class_t.values())
                def _mn(key):
                    v = [x[key] for x in fg_t if x[key] == x[key]]
                    return float(np.mean(v)) if v else float("nan")
                test_metrics = {
                    "n_images":       len(test_coco.get("images", [])),
                    "mean_precision": _mn("precision"),
                    "mean_recall":    _mn("recall"),
                    "mean_f1":        _mn("f1"),
                    "mean_iou":       _mn("iou"),
                    "per_class":      per_class_t,
                }
                self.log_cb(
                    f"Test set ({test_metrics['n_images']} images):  "
                    f"mF1={test_metrics['mean_f1']:.3f}  mIoU={test_metrics['mean_iou']:.3f}"
                )
            except Exception as exc:
                self.log_cb(f"Test evaluation failed: {exc}")
        else:
            self.log_cb("No test split found — skipping test evaluation.")

        # ── figures ───────────────────────────────────────────────────────────
        from acorn.core.unet_trainer import _save_figures
        _save_figures(
            project_dir=self.project_dir,
            metrics_csv=metrics_csv_path,
            best_metrics=best_metrics,
            test_metrics=test_metrics,
            class_names=class_names,
            log_cb=self.log_cb,
        )

        # Save metadata
        info = {
            "model_type":   "yolo",
            "base_model":   self.base_model,
            "class_names":  class_names,
            "epochs":       self.epochs,
            "batch":        self.batch,
            "imgsz":        self.imgsz,
            "best_weights": str(best_pt),
            "best_metrics": best_metrics,
            "test_metrics": test_metrics,
        }
        (self.project_dir / "training_info.json").write_text(
            json.dumps(info, indent=2)
        )

        # ── human-readable summary report ─────────────────────────────────────
        from acorn.core.unet_trainer import _format_summary_report
        report = _format_summary_report(
            model_label=f"YOLO  {self.base_model}",
            dataset_dir=self.dataset_dir,
            val_metrics=best_metrics,
            test_metrics=test_metrics,
            metrics_csv=metrics_csv_path,
        )
        report_path = self.project_dir / "training_report.txt"
        report_path.write_text(report)
        self.log_cb(f"\n{report}\nReport saved: {report_path}")

        return best_pt
