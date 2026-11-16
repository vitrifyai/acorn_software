"""YOLO and SAM pickers for exact-truth measurement-bias pilot studies."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import numpy as np
import tifffile
import yaml
from PIL import Image

from acorn_sim_common.classical_pickers import _dataset_items, mask_outline


def train_yolo(
    dataset_dir: Path,
    project_dir: Path,
    *,
    base_model: str,
    name: str,
    epochs: int = 30,
    imgsz: int = 256,
    batch: int = 4,
    device: str = "cpu",
    seed: int = 0,
) -> Path:
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    project_dir = Path(project_dir)
    data_yaml = prepare_yolo_rgb_dataset(dataset_dir, project_dir / f"{name}_rgb_data")
    model = YOLO(base_model)
    model.train(
        data=str(data_yaml),
        epochs=int(epochs),
        imgsz=int(imgsz),
        batch=int(batch),
        device=str(device),
        project=str(project_dir),
        name=name,
        exist_ok=True,
        seed=int(seed),
        verbose=False,
    )
    best = project_dir / name / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"YOLO training did not write {best}")
    return best


def predict_yolo(
    dataset_dir: Path,
    weights: Path,
    output_dir: Path,
    *,
    class_name: str,
    imgsz: int = 256,
    conf: float = 0.15,
    iou: float = 0.50,
    device: str = "cpu",
    class_id: int = 0,
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    from ultralytics import YOLO

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    items = _dataset_items(dataset_dir, splits=splits)
    model = YOLO(str(weights))
    written: list[Path] = []
    for item in items:
        rgb = _read_rgb(item.image_path)
        size = int(imgsz)
        if size <= 0:
            # Native resolution: pad bottom/right to the network stride so the
            # letterbox scale is exactly 1 and objects keep the pixel size the
            # model was trained at. Padding adds no image content.
            h, w = rgb.shape[:2]
            ph, pw = (-h) % 32, (-w) % 32
            rgb = np.pad(rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
            size = [h + ph, w + pw]
        results = model.predict(
            source=rgb,
            imgsz=size,
            conf=float(conf),
            iou=float(iou),
            device=str(device),
            verbose=False,
            retina_masks=True,
        )
        predictions = _predictions_from_yolo_result(results[0], item.pixel_size_nm, class_name, class_id=class_id)
        payload = {
            "stem": item.stem,
            "method": "yolo-seg",
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
    _write_manifest(output_dir, "yolo-seg", class_name, dataset_dir, written)
    return written


def prepare_yolo_rgb_dataset(dataset_dir: Path, output_dir: Path) -> Path:
    """Create a YOLO-compatible RGB copy of a simulator dataset.

    The simulators intentionally write single-channel microscopy images. YOLO's
    pretrained segmentation models expect RGB input, so this creates a derived
    training dataset without touching the scientific source exports.
    """

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    source_yaml = dataset_dir / "data.yaml"
    if not source_yaml.exists():
        raise FileNotFoundError(source_yaml)

    data = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    for split_dir in sorted((dataset_dir / "images").iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        out_image_dir = output_dir / "images" / split
        out_label_dir = output_dir / "labels" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for image_path in sorted(split_dir.iterdir()):
            if not image_path.is_file():
                continue
            rgb = _read_rgb(image_path)
            Image.fromarray(rgb).save(out_image_dir / f"{image_path.stem}.png")
            label_path = dataset_dir / "labels" / split / f"{image_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, out_label_dir / label_path.name)

    derived = dict(data)
    derived["path"] = str(output_dir)
    derived["train"] = "images/train"
    derived["val"] = "images/val"
    if (output_dir / "images" / "test").exists():
        derived["test"] = "images/test"
    (output_dir / "data.yaml").write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
    return output_dir / "data.yaml"


def predict_sam_points(
    dataset_dir: Path,
    output_dir: Path,
    *,
    class_name: str,
    checkpoint: Path,
    model_type: str = "vit_h",
    device: str = "cpu",
    splits: tuple[str, ...] | None = None,
    max_area_fraction: float = 0.05,
) -> list[Path]:
    """Prompt SAM once at each simulated truth center.

    This intentionally measures SAM boundary behavior given a correct object
    prompt, so B_det is zero by construction and the bias is all delineation.
    It is not an independent detection score.

    SAM returns three nested candidates per point and its own score usually
    favours the largest, which on a sparse field is the whole image. The
    candidate is chosen by score among those covering at most
    ``max_area_fraction`` of the image; the rule uses no truth size.
    """

    import torch
    from segment_anything import SamPredictor, sam_model_registry

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    items = _dataset_items(dataset_dir, splits=splits)
    sam = sam_model_registry[model_type](checkpoint=str(checkpoint))
    sam.to(device=device)
    predictor = SamPredictor(sam)
    written: list[Path] = []
    try:
        for item in items:
            image = _read_rgb(item.image_path)
            predictor.set_image(image)
            truth_payload = json.loads(item.truth_path.read_text(encoding="utf-8"))
            predictions = []
            for obj in truth_payload.get("objects", []):
                if obj.get("fully_in_frame") is False:
                    continue
                point = np.asarray([[float(obj["cx_px"]), float(obj["cy_px"])]], dtype=np.float32)
                labels = np.asarray([1], dtype=np.int32)
                with torch.inference_mode():
                    masks, scores, _logits = predictor.predict(
                        point_coords=point,
                        point_labels=labels,
                        multimask_output=True,
                    )
                idx = _choose_point_mask(masks, scores, max_area_fraction=max_area_fraction, point_xy=point[0])
                pred = _prediction_from_mask(
                    masks[idx].astype(bool),
                    item.pixel_size_nm,
                    class_name=class_name,
                    score=float(scores[idx]),
                    source="sam-point-truth-center",
                )
                if pred is not None:
                    pred["truth_id"] = str(obj.get("id", len(predictions) + 1))
                    pred["prompt_cx_px"] = float(obj["cx_px"])
                    pred["prompt_cy_px"] = float(obj["cy_px"])
                    predictions.append(pred)
            payload = {
                "stem": item.stem,
                "method": "sam-point-truth-center",
                "class_name": class_name,
                "pixel_size_nm": item.pixel_size_nm,
                "source_image": str(item.image_path),
                "checkpoint": str(checkpoint),
                "prompt_policy": "one positive point at each simulated truth center; boundary test, not independent detection",
                "predictions": predictions,
            }
            out_path = output_dir / item.split / f"{item.stem}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            written.append(out_path)
    finally:
        del predictor
        del sam
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    _write_manifest(output_dir, "sam-point-truth-center", class_name, dataset_dir, written)
    return written


def predict_sam3_text(
    dataset_dir: Path,
    output_dir: Path,
    *,
    class_name: str,
    prompt: str,
    confidence: float = 0.5,
    checkpoint: Path | None = None,
    device: str = "cuda",
    splits: tuple[str, ...] | None = None,
) -> list[Path]:
    """Segment every image with one literal SAM 3 text prompt.

    The prompt is fixed in advance and sent without vocabulary remapping, so
    it is a stated method parameter rather than something tuned per image.
    """

    import torch
    from PIL import Image as _Image

    from acorn.core.sam_predictor import SAMPredictor

    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    predictor = SAMPredictor(backend="sam3", checkpoint_path=checkpoint, device=device)
    predictor.load_model()
    proc = predictor._sam3_processor
    proc.set_confidence_threshold(float(confidence))
    written: list[Path] = []
    for item in _dataset_items(dataset_dir, splits=splits):
        rgb = _read_rgb(item.image_path)
        with torch.inference_mode():
            state = proc.set_image(_Image.fromarray(rgb))
            out = proc.set_text_prompt(prompt, state)
        predictions = []
        masks = out.get("masks")
        scores = out.get("scores")
        for k, m in enumerate([] if masks is None else masks):
            mask = np.asarray(m.squeeze().float().cpu().numpy() > 0.5, dtype=bool)
            score = None if scores is None else float(scores[k])
            pred = _prediction_from_mask(mask, item.pixel_size_nm, class_name=class_name, score=score, source="sam3-text")
            if pred is not None:
                predictions.append(pred)
        payload = {
            "stem": item.stem,
            "method": "sam3-text",
            "class_name": class_name,
            "prompt": prompt,
            "confidence": float(confidence),
            "pixel_size_nm": item.pixel_size_nm,
            "source_image": str(item.image_path),
            "predictions": predictions,
        }
        out_path = output_dir / item.split / f"{item.stem}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(out_path)
    _write_manifest(output_dir, "sam3-text", class_name, dataset_dir, written)
    return written


def _choose_point_mask(
    masks: np.ndarray,
    scores: np.ndarray,
    *,
    max_area_fraction: float,
    point_xy: np.ndarray | None = None,
) -> int:
    areas = np.asarray([np.count_nonzero(m) for m in masks], dtype=float)
    allowed = areas <= max_area_fraction * masks[0].size
    if point_xy is not None:
        x, y = (int(round(float(v))) for v in point_xy)
        h, w = masks[0].shape
        if 0 <= y < h and 0 <= x < w:
            covers = np.asarray([bool(m[y, x]) for m in masks])
            if (allowed & covers).any():
                allowed = allowed & covers
    if not allowed.any():
        return int(np.argmin(areas))
    return int(np.flatnonzero(allowed)[np.argmax(np.asarray(scores)[allowed])])


def _predictions_from_yolo_result(result, pixel_size_nm: float, class_name: str, *, class_id: int = 0) -> list[dict]:
    predictions: list[dict] = []
    if result.masks is None:
        return predictions
    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    boxes = result.boxes
    scores = boxes.conf.detach().cpu().numpy().tolist() if boxes is not None and boxes.conf is not None else [None] * len(masks)
    classes = boxes.cls.detach().cpu().numpy().astype(int).tolist() if boxes is not None and boxes.cls is not None else [class_id] * len(masks)
    for i, mask in enumerate(masks):
        if int(classes[i]) != int(class_id):
            continue
        pred = _prediction_from_mask(
            mask,
            pixel_size_nm,
            class_name=class_name,
            score=None if scores[i] is None else float(scores[i]),
            source="yolo-seg",
        )
        if pred is not None:
            predictions.append(pred)
    return predictions


def _prediction_from_mask(
    mask: np.ndarray,
    pixel_size_nm: float,
    *,
    class_name: str,
    score: float | None,
    source: str,
) -> dict | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    area_px = float(xs.size)
    return {
        "id": 0,
        "class_name": class_name,
        "cx_px": float(xs.mean()),
        "cy_px": float(ys.mean()),
        "diameter_nm": float(2.0 * math.sqrt(area_px / math.pi) * pixel_size_nm),
        "area_px": area_px,
        "polygon": mask_outline(mask),
        "score": score,
        "source": source,
    }


def _read_rgb(path: Path) -> np.ndarray:
    try:
        arr = tifffile.imread(path)
    except Exception:
        arr = np.asarray(Image.open(path))
    if arr.ndim == 2:
        arr = _uint8(arr)
        return np.repeat(arr[..., None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    return _uint8(arr)


def _uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.percentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return (np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8)


def _write_manifest(output_dir: Path, method: str, class_name: str, dataset_dir: Path, paths: list[Path]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "method": method,
                "class_name": class_name,
                "source_dataset": str(dataset_dir),
                "n_images": len(paths),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run YOLO or SAM pickers on an exact-truth simulator dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    ytrain = sub.add_parser("train-yolo")
    ytrain.add_argument("--dataset", required=True, type=Path)
    ytrain.add_argument("--project", required=True, type=Path)
    ytrain.add_argument("--base-model", required=True)
    ytrain.add_argument("--name", required=True)
    ytrain.add_argument("--epochs", type=int, default=30)
    ytrain.add_argument("--imgsz", type=int, default=256)
    ytrain.add_argument("--batch", type=int, default=4)
    ytrain.add_argument("--device", default="cpu")
    ytrain.add_argument("--seed", type=int, default=0)

    ypred = sub.add_parser("predict-yolo")
    ypred.add_argument("--dataset", required=True, type=Path)
    ypred.add_argument("--weights", required=True, type=Path)
    ypred.add_argument("--output", required=True, type=Path)
    ypred.add_argument("--class-name", required=True)
    ypred.add_argument("--imgsz", type=int, default=256, help="0 = native resolution (no rescaling).")
    ypred.add_argument("--conf", type=float, default=0.15)
    ypred.add_argument("--iou", type=float, default=0.50)
    ypred.add_argument("--device", default="cpu")
    ypred.add_argument("--class-id", type=int, default=0)
    ypred.add_argument("--splits", nargs="+", default=None)

    sam = sub.add_parser("predict-sam-points")
    sam.add_argument("--dataset", required=True, type=Path)
    sam.add_argument("--output", required=True, type=Path)
    sam.add_argument("--class-name", required=True)
    sam.add_argument("--checkpoint", required=True, type=Path)
    sam.add_argument("--model-type", default="vit_h")
    sam.add_argument("--device", default="cpu")
    sam.add_argument("--splits", nargs="+", default=None)
    sam.add_argument("--max-area-fraction", type=float, default=0.05)

    sam3 = sub.add_parser("predict-sam3-text")
    sam3.add_argument("--dataset", required=True, type=Path)
    sam3.add_argument("--output", required=True, type=Path)
    sam3.add_argument("--class-name", required=True)
    sam3.add_argument("--prompt", required=True, help="Literal text prompt, sent unchanged.")
    sam3.add_argument("--confidence", type=float, default=0.5)
    sam3.add_argument("--checkpoint", type=Path, default=None)
    sam3.add_argument("--device", default="cuda")
    sam3.add_argument("--splits", nargs="+", default=None)

    args = parser.parse_args(argv)
    if args.cmd == "train-yolo":
        best = train_yolo(
            args.dataset,
            args.project,
            base_model=args.base_model,
            name=args.name,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            seed=args.seed,
        )
        print(best)
        return 0
    if args.cmd == "predict-yolo":
        paths = predict_yolo(
            args.dataset,
            args.weights,
            args.output,
            class_name=args.class_name,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            class_id=args.class_id,
            splits=None if args.splits is None else tuple(args.splits),
        )
        print(f"Wrote {len(paths)} YOLO prediction files to {args.output}")
        return 0
    if args.cmd == "predict-sam-points":
        paths = predict_sam_points(
            args.dataset,
            args.output,
            class_name=args.class_name,
            checkpoint=args.checkpoint,
            model_type=args.model_type,
            device=args.device,
            splits=None if args.splits is None else tuple(args.splits),
            max_area_fraction=args.max_area_fraction,
        )
        print(f"Wrote {len(paths)} SAM point-prompt prediction files to {args.output}")
        return 0
    if args.cmd == "predict-sam3-text":
        paths = predict_sam3_text(
            args.dataset,
            args.output,
            class_name=args.class_name,
            prompt=args.prompt,
            confidence=args.confidence,
            checkpoint=args.checkpoint,
            device=args.device,
            splits=None if args.splits is None else tuple(args.splits),
        )
        print(f"Wrote {len(paths)} SAM 3 text-prompt prediction files to {args.output}")
        return 0
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
