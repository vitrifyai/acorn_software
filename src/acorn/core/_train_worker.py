"""Standalone training worker — launched as a detached subprocess by the GUI.

Usage (internal):
    python -m acorn.core._train_worker <config_json_path>

Writes log lines to stdout. Special-format lines are parsed by the GUI:
    PROGRESS:{epoch}/{total}
    METRIC:{epoch},{loss},{metric}
    DONE:{model_path}
    ERROR:{message}
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m acorn.core._train_worker <config.json>", file=sys.stderr)
        sys.exit(1)

    config = json.loads(Path(sys.argv[1]).read_text())
    model_type = config["model_type"]

    def log(msg: str) -> None:
        print(msg, flush=True)

    def progress(epoch: int, total: int) -> None:
        print(f"PROGRESS:{epoch}/{total}", flush=True)

    def metrics(epoch: int, loss: float, metric: float) -> None:
        print(f"METRIC:{epoch},{loss},{metric}", flush=True)

    if model_type == "yolo":
        from acorn.core.yolo_trainer import YOLOTrainer
        trainer = YOLOTrainer(
            dataset_dir=config["dataset_dir"],
            base_model=config["base_model"],
            epochs=config["epochs"],
            batch=config["batch"],
            devices=config["devices"],
            imgsz=config.get("imgsz", 640),
            log_cb=log,
            progress_cb=progress,
            metrics_cb=metrics,
        )
    elif model_type == "unet":
        from acorn.core.unet_trainer import UNetTrainer
        trainer = UNetTrainer(
            dataset_dir=config["dataset_dir"],
            arch=config.get("arch", "Unet"),
            encoder=config.get("encoder", "resnet34"),
            lr=config.get("lr", 1e-4),
            epochs=config["epochs"],
            batch=config["batch"],
            devices=config["devices"],
            imgsz=config.get("imgsz", 512),
            log_cb=log,
            progress_cb=progress,
            metrics_cb=metrics,
        )
    else:
        print(f"ERROR:Unknown model type: {model_type}", flush=True)
        sys.exit(1)

    try:
        result = trainer.train()
        print(f"DONE:{result}", flush=True)
    except Exception as exc:
        print(f"ERROR:{exc}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
