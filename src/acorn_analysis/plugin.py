"""Analysis plugin — surface area estimation and population statistics."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class AnalysisPlugin(AcornPlugin):
    TAB_LABEL = "Analysis"
    PLUGIN_ID = "acorn_analysis"

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._thread = None
        context.image_loaded.connect(self._on_image_loaded)
        context.annotations_changed.connect(self._on_annotations_changed)

    def _on_image_loaded(self, img) -> None:
        if self._panel is not None:
            self._panel.set_pixel_size(img.pixel_size)

    def _on_annotations_changed(self, store) -> None:
        if self._panel is None:
            return
        labels: list[str] = []
        for ann in store:
            if getattr(ann, "type", None) == "roi":
                labels.append(getattr(ann, "label", ""))
        idx = self._context.current_image_index
        for i, state_list in self._context.all_annotation_states.items():
            if i == idx:
                continue
            for ann in state_list:
                if getattr(ann, "type", None) == "roi":
                    labels.append(getattr(ann, "label", ""))
        self._panel.refresh_labels(labels)

    def create_panel(self) -> QWidget:
        from acorn_analysis.panel import AnalysisPanel
        self._panel = AnalysisPanel()
        self._panel.analysis_requested.connect(self._on_analysis_requested)
        return self._panel

    def _on_analysis_requested(self, config: dict) -> None:
        from acorn_analysis.thread import AnalysisThread

        mode             = config["mode"]
        selected_labels  = set(config["selected_labels"])
        pixel_size_nm    = config["pixel_size_nm"]
        pixel_size_unc   = config["pixel_size_uncertainty_nm"]
        out_dir_str      = config["output_dir"]

        if mode not in ("folder",) and pixel_size_nm <= 0:
            QMessageBox.warning(
                None, "Analysis",
                "Pixel size is 0. Set a valid pixel size before running analysis."
            )
            return

        items: list[dict] = []

        def _collect(store, img_name: str, img_path: str, px_nm: float) -> None:
            for ann in store:
                if getattr(ann, "type", None) != "roi":
                    continue
                lbl = getattr(ann, "label", "")
                if lbl not in selected_labels:
                    continue
                verts = getattr(ann, "vertices", [])
                if len(verts) >= 3:
                    items.append({
                        "vertices": [list(v) for v in verts],
                        "label": lbl,
                        "image_name": img_name,
                        "image_path": img_path,
                        "pixel_size_nm": px_nm,
                    })

        paths = self._context.image_paths
        idx   = self._context.current_image_index
        store = self._context.annotation_store

        if mode == "single":
            px = self._context.pixel_size_for_index(idx)
            if idx >= 0 and store is not None:
                _collect(store, paths[idx].stem, str(paths[idx]), px)

        elif mode == "batch":
            if idx >= 0 and store is not None:
                px = self._context.pixel_size_for_index(idx)
                _collect(store, paths[idx].stem, str(paths[idx]), px)
            for i, state_list in self._context.all_annotation_states.items():
                if i == idx:
                    continue
                if i < len(paths):
                    px = self._context.pixel_size_for_index(i)
                    _collect(state_list, paths[i].stem, str(paths[i]), px)

        elif mode == "folder":
            from pathlib import Path as _Path
            from acorn.core.annotations import AnnotationStore
            folder_items = config.get("folder_items", [])
            for fi in folder_items:
                fpath = _Path(fi["path"])
                px = float(fi.get("pixel_size_nm") or pixel_size_nm or 1.0)
                sidecar = fpath.parent / f".{fpath.stem}.acorn.json"
                if not sidecar.exists():
                    continue
                try:
                    raw = json.loads(sidecar.read_text())
                    ann_data = raw.get("annotations", raw) if isinstance(raw, dict) else raw
                    s = AnnotationStore.from_json(json.dumps(ann_data))
                    _collect(list(s), fpath.stem, str(fpath), px)
                except Exception:
                    continue

        if not items:
            QMessageBox.information(
                None, "Analysis",
                "No ROI annotations found for the selected label(s).\n"
                "Add polygon annotations via the Annotate, SAM, YOLO, or UNet tabs."
            )
            return

        if not out_dir_str and idx >= 0 and mode != "folder":
            from datetime import datetime
            ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
            img_path = paths[idx]
            out_dir_str = str(img_path.parent / "acorn_analysis" / f"{img_path.stem}_{ts}")
        elif not out_dir_str and mode == "folder":
            from datetime import datetime
            folder_path = config.get("folder_path", "")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_dir_str = str(_Path(folder_path) / "acorn_analysis" / ts) if folder_path else ""

        self._panel.set_running(True)
        self._thread = AnalysisThread(
            items=items,
            pixel_size_nm=pixel_size_nm,
            pixel_size_uncertainty_nm=pixel_size_unc,
            output_dir=out_dir_str,
            method=config.get("method", "auto"),
            compound_mode=config.get("compound_mode", "separate"),
        )
        self._thread.progress.connect(self._panel.show_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _on_finished(self, df, stats_dict, out_dir_str: str) -> None:
        self._panel.set_running(False)
        out = Path(out_dir_str) if out_dir_str else None
        self._panel.show_results(df, stats_dict, out)
        msg = f"Analysis complete — {len(df)} particles"
        if out:
            msg += f"  |  results saved to {out}"
        self._context.set_status(msg)

    def _on_error(self, msg: str) -> None:
        self._panel.set_running(False)
        QMessageBox.critical(None, "Analysis error", msg)

    def teardown(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
