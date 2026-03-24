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
        self._panel     = None
        self._sem_panel = None
        self._thread     = None
        self._sem_thread = None
        context.image_loaded.connect(self._on_image_loaded)
        context.annotations_changed.connect(self._on_annotations_changed)

    def _on_image_loaded(self, img) -> None:
        if self._panel is not None:
            self._panel.set_pixel_size(img.pixel_size)
        if self._sem_panel is not None:
            self._sem_panel.set_pixel_size(img.pixel_size)

    def _on_annotations_changed(self, store) -> None:
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
        if self._panel is not None:
            self._panel.refresh_labels(labels)
        if self._sem_panel is not None:
            self._sem_panel.refresh_labels(labels)

    def create_panel(self) -> QWidget:
        from PyQt6.QtWidgets import QTabWidget
        from acorn_analysis.panel import AnalysisPanel
        from acorn_analysis.sem_panel import SEMPanel

        self._panel = AnalysisPanel()
        self._panel.analysis_requested.connect(self._on_analysis_requested)

        self._sem_panel = SEMPanel()
        self._sem_panel.sem_requested.connect(self._on_sem_requested)
        self._sem_panel.train_requested.connect(self._on_sem_train_requested)
        self._sem_panel.set_calibrate_callback(self._do_sem_calibrate)
        self._sem_panel.pick_flat_region_requested.connect(self._on_pick_flat_region)

        tabs = QTabWidget()
        tabs.addTab(self._panel,     "Mask-Based")
        tabs.addTab(self._sem_panel, "SEM 3D")
        return tabs

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

    # ------------------------------------------------------------------
    # SEM 3D analysis
    # ------------------------------------------------------------------

    def _on_pick_flat_region(self) -> None:
        """Start canvas rect-pick mode; result routed to _on_flat_region_picked."""
        cw = self._context.canvas_widget()
        if cw is None:
            return
        # Connect one-shot; disconnect after first pick in the handler
        try:
            cw.flat_region_picked.disconnect(self._on_flat_region_picked)
        except Exception:
            pass
        cw.flat_region_picked.connect(self._on_flat_region_picked)
        cw.start_flat_region_pick()
        self._context.set_status(
            "SEM calibration: draw a rectangle on a flat substrate area, then release."
        )

    def _on_flat_region_picked(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Extract the selected region and calibrate I_bg / eta0."""
        cw = self._context.canvas_widget()
        if cw is not None:
            try:
                cw.flat_region_picked.disconnect(self._on_flat_region_picked)
            except Exception:
                pass

        img = self._context.current_image
        if img is None or self._sem_panel is None:
            return

        import numpy as np
        from acorn.analysis.sem_physics import estimate_params_from_image

        raw = img.raw.astype(np.float32)
        ry0, ry1 = max(0, int(y0)), min(raw.shape[0], int(y1) + 1)
        rx0, rx1 = max(0, int(x0)), min(raw.shape[1], int(x1) + 1)
        if ry1 <= ry0 or rx1 <= rx0:
            return

        flat_mask = np.zeros(raw.shape[:2], dtype=bool)
        flat_mask[ry0:ry1, rx0:rx1] = True
        p = estimate_params_from_image(raw, flat_region_mask=flat_mask)
        self._sem_panel.apply_calibration(p.I_bg, p.eta0)
        self._context.set_status(
            f"SEM calibration from selected region: I_bg={p.I_bg:.4g}  eta0={p.eta0:.4g}"
        )

    def _do_sem_calibrate(self) -> None:
        """Auto-estimate I_bg and eta0 from current image + annotations."""
        img = self._context.current_image
        if img is None or self._sem_panel is None:
            return
        import numpy as np
        from acorn.analysis.sem_physics import estimate_params_from_image
        store = self._context.annotation_store
        mask  = None
        if store is not None:
            # Build combined mask from all ROIs on this image
            try:
                import cv2
                H, W = img.raw.shape[:2]
                combined = np.zeros((H, W), dtype=np.uint8)
                for ann in store:
                    if getattr(ann, "type", None) == "roi":
                        verts = getattr(ann, "vertices", [])
                        if len(verts) >= 3:
                            pts = np.array(verts, dtype=np.int32)
                            cv2.fillPoly(combined, [pts], 1)
                mask = combined.astype(bool)
            except Exception:
                mask = None
        p = estimate_params_from_image(img.raw.astype(float), mask)
        self._sem_panel.apply_calibration(p.I_bg, p.eta0)
        self._context.set_status(
            f"SEM calibration: I_bg={p.I_bg:.4g}  eta0={p.eta0:.4g}"
        )

    def _on_sem_requested(self, config: dict) -> None:
        from acorn_analysis.sem_thread import SEMAnalysisThread

        selected_labels = set(config["selected_labels"])
        px_nm           = config["pixel_size_nm"]
        out_dir_str     = config.get("output_dir", "")

        items: list[dict] = []
        paths = self._context.image_paths
        idx   = self._context.current_image_index
        store = self._context.annotation_store

        def _collect(anns, img_name, img_path, px):
            for ann in anns:
                if getattr(ann, "type", None) != "roi":
                    continue
                lbl = getattr(ann, "label", "")
                if lbl not in selected_labels:
                    continue
                verts = getattr(ann, "vertices", [])
                if len(verts) >= 3:
                    items.append({
                        "vertices":     [list(v) for v in verts],
                        "label":        lbl,
                        "image_name":   img_name,
                        "image_path":   img_path,
                        "pixel_size_nm": px,
                    })

        if idx >= 0 and store is not None:
            px = self._context.pixel_size_for_index(idx)
            _collect(store, paths[idx].stem, str(paths[idx]), px)
        for i, state_list in self._context.all_annotation_states.items():
            if i == idx:
                continue
            if i < len(paths):
                px = self._context.pixel_size_for_index(i)
                _collect(state_list, paths[i].stem, str(paths[i]), px)

        if not items:
            QMessageBox.information(
                None, "SEM 3D Analysis",
                "No ROI annotations found for the selected label(s)."
            )
            return

        if not out_dir_str and idx >= 0:
            from datetime import datetime
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ip = paths[idx]
            out_dir_str = str(ip.parent / "acorn_sem_analysis" / f"{ip.stem}_{ts}")

        det_cfg = {k: config[k] for k in
                   ("alpha_deg", "phi_deg", "I_bg", "eta0", "lam", "learn_detector")}
        sfs_cfg = {"n_iters": config["n_iters"],
                   "smoothness": config["smoothness"],
                   "lr": config["lr"]}
        nn_cfg  = {"use_nn": config["use_nn"],
                   "checkpoint": config["checkpoint"]}

        self._sem_panel.set_running(True)
        self._sem_thread = SEMAnalysisThread(
            items          = items,
            detector_config = det_cfg,
            sfs_config     = sfs_cfg,
            nn_config      = nn_cfg,
            output_dir     = out_dir_str,
        )
        self._sem_thread.progress.connect(self._sem_panel.show_progress)
        self._sem_thread.finished.connect(self._on_sem_finished)
        self._sem_thread.error.connect(self._on_sem_error)
        self._sem_thread.start()

    def _on_sem_finished(self, df, out_dir_str: str) -> None:
        self._sem_panel.set_running(False)
        out = Path(out_dir_str) if out_dir_str else None
        self._sem_panel.show_results(df, out)
        self._context.set_status(
            f"SEM 3D analysis complete — {len(df)} particles"
            + (f"  |  results in {out}" if out else "")
        )

    def _on_sem_error(self, msg: str) -> None:
        self._sem_panel.set_running(False)
        QMessageBox.critical(None, "SEM Analysis error", msg)

    def _on_sem_train_requested(self, config: dict) -> None:
        from acorn_analysis.sem_thread import SEMTrainThread
        from acorn.analysis.sem_physics import DetectorParams

        if not config.get("output_dir"):
            QMessageBox.warning(None, "Train U-Net", "Choose an output directory first.")
            return

        dp = None
        if self._sem_panel is not None:
            dp = DetectorParams(
                alpha_deg = self._sem_panel._alpha.value(),
                phi_deg   = self._sem_panel._phi.value(),
                lam       = self._sem_panel._lam.value(),
            )

        self._sem_train_thread = SEMTrainThread(config, detector_params=dp)
        self._sem_train_thread.progress.connect(
            lambda pct, msg: self._context.set_status(f"SEM train: {msg}")
        )
        self._sem_train_thread.finished.connect(self._on_sem_train_done)
        self._sem_train_thread.error.connect(
            lambda msg: QMessageBox.critical(None, "SEM Train error", msg)
        )
        self._sem_train_thread.start()
        self._context.set_status("SEM U-Net training started…")

    def _on_sem_train_done(self, ckpt_path: str) -> None:
        self._context.set_status(f"SEM U-Net training complete: {ckpt_path}")
        if self._sem_panel is not None:
            self._sem_panel._ckpt_edit.setText(ckpt_path)
            self._sem_panel._checkpoint = ckpt_path
            self._sem_panel._use_nn.setEnabled(True)
            self._sem_panel._use_nn.setChecked(True)

    def teardown(self) -> None:
        if self._thread and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(3000)
        if self._sem_thread and self._sem_thread.isRunning():
            self._sem_thread.stop()
            self._sem_thread.wait(3000)
