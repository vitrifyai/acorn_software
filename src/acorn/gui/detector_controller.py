"""
YOLO and UNet: loading a model, running it on one image or a whole folder,
and turning its output into annotations the user can accept or reject.

Both share the SAM worker thread and the same accept/reject flow, so they live
together rather than in two near-identical files.
"""
from __future__ import annotations

import json
from pathlib import Path

from acorn.core.annotations import ROIAnnotation
from acorn.gui.threads import SAMThread

class DetectorControllerMixin:
    """Mixed into MainWindow — these methods use its state directly."""

    def _yolo_busy(self) -> bool:
        return self._yolo_thread is not None and self._yolo_thread.isRunning()
    def _on_yolo_load_model(self, model_path: str) -> None:
        if self._yolo_busy():
            return
        from acorn.core.yolo_predictor import YOLOPredictor
        predictor = YOLOPredictor()
        self._yolo_panel.set_model_status("Loading…", loaded=False)
        self._seg_panel.set_loaded("yolo", False)

        def _run():
            predictor.load_model(model_path)
            return predictor

        def _done(p):
            self._yolo_predictor = p
            from acorn.core import provenance as _prov
            self._yolo_source_model = _prov.source_model(model_path)
            seg_note = " (seg)" if p.is_seg else ""
            self._yolo_panel.set_model_status(
                f"Loaded{seg_note}: {model_path}", loaded=True
            )
            self._seg_panel.set_loaded("yolo", True)
            self._statusbar.showMessage(f"YOLO model loaded: {model_path}")

        def _err(msg):
            self._yolo_panel.set_model_status(f"Load failed: {msg}", loaded=False)
            self._seg_panel.set_loaded("yolo", False)

        self._yolo_thread = SAMThread(_run, self)
        self._yolo_thread.finished.connect(_done)
        self._yolo_thread.error.connect(_err)
        self._yolo_thread.start()
    def _on_yolo_detect(self) -> None:
        self._run_yolo(segmentation=False)
    def _on_yolo_detect_seg(self) -> None:
        self._run_yolo(segmentation=True)
    def _run_yolo(self, segmentation: bool, on_complete=None) -> None:
        if self._yolo_busy():
            self._yolo_panel.set_status("YOLO is running — please wait.")
            if on_complete is not None:
                on_complete()
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._yolo_panel.set_status("No image loaded.")
            if on_complete is not None:
                on_complete()
            return
        if self._yolo_predictor is None or not self._yolo_predictor.is_loaded:
            self._yolo_panel.set_status("Load a YOLO model first.")
            if on_complete is not None:
                on_complete()
            return

        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm = apply_contrast(img.raw, self._contrast_panel.params())
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        conf = self._yolo_panel.conf_thresh
        iou  = self._yolo_panel.iou_thresh
        self._yolo_panel.set_status("Running…")

        def _run():
            if segmentation:
                return self._yolo_predictor.detect_and_segment(
                    img8, conf_thresh=conf, iou_thresh=iou
                )
            return self._yolo_predictor.detect(img8, conf_thresh=conf, iou_thresh=iou)

        def _done(detections):
            self._last_yolo_detections = detections
            self._add_yolo_detections_to_store(detections, segmentation)
            self._yolo_panel.set_status(
                f"{len(detections)} detection(s). Undo unwanted, then Accept All."
            )
            self._statusbar.showMessage(f"YOLO: {len(detections)} detection(s).")
            if detections:
                self._report_clu(f"YOLO produced {len(detections)} detection(s), added as "
                                 "pending annotations. Tell the user to Accept All to keep them.")
            else:
                self._report_clu("YOLO produced 0 detections — nothing was added. Suggest the "
                                 "user lower the confidence threshold or check the model. Do NOT "
                                 "claim detections were made.")
            if on_complete is not None:
                on_complete()

        def _err(msg):
            self._yolo_panel.set_status(f"Error: {msg}")
            self._report_clu(f"YOLO detection failed: {msg}")
            if on_complete is not None:
                on_complete()

        self._yolo_thread = SAMThread(_run, self)
        self._yolo_thread.finished.connect(_done)
        self._yolo_thread.error.connect(_err)
        self._yolo_thread.start()
    def _add_yolo_detections_to_store(
        self, detections: list, use_masks: bool
    ) -> None:
        canvas = self._canvas_widget.canvas
        store  = canvas.store
        label  = self._yolo_panel.label
        color  = "#4dbb78"
        self._pending_yolo_anns.clear()
        canvas._loading = True
        try:
            has_masks = use_masks and any("mask" in d for d in detections)
            with self._pred_context("yolo"):
                if has_masks:
                    from acorn.core.yolo_predictor import masks_to_roi_annotations
                    n = masks_to_roi_annotations(detections, store, label=label, color=color)
                    for _ in range(n):
                        self._pending_yolo_anns.append(True)
                else:
                    from acorn.core.yolo_predictor import boxes_to_roi_annotations
                    n = boxes_to_roi_annotations(
                        detections, store, label=label, color=color,
                        as_rectangles=self._yolo_panel.as_rectangles,
                    )
                    for _ in range(n):
                        self._pending_yolo_anns.append(True)
        finally:
            canvas._loading = False
        # Compute real areas for ROIs added with area_nm2=0.0
        px = self._engine.pixel_size
        if px > 0:
            from acorn.core.annotations import ROIAnnotation as _ROI
            from acorn.core.measurements import polygon_area_nm2 as _poly_area
            for ann in store:
                if isinstance(ann, _ROI) and ann.area_nm2 == 0.0 and ann.vertices:
                    ann.area_nm2 = _poly_area(ann.vertices, px)
        if canvas.renderer is not None:
            canvas.renderer.render_noblit(canvas.store, canvas)
        else:
            canvas.fig.canvas.draw_idle()
    def _on_yolo_accept(self) -> None:
        self._pending_yolo_anns.clear()
        self._yolo_panel.set_status("Detections accepted as ROI annotations.")
    def _on_yolo_reject(self) -> None:
        self._remove_pending_annotations(self._pending_yolo_anns)
        self._pending_yolo_anns.clear()
        self._yolo_panel.set_status("YOLO detections removed.")
    def _on_yolo_batch(self) -> None:
        """'Run on ALL Images' button — batch the loaded YOLO model over the dataset."""
        self._start_batch_model(
            {"label": self._yolo_panel.label, "segmentation": False,
             "skip_annotated": True, "queue_after": False},
            "yolo",
        )
    def _unet_busy(self) -> bool:
        return self._unet_thread is not None and self._unet_thread.isRunning()
    def _on_unet_load_model(
        self, arch: str, encoder: str, in_channels: int,
        n_classes: int, ckpt_path: str,
    ) -> None:
        if self._unet_busy():
            return
        from acorn.core.unet_predictor import UNetPredictor
        tile_size = self._unet_panel.tile_size
        predictor = UNetPredictor(
            architecture=arch, encoder=encoder,
            in_channels=in_channels, n_classes=n_classes,
            tile_size=tile_size,
        )
        self._unet_panel.set_model_status("Loading…", loaded=False)
        self._seg_panel.set_loaded("unet", False)

        def _run():
            predictor.load_model(ckpt_path)
            return predictor

        def _done(p):
            self._unet_predictor = p
            from acorn.core import provenance as _prov
            self._unet_source_model = _prov.source_model(ckpt_path)
            self._unet_panel.set_model_status(
                f"Loaded ({arch}/{encoder}, {in_channels}ch, {n_classes} cls)",
                loaded=True,
            )
            self._seg_panel.set_loaded("unet", True)
            self._statusbar.showMessage(f"UNet model loaded: {ckpt_path}")

        def _err(msg):
            self._unet_panel.set_model_status(f"Load failed: {msg}", loaded=False)
            self._seg_panel.set_loaded("unet", False)

        self._unet_thread = SAMThread(_run, self)
        self._unet_thread.finished.connect(_done)
        self._unet_thread.error.connect(_err)
        self._unet_thread.start()
    def _on_unet_segment(self, on_complete=None) -> None:
        if self._unet_busy():
            self._unet_panel.set_status("UNet is running — please wait.")
            if on_complete is not None:
                on_complete()
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._unet_panel.set_status("No image loaded.")
            if on_complete is not None:
                on_complete()
            return
        if self._unet_predictor is None or not self._unet_predictor.is_loaded:
            self._unet_panel.set_status("Load a UNet model first.")
            if on_complete is not None:
                on_complete()
            return

        from acorn.core.contrast import apply_contrast
        import numpy as np
        norm  = apply_contrast(img.raw, self._contrast_panel.params())
        img8  = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        threshold = self._unet_panel.threshold
        fg_class  = self._unet_panel.foreground_class
        min_area  = self._unet_panel.min_area
        self._unet_panel.set_status("Running segmentation…")

        def _run():
            return self._unet_predictor.predict(
                img8, threshold=threshold,
                foreground_class=fg_class, min_area=min_area,
            )

        def _done(masks):
            self._add_unet_masks_to_store(masks)
            self._unet_panel.set_status(
                f"{len(masks)} mask(s) found. Undo unwanted, then Accept All."
            )
            self._statusbar.showMessage(f"UNet: {len(masks)} instance mask(s).")
            if masks:
                self._report_clu(f"UNet produced {len(masks)} mask(s), added as pending "
                                 "annotations. Tell the user to Accept All to keep them.")
            else:
                self._report_clu("UNet produced 0 masks — nothing was added. Do NOT claim "
                                 "masks were made; suggest adjusting the threshold/min-area.")
            if on_complete is not None:
                on_complete()

        def _err(msg):
            self._unet_panel.set_status(f"Error: {msg}")
            self._report_clu(f"UNet segmentation failed: {msg}")
            if on_complete is not None:
                on_complete()

        self._unet_thread = SAMThread(_run, self)
        self._unet_thread.finished.connect(_done)
        self._unet_thread.error.connect(_err)
        self._unet_thread.start()
    def _add_unet_masks_to_store(self, masks: list) -> None:
        canvas = self._canvas_widget.canvas
        store  = canvas.store
        label  = self._unet_panel.label
        color  = "#1a5fa8"
        self._pending_unet_masks.clear()
        canvas._loading = True
        try:
            with self._pred_context("unet"):
                for mask in masks:
                    vertices = self._unet_predictor.mask_to_polygon(mask)
                    if len(vertices) < 3:
                        continue
                    from acorn.core.annotations import ROIAnnotation
                    roi = ROIAnnotation(
                        vertices=vertices, area_nm2=0.0, stats={},
                        color=color, linewidth=1.5, label=label,
                    )
                    store.add(roi)
                    self._pending_unet_masks.append(roi)
        finally:
            canvas._loading = False
        if canvas.renderer is not None:
            canvas.renderer.render_noblit(canvas.store, canvas)
        else:
            canvas.fig.canvas.draw_idle()
    def _on_unet_batch(self) -> None:
        """'Run on ALL Images' button — batch the loaded UNet model over the dataset."""
        self._start_batch_model(
            {"label": self._unet_panel.label, "skip_annotated": True, "queue_after": False},
            "unet",
        )
    def _on_unet_accept(self) -> None:
        self._pending_unet_masks.clear()
        self._unet_panel.set_status("Masks accepted as ROI annotations.")
    def _on_unet_reject(self) -> None:
        self._remove_pending_annotations(self._pending_unet_masks)
        self._pending_unet_masks.clear()
        self._unet_panel.set_status("UNet masks removed.")
    def _on_train_load_yolo(self, model_path: str) -> None:
        """Auto-load the freshly trained YOLO model into the YOLO tab."""
        from acorn.core.yolo_predictor import YOLOPredictor
        self._yolo_panel.set_model_status("Loading trained model…", loaded=False)
        self._seg_panel.set_loaded("yolo", False)

        def _run():
            predictor = YOLOPredictor(model_path=model_path)
            predictor.load_model()
            return predictor

        def _done(p):
            self._yolo_predictor = p
            self._yolo_panel.set_model_status(
                f"Trained model loaded: {Path(model_path).name}", loaded=True
            )
            self._seg_panel.set_loaded("yolo", True)
            self._statusbar.showMessage("Trained YOLO model loaded into YOLO tab.")

        def _err(msg):
            self._yolo_panel.set_model_status(f"Auto-load failed: {msg}", loaded=False)
            self._seg_panel.set_loaded("yolo", False)

        t = SAMThread(_run, self)
        t.finished.connect(_done)
        t.error.connect(_err)
        t.start()
    def _on_train_load_unet(self, model_path: str) -> None:
        """Auto-load the freshly trained UNet model into the UNet tab."""
        info_path = Path(model_path).parent / "training_info.json"
        arch, encoder, n_classes = "Unet", "resnet34", 2
        if info_path.exists():
            try:
                info = json.loads(info_path.read_text())
                arch      = info.get("arch", arch)
                encoder   = info.get("encoder", encoder)
                n_classes = info.get("n_classes", n_classes)
            except Exception:
                pass

        from acorn.core.unet_predictor import UNetPredictor
        predictor = UNetPredictor(
            architecture=arch, encoder=encoder,
            in_channels=1, n_classes=n_classes,
        )
        self._unet_panel.set_model_status("Loading trained model…", loaded=False)
        self._seg_panel.set_loaded("unet", False)

        def _run():
            predictor.load_model(model_path)
            return predictor

        def _done(p):
            self._unet_predictor = p
            self._unet_panel.set_model_status(
                f"Trained model loaded: {Path(model_path).name}", loaded=True
            )
            self._seg_panel.set_loaded("unet", True)
            self._statusbar.showMessage("Trained UNet model loaded into UNet tab.")

        def _err(msg):
            self._unet_panel.set_model_status(f"Auto-load failed: {msg}", loaded=False)
            self._seg_panel.set_loaded("unet", False)

        t = SAMThread(_run, self)
        t.finished.connect(_done)
        t.error.connect(_err)
        t.start()
