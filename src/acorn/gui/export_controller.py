"""
Getting work out of ACORN: the export queue, batch export, mask and NEXUS
writers, dataset finalisation, measurement CSVs, and pushing to the Hub.
"""
from __future__ import annotations

from pathlib import Path

from acorn.export import MEASUREMENTS_CSV as _MEAS_CSV, measurements_dir as _meas_dir
from acorn.gui.threads import BatchExportThread, TrainingThread

class ExportControllerMixin:
    """Mixed into MainWindow — these methods use its state directly."""

    def _on_export(self, path: str, fmt: str, dpi: int) -> None:
        if self._canvas_widget.canvas.dm4 is None:
            self._export_panel.set_status("No image loaded.")
            return
        try:
            out = self._canvas_widget.canvas.save(path, dpi=dpi, fmt=fmt)
            self._export_panel.set_status(f"Saved: {out.name}")
            self._statusbar.showMessage(f"Exported → {out}")
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, path))
    def _on_export_raw(self, path: str) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return
        try:
            import tifffile
            tifffile.imwrite(path, img.raw,
                             metadata={"pixel_size_nm": str(img.pixel_size)})
            self._export_panel.set_status(f"Raw TIFF saved: {Path(path).name}")
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, path))
    def _on_export_masks(self, stem: str) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_status("No image loaded.")
            return
        store = self._canvas_widget.canvas.store
        rois = [a for a in store if getattr(a, "type", None) == "roi"]
        if not rois:
            self._export_panel.set_status("No ROI regions to export.")
            self._report_clu("No ROI regions on the image — nothing to export as masks.")
            return
        try:
            from acorn.export.mask_exporter import export_masks
            result = export_masks(store, img.shape, stem)
            n = result["n_regions"]
            self._export_panel.set_status(f"Masks saved: {n} region(s)")
            self._statusbar.showMessage(
                f"Mask → {result['mask_path']}  Labels → {result['json_path']}"
            )
            self._report_clu(f"Exported {n} ROI region(s) to {result['mask_path']} "
                             f"(+ labels JSON).")
        except Exception as e:
            self._export_panel.set_status(self._format_write_error(e, stem))
            self._report_clu(f"Mask export failed: {self._format_write_error(e, stem)}")
    def _on_training_export(self, dataset_dir: str) -> None:
        import shutil
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_train_status("No image loaded.")
            return

        # Disk-space pre-check
        ds_path = Path(dataset_dir)
        ds_path.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(str(ds_path)).free
        if free_bytes < 500 * 1024 * 1024:
            free_mb = free_bytes // (1024 * 1024)
            self._export_panel.set_train_status(
                f"Low disk space: only {free_mb} MB free on "
                f"{ds_path.anchor}. Export cancelled — free at least 500 MB."
            )
            return

        store_snapshot = list(self._canvas_widget.canvas.store)

        # Warn immediately if there are no ROI annotations on this image
        n_rois = sum(1 for a in store_snapshot if getattr(a, "type", None) == "roi")
        if n_rois == 0:
            self._export_panel.set_train_status(
                "No ROI annotations on this image — pick particles in the SAM tab first, "
                "then Commit & New (or Accept All) before exporting."
            )
            return

        params   = self._contrast_panel.params()
        cfg_dict = self._export_panel.training_config()

        from acorn.export.training_exporter import TrainingConfig
        config = TrainingConfig(**cfg_dict)

        self._export_panel.set_train_status(
            f"Exporting {n_rois} ROI(s)… (running in background)"
        )
        self._statusbar.showMessage(f"Training export started -> {dataset_dir}")

        self._train_thread = TrainingThread(
            dataset_dir, img, store_snapshot, params, config, parent=self
        )
        self._train_thread.progress.connect(self._export_panel.set_train_status)
        self._train_thread.progress_int.connect(self._export_panel.set_train_progress)
        self._train_thread.finished.connect(self._on_training_export_done)
        self._train_thread.error.connect(
            lambda msg: (
                self._export_panel.set_train_status(f"Error: {msg}"),
                self._export_panel.reset_train_progress(),
            )
        )
        self._train_thread.start()
    def _on_training_export_done(self, result: dict) -> None:
        n_aug   = result["n_augmented"]
        n_tiles = result["n_tiles"]
        n_inst  = result["n_instances_total"]
        n_skip  = result["n_skipped_tiles"]
        self._export_panel.set_train_status(
            f"Done. {n_aug} entries ({n_tiles} tiles, {n_inst} instance(s), "
            f"{n_skip} empty skipped)."
        )
        self._export_panel.reset_train_progress()
        self._statusbar.showMessage(
            f"Training export done  ({n_aug} tiles/augs, {n_inst} masks)"
        )
    def _on_queue_image(self, _dataset_dir: str) -> None:
        """Snapshot the current image + annotations into the export queue."""
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._export_panel.set_train_status("No image loaded.")
            return

        store_snapshot = list(self._canvas_widget.canvas.store)
        n_rois = sum(1 for a in store_snapshot if getattr(a, "type", None) == "roi")

        stem = (
            self._image_paths[self._img_idx].stem
            if 0 <= self._img_idx < len(self._image_paths)
            else "image"
        )
        if any(item["stem"] == stem for item in self._export_queue):
            self._export_panel.set_train_status(f"{stem} is already queued.")
            return

        self._export_queue.append({
            "dm4img":         img,
            "store_snapshot": store_snapshot,
            "params":         self._contrast_panel.params(),
            "stem":           stem,
            "path":           str(img.filepath) if img is not None else "",
            "n_rois":         n_rois,
        })
        names = [item["stem"] for item in self._export_queue]
        self._export_panel.set_queue_status(len(self._export_queue), names)
        self._export_panel.update_queue_table(self._export_queue)
        ann_note = f"{n_rois} ROI(s)" if n_rois > 0 else "no annotations — will contribute negative tiles"
        self._export_panel.set_train_status(
            f"Queued: {stem} ({ann_note})  —  {len(self._export_queue)} total in queue."
        )
    def _on_clear_queue(self) -> None:
        self._export_queue.clear()
        self._export_panel.set_queue_status(0, [])
        self._export_panel.update_queue_table([])
        self._export_panel.set_train_status("Queue cleared.")
    def _on_batch_export(self, dataset_dir: str) -> None:
        """Export all queued images to the training dataset."""
        if not self._export_queue:
            return

        if self._batch_export_thread and self._batch_export_thread.isRunning():
            self._export_panel.set_train_status("Export already running — please wait.")
            return

        from acorn.export.training_exporter import TrainingConfig
        cfg_dict = self._export_panel.training_config()
        config   = TrainingConfig(**cfg_dict)

        n = len(self._export_queue)
        self._export_panel.set_train_status(f"Starting batch export of {n} image(s)…")
        self._statusbar.showMessage(f"Batch training export started — {n} image(s)")

        self._batch_export_thread = BatchExportThread(
            items=list(self._export_queue),
            dataset_dir=dataset_dir,
            config=config,
            parent=self,
        )
        self._batch_export_thread.image_status.connect(self._export_panel.set_train_status)
        self._batch_export_thread.image_progress.connect(self._export_panel.set_image_progress)
        self._batch_export_thread.tile_progress.connect(self._export_panel.set_train_progress)
        self._batch_export_thread.item_done.connect(
            lambda idx, stem: self._statusbar.showMessage(
                f"Exported {stem} ({idx + 1}/{n})"
            )
        )
        self._batch_export_thread.error.connect(
            lambda idx, msg: self._export_panel.set_train_status(f"Error: {msg}")
        )
        self._batch_export_thread.finished.connect(self._on_batch_export_done)
        self._export_panel.set_export_running(True)
        self._batch_export_thread.start()
    def _on_batch_export_done(self, results: list) -> None:
        self._export_panel.set_export_running(False)
        self._export_panel.reset_train_progress()
        total_aug   = sum(r.get("n_augmented", 0) for r in results)
        total_inst  = sum(r.get("n_instances_total", 0) for r in results)
        n_images    = len(results)
        self._export_panel.set_train_status(
            f"Batch export done — {n_images} image(s), {total_aug} tiles/augs, "
            f"{total_inst} instance(s) total."
        )
        self._statusbar.showMessage(
            f"Batch export complete: {n_images} images, {total_aug} tiles"
        )
        # Clear queue after successful export
        self._export_queue.clear()
        self._export_panel.set_queue_status(0, [])
        self._export_panel.update_queue_table([])
    def _on_push_hub(self, dataset_dir: str, repo_id: str, token: str) -> None:
        try:
            from acorn.export.hub_exporter import push_to_hub
            token_arg = token if token else None
            url = push_to_hub(dataset_dir, repo_id, token=token_arg)
            self._export_panel.set_hub_status(f"Pushed. URL: {url}")
            self._statusbar.showMessage(f"Dataset pushed to HuggingFace Hub: {url}")
        except Exception as exc:
            self._export_panel.set_hub_status(f"Error: {exc}")
    def _on_display_export(self) -> None:
        """Export 8-bit contrast-normalised PNG next to the source file for external annotation."""
        img = self._canvas_widget.canvas.dm4
        norm = self._canvas_widget.canvas.norm_image
        if img is None or norm is None:
            self._export_panel.set_status("No image loaded.")
            return
        import numpy as np
        from PIL import Image as _PILImage
        img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
        if img8.ndim == 2:
            pil_img = _PILImage.fromarray(img8, mode="L")
        else:
            pil_img = _PILImage.fromarray(img8)
        out_path = img.filepath.parent / f"{img.filepath.stem}_display.png"
        pil_img.save(str(out_path))
        self._export_panel.set_status(f"Saved: {out_path.name}")
        self._statusbar.showMessage(f"Display image saved: {out_path}")
    def _on_finalize_dataset(self, dataset_dir: str, val_frac: float, test_frac: float, assignments: dict) -> None:
        try:
            from acorn.export.dataset_finalizer import finalize_dataset
            result = finalize_dataset(
                dataset_dir, val_frac=val_frac, test_frac=test_frac,
                explicit_splits=assignments or None,
            )
            sc = result["split_counts"]
            self._export_panel.set_fin_status(
                f"Done. Train={sc['train']}  Val={sc['val']}  Test={sc['test']} tiles."
            )
            self._statusbar.showMessage(
                f"Dataset finalized -> {dataset_dir}/splits/  "
                f"train={sc['train']} val={sc['val']} test={sc['test']}"
            )
            self._report_clu(f"Dataset finalized at {dataset_dir}: "
                             f"train={sc['train']}, val={sc['val']}, test={sc['test']} tiles.")
        except Exception as e:
            self._export_panel.set_fin_status(f"Error: {e}")
            self._report_clu(f"Dataset finalize failed: {e}")
    def _on_export_nexus(self, params: dict) -> None:
        """Write a NeXus-compatible HDF5 file for the loaded dataset."""
        if not self._image_paths:
            self._statusbar.showMessage("No images loaded — cannot export NeXus.")
            return
        from pathlib import Path as _Path
        from acorn.export.nexus_exporter import export_nexus

        img_dir    = _Path(self._image_paths[0]).parent
        meas_root  = _meas_dir(img_dir)
        meas_root.mkdir(parents=True, exist_ok=True)
        out_path   = meas_root / (img_dir.name + "_acorn.nxs")

        # Snapshot current annotations
        all_ann_states = dict(self._ann_states)
        if self._img_idx >= 0:
            all_ann_states[self._img_idx] = list(self._canvas_widget.canvas.store)

        # Gather measurements DataFrame if available
        df = None
        csv_path = meas_root / _MEAS_CSV
        if csv_path.exists():
            try:
                import pandas as _pd
                df = _pd.read_csv(str(csv_path))
            except Exception:
                pass

        include_images = bool(params.get("include_images", True))
        sample_name    = str(params.get("sample_name", ""))
        title          = str(params.get("title", img_dir.name))

        try:
            export_nexus(
                output_path   = out_path,
                image_paths   = list(self._image_paths),
                ann_states    = all_ann_states,
                px_overrides  = dict(self._px_overrides),
                image_cache   = dict(self._image_cache),
                measurements_df = df,
                include_images  = include_images,
                title           = title,
                sample_name     = sample_name,
            )
            self._statusbar.showMessage(
                f"NeXus export → acorn_measurements/{out_path.name}"
            )
        except Exception as exc:
            self._statusbar.showMessage(f"NeXus export failed: {exc}")
            import logging
            logging.getLogger(__name__).exception("NeXus export error")
