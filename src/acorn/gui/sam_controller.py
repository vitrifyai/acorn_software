"""
SAM prompting and mask handling.

Everything between a click on the canvas and a mask in the annotation store:
loading a checkpoint, warming up the encoder, the point / box / scribble prompt
modes, exclude and crop regions, batch runs, and accept / reject.

This is a mixin, not a standalone class - the methods read and write MainWindow
state (the canvas, the annotation store, the pending-mask lists) and are mixed
into it. Splitting them out keeps 30 methods and their prompt bookkeeping in one
file instead of scattered through a 6,000-line window class.
"""
from __future__ import annotations


from PyQt6.QtWidgets import QMessageBox

from acorn.core.annotations import ROIAnnotation
from acorn.gui.threads import SAMThread
from acorn.render import palette as _PAL


def _sample_path(pts: list, spacing: float = 20.0) -> list:
    """Return a subset of pts sampled at roughly `spacing`-pixel intervals."""
    import math
    if len(pts) < 2:
        return list(pts)
    result = [pts[0]]
    accumulated = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dy = pts[i][1] - pts[i - 1][1]
        accumulated += math.sqrt(dx * dx + dy * dy)
        if accumulated >= spacing:
            result.append(pts[i])
            accumulated = 0.0
    if result[-1] != pts[-1]:
        result.append(pts[-1])
    return result


def _offset_vertices(vertices: list, ox: float, oy: float) -> list:
    """Map vertices from SAM crop-space back into full-image coordinates."""
    if ox == 0 and oy == 0:
        return vertices
    return [(vx + ox, vy + oy) for vx, vy in vertices]


def _predict_point_preview(predictor, img8, points, labels, ox: float, oy: float) -> dict:
    """Run point-prompt prediction and contour extraction off the GUI thread."""
    masks = predictor.predict_points(img8, points, labels=labels)
    if not masks:
        return {"has_mask": False, "vertices": []}
    vertices = predictor.mask_to_polygon(masks[0])
    return {"has_mask": True, "vertices": _offset_vertices(vertices, ox, oy)}


class SAMControllerMixin:
    """SAM prompting, previewing and committing. Mixed into MainWindow."""

    def _remove_sam_preview(self) -> None:
        """Remove only the live SAM preview annotation, leaving other edits alone."""
        if self._sam_current_preview is None:
            return
        store = self._canvas_widget.canvas.store
        store.remove(self._sam_current_preview)
        if self._sam_current_preview in self._pending_sam_masks:
            self._pending_sam_masks.remove(self._sam_current_preview)
        self._sam_current_preview = None

    def _on_sam_scribble_commit(self, pts: list, positive: bool = True) -> None:
        """Convert a freehand scribble stroke into SAM point prompts.

        positive=True  → foreground prompts (label 1)
        positive=False → background/negative prompts (label 0)
        """
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return

        sampled = _sample_path(pts, spacing=20.0)
        if not sampled:
            return

        point_label = self._sam_panel.point_label
        label_int = 1 if positive else 0
        for x, y in sampled:
            self._sam_prompt_points.append((x, y))
            self._sam_prompt_labels.append(label_int)
            markers = self._canvas_widget.add_sam_point_marker(x, y, positive, label="")
            self._sam_point_artists.append(markers)

        points_snap = list(self._sam_prompt_points)
        labels_snap = list(self._sam_prompt_labels)
        points_for_sam = [(px - ox, py - oy) for px, py in points_snap]
        self._sam_panel.set_sam_status(f"Running SAM with {len(points_snap)} point(s)…")

        def _run():
            return _predict_point_preview(
                self._sam_predictor, img8, points_for_sam, labels_snap, ox, oy
            )

        def _done(result):
            if not result["has_mask"]:
                self._sam_panel.set_sam_status(
                    "No mask returned — try adding more strokes or a positive point."
                )
                return
            store = self._canvas_widget.canvas.store
            self._remove_sam_preview()
            vertices = result["vertices"]
            if len(vertices) >= 3:
                roi = self._roi_from_sam(vertices, point_label)
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            self._sam_panel.set_sam_status(
                f"Preview updated ({len(points_snap)} prompt point(s)).  "
                "Draw more strokes to refine, or Commit & New / Accept All."
            )
            # Stay in scribble/freehand mode so next stroke adds more prompts
            self._canvas_widget.set_tool("freehand")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()
    def _sam_busy(self) -> bool:
        """Return True if a SAM thread is currently running."""
        return self._sam_thread is not None and self._sam_thread.isRunning()
    def _sam_warmup_encode(self) -> None:
        """Encode the current image into the SAM embedding cache in the background.

        Called after model load and after image switch so the first point/box
        prompt skips the expensive ViT encoder pass.
        """
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return
        if self._sam_busy():
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return
        params = self._contrast_panel.params()
        self._sam_panel.set_sam_status("Encoding image…")

        def _run():
            # Compute the SAME full-res working image the prompt path uses, off
            # the UI thread, and encode it — so the first click hits both the
            # img8 cache and the embedding cache instead of freezing the GUI.
            from acorn.core.contrast import apply_contrast
            import numpy as np
            norm = apply_contrast(img.raw, params, pixel_size_nm=img.pixel_size)
            img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
            from_cache = self._sam_predictor.encode_image(img8)
            return img8, from_cache

        def _done(result):
            img8, from_cache = result
            # Prime the working-image cache so _get_sam_working_image is instant.
            if self._canvas_widget.canvas.dm4 is img:
                self._sam_img8_cache = (img, params, img8)
            if from_cache is True:
                self._sam_panel.set_sam_status("Ready  (embedding loaded from cache).")
            else:
                self._sam_panel.set_sam_status("Ready.")

        def _err(msg):
            self._sam_panel.set_sam_status(f"Encode failed: {msg}")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()
    def _clear_sam_point_artists(self, blit: bool = True) -> None:
        """Remove all SAM point-marker dots from the canvas.

        blit=False skips the redraw — used during an image switch, where
        _finish_switch's load_image() does the authoritative render instead
        (avoids a redundant blit on the transitional canvas).
        """
        for group in self._sam_point_artists:
            for a in group:
                try:
                    a.remove()
                except (ValueError, AttributeError):
                    pass
        self._sam_point_artists.clear()
        self._canvas_widget.canvas._overlay_artists.clear()
        if blit:
            self._canvas_widget.canvas.blit_annotations()
    @staticmethod
    def _sam_color_for_label(label: str) -> str:
        """Return a consistent colour for a SAM mask label.

        Matches the quick-select button colours in SAMPanel so the mask on
        canvas always corresponds visually to the button used to create it.
        """
        _FIXED = {
            "foreground": "#00703C",
            "background": "#c0392b",
        }
        key = label.strip().lower()
        if key in _FIXED:
            return _FIXED[key]
        # Deterministic colour for any custom label — cycle through the same
        # palette used by SAMPanel._user_label_colors
        # One palette for everything drawn on an image — see acorn.render.palette.
        # The old local list used Python's salted str hash, so a label changed
        # colour every time the application restarted.
        return _PAL.color_for_label(key)
    def _on_sam_load_model(self, checkpoint: str, model_cfg: str, backend: str) -> None:
        if self._sam_busy():
            # Say so. A SAM 3 load takes tens of seconds and the button stays
            # enabled throughout, so a second click is normal -- swallowing it
            # without a word makes the control look broken.
            self._sam_panel.set_model_status(
                "A model is already loading — wait for it to finish.", loaded=False)
            return

        ckpt_arg = checkpoint if checkpoint else None

        if backend == "usam":
            from acorn.core.usam_predictor import MicroSAMPredictor
            model_type = self._sam_panel.usam_model_type
            predictor  = MicroSAMPredictor(
                model_type=model_type,
                checkpoint_path=ckpt_arg,
            )
            label = f"micro-SAM ({model_type})"
        else:
            from acorn.core.sam_predictor import SAMPredictor
            predictor = SAMPredictor(
                checkpoint_path=ckpt_arg, model_cfg=model_cfg, backend=backend
            )
            label = backend

        self._sam_panel.set_model_status("Loading model…", loaded=False)
        self._seg_panel.set_loaded("sam", False)

        # For usam, emit download progress through the thread's status signal.
        # _emit is filled after the thread is constructed (thread-safe via Qt signal queue).
        _emit: list = [None]

        def _run():
            if backend == "usam":
                def _progress(pct):
                    fn = _emit[0]
                    if fn:
                        fn(f"Downloading {model_type}… {pct}%")
                predictor.load_model(progress_cb=_progress)
            else:
                predictor.load_model()
            return predictor

        def _done(p):
            self._sam_predictor = p
            active = getattr(p, "backend", None) or label
            self._sam_panel.set_model_status(f"Model loaded ({active}).", loaded=True)
            self._seg_panel.set_loaded("sam", True)
            self._statusbar.showMessage(f"SAM model loaded ({active}).")
            self._sam_warmup_encode()

        def _err(msg):
            self._sam_panel.set_model_status(f"Load failed: {msg}", loaded=False)
            self._seg_panel.set_loaded("sam", False)
            QMessageBox.critical(self, "SAM model failed to load",
                f"The model could not be loaded:\n\n{msg}\n\n"
                "Check that the model file exists and you have read access to it.")

        self._sam_thread = SAMThread(_run, self)
        if backend == "usam":
            _emit[0] = self._sam_thread.status.emit
            self._sam_thread.status.connect(
                lambda msg: self._sam_panel.set_model_status(msg, loaded=False)
            )
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()
    def _get_sam_working_image(self):
        """Return (img8, offset_x, offset_y).

        If a crop region is set, img8 is the cropped sub-image and (offset_x, offset_y)
        is the top-left corner in full-image pixel coordinates.  Callers must add the
        offset to all polygon vertices returned by SAM.
        """
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            return None, 0, 0
        import numpy as np
        params = self._contrast_panel.params()
        # Cache the full-res contrasted uint8 image; it depends only on the image
        # and contrast params, so consecutive SAM clicks needn't recompute it.
        cache = getattr(self, "_sam_img8_cache", None)
        if cache is not None and cache[0] is img and cache[1] == params:
            img8 = cache[2]
        else:
            from acorn.core.contrast import apply_contrast
            norm = apply_contrast(img.raw, params, pixel_size_nm=img.pixel_size)
            img8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
            self._sam_img8_cache = (img, params, img8)
        if self._sam_crop_region is None:
            return img8, 0, 0
        x0, y0, x1, y1 = self._sam_crop_region
        h, w = img8.shape[:2]
        cx0 = max(0, int(round(min(x0, x1))))
        cy0 = max(0, int(round(min(y0, y1))))
        cx1 = min(w, int(round(max(x0, x1))))
        cy1 = min(h, int(round(max(y0, y1))))
        if cx1 <= cx0 or cy1 <= cy0:
            return img8, 0, 0
        return img8[cy0:cy1, cx0:cx1], cx0, cy0
    def _on_sam_exclude_mode(self) -> None:
        self._sam_mode = "exclude_zone"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM exclude zone: drag to mark region SAM should ignore")
    def _on_sam_exclude_clear(self) -> None:
        self._sam_exclude_zone = None
        self._canvas_widget.clear_exclude_zone()
        if self._img_idx >= 0:
            self._sam_exclude_zones.pop(self._img_idx, None)
            self._autosave_timer.start()
        self._sam_panel.reset_region_btns()
        if self._sam_mode == "exclude_zone":
            self._sam_mode = None
        self._statusbar.showMessage("SAM exclude zone cleared.")
    def _on_sam_crop_mode(self) -> None:
        self._sam_mode = "crop_region"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM crop region: drag to restrict SAM to a sub-area")
    def _on_sam_crop_clear(self) -> None:
        self._sam_crop_region = None
        self._canvas_widget.clear_crop_region()
        if self._img_idx >= 0:
            self._sam_crop_regions_saved.pop(self._img_idx, None)
            self._autosave_timer.start()
        self._sam_panel.reset_region_btns()
        self._ann_panel.set_crop_region(None)
        if self._sam_mode == "crop_region":
            self._sam_mode = None
        self._statusbar.showMessage("SAM crop region cleared — SAM will use the full image.")
    def _on_sam_auto_segment(self, _batch_done_cb=None) -> None:
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is busy — please wait.")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None or img.raw is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._sam_panel.set_sam_status("Load the SAM model first.")
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return

        # SAM's image encoder processes at 1024×1024 internally.  Passing a
        # larger image only causes SAM2's mask generator to allocate O(N_points
        # × H × W) float32 tensors at the input resolution — 67 GiB for a
        # typical 4096×4096 cryo-EM image.  Cap here; upscale masks in _done.
        import numpy as np
        _SAM_MAX_DIM = 1024
        h0, w0 = img8.shape[:2]
        _sam_scale = min(1.0, _SAM_MAX_DIM / max(h0, w0, 1))
        if _sam_scale < 1.0:
            from PIL import Image as _PILImg
            _nh, _nw = max(1, int(h0 * _sam_scale)), max(1, int(w0 * _sam_scale))
            img8_sam = np.array(_PILImg.fromarray(img8).resize((_nw, _nh), _PILImg.LANCZOS))
        else:
            img8_sam = img8

        params = self._sam_panel.auto_params
        active = self._sam_predictor.backend or "SAM"
        crop_note = " (cropped region)" if self._sam_crop_region is not None else ""
        # Say when the image is being reduced. SAM's own encoder works at about
        # 1000px whatever it is given, so on a 24-megapixel micrograph it sees
        # roughly a sixth of the linear detail — which silently limits how small
        # a feature can be found, and used to be invisible.
        scale_note = ""
        if _sam_scale < 1.0 and self._sam_crop_region is None:
            scale_note = (f" — image reduced {1/_sam_scale:.1f}× to {_nw}×{_nh}; "
                          f"crop a region for small features")
        self._sam_panel.set_sam_status(f"Running {active}{crop_note}{scale_note}…")

        def _run():
            return self._sam_predictor.predict_everything(img8_sam, **params)

        def _done(masks):
            # Upscale masks from SAM-input space back to working-image space.
            if _sam_scale < 1.0 and masks:
                from PIL import Image as _PILImg
                import numpy as np
                upscaled = []
                for m in masks:
                    m_up = np.array(
                        _PILImg.fromarray(m.astype(np.uint8) * 255).resize(
                            (w0, h0), _PILImg.NEAREST
                        )
                    ).astype(bool)
                    upscaled.append(m_up)
                masks = upscaled
            self._add_sam_masks_to_store(masks, offset=(ox, oy))
            n = len(self._pending_sam_masks)
            # Switch to select mode so user can click masks to delete individually.
            # Also reset prompt-mode buttons so they can be re-activated cleanly.
            self._canvas_widget.set_tool("none")
            self._sam_mode = None
            self._sam_panel.reset_prompt_mode()
            self._sam_panel.set_sam_status(
                f"{n} mask(s) added.  Click a mask to select it, then press Delete to remove it.  "
                "Accept All to keep all remaining masks."
            )
            self._statusbar.showMessage(f"{active} auto-segment: {n} masks found — click to select, Delete to remove.")
            if n:
                self._report_clu(f"SAM auto-segment produced {n} mask(s), added as pending "
                                 "annotations. Tell the user to Accept All to keep them.")
            else:
                self._report_clu("SAM auto-segment produced 0 masks — nothing was added. "
                                 "Do NOT claim masks were made.")
            if _batch_done_cb is not None:
                _batch_done_cb()

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(
            lambda e: (
                self._sam_panel.set_sam_status(f"Error: {e}"),
                self._report_clu(f"SAM auto-segment failed: {e}"),
                _batch_done_cb() if _batch_done_cb else None,
            )
        )
        self._sam_thread.start()
    def _start_batch_sam(self, params: dict) -> None:
        """Kick off the batch SAM pipeline: segment → accept → queue across all images."""
        label           = params.get("label", "")
        points_per_side = int(params.get("points_per_side", 32))
        skip_annotated  = bool(params.get("skip_annotated", True))

        queue: list[int] = []
        for i in range(len(self._image_paths)):
            if skip_annotated:
                existing = self._ann_states.get(i, [])
                if i == self._img_idx:
                    existing = list(self._canvas_widget.canvas.store)
                if existing:
                    continue
            queue.append(i)

        if not queue:
            self._statusbar.showMessage("Batch SAM: all images already annotated — nothing to do.")
            return

        self._batch_proc = {
            "queue":          queue,
            "label":          label,
            "points_per_side": points_per_side,
            "processed":      0,
            "total":          len(queue),
        }
        n = len(queue)
        self._statusbar.showMessage(f"Batch SAM: starting — {n} image(s) to process…")
        self._batch_next_image()
    def _batch_after_sam(self) -> None:
        bp = self._batch_proc
        if bp is None:
            return
        if self._pending_sam_masks:
            self._on_sam_accept()
        self._on_queue_image("")
        if bp["queue"]:
            bp["queue"].pop(0)
        bp["processed"] += 1
        self._batch_next_image()
    def _on_sam_point_mode(self, positive: bool) -> None:
        self._sam_mode      = "pos_point" if positive else "neg_point"
        self._sam_box_click = None
        self._canvas_widget.set_tool("sam")
        label = "positive" if positive else "negative"
        self._statusbar.showMessage(f"SAM: click on canvas to add a {label} point prompt")
    def _on_sam_box_mode(self) -> None:
        self._sam_mode      = "box"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage("SAM box: drag around object, or click two corners")
    def _on_sam_scribble_mode(self) -> None:
        self._sam_mode      = "scribble"
        self._sam_box_click = None
        self._canvas_widget.set_tool("freehand")
        self._statusbar.showMessage(
            "SAM scribble: draw along the feature — stroke points become positive prompts"
        )
    def _on_sam_neg_scribble_mode(self) -> None:
        self._sam_mode      = "scribble_neg"
        self._sam_box_click = None
        self._canvas_widget.set_tool("freehand")
        self._statusbar.showMessage(
            "SAM negative scribble: draw over background — stroke points become negative prompts"
        )
    def _on_sam_neg_box_mode(self) -> None:
        self._sam_mode      = "neg_box"
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        self._canvas_widget.set_tool("sam")
        self._statusbar.showMessage(
            "SAM negative box: drag over background area to add negative prompts at its centre"
        )
    def _on_sam_accept(self) -> None:
        self._pending_sam_masks.clear()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")
        self._sam_panel.set_sam_status("Masks accepted as ROI annotations.")
    def _on_sam_accept_and_queue(self) -> None:
        """Accept all pending SAM masks then immediately queue the image for export."""
        self._on_sam_accept()
        ds_dir = self._export_panel.dataset_dir
        if not ds_dir:
            self._sam_panel.set_sam_status(
                "Masks accepted. Set a dataset directory in the Export tab to enable queuing."
            )
            return
        self._on_queue_image(ds_dir)
        n = len(self._export_queue)
        self._sam_panel.set_sam_status(
            f"Masks accepted and image queued ({n} total in queue)."
        )
    def _on_sam_reject(self) -> None:
        self._remove_pending_annotations(self._pending_sam_masks)
        self._pending_sam_masks.clear()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")
        self._sam_panel.set_sam_status("SAM masks removed.")
    def _on_sam_commit_new(self) -> None:
        """Lock current preview, switch to select mode for vertex editing."""
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._sam_current_preview = None
        self._sam_mode = None
        self._sam_panel.reset_prompt_mode()
        self._clear_sam_point_artists()
        self._canvas_widget.set_tool("none")   # select mode — user can now edit vertices
        count = len(self._pending_sam_masks)
        self._sam_panel.set_sam_status(
            f"{count} mask(s) committed. Edit vertices if needed, then click + Positive Point for the next object."
        )
    def _on_sam_undo_point(self) -> None:
        """Remove the last added point and re-run SAM with the remaining points."""
        if not self._sam_prompt_points:
            self._sam_panel.set_sam_status("No points to undo.")
            return
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return

        # Remove the last point from state
        self._sam_prompt_points.pop()
        self._sam_prompt_labels.pop()

        # Remove its canvas marker
        if self._sam_point_artists:
            for a in self._sam_point_artists.pop():
                self._canvas_widget.remove_artist(a)

        # Remove the current preview mask from the store
        self._remove_sam_preview()

        # No points left — just report and stop
        if not self._sam_prompt_points:
            self._sam_panel.set_sam_status("All points removed. Click to start a new prompt.")
            return

        # Re-run SAM with the remaining points
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        points_for_sam = [(px - ox, py - oy) for px, py in self._sam_prompt_points]
        labels_snap    = list(self._sam_prompt_labels)
        point_label    = self._sam_panel.point_label
        self._sam_panel.set_sam_status("Re-running SAM…")

        def _run():
            return _predict_point_preview(
                self._sam_predictor, img8, points_for_sam, labels_snap, ox, oy
            )

        def _done(result):
            if not result["has_mask"]:
                self._sam_panel.set_sam_status("SAM returned no mask — add more points.")
                return
            store = self._canvas_widget.canvas.store
            vertices = result["vertices"]
            if len(vertices) >= 3:
                roi = self._roi_from_sam(vertices, point_label)
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            n_pos = labels_snap.count(1)
            n_neg = labels_snap.count(0)
            self._sam_panel.set_sam_status(
                f"Preview: {n_pos} pos + {n_neg} neg point(s).  "
                "Add more points, Commit & New to lock and edit, or Accept All."
            )

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()
    def _on_sam_clear_points(self) -> None:
        """Discard accumulated point prompts and remove the current preview mask."""
        self._remove_sam_preview()
        self._sam_prompt_points.clear()
        self._sam_prompt_labels.clear()
        self._clear_sam_point_artists()
        self._sam_panel.set_sam_status("Points cleared. Click to start a new prompt.")
    def run_sam_text(self, word: str, confidence: float = 0.5) -> None:
        """
        Segment everything matching *word*, using SAM 3's text prompt.

        This is what turns a description into training data: accept the result and
        it becomes annotations, which export and train a YOLO or UNet that then
        knows the class by name. SAM 3 is the only backend that can be steered by
        a word — see acorn.core.vocabulary for what the others do instead.
        """
        from acorn.core import vocabulary

        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is busy — please wait.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._sam_panel.set_sam_status("Load the SAM model first.")
            return

        backend = self._sam_predictor.backend or ""
        if not vocabulary.backend_uses_text(backend):
            note = vocabulary.backend_note(backend)
            self._sam_panel.set_sam_status(note)
            self._statusbar.showMessage(note, 8000)
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return

        term = vocabulary.resolve(word)
        shown = term.canonical if term else word
        self._sam_panel.set_sam_status(f"Looking for {shown}…")

        def _run():
            return self._sam_predictor.predict_text(img8, word, confidence=confidence)

        def _done(result):
            masks, phrase = result
            if not masks:
                self._sam_panel.set_sam_status(
                    f"No {shown} found (tried \"{phrase}\"). Lower the confidence, "
                    f"try a different word, or use a point or box prompt."
                )
                return
            self._pending_sam_label = word
            self._add_sam_masks_to_store(masks, offset=(ox, oy))
            self._sam_panel.set_sam_status(
                f"{len(masks)} {shown} found using \"{phrase}\" — "
                f"Accept All to keep them."
            )

        def _err(msg):
            self._sam_panel.set_sam_status(f"Text prompt failed: {msg}")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.done.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()

    def _roi_from_sam(self, vertices, label):
        """Build an ROIAnnotation from SAM vertices, carrying its measurements.

        Every SAM path used to store `area_nm2=0.0, stats={}`, so a mask you
        accepted had no size attached — the outline was saved to the sidecar and
        the measurement had to be recomputed elsewhere to get a diameter out of
        it. `polygon_metrics` is the same function the Measure tools use, so a
        SAM region and a hand-drawn one now report size the same way.

        Shape metrics go in `stats`, which is a free dict already serialised with
        the annotation; intensity stats (mean/std/min/max) are added separately
        by the ROI tools and are not overwritten here.
        """
        from acorn.core.annotations import ROIAnnotation
        from acorn.core.measurements import polygon_metrics

        px_nm = getattr(self._engine, "pixel_size", 0.0) or 0.0
        metrics = polygon_metrics(vertices, px_nm) if px_nm > 0 else {}
        return ROIAnnotation(
            vertices  = vertices,
            area_nm2  = float(metrics.get("area_nm2", 0.0)),
            stats     = dict(metrics),
            color     = self._sam_color_for_label(label),
            linewidth = 1.5,
            label     = label,
        )

    def _add_sam_masks_to_store(self, masks, offset: tuple = (0, 0)) -> None:
        """Convert SAM masks to ROIAnnotations and add to the store.

        Parameters
        ----------
        masks  : list of masks returned by SAMPredictor
        offset : (ox, oy) pixel offset to add to all polygon vertices.
                 Non-zero when SAM was run on a cropped sub-image.
        """
        canvas = self._canvas_widget.canvas
        store  = canvas.store
        label  = self._sam_panel.label
        ox, oy = offset
        self._pending_sam_masks.clear()

        # Suppress incremental renders while adding many masks so we don't pay
        # O(N) blits.  A single render_noblit at the end is cheaper and more
        # reliable (incremental draw_artist calls can silently fail for patches
        # that haven't gone through a full draw() cycle).
        canvas._loading = True
        try:
            for mask in masks:
                vertices = self._sam_predictor.mask_to_polygon(mask)
                if len(vertices) < 3:
                    continue
                if ox != 0 or oy != 0:
                    vertices = [(vx + ox, vy + oy) for vx, vy in vertices]
                # Filter: discard masks whose centroid falls inside the exclude zone
                if self._sam_exclude_zone is not None:
                    ex0, ey0, ex1, ey1 = self._sam_exclude_zone
                    cx = sum(v[0] for v in vertices) / len(vertices)
                    cy = sum(v[1] for v in vertices) / len(vertices)
                    if ex0 <= cx <= ex1 and ey0 <= cy <= ey1:
                        continue
                roi = self._roi_from_sam(vertices, label)
                store.add(roi)
                self._pending_sam_masks.append(roi)
        finally:
            canvas._loading = False

        # Single authoritative redraw after all masks are in the store
        if canvas.renderer is not None:
            canvas.renderer.render_noblit(canvas.store, canvas)
        else:
            canvas.fig.canvas.draw_idle()
    def _sam_point_prompt(self, x: float, y: float, positive: bool) -> None:
        if self._sam_busy():
            self._sam_panel.set_sam_status("SAM is running — please wait.")
            return
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            self._sam_panel.set_sam_status("Load the SAM model first (click 'Load Model').")
            return
        img = self._canvas_widget.canvas.dm4
        if img is None:
            self._sam_panel.set_sam_status("No image loaded.")
            return

        # Accumulate point (stored in full image coords), draw marker immediately
        point_label = self._sam_panel.point_label
        self._sam_prompt_points.append((x, y))
        self._sam_prompt_labels.append(1 if positive else 0)
        display_label = point_label if positive else ""
        markers = self._canvas_widget.add_sam_point_marker(x, y, positive, label=display_label)
        self._sam_point_artists.append(markers)   # list-of-lists: one group per point

        # Snapshot in crop-space (full coords minus crop offset)
        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        points_snap_full = list(self._sam_prompt_points)
        labels_snap      = list(self._sam_prompt_labels)
        points_for_sam   = [(px - ox, py - oy) for px, py in points_snap_full]
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return _predict_point_preview(
                self._sam_predictor, img8, points_for_sam, labels_snap, ox, oy
            )

        def _done(result):
            if not result["has_mask"]:
                n_pos = labels_snap.count(1)
                n_neg = labels_snap.count(0)
                if n_pos == 0:
                    self._sam_panel.set_sam_status(
                        "No mask — add at least one positive point first, "
                        "then use negative points to refine."
                    )
                else:
                    self._sam_panel.set_sam_status(
                        "SAM returned no mask for these points — try repositioning."
                    )
                return
            store = self._canvas_widget.canvas.store
            self._remove_sam_preview()
            vertices = result["vertices"]
            if len(vertices) >= 3:
                roi = self._roi_from_sam(vertices, point_label)
                store.add(roi)
                self._pending_sam_masks.append(roi)
                self._sam_current_preview = roi
            n_pos = labels_snap.count(1)
            n_neg = labels_snap.count(0)
            self._sam_panel.set_sam_status(
                f"Preview: {n_pos} pos + {n_neg} neg point(s).  "
                "Add more points, Commit & New to lock and edit, or Accept All."
            )

        def _err(msg):
            self._sam_prompt_points.pop()
            self._sam_prompt_labels.pop()
            if self._sam_point_artists:
                for a in self._sam_point_artists.pop():
                    self._canvas_widget.remove_artist(a)
            self._sam_panel.set_sam_status(f"Error: {msg}")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(_err)
        self._sam_thread.start()
    def _sam_box_prompt_first_click(self, x: float, y: float) -> None:
        self._sam_box_click = (x, y)
        self._canvas_widget.set_sam_box_anchor(x, y)
        self._statusbar.showMessage(
            f"SAM box: first corner at ({x:.0f}, {y:.0f}) — drag or click second corner"
        )
    def _sam_box_prompt_second_click(self, x: float, y: float) -> None:
        if self._sam_box_click is None:
            return
        if self._sam_busy():
            return
        x0, y0 = self._sam_box_click
        x1, y1 = x, y
        self._sam_box_click = None
        self._canvas_widget.clear_sam_box_anchor()
        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        box = (min(x0, x1) - ox, min(y0, y1) - oy, max(x0, x1) - ox, max(y0, y1) - oy)
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return self._sam_predictor.predict_box(img8, box)

        def _done(mask):
            self._add_sam_masks_to_store([mask], offset=(ox, oy))
            self._sam_panel.set_sam_status("Box prompt: 1 mask. Undo if wrong, or Accept All.")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()
    def _on_sam_box_drag(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Handle a drag-drawn SAM box (from canvas sam_box_commit signal).

        Routes to exclude-zone, crop-region, or normal box-prompt handling
        depending on the current _sam_mode.
        """
        if self._sam_mode == "exclude_zone":
            self._sam_exclude_zone = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self._canvas_widget.set_exclude_zone(*self._sam_exclude_zone)
            if self._img_idx >= 0:
                self._sam_exclude_zones[self._img_idx] = self._sam_exclude_zone
                self._autosave_timer.start()
            self._sam_panel.reset_region_btns()
            self._sam_mode = None
            self._canvas_widget.set_tool("none")
            self._statusbar.showMessage(
                f"Exclude zone set: ({self._sam_exclude_zone[0]:.0f}, {self._sam_exclude_zone[1]:.0f}) — "
                f"({self._sam_exclude_zone[2]:.0f}, {self._sam_exclude_zone[3]:.0f})"
            )
            return

        if self._sam_mode == "crop_region":
            self._sam_crop_region = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self._canvas_widget.set_crop_region(*self._sam_crop_region)
            self._ann_panel.set_crop_region(self._sam_crop_region)
            if self._img_idx >= 0:
                self._sam_crop_regions_saved[self._img_idx] = self._sam_crop_region
                self._autosave_timer.start()
            self._sam_panel.reset_region_btns()
            self._sam_mode = None
            self._canvas_widget.set_tool("none")
            self._statusbar.showMessage(
                f"Crop region set: ({self._sam_crop_region[0]:.0f}, {self._sam_crop_region[1]:.0f}) — "
                f"({self._sam_crop_region[2]:.0f}, {self._sam_crop_region[3]:.0f})"
            )
            return

        if self._sam_mode == "neg_box":
            # Add the centre of the dragged box as a negative point prompt
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            self._sam_prompt_points.append((cx, cy))
            self._sam_prompt_labels.append(0)
            markers = self._canvas_widget.add_sam_point_marker(cx, cy, False, label="")
            self._sam_point_artists.append(markers)
            if (self._sam_predictor is not None and self._sam_predictor.is_loaded
                    and not self._sam_busy()):
                img8, ox, oy = self._get_sam_working_image()
                if img8 is not None:
                    points_snap = list(self._sam_prompt_points)
                    labels_snap = list(self._sam_prompt_labels)
                    points_for_sam = [(px - ox, py - oy) for px, py in points_snap]
                    self._sam_panel.set_sam_status("Running SAM with negative box point…")

                    def _run_nb():
                        return _predict_point_preview(
                            self._sam_predictor, img8, points_for_sam, labels_snap, ox, oy
                        )

                    def _done_nb(result):
                        if not result["has_mask"]:
                            self._sam_panel.set_sam_status("No mask — add a positive point first.")
                            return
                        store = self._canvas_widget.canvas.store
                        self._remove_sam_preview()
                        vertices = result["vertices"]
                        if len(vertices) >= 3:
                            point_label = self._sam_panel.point_label
                            roi = self._roi_from_sam(vertices, point_label)
                            store.add(roi)
                            self._pending_sam_masks.append(roi)
                            self._sam_current_preview = roi
                        self._sam_panel.set_sam_status("Updated mask with negative box point.")

                    self._sam_thread = SAMThread(_run_nb, self)
                    self._sam_thread.finished.connect(_done_nb)
                    self._sam_thread.error.connect(
                        lambda e: self._sam_panel.set_sam_status(f"Error: {e}")
                    )
                    self._sam_thread.start()
            return

        if self._sam_mode != "box":
            return
        if self._sam_busy():
            return
        self._sam_box_click = None   # cancel any pending two-click state

        if self._sam_predictor is None or not self._sam_predictor.is_loaded:
            return

        img8, ox, oy = self._get_sam_working_image()
        if img8 is None:
            return
        box = (x0 - ox, y0 - oy, x1 - ox, y1 - oy)
        self._sam_panel.set_sam_status("Running SAM…")

        def _run():
            return self._sam_predictor.predict_box(img8, box)

        def _done(mask):
            self._add_sam_masks_to_store([mask], offset=(ox, oy))
            self._sam_panel.set_sam_status("Box prompt: 1 mask. Undo if wrong, or Accept All.")

        self._sam_thread = SAMThread(_run, self)
        self._sam_thread.finished.connect(_done)
        self._sam_thread.error.connect(lambda e: self._sam_panel.set_sam_status(f"Error: {e}"))
        self._sam_thread.start()
