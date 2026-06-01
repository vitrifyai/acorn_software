"""Plugin context — exposes core application state to ACORN plugins."""
from __future__ import annotations
import threading
import weakref
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Callable

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMenu

if TYPE_CHECKING:
    from acorn.gui.main_window import MainWindow
    from acorn.gui.canvas_widget import CanvasWidget
    from acorn.core.dm4_loader import DM4Image
    from acorn.core.annotations import AnnotationStore
    from acorn.core.contrast import ContrastParams


class AcornContext(QObject):
    """
    Read/write access to core application state, exposed to plugins.

    Plugins receive this object at construction time and must not import
    MainWindow directly.
    """

    # Emitted by the core when state changes (plugins connect to these)
    image_loaded        = pyqtSignal(object)   # DM4Image
    annotations_changed = pyqtSignal(object)   # AnnotationStore
    pixel_size_changed  = pyqtSignal(float)
    slice_changed       = pyqtSignal(int)      # z-slice index (for acorn_3d)

    # Emitted by the LLM assistant to request tool actions (action_name, params)
    action_requested    = pyqtSignal(str, dict)

    def __init__(self, main_window: "MainWindow") -> None:
        super().__init__()
        self._window_ref = weakref.ref(main_window)
        self._nav_loaded_event = threading.Event()
        self.image_loaded.connect(self._on_image_loaded_set_event)

    def _on_image_loaded_set_event(self, _img=None) -> None:
        self._nav_loaded_event.set()

    def arm_nav_wait(self) -> None:
        """Clear the event before emitting a navigation tool — call from agent thread."""
        self._nav_loaded_event.clear()

    def wait_for_image_load(self, timeout: float = 10.0) -> bool:
        """Block until image_loaded fires (or timeout). Call after arm_nav_wait()."""
        return self._nav_loaded_event.wait(timeout=timeout)

    def _w(self) -> Optional["MainWindow"]:
        return self._window_ref()

    # ── read-only state ───────────────────────────────────────────────────────

    @property
    def current_image(self) -> Optional["DM4Image"]:
        w = self._w()
        if w is None or w._img_idx < 0:
            return None
        return w._canvas_widget.canvas.dm4

    @property
    def current_pixel_size_nm(self) -> float:
        w = self._w()
        if w is None:
            return 1.0
        return w._engine.pixel_size if w._engine.pixel_size > 0 else 1.0

    @property
    def annotation_store(self) -> Optional["AnnotationStore"]:
        w = self._w()
        if w is None:
            return None
        return w._canvas_widget.canvas.store

    @property
    def all_annotation_states(self) -> dict[int, list]:
        """
        Dict of image_index -> annotation list for all loaded images.
        The current image's live store is overlaid at its index.
        Plugins must not mutate this dict.
        """
        w = self._w()
        if w is None:
            return {}
        result = dict(w._ann_states)
        if w._img_idx >= 0:
            result[w._img_idx] = list(w._canvas_widget.canvas.store)
        return result

    @property
    def image_paths(self) -> list[Path]:
        w = self._w()
        return list(w._image_paths) if w else []

    @property
    def current_image_index(self) -> int:
        w = self._w()
        return w._img_idx if w else -1

    def pixel_size_for_index(self, idx: int) -> float:
        """Return pixel size for image at idx — manual override takes priority."""
        w = self._w()
        if w is None:
            return 1.0
        override = w._px_overrides.get(idx)
        if override is not None and override > 0:
            return float(override)
        if idx == w._img_idx:
            img = w._canvas_widget.canvas.dm4
            if img is not None and img.pixel_size > 0:
                return float(img.pixel_size)
        # For unloaded images, try the sidecar file
        if 0 <= idx < len(w._image_paths):
            sidecar = w._image_paths[idx].parent / f".{w._image_paths[idx].stem}.acorn.json"
            try:
                import json as _json
                data = _json.loads(sidecar.read_text())
                px = data.get("pixel_size_nm")
                if px and float(px) > 0:
                    return float(px)
            except Exception:
                pass
        return 1.0

    @property
    def current_contrast_params(self) -> Optional["ContrastParams"]:
        w = self._w()
        if w is None:
            return None
        return w._canvas_widget.canvas._params if hasattr(w._canvas_widget.canvas, "_params") else None

    # ── write API ─────────────────────────────────────────────────────────────

    def canvas_widget(self) -> Optional["CanvasWidget"]:
        w = self._w()
        return w._canvas_widget if w else None

    def set_status(self, message: str, timeout_ms: int = 0) -> None:
        w = self._w()
        if w:
            w._statusbar.showMessage(message, timeout_ms)

    def register_menu_action(
        self,
        menu_name: str,
        label: str,
        callback: Callable,
        shortcut: Optional[str] = None,
    ) -> Optional[QAction]:
        """
        Add an action to an existing menu (e.g. 'File', 'View') or create
        a new top-level menu if it doesn't exist.
        """
        w = self._w()
        if w is None:
            return None
        menubar = w.menuBar()
        target_menu: Optional[QMenu] = None
        for action in menubar.actions():
            if action.text().replace("&", "") == menu_name:
                target_menu = action.menu()
                break
        if target_menu is None:
            target_menu = menubar.addMenu(menu_name)
        action = QAction(label, w)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(callback)
        target_menu.addAction(action)
        return action

    def get_nav_state(self) -> dict:
        """Thread-safe subset of state — only Python primitives, safe to call from QThread.

        Used by LLMAgent after navigation tools to refresh pixel size / filename
        without touching Qt widget methods.
        """
        w = self._w()
        if w is None:
            return {}
        idx = w._img_idx
        paths = w._image_paths
        state: dict = {
            "current_image_index": idx,
            "image_count": len(paths),
        }
        if 0 <= idx < len(paths):
            state["image_name"] = paths[idx].name
            engine_px = w._engine.pixel_size
            override  = w._px_overrides.get(idx)
            state["pixel_size_nm"] = float(override if override and override > 0 else
                                           (engine_px if engine_px > 0 else 1.0))
            # Include annotation summary so CLU knows what's already on the new image
            anns = w._ann_states.get(idx) or []
            state["annotation_count"] = len(anns)
            labels: dict[str, int] = {}
            for a in anns:
                lbl = getattr(a, "label", None) or getattr(a, "text", None) or "unknown"
                labels[lbl] = labels.get(lbl, 0) + 1
            if labels:
                state["annotation_labels"] = labels
            # Clear stale pending counts from the previous image
            state["pending_sam"]  = 0
            state["pending_yolo"] = 0
            state["pending_unet"] = 0

        # Rebuild image_list so CLU's running list always has correct pixel sizes
        # and annotation counts — reads only Python dicts, thread-safe.
        image_list: list[dict] = []
        for i, path in enumerate(paths):
            img_anns = w._ann_states.get(i, []) or []
            if i == idx:
                img_anns = anns if 0 <= idx < len(paths) else img_anns
            img_labels: dict[str, int] = {}
            for a in img_anns:
                lbl = getattr(a, "label", None) or getattr(a, "text", None) or "unknown"
                img_labels[lbl] = img_labels.get(lbl, 0) + 1
            image_list.append({
                "index":            i,
                "filename":         path.name,
                "annotation_count": len(img_anns),
                "label_counts":     img_labels,
                "pixel_size_nm":    self.pixel_size_for_index(i),
            })
        state["image_list"] = image_list
        return state

    def get_llm_state(self) -> dict:
        """Return a snapshot of current application state for the LLM system prompt."""
        w = self._w()
        state: dict = {}
        if w is None:
            return state

        img = self.current_image
        if img is not None:
            state["image_name"] = img.filepath.name if img.filepath else "untitled"
            state["image_shape"] = list(img.raw.shape) if img.raw is not None else []
            engine_px  = w._engine.pixel_size
            img_px     = img.pixel_size if img.pixel_size > 0 else None
            override   = w._px_overrides.get(w._img_idx)
            best_px    = override or img_px or (engine_px if engine_px > 0 else 1.0)
            state["pixel_size_nm"] = float(best_px)
            state["is_movie"]  = img.is_movie
            state["n_frames"]  = img.n_frames
        state["image_count"]         = len(self.image_paths)
        state["current_image_index"] = self.current_image_index

        state["sam_loaded"]  = getattr(w, "_sam_predictor",  None) is not None and getattr(w._sam_predictor,  "is_loaded", False)
        state["yolo_loaded"] = getattr(w, "_yolo_predictor", None) is not None and getattr(w._yolo_predictor, "is_loaded", False)
        state["unet_loaded"] = getattr(w, "_unet_predictor", None) is not None and getattr(w._unet_predictor, "is_loaded", False)

        state["pending_sam"]  = len(getattr(w, "_pending_sam_masks",  []))
        state["pending_yolo"] = len(getattr(w, "_pending_yolo_anns",  []))
        state["pending_unet"] = len(getattr(w, "_pending_unet_masks", []))

        try:
            state["contrast_method"] = w._contrast_panel.params().method
            state["contrast_presets"] = list(w._contrast_panel._all_presets().keys())
        except Exception:
            pass

        # SAM configuration
        try:
            sp = w._sam_panel
            state["sam_backend"]    = sp._backend_combo.currentData()
            state["sam_checkpoint"] = sp._ckpt_combo.currentText()
            ckpts = [sp._ckpt_combo.itemText(i) for i in range(sp._ckpt_combo.count())]
            state["sam_checkpoints_available"] = ckpts
            state["sam_points_per_side"]  = sp._pts_per_side.value()
            state["sam_iou_thresh"]       = sp._iou_thresh.value()
            state["sam_stability_thresh"] = sp._stability_thresh.value()
        except Exception:
            pass

        # YOLO configuration
        try:
            yp = w._yolo_panel
            state["yolo_model_path"] = yp._model_combo.currentText() if hasattr(yp, "_model_combo") else ""
        except Exception:
            pass

        # UNet configuration
        try:
            up = w._unet_panel
            state["unet_arch"]     = up._arch_combo.currentText()    if hasattr(up, "_arch_combo")    else ""
            state["unet_encoder"]  = up._encoder_combo.currentText() if hasattr(up, "_encoder_combo") else ""
            state["unet_ckpt"]     = up._ckpt_edit.text()            if hasattr(up, "_ckpt_edit")     else ""
        except Exception:
            pass

        # Export queue and dataset state
        try:
            ep = w._export_panel
            export_dir = ep.dataset_dir
            state["export_dataset_dir"] = export_dir
            state["export_queue_count"] = len(w._export_queue)
            state["export_val_frac"]    = ep._val_frac.value()
            state["export_test_frac"]   = ep._test_frac.value()
            if export_dir:
                from pathlib import Path as _Path
                state["dataset_finalized"] = (_Path(export_dir) / "splits" / "train.json").exists()
            else:
                state["dataset_finalized"] = False
        except Exception:
            pass

        # Train tab configuration
        try:
            tp = w._train_panel
            state["train_model_type"]    = "yolo" if tp._yolo_radio.isChecked() else "unet"
            state["train_dataset_dir"]   = tp._dir_edit.text().strip()
            state["train_epochs"]        = tp._epochs.value()
            state["train_batch"]         = tp._batch.value()
            state["train_yolo_base"]     = tp._yolo_base.currentText()
            state["train_yolo_imgsz"]    = tp._yolo_imgsz.value()
            state["train_unet_arch"]     = tp._unet_arch.currentText()
            state["train_unet_encoder"]  = tp._unet_encoder.currentText()
            state["train_unet_imgsz"]    = tp._unet_imgsz.value()
        except Exception:
            pass

        # Current image annotations (detailed)
        store = self.annotation_store
        if store is not None:
            anns = list(store)
            state["annotation_count"] = len(anns)
            labels: dict[str, int] = {}
            ann_types: dict[str, int] = {}
            distances: list[dict] = []
            roi_areas: list[dict] = []
            for a in anns:
                lbl = getattr(a, "label", None) or getattr(a, "text", None) or "unknown"
                labels[lbl] = labels.get(lbl, 0) + 1
                typ = getattr(a, "type", "unknown")
                ann_types[typ] = ann_types.get(typ, 0) + 1
                if getattr(a, "type", None) == "distance":
                    distances.append({
                        "distance_nm": round(a.distance_nm, 4),
                        "distance_px": round(a.distance_px, 2),
                        "calibrated": a.calibrated,
                    })
                if getattr(a, "type", None) == "roi" and getattr(a, "area_nm2", 0) > 0:
                    roi_areas.append({
                        "label": getattr(a, "label", ""),
                        "area_nm2": round(a.area_nm2, 2),
                    })
            state["annotation_labels"] = labels
            state["annotation_types"]  = ann_types
            if distances:
                state["distance_measurements"] = distances
            if roi_areas:
                state["roi_areas"] = roi_areas

            # Per-annotation shape metrics for CLU measurement queries
            px = float(self.current_pixel_size_nm)
            shape_measurements: list[dict] = []
            try:
                from acorn_analysis.particle_panel import (
                    _polygon_metrics, _circle_metrics, _rect_metrics,
                )
                for a in anns:
                    t   = getattr(a, "type", "")
                    lbl = getattr(a, "label", "") or ""
                    m: dict = {}
                    if t == "roi":
                        verts = getattr(a, "vertices", [])
                        if len(verts) >= 3:
                            m = _polygon_metrics(verts, px)
                    elif t == "circle":
                        r = getattr(a, "r", 0.0)
                        if r > 0:
                            m = _circle_metrics(r, px)
                    elif t == "rectangle":
                        m = _rect_metrics(
                            getattr(a, "x0", 0), getattr(a, "y0", 0),
                            getattr(a, "x1", 0), getattr(a, "y1", 0),
                            px,
                        )
                    if m:
                        shape_measurements.append({"type": t, "label": lbl, **m})
            except Exception:
                pass
            if shape_measurements:
                state["shape_measurements"] = shape_measurements
        else:
            state["annotation_count"]  = 0
            state["annotation_labels"] = {}

        # Full image list — every loaded image with annotation, pixel size, and queue status.
        # The agent must see this to reason about the whole dataset, navigate by filename,
        # and know which images still need work.
        all_states = self.all_annotation_states
        paths = self.image_paths
        queue_stems: set = set()
        if w is not None and hasattr(w, "_export_queue"):
            queue_stems = {item["stem"] for item in w._export_queue}

        dataset_total   = 0
        dataset_labels: dict[str, int] = {}
        images_annotated = 0
        image_list: list[dict] = []

        for i, path in enumerate(paths):
            img_anns = all_states.get(i, [])
            n = len(img_anns)
            dataset_total += n
            if n > 0:
                images_annotated += 1
            img_labels: dict[str, int] = {}
            for a in img_anns:
                lbl = getattr(a, "label", None) or getattr(a, "text", None) or "unknown"
                img_labels[lbl] = img_labels.get(lbl, 0) + 1
                dataset_labels[lbl] = dataset_labels.get(lbl, 0) + 1
            image_list.append({
                "index":            i,
                "filename":         path.name,
                "annotation_count": n,
                "label_counts":     img_labels,
                "pixel_size_nm":    self.pixel_size_for_index(i),
                "in_export_queue":  path.stem in queue_stems,
            })

        state["image_list"]                = image_list
        state["dataset_total_annotations"] = dataset_total
        state["dataset_images_annotated"]  = images_annotated
        state["dataset_label_counts"]      = dataset_labels
        state["export_queue_filenames"]    = sorted(queue_stems)

        # Finalized dataset statistics (from dataset_stats.json if it exists)
        try:
            import json as _json
            from pathlib import Path as _Path
            ep = w._export_panel
            ds_dir = ep.dataset_dir if ep is not None else ""
            if ds_dir:
                stats_file = _Path(ds_dir) / "dataset_stats.json"
                if stats_file.exists():
                    state["dataset_stats"] = _json.loads(stats_file.read_text())
        except Exception:
            pass

        return state

    def get_thumbnail(self, max_px: int = 1024) -> Optional[str]:
        """Return a base64-encoded JPEG thumbnail of the current displayed image, or None."""
        import base64
        import io
        import numpy as np

        w = self._w()
        if w is None:
            return None
        img = self.current_image
        if img is None or img.raw is None:
            return None

        try:
            from acorn.core.contrast import apply_contrast
            from PIL import Image

            if img.is_color:
                # Color image: use raw RGB directly
                arr8 = (np.clip(img.raw, 0.0, 1.0) * 255).astype(np.uint8)
                pil = Image.fromarray(arr8, mode="RGB")
            else:
                params = w._canvas_widget.canvas._params if hasattr(w._canvas_widget.canvas, "_params") else None
                if params is not None:
                    norm = apply_contrast(img.raw, params)
                else:
                    raw = img.raw.astype(float)
                    raw -= raw.min()
                    if raw.max() > 0:
                        raw /= raw.max()
                    norm = raw
                arr8 = (np.clip(norm, 0.0, 1.0) * 255).astype(np.uint8)
                pil = Image.fromarray(arr8, mode="L").convert("RGB")
            h, w_px = arr8.shape[:2]
            scale = min(1.0, max_px / max(h, w_px))
            if scale < 1.0:
                pil = pil.resize((int(w_px * scale), int(h * scale)), Image.LANCZOS)

            buf = io.BytesIO()
            pil.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode()
        except Exception:
            return None
