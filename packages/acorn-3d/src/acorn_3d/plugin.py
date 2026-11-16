"""3D volume plugin for ACORN: slice viewing, tomogram annotation, propagation, export, rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QFileDialog, QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from acorn_3d.volume_loader import VolumeImage
    from acorn_3d.labels import VolumeAnnotations


class _ZWheelFilter(QObject):
    """Shift + mouse wheel over the canvas steps through z (plain wheel keeps zooming)."""

    def __init__(self, plugin: "VolumePlugin") -> None:
        super().__init__()
        self._plugin = plugin

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 (Qt name)
        if event.type() == QEvent.Type.Wheel and self._plugin._current_vol is not None \
                and event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            dy = event.angleDelta().y() or event.angleDelta().x()
            if dy:
                step = 1 if dy < 0 else -1
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                    step *= 10
                self._plugin._panel._step(step)
                return True
        return False


class VolumePlugin(AcornPlugin):
    TAB_LABEL         = "3D"
    PLUGIN_ID         = "acorn_3d"
    sort_order        = 30
    FLOATING          = True
    FLOATING_TITLE    = "3D Viewer"
    FLOATING_SHORTCUT = "Ctrl+Shift+3"

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel: Optional["VolumePanel"] = None
        self._current_vol: Optional["VolumeImage"] = None
        self._va: Optional["VolumeAnnotations"] = None
        self._z: int = 0
        self._thread = None
        self._own_predictor = None
        self._u8_cache: Optional[np.ndarray] = None    # uint8 stack for SAM 3 (denoised, stretched)
        self._wheel_filter = _ZWheelFilter(self)
        self._wheel_installed_on: list = []

    # ── plugin API ─────────────────────────────────────────────────────────────

    def create_panel(self) -> QWidget:
        from acorn_3d.panel import VolumePanel
        self._panel = VolumePanel()
        p = self._panel
        p.slice_changed.connect(self._on_slice_changed)
        p.view_changed.connect(self._on_view_changed)
        p.projection_requested.connect(self._on_projection_requested)
        p.segment_slice_requested.connect(self._on_segment_slice)
        p.propagate_requested.connect(self._on_propagate)
        p.clear_class_requested.connect(self._on_clear_class)
        p.export_requested.connect(self._on_export)
        p.render_requested.connect(self._on_render)
        return p

    def setup_menus(self, menubar) -> None:
        self._context.register_menu_action("File", "Open as Volume (MRC / TIFF)...", self._open_volume_dialog)

    def teardown(self) -> None:
        self._save_current_slice()
        self._save_sidecar()

    # ── loading ────────────────────────────────────────────────────────────────

    def _open_volume_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(None, "Open Volume", "", "Volume files (*.mrc *.mrcs *.rec *.tif *.tiff)")
        if path:
            self._load_volume(Path(path))

    def _load_volume(self, path: Path) -> None:
        from acorn_3d.volume_loader import VolumeImage
        from acorn_3d.labels import VolumeAnnotations
        if self._current_vol is not None:
            self._save_current_slice(); self._save_sidecar()
        try:
            vol = VolumeImage.from_file(path)
        except Exception as exc:
            QMessageBox.critical(None, "Volume load error", str(exc))
            return
        self._current_vol = vol
        self._u8_cache = None
        self._install_wheel_filter()
        self._va = VolumeAnnotations.load(path, vol.n_slices)
        self._va.pixel_size_nm = vol.pixel_size
        self._va.voxel_depth_nm = vol.meta.voxel_depth_nm
        for lab in self._va.labels():
            self._panel.add_class(lab)
        self._panel.set_volume(vol)
        # Detach the main window's per-image autosave while a volume is shown, so slice
        # ROIs are not written into whatever 2D image was open before.
        w = getattr(self._context, "_w", None)
        mw = w() if callable(w) else None
        if mw is not None and hasattr(mw, "_img_idx"):
            try:
                mw._img_idx = -1
            except Exception:
                pass
        self._z = 0
        self._show_slice(0, save_previous=False)
        self._context.set_status(f"Volume loaded: {vol.filename}  ({vol.n_slices} slices, "
                                 f"{self._va.count()} saved ROIs on {len(self._va.slices_with_annotations())} slices)")

    def _install_wheel_filter(self) -> None:
        """Shift+wheel on the canvas = z step. Installed on the canvas widget and its children."""
        cw = self._context.canvas_widget()
        if cw is None:
            return
        targets = [cw] + [c for c in cw.findChildren(QWidget)]
        for t in targets:
            if t not in self._wheel_installed_on:
                t.installEventFilter(self._wheel_filter)
                self._wheel_installed_on.append(t)

    # ── slice navigation with per-z annotation stores ─────────────────────────

    def _store(self):
        return self._context.annotation_store

    def _save_current_slice(self) -> None:
        store = self._store()
        if self._va is None or store is None:
            return
        self._va.set(self._z, list(store))

    def _show_slice(self, z: int, save_previous: bool = True) -> None:
        if self._current_vol is None:
            return
        if save_previous:
            self._save_current_slice()
        self._z = int(z)
        self._current_vol.set_slice(self._z)
        cw = self._context.canvas_widget()
        if cw is not None:
            cw.canvas.load_image(self._current_vol, self._context.current_contrast_params)
            store = cw.canvas.store
            if store is not None:
                cw.canvas._loading = True
                try:
                    store.replace_all(self._va.get(self._z) if self._va else [])
                finally:
                    cw.canvas._loading = False
                try:
                    cw.canvas.renderer.render_noblit(store, cw.canvas)
                except Exception:
                    pass
                self._context.annotations_changed.emit(store)
        self._panel.set_annotation_summary(self._summary())
        self._context.slice_changed.emit(self._z)

    def _summary(self) -> str:
        if self._va is None:
            return ""
        zs = self._va.slices_with_annotations()
        here = len(self._va.get(self._z))
        return f"ROIs on this slice: {here}   |   {self._va.count()} ROIs on {len(zs)} slices" + \
               (f"  (z {zs[0]}..{zs[-1]})" if zs else "")

    def _on_slice_changed(self, z: int) -> None:
        self._show_slice(z)
        self._save_sidecar()

    def _on_view_changed(self, slab: int, smooth_px: float) -> None:
        if self._current_vol is None:
            return
        self._current_vol.slab = int(slab)
        self._current_vol.smooth_px = float(smooth_px)
        self._show_slice(self._z, save_previous=True)

    def _save_sidecar(self) -> None:
        if self._va is None or self._current_vol is None:
            return
        try:
            self._va.save()
        except OSError as exc:
            self._context.set_status(f"Could not save volume annotations: {exc}", 8000)

    # ── SAM 3 on one slice ────────────────────────────────────────────────────

    def _predictor(self):
        """The main window's SAM predictor if loaded, else a private SAM 3 image predictor."""
        w = getattr(self._context, "_w", None)
        mw = w() if callable(w) else None
        pred = getattr(mw, "_sam_predictor", None) if mw is not None else None
        if pred is not None and getattr(pred, "is_loaded", False):
            return pred
        if self._own_predictor is None:
            from acorn.core.sam_predictor import SAMPredictor
            self._own_predictor = SAMPredictor(backend="sam3")
            self._own_predictor.load_model()
        return self._own_predictor

    def _slice_u8(self, z: int) -> np.ndarray:
        from acorn_3d.propagate import slices_to_uint8
        self._current_vol.set_slice(z)
        img = self._current_vol.raw
        self._current_vol.set_slice(self._z)
        return slices_to_uint8(img[None], denoise_sigma=1.0)[0]

    def _on_segment_slice(self, prompt: str, label: str, confidence: float) -> None:
        if self._current_vol is None:
            return
        z = self._z
        img8 = self._slice_u8(z)
        self._panel.set_busy(True)
        self._panel.set_seg_status(f"SAM 3 text prompt '{prompt}' on slice {z}...")

        def work():
            pred = self._predictor()
            masks, phrase = pred.predict_text(img8, prompt, confidence=confidence)
            return masks, phrase

        def done(result):
            masks, phrase = result
            self._add_masks_as_rois(z, masks, label)
            self._panel.set_seg_status(f"'{phrase}': {len(masks)} mask(s) added as '{label}' on slice {z}. "
                                       "Edit or delete them, then propagate.")
            self._panel.set_busy(False)

        self._run_in_thread(work, done)

    def _add_masks_as_rois(self, z: int, masks, label: str) -> None:
        from acorn_3d.labels import masks_to_rois
        from acorn.render.palette import color_for_label
        rois = masks_to_rois(masks, label, color_for_label(label), self._current_vol.pixel_size)
        existing = self._va.get(z)
        self._va.set(z, existing + rois)
        self._panel.add_class(label)
        if z == self._z:
            self._show_slice(z, save_previous=False)
        self._save_sidecar()

    # ── propagation through z ─────────────────────────────────────────────────

    def _u8_stack(self, z_from: int, z_to: int, status) -> np.ndarray:
        from acorn_3d.propagate import slices_to_uint8
        vol = self._current_vol
        n = z_to - z_from + 1
        out = []
        for i, z in enumerate(range(z_from, z_to + 1)):
            vol.set_slice(z)
            out.append(np.asarray(vol.raw, dtype=np.float32))
            if status and i % 25 == 0:
                status(f"reading slices {i + 1}/{n}")
        vol.set_slice(self._z)
        return slices_to_uint8(np.stack(out), denoise_sigma=1.0)

    def _on_propagate(self, mode: str, label: str, z_from: int, z_to: int) -> None:
        if self._current_vol is None:
            return
        z_from, z_to = sorted((int(z_from), int(z_to)))
        z0 = self._z
        if not (z_from <= z0 <= z_to):
            QMessageBox.information(None, "Propagate", "Move to a slice inside the z range first; "
                                                          "propagation starts from the current slice.")
            return
        self._save_current_slice()
        prompt = self._panel.prompt_text
        seeds_rois = [a for a in self._va.get(z0) if (getattr(a, "label", "") or "Unlabelled").lower() == label.lower()]
        if mode == "track" and not seeds_rois:
            QMessageBox.information(None, "Propagate", f"No '{label}' ROIs on slice {z0} to track.")
            return
        h, w = self._current_vol.shape
        self._panel.set_busy(True)
        self._panel.set_seg_status("starting propagation...")

        def work(status=None):
            from acorn_3d.propagate import shared_propagator, ZPrompt
            from skimage.draw import polygon as _poly
            u8 = self._u8_stack(z_from, z_to, status)
            zp = shared_propagator("cuda")
            if status:
                status("loading SAM 3 video model...")
            zp.load()
            prog = (lambda d, n: status(f"propagating {d}/{n}")) if status else None
            if mode == "text":
                res = zp.run(u8, [ZPrompt(z=z0 - z_from, text=prompt)], progress=prog)
            else:
                seeds = []
                for i, a in enumerate(seeds_rois):
                    m = np.zeros((h, w), bool)
                    xs = np.array([v[0] for v in a.vertices]); ys = np.array([v[1] for v in a.vertices])
                    rr, cc = _poly(ys, xs, shape=(h, w)); m[rr, cc] = True
                    if m.any():
                        seeds.append((z0 - z_from, 100 + i, m))
                res = zp.run_masks(u8, seeds, warmup_text=prompt, progress=prog)
            return res

        def done(res):
            from acorn_3d.labels import masks_to_rois
            from acorn.render.palette import color_for_label
            col = color_for_label(label)
            px = self._current_vol.pixel_size
            n_rois = 0
            for zz, objs in res.masks.items():
                z = zz + z_from
                keep = [a for a in self._va.get(z) if (getattr(a, "label", "") or "").lower() != label.lower()]
                rois = masks_to_rois(list(objs.values()), label, col, px)
                for r, oid in zip(rois, objs.keys()):
                    r.stats = dict(r.stats or {}, track_id=int(oid), z=int(z))
                self._va.set(z, keep + rois)
                n_rois += len(rois)
            self._va.prompts.append({"mode": mode, "label": label, "prompt": prompt, "z0": z0,
                                     "z_from": z_from, "z_to": z_to, "objects": len(res.object_ids())})
            self._save_sidecar()
            self._show_slice(self._z, save_previous=False)
            self._panel.set_seg_status(f"Propagated '{label}': {len(res.object_ids())} object(s), "
                                       f"{n_rois} ROIs on {len(res.masks)} slices.")
            self._panel.set_busy(False)

        self._run_in_thread(work, done, with_status=True)

    def _on_clear_class(self, label: str) -> None:
        if self._va is None:
            return
        for z in self._va.slices_with_annotations():
            self._va.set(z, [a for a in self._va.get(z) if (getattr(a, "label", "") or "").lower() != label.lower()])
        self._save_sidecar()
        self._show_slice(self._z, save_previous=False)
        self._panel.set_seg_status(f"Removed every '{label}' ROI.")

    # ── export / render ───────────────────────────────────────────────────────

    def _label_volume(self):
        self._save_current_slice()
        h, w = self._current_vol.shape
        return self._va.to_label_volume((h, w))

    def _default_out(self, suffix: str) -> Path:
        p = self._current_vol.filepath
        return p.with_name(f"{p.stem}_labels{suffix}")

    def _on_export(self) -> None:
        if self._current_vol is None or self._va is None or self._va.count() == 0:
            QMessageBox.information(None, "Export", "No ROIs to export yet.")
            return
        from acorn_3d.labels import write_label_mrc
        path, _ = QFileDialog.getSaveFileName(None, "Save label volume", str(self._default_out(".mrc")), "MRC (*.mrc)")
        if not path:
            return
        vol, names = self._label_volume()
        vz = self._current_vol.meta.voxel_depth_nm or self._current_vol.pixel_size
        write_label_mrc(vol, Path(path), (vz, self._current_vol.pixel_size, self._current_vol.pixel_size))
        Path(path).with_suffix(".classes.txt").write_text("".join(f"{k}\t{v}\n" for k, v in sorted(names.items())))
        self._context.set_status(f"Label volume written: {path}  classes: {names}", 8000)

    def _on_render(self, style: str, seconds: float, zexag: float) -> None:
        if self._current_vol is None or self._va is None or self._va.count() == 0:
            QMessageBox.information(None, "Render", "Annotate and propagate first; there is nothing to render.")
            return
        path, _ = QFileDialog.getSaveFileName(None, "Save movie", str(self._default_out(f"_{style}.mp4")), "MP4 (*.mp4)")
        if not path:
            return
        vol, names = self._label_volume()
        vz = self._current_vol.meta.voxel_depth_nm or self._current_vol.pixel_size
        voxel = (vz, self._current_vol.pixel_size, self._current_vol.pixel_size)
        self._panel.set_busy(True)

        def work(status=None):
            from acorn_3d.render import render_spin_movie
            return render_spin_movie(vol, voxel, Path(path), names, style=style, seconds=seconds,
                                     z_exaggeration=zexag, progress=status)

        def done(res):
            self._panel.set_busy(False)
            self._panel.set_seg_status(f"Movie written: {res['mp4']}" + (f", GIF: {res['gif']}" if res.get("gif") else ""))
            self._context.set_status(f"Rendered {res['mp4']}", 8000)

        self._run_in_thread(work, done, with_status=True)

    # ── threading ─────────────────────────────────────────────────────────────

    def _run_in_thread(self, fn, on_done, with_status: bool = False) -> None:
        from acorn.gui.threads import SAMThread
        t = SAMThread(lambda: fn(status=t.status.emit) if with_status else fn())
        self._thread = t
        t.finished.connect(on_done)
        t.status.connect(self._panel.set_seg_status)

        def failed(msg: str):
            self._panel.set_busy(False)
            self._panel.set_seg_status(f"Error: {msg}")
            QMessageBox.critical(None, "acorn_3d", msg)
        t.error.connect(failed)
        t.start()

    # ── projection (unchanged) ────────────────────────────────────────────────

    def _on_projection_requested(self, method: str, z_from: int, z_to: int) -> None:
        if self._current_vol is None:
            return
        from acorn.core.dm4_loader import DM4Image
        try:
            arr = self._current_vol.projection(method=method, z_from=z_from, z_to=z_to)
        except Exception as exc:
            QMessageBox.critical(None, "Projection error", str(exc))
            return
        vol = self._current_vol
        proj = DM4Image()
        proj.raw = arr.astype(np.float32)
        proj.meta.pixel_size = vol.pixel_size
        proj.meta.pixel_size_from_header = vol.meta.pixel_size_from_header
        proj.meta.shape = arr.shape
        proj.meta.filepath = vol.filepath
        proj.meta.filename = f"{vol.filename}_proj_{method}_{z_from}-{z_to}"
        cw = self._context.canvas_widget()
        if cw is not None:
            cw.canvas.load_image(proj, self._context.current_contrast_params)
        self._context.set_status(f"Projection ({method}, z={z_from}-{z_to}) displayed. Use Export tab to save.")
