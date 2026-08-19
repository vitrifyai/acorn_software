"""
Background worker threads.

Every long job in ACORN - validating a folder, loading a stack, running a model,
training, batch export - happens on one of these so the window keeps repainting.
They were defined inline in main_window.py, where they had nothing to do with the
window itself; SAMThread in particular is a generic worker that YOLO, UNet and
training all use despite the name.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from PyQt6.QtCore import QThread, pyqtSignal

from acorn.core.annotations import AnnotationStore
from acorn.core.dm4_loader import DM4Image


# ── background loader thread ──────────────────────────────────────────────────

class LoadThread(QThread):
    """Validate image paths in a background thread; actual loading is lazy."""

    progress = pyqtSignal(int, int, str)            # n_done, n_total, filename
    finished = pyqtSignal(list, list)               # valid_paths, error_pairs

    def __init__(self, paths: list[Path], parent=None):
        super().__init__(parent)
        self._paths = paths

    def run(self) -> None:
        valid = []
        errors = []
        n = len(self._paths)
        for i, p in enumerate(self._paths):
            self.progress.emit(i, n, p.name)
            if p.exists():
                valid.append(p)
            else:
                errors.append((p, "File not found"))
        self.finished.emit(valid, errors)


# ── background training-export thread ────────────────────────────────────────

class TrainingThread(QThread):
    """Run training export on a background thread so the GUI stays responsive."""

    progress     = pyqtSignal(str)       # status message
    progress_int = pyqtSignal(int, int)  # (current_tile, total_tiles)
    finished     = pyqtSignal(dict)      # result summary dict
    error        = pyqtSignal(str)       # error message string

    def __init__(self, dataset_dir, dm4img, store_snapshot, params, config, parent=None):
        super().__init__(parent)
        self._dataset_dir    = dataset_dir
        self._dm4img         = dm4img
        self._store_snapshot = store_snapshot
        self._params         = params
        self._config         = config

    def run(self) -> None:
        try:
            from acorn.export.training_exporter import add_image, TrainingConfig
            from acorn.core.annotations import AnnotationStore
            # Reconstruct a store from the snapshot
            store = AnnotationStore()
            store.replace_all(self._store_snapshot)

            def _progress_cb(current: int, total: int) -> None:
                self.progress.emit(f"Exporting tile {current}/{total}…")
                self.progress_int.emit(current, total)

            result = add_image(
                self._dataset_dir, self._dm4img, store, self._params, self._config,
                progress_callback=_progress_cb,
            )
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ── batch training export thread ──────────────────────────────────────────────

class BatchExportThread(QThread):
    """Process a queue of annotated images into a training dataset sequentially."""

    image_status    = pyqtSignal(str)        # human-readable status per image
    image_progress  = pyqtSignal(int, int)  # (current_image, total_images) overall
    tile_progress   = pyqtSignal(int, int)  # (current_tile, total_tiles) within image
    item_done       = pyqtSignal(int, str)  # (item_idx, stem) when one image finishes
    finished        = pyqtSignal(list)      # list of result dicts (one per image)
    error           = pyqtSignal(int, str)  # (item_idx, error message) — continues

    def __init__(self, items: list[dict], dataset_dir: str, config, parent=None):
        super().__init__(parent)
        self._cancelled   = False
        self._items       = items        # list of {dm4img, store_snapshot, params, stem}
        self._dataset_dir = dataset_dir
        self._config      = config

    def cancel(self) -> None:
        """Request the export stop at the next image / tile boundary."""
        self._cancelled = True

    def run(self) -> None:
        from acorn.export.training_exporter import add_image
        from acorn.core.annotations import AnnotationStore

        results: list[dict] = []
        n = len(self._items)

        for i, item in enumerate(self._items):
            if self._cancelled:
                self.image_status.emit(f"Export cancelled after {i} of {n} image(s).")
                break
            stem = item["stem"]
            self.image_progress.emit(i + 1, n)
            self.image_status.emit(f"[{i + 1}/{n}] {stem} — preparing…")

            # Negative images imported by path are loaded on demand here
            dm4img = item["dm4img"]
            if dm4img is None:
                try:
                    dm4img = DM4Image.from_file(Path(item["path"]))
                except Exception as exc:
                    self.error.emit(i, f"{stem}: could not load image — {exc}")
                    continue

            store = AnnotationStore()
            store.replace_all(item["store_snapshot"])

            # Unannotated images (negatives) must never skip empty tiles —
            # that would produce zero output. Override skip_empty_tiles for them.
            n_rois = sum(1 for a in store if getattr(a, "type", None) == "roi")
            if n_rois == 0 and self._config.skip_empty_tiles:
                from dataclasses import replace as _dc_replace
                cfg = _dc_replace(self._config, skip_empty_tiles=False)
                self.image_status.emit(
                    f"[{i + 1}/{n}] {stem} — no annotations, exporting as negative tiles…"
                )
            else:
                cfg = self._config

            def _cb(current: int, total: int, _s=stem, _i=i, _n=n) -> None:
                self.image_status.emit(f"[{_i + 1}/{_n}] {_s} — tile {current}/{total}")
                self.tile_progress.emit(current, total)

            try:
                result = add_image(
                    self._dataset_dir,
                    dm4img,
                    store,
                    item["params"],
                    cfg,
                    progress_callback=_cb,
                    should_cancel=lambda: self._cancelled,
                )
                results.append(result)
                self.item_done.emit(i, stem)
            except Exception as exc:
                self.error.emit(i, f"{stem}: {exc}")

        self.finished.emit(results)


# ── background SAM thread ─────────────────────────────────────────────────────

class SAMThread(QThread):
    """Run any SAM inference or model-load call on a background thread."""

    finished = pyqtSignal(object)   # result (type depends on task)
    error    = pyqtSignal(str)
    status   = pyqtSignal(str)      # intermediate status messages (e.g. download progress)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            self.finished.emit(self._fn())
        except Exception as exc:
            self.error.emit(str(exc))


# ── background image-load thread ──────────────────────────────────────────────

class ImageLoadThread(QThread):
    """Load a DM4Image and pre-compute contrast normalisation in a background thread.

    Both the file I/O (DM4Image.from_file) and the contrast calculation
    (apply_contrast, which can involve a slow bandpass FFT) happen off the
    main thread so the GUI stays responsive throughout.
    """

    finished = pyqtSignal(int, object, object)  # (idx, DM4Image, norm_array)
    error    = pyqtSignal(int, str)             # (idx, error message)

    def __init__(self, idx: int, path: Path, contrast_params, parent=None):
        super().__init__(parent)
        self._idx     = idx
        self._path    = path
        self._params  = contrast_params

    def run(self) -> None:
        try:
            from acorn.core.contrast import apply_contrast
            from acorn.render.canvas import _DISPLAY_MAX_DIM
            img = DM4Image.from_file(self._path)

            # Compute contrast on full-res raw so the precomputed norm is
            # full-resolution (required for correct ROI stats, line profiles,
            # and display export).  Scale spatial sigma/cutoff params by the
            # same stride factor used for display so bandpass / Fourier methods
            # give the same visual result as they would on the downsampled image.
            raw = img.raw
            h, w = raw.shape[:2]
            step = max(1, (max(h, w) + _DISPLAY_MAX_DIM - 1) // _DISPLAY_MAX_DIM)
            if step > 1:
                from dataclasses import replace
                scaled_params = replace(self._params,
                    bp_low_sigma=self._params.bp_low_sigma * step,
                    bp_high_sigma=self._params.bp_high_sigma * step,
                    fbp_hp_px=self._params.fbp_hp_px * step,
                    fbp_lp_px=self._params.fbp_lp_px * step,
                )
            else:
                scaled_params = self._params
            norm = apply_contrast(raw, scaled_params)
            self.finished.emit(self._idx, img, norm)
        except Exception as exc:
            self.error.emit(self._idx, str(exc))


# ── background frame-processing thread ───────────────────────────────────────

class FrameProcessThread(QThread):
    """Run frame averaging/motion correction off the main thread."""

    finished         = pyqtSignal(object)   # np.ndarray averaged result
    shifts_available = pyqtSignal(object)   # np.ndarray (n,2) shifts — motion_corrected only
    error            = pyqtSignal(str)

    def __init__(self, frames, method: str, dose_per_frame: float = 1.0,
                 pixel_size_nm: float = 1.0, parent=None) -> None:
        super().__init__(parent)
        self._frames         = frames
        self._method         = method
        self._dose_per_frame = dose_per_frame
        self._pixel_size_nm  = pixel_size_nm

    def run(self) -> None:
        try:
            from acorn.core.frame_processor import (
                mean_average, motion_correct_frames, dose_weighted_average,
            )
            if self._method == "motion_corrected":
                result, shifts = motion_correct_frames(self._frames)
                self.finished.emit(result)
                self.shifts_available.emit(shifts)
            elif self._method == "dose_weighted":
                result = dose_weighted_average(
                    self._frames, self._dose_per_frame, self._pixel_size_nm
                )
                self.finished.emit(result)
            else:
                self.finished.emit(mean_average(self._frames))
        except Exception as exc:
            self.error.emit(str(exc))
