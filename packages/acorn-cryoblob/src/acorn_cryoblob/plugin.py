"""ACORN plugin that exposes CryoBLOB in the GUI."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin
from acorn_cryoblob.thread import SUPPORTED_EXTS, CryoBlobThread

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext

# A folder pick that recurses into a parent dir can resolve hundreds of images.
# CryoBLOB detection is CPU-bound (skimage/scipy stages), so a big accidental
# sweep pins a core for many minutes and holds the Run button disabled the whole
# time. Warn (interactive) or refuse (CLU) above this count so a mistaken folder
# pick can't silently lock the UI.
_MAX_FILES_WARN = 50


# Fallback detection params for a CLU-driven run when the panel isn't built yet
# (in the running app the panel supplies the live values via current_params()).
_CLU_DEFAULTS = {
    "mode": "loaded", "run_mode": "final", "detection_mode": "log",
    "folder": "", "output_csv": "", "pixel_size_nm": 0.0,
    "blob_downscale": 2.0, "min_sigma": 3.0, "max_sigma": 30.0, "blob_step": 1.0,
    "threshold_rel": 0.1, "max_detections": 300, "refine_sizes": True,   # 0.1 is more sensitive than the panel's 0.22 default -> better out-of-box recall for CLU runs
    "size_scale": 1.0, "ridge_threshold": 0.006, "ridge_scales": 20,
    "min_marker_distance": 4.0, "use_ridge_detection": False, "use_watershed": False,
    "gblur": 2, "background": 10, "apply_filter": 1, "exponential": False,
    "logarizer": True, "stream_large_files": False, "cache_results": True,
    "add_annotations": True,
    # auto | dark | light — CryoBLOB finds density minima, so inverted data
    # returns nothing unless the image is flipped first.
    "contrast_polarity": "auto",
}


class CryoBlobPlugin(AcornPlugin):
    TAB_LABEL              = "CryoBLOB"
    PLUGIN_ID              = "acorn_cryoblob"
    WORKFLOW_STAGE         = "Annotate"
    WORKFLOW_SECTION_LABEL = "Blob Detection"

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._thread = None
        self._last_params = {}
        context.image_loaded.connect(self._on_image_loaded)
        context.action_requested.connect(self._on_action_requested)   # CLU: "run cryoblob"

    @property
    def sort_order(self) -> int:
        return 30

    def create_panel(self) -> QWidget:
        from acorn_cryoblob.panel import CryoBlobPanel

        self._panel = CryoBlobPanel()
        self._panel.run_requested.connect(self._on_run_requested)
        self._panel.clear_requested.connect(self._on_clear_requested)
        self._sync_current_image()
        return self._panel

    def teardown(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.quit()
            self._thread.wait(2000)

    def _on_image_loaded(self, _img) -> None:
        self._sync_current_image()

    def _sync_current_image(self) -> None:
        if self._panel is None:
            return
        img = self._context.current_image
        path = str(img.filepath) if img is not None else None
        is_supported = bool(path and Path(path).suffix.lower() in SUPPORTED_EXTS)
        self._panel.set_current_image(path, is_supported)
        self._panel.set_pixel_size(self._context.current_pixel_size_nm)

    def _on_action_requested(self, action: str, params: dict) -> None:
        """CLU entry point. `run_cryoblob` launches detection with the panel's
        live settings (or built-in defaults) plus any overrides CLU supplies —
        JAX/GPU blob detection, no SAM/torch involved."""
        if action != "run_cryoblob":
            return
        print(f"[CryoBLOB] action received: run_cryoblob params={params} panel={'yes' if self._panel else 'None'}", flush=True)
        if self._thread is not None and self._thread.isRunning():
            self._context.set_status("CryoBLOB is already running.")
            return
        merged = self._panel.current_params() if self._panel is not None else dict(_CLU_DEFAULTS)
        for key, value in (params or {}).items():
            if key in merged:
                merged[key] = value
        merged.setdefault("add_annotations", True)
        self._on_run_requested(merged)

    def _on_run_requested(self, params: dict) -> None:
        self._last_params = dict(params)
        files = self._resolve_files(params)
        print(f"[CryoBLOB] resolved {len(files)} file(s) for mode={params.get('mode')} "
              f"folder={params.get('folder')!r}", flush=True)
        if not files:
            msg = "No supported image files were found for the selected source."
            if self._panel is not None:
                QMessageBox.information(self._panel, "CryoBLOB", msg)
            self._context.set_status(f"CryoBLOB: {msg}")
            return

        # Guard against an accidental huge sweep (e.g. a folder pick that recurses
        # into a parent holding many run dirs). Detection is CPU-bound and blocks
        # the Run button for the whole run, so a mistaken pick can lock the UI.
        if len(files) > _MAX_FILES_WARN:
            if self._panel is not None:
                reply = QMessageBox.question(
                    self._panel,
                    "CryoBLOB — large batch",
                    f"This will process {len(files)} images. Detection runs on CPU and "
                    f"can take a long time; the Run button stays disabled until it "
                    f"finishes.\n\n"
                    f"Tip: point at a single run's images/ folder to narrow the batch.\n\n"
                    f"Proceed with all {len(files)} images?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply != QMessageBox.StandardButton.Yes:
                    self._context.set_status(
                        f"CryoBLOB: cancelled ({len(files)} images — batch too large)."
                    )
                    return
            else:
                # CLU / headless run: refuse rather than silently grind for minutes.
                msg = (
                    f"CryoBLOB resolved {len(files)} images (> {_MAX_FILES_WARN}). "
                    f"Point at a single run's images/ folder to narrow the batch, "
                    f"then run again."
                )
                self._context.set_status(f"CryoBLOB: {msg}")
                print(f"[CryoBLOB] refused large batch: {len(files)} files", flush=True)
                return

        output_csv = params["output_csv"] or self._default_output_csv(params, files)

        # Panel is None when the CryoBLOB tab was never opened — CLU runs must still work.
        if self._panel is not None:
            self._panel.set_running(True)
            self._panel.show_progress(0, f"Queued {len(files)} file(s) for CryoBLOB.")

        self._thread = CryoBlobThread(
            files=files,
            output_csv=output_csv,
            pixel_size_nm=params["pixel_size_nm"],
            # CryoBLOB re-reads files from disk, so it has to be told the
            # binning the window applied; otherwise it analyses native pixels
            # while the operator believes 4x is in force.
            bin_factor=getattr(self._context, "bin_factor", 1),
            run_mode=params["run_mode"],
            detection_mode=params["detection_mode"],
            blob_downscale=params["blob_downscale"],
            min_sigma=params["min_sigma"],
            max_sigma=params["max_sigma"],
            blob_step=params["blob_step"],
            threshold_rel=params["threshold_rel"],
            max_detections=params["max_detections"],
            refine_sizes=params["refine_sizes"],
            contrast_polarity=params.get("contrast_polarity", "auto"),
            size_scale=params["size_scale"],
            ridge_threshold=params["ridge_threshold"],
            ridge_scales=params["ridge_scales"],
            min_marker_distance=params["min_marker_distance"],
            use_ridge_detection=params["use_ridge_detection"],
            use_watershed=params["use_watershed"],
            stream_large_files=params["stream_large_files"],
            exponential=params["exponential"],
            logarizer=params["logarizer"],
            gblur=params["gblur"],
            background=params["background"],
            apply_filter=params["apply_filter"],
            cache_results=params["cache_results"],
        )
        if self._panel is not None:
            self._thread.progress.connect(self._panel.show_progress)
        self._thread.finished.connect(self._on_finished)
        self._thread.error.connect(self._on_error)
        self._thread.start()

    def _resolve_files(self, params: dict) -> list[str]:
        """Resolve the image list for a run. Robust + cascading so a CLU call
        finds images regardless of the exact mode/path it picked: folder scans
        RECURSIVELY (generated sim images live in an images/ subfolder), and each
        mode falls back to the others rather than returning nothing."""
        mode = params.get("mode", "loaded")

        def from_current() -> list[str]:
            img = self._context.current_image
            if img is None:
                return []
            path = Path(img.filepath)
            return [str(path)] if path.suffix.lower() in SUPPORTED_EXTS else []

        def from_loaded() -> list[str]:
            return [
                str(path)
                for path in self._context.image_paths
                if path.suffix.lower() in SUPPORTED_EXTS
            ]

        def from_folder(raw: str) -> list[str]:
            folder = Path(raw or "").expanduser()
            if folder.is_file() and folder.suffix.lower() in SUPPORTED_EXTS:
                return [str(folder)]
            if not folder.is_dir():
                return []
            # rglob: works whether the caller passed the run dir or its images/ subdir.
            return [
                str(path)
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
            ]

        if mode == "current":
            return from_current() or from_loaded()
        if mode == "folder":
            return from_folder(params.get("folder", "")) or from_loaded() or from_current()
        # "loaded" (default)
        return from_loaded() or from_current()

    def _default_output_csv(self, params: dict, files: list[str]) -> str:
        mode = params["mode"]
        if mode == "folder":
            return str(Path(params["folder"]).expanduser() / "cryoblob_results.csv")
        if len(files) == 1:
            path = Path(files[0])
            return str(path.with_name(f"{path.stem}_cryoblob.csv"))
        return str(Path(files[0]).parent / "cryoblob_results.csv")

    def _on_finished(self, df, output_csv: str, failures: list[tuple[str, str]]) -> None:
        if self._panel is not None:
            self._panel.set_running(False)
            self._panel.show_results(df, output_csv, failures)
        n_added = 0
        written: list[str] = []
        print(f"[CryoBLOB] finished: {len(df)} blob(s), {len(failures)} failure(s), "
              f"add_annotations={self._last_params.get('add_annotations')}", flush=True)
        if self._last_params.get("add_annotations"):
            n_added = self._annotate_current_image(df)      # live overlay on current image
            written = self._write_all_sidecars(df)          # persist for ALL images + training
            print(f"[CryoBLOB] annotated current image: {n_added} ROI(s); wrote sidecars for {len(written)} image(s)", flush=True)
            # Tell the app to drop its in-memory annotation cache for these images and
            # reload the current one from disk, so the new ROIs actually appear (and on
            # navigation too) instead of the stale cached state.
            if written:
                try:
                    self._context.action_requested.emit("reload_annotations_from_disk", {"paths": written})
                except Exception as exc:
                    print(f"[CryoBLOB] reload emit failed: {exc}", flush=True)
        status = f"CryoBLOB complete — {len(df)} blob(s) saved to {output_csv}"
        if written:
            status += f" | wrote annotations for {len(written)} image(s)"
        elif n_added:
            status += f" | added {n_added} ROI(s) to current image"
        if failures:
            status += f" ({len(failures)} file failures)"
        self._context.set_status(status)

    def _on_error(self, message: str) -> None:
        print(f"[CryoBLOB] ERROR: {message}", flush=True)
        if self._panel is not None:
            self._panel.set_running(False)
            QMessageBox.critical(self._panel, "CryoBLOB", message)
        self._context.set_status(f"CryoBLOB failed: {message}")

    def _on_clear_requested(self) -> None:
        n_removed = self._clear_current_annotations()
        if self._panel is not None:
            self._panel.clear_results()
        self._context.set_status(f"Cleared {n_removed} CryoBLOB ROI(s) from current image.")

    @staticmethod
    def _row_path(row):
        try:
            return Path(str(row["File Location"])).expanduser().resolve()
        except Exception:
            return None

    def _rows_to_annotations(self, rows, img_path: Path) -> list:
        """Convert CryoBLOB result rows for one image into ACORN annotation objects."""
        from acorn.core.annotations import LineAnnotation, ROIAnnotation
        try:
            from acorn.core.annotations import PolylineAnnotation
        except ImportError:
            PolylineAnnotation = None      # older/forked ACORN lacks it; fall back to LineAnnotation

        def _circle_poly(cx, cy, r, n=20):
            return [(cx + r * math.cos(2 * math.pi * k / n),
                     cy + r * math.sin(2 * math.pi * k / n)) for k in range(n)]

        new_annotations = []
        for row in rows:
            y_px_nm = float(row.get("Pixel Size Y (nm/px)", 0.0) or 0.0)
            x_px_nm = float(row.get("Pixel Size X (nm/px)", 0.0) or 0.0)
            if y_px_nm <= 0 or x_px_nm <= 0:
                if img_path.suffix.lower() in {".mrc", ".mrcs"}:
                    fallback_px_nm = float(self._context.current_pixel_size_nm or 1.0)
                else:
                    fallback_px_nm = float(self._last_params.get("pixel_size_nm") or 1.0)
                y_px_nm = y_px_nm if y_px_nm > 0 else fallback_px_nm
                x_px_nm = x_px_nm if x_px_nm > 0 else fallback_px_nm

            detection_type = str(row.get("Detection Type", ""))
            size_px_nm = math.sqrt(y_px_nm * x_px_nm)
            if detection_type == "ridge" and row.get("Ridge Length (nm)"):
                ridge_path_raw = row.get("Ridge Path (nm)")
                ridge_vertices = []
                if ridge_path_raw:
                    try:
                        ridge_path_nm = json.loads(ridge_path_raw)
                        ridge_vertices = [
                            (float(x_nm) / x_px_nm, float(y_nm) / y_px_nm)
                            for y_nm, x_nm in ridge_path_nm
                        ]
                    except Exception:
                        ridge_vertices = []
                if len(ridge_vertices) >= 2 and PolylineAnnotation is not None:
                    ann = PolylineAnnotation(
                        vertices=ridge_vertices,
                        color="#00FF88",
                        linewidth=2.0,
                        linestyle="--",
                    )
                else:
                    ann = LineAnnotation(
                        p1=(
                            float(row["Ridge Start X (nm)"]) / x_px_nm,
                            float(row["Ridge Start Y (nm)"]) / y_px_nm,
                        ),
                        p2=(
                            float(row["Ridge End X (nm)"]) / x_px_nm,
                            float(row["Ridge End Y (nm)"]) / y_px_nm,
                        ),
                        color="#00FF88",
                        linewidth=2.0,
                        linestyle="--",
                    )
            else:
                x_px = float(row["Center X (nm)"]) / x_px_nm
                y_px = float(row["Center Y (nm)"]) / y_px_nm
                radius_px = max(3.0, 0.5 * float(row["Size (nm)"]) / max(size_px_nm, 1e-9))
                # ROI circle-polygon (not CircleAnnotation): ROI carries stats+label
                # so the CryoBLOB marker persists to the sidecar and exports to masks.
                ann = ROIAnnotation(
                    vertices=_circle_poly(x_px, y_px, radius_px),
                    color="#00FF88",
                    linewidth=1.5,
                    label="",              # no on-image text; CryoBLOB identity lives in stats.source
                )
            ann.stats = {
                "source": "CryoBLOB",
                "detection_mode": str(row.get("Detection Mode", "")),
                "detection_type": detection_type,
                "center_x_nm": float(row["Center X (nm)"]),
                "center_y_nm": float(row["Center Y (nm)"]),
                "radius_nm": float(row.get("Radius (nm)", 0.5 * float(row["Size (nm)"]))),
                "size_nm": float(row["Size (nm)"]),
                "raw_size_nm": float(row.get("Raw CryoBLOB Size (nm)", row["Size (nm)"])),
                "ridge_start_x_nm": row.get("Ridge Start X (nm)"),
                "ridge_start_y_nm": row.get("Ridge Start Y (nm)"),
                "ridge_end_x_nm": row.get("Ridge End X (nm)"),
                "ridge_end_y_nm": row.get("Ridge End Y (nm)"),
                "ridge_length_nm": row.get("Ridge Length (nm)"),
                "ridge_path_nm": row.get("Ridge Path (nm)"),
                "size_measurement": str(row.get("Size Measurement", "")),
                "pixel_size_source": str(row.get("Pixel Size Source", "")),
            }
            new_annotations.append(ann)
        return new_annotations

    def _annotate_current_image(self, df) -> int:
        img = self._context.current_image
        store = self._context.annotation_store
        if img is None or store is None or df is None or df.empty:
            print(f"[CryoBLOB] annotate skipped: img={img is not None} store={store is not None} "
                  f"df_rows={0 if df is None else len(df)}", flush=True)
            return 0

        current_path = Path(img.filepath).expanduser().resolve()
        # Primary match on resolved path; fall back to filename so a batch/CLU run
        # still overlays the open image even if paths differ (symlinks, /tmp, etc.).
        current_rows = [row for _, row in df.iterrows()
                        if self._row_path(row) == current_path]
        if not current_rows:
            current_rows = [row for _, row in df.iterrows()
                            if (self._row_path(row) or Path("")).name == current_path.name]
        print(f"[CryoBLOB] current image {current_path.name}: matched {len(current_rows)} "
              f"detection row(s) of {len(df)} total", flush=True)
        if not current_rows:
            return 0

        new_annotations = self._rows_to_annotations(current_rows, current_path)
        kept_annotations, _ = self._split_cryoblob_annotations(store)
        store.replace_all(kept_annotations + new_annotations)
        # Nudge the app to redraw/persist even if some render path is gated.
        try:
            self._context.annotations_changed.emit(store)
        except Exception:
            pass
        return len(new_annotations)

    def _write_all_sidecars(self, df) -> list[str]:
        """Persist CryoBLOB detections as .<stem>.acorn.json sidecars for EVERY
        processed image (not just the current one), so a batch/CLU run shows up on
        each image and is available for training export. Non-CryoBLOB annotations
        already in a sidecar are preserved. Returns the image paths written."""
        if df is None or df.empty:
            return []
        from dataclasses import asdict

        by_file: dict = {}
        for _, row in df.iterrows():
            p = self._row_path(row)
            if p is not None:
                by_file.setdefault(p, []).append(row)

        written: list[str] = []
        for img_path, rows in by_file.items():
            anns = self._rows_to_annotations(rows, img_path)
            if not anns:
                continue
            side = img_path.parent / f".{img_path.stem}.acorn.json"
            existing, px_nm = [], None
            if side.exists():
                try:
                    data = json.loads(side.read_text())
                    px_nm = data.get("pixel_size_nm")
                    existing = [d for d in data.get("annotations", [])
                                if (d.get("stats") or {}).get("source") != "CryoBLOB"]
                except Exception:
                    existing = []
            if px_nm is None:
                x = float(rows[0].get("Pixel Size X (nm/px)", 0.0) or 0.0)
                px_nm = x if x > 0 else (float(self._last_params.get("pixel_size_nm") or 0.0) or None)
            payload = {
                "version": 3,
                "annotations": existing + [asdict(a) for a in anns],
                "pixel_size_nm": px_nm,
                "exclude_zone": None,
                "crop_region": None,
            }
            try:
                side.write_text(json.dumps(payload))
                written.append(str(img_path))
            except OSError:
                pass
        return written

    def _clear_current_annotations(self) -> int:
        store = self._context.annotation_store
        if store is None:
            return 0
        kept_annotations, removed_annotations = self._split_cryoblob_annotations(store)
        if removed_annotations:
            store.replace_all(kept_annotations)
        return len(removed_annotations)

    @staticmethod
    def _split_cryoblob_annotations(store) -> tuple[list, list]:
        kept_annotations = []
        removed_annotations = []
        for ann in list(store):
            stats = getattr(ann, "stats", {}) or {}
            if stats.get("source") == "CryoBLOB" or (
                getattr(ann, "type", None) == "roi"
                and getattr(ann, "label", "") == "CryoBLOB"
            ):
                removed_annotations.append(ann)
            else:
                kept_annotations.append(ann)
        return kept_annotations, removed_annotations
