"""AnalysisThread — background worker for surface area estimation."""
from __future__ import annotations
from typing import Optional
import numpy as np  # imported inside run() but add here for clarity
from PyQt6.QtCore import QThread, pyqtSignal

from acorn.analysis.surface_area import estimate_surface_area
from acorn.analysis.surface_area_stats import compare_groups, export_stats_report


class AnalysisThread(QThread):
    """
    Background thread that converts ROI polygon annotations to binary masks,
    runs per-particle surface area estimation, and saves publication figures.

    Input: list of dicts with keys  vertices, label, image_name.
    Emits: progress(int, str) and finished(object, object, str).
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object, object, str)  # (particles_df, stats_dict|None, out_dir_str)
    error    = pyqtSignal(str)

    def __init__(
        self,
        items: list[dict],
        pixel_size_nm: float,
        pixel_size_uncertainty_nm: float,
        output_dir: str,
        method: str = "auto",
        compound_mode: str = "separate",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._items          = items
        self._px_nm          = pixel_size_nm
        self._px_unc_nm      = pixel_size_uncertainty_nm
        self._output_dir     = output_dir
        self._method         = method
        self._compound_mode  = compound_mode

    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")

    def _run(self) -> None:
        import numpy as np
        import pandas as pd

        try:
            import cv2
        except ImportError as exc:
            self.error.emit(
                f"opencv-python is required for surface area analysis:\n  pip install opencv-python\n\n{exc}"
            )
            return

        from acorn.analysis.surface_area import estimate_surface_area
        from acorn.analysis.surface_area_stats import compare_groups, export_stats_report
        from acorn.core.dm4_loader import DM4Image

        self.progress.emit(5, "Preparing masks...")

        # ── compound mask grouping ─────────────────────────────────────────────
        if self._compound_mode != "separate":
            from collections import defaultdict
            groups: dict = defaultdict(list)
            for item in self._items:
                groups[(item["label"], item["image_name"])].append(item)

            merged_items: list[dict] = []
            for (lbl, img_name), group in groups.items():
                if len(group) == 1:
                    merged_items.append(group[0])
                    continue

                # Build all masks in a shared coordinate space
                all_pts = [np.array(it["vertices"], dtype=np.float32) for it in group]
                stacked = np.vstack(all_pts)
                gx_min, gy_min = stacked.min(axis=0)
                gx_max, gy_max = stacked.max(axis=0)
                gpad = 4
                gW = int(gx_max - gx_min) + 2 * gpad + 1
                gH = int(gy_max - gy_min) + 2 * gpad + 1
                offset = np.array([gx_min - gpad, gy_min - gpad])

                poly_masks = []
                for pts in all_pts:
                    m = np.zeros((gH, gW), dtype=np.uint8)
                    cv2.fillPoly(m, [(pts - offset).astype(np.int32)], 1)
                    poly_masks.append(m)

                areas = [int(m.sum()) for m in poly_masks]
                order = sorted(range(len(poly_masks)), key=lambda i: areas[i], reverse=True)

                if self._compound_mode == "union":
                    compound = poly_masks[0].astype(bool)
                    for m in poly_masks[1:]:
                        compound = compound | m.astype(bool)
                else:
                    # subtract_inner or auto
                    compound = poly_masks[order[0]].astype(bool)
                    for j in order[1:]:
                        inner = poly_masks[j].astype(bool)
                        contained = float(np.sum(inner & compound)) / max(float(np.sum(inner)), 1)
                        if self._compound_mode == "subtract_inner" or contained > 0.8:
                            compound = compound & ~inner
                        else:
                            compound = compound | inner

                merged_items.append({
                    "label":       lbl,
                    "image_name":  img_name,
                    "image_path":  group[0].get("image_path", ""),
                    "vertices":    group[order[0]]["vertices"],  # largest polygon vertices kept for reference
                    "_mask":       compound,
                    "_mask_offset": offset,
                })

            self._items = merged_items

        n = len(self._items)
        rows: list[dict] = []

        _cached_path: str = ""
        _cached_raw: Optional[np.ndarray] = None

        for i, item in enumerate(self._items):
            vertices  = item.get("vertices", [])
            label     = item.get("label", "")
            img_name  = item.get("image_name", "")
            img_path  = item.get("image_path", "")

            # Use pre-built compound mask if available, otherwise build from vertices
            if "_mask" in item:
                mask = item["_mask"]
                mask_offset = item.get("_mask_offset", np.zeros(2))
                x_min = float(mask_offset[0])
                y_min = float(mask_offset[1])
            else:
                if not vertices or len(vertices) < 3:
                    continue
                pts = np.array(vertices, dtype=np.float32)
                x_min, y_min = pts.min(axis=0)
                x_max, y_max = pts.max(axis=0)
                pad = 2
                w   = int(x_max - x_min) + 2 * pad + 1
                h   = int(y_max - y_min) + 2 * pad + 1
                shifted = pts - np.array([x_min - pad, y_min - pad])
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask, [shifted.astype(np.int32)], 1)
                mask = mask.astype(bool)

            # Load raw image crop for hollow detection via radial intensity profile
            raw_crop: Optional[np.ndarray] = None
            if img_path:
                if img_path != _cached_path:
                    try:
                        _cached_raw = DM4Image.from_file(img_path).raw
                        _cached_path = img_path
                    except Exception:
                        _cached_raw = None
                if _cached_raw is not None:
                    h_mask, w_mask = mask.shape[:2]
                    ry0 = max(0, int(y_min))
                    ry1 = min(_cached_raw.shape[0], ry0 + h_mask)
                    rx0 = max(0, int(x_min))
                    rx1 = min(_cached_raw.shape[1], rx0 + w_mask)
                    raw_crop = _cached_raw[ry0:ry1, rx0:rx1].astype(np.float32)

            pct = 5 + int(85 * i / max(n, 1))
            self.progress.emit(pct, f"Estimating SA {i + 1}/{n}  [{label}]...")

            result = estimate_surface_area(
                mask,
                pixel_size_nm=self._px_nm,
                pixel_size_uncertainty_nm=self._px_unc_nm,
                particle_id=i,
                method=self._method,
                raw_image=raw_crop,
            )
            row = {"label": label, "image": img_name}
            row.update(vars(result))
            row["particle_id"] = i
            rows.append(row)

        if not rows:
            self.error.emit(
                "No valid ROI annotations found for the selected labels.\n"
                "Make sure annotations are ROI polygons with a matching label."
            )
            return

        df = pd.DataFrame(rows)

        self.progress.emit(92, "Running statistical analysis...")

        stats_dict = None
        if df["label"].nunique() >= 2:
            try:
                stats_dict = compare_groups(df, group_col="label")
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("compare_groups failed: %s", exc)

        self.progress.emit(95, "Saving figures and tables...")

        out_dir = self._output_dir or ""
        if out_dir:
            from pathlib import Path as _Path
            try:
                export_stats_report(
                    df,
                    group_col="label",
                    output_dir=out_dir,
                    sample_name="acorn_sa",
                )
                # Also save the raw per-particle CSV
                df.to_csv(str(_Path(out_dir) / "per_particle.csv"), index=False)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("export_stats_report failed: %s", exc)

        self.progress.emit(100, "Done.")
        self.finished.emit(df, stats_dict, out_dir)
