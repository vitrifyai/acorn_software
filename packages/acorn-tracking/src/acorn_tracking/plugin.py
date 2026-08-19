"""Tracking plugin — particle/cell tracking across image sequences."""
from __future__ import annotations
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QMessageBox, QWidget

from acorn.plugin_base import AcornPlugin

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


class TrackingPlugin(AcornPlugin):
    TAB_LABEL         = "Track"
    PLUGIN_ID         = "acorn_tracking"
    sort_order        = 20
    FLOATING          = True
    FLOATING_TITLE    = "Particle Tracking"
    FLOATING_SHORTCUT = "Ctrl+Shift+T"

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._panel = None
        self._tracking_df = None
        context.image_loaded.connect(self._refresh_status)
        context.annotations_changed.connect(self._refresh_status)
        context.action_requested.connect(self._on_action_requested)

    def _on_action_requested(self, action: str, params: dict) -> None:
        if action != "track_particles":
            return
        track_params = {
            "max_displacement_nm": float(params.get("max_displacement_nm", 500.0)),
            "min_frames":          int(params.get("min_frames", 2)),
            "max_gap":             int(params.get("max_gap", 1)),
        }
        if self._panel is not None:
            self._panel._max_disp_spin.setValue(track_params["max_displacement_nm"])
            self._panel._min_frames_spin.setValue(track_params["min_frames"])
            self._panel._max_gap_spin.setValue(track_params["max_gap"])
        self._on_track_requested(track_params)

    def _refresh_status(self, *_) -> None:
        if self._panel is None:
            return
        n = len(self._context.image_paths)
        states = self._context.all_annotation_states
        annotated = sum(1 for s in states.values() if s)
        self._panel.update_status(n, annotated)

    def create_panel(self) -> QWidget:
        from acorn_tracking.panel import TrackingPanel
        self._panel = TrackingPanel()
        self._panel.track_requested.connect(self._on_track_requested)
        self._panel.export_requested.connect(self._on_export)
        return self._panel

    def _on_track_requested(self, params: dict) -> None:
        from acorn.analysis.tracking import track_annotations, track_statistics
        from acorn.core.annotations import AnnotationStore

        paths = self._context.image_paths
        if len(paths) < 2:
            QMessageBox.information(None, "Tracking", "Load at least two images before tracking.")
            return

        px = self._context.current_pixel_size_nm

        states = self._context.all_annotation_states
        stores = []
        for i in range(len(paths)):
            state = states.get(i)
            if state:
                s = AnnotationStore()
                s.replace_all(state)
                stores.append(s)
            else:
                stores.append(AnnotationStore())

        try:
            df = track_annotations(
                stores,
                pixel_size_nm=px,
                max_displacement_nm=params["max_displacement_nm"],
                min_frames=params["min_frames"],
                max_gap=params["max_gap"],
            )
            stats = track_statistics(df)
        except Exception as exc:
            QMessageBox.critical(None, "Tracking error", str(exc))
            return

        self._tracking_df = df
        self._panel.set_tracks(df, stats)
        n = 0 if df.empty else df["track_id"].nunique()
        self._context.set_status(f"Tracking complete — {n} track(s) found.")

    def _on_export(self, path: str) -> None:
        if self._tracking_df is None or self._tracking_df.empty:
            return
        try:
            self._tracking_df.to_csv(path, index=False)
            self._context.set_status(f"Tracks exported to {path}")
        except Exception as exc:
            QMessageBox.critical(None, "Export error", str(exc))
