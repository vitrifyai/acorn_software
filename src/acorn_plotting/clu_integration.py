"""CLU tool + Plot tab plugin for acorn_plotting."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

from PyQt6.QtWidgets import QWidget

from acorn.plugin_base import AcornPlugin
from acorn.export import ACORN_MEASUREMENTS_DIR, MEASUREMENTS_CSV

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext


# ---------------------------------------------------------------------------
# Tool schema (inlined so agent.py can import it directly)
# ---------------------------------------------------------------------------

PLOT_TOOL: dict = {
    "name": "plot_measurements",
    "description": (
        "Generate a publication-quality figure from particle measurement data. "
        "Shows in the floating Plot window and saves a file next to the images. "
        "Use after run_particle_analysis when the user asks to plot, visualise, "
        "chart, or export a figure of the data."
    ),
    "properties": {
        "plot_type": {
            "type": "string",
            "enum": ["scatter", "histogram", "box+jitter", "violin", "box", "waterfall"],
            "description": (
                "Chart type. scatter=x vs y (default), histogram=count histogram, "
                "box+jitter=box plot with data points + significance brackets, "
                "violin=violin, box=box-and-whisker, waterfall=ridge plot."
            ),
        },
        "metric": {
            "type": "string",
            "enum": ["ecd_nm", "feret_nm", "area_nm2", "perimeter_nm",
                     "circularity", "aspect_ratio", "bbox_w_nm", "bbox_h_nm"],
            "description": (
                "Primary metric / x-axis column. "
                "ecd_nm=diameter, feret_nm=Feret length, area_nm2=area."
            ),
        },
        "scatter_y": {
            "type": "string",
            "enum": ["ecd_nm", "feret_nm", "area_nm2", "perimeter_nm",
                     "circularity", "aspect_ratio", "bbox_w_nm", "bbox_h_nm"],
            "description": "Y-axis column when plot_type='scatter'.",
        },
        "n_bins": {
            "type": "integer",
            "description": "Number of bins for histogram/waterfall (5–200). Default 30.",
        },
        "title": {
            "type": "string",
            "description": "Optional figure title.",
        },
    },
    "required": [],
    "needs_confirm": False,
}


STATS_TOOL: dict = {
    "name": "run_statistics",
    "description": (
        "Run statistical analysis on particle measurement data and show results "
        "in the Stats tab of the Plot window. "
        "Automatically selects the right test: t-test / Mann-Whitney (2 groups), "
        "ANOVA / Kruskal-Wallis (3+ groups), with post-hoc comparisons if significant. "
        "Use when the user asks for statistics, p-values, significance, whether groups "
        "differ, or comparison of datasets."
    ),
    "properties": {
        "metric": {
            "type": "string",
            "enum": ["ecd_nm", "feret_nm", "area_nm2", "perimeter_nm",
                     "circularity", "aspect_ratio", "bbox_w_nm", "bbox_h_nm"],
            "description": "Metric to analyse. ecd_nm=diameter, feret_nm=Feret length.",
        },
    },
    "required": [],
    "needs_confirm": False,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_measurements_df(context: "AcornContext", analysis_plugin=None):
    """
    Return a pandas DataFrame with measurement results.

    Priority order:
    1. In-memory DataFrame from the Analysis plugin's particle_panel (_df attribute).
    2. acorn_measurements/measurements.csv next to the loaded image files.
    3. None if neither source is available.
    """
    # 1. In-memory DF from AnalysisPlugin._particle_panel._df
    if analysis_plugin is not None:
        pp = getattr(analysis_plugin, "_particle_panel", None)
        if pp is not None:
            df = getattr(pp, "_df", None)
            if df is not None and not df.empty:
                return df

    # 2. CSV fallback
    paths = context.image_paths
    if paths:
        csv_path = paths[0].parent / ACORN_MEASUREMENTS_DIR / MEASUREMENTS_CSV
        if csv_path.exists():
            try:
                import pandas as pd
                return pd.read_csv(csv_path)
            except Exception:
                pass

    return None


def _figure_output_path(context: "AcornContext", metric: str) -> Optional[Path]:
    """Return path for saving the output figure, next to the image files."""
    paths = context.image_paths
    if not paths:
        return None
    out_dir = paths[0].parent / ACORN_MEASUREMENTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"plot_{metric}.png"


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------

class PlottingPlugin(AcornPlugin):
    """Floating plot window — pops up when CLU generates a figure."""

    PLUGIN_ID  = "acorn_plotting"
    TAB_LABEL  = "Plot"   # kept so the base class is happy; create_panel returns None

    def __init__(self, context: "AcornContext") -> None:
        super().__init__(context)
        self._dock            = None   # QDockWidget, created lazily
        self._panel           = None   # PlotPanel inside the dock
        self._analysis_plugin = None
        context.action_requested.connect(self._on_action_requested)
        # Navigation from panel click is connected after dock is created

    def create_panel(self) -> Optional[QWidget]:
        """Return None — no permanent tab; we use a floating dock instead."""
        return None

    # ------------------------------------------------------------------
    # Lazy dock creation
    # ------------------------------------------------------------------

    def _ensure_dock(self) -> bool:
        """Create (once) and attach the floating QDockWidget to the main window."""
        if self._dock is not None:
            return True
        w = self._context._w()
        if w is None:
            return False

        from PyQt6.QtWidgets import QDockWidget
        from PyQt6.QtCore import Qt
        from acorn_plotting.panel import PlotPanel

        self._panel = PlotPanel()
        self._panel.navigate_requested.connect(self._on_navigate_requested)

        self._dock = QDockWidget("Plot", w)
        self._dock.setObjectName("acorn_plot_dock")
        self._dock.setWidget(self._panel)
        self._dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)

        # Start as a floating window, sized reasonably
        w.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._dock)
        self._dock.setFloating(True)

        # Position it offset from the main window so it doesn't fully overlap
        mw_geo = w.geometry()
        self._dock.resize(660, 520)
        self._dock.move(mw_geo.right() - 680, mw_geo.top() + 60)

        self._dock.hide()
        return True

    # ------------------------------------------------------------------
    # Helper: find AnalysisPlugin
    # ------------------------------------------------------------------

    def _resolve_analysis_plugin(self):
        if self._analysis_plugin is not None:
            return self._analysis_plugin
        w = self._context._w()
        if w is None:
            return None
        plugins = getattr(w, "_plugins", None) or []
        for p in plugins:
            if type(p).__name__ == "AnalysisPlugin":
                self._analysis_plugin = p
                return p
        return None

    # ------------------------------------------------------------------
    # CLU tool handler
    # ------------------------------------------------------------------

    def _on_navigate_requested(self, image_name: str, row: dict) -> None:
        """Navigate ACORN to the image/particle the user clicked in the plot."""
        paths = self._context.image_paths
        for i, p in enumerate(paths):
            if p.name == image_name:
                self._context.action_requested.emit("go_to_image", {"index": i + 1})
                self._context.set_status(
                    f"Plot → {image_name}  |  "
                    + "  ".join(f"{k}={v:.2f}" for k, v in row.items()
                                if k not in ("image", "label", "type", "calibrated")
                                and isinstance(v, (int, float)) and v == v)
                )
                return

    def _on_action_requested(self, action: str, params: dict) -> None:
        if action == "run_statistics":
            self._handle_run_statistics(params)
        elif action == "plot_measurements":
            self._handle_plot_measurements(params)

    def _handle_run_statistics(self, params: dict) -> None:
        metric          = params.get("metric", "ecd_nm")
        analysis_plugin = self._resolve_analysis_plugin()
        df = _get_measurements_df(self._context, analysis_plugin)
        if df is None or df.empty:
            self._context.set_status("Stats: no measurements — run particle analysis first.")
            return
        from acorn_plotting.stats import run_statistics, format_stats_report
        result = run_statistics(df, metric)
        report = format_stats_report(result)
        if self._ensure_dock() and self._panel is not None:
            self._panel.show_stats(report)
            self._dock.show()
            self._dock.raise_()
            self._dock.activateWindow()
        self._context.set_status(
            f"Stats: {result.get('comparison', {}).get('test', 'done')} — "
            f"{result.get('comparison', {}).get('significance', '')}"
        )

    def _handle_plot_measurements(self, params: dict) -> None:
        # Always open the dock first so the user sees it regardless of data state
        if not self._ensure_dock() or self._panel is None:
            return
        self._dock.show()
        self._dock.raise_()
        self._dock.activateWindow()

        plot_type       = params.get("plot_type", "scatter")
        metric          = params.get("metric", "ecd_nm")
        scatter_y       = params.get("scatter_y", "aspect_ratio")
        n_bins          = int(params.get("n_bins", 30))
        analysis_plugin = self._resolve_analysis_plugin()

        # If particle analysis is still running, retry briefly
        df = None
        for _attempt in range(5):
            df = _get_measurements_df(self._context, analysis_plugin)
            if df is not None and not df.empty:
                break
            import time
            time.sleep(0.5)

        if df is None or df.empty:
            self._context.set_status(
                "Plot window open — no measurements yet. Run particle analysis first, then ask to plot again."
            )
            return

        from acorn_plotting.figures import build_figure_new
        out_path = _figure_output_path(self._context, f"{plot_type}_{metric}")
        fig = build_figure_new(
            df=df, plot_type=plot_type, metric=metric,
            scatter_y=scatter_y, n_bins=n_bins,
            output_path=out_path,
        )
        self._panel.show_figure(fig, df=df)
        saved_msg = f"  Saved → {out_path}" if out_path else ""
        self._context.set_status(f"Plot: {plot_type} of {metric} ready.{saved_msg}")
