"""Plot panel — live matplotlib canvas with hover, click-to-navigate, stats tab."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QSpinBox, QTabWidget, QTextEdit, QToolButton,
    QVBoxLayout, QWidget,
)

from acorn_plotting.style import PALETTE
from acorn_plotting.figures import PLOT_TYPES, _XLABEL_MAP

# ---------------------------------------------------------------------------
# Palette presets
# ---------------------------------------------------------------------------

PALETTES: dict[str, list[str]] = {
    "ACORN":        ["#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"],
    "Colorblind":   ["#0072B2", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#CC79A7"],
    "TEM greens":   ["#1B7837", "#4DAC26", "#A6DBA0", "#008837", "#7FBF7B", "#D9F0D3"],
    "Warm":         ["#D73027", "#FC8D59", "#FEE090", "#E75480", "#FF6B35", "#FFB347"],
    "Cool":         ["#313695", "#4575B4", "#74ADD1", "#ABD9E9", "#7B2D8B", "#A6539C"],
    "Grayscale":    ["#222222", "#555555", "#888888", "#AAAAAA", "#CCCCCC", "#EEEEEE"],
}


class _SwatchButton(QToolButton):
    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self.color = color
        self.setFixedSize(18, 18)
        self.setStyleSheet(
            f"background:{color};border:1px solid #555;border-radius:2px;"
        )
        self.setToolTip(color)


class PlotPanel(QWidget):
    # Emitted when the user clicks a data point — (image_name, row_as_dict)
    navigate_requested = pyqtSignal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[dict] = []   # [{"label": str, "df": DataFrame, "color_offset": int}]
        self._fig          = None
        self._canvas_mpl   = None         # FigureCanvasQTAgg
        self._point_meta: list[tuple] = []  # [(x, y, row_dict), ...] for scatter hover/click
        self._bin_meta: list[tuple] = []    # [(x_left, x_right, [row_dicts]), ...] for histogram
        self._hover_annot  = None
        self._markers: list[tuple] = []
        self._add_mode     = False
        self._palette      = list(PALETTES["ACORN"])
        self._cid_click    = None
        self._cid_hover    = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # ── top toolbar ──────────────────────────────────────────────
        top = QHBoxLayout()

        top.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for t in PLOT_TYPES:
            self._type_combo.addItem(t.capitalize(), t)
        self._type_combo.setFixedWidth(110)
        self._type_combo.currentIndexChanged.connect(self._on_controls_changed)
        top.addWidget(self._type_combo)

        top.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._metric_combo.addItem(lbl, k)
        self._metric_combo.setFixedWidth(200)
        self._metric_combo.currentIndexChanged.connect(self._on_controls_changed)
        top.addWidget(self._metric_combo)

        self._y_label = QLabel("Y:")
        self._y_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._y_combo.addItem(lbl, k)
        self._y_combo.setCurrentIndex(4)
        self._y_combo.setFixedWidth(160)
        self._y_combo.currentIndexChanged.connect(self._on_controls_changed)
        top.addWidget(self._y_label)
        top.addWidget(self._y_combo)

        top.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 200)
        self._bins_spin.setValue(30)
        self._bins_spin.setFixedWidth(52)
        self._bins_spin.valueChanged.connect(self._on_controls_changed)
        top.addWidget(self._bins_spin)

        top.addStretch()

        self._add_btn = QPushButton("+ Marker")
        self._add_btn.setCheckable(True)
        self._add_btn.setFixedWidth(72)
        self._add_btn.setToolTip("Click plot to drop a reference line")
        self._add_btn.toggled.connect(self._on_add_mode_toggled)
        top.addWidget(self._add_btn)

        clr_btn = QPushButton("Clear")
        clr_btn.setFixedWidth(48)
        clr_btn.setToolTip("Clear reference markers")
        clr_btn.clicked.connect(self._clear_markers)
        top.addWidget(clr_btn)

        load_btn = QPushButton("+ Dataset")
        load_btn.setFixedWidth(72)
        load_btn.setToolTip("Load a second CSV and overlay it on the plot")
        load_btn.clicked.connect(self._load_extra_dataset)
        top.addWidget(load_btn)

        save_btn = QPushButton("Save PDF")
        save_btn.setFixedWidth(72)
        save_btn.clicked.connect(self._save_pdf)
        top.addWidget(save_btn)

        outer.addLayout(top)

        # ── palette row ──────────────────────────────────────────────
        pal_row = QHBoxLayout()
        pal_row.addWidget(QLabel("Palette:"))
        self._pal_combo = QComboBox()
        for name in PALETTES:
            self._pal_combo.addItem(name)
        self._pal_combo.setFixedWidth(110)
        self._pal_combo.currentTextChanged.connect(self._on_palette_changed)
        pal_row.addWidget(self._pal_combo)
        self._swatch_row = QHBoxLayout()
        self._swatch_row.setSpacing(2)
        self._rebuild_swatches()
        pal_row.addLayout(self._swatch_row)
        pal_row.addStretch()
        outer.addLayout(pal_row)

        # ── tab widget (Plot / Stats) ─────────────────────────────────
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        # Plot tab
        self._plot_tab = QWidget()
        plot_layout = QVBoxLayout(self._plot_tab)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self._canvas_placeholder = QLabel(
            "No plot yet.\nRun particle analysis, then ask CLU to plot.",
            alignment=Qt.AlignmentFlag.AlignCenter,
        )
        self._canvas_placeholder.setWordWrap(True)
        self._canvas_placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        plot_layout.addWidget(self._canvas_placeholder)
        self._plot_layout = plot_layout
        self._tabs.addTab(self._plot_tab, "Plot")

        # Stats tab
        self._stats_tab = QWidget()
        stats_layout = QVBoxLayout(self._stats_tab)
        stats_layout.setContentsMargins(4, 4, 4, 4)

        stats_ctrl = QHBoxLayout()
        stats_ctrl.addWidget(QLabel("Metric:"))
        self._stats_metric_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._stats_metric_combo.addItem(lbl, k)
        self._stats_metric_combo.setFixedWidth(200)
        stats_ctrl.addWidget(self._stats_metric_combo)
        run_stats_btn = QPushButton("Run Statistics")
        run_stats_btn.clicked.connect(self._run_stats)
        stats_ctrl.addWidget(run_stats_btn)
        stats_ctrl.addStretch()
        stats_layout.addLayout(stats_ctrl)

        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setFontFamily("monospace")
        self._stats_text.setPlaceholderText(
            "Click 'Run Statistics' or ask CLU to run stats on the current data."
        )
        stats_layout.addWidget(self._stats_text)
        self._tabs.addTab(self._stats_tab, "Stats")

        self._sync_y_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_figure(self, fig, df=None) -> None:
        if df is not None:
            if not self._datasets:
                self._datasets.append({"label": "data", "df": df, "color_offset": 0})
            else:
                self._datasets[0] = {"label": "data", "df": df, "color_offset": 0}
        self._install_figure(fig)

    def set_dataframe(self, df) -> None:
        if not self._datasets:
            self._datasets.append({"label": "data", "df": df, "color_offset": 0})
        else:
            self._datasets[0]["df"] = df
        if df is not None and not df.empty:
            self._redraw()

    def show_stats(self, text: str) -> None:
        """Display pre-formatted stats text and switch to Stats tab."""
        self._stats_text.setPlainText(text)
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Canvas installation
    # ------------------------------------------------------------------

    def _install_figure(self, fig) -> None:
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._disconnect_events()
        self._fig = fig
        axs = fig.get_axes()
        self._ax = axs[0] if axs else None

        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Replace whatever is currently in the plot tab layout
        while self._plot_layout.count():
            item = self._plot_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._plot_layout.addWidget(canvas)
        self._canvas_mpl = canvas

        self._setup_hover_annot()
        self._connect_events()
        self._redraw_markers()
        canvas.draw_idle()
        self._tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Hover & click
    # ------------------------------------------------------------------

    def _setup_hover_annot(self) -> None:
        if self._ax is None:
            return
        self._hover_annot = self._ax.annotate(
            "", xy=(0, 0), xytext=(12, 12),
            textcoords="offset points",
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFDE7", ec="#999", lw=0.8),
            arrowprops=dict(arrowstyle="->", color="#999", lw=0.8),
            fontsize=8, zorder=20,
        )
        self._hover_annot.set_visible(False)

    def _connect_events(self) -> None:
        if self._fig is None:
            return
        self._cid_click = self._fig.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )
        self._cid_hover = self._fig.canvas.mpl_connect(
            "motion_notify_event", self._on_canvas_hover
        )

    def _disconnect_events(self) -> None:
        if self._fig is None:
            return
        for cid in (self._cid_click, self._cid_hover):
            if cid is not None:
                try:
                    self._fig.canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._cid_click = self._cid_hover = None

    def _on_canvas_hover(self, event) -> None:
        if self._hover_annot is None or self._ax is None:
            return
        if event.inaxes != self._ax or event.xdata is None:
            self._hover_annot.set_visible(False)
            if self._canvas_mpl:
                self._canvas_mpl.draw_idle()
            return

        plot_type = self._type_combo.currentData()
        info = self._find_point_info(event.xdata, event.ydata, plot_type)
        if info is None:
            self._hover_annot.set_visible(False)
        else:
            text, xy = info
            self._hover_annot.set_text(text)
            self._hover_annot.xy = xy
            self._hover_annot.set_visible(True)
        if self._canvas_mpl:
            self._canvas_mpl.draw_idle()

    def _on_canvas_click(self, event) -> None:
        if event.inaxes != self._ax or event.xdata is None:
            return

        plot_type = self._type_combo.currentData()

        # Marker mode — drop reference line
        if self._add_mode:
            if plot_type not in ("scatter", "violin", "box"):
                self._markers.append((event.xdata, f"{event.xdata:.2f}"))
                self._redraw_markers()
                if self._canvas_mpl:
                    self._canvas_mpl.draw_idle()
            return

        # Navigation mode — find nearest point and emit signal
        row = self._find_nearest_row(event.xdata, event.ydata, plot_type)
        if row is not None:
            image_name = row.get("image", "")
            if image_name:
                self.navigate_requested.emit(image_name, row)

    def _find_point_info(self, xdata, ydata, plot_type):
        """Return (tooltip_text, (x, y)) for the nearest data point, or None."""
        if plot_type == "scatter" and self._point_meta:
            return self._nearest_scatter_info(xdata, ydata)
        if plot_type in ("histogram", "waterfall") and self._bin_meta:
            return self._bin_info(xdata)
        return None

    def _nearest_scatter_info(self, xdata, ydata):
        if not self._point_meta:
            return None
        metric_x = self._metric_combo.currentData() or "ecd_nm"
        metric_y = self._y_combo.currentData() or "aspect_ratio"
        xs = np.array([p[0] for p in self._point_meta])
        ys = np.array([p[1] for p in self._point_meta])
        # Normalise by axis range to compare distances fairly
        ax_xr = self._ax.get_xlim()
        ax_yr = self._ax.get_ylim()
        rx = ax_xr[1] - ax_xr[0] or 1
        ry = ax_yr[1] - ax_yr[0] or 1
        dists = ((xs - xdata) / rx) ** 2 + ((ys - ydata) / ry) ** 2
        idx = int(np.argmin(dists))
        if dists[idx] > 0.01:
            return None
        x, y, row = self._point_meta[idx]
        lines = [f"{row.get('image','?')}", f"{row.get('label','')}"]
        for k in (metric_x, metric_y, "ecd_nm", "feret_nm", "area_nm2"):
            if k in row and k not in (metric_x, metric_y):
                continue
            v = row.get(k)
            if v is not None and v != "":
                try:
                    lines.append(f"{k}: {float(v):.2f}")
                except (TypeError, ValueError):
                    pass
        return "\n".join(lines), (x, y)

    def _bin_info(self, xdata):
        for x_left, x_right, rows in self._bin_meta:
            if x_left <= xdata <= x_right and rows:
                r = rows[0]
                lines = [f"{len(rows)} particle(s) in bin", f"[{x_left:.1f} – {x_right:.1f}]"]
                if r.get("image"):
                    lines.append(f"First: {r['image']}")
                return "\n".join(lines), (0.5 * (x_left + x_right), 0)
        return None

    def _find_nearest_row(self, xdata, ydata, plot_type):
        if plot_type == "scatter" and self._point_meta:
            xs = np.array([p[0] for p in self._point_meta])
            ys = np.array([p[1] for p in self._point_meta])
            ax_xr = self._ax.get_xlim()
            ax_yr = self._ax.get_ylim()
            rx = ax_xr[1] - ax_xr[0] or 1
            ry = ax_yr[1] - ax_yr[0] or 1
            dists = ((xs - xdata) / rx) ** 2 + ((ys - ydata) / ry) ** 2
            idx = int(np.argmin(dists))
            if dists[idx] < 0.05:
                return self._point_meta[idx][2]
        elif plot_type in ("histogram", "waterfall"):
            for x_left, x_right, rows in self._bin_meta:
                if x_left <= xdata <= x_right and rows:
                    return rows[0]
        return None

    # ------------------------------------------------------------------
    # Redraw helpers
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        if not self._datasets or self._datasets[0]["df"] is None:
            return
        import matplotlib.pyplot as plt
        plt.close("all")

        plot_type = self._type_combo.currentData() or "histogram"
        metric    = self._metric_combo.currentData() or "ecd_nm"
        scatter_y = self._y_combo.currentData() or "aspect_ratio"
        n_bins    = self._bins_spin.value()

        # Merge all datasets with a source label for colouring
        import pandas as pd
        frames = []
        for ds in self._datasets:
            df = ds["df"]
            if df is None or df.empty:
                continue
            tmp = df.copy()
            if len(self._datasets) > 1:
                tmp["_source"] = ds["label"]
            frames.append(tmp)
        if not frames:
            return
        merged = pd.concat(frames, ignore_index=True)
        group_col = "_source" if len(self._datasets) > 1 else "label"

        from acorn_plotting.figures import build_figure
        fig = build_figure(
            df=merged, plot_type=plot_type, metric=metric,
            scatter_y=scatter_y, n_bins=n_bins,
            label_col=group_col, palette=self._palette,
        )

        # Cache point/bin metadata for interaction
        self._point_meta = []
        self._bin_meta   = []
        if plot_type == "scatter":
            for _, row in merged.iterrows():
                x = row.get(metric)
                y = row.get(scatter_y)
                if x is not None and y is not None:
                    try:
                        self._point_meta.append((float(x), float(y), dict(row)))
                    except (TypeError, ValueError):
                        pass
        elif plot_type in ("histogram", "waterfall"):
            vals = merged[metric].dropna().values
            if len(vals):
                bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
                for i in range(len(bins) - 1):
                    mask = (merged[metric] >= bins[i]) & (merged[metric] < bins[i + 1])
                    rows = [dict(r) for _, r in merged[mask].iterrows()]
                    self._bin_meta.append((float(bins[i]), float(bins[i + 1]), rows))

        self._install_figure(fig)

    def _redraw_markers(self) -> None:
        if self._ax is None or not self._markers:
            return
        plot_type = self._type_combo.currentData()
        if plot_type in ("scatter", "violin", "box"):
            return
        ylim = self._ax.get_ylim()
        for x, label in self._markers:
            self._ax.axvline(x, color="#E63946", lw=1.2, ls="--", alpha=0.8, zorder=10)
            self._ax.text(x, ylim[1] * 0.97, label, color="#E63946",
                          fontsize=7, ha="center", va="top", rotation=90, zorder=11)

    def _clear_markers(self) -> None:
        self._markers.clear()
        self._redraw()

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def _run_stats(self) -> None:
        if not self._datasets or self._datasets[0]["df"] is None:
            self._stats_text.setPlainText("No data loaded.")
            return
        import pandas as pd
        frames = [ds["df"] for ds in self._datasets if ds["df"] is not None and not ds["df"].empty]
        if not frames:
            return
        df = pd.concat(frames, ignore_index=True)
        metric = self._stats_metric_combo.currentData() or "ecd_nm"

        from acorn_plotting.stats import run_statistics, format_stats_report
        result = run_statistics(df, metric)
        self._stats_text.setPlainText(format_stats_report(result))
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _sync_y_visibility(self) -> None:
        scatter = self._type_combo.currentData() == "scatter"
        self._y_label.setVisible(scatter)
        self._y_combo.setVisible(scatter)
        self._bins_spin.setVisible(
            self._type_combo.currentData() in ("histogram", "waterfall")
        )

    def _on_controls_changed(self) -> None:
        self._sync_y_visibility()
        if self._datasets:
            self._redraw()

    def _on_palette_changed(self, name: str) -> None:
        self._palette = list(PALETTES.get(name, PALETTES["ACORN"]))
        self._rebuild_swatches()
        if self._datasets:
            self._redraw()

    def _pick_color(self, color: str) -> None:
        from PyQt6.QtWidgets import QColorDialog
        chosen = QColorDialog.getColor(QColor(color), self, "Pick colour")
        if chosen.isValid():
            try:
                idx = self._palette.index(color)
                self._palette[idx] = chosen.name()
            except ValueError:
                self._palette[0] = chosen.name()
            self._rebuild_swatches()
            if self._datasets:
                self._redraw()

    def _rebuild_swatches(self) -> None:
        while self._swatch_row.count():
            item = self._swatch_row.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for color in self._palette:
            sw = _SwatchButton(color)
            sw.clicked.connect(lambda _, c=color: self._pick_color(c))
            self._swatch_row.addWidget(sw)

    def _on_add_mode_toggled(self, checked: bool) -> None:
        self._add_mode = checked
        self._add_btn.setText("Marker ON" if checked else "+ Marker")

    def _load_extra_dataset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load dataset CSV", "", "CSV (*.csv);;All files (*)"
        )
        if not path:
            return
        try:
            import pandas as pd
            df = pd.read_csv(path)
            import os
            label = os.path.splitext(os.path.basename(path))[0]
            self._datasets.append({
                "label": label,
                "df": df,
                "color_offset": len(self._datasets),
            })
            self._redraw()
        except Exception as exc:
            self._stats_text.setPlainText(f"Error loading {path}:\n{exc}")
            self._tabs.setCurrentIndex(1)

    def _save_pdf(self) -> None:
        if self._fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", "", "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)"
        )
        if path:
            self._fig.savefig(path, bbox_inches="tight")
