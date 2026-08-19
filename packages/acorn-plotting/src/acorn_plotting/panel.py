"""Plot panel — persistent matplotlib canvas, full controls, hover/click navigation."""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QSizePolicy, QSpinBox, QTabWidget,
    QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from acorn_plotting.style import PALETTE
from acorn_plotting.figures import PLOT_TYPES, _XLABEL_MAP

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
        self.setFixedSize(16, 16)
        self.setStyleSheet(
            f"background:{color};border:1px solid #555;border-radius:2px;"
        )
        self.setToolTip(color)


class PlotPanel(QWidget):
    navigate_requested = pyqtSignal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._datasets: list[dict] = []
        self._mpl_fig    = None   # single persistent Figure
        self._canvas_mpl = None   # single persistent FigureCanvasQTAgg
        self._ax         = None   # primary axes reference (updated each redraw)
        self._point_meta: list[tuple] = []
        self._bin_meta:   list[tuple] = []
        self._hover_annot = None
        self._markers:    list[tuple] = []
        self._add_mode    = False
        self._palette     = list(PALETTES["ACORN"])
        self._cid_click   = None
        self._cid_hover   = None
        self._suppress_redraw = False   # block redraws during bulk init

        self._build_ui()
        self._init_canvas()

    # ------------------------------------------------------------------
    # Canvas initialisation (once)
    # ------------------------------------------------------------------

    def _init_canvas(self) -> None:
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        self._mpl_fig = plt.figure(figsize=(6, 4.5))
        self._canvas_mpl = FigureCanvasQTAgg(self._mpl_fig)
        self._canvas_mpl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Add placeholder text on empty figure
        self._mpl_fig.text(0.5, 0.5,
                           "No plot yet.\nRun particle analysis, then ask CLU to plot.",
                           ha="center", va="center", fontsize=10, color="#888888",
                           transform=self._mpl_fig.transFigure)

        self._plot_layout.addWidget(self._canvas_mpl)
        self._connect_events()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(3)

        # ── Row 1: type / primary metric / Y-axis / bins ─────────────
        row1 = QHBoxLayout()
        row1.setSpacing(4)

        row1.addWidget(QLabel("Type:"))
        self._type_combo = QComboBox()
        for t in PLOT_TYPES:
            label = "Box + points" if t == "box+jitter" else t.capitalize()
            self._type_combo.addItem(label, t)
        self._type_combo.setFixedWidth(115)
        self._type_combo.currentIndexChanged.connect(self._on_controls_changed)
        row1.addWidget(self._type_combo)

        row1.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._metric_combo.addItem(lbl, k)
        self._metric_combo.setFixedWidth(205)
        self._metric_combo.currentIndexChanged.connect(self._on_controls_changed)
        row1.addWidget(self._metric_combo)

        self._y_label = QLabel("Y:")
        self._y_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._y_combo.addItem(lbl, k)
        self._y_combo.setCurrentIndex(4)   # circularity
        self._y_combo.setFixedWidth(155)
        self._y_combo.currentIndexChanged.connect(self._on_controls_changed)
        row1.addWidget(self._y_label)
        row1.addWidget(self._y_combo)

        row1.addWidget(QLabel("Bins:"))
        self._bins_spin = QSpinBox()
        self._bins_spin.setRange(5, 200)
        self._bins_spin.setValue(30)
        self._bins_spin.setFixedWidth(50)
        self._bins_spin.valueChanged.connect(self._on_controls_changed)
        row1.addWidget(self._bins_spin)

        self._sig_chk = QCheckBox("Sig.")
        self._sig_chk.setChecked(True)
        self._sig_chk.setToolTip("Show significance brackets on box+jitter plots")
        self._sig_chk.stateChanged.connect(self._on_controls_changed)
        row1.addWidget(self._sig_chk)

        row1.addStretch()
        outer.addLayout(row1)

        # ── Row 2: axis labels / log / palette ───────────────────────
        row2 = QHBoxLayout()
        row2.setSpacing(4)

        row2.addWidget(QLabel("X label:"))
        self._xlabel_edit = QLineEdit()
        self._xlabel_edit.setPlaceholderText("auto")
        self._xlabel_edit.setFixedWidth(140)
        self._xlabel_edit.editingFinished.connect(self._on_controls_changed)
        row2.addWidget(self._xlabel_edit)

        row2.addWidget(QLabel("Y label:"))
        self._ylabel_edit = QLineEdit()
        self._ylabel_edit.setPlaceholderText("auto")
        self._ylabel_edit.setFixedWidth(140)
        self._ylabel_edit.editingFinished.connect(self._on_controls_changed)
        row2.addWidget(self._ylabel_edit)

        self._logx_chk = QCheckBox("Log X")
        self._logx_chk.stateChanged.connect(self._on_controls_changed)
        row2.addWidget(self._logx_chk)

        self._logy_chk = QCheckBox("Log Y")
        self._logy_chk.stateChanged.connect(self._on_controls_changed)
        row2.addWidget(self._logy_chk)

        row2.addWidget(QLabel("Palette:"))
        self._pal_combo = QComboBox()
        for name in PALETTES:
            self._pal_combo.addItem(name)
        self._pal_combo.setFixedWidth(105)
        self._pal_combo.currentTextChanged.connect(self._on_palette_changed)
        row2.addWidget(self._pal_combo)

        self._swatch_row = QHBoxLayout()
        self._swatch_row.setSpacing(2)
        self._rebuild_swatches()
        row2.addLayout(self._swatch_row)

        row2.addStretch()
        outer.addLayout(row2)

        # ── Row 3: action buttons ─────────────────────────────────────
        row3 = QHBoxLayout()
        row3.setSpacing(4)

        self._add_btn = QPushButton("+ Marker")
        self._add_btn.setCheckable(True)
        self._add_btn.setToolTip("Click plot to drop a reference line")
        self._add_btn.toggled.connect(self._on_add_mode_toggled)
        row3.addWidget(self._add_btn)

        clr_btn = QPushButton("Clear markers")
        clr_btn.clicked.connect(self._clear_markers)
        row3.addWidget(clr_btn)

        load_btn = QPushButton("+ Dataset")
        load_btn.setToolTip("Load a second CSV and overlay it")
        load_btn.clicked.connect(self._load_extra_dataset)
        row3.addWidget(load_btn)

        save_btn = QPushButton("Save PDF")
        save_btn.clicked.connect(self._save_pdf)
        row3.addWidget(save_btn)

        row3.addStretch()
        outer.addLayout(row3)

        # ── Tab widget: Plot / Stats ──────────────────────────────────
        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, 1)

        # Plot tab
        plot_tab = QWidget()
        self._plot_layout = QVBoxLayout(plot_tab)
        self._plot_layout.setContentsMargins(0, 0, 0, 0)
        self._tabs.addTab(plot_tab, "Plot")

        # Stats tab
        stats_tab = QWidget()
        sl = QVBoxLayout(stats_tab)
        sl.setContentsMargins(4, 4, 4, 4)
        sc = QHBoxLayout()
        sc.addWidget(QLabel("Metric:"))
        self._stats_metric_combo = QComboBox()
        for k, lbl in _XLABEL_MAP.items():
            self._stats_metric_combo.addItem(lbl, k)
        self._stats_metric_combo.setFixedWidth(200)
        sc.addWidget(self._stats_metric_combo)
        run_btn = QPushButton("Run Statistics")
        run_btn.clicked.connect(self._run_stats)
        sc.addWidget(run_btn)
        sc.addStretch()
        sl.addLayout(sc)
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setFontFamily("monospace")
        self._stats_text.setPlaceholderText(
            "Click 'Run Statistics' or ask CLU to analyse the data.")
        sl.addWidget(self._stats_text)
        self._tabs.addTab(stats_tab, "Stats")

        self._sync_visibility()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_figure(self, fig_ignored, df=None) -> None:
        """Called by CLU. Store df and redraw using panel controls."""
        if df is not None:
            if not self._datasets:
                self._datasets.append({"label": "data", "df": df, "color_offset": 0})
            else:
                self._datasets[0] = {"label": "data", "df": df, "color_offset": 0}
        self._redraw()

    def set_dataframe(self, df) -> None:
        if not self._datasets:
            self._datasets.append({"label": "data", "df": df, "color_offset": 0})
        else:
            self._datasets[0]["df"] = df
        self._redraw()

    def show_stats(self, text: str) -> None:
        self._stats_text.setPlainText(text)
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Redraw — clears figure and redraws in-place
    # ------------------------------------------------------------------

    def _redraw(self) -> None:
        if self._suppress_redraw or not self._datasets:
            return
        import pandas as pd
        frames = [ds["df"] for ds in self._datasets
                  if ds.get("df") is not None and not ds["df"].empty]
        if not frames:
            return

        merged = pd.concat(frames, ignore_index=True)
        group_col = "_source" if len(self._datasets) > 1 else "label"
        if len(self._datasets) > 1:
            parts = []
            for ds in self._datasets:
                if ds.get("df") is None or ds["df"].empty:
                    continue
                tmp = ds["df"].copy()
                tmp["_source"] = ds["label"]
                parts.append(tmp)
            merged = pd.concat(parts, ignore_index=True)

        plot_type = self._type_combo.currentData() or "scatter"
        metric    = self._metric_combo.currentData() or "ecd_nm"
        scatter_y = self._y_combo.currentData() or "circularity"
        n_bins    = self._bins_spin.value()
        log_x     = self._logx_chk.isChecked()
        log_y     = self._logy_chk.isChecked()
        show_sig  = self._sig_chk.isChecked()
        xlabel    = self._xlabel_edit.text().strip() or None
        ylabel    = self._ylabel_edit.text().strip() or None

        # Cache interaction metadata before redraw
        self._point_meta = []
        self._bin_meta   = []
        if plot_type == "scatter" and metric in merged.columns and scatter_y in merged.columns:
            for _, row in merged.iterrows():
                x, y = row.get(metric), row.get(scatter_y)
                if x is not None and y is not None:
                    try:
                        self._point_meta.append((float(x), float(y), dict(row)))
                    except (TypeError, ValueError):
                        pass
        elif plot_type in ("histogram", "waterfall") and metric in merged.columns:
            vals = merged[metric].dropna().values
            if len(vals):
                bins = np.linspace(vals.min(), vals.max(), n_bins + 1)
                for i in range(len(bins) - 1):
                    mask = (merged[metric] >= bins[i]) & (merged[metric] < bins[i + 1])
                    rows = [dict(r) for _, r in merged[mask].iterrows()]
                    self._bin_meta.append((float(bins[i]), float(bins[i + 1]), rows))

        from acorn_plotting.figures import build_figure
        build_figure(
            df=merged, fig=self._mpl_fig,
            plot_type=plot_type, metric=metric,
            scatter_y=scatter_y, n_bins=n_bins,
            label_col=group_col, palette=self._palette,
            log_x=log_x, log_y=log_y,
            xlabel=xlabel, ylabel=ylabel,
            show_sig=show_sig,
        )

        # Update primary axes reference
        axs = self._mpl_fig.get_axes()
        self._ax = axs[0] if axs else None
        self._setup_hover_annot()
        self._redraw_markers()
        self._canvas_mpl.draw_idle()
        self._tabs.setCurrentIndex(0)

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def _connect_events(self) -> None:
        if self._mpl_fig is None:
            return
        self._cid_click = self._mpl_fig.canvas.mpl_connect(
            "button_press_event", self._on_canvas_click)
        self._cid_hover = self._mpl_fig.canvas.mpl_connect(
            "motion_notify_event", self._on_canvas_hover)

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

    def _on_canvas_hover(self, event) -> None:
        if self._hover_annot is None or self._ax is None:
            return
        if event.inaxes != self._ax or event.xdata is None:
            self._hover_annot.set_visible(False)
            self._canvas_mpl.draw_idle()
            return
        plot_type = self._type_combo.currentData()
        info = self._find_point_info(event.xdata, event.ydata, plot_type)
        if info:
            text, xy = info
            self._hover_annot.set_text(text)
            self._hover_annot.xy = xy
            self._hover_annot.set_visible(True)
        else:
            self._hover_annot.set_visible(False)
        self._canvas_mpl.draw_idle()

    def _on_canvas_click(self, event) -> None:
        if self._ax is None or event.inaxes != self._ax or event.xdata is None:
            return
        plot_type = self._type_combo.currentData()
        if self._add_mode:
            if plot_type not in ("scatter", "violin", "box", "box+jitter"):
                self._markers.append((event.xdata, f"{event.xdata:.2f}"))
                self._redraw_markers()
                self._canvas_mpl.draw_idle()
            return
        row = self._find_nearest_row(event.xdata, event.ydata, plot_type)
        if row:
            image_name = row.get("image", "")
            if image_name:
                self.navigate_requested.emit(image_name, row)

    def _find_point_info(self, xdata, ydata, plot_type):
        if plot_type == "scatter" and self._point_meta:
            return self._nearest_scatter_info(xdata, ydata)
        if plot_type in ("histogram", "waterfall") and self._bin_meta:
            return self._bin_info(xdata)
        return None

    def _nearest_scatter_info(self, xdata, ydata):
        if not self._point_meta:
            return None
        metric_x = self._metric_combo.currentData() or "ecd_nm"
        metric_y = self._y_combo.currentData() or "circularity"
        xs = np.array([p[0] for p in self._point_meta])
        ys = np.array([p[1] for p in self._point_meta])
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()
        rx = xlim[1] - xlim[0] or 1
        ry = ylim[1] - ylim[0] or 1
        dists = ((xs - xdata) / rx) ** 2 + ((ys - ydata) / ry) ** 2
        idx = int(np.argmin(dists))
        if dists[idx] > 0.01:
            return None
        x, y, row = self._point_meta[idx]
        lines = [row.get("image", "?"), row.get("label", "")]
        for k in (metric_x, metric_y):
            v = row.get(k)
            if v is not None and v != "":
                try:
                    lines.append(f"{k}: {float(v):.2f}")
                except (TypeError, ValueError):
                    pass
        return "\n".join(l for l in lines if l), (x, y)

    def _bin_info(self, xdata):
        for x_left, x_right, rows in self._bin_meta:
            if x_left <= xdata <= x_right and rows:
                r = rows[0]
                return (f"{len(rows)} particle(s)\n[{x_left:.1f}–{x_right:.1f}]\n"
                        + (r.get("image", "") or "")), \
                       (0.5 * (x_left + x_right), 0)
        return None

    def _find_nearest_row(self, xdata, ydata, plot_type):
        if plot_type == "scatter" and self._point_meta:
            xs = np.array([p[0] for p in self._point_meta])
            ys = np.array([p[1] for p in self._point_meta])
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()
            rx = xlim[1] - xlim[0] or 1
            ry = ylim[1] - ylim[0] or 1
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
    # Markers
    # ------------------------------------------------------------------

    def _redraw_markers(self) -> None:
        if self._ax is None or not self._markers:
            return
        plot_type = self._type_combo.currentData()
        if plot_type in ("scatter", "violin", "box", "box+jitter"):
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
        if not self._datasets:
            self._stats_text.setPlainText("No data loaded.")
            return
        import pandas as pd
        frames = [ds["df"] for ds in self._datasets
                  if ds.get("df") is not None and not ds["df"].empty]
        if not frames:
            return
        df     = pd.concat(frames, ignore_index=True)
        metric = self._stats_metric_combo.currentData() or "ecd_nm"
        from acorn_plotting.stats import run_statistics, format_stats_report
        result = run_statistics(df, metric)
        self._stats_text.setPlainText(format_stats_report(result))
        self._tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _sync_visibility(self) -> None:
        pt = self._type_combo.currentData() if self._type_combo.count() else "scatter"
        self._y_label.setVisible(pt == "scatter")
        self._y_combo.setVisible(pt == "scatter")
        self._bins_spin.setVisible(pt in ("histogram", "waterfall"))
        self._sig_chk.setVisible(pt == "box+jitter")
        self._logx_chk.setVisible(pt in ("scatter", "histogram", "waterfall"))

    def _on_controls_changed(self) -> None:
        self._sync_visibility()
        self._redraw()

    def _on_palette_changed(self, name: str) -> None:
        self._palette = list(PALETTES.get(name, PALETTES["ACORN"]))
        self._rebuild_swatches()
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
            self, "Load dataset CSV", "", "CSV (*.csv);;All files (*)")
        if not path:
            return
        try:
            import pandas as pd, os
            df    = pd.read_csv(path)
            label = os.path.splitext(os.path.basename(path))[0]
            self._datasets.append({"label": label, "df": df,
                                    "color_offset": len(self._datasets)})
            self._redraw()
        except Exception as exc:
            self._stats_text.setPlainText(f"Error: {exc}")
            self._tabs.setCurrentIndex(1)

    def _save_pdf(self) -> None:
        if self._mpl_fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save figure", "",
            "PDF (*.pdf);;SVG (*.svg);;PNG (*.png)")
        if path:
            self._mpl_fig.savefig(path, bbox_inches="tight")
