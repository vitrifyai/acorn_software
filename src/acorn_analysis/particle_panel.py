"""Particle Measurements panel -- 2D shape metrics for TEM/STEM nanoparticles."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

# Shape metric functions live in acorn.core.measurements — import here for
# backward compatibility with any code that imports them from particle_panel.
from acorn.core.measurements import (
    polygon_metrics  as _polygon_metrics,
    circle_metrics   as _circle_metrics,
    rect_metrics     as _rect_metrics,
)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class ParticleThread(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)   # pandas DataFrame
    error    = pyqtSignal(str)

    def __init__(self, items: list[dict], selected_labels: set[str], parent=None):
        super().__init__(parent)
        self._items = items
        self._selected = selected_labels

    def run(self) -> None:
        try:
            import pandas as pd
        except ImportError:
            self.error.emit("pandas is required for particle measurements.")
            return

        rows = []
        total = len(self._items)
        for i, item in enumerate(self._items):
            self.progress.emit(int(100 * i / max(total, 1)), item.get("image", ""))
            store   = item["store"]
            px_nm   = item["px_nm"]
            img_name = item["image"]

            for ann in store:
                lbl = getattr(ann, "label", "") or ""
                if self._selected and lbl not in self._selected:
                    continue
                atype = getattr(ann, "type", "")
                metrics = {}
                if atype == "roi":
                    verts = getattr(ann, "vertices", [])
                    if len(verts) >= 3:
                        metrics = _polygon_metrics(verts, px_nm)
                elif atype == "circle":
                    metrics = _circle_metrics(getattr(ann, "r", 0.0), px_nm)
                elif atype == "rectangle":
                    metrics = _rect_metrics(
                        getattr(ann, "x0", 0), getattr(ann, "y0", 0),
                        getattr(ann, "x1", 0), getattr(ann, "y1", 0),
                        px_nm,
                    )
                if not metrics:
                    continue
                rows.append({"image": img_name, "label": lbl, **metrics})

        self.progress.emit(100, "done")
        self.finished.emit(pd.DataFrame(rows) if rows else pd.DataFrame())


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------

_METRICS = [
    ("ecd_nm",        "ECD (nm)",             "Equivalent Circular Diameter"),
    ("feret_nm",      "Feret diam. (nm)",      "Max caliper diameter"),
    ("area_nm2",      "Area (nm^2)",           "2D projected area"),
    ("perimeter_nm",  "Perimeter (nm)",        "Polygon perimeter"),
    ("circularity",   "Circularity",           "4*pi*A/P^2 -- 1.0 = perfect circle"),
    ("aspect_ratio",  "Aspect ratio",          "Long / short axis (bounding box)"),
    ("bbox_w_nm",     "BBox width (nm)",       "Bounding box width"),
    ("bbox_h_nm",     "BBox height (nm)",      "Bounding box height"),
]
_METRIC_KEYS  = [k for k, _, _ in _METRICS]
_METRIC_LABEL = {k: lbl for k, lbl, _ in _METRICS}


class ParticlePanel(QWidget):
    """2D shape measurement panel for TEM/STEM nanoparticles."""

    analysis_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label_checks: dict[str, QCheckBox] = {}
        self._df = None
        self._fig = None
        self._fig_canvas_widget = None
        self._figures_layout = None
        self._thread: Optional[ParticleThread] = None
        self._build_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_label_group())
        outer.addWidget(self._build_param_group())

        self._run_btn = QPushButton("Run Measurements")
        self._run_btn.clicked.connect(self._on_run)
        outer.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setStyleSheet("font-size:11px;color:palette(mid);")
        self._status.setVisible(False)
        outer.addWidget(self._status)

        self._results_tabs = self._build_results_tabs()
        self._results_tabs.setVisible(False)
        outer.addWidget(self._results_tabs, 1)

    def _build_label_group(self) -> QGroupBox:
        box = QGroupBox("Labels / Annotation Types")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        for txt, slot in (("All", self._select_all), ("None", self._select_none)):
            b = QPushButton(txt)
            b.setFixedWidth(56 if txt == "All" else 64)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._label_container = QWidget()
        self._label_layout = QVBoxLayout(self._label_container)
        self._label_layout.setSpacing(2)
        self._label_layout.setContentsMargins(2, 2, 2, 2)
        self._label_layout.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(110)
        scroll.setWidget(self._label_container)
        layout.addWidget(scroll)

        self._no_labels = QLabel("No ROI, circle, or rectangle annotations found.")
        self._no_labels.setStyleSheet("font-size:11px;color:palette(mid);")
        self._no_labels.setWordWrap(True)
        layout.addWidget(self._no_labels)

        scroll.setVisible(False)
        self._label_scroll = scroll
        return box

    def _build_param_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        form.setSpacing(6)

        mode_w = QWidget()
        row = QHBoxLayout(mode_w)
        row.setContentsMargins(0, 0, 0, 0)
        self._mode_single = QRadioButton("Current image")
        self._mode_batch  = QRadioButton("All images")
        self._mode_single.setChecked(True)
        row.addWidget(self._mode_single)
        row.addWidget(self._mode_batch)
        row.addStretch()
        form.addRow("Mode:", mode_w)

        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.0, 10000.0)
        self._px_spin.setDecimals(4)
        self._px_spin.setSuffix(" nm/px")
        self._px_spin.setToolTip(
            "Fallback pixel size. In session mode each image uses its own calibrated value."
        )
        form.addRow("Default pixel size:", self._px_spin)

        note = QLabel("Accepts ROI polygons, circles, and rectangles.")
        note.setStyleSheet("font-size:10px;color:palette(mid);")
        form.addRow("", note)
        return box

    def _build_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # Table tab
        tbl_widget = QWidget()
        tbl_layout = QVBoxLayout(tbl_widget)
        tbl_layout.setContentsMargins(4, 4, 4, 4)

        export_row = QHBoxLayout()
        export_row.addStretch()
        exp_btn = QPushButton("Export CSV")
        exp_btn.setFixedWidth(90)
        exp_btn.clicked.connect(self._export_csv)
        export_row.addWidget(exp_btn)
        tbl_layout.addLayout(export_row)

        self._table = QTableWidget()
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.setSortingEnabled(True)
        tbl_layout.addWidget(self._table)
        tabs.addTab(tbl_widget, "Table")

        # Figures tab
        fig_widget = QWidget()
        fig_layout = QVBoxLayout(fig_widget)
        fig_layout.setContentsMargins(4, 4, 4, 4)
        fig_layout.setSpacing(4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Metric:"))
        self._metric_combo = QComboBox()
        for k, lbl, tip in _METRICS:
            self._metric_combo.addItem(lbl, userData=k)
            self._metric_combo.setItemData(
                self._metric_combo.count() - 1, tip, Qt.ItemDataRole.ToolTipRole
            )
        self._metric_combo.currentIndexChanged.connect(self._refresh_figure)
        ctrl.addWidget(self._metric_combo)

        ctrl.addWidget(QLabel("Y-axis:"))
        self._plot_type_combo = QComboBox()
        self._plot_type_combo.addItem("Count",   "count")
        self._plot_type_combo.addItem("Density", "density")
        self._plot_type_combo.setFixedWidth(80)
        self._plot_type_combo.currentIndexChanged.connect(self._refresh_figure)
        ctrl.addWidget(self._plot_type_combo)

        ctrl.addWidget(QLabel("Bins:"))
        self._bins_spin = QDoubleSpinBox()
        self._bins_spin.setDecimals(0)
        self._bins_spin.setRange(5, 200)
        self._bins_spin.setValue(30)
        self._bins_spin.setFixedWidth(56)
        self._bins_spin.valueChanged.connect(self._refresh_figure)
        ctrl.addWidget(self._bins_spin)

        ctrl.addStretch()
        for fmt in ("PNG", "SVG", "PDF"):
            b = QPushButton(fmt)
            b.setFixedWidth(46)
            b.clicked.connect(lambda _, f=fmt.lower(): self._export_fig(f))
            ctrl.addWidget(b)
        fig_layout.addLayout(ctrl)

        placeholder = QLabel("Run measurements to generate figures.")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setStyleSheet("color:palette(mid);font-size:11px;")
        fig_layout.addWidget(placeholder, 1)
        self._fig_canvas_widget = placeholder
        self._figures_layout = fig_layout
        tabs.addTab(fig_widget, "Figures")

        # Stats tab
        self._stats_text = QTextEdit()
        self._stats_text.setReadOnly(True)
        self._stats_text.setFontFamily("Monospace")
        self._stats_text.setFontPointSize(10)
        tabs.addTab(self._stats_text, "Stats")

        return tabs

    # ------------------------------------------------------------------
    # Public API (called by plugin)
    # ------------------------------------------------------------------

    def refresh_labels(self, labels: list[str]) -> None:
        prev = {lbl for lbl, cb in self._label_checks.items() if cb.isChecked()}
        for cb in list(self._label_checks.values()):
            self._label_layout.removeWidget(cb)
            cb.deleteLater()
        self._label_checks.clear()

        unique = sorted({l for l in labels if l is not None})
        has = bool(unique)
        self._no_labels.setVisible(not has)
        self._label_scroll.setVisible(has)
        for lbl in unique:
            cb = QCheckBox(lbl or "(unlabeled)")
            cb.setChecked(lbl in prev or not prev)
            self._label_checks[lbl] = cb
            self._label_layout.insertWidget(self._label_layout.count() - 1, cb)

    def set_pixel_size(self, ps_nm: float) -> None:
        if ps_nm > 0:
            self._px_spin.setValue(ps_nm)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)
        self._status.setVisible(running)
        if not running:
            self._progress.setValue(0)
            self._status.setText("")

    def show_measurements(self, df) -> None:
        """Load *df* into the panel, pre-select best metric, show Particles tab."""
        for prefer in ("ecd_nm", "area_nm2"):
            idx = self._metric_combo.findData(prefer)
            if idx >= 0 and prefer in df.columns:
                self._metric_combo.setCurrentIndex(idx)
                break
        self.show_results(df)
        self._results_tabs.setCurrentIndex(0)  # Particles table, not Figures

    def set_histogram_metric(self, key: str) -> None:
        """Select histogram x-axis metric by column key."""
        idx = self._metric_combo.findData(key)
        if idx >= 0:
            self._metric_combo.setCurrentIndex(idx)

    def set_histogram_bins(self, n: int) -> None:
        self._bins_spin.setValue(int(n))

    def set_plot_type(self, plot_type: str) -> None:
        """Switch histogram display type ('count' or 'density')."""
        idx = self._plot_type_combo.findData(plot_type)
        if idx >= 0:
            self._plot_type_combo.setCurrentIndex(idx)

    def show_progress(self, pct: int, msg: str) -> None:
        self._progress.setValue(pct)
        self._status.setText(msg)

    def show_results(self, df) -> None:
        self._df = df
        self._results_tabs.setVisible(True)
        self._populate_table(df)
        self._populate_stats(df)
        self._refresh_figure()
        self._results_tabs.setCurrentIndex(1)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _select_all(self)  -> None:
        for cb in self._label_checks.values(): cb.setChecked(True)

    def _select_none(self) -> None:
        for cb in self._label_checks.values(): cb.setChecked(False)

    def _on_run(self) -> None:
        selected = {lbl for lbl, cb in self._label_checks.items() if cb.isChecked()}
        if not selected:
            return
        mode = "batch" if self._mode_batch.isChecked() else "single"
        self.analysis_requested.emit({
            "mode":          mode,
            "selected_labels": selected,
            "pixel_size_nm": self._px_spin.value(),
        })

    def _populate_table(self, df) -> None:
        if df is None or df.empty:
            self._table.setRowCount(0)
            return
        cols_order = ["image", "label"] + [k for k in _METRIC_KEYS if k in df.columns]
        header_map = {k: _METRIC_LABEL.get(k, k) for k in cols_order}
        available = [c for c in cols_order if c in df.columns]

        self._table.setSortingEnabled(False)
        self._table.setColumnCount(len(available))
        self._table.setHorizontalHeaderLabels([header_map.get(c, c) for c in available])
        self._table.setRowCount(len(df))

        for row_i, (_, row) in enumerate(df.iterrows()):
            for col_i, col in enumerate(available):
                val = row.get(col, "")
                if isinstance(val, float):
                    text = "" if val != val else (f"{val:.4f}" if abs(val) < 1e5 else f"{val:.4e}")
                else:
                    text = str(val)
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(row_i, col_i, item)

        self._table.setSortingEnabled(True)
        self._table.resizeColumnsToContents()

    def _populate_stats(self, df) -> None:
        if df is None or df.empty:
            self._stats_text.setPlainText("No data.")
            return
        lines = []
        groups = sorted(df["label"].dropna().unique().tolist()) if "label" in df.columns else []
        for grp in groups:
            sub = df[df["label"] == grp]
            lines.append(f"[{grp}]  n={len(sub)}")
            for k, lbl, _ in _METRICS:
                if k not in sub.columns:
                    continue
                vals = sub[k].dropna()
                if len(vals) == 0:
                    continue
                lines.append(
                    f"  {lbl:<22}  mean={vals.mean():.4g}  median={vals.median():.4g}"
                    f"  std={vals.std():.4g}  min={vals.min():.4g}  max={vals.max():.4g}"
                )
            lines.append("")
        self._stats_text.setPlainText("\n".join(lines))

    def _refresh_figure(self) -> None:
        if self._df is None or self._df.empty:
            return
        key = self._metric_combo.currentData()
        if key not in self._df.columns:
            return
        try:
            fig = self._make_figure(key)
        except Exception:
            return
        if fig is None:
            return
        self._fig = fig
        self._install_canvas(fig)

    def _make_figure(self, key: str):
        df = self._df
        label_col = "label" if "label" in df.columns else None
        groups = sorted(df[label_col].dropna().unique().tolist()) if label_col else ["all"]
        palette = ["#4878D0","#EE854A","#6ACC65","#D65F5F","#956CB4","#8C613C","#DC7EC0","#797979"]
        colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}
        xlabel = _METRIC_LABEL.get(key, key)
        use_density = getattr(self, "_plot_type_combo", None) and self._plot_type_combo.currentData() == "density"
        n_bins = int(getattr(self, "_bins_spin", None) and self._bins_spin.value() or 30)

        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fig, ax = plt.subplots(figsize=(5.5, 3.8))
        all_vals = df[key].dropna().values
        if len(all_vals) == 0:
            plt.close(fig)
            return None

        lo, hi = all_vals.min(), all_vals.max()
        bins = np.linspace(lo, hi, n_bins + 1) if hi > lo else n_bins

        for grp in groups:
            sub = df[df[label_col] == grp][key].dropna().values if label_col else all_vals
            if len(sub) == 0:
                continue
            c = colors[grp]
            ax.hist(sub, bins=bins, alpha=0.38, color=c, density=use_density)
            if use_density and len(sub) > 3 and sub.std() > 1e-12:
                try:
                    from scipy.stats import gaussian_kde
                    kde = gaussian_kde(sub)
                    xs = np.linspace(lo, hi, 300)
                    ax.plot(xs, kde(xs), color=c, lw=1.8)
                except Exception:
                    pass
            ax.axvline(sub.mean(),     color=c, lw=1.2, ls="-")
            ax.axvline(np.median(sub), color=c, lw=1.2, ls="--")

        legend_handles = [
            *[plt.Rectangle((0,0),1,1,color=colors[g],alpha=0.5,
                             label=f"{g} (n={len(df[df[label_col]==g]) if label_col else len(df)})") for g in groups],
            Line2D([0],[0],color="k",lw=1.2,ls="-", label="mean"),
            Line2D([0],[0],color="k",lw=1.2,ls="--",label="median"),
        ]
        ax.legend(handles=legend_handles, fontsize=7, frameon=False)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel("Density" if use_density else "Count", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(direction="out", length=3, width=0.8, labelsize=8)
        fig.tight_layout(pad=0.8)
        return fig

    def _install_canvas(self, fig) -> None:
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        old = self._fig_canvas_widget
        canvas = FigureCanvasQTAgg(fig)
        canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        idx = self._figures_layout.indexOf(old)
        self._figures_layout.removeWidget(old)
        old.hide()
        old.deleteLater()
        self._figures_layout.insertWidget(idx, canvas, 1)
        self._fig_canvas_widget = canvas
        canvas.draw()

    def _export_fig(self, fmt: str) -> None:
        if self._fig is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export figure", str(Path.home() / f"particle_figure.{fmt}"),
            f"{fmt.upper()} files (*.{fmt});;All files (*)",
        )
        if path:
            try:
                self._fig.savefig(path, dpi=300, bbox_inches="tight")
            except Exception as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export failed", str(exc))

    def _export_csv(self) -> None:
        if self._df is None or self._df.empty:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export measurements", str(Path.home() / "particle_measurements.csv"),
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            try:
                self._df.to_csv(path, index=False)
            except Exception as exc:
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Export failed", str(exc))
