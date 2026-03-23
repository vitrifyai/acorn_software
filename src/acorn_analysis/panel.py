"""Analysis panel — surface area estimation and population statistics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
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


class AnalysisPanel(QWidget):
    """Tab panel for surface area estimation and population statistics."""

    analysis_requested = pyqtSignal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._label_checks: dict[str, QCheckBox] = {}
        self._particles_df = None
        self._stats_dict = None
        self._output_dir: Optional[Path] = None
        self._fig = None           # current matplotlib Figure
        self._fig_canvas = None    # FigureCanvasQTAgg
        self._folder_items: list[dict] = []   # [{path, pixel_size_nm, n_rois, labels}]
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_input_source_group())
        outer.addWidget(self._build_label_group())
        outer.addWidget(self._build_param_group())
        outer.addWidget(self._build_output_group())

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.clicked.connect(self._on_run_clicked)
        outer.addWidget(self._run_btn)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setVisible(False)
        outer.addWidget(self._progress)

        self._status_label = QLabel("")
        self._status_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._status_label.setVisible(False)
        outer.addWidget(self._status_label)

        self._results_tabs = self._build_results_tabs()
        self._results_tabs.setVisible(False)
        outer.addWidget(self._results_tabs, 1)

    def _build_input_source_group(self) -> QGroupBox:
        box = QGroupBox("Input Source")
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        radio_row = QHBoxLayout()
        self._src_session = QRadioButton("Session (loaded images)")
        self._src_folder  = QRadioButton("Folder")
        self._src_session.setChecked(True)
        self._src_session.toggled.connect(self._on_src_changed)
        radio_row.addWidget(self._src_session)
        radio_row.addWidget(self._src_folder)
        radio_row.addStretch()
        layout.addLayout(radio_row)

        # Folder controls (hidden by default)
        self._folder_widget = QWidget()
        folder_layout = QVBoxLayout(self._folder_widget)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(4)

        picker_row = QHBoxLayout()
        self._folder_edit = QLineEdit()
        self._folder_edit.setPlaceholderText("Select folder containing annotated images...")
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(70)
        browse_btn.clicked.connect(self._browse_folder)
        scan_btn = QPushButton("Scan")
        scan_btn.setFixedWidth(55)
        scan_btn.clicked.connect(self._on_scan_folder)
        picker_row.addWidget(self._folder_edit)
        picker_row.addWidget(browse_btn)
        picker_row.addWidget(scan_btn)
        folder_layout.addLayout(picker_row)

        self._folder_table = QTableWidget()
        self._folder_table.setColumnCount(4)
        self._folder_table.setHorizontalHeaderLabels(["Image", "Pixel size (nm/px)", "ROIs", "Labels found"])
        self._folder_table.setFixedHeight(130)
        self._folder_table.setAlternatingRowColors(True)
        self._folder_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self._folder_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self._folder_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self._folder_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._folder_table.itemChanged.connect(self._on_folder_table_item_changed)
        folder_layout.addWidget(self._folder_table)

        self._folder_widget.setVisible(False)
        layout.addWidget(self._folder_widget)
        return box

    def _build_label_group(self) -> QGroupBox:
        box = QGroupBox("Labels to Analyze")
        layout = QVBoxLayout(box)
        layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        sel_all = QPushButton("All")
        sel_all.setFixedWidth(56)
        sel_all.clicked.connect(self._select_all_labels)
        sel_none = QPushButton("None")
        sel_none.setFixedWidth(64)
        sel_none.clicked.connect(self._select_no_labels)
        btn_row.addWidget(sel_all)
        btn_row.addWidget(sel_none)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._label_container = QWidget()
        self._label_container_layout = QVBoxLayout(self._label_container)
        self._label_container_layout.setSpacing(2)
        self._label_container_layout.setContentsMargins(2, 2, 2, 2)
        self._label_container_layout.addStretch()

        self._label_scroll = QScrollArea()
        self._label_scroll.setWidgetResizable(True)
        self._label_scroll.setMaximumHeight(110)
        self._label_scroll.setWidget(self._label_container)
        layout.addWidget(self._label_scroll)

        self._no_labels_label = QLabel("No ROI annotations found in loaded images.")
        self._no_labels_label.setStyleSheet("font-size: 11px; color: palette(mid);")
        self._no_labels_label.setWordWrap(True)
        layout.addWidget(self._no_labels_label)

        self._label_scroll.setVisible(False)
        return box

    def _build_param_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        form.setSpacing(6)

        mode_widget = QWidget()
        mode_row = QHBoxLayout(mode_widget)
        mode_row.setContentsMargins(0, 0, 0, 0)
        mode_row.setSpacing(12)
        self._mode_single = QRadioButton("Current image")
        self._mode_batch = QRadioButton("All images")
        self._mode_single.setChecked(True)
        mode_row.addWidget(self._mode_single)
        mode_row.addWidget(self._mode_batch)
        mode_row.addStretch()
        form.addRow("Mode:", mode_widget)

        self._px_spin = QDoubleSpinBox()
        self._px_spin.setRange(0.0, 10000.0)
        self._px_spin.setDecimals(4)
        self._px_spin.setSuffix(" nm/px")
        self._px_spin.setToolTip(
            "Default pixel size in nm/px — used as fallback when per-image pixel size\n"
            "is not set. In session mode, each image uses its own saved pixel size automatically."
        )
        form.addRow("Default pixel size:", self._px_spin)

        self._px_unc_spin = QDoubleSpinBox()
        self._px_unc_spin.setRange(0.0, 100.0)
        self._px_unc_spin.setDecimals(4)
        self._px_unc_spin.setSuffix(" nm")
        self._px_unc_spin.setToolTip(
            "Absolute uncertainty in pixel size (nm) used for error propagation. "
            "Leave at 0 to skip uncertainty estimation."
        )
        form.addRow("Px uncertainty:", self._px_unc_spin)

        self._method_combo = QComboBox()
        _METHODS = [
            ("Auto",              "auto",          "Auto-select based on shape metrics (recommended)"),
            ("Ellipsoid",         "ellipsoid",     "Fit prolate/oblate spheroid — best for smooth, round particles"),
            ("Cauchy",            "cauchy",        "Cauchy-Crofton theorem — convex irregular particles"),
            ("Fourier",           "fourier",       "Radial Fourier decomposition — rough/irregular surfaces"),
            ("Fourier (spiky)",   "fourier_spiky", "Fourier + spike modelling — fractal or highly spiky surfaces"),
            ("Capsule",           "capsule",       "Cylindrical capsule model — rod-shaped particles"),
            ("Perimeter",         "perimeter",     "Direct perimeter scaled by Cauchy factor"),
        ]
        for label, key, tip in _METHODS:
            self._method_combo.addItem(label, userData=key)
            self._method_combo.setItemData(
                self._method_combo.count() - 1, tip,
                Qt.ItemDataRole.ToolTipRole,
            )
        self._method_combo.setToolTip("Surface area estimation method.")
        form.addRow("SA method:", self._method_combo)

        self._compound_check = QCheckBox("Combine same-label annotations into one mask")
        self._compound_check.setToolTip(
            "When multiple ROI annotations share a label on the same image,\n"
            "combine them into a single compound mask before estimating SA.\n"
            "Use this for hollow particles (draw outer + inner boundary)\n"
            "or particles with separate dense regions."
        )
        self._compound_check.toggled.connect(self._on_compound_toggled)
        form.addRow("Compound:", self._compound_check)

        self._compound_mode_widget = QWidget()
        cm_row = QHBoxLayout(self._compound_mode_widget)
        cm_row.setContentsMargins(0, 0, 0, 0)
        cm_row.setSpacing(10)
        self._cm_auto   = QRadioButton("Auto")
        self._cm_sub    = QRadioButton("Subtract inner (donut / hole)")
        self._cm_union  = QRadioButton("Add / union (stacked regions)")
        self._cm_auto.setChecked(True)
        self._cm_auto.setToolTip("If a smaller polygon is fully inside a larger one: subtract it.")
        self._cm_sub.setToolTip("Always subtract smaller polygons from the largest.")
        self._cm_union.setToolTip("Always union all polygons.")
        cm_row.addWidget(self._cm_auto)
        cm_row.addWidget(self._cm_sub)
        cm_row.addWidget(self._cm_union)
        cm_row.addStretch()
        self._compound_mode_widget.setEnabled(False)
        form.addRow("", self._compound_mode_widget)

        return box

    def _on_compound_toggled(self, checked: bool) -> None:
        self._compound_mode_widget.setEnabled(checked)

    def _build_output_group(self) -> QGroupBox:
        box = QGroupBox("Output Folder")
        row = QHBoxLayout(box)
        row.setSpacing(4)
        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText("Default: <image folder>/acorn_analysis/<stem>_<timestamp>/")
        browse = QPushButton("Browse")
        browse.setFixedWidth(75)
        browse.clicked.connect(self._browse_output)
        row.addWidget(self._out_edit)
        row.addWidget(browse)
        return box

    def _build_results_tabs(self) -> QTabWidget:
        tabs = QTabWidget()

        # Particles tab
        particles_widget = QWidget()
        particles_layout = QVBoxLayout(particles_widget)
        particles_layout.setContentsMargins(4, 4, 4, 4)
        particles_layout.setSpacing(4)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Display units:"))
        self._unit_combo = QComboBox()
        self._unit_combo.addItem("\u03bcm\u00b2", userData="um2")
        self._unit_combo.addItem("nm\u00b2", userData="nm2")
        self._unit_combo.setFixedWidth(70)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch()
        particles_layout.addLayout(unit_row)

        self._particles_table = QTableWidget()
        self._particles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._particles_table.setAlternatingRowColors(True)
        self._particles_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._particles_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._particles_table.setSortingEnabled(True)
        particles_layout.addWidget(self._particles_table)
        tabs.addTab(particles_widget, "Particles")

        # Groups tab
        self._groups_text = QTextEdit()
        self._groups_text.setReadOnly(True)
        self._groups_text.setFontFamily("Monospace")
        self._groups_text.setFontPointSize(10)
        tabs.addTab(self._groups_text, "Groups")

        # Figures tab — embedded matplotlib canvas
        figures_widget = QWidget()
        figures_layout = QVBoxLayout(figures_widget)
        figures_layout.setSpacing(4)
        figures_layout.setContentsMargins(4, 4, 4, 4)

        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Plot:"))
        self._fig_type_combo = QComboBox()
        self._fig_type_combo.addItems(["Histogram (KDE)", "Violin / Box", "ECDF", "Summary panel"])
        self._fig_type_combo.currentIndexChanged.connect(self._refresh_figure)
        ctrl_row.addWidget(self._fig_type_combo)
        ctrl_row.addStretch()
        for fmt in ("PNG", "SVG", "PDF"):
            btn = QPushButton(fmt)
            btn.setFixedWidth(46)
            btn.clicked.connect(lambda _, f=fmt.lower(): self._export_figure(f))
            ctrl_row.addWidget(btn)
        figures_layout.addLayout(ctrl_row)

        # Matplotlib canvas placeholder — real canvas created on first use
        self._fig_placeholder = QLabel("Run analysis to generate figures.")
        self._fig_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._fig_placeholder.setStyleSheet("color: palette(mid); font-size: 11px;")
        figures_layout.addWidget(self._fig_placeholder, 1)
        self._fig_canvas_widget = self._fig_placeholder   # updated when canvas is created

        self._figures_text = QTextEdit()
        self._figures_text.setReadOnly(True)
        self._figures_text.setFontFamily("Monospace")
        self._figures_text.setFontPointSize(9)
        self._figures_text.setMaximumHeight(60)
        figures_layout.addWidget(self._figures_text)

        self._figures_layout = figures_layout
        tabs.addTab(figures_widget, "Figures")

        return tabs

    # ── input source ──────────────────────────────────────────────────────────

    def _on_src_changed(self) -> None:
        self._folder_widget.setVisible(not self._src_session.isChecked())

    def _browse_folder(self) -> None:
        start = self._folder_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Select folder of annotated images", start)
        if d:
            self._folder_edit.setText(d)

    def _on_scan_folder(self) -> None:
        folder = self._folder_edit.text().strip()
        if not folder or not Path(folder).exists():
            return

        from acorn.core.dm4_loader import scan_folder as _scan
        image_paths = _scan(Path(folder))

        self._folder_items = []
        all_labels: set[str] = set()

        self._folder_table.blockSignals(True)
        self._folder_table.setRowCount(0)

        for img_path in image_paths:
            sidecar = img_path.parent / f".{img_path.stem}.acorn.json"
            px_nm = 0.0
            n_rois = 0
            labels: list[str] = []

            if sidecar.exists():
                try:
                    raw = json.loads(sidecar.read_text())
                    if isinstance(raw, dict):
                        px_nm = float(raw.get("pixel_size_nm") or 0.0)
                        for ann in raw.get("annotations", []):
                            if ann.get("type") == "roi":
                                n_rois += 1
                                lbl = ann.get("label", "")
                                if lbl:
                                    labels.append(lbl)
                    all_labels.update(labels)
                except Exception:
                    pass

            self._folder_items.append({
                "path":          str(img_path),
                "pixel_size_nm": px_nm,
                "n_rois":        n_rois,
                "labels":        labels,
            })

            row = self._folder_table.rowCount()
            self._folder_table.insertRow(row)

            name_item = QTableWidgetItem(img_path.name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._folder_table.setItem(row, 0, name_item)

            px_item = QTableWidgetItem(f"{px_nm:.4f}" if px_nm > 0 else "")
            px_item.setToolTip("Double-click to edit pixel size for this image")
            self._folder_table.setItem(row, 1, px_item)

            roi_item = QTableWidgetItem(str(n_rois))
            roi_item.setFlags(roi_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._folder_table.setItem(row, 2, roi_item)

            lbl_item = QTableWidgetItem(", ".join(sorted(set(labels))))
            lbl_item.setFlags(lbl_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self._folder_table.setItem(row, 3, lbl_item)

        self._folder_table.blockSignals(False)

        if all_labels:
            self.refresh_labels(sorted(all_labels))

    def _on_folder_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != 1:
            return
        row = item.row()
        if row >= len(self._folder_items):
            return
        try:
            val = float(item.text().strip())
            self._folder_items[row]["pixel_size_nm"] = val
        except (ValueError, TypeError):
            self._folder_items[row]["pixel_size_nm"] = 0.0

    # ── public API ────────────────────────────────────────────────────────────

    def refresh_labels(self, labels: list[str]) -> None:
        previously_checked = {lbl for lbl, cb in self._label_checks.items() if cb.isChecked()}
        for cb in list(self._label_checks.values()):
            self._label_container_layout.removeWidget(cb)
            cb.deleteLater()
        self._label_checks.clear()

        unique = sorted({lbl for lbl in labels if lbl is not None})
        has = bool(unique)
        self._no_labels_label.setVisible(not has)
        self._label_scroll.setVisible(has)

        for lbl in unique:
            display = lbl if lbl else "(unlabeled)"
            cb = QCheckBox(display)
            cb.setChecked(lbl in previously_checked or not previously_checked)
            self._label_checks[lbl] = cb
            self._label_container_layout.insertWidget(
                self._label_container_layout.count() - 1, cb
            )

    def set_pixel_size(self, ps_nm: float) -> None:
        if ps_nm > 0:
            self._px_spin.setValue(ps_nm)

    def set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._progress.setVisible(running)
        self._status_label.setVisible(running)
        if not running:
            self._progress.setValue(0)
            self._status_label.setText("")

    def show_progress(self, value: int, message: str) -> None:
        self._progress.setValue(value)
        self._status_label.setText(message)

    def show_results(
        self,
        particles_df,
        stats_dict: dict | None,
        output_dir: Path | None,
    ) -> None:
        self._results_tabs.setVisible(True)
        self._output_dir = output_dir
        self._particles_df = particles_df
        self._stats_dict = stats_dict

        self._populate_particles_table(particles_df)
        self._populate_groups_text(stats_dict)
        self._populate_figures_text(output_dir)
        self._refresh_figure()

        self._results_tabs.setCurrentIndex(2)  # jump to Figures tab

    # ── figure ────────────────────────────────────────────────────────────────

    def _refresh_figure(self) -> None:
        if self._particles_df is None:
            return
        plot_type = self._fig_type_combo.currentText()
        try:
            fig = self._generate_inline_figure(plot_type)
        except Exception:
            return
        if fig is None:
            return
        self._fig = fig
        self._install_canvas(fig)

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
        self._fig_canvas = canvas
        canvas.draw()

    def _generate_inline_figure(self, plot_type: str):
        df = self._particles_df
        if df is None or "SA_nm2" not in df.columns:
            return None

        use_um2 = self._unit_combo.currentData() == "um2"
        scale   = 1e-6 if use_um2 else 1.0
        unit    = "\u03bcm\u00b2" if use_um2 else "nm\u00b2"

        import pandas as _pd
        plot_df = df.copy()
        plot_df["_sa"] = plot_df["SA_nm2"] * scale

        groups = sorted(plot_df["label"].dropna().unique().tolist())
        palette = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F",
                   "#956CB4", "#8C613C", "#DC7EC0", "#797979"]
        colors = {g: palette[i % len(palette)] for i, g in enumerate(groups)}

        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from matplotlib.lines import Line2D

        if plot_type == "Histogram (KDE)":
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            all_vals = plot_df["_sa"].dropna().values
            if len(all_vals) == 0:
                plt.close(fig)
                return None
            bin_edges = _smart_bins(all_vals, n=30)
            for name in groups:
                data = plot_df[plot_df["label"] == name]["_sa"].dropna().values
                color = colors[name]
                ax.hist(data, bins=bin_edges, alpha=0.38, color=color, density=True)
                if len(data) > 3 and data.std() > 1e-12:
                    try:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(data)
                        xs = _np_linspace(all_vals.min(), all_vals.max(), 300)
                        ax.plot(xs, kde(xs), color=color, lw=1.8, label=f"{name} (n={len(data)})")
                    except Exception:
                        ax.plot([], [], color=color, lw=1.8, label=f"{name} (n={len(data)})")
                else:
                    ax.plot([], [], color=color, lw=1.8, label=f"{name} (n={len(data)})")
                m = float(data.mean()) if len(data) > 0 else 0
                med = float(_np_median(data)) if len(data) > 0 else 0
                ax.axvline(m,   color=color, lw=1.2, ls="-")
                ax.axvline(med, color=color, lw=1.2, ls="--")

            legend_handles = [
                *[plt.Rectangle((0,0),1,1, color=colors[g], alpha=0.5, label=g) for g in groups],
                Line2D([0],[0], color="k", lw=1.2, ls="-",  label="mean"),
                Line2D([0],[0], color="k", lw=1.2, ls="--", label="median"),
            ]
            ax.legend(handles=legend_handles, fontsize=7, frameon=False)
            ax.set_xlabel(f"Surface area ({unit})", fontsize=9)
            ax.set_ylabel("Density", fontsize=9)
            _pub_style(ax)
            fig.tight_layout(pad=0.8)

        elif plot_type == "Violin / Box":
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            box_data = [plot_df[plot_df["label"] == g]["_sa"].dropna().values for g in groups]
            valid = [(g, d) for g, d in zip(groups, box_data) if len(d) > 1]
            if not valid:
                plt.close(fig)
                return None
            v_groups, v_data = zip(*valid)
            parts = ax.violinplot(v_data, positions=range(1, len(v_data)+1),
                                  showmedians=True, showextrema=True)
            for i, (pc, g) in enumerate(zip(parts["bodies"], v_groups)):
                pc.set_facecolor(colors[g])
                pc.set_alpha(0.65)
            for key in ("cmedians", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color("#333333")
                    parts[key].set_linewidth(0.8)
            ax.set_xticks(range(1, len(v_groups)+1))
            ax.set_xticklabels(v_groups, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(f"Surface area ({unit})", fontsize=9)
            # Annotate n
            for i, (g, d) in enumerate(zip(v_groups, v_data), start=1):
                ax.text(i, ax.get_ylim()[0], f"n={len(d)}", ha="center", va="bottom",
                        fontsize=7, color="#555555")
            _pub_style(ax)
            fig.tight_layout(pad=0.8)

        elif plot_type == "ECDF":
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            import numpy as _np
            for name in groups:
                data = _np.sort(plot_df[plot_df["label"] == name]["_sa"].dropna().values)
                if len(data) == 0:
                    continue
                y = _np.arange(1, len(data)+1) / len(data)
                ax.step(data, y, where="post", color=colors[name], lw=1.8,
                        label=f"{name} (n={len(data)})")
            ax.set_xlabel(f"Surface area ({unit})", fontsize=9)
            ax.set_ylabel("Cumulative fraction", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=7, frameon=False)
            _pub_style(ax)
            fig.tight_layout(pad=0.8)

        elif plot_type == "Summary panel":
            fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.8))
            import numpy as _np
            # Left: bar chart of mean +/- std
            means = [plot_df[plot_df["label"]==g]["_sa"].mean() for g in groups]
            stds  = [plot_df[plot_df["label"]==g]["_sa"].std()  for g in groups]
            ns    = [int((plot_df["label"]==g).sum()) for g in groups]
            x = range(len(groups))
            axes[0].bar(x, means, yerr=stds, color=[colors[g] for g in groups],
                        alpha=0.75, capsize=4, error_kw={"lw": 1.2})
            axes[0].set_xticks(x)
            axes[0].set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
            axes[0].set_ylabel(f"Mean SA ({unit})", fontsize=9)
            axes[0].set_title("Mean +/- SD", fontsize=9)
            for xi, (m, n) in enumerate(zip(means, ns)):
                axes[0].text(xi, 0, f"n={n}", ha="center", va="bottom", fontsize=7, color="#555")
            # Right: box
            box_data = [plot_df[plot_df["label"]==g]["_sa"].dropna().values for g in groups]
            bp = axes[1].boxplot(box_data, patch_artist=True, widths=0.5,
                                 medianprops={"color": "#333", "lw": 1.5})
            for patch, g in zip(bp["boxes"], groups):
                patch.set_facecolor(colors[g])
                patch.set_alpha(0.7)
            axes[1].set_xticks(range(1, len(groups)+1))
            axes[1].set_xticklabels(groups, rotation=30, ha="right", fontsize=8)
            axes[1].set_ylabel(f"Surface area ({unit})", fontsize=9)
            axes[1].set_title("Distribution", fontsize=9)
            for ax in axes:
                _pub_style(ax)
            fig.tight_layout(pad=0.8)

        else:
            return None

        return fig

    def _export_figure(self, fmt: str) -> None:
        if self._fig is None:
            return
        ext = fmt.lower()
        path, _ = QFileDialog.getSaveFileName(
            self, f"Export figure as {ext.upper()}",
            str(Path.home() / f"figure.{ext}"),
            f"{ext.upper()} files (*.{ext});;All files (*)",
        )
        if not path:
            return
        try:
            self._fig.savefig(path, dpi=300, bbox_inches="tight")
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Export failed", str(exc))

    # ── private helpers ───────────────────────────────────────────────────────

    def _select_all_labels(self) -> None:
        for cb in self._label_checks.values():
            cb.setChecked(True)

    def _select_no_labels(self) -> None:
        for cb in self._label_checks.values():
            cb.setChecked(False)

    def _browse_output(self) -> None:
        start = self._out_edit.text().strip() or str(Path.home())
        d = QFileDialog.getExistingDirectory(self, "Select output folder", start)
        if d:
            self._out_edit.setText(d)

    def _on_run_clicked(self) -> None:
        selected = [lbl for lbl, cb in self._label_checks.items() if cb.isChecked()]
        if not selected:
            return
        if self._compound_check.isChecked():
            if self._cm_sub.isChecked():
                compound_mode = "subtract_inner"
            elif self._cm_union.isChecked():
                compound_mode = "union"
            else:
                compound_mode = "auto"
        else:
            compound_mode = "separate"

        if self._src_folder.isChecked():
            mode = "folder"
        elif self._mode_batch.isChecked():
            mode = "batch"
        else:
            mode = "single"

        self.analysis_requested.emit({
            "mode":                       mode,
            "selected_labels":            selected,
            "pixel_size_nm":              self._px_spin.value(),
            "pixel_size_uncertainty_nm":  self._px_unc_spin.value(),
            "output_dir":                 self._out_edit.text().strip(),
            "method":                     self._method_combo.currentData(),
            "compound_mode":              compound_mode,
            "folder_items":               list(self._folder_items),
            "folder_path":                self._folder_edit.text().strip(),
        })

    def _on_open_output_folder(self) -> None:
        d = getattr(self, "_output_dir", None)
        if d and Path(d).exists():
            import subprocess
            subprocess.Popen(["xdg-open", str(d)])

    def _on_unit_changed(self) -> None:
        df = getattr(self, "_particles_df", None)
        if df is not None:
            self._populate_particles_table(df)
        stats = getattr(self, "_stats_dict", None)
        self._populate_groups_text(stats)
        self._refresh_figure()

    def _populate_particles_table(self, df) -> None:
        use_um2 = self._unit_combo.currentData() == "um2"
        sa_unit = "\u03bcm\u00b2" if use_um2 else "nm\u00b2"
        sa_scale = 1e-6 if use_um2 else 1.0
        _SA_COLS = {"SA_nm2", "SA_nm2_uncertainty", "SA_outer_nm2", "SA_inner_nm2"}

        COLS = [
            ("label",                       "Label"),
            ("image",                       "Image"),
            ("particle_id",                 "ID"),
            ("SA_nm2",                      f"SA total ({sa_unit})"),
            ("SA_outer_nm2",                f"SA outer ({sa_unit})"),
            ("SA_inner_nm2",                f"SA inner ({sa_unit})"),
            ("SA_nm2_uncertainty",          f"SA uncertainty ({sa_unit})"),
            ("method_used",                 "Method"),
            ("a_nm",                        "Semi-axis a (nm)"),
            ("b_nm",                        "Semi-axis b (nm)"),
            ("coverage_score",              "Coverage (0-1)"),
            ("is_hollow",                   "Hollow"),
            ("shell_thickness_estimate_nm", "Shell thickness (nm)"),
            ("aggregate_score",             "Aggregate (0-1)"),
            ("sem_roughness_index",         "SEM roughness index"),
            ("flagged",                     "Flagged"),
            ("flag_reason",                 "Flag reason"),
        ]
        _hollow_cols = {"SA_outer_nm2", "SA_inner_nm2"}
        _any_hollow = bool(df.get("is_hollow", False).any()) if "is_hollow" in df.columns else False
        available = [
            (k, h) for k, h in COLS
            if k in df.columns and (k not in _hollow_cols or _any_hollow)
        ]
        self._particles_table.setSortingEnabled(False)
        self._particles_table.setColumnCount(len(available))
        self._particles_table.setHorizontalHeaderLabels([h for _, h in available])
        self._particles_table.setRowCount(len(df))

        for row_i, (_, row) in enumerate(df.iterrows()):
            for col_i, (key, _) in enumerate(available):
                val = row.get(key, "")
                if isinstance(val, bool):
                    text = "yes" if val else "no"
                elif isinstance(val, float):
                    if key in _SA_COLS:
                        v = val * sa_scale
                    else:
                        v = val
                    if v != v:
                        text = ""
                    elif v == 0.0:
                        text = "0"
                    elif abs(v) >= 1e4 or (abs(v) < 0.001 and v != 0.0):
                        text = f"{v:.4e}"
                    else:
                        text = f"{v:.3f}"
                else:
                    text = str(val) if val != "" else ""
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._particles_table.setItem(row_i, col_i, item)

        self._particles_table.setSortingEnabled(True)
        self._particles_table.resizeColumnsToContents()

    def _populate_groups_text(self, stats_dict: dict | None) -> None:
        if stats_dict is None:
            self._groups_text.setPlainText("Single group — no between-group statistics computed.")
            return

        use_um2 = self._unit_combo.currentData() == "um2"
        sa_unit = "\u03bcm\u00b2" if use_um2 else "nm\u00b2"
        sa_scale = 1e-6 if use_um2 else 1.0

        lines: list[str] = []
        test = stats_dict.get("test_used", "none")
        n_groups = stats_dict.get("n_groups", 0)
        lines.append(f"Test: {test}   ({n_groups} groups)")

        omnibus = stats_dict.get("omnibus")
        if omnibus:
            stat = omnibus.get("statistic", float("nan"))
            p = omnibus.get("p_value", float("nan"))
            eta2 = omnibus.get("eta_squared", float("nan"))
            stars = _p_stars(p)
            lines.append(f"Omnibus: statistic={stat:.4f}, p={p:.4g} ({stars}), eta2={eta2:.4f}")

        lines.append("")
        group_stats = stats_dict.get("group_stats", {})
        for grp, s in group_stats.items():
            if not isinstance(s, dict) or s.get("n", 0) == 0:
                continue
            lines.append(f"[{grp}]")
            lines.append(f"  n           = {s['n']}")
            lines.append(f"  mean SA     = {s.get('mean', float('nan')) * sa_scale:.4g} {sa_unit}")
            lines.append(f"  median SA   = {s.get('median', float('nan')) * sa_scale:.4g} {sa_unit}")
            lines.append(f"  std         = {s.get('std', float('nan')) * sa_scale:.4g} {sa_unit}")
            lines.append(f"  IQR         = {s.get('iqr', float('nan')) * sa_scale:.4g} {sa_unit}")
            ci_lo = s.get("ci95_lo", float("nan")) * sa_scale
            ci_hi = s.get("ci95_hi", float("nan")) * sa_scale
            lines.append(f"  95% CI      = [{ci_lo:.4g}, {ci_hi:.4g}] {sa_unit}")
            lines.append("")

        pairwise = stats_dict.get("pairwise")
        if pairwise is not None and hasattr(pairwise, "iterrows") and len(pairwise) > 0 and n_groups >= 3:
            lines.append("Pairwise comparisons (Dunn + BH correction):")
            for _, pw_row in pairwise.iterrows():
                p_fdr = pw_row.get("p_fdr", float("nan"))
                sig = pw_row.get("significant", False)
                marker = " *" if sig else ""
                lines.append(
                    f"  {pw_row.get('group_a','?')} vs {pw_row.get('group_b','?')}: "
                    f"p_fdr={p_fdr:.4g}{marker}"
                )

        self._groups_text.setPlainText("\n".join(lines))

    def _populate_figures_text(self, output_dir: Path | None) -> None:
        if output_dir is None or not Path(output_dir).exists():
            self._figures_text.setPlainText("No output folder set — figures not saved to disk.")
            return
        figs = sorted(Path(output_dir).glob("*.png")) + sorted(Path(output_dir).glob("*.svg"))
        lines = [f"Saved: {output_dir}"]
        for f in figs[:8]:
            lines.append(f"  {f.name}")
        if len(figs) > 8:
            lines.append(f"  ... and {len(figs)-8} more")
        self._figures_text.setPlainText("\n".join(lines))


# ── module-level helpers ──────────────────────────────────────────────────────

def _p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _pub_style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=8)


def _smart_bins(vals, n: int = 30):
    import numpy as _np
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        return n
    return _np.linspace(lo, hi, n + 1)


def _np_linspace(a, b, n):
    import numpy as _np
    return _np.linspace(a, b, n)


def _np_median(arr):
    import numpy as _np
    return _np.median(arr)
