"""Analysis panel — surface area estimation and population statistics.

Displayed as the "Analysis" tab in the main QTabWidget.  The panel owns its
own UI; the heavy computation runs in AnalysisThread (main_window.py) so the
GUI stays responsive.

Public API (called by MainWindow)
----------------------------------
refresh_labels(labels)      -- rebuild label checkbox list from annotation store
set_pixel_size(ps_nm)       -- update pixel-size spinbox when image changes
set_running(bool)           -- toggle progress / disable run button
show_progress(pct, msg)     -- update progress bar + status label
show_results(df, stats, dir)-- populate results tabs after analysis completes
"""

from __future__ import annotations

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
        self._build_ui()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

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

        # Scrollable checkbox container
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

        # Initial state: no images loaded
        self._label_scroll.setVisible(False)
        return box

    def _build_param_group(self) -> QGroupBox:
        box = QGroupBox("Parameters")
        form = QFormLayout(box)
        form.setSpacing(6)

        # Mode
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
            "Pixel size in nm per pixel. Set automatically from image header; "
            "override here if needed."
        )
        form.addRow("Pixel size:", self._px_spin)

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
        self._method_combo.setToolTip("Surface area estimation method.\n'Auto' selects the best method per particle based on circularity, convexity, and fractal dimension.")
        form.addRow("SA method:", self._method_combo)

        # Compound mask controls
        self._compound_check = QCheckBox("Combine same-label annotations into one mask")
        self._compound_check.setToolTip(
            "When multiple ROI annotations share a label on the same image,\n"
            "combine them into a single compound mask before estimating SA.\n"
            "Use this for hollow particles (draw outer + inner boundary)\n"
            "or particles with separate dense regions (draw each region separately)."
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
        self._cm_auto.setToolTip("If a smaller polygon is fully inside a larger one: subtract it.\nIf polygons overlap or are adjacent: union them.")
        self._cm_sub.setToolTip("Always subtract smaller polygons from the largest — use for hollow particles, donuts, liposomes.")
        self._cm_union.setToolTip("Always union all polygons — use for particles with dark internal regions or overlapping/touching particles.")
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

        # Particles tab — unit toggle + scrollable table
        particles_widget = QWidget()
        particles_layout = QVBoxLayout(particles_widget)
        particles_layout.setContentsMargins(4, 4, 4, 4)
        particles_layout.setSpacing(4)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Display units:"))
        self._unit_combo = QComboBox()
        self._unit_combo.addItem("nm\u00b2", userData="nm2")
        self._unit_combo.addItem("\u03bcm\u00b2", userData="um2")
        self._unit_combo.setFixedWidth(70)
        self._unit_combo.currentIndexChanged.connect(self._on_unit_changed)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch()
        particles_layout.addLayout(unit_row)

        self._particles_table = QTableWidget()
        self._particles_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._particles_table.setAlternatingRowColors(True)
        self._particles_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self._particles_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )
        self._particles_table.setSortingEnabled(True)
        particles_layout.addWidget(self._particles_table)
        tabs.addTab(particles_widget, "Particles")

        # Groups tab — formatted plain text
        self._groups_text = QTextEdit()
        self._groups_text.setReadOnly(True)
        self._groups_text.setFontFamily("Monospace")
        self._groups_text.setFontPointSize(10)
        tabs.addTab(self._groups_text, "Groups")

        # Figures tab — file list + open folder button
        figures_widget = QWidget()
        figures_layout = QVBoxLayout(figures_widget)
        figures_layout.setSpacing(4)
        figures_layout.setContentsMargins(4, 4, 4, 4)
        self._open_folder_btn = QPushButton("Open Output Folder")
        self._open_folder_btn.clicked.connect(self._on_open_output_folder)
        figures_layout.addWidget(self._open_folder_btn)
        self._figures_text = QTextEdit()
        self._figures_text.setReadOnly(True)
        self._figures_text.setFontFamily("Monospace")
        self._figures_text.setFontPointSize(10)
        figures_layout.addWidget(self._figures_text, 1)
        tabs.addTab(figures_widget, "Figures")

        return tabs

    # ── public API ────────────────────────────────────────────────────────────

    def refresh_labels(self, labels: list[str]) -> None:
        """Rebuild label checkboxes. Previously checked labels stay checked."""
        previously_checked = {
            lbl for lbl, cb in self._label_checks.items() if cb.isChecked()
        }

        # Clear existing checkboxes (leave the trailing stretch)
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
            # Default: checked if previously checked, or if this is a fresh load (nothing was checked before)
            cb.setChecked(lbl in previously_checked or not previously_checked)
            self._label_checks[lbl] = cb
            # Insert before the stretch (last item)
            self._label_container_layout.insertWidget(
                self._label_container_layout.count() - 1, cb
            )

    def set_pixel_size(self, ps_nm: float) -> None:
        """Called by MainWindow when the current image changes."""
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
        """Populate the three results tabs and make them visible."""
        self._results_tabs.setVisible(True)
        self._output_dir = output_dir
        self._particles_df = particles_df  # kept for unit toggle re-render

        self._populate_particles_table(particles_df)
        self._populate_groups_text(stats_dict)
        self._populate_figures_text(output_dir)

        self._results_tabs.setCurrentIndex(0)

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

        self.analysis_requested.emit({
            "mode": "batch" if self._mode_batch.isChecked() else "single",
            "selected_labels": selected,
            "pixel_size_nm": self._px_spin.value(),
            "pixel_size_uncertainty_nm": self._px_unc_spin.value(),
            "output_dir": self._out_edit.text().strip(),
            "method": self._method_combo.currentData(),
            "compound_mode": compound_mode,
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

    def _populate_particles_table(self, df) -> None:
        use_um2 = self._unit_combo.currentData() == "um2"
        sa_unit = "\u03bcm\u00b2" if use_um2 else "nm\u00b2"
        sa_scale = 1e-6 if use_um2 else 1.0

        # SA columns that need unit conversion
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
                    if v != v:          # NaN
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
                item.setTextAlignment(
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
                )
                self._particles_table.setItem(row_i, col_i, item)

        self._particles_table.setSortingEnabled(True)
        self._particles_table.resizeColumnsToContents()

    def _populate_groups_text(self, stats_dict: dict | None) -> None:
        if stats_dict is None:
            self._groups_text.setPlainText(
                "Single group — no between-group statistics computed."
            )
            return

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
        else:
            pairwise = stats_dict.get("pairwise")
            if pairwise is not None and hasattr(pairwise, "iterrows"):
                for _, pw_row in pairwise.iterrows():
                    p = pw_row.get("p_fdr", pw_row.get("p_raw", float("nan")))
                    eff = pw_row.get("effect_size", float("nan"))
                    stars = _p_stars(p)
                    lines.append(
                        f"{pw_row.get('group_a','?')} vs {pw_row.get('group_b','?')}: "
                        f"p={p:.4g} ({stars}), effect size={eff:.4f}"
                    )

        lines.append("")
        group_stats = stats_dict.get("group_stats", {})
        for grp, s in group_stats.items():
            if not isinstance(s, dict) or s.get("n", 0) == 0:
                continue
            lines.append(f"[{grp}]")
            lines.append(f"  n           = {s['n']}")
            lines.append(f"  mean SA     = {s.get('mean', float('nan')):.2f} nm\u00b2")
            lines.append(f"  median SA   = {s.get('median', float('nan')):.2f} nm\u00b2")
            lines.append(f"  std         = {s.get('std', float('nan')):.2f} nm\u00b2")
            lines.append(f"  IQR         = {s.get('iqr', float('nan')):.2f} nm\u00b2")
            ci_lo = s.get("ci95_lo", float("nan"))
            ci_hi = s.get("ci95_hi", float("nan"))
            lines.append(f"  95% CI      = [{ci_lo:.2f}, {ci_hi:.2f}] nm\u00b2")
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
            self._figures_text.setPlainText("No output folder.")
            return

        figs = sorted(Path(output_dir).glob("*.png"))
        csvs = sorted(Path(output_dir).glob("*.csv"))

        lines = [f"Saved to: {output_dir}", ""]
        if figs:
            lines.append("Figures (PNG + SVG):")
            for f in figs:
                lines.append(f"  {f.name}")
        if csvs:
            lines.append("")
            lines.append("Tables (CSV):")
            for f in csvs:
                lines.append(f"  {f.name}")

        self._figures_text.setPlainText("\n".join(lines))


# ── module-level helper ───────────────────────────────────────────────────────

def _p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"
