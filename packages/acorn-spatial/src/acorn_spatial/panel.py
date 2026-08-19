"""Spatial analysis panel — clustering, hotspots, nearest-neighbour, and
cross-label association of detected features."""

from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QTextEdit, QVBoxLayout, QWidget,
)

from acorn_spatial import spatial as S


class SpatialPanel(QWidget):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self._context = context
        self._fig = None
        self._canvas = None
        self._build_ui()
        context.annotations_changed.connect(lambda *_: self._refresh_labels())
        context.image_loaded.connect(self._on_image_loaded)

    # ── UI ──────────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # scope
        scope_box = QGroupBox("Source")
        scope_form = QFormLayout(scope_box)
        self._scope = QComboBox()
        self._scope.addItem("Current image", "current")
        self._scope.addItem("All images (per-image + compare conditions)", "all")
        scope_form.addRow("Analyze:", self._scope)
        from PyQt6.QtWidgets import QLineEdit
        self._group_re = QLineEdit(r"^\d+_([A-Za-z0-9]+)")
        self._group_re.setToolTip(
            "Regex applied to each filename; capture group 1 = the condition/group.\n"
            "Default captures the strain token, e.g. '10A5' from '0001_10A5_009_….png'.\n"
            "Leave blank to treat all images as one group.")
        scope_form.addRow("Group by (regex):", self._group_re)
        layout.addWidget(scope_box)

        # labels
        lbl_box = QGroupBox("Feature labels")
        lbl_layout = QVBoxLayout(lbl_box)
        info = QLabel(
            "Tick one label for clustering/hotspots, or two+ labels to also test "
            "cross-label association (e.g. are Spores near Nanopillars?)."
        )
        info.setWordWrap(True)
        info.setStyleSheet("font-size:11px; color: palette(mid);")
        lbl_layout.addWidget(info)
        self._label_list = QListWidget()
        self._label_list.setMaximumHeight(120)
        lbl_layout.addWidget(self._label_list)
        refresh_btn = QPushButton("Refresh labels")
        refresh_btn.clicked.connect(self._refresh_labels)
        lbl_layout.addWidget(refresh_btn)
        layout.addWidget(lbl_box)

        # parameters
        param_box = QGroupBox("Parameters (nm)")
        pform = QFormLayout(param_box)
        self._eps = QDoubleSpinBox()
        self._eps.setRange(1.0, 1e7); self._eps.setValue(200.0); self._eps.setDecimals(1)
        self._eps.setToolTip("DBSCAN neighbourhood radius — features within this distance group together.")
        pform.addRow("Cluster radius:", self._eps)
        self._min_samples = QSpinBox()
        self._min_samples.setRange(1, 100); self._min_samples.setValue(3)
        self._min_samples.setToolTip("Minimum features within the radius to seed a cluster.")
        pform.addRow("Min cluster size:", self._min_samples)
        self._local_r = QDoubleSpinBox()
        self._local_r.setRange(1.0, 1e7); self._local_r.setValue(300.0); self._local_r.setDecimals(1)
        self._local_r.setToolTip("Radius for per-feature local-crowding count.")
        pform.addRow("Local-density radius:", self._local_r)
        self._bandwidth = QDoubleSpinBox()
        self._bandwidth.setRange(0.0, 1e7); self._bandwidth.setValue(0.0); self._bandwidth.setDecimals(1)
        self._bandwidth.setToolTip("Hotspot (KDE) smoothing. 0 = automatic.")
        pform.addRow("Hotspot bandwidth:", self._bandwidth)
        self._mc = QSpinBox()
        self._mc.setRange(0, 9999); self._mc.setValue(199)
        self._mc.setToolTip("Monte-Carlo CSR simulations for p-values + Ripley envelope.\n"
                            "0 = fast analytic only (no edge correction / envelope).")
        pform.addRow("Monte-Carlo sims:", self._mc)
        layout.addWidget(param_box)

        run_btn = QPushButton("Run Spatial Analysis")
        run_btn.setStyleSheet("background:#1a5fa8;color:white;font-weight:bold;")
        run_btn.clicked.connect(self._run)
        layout.addWidget(run_btn)

        ov_row = QHBoxLayout()
        self._overlay_chk = QCheckBox("Draw clusters + hotspot on the image")
        self._overlay_chk.setChecked(True)
        self._overlay_chk.setToolTip("Overlay the cluster colours and hotspot heatmap on the canvas.")
        clear_ov_btn = QPushButton("Clear overlay")
        clear_ov_btn.setFixedWidth(110)
        clear_ov_btn.clicked.connect(self._clear_canvas_overlay)
        ov_row.addWidget(self._overlay_chk, 1)
        ov_row.addWidget(clear_ov_btn)
        layout.addLayout(ov_row)

        focus_row = QHBoxLayout()
        focus_row.addWidget(QLabel("Focus cluster:"))
        self._focus_spin = QSpinBox()
        self._focus_spin.setRange(-1, -1); self._focus_spin.setValue(-1)
        self._focus_spin.setSpecialValueText("all")     # -1 shows all clusters
        self._focus_spin.setEnabled(False)
        self._focus_spin.setToolTip("Show only one cluster's features on the image (−1 = all).")
        self._focus_spin.valueChanged.connect(self._on_focus_changed)
        focus_row.addWidget(self._focus_spin, 1)
        layout.addLayout(focus_row)

        self._save_btn = QPushButton("Save results (CSV + summary)…")
        self._save_btn.setToolTip("Write a per-feature CSV and a summary text file.")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_results)
        layout.addWidget(self._save_btn)
        self._last = None

        # results
        self._stats = QTextEdit()
        self._stats.setReadOnly(True)
        self._stats.setFontFamily("Monospace")
        self._stats.setMaximumHeight(220)
        layout.addWidget(self._stats)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        # Match ACORN's dark theme (a QSS stylesheet, not a palette, so we can't
        # read it from palette()); #1a1a1a is the in-app matplotlib-on-dark colour.
        self._fig_bg = "#1e1e1e"
        self._fig = Figure(figsize=(5.5, 6.5), facecolor=self._fig_bg)
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setStyleSheet("background-color: transparent;")
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumHeight(360)
        layout.addWidget(self._canvas, 1)
        self._show_placeholder()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self._refresh_labels()

    # ── label discovery ──────────────────────────────────────────────────────────

    def _all_annotations(self):
        scope = self._scope.currentData()
        if scope == "all":
            out = []
            for anns in self._context.all_annotation_states.values():
                out.extend(anns)
            return out
        store = self._context.annotation_store
        return list(store) if store is not None else []

    def _refresh_labels(self) -> None:
        prev = set(self._checked_labels())
        labels = sorted({S.feature_label(a) for a in self._all_annotations()
                         if getattr(a, "type", None) in ("roi", "circle", "rectangle")})
        self._label_list.clear()
        for lbl in labels:
            it = QListWidgetItem(lbl)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if (not prev or lbl in prev)
                             else Qt.CheckState.Unchecked)
            self._label_list.addItem(it)

    def _checked_labels(self) -> list[str]:
        return [self._label_list.item(i).text()
                for i in range(self._label_list.count())
                if self._label_list.item(i).checkState() == Qt.CheckState.Checked]

    # ── run ───────────────────────────────────────────────────────────────────────

    def _image_area_nm2(self) -> tuple[float, float, float]:
        img = self._context.current_image
        px = self._context.current_pixel_size_nm or 1.0
        if img is None or not img.shape:
            return 0.0, 0.0, 0.0
        h, w = img.shape[:2]
        wn, hn = w * px, h * px
        return wn, hn, wn * hn

    def _run(self) -> None:
        labels = set(self._checked_labels())
        if not labels:
            self._stats.setPlainText("Select at least one feature label.")
            return
        if self._scope.currentData() == "all":
            self._run_batch(labels)
            return
        px = self._context.current_pixel_size_nm or 1.0
        wn, hn, area = self._image_area_nm2()
        if area <= 0:
            self._stats.setPlainText("No image loaded.")
            return
        pts_by_label = S.extract_points(self._all_annotations(), px_nm=px, labels=labels)
        keys = list(pts_by_label.keys())
        pooled = (np.vstack([pts_by_label[k] for k in keys]) if keys else np.empty((0, 2)))
        point_labels = np.concatenate([[k] * len(pts_by_label[k]) for k in keys]) if keys \
            else np.array([], dtype=object)
        if len(pooled) < 2:
            self._stats.setPlainText("Need at least 2 features in the selected labels.")
            return

        calibrated = self._context.current_pixel_size_nm not in (0, 1.0)
        unit = "nm" if calibrated else "px"
        lines = [f"Field of view: {wn:.0f} × {hn:.0f} {unit}   "
                 f"({sum(len(p) for p in pts_by_label.values())} features, {unit})",
                 ""]

        mc = self._mc.value()
        nnd = S.nearest_neighbour(pooled, area, wn, hn, n_mc=mc)
        lines += ["── Clustering (all selected features pooled) ──",
                  f"  features:        {nnd.n}",
                  f"  mean NN dist:    {nnd.mean_nnd_nm:.1f} {unit}",
                  f"  expected(random):{nnd.expected_nnd_nm:.1f} {unit}",
                  f"  Clark-Evans R:   {nnd.clark_evans_R:.3f}",
                  f"  verdict:         {nnd.verdict}", ""]

        cl = S.dbscan(pooled, self._eps.value(), self._min_samples.value())
        lines += ["── Proximity clusters (DBSCAN) ──",
                  f"  clusters:        {cl.n_clusters}",
                  f"  cluster sizes:   {cl.cluster_sizes}",
                  f"  isolated (noise):{cl.n_noise}", ""]

        ld = S.local_density(pooled, self._local_r.value())
        if len(ld):
            lines += [f"── Local crowding (within {self._local_r.value():.0f} {unit}) ──",
                      f"  neighbours/feature: mean {ld.mean():.1f}, max {int(ld.max())}", ""]

        # cross-label association (each ordered pair of labels with ≥1 point)
        present = {k: v for k, v in pts_by_label.items() if len(v) > 0}
        if len(present) >= 2:
            lines.append("── Cross-label association ──")
            for a, b in itertools.permutations(present.keys(), 2):
                cr = S.cross_nearest_neighbour(present[a], present[b], area, a, b,
                                               wn, hn, n_mc=mc)
                lines.append(f"  {cr.verdict}")
            lines.append("")

        img = self._context.current_image
        img_name = img.filename if (img is not None and getattr(img, "filename", "")) else "image"
        self._last = {
            "image": img_name, "unit": unit, "px": px, "wn": wn, "hn": hn,
            "labels": point_labels, "points_nm": pooled,
            "cluster": cl.labels, "local_density": ld,
            "nnd": nnd.nnd_nm, "summary": "\n".join(lines),
        }
        self._save_btn.setEnabled(True)
        self._focus_spin.setRange(-1, max(cl.n_clusters - 1, -1))
        self._focus_spin.setValue(-1)
        self._focus_spin.setEnabled(cl.n_clusters > 0)

        self._stats.setPlainText("\n".join(lines))
        self._draw_figures(pts_by_label, pooled, nnd, cl, area, wn, hn, unit)
        if self._overlay_chk.isChecked():
            self._overlay_on_canvas(pooled, cl.labels, px, wn, hn)
        else:
            self._clear_canvas_overlay()
        self._context.set_status(f"Spatial analysis: {nnd.verdict}", timeout_ms=4000)

    # ── canvas overlay ───────────────────────────────────────────────────────────

    def _overlay_on_canvas(self, pooled_nm, labels, px, wn, hn, only_cluster=None) -> None:
        cw = self._context.canvas_widget()
        if cw is None:
            return
        import numpy as np
        pts_px = pooled_nm / px                          # nm → image pixels (aligned with labels)
        labels = np.asarray(labels)
        if only_cluster is not None and only_cluster >= 0:
            keep = labels == only_cluster
            pts_px, labels = pts_px[keep], labels[keep]
            kde = extent = None                          # focus mode: just the cluster points
        else:
            w_px, h_px = wn / px, hn / px
            kde, _ = S.kde_grid(pts_px, w_px, h_px, n_grid=160,
                                bandwidth_nm=(self._bandwidth.value() / px) if self._bandwidth.value() else None)
            extent = (0, w_px, h_px, 0) if kde is not None else None
        cw.show_spatial_overlay(pts_px, labels, kde=kde, kde_extent=extent)

    def _on_focus_changed(self, val: int) -> None:
        if not self._last or not self._overlay_chk.isChecked():
            return
        d = self._last
        if d.get("points_nm") is None or len(d["points_nm"]) == 0:
            return
        self._overlay_on_canvas(d["points_nm"], d["cluster"], d["px"], d["wn"], d["hn"],
                                only_cluster=(val if val >= 0 else None))

    def _clear_canvas_overlay(self) -> None:
        cw = self._context.canvas_widget()
        if cw is not None and hasattr(cw, "clear_spatial_overlay"):
            cw.clear_spatial_overlay()

    # ── export ───────────────────────────────────────────────────────────────────

    def _save_results(self) -> None:
        if not self._last:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save spatial results (base name)",
            f"{self._last['image']}_spatial.csv", "CSV files (*.csv);;All files (*)")
        if not path:
            return
        base = Path(path)
        if base.suffix.lower() == ".csv":
            base = base.with_suffix("")
        feat_path = base.with_name(base.name + "_features.csv")
        summ_path = base.with_name(base.name + "_summary.txt")
        d = self._last
        try:
            with open(feat_path, "w", newline="") as f:
                wtr = csv.writer(f)
                u = d["unit"]
                wtr.writerow(["image", "label", f"x_{u}", f"y_{u}",
                              "cluster_id", "n_neighbors", f"nearest_neighbor_{u}"])
                for i in range(len(d["points_nm"])):
                    x, y = d["points_nm"][i]
                    cid = int(d["cluster"][i])
                    nbr = int(d["local_density"][i]) if i < len(d["local_density"]) else ""
                    nn = round(float(d["nnd"][i]), 3) if i < len(d["nnd"]) else ""
                    wtr.writerow([d["image"], str(d["labels"][i]), round(float(x), 3),
                                  round(float(y), 3),
                                  cid if cid >= 0 else "isolated", nbr, nn])
            with open(summ_path, "w") as f:
                f.write(d["summary"] + "\n")
        except OSError as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return
        self._context.set_status(
            f"Saved {feat_path.name} + {summ_path.name}", timeout_ms=5000)
        self._stats.append(f"\n[saved → {feat_path}  and  {summ_path.name}]")

    def _on_image_loaded(self, *_) -> None:
        self._clear_canvas_overlay()   # stale overlay belongs to the previous image
        self._refresh_labels()

    def run_from_clu(self, labels=None) -> None:
        """Entry point for the CLU assistant: tick labels (or all), run, and
        report a concise real result back to the agent."""
        self._refresh_labels()
        want = {str(x) for x in labels} if labels else None
        for i in range(self._label_list.count()):
            it = self._label_list.item(i)
            checked = (want is None) or (it.text() in want)
            it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._scope.setCurrentIndex(0)   # current image
        self._run()

    def clu_result_text(self) -> str:
        """Concise outcome string for the CLU assistant after run_from_clu()."""
        return (self._stats.toPlainText().strip()
                or "Spatial analysis produced no result.")

    def _condition_for(self, name: str) -> str:
        pat = self._group_re.text().strip()
        if not pat:
            return "all"
        import re
        try:
            m = re.search(pat, name)
        except re.error:
            return "all"
        if not m:
            return "(no match)"
        return m.group(1) if m.groups() else m.group(0)

    def _run_batch(self, labels: set[str]) -> None:
        ctx = self._context
        states = ctx.all_annotation_states
        paths = ctx.image_paths
        rows = ["Per-image spatial summary (selected labels):", ""]
        rows.append(f"{'image':<26} {'cond':>8} {'n':>4} {'R':>6} {'clust':>6}  verdict")
        per_cond: dict[str, list[float]] = {}
        per_image_feats: list = []   # (image, label, x, y, cluster) for export
        for idx in sorted(states.keys()):
            anns = states[idx]
            px = ctx.pixel_size_for_index(idx) or 1.0
            name = paths[idx].name if idx < len(paths) else str(idx)
            cond = self._condition_for(name)
            pts_by = S.extract_points(anns, px_nm=px, labels=labels)
            keys = list(pts_by.keys())
            pooled = np.vstack([pts_by[k] for k in keys]) if keys else np.empty((0, 2))
            if pooled.shape[0] >= 2:
                # Prefer the true field of view; fall back to the point bbox.
                shape = ctx.image_shape_for_index(idx)
                if shape is not None:
                    hn, wn = shape[0] * px, shape[1] * px
                else:
                    lo = pooled.min(0); span = pooled.max(0) - lo
                    wn, hn = float(max(span[0], 1)), float(max(span[1], 1))
                nnd = S.nearest_neighbour(pooled, wn * hn, wn, hn, n_mc=self._mc.value())
                cl = S.dbscan(pooled, self._eps.value(), self._min_samples.value())
                per_cond.setdefault(cond, []).append(nnd.clark_evans_R)
                rows.append(f"{name[:26]:<26} {cond[:8]:>8} {nnd.n:>4} "
                            f"{nnd.clark_evans_R:>6.2f} {cl.n_clusters:>6}  {nnd.verdict}")
            else:
                rows.append(f"{name[:26]:<26} {cond[:8]:>8} {pooled.shape[0]:>4} "
                            f"{'—':>6} {'—':>6}  (too few)")

        # ── condition comparison ──────────────────────────────────────────────
        if len(per_cond) >= 1:
            rows += ["", "── Clustering by condition (Clark-Evans R; R<1 = clustered) ──",
                     f"{'condition':>10} {'images':>7} {'meanR':>7} {'±SE':>7}"]
            for cond in sorted(per_cond):
                vals = np.array(per_cond[cond])
                se = float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
                rows.append(f"{cond[:10]:>10} {len(vals):>7} {vals.mean():>7.3f} {se:>7.3f}")
            groups = [c for c in sorted(per_cond) if len(per_cond[c]) >= 1]
            if len(groups) == 2 and all(len(per_cond[g]) >= 2 for g in groups):
                from scipy.stats import mannwhitneyu
                a, b = per_cond[groups[0]], per_cond[groups[1]]
                try:
                    u, p = mannwhitneyu(a, b, alternative="two-sided")
                    verdict = ("differ significantly" if p < 0.05
                               else "not significantly different")
                    rows += ["", f"  {groups[0]} vs {groups[1]} clustering: "
                                 f"{verdict} (Mann-Whitney p={p:.3f})"]
                except ValueError:
                    pass
            elif len(groups) > 2:
                from scipy.stats import kruskal
                try:
                    h, p = kruskal(*[per_cond[g] for g in groups])
                    rows += ["", f"  Across {len(groups)} conditions: "
                                 f"Kruskal-Wallis p={p:.3f}"]
                except ValueError:
                    pass

        self._stats.setPlainText("\n".join(rows))
        self._last = {"image": "batch", "unit": "nm", "summary": "\n".join(rows),
                      "labels": np.array([]), "points_nm": np.empty((0, 2)),
                      "cluster": np.array([]), "local_density": np.array([]),
                      "nnd": np.array([])}
        self._save_btn.setEnabled(True)
        self._clear_canvas_overlay()
        self._show_placeholder("Per-image summary shown above.\n(No figure in batch mode.)")

    # ── figures ────────────────────────────────────────────────────────────────────

    def _show_placeholder(self, msg=None) -> None:
        """Dark, themed empty state instead of a glaring white canvas."""
        self._fig.clear()
        self._fig.patch.set_facecolor(self._fig_bg)
        ax = self._fig.add_subplot(111)
        ax.set_facecolor(self._fig_bg)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.5, 0.5,
                msg or "Run Spatial Analysis to see cluster, hotspot,\n"
                       "nearest-neighbour and Ripley figures here.",
                ha="center", va="center", color="#888888", fontsize=10)
        self._canvas.draw_idle()

    def _draw_figures(self, pts_by_label, pooled, nnd, cl, area, wn, hn, unit) -> None:
        fig = self._fig
        fig.clear()
        fig.patch.set_facecolor(self._fig_bg)
        ax1 = fig.add_subplot(2, 2, 1)   # cluster scatter
        ax2 = fig.add_subplot(2, 2, 2)   # hotspot KDE
        ax3 = fig.add_subplot(2, 2, 3)   # NND histogram
        ax4 = fig.add_subplot(2, 2, 4)   # Ripley's L

        # cluster scatter
        palette = ["#4878D0", "#EE854A", "#6ACC65", "#D65F5F", "#956CB4",
                   "#8C613C", "#DC7EC0", "#797979", "#D5BB67", "#82C6E2"]
        for c in range(cl.n_clusters):
            m = cl.labels == c
            ax1.scatter(pooled[m, 0], pooled[m, 1], s=14,
                        color=palette[c % len(palette)], edgecolors="none")
        noise = cl.labels == -1
        if noise.any():
            ax1.scatter(pooled[noise, 0], pooled[noise, 1], s=10, color="#bbbbbb",
                        edgecolors="none", label="isolated")
        ax1.set_title(f"{cl.n_clusters} clusters", fontsize=9)
        ax1.set_xlim(0, wn); ax1.set_ylim(hn, 0); ax1.set_aspect("equal")
        ax1.tick_params(labelsize=7)

        # hotspot
        dens, extent = S.kde_grid(pooled, wn, hn, n_grid=160,
                                  bandwidth_nm=self._bandwidth.value() or None)
        if dens is not None:
            ax2.imshow(dens, extent=extent, origin="upper", cmap="inferno", aspect="equal")
            ax2.scatter(pooled[:, 0], pooled[:, 1], s=3, color="white", alpha=0.5)
        ax2.set_title("Hotspot density", fontsize=9)
        ax2.set_xlim(0, wn); ax2.set_ylim(hn, 0); ax2.tick_params(labelsize=7)

        # NND histogram
        if len(nnd.nnd_nm):
            ax3.hist(nnd.nnd_nm, bins=24, color="#4878D0", edgecolor="#1a1a1a", linewidth=0.4)
            ax3.axvline(nnd.expected_nnd_nm, color="#D65F5F", ls="--", lw=1.2, label="random")
            ax3.legend(fontsize=7)
        ax3.set_title("Nearest-neighbour dist", fontsize=9)
        ax3.set_xlabel(f"distance ({unit})", fontsize=8); ax3.tick_params(labelsize=7)

        # Ripley's L with Monte-Carlo CSR envelope
        rmax = min(wn, hn) / 2.0
        radii = np.linspace(rmax / 40, rmax, 40)
        L, lo, hi = S.ripleys_l(pooled, area, radii, wn, hn, n_mc=self._mc.value())
        if L is not None:
            if lo is not None:
                ax4.fill_between(radii, lo, hi, color="#888888", alpha=0.25,
                                 label="95% CSR envelope")
                ax4.legend(fontsize=7)
            ax4.plot(radii, L, color="#6ACC65", lw=1.5)
            ax4.axhline(0, color="#888888", lw=0.8)
        ax4.set_title("Ripley's L − r  (>0 clustered)", fontsize=9)
        ax4.set_xlabel(f"radius ({unit})", fontsize=8); ax4.tick_params(labelsize=7)

        # Dark theme: dark plot panels + light text, to match the rest of ACORN.
        for ax in fig.axes:
            ax.set_facecolor(self._fig_bg)
            ax.tick_params(colors="#cccccc", labelsize=7)
            ax.title.set_color("#e6e6e6")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            for s in ax.spines.values():
                s.set_edgecolor("#555555")
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(self._fig_bg)
                leg.get_frame().set_edgecolor("#555555")
                for t in leg.get_texts():
                    t.set_color("#cccccc")

        fig.tight_layout()
        self._canvas.draw_idle()
