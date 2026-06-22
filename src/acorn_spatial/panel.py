"""Spatial analysis panel — clustering, hotspots, nearest-neighbour, and
cross-label association of detected features."""

from __future__ import annotations

import itertools

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QTextEdit, QVBoxLayout, QWidget,
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
        self._scope.addItem("All images (per-image summary)", "all")
        scope_form.addRow("Analyze:", self._scope)
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

        # results
        self._stats = QTextEdit()
        self._stats.setReadOnly(True)
        self._stats.setFontFamily("Monospace")
        self._stats.setMaximumHeight(220)
        layout.addWidget(self._stats)

        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        self._fig = Figure(figsize=(5.5, 6.5), facecolor="none")
        self._canvas = FigureCanvasQTAgg(self._fig)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._canvas.setMinimumHeight(420)
        layout.addWidget(self._canvas, 1)

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
        pooled = (np.vstack(list(pts_by_label.values()))
                  if pts_by_label else np.empty((0, 2)))
        if len(pooled) < 2:
            self._stats.setPlainText("Need at least 2 features in the selected labels.")
            return

        calibrated = self._context.current_pixel_size_nm not in (0, 1.0)
        unit = "nm" if calibrated else "px"
        lines = [f"Field of view: {wn:.0f} × {hn:.0f} {unit}   "
                 f"({sum(len(p) for p in pts_by_label.values())} features, {unit})",
                 ""]

        nnd = S.nearest_neighbour(pooled, area)
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
                cr = S.cross_nearest_neighbour(present[a], present[b], area, a, b)
                lines.append(f"  {cr.verdict}")
            lines.append("")

        self._stats.setPlainText("\n".join(lines))
        self._draw_figures(pts_by_label, pooled, nnd, cl, area, wn, hn, unit)
        if self._overlay_chk.isChecked():
            self._overlay_on_canvas(pooled, cl.labels, px, wn, hn)
        else:
            self._clear_canvas_overlay()
        self._context.set_status(f"Spatial analysis: {nnd.verdict}", timeout_ms=4000)

    # ── canvas overlay ───────────────────────────────────────────────────────────

    def _overlay_on_canvas(self, pooled_nm, labels, px, wn, hn) -> None:
        cw = self._context.canvas_widget()
        if cw is None:
            return
        import numpy as np
        pts_px = pooled_nm / px                          # nm → image pixels (aligned with labels)
        w_px, h_px = wn / px, hn / px
        kde, _ = S.kde_grid(pts_px, w_px, h_px, n_grid=160,
                            bandwidth_nm=(self._bandwidth.value() / px) if self._bandwidth.value() else None)
        extent = (0, w_px, h_px, 0) if kde is not None else None
        cw.show_spatial_overlay(pts_px, labels, kde=kde, kde_extent=extent)

    def _clear_canvas_overlay(self) -> None:
        cw = self._context.canvas_widget()
        if cw is not None and hasattr(cw, "clear_spatial_overlay"):
            cw.clear_spatial_overlay()

    def _on_image_loaded(self, *_) -> None:
        self._clear_canvas_overlay()   # stale overlay belongs to the previous image
        self._refresh_labels()

    def _run_batch(self, labels: set[str]) -> None:
        ctx = self._context
        states = ctx.all_annotation_states
        paths = ctx.image_paths
        rows = ["Per-image spatial summary (selected labels):", ""]
        rows.append(f"{'image':<28} {'n':>4} {'meanNND':>9} {'R':>6} {'clusters':>8}  verdict")
        for idx in sorted(states.keys()):
            anns = states[idx]
            px = ctx.pixel_size_for_index(idx) or 1.0
            name = paths[idx].name if idx < len(paths) else str(idx)
            pts_by = S.extract_points(anns, px_nm=px, labels=labels)
            pooled = np.vstack(list(pts_by.values())) if pts_by else np.empty((0, 2))
            img = ctx.current_image
            # area per image: use that image if it's current, else approximate from points bbox
            if pooled.shape[0] >= 2:
                span = pooled.max(0) - pooled.min(0)
                area = float(max(span[0], 1) * max(span[1], 1))
                nnd = S.nearest_neighbour(pooled, area)
                cl = S.dbscan(pooled, self._eps.value(), self._min_samples.value())
                rows.append(f"{name[:28]:<28} {nnd.n:>4} {nnd.mean_nnd_nm:>9.1f} "
                            f"{nnd.clark_evans_R:>6.2f} {cl.n_clusters:>8}  {nnd.verdict}")
            else:
                rows.append(f"{name[:28]:<28} {pooled.shape[0]:>4} {'—':>9} {'—':>6} {'—':>8}  (too few)")
        self._stats.setPlainText("\n".join(rows))
        self._clear_canvas_overlay()
        self._fig.clear(); self._canvas.draw_idle()

    # ── figures ────────────────────────────────────────────────────────────────────

    def _draw_figures(self, pts_by_label, pooled, nnd, cl, area, wn, hn, unit) -> None:
        fig = self._fig
        fig.clear()
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

        # Ripley's L
        rmax = min(wn, hn) / 2.0
        radii = np.linspace(rmax / 40, rmax, 40)
        L = S.ripleys_l(pooled, area, radii)
        if L is not None:
            ax4.plot(radii, L, color="#6ACC65", lw=1.5)
            ax4.axhline(0, color="#888888", lw=0.8)
        ax4.set_title("Ripley's L − r  (>0 clustered)", fontsize=9)
        ax4.set_xlabel(f"radius ({unit})", fontsize=8); ax4.tick_params(labelsize=7)

        fig.tight_layout()
        self._canvas.draw_idle()
