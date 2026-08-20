"""
Multi-frame images: averaging, motion correction, dose weighting, and the two
dialogs that show the result.

Movies are the one part of ACORN where a single file is really a stack, and the
handling - frame ranges, drift plots, dose series, compression - is self
contained enough to keep out of the window class.
"""
from __future__ import annotations

from acorn.gui import buttons

import numpy as np

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QHBoxLayout, QLabel,
    QPushButton, QSpinBox, QVBoxLayout, QWidget,
)

from acorn.gui.threads import FrameProcessThread


class MotionPlotDialog(QDialog):
    """Drift trajectory and per-frame displacement from motion correction."""

    def __init__(
        self,
        shifts: "np.ndarray",       # (n_frames, 2) total (dy, dx) applied shifts
        pixel_size_nm: float = 1.0,
        start_frame: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Motion Correction — Drift Analysis")
        self.setMinimumSize(960, 460)

        import numpy as np
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        import matplotlib.cm as cm

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        n = len(shifts)
        px_A  = pixel_size_nm * 10.0
        calibrated = pixel_size_nm > 0 and pixel_size_nm != 1.0
        unit  = "Å" if calibrated else "px"
        scale = px_A if calibrated else 1.0

        # Sample drift = where the sample WAS before correction (negative of applied shift)
        drift_y =  shifts[:, 0] * scale   # row direction → Y
        drift_x =  shifts[:, 1] * scale   # col direction → X
        magnitude = np.sqrt(drift_y**2 + drift_x**2)
        frame_nums = np.arange(start_frame, start_frame + n)

        fig = Figure(figsize=(11.5, 4.2), facecolor="#1a1a1a")
        canvas = FigureCanvasQTAgg(fig)
        layout.addWidget(canvas, 1)

        cmap = cm.plasma

        # ── Left: drift trajectory ─────────────────────────────────────────
        ax1 = fig.add_subplot(121)
        ax1.set_facecolor("#1e2a30")

        ax1.plot(drift_x, drift_y, color="#333", linewidth=0.9, zorder=1)
        sc = ax1.scatter(
            drift_x, drift_y,
            c=frame_nums, cmap=cmap, s=22, zorder=2, edgecolors="none",
        )
        ax1.scatter(drift_x[0],  drift_y[0],  s=70, color="#4dbb78",
                    zorder=3, marker="o", label=f"start (frame {start_frame})")
        ax1.scatter(drift_x[-1], drift_y[-1], s=70, color="#c0392b",
                    zorder=3, marker="s", label=f"end (frame {start_frame + n - 1})")

        cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
        cbar.set_label("Frame", color="#e0e0e0", fontsize=9)
        cbar.ax.yaxis.set_tick_params(color="#888", labelcolor="#e0e0e0")

        ax1.axhline(0, color="#444", linewidth=0.5, linestyle="--")
        ax1.axvline(0, color="#444", linewidth=0.5, linestyle="--")
        ax1.set_xlabel(f"X drift ({unit})", color="#e0e0e0")
        ax1.set_ylabel(f"Y drift ({unit})", color="#e0e0e0")
        ax1.set_title("Drift trajectory", color="#4dbb78", fontweight="bold")
        ax1.tick_params(colors="#888", labelcolor="#e0e0e0")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333")
        ax1.legend(fontsize=8, facecolor="#1e2a30", edgecolor="#444",
                   labelcolor="#e0e0e0", loc="best")

        # ── Right: per-frame displacement ──────────────────────────────────
        ax2 = fig.add_subplot(122)
        ax2.set_facecolor("#1e2a30")

        bar_colors = [cmap(i / max(n - 1, 1)) for i in range(n)]
        ax2.bar(frame_nums, magnitude, color=bar_colors, width=max(0.8, n * 0.006))
        ax2.set_xlabel("Frame", color="#e0e0e0")
        ax2.set_ylabel(f"Displacement ({unit})", color="#e0e0e0")
        ax2.set_title("Per-frame displacement", color="#4dbb78", fontweight="bold")
        ax2.tick_params(colors="#888", labelcolor="#e0e0e0")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333")

        # ── Summary footer ─────────────────────────────────────────────────
        total_path   = float(np.sum(np.sqrt(np.diff(drift_x)**2 + np.diff(drift_y)**2)))
        max_disp     = float(magnitude.max())
        mean_disp    = float(magnitude.mean())
        info = (
            f"Frames {start_frame}–{start_frame + n - 1}  |  "
            f"Max displacement: {max_disp:.2f} {unit}  |  "
            f"Mean: {mean_disp:.2f} {unit}  |  "
            f"Total drift path: {total_path:.2f} {unit}"
        )
        fig.text(0.5, 0.005, info, ha="center", color="#888", fontsize=9)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        canvas.draw()

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)


class DoseSeriesDialog(QDialog):
    """
    Splits the movie into equal dose bins and displays per-bin averages plus
    difference images to visualise dose-dependent structural changes.
    """

    def __init__(
        self,
        frames: "np.ndarray",       # (n, H, W) float32
        pixel_size_nm: float = 1.0,
        dose_per_frame: float = 1.0,
        start_frame: int = 1,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dose Series — Frame Comparison")
        self.setMinimumSize(1100, 560)

        self._frames        = frames
        self._px_nm         = pixel_size_nm
        self._start_frame   = start_frame

        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        self._Figure        = Figure
        self._FigureCanvas  = FigureCanvasQTAgg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── controls ──────────────────────────────────────────────────────
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        ctrl.addWidget(QLabel("Bins:"))
        self._n_bins_spin = QSpinBox()
        self._n_bins_spin.setRange(2, min(20, len(frames)))
        self._n_bins_spin.setValue(min(4, len(frames)))
        self._n_bins_spin.setFixedWidth(56)
        self._n_bins_spin.setToolTip("Number of equal-dose bins to split the movie into")
        ctrl.addWidget(self._n_bins_spin)

        ctrl.addWidget(QLabel("Dose/frame (e/Å²):"))
        self._dose_spin = QDoubleSpinBox()
        self._dose_spin.setRange(0.0, 200.0)
        self._dose_spin.setValue(max(0.0, dose_per_frame))
        self._dose_spin.setDecimals(2)
        self._dose_spin.setFixedWidth(72)
        self._dose_spin.setToolTip(
            "Electron dose per frame — used to label cumulative dose on each bin. "
            "Set to 0 to show frame numbers only."
        )
        ctrl.addWidget(self._dose_spin)

        self._diff_chk = QCheckBox("Show difference from first bin")
        self._diff_chk.setChecked(True)
        self._diff_chk.setToolTip(
            "Show a second row with (bin N) − (bin 1) difference images.\n"
            "Blue = signal decreased; red = signal increased with dose."
        )
        ctrl.addWidget(self._diff_chk)

        ctrl.addStretch()

        update_btn = QPushButton("Update")
        buttons.primary(update_btn)
        update_btn.setFixedWidth(70)
        update_btn.clicked.connect(self._update_figure)
        ctrl.addWidget(update_btn)
        layout.addLayout(ctrl)

        # ── figure ────────────────────────────────────────────────────────
        self._fig    = self._Figure(facecolor="#1a1a1a")
        self._canvas = self._FigureCanvas(self._fig)
        layout.addWidget(self._canvas, 1)

        # ── close button ──────────────────────────────────────────────────
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(70)
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        # Auto-update when controls change
        self._n_bins_spin.valueChanged.connect(self._update_figure)
        self._diff_chk.stateChanged.connect(self._update_figure)
        self._update_figure()

    def _update_figure(self) -> None:
        import numpy as np
        from acorn.core.frame_processor import dose_series

        n_bins   = self._n_bins_spin.value()
        dose_pf  = self._dose_spin.value()
        show_diff = self._diff_chk.isChecked()
        n_total  = len(self._frames)

        averages, ranges = dose_series(self._frames, n_bins)

        # Global contrast for the raw-average row (consistent across bins)
        all_vals = np.concatenate([a.ravel() for a in averages])
        vmin, vmax = np.percentile(all_vals, [0.5, 99.5])

        # Difference row: symmetric ± range across all non-reference diffs
        ref = averages[0]
        diffs = [a - ref for a in averages]
        if n_bins > 1:
            max_abs = max(float(np.abs(d).max()) for d in diffs[1:])
            max_abs = max(max_abs, 1e-6)
        else:
            max_abs = 1.0

        n_rows = 2 if show_diff else 1
        self._fig.clear()

        calibrated = self._px_nm > 0 and self._px_nm != 1.0

        for col, (avg, (s, e)) in enumerate(zip(averages, ranges)):
            frame_s = s + self._start_frame
            frame_e = e + self._start_frame - 1

            # cumulative dose label
            if dose_pf > 0:
                cum_start = s       * dose_pf
                cum_end   = (e - 1) * dose_pf
                dose_label = f"\n{cum_start:.1f}–{cum_end:.1f} e/Å²"
            else:
                dose_label = ""

            title = (
                f"Bin {col + 1}{'  (ref)' if col == 0 else ''}\n"
                f"Frames {frame_s}–{frame_e}{dose_label}"
            )

            ax = self._fig.add_subplot(n_rows, n_bins, col + 1)
            ax.imshow(avg, cmap="gray", vmin=vmin, vmax=vmax,
                      aspect="equal", interpolation="nearest")
            ax.set_facecolor("#1e2a30")
            ax.set_title(title, color="#4dbb78" if col == 0 else "#e0e0e0",
                         fontsize=8, pad=3)
            ax.axis("off")

            if show_diff:
                ax2 = self._fig.add_subplot(n_rows, n_bins, n_bins + col + 1)
                ax2.set_facecolor("#1e2a30")
                if col == 0:
                    ax2.text(
                        0.5, 0.5, "reference\n(no diff)",
                        ha="center", va="center", color="#666",
                        fontsize=9, transform=ax2.transAxes,
                    )
                    ax2.axis("off")
                else:
                    diff = diffs[col]
                    im = ax2.imshow(
                        diff, cmap="RdBu_r",
                        vmin=-max_abs, vmax=max_abs,
                        aspect="equal", interpolation="nearest",
                    )
                    self._fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.02,
                                       format="%.2g").ax.yaxis.set_tick_params(
                        color="#888", labelcolor="#aaa", labelsize=7)
                    ax2.set_title(
                        f"Δ bin {col + 1} − bin 1",
                        color="#c0392b", fontsize=8, pad=3,
                    )
                    ax2.axis("off")

        # Row labels on the left edge
        if n_rows >= 1:
            self._fig.text(0.005, 0.75 if show_diff else 0.5,
                           "Dose average", va="center", rotation=90,
                           color="#4dbb78", fontsize=9)
        if show_diff:
            self._fig.text(0.005, 0.25,
                           "Δ vs bin 1", va="center", rotation=90,
                           color="#c0392b", fontsize=9)

        # Footer
        frame_range = f"Frames {self._start_frame}–{self._start_frame + n_total - 1}"
        dose_total  = f"  |  Total dose: {n_total * dose_pf:.1f} e/Å²" if dose_pf > 0 else ""
        self._fig.text(
            0.5, 0.003,
            f"{frame_range}  |  {n_bins} bins  ({n_total // n_bins} frames/bin approx){dose_total}",
            ha="center", color="#888", fontsize=8,
        )

        self._fig.tight_layout(rect=[0.015, 0.03, 1, 1])
        self._canvas.draw()


class MovieControllerMixin:
    """Mixed into MainWindow — these methods use its state directly."""

    def _build_movie_bar(self) -> QWidget:
        bar = QWidget()
        bar.setStyleSheet("background:#1a2535;border-top:1px solid #2a3a4a;")
        bar.hide()

        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(10)

        self._movie_label = QLabel("Movie: 0 frames")
        self._movie_label.setStyleSheet("color:#aaaaaa;font-size:11px;")
        row.addWidget(self._movie_label)

        row.addWidget(QLabel("Method:"))
        self._movie_method_combo = QComboBox()
        self._movie_method_combo.addItem("Mean average",      "mean")
        self._movie_method_combo.addItem("Motion-corrected",  "motion_corrected")
        self._movie_method_combo.addItem("Dose-weighted",     "dose_weighted")
        self._movie_method_combo.setFixedWidth(160)
        self._movie_method_combo.currentIndexChanged.connect(self._on_movie_method_ui_changed)
        row.addWidget(self._movie_method_combo)

        self._movie_dose_label = QLabel("Dose e/A²/frame:")
        self._movie_dose_label.setStyleSheet("font-size:11px;")
        self._movie_dose_label.hide()
        row.addWidget(self._movie_dose_label)

        self._movie_dose_spin = QDoubleSpinBox()
        self._movie_dose_spin.setRange(0.01, 200.0)
        self._movie_dose_spin.setValue(1.0)
        self._movie_dose_spin.setDecimals(2)
        self._movie_dose_spin.setFixedWidth(72)
        self._movie_dose_spin.hide()
        row.addWidget(self._movie_dose_spin)

        row.addWidget(QLabel("Frames:"))
        self._movie_start_spin = QSpinBox()
        self._movie_start_spin.setRange(1, 1)
        self._movie_start_spin.setValue(1)
        self._movie_start_spin.setFixedWidth(64)
        self._movie_start_spin.setToolTip("First frame to include in compression (1 = frame 1)")
        row.addWidget(self._movie_start_spin)
        row.addWidget(QLabel("to"))
        self._movie_end_spin = QSpinBox()
        self._movie_end_spin.setRange(1, 1)
        self._movie_end_spin.setValue(1)
        self._movie_end_spin.setFixedWidth(64)
        self._movie_end_spin.setToolTip("Last frame to include in compression")
        row.addWidget(self._movie_end_spin)

        apply_btn = QPushButton("Apply")
        apply_btn.setFixedWidth(56)
        buttons.primary(apply_btn)
        apply_btn.clicked.connect(self._on_movie_apply_clicked)
        row.addWidget(apply_btn)

        row.addSpacing(16)
        row.addWidget(QLabel("View frame:"))
        self._movie_frame_spin = QSpinBox()
        self._movie_frame_spin.setRange(0, 0)
        self._movie_frame_spin.setSpecialValueText("avg")
        self._movie_frame_spin.setFixedWidth(72)
        self._movie_frame_spin.setToolTip("0 = averaged view  |  1..N = jump to individual frame")
        self._movie_frame_spin.valueChanged.connect(self._on_movie_frame_changed)
        row.addWidget(self._movie_frame_spin)

        row.addSpacing(16)
        self._motion_plot_btn = QPushButton("Motion plot")
        self._motion_plot_btn.setFixedWidth(96)
        self._motion_plot_btn.setStyleSheet(
            "background:#1a5fa8;color:white;font-weight:bold;"
        )
        self._motion_plot_btn.setToolTip(
            "Show drift trajectory and per-frame displacement from the last motion correction"
        )
        self._motion_plot_btn.clicked.connect(self._on_motion_plot_clicked)
        self._motion_plot_btn.hide()
        row.addWidget(self._motion_plot_btn)

        self._dose_series_btn = QPushButton("Dose series")
        self._dose_series_btn.setFixedWidth(96)
        self._dose_series_btn.setStyleSheet(
            "background:#5a3a8a;color:white;font-weight:bold;"
        )
        self._dose_series_btn.setToolTip(
            "Split the movie into equal-dose bins and compare averaged images "
            "to visualise dose-dependent structural changes"
        )
        self._dose_series_btn.clicked.connect(self._on_dose_series_clicked)
        row.addWidget(self._dose_series_btn)

        row.addStretch()
        return bar
    def _update_movie_bar(self, img) -> None:
        if img is None or not img.is_movie:
            self._movie_bar.hide()
            return
        n = img.n_frames
        self._movie_label.setText(f"Movie: {n} frames")
        for spin in (self._movie_start_spin, self._movie_end_spin, self._movie_frame_spin):
            spin.blockSignals(True)
        self._movie_start_spin.setRange(1, n)
        self._movie_start_spin.setValue(1)
        self._movie_end_spin.setRange(1, n)
        self._movie_end_spin.setValue(n)
        self._movie_frame_spin.setRange(0, n)
        self._movie_frame_spin.setValue(0)
        for spin in (self._movie_start_spin, self._movie_end_spin, self._movie_frame_spin):
            spin.blockSignals(False)
        self._last_motion_shifts = None
        self._motion_plot_btn.hide()
        self._movie_bar.show()
        self._statusbar.showMessage(
            f"Movie loaded: {n} frames averaged for display. "
            "Use the bar below to change method or step through frames. "
            "Ask CLU to fix contrast if the image is hard to see.",
            8000,
        )
    def _on_movie_method_ui_changed(self) -> None:
        is_dose = self._movie_method_combo.currentData() == "dose_weighted"
        self._movie_dose_label.setVisible(is_dose)
        self._movie_dose_spin.setVisible(is_dose)
    def _on_movie_frame_changed(self, value: int) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or not img.is_movie:
            return
        if value == 0:
            img.raw = img._frames.mean(axis=0)
        else:
            img.raw = img.get_frame(value - 1)
        self._pending_contrast = self._contrast_panel.params()
        self._contrast_timer.start()
    def _on_movie_apply_clicked(self) -> None:
        method     = self._movie_method_combo.currentData()
        dose       = self._movie_dose_spin.value()
        start_frame = self._movie_start_spin.value()
        end_frame   = self._movie_end_spin.value()
        self._compress_frames(method, dose, start_frame, end_frame)
    def _compress_frames(
        self,
        method: str = "mean",
        dose_per_frame: float = 1.0,
        start_frame: int = 1,
        end_frame: int = 0,   # 0 = use all
    ) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or not img.is_movie:
            self._statusbar.showMessage("No movie loaded — open a multi-frame file first.")
            return

        n = img.n_frames
        s = max(1, start_frame) - 1              # convert 1-indexed to 0-indexed
        e = (n if end_frame <= 0 else min(end_frame, n))  # inclusive end
        if s >= e:
            self._statusbar.showMessage("Invalid frame range — start must be < end.")
            return

        frames_slice = img._frames[s:e]
        n_used = e - s

        method_labels = {
            "mean": "mean average",
            "motion_corrected": "motion correction",
            "dose_weighted": "dose-weighted average",
        }
        range_label = f"frames {s+1}–{e}" if (s > 0 or e < n) else f"all {n} frames"
        self._statusbar.showMessage(
            f"Processing {n_used} frames ({range_label}, {method_labels.get(method, method)})…"
        )

        # Update UI to match params
        idx = self._movie_method_combo.findData(method)
        if idx >= 0:
            self._movie_method_combo.blockSignals(True)
            self._movie_method_combo.setCurrentIndex(idx)
            self._movie_method_combo.blockSignals(False)
        self._movie_dose_spin.setValue(dose_per_frame)
        for spin, val in ((self._movie_start_spin, s + 1), (self._movie_end_spin, e)):
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self._last_motion_start_frame = s + 1  # remember for the plot x-axis
        self._frame_proc_thread = FrameProcessThread(
            frames_slice, method, dose_per_frame, img.pixel_size, parent=self
        )
        self._frame_proc_thread.finished.connect(self._on_frame_proc_done)
        self._frame_proc_thread.shifts_available.connect(self._on_motion_shifts_ready)
        self._frame_proc_thread.error.connect(self._on_frame_proc_error)
        self._frame_proc_thread.start()
    def _on_frame_proc_done(self, result) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is not None:
            img.raw = result
            self._movie_frame_spin.blockSignals(True)
            self._movie_frame_spin.setValue(0)
            self._movie_frame_spin.blockSignals(False)
            self._pending_contrast = self._contrast_panel.params()
            self._contrast_timer.start()
        self._statusbar.showMessage("Frame processing complete.", 3000)
    def _on_frame_proc_error(self, msg: str) -> None:
        self._statusbar.showMessage(f"Frame processing failed: {msg}", 5000)
    def _on_motion_shifts_ready(self, shifts) -> None:
        self._last_motion_shifts = shifts
        self._motion_plot_btn.show()
        img = self._canvas_widget.canvas.dm4
        px_nm = img.pixel_size if img else 1.0
        px_A  = px_nm * 10.0
        import numpy as np
        mag = np.sqrt((shifts**2).sum(axis=1))
        max_drift_A = float(mag.max()) * px_A
        unit = "Å" if (px_nm > 0 and px_nm != 1.0) else "px"
        val  = max_drift_A if unit == "Å" else float(mag.max())
        self._statusbar.showMessage(
            f"Motion correction complete. Max drift: {val:.1f} {unit}. "
            "Click 'Motion plot' to view the drift trajectory.",
            6000,
        )
    def _on_motion_plot_clicked(self) -> None:
        if self._last_motion_shifts is None:
            return
        img = self._canvas_widget.canvas.dm4
        px_nm = img.pixel_size if img else 1.0
        dlg = MotionPlotDialog(
            self._last_motion_shifts,
            pixel_size_nm=px_nm,
            start_frame=self._last_motion_start_frame,
            parent=self,
        )
        dlg.exec()
    def _on_dose_series_clicked(self) -> None:
        img = self._canvas_widget.canvas.dm4
        if img is None or not img.is_movie:
            return
        s = max(0, self._movie_start_spin.value() - 1)
        e = min(img.n_frames, self._movie_end_spin.value())
        frames_slice = img._frames[s:e]
        if len(frames_slice) < 2:
            self._statusbar.showMessage("Need at least 2 frames for dose series.", 3000)
            return
        dlg = DoseSeriesDialog(
            frames_slice,
            pixel_size_nm=img.pixel_size,
            dose_per_frame=self._movie_dose_spin.value(),
            start_frame=s + 1,
            parent=self,
        )
        dlg.exec()
