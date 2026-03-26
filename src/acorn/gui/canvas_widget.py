"""Canvas widget — embeds CryoCanvas Figure into PyQt6."""

from __future__ import annotations

import math
from typing import Optional

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvasQtAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT

from acorn.render.canvas import CryoCanvas


# Tools that use click-drag instead of click-click
_DRAG_TOOLS = frozenset({"line", "arrow", "circle", "rectangle", "freehand"})

# All tools that change the cursor to a crosshair
_ANNOTATION_TOOLS = _DRAG_TOOLS | frozenset({
    "text", "scalebar", "distance", "line_profile", "angle", "roi",
})

_PREVIEW_COLOR = "#FFFF00"
_PREVIEW_LW    = 1.5
_PREVIEW_ALPHA = 0.75


class CanvasWidget(QWidget):
    """
    Wraps a CryoCanvas Figure inside a Qt widget.

    Exposes:
    - canvas               : the underlying CryoCanvas
    - click_event          : pyqtSignal emitted for click-only tools
    - drag_commit          : pyqtSignal emitted when a drag annotation is finished
    - freehand_commit      : pyqtSignal emitted when a freehand stroke is finished
    - prev_requested / next_requested : navigation signals
    """

    click_event               = pyqtSignal(float, float, int)
    # tool, x1, y1, x2, y2, shift_held
    drag_commit               = pyqtSignal(str, float, float, float, float, bool)
    freehand_commit           = pyqtSignal(object)   # list of (x, y) tuples
    sam_box_commit            = pyqtSignal(float, float, float, float)  # drag-drawn SAM box
    flat_region_picked        = pyqtSignal(float, float, float, float)  # one-shot rect pick
    prev_requested            = pyqtSignal()
    next_requested            = pyqtSignal()
    annotation_selected       = pyqtSignal(object)   # AnyAnnotation or None
    annotation_delete_requested = pyqtSignal(object) # AnyAnnotation

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # CryoCanvas + Qt embedding
        self.canvas = CryoCanvas(figsize=(8, 8))
        self._mpl_canvas = FigureCanvasQtAgg(self.canvas.fig)
        self._mpl_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        # Allow the canvas to receive keyboard events for Shift detection
        self._mpl_canvas.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        layout.addWidget(self._mpl_canvas, 1)

        # Navigation toolbar (matplotlib built-in: zoom, pan, home, save)
        self._toolbar = NavigationToolbar2QT(self._mpl_canvas, self)
        layout.addWidget(self._toolbar)

        # Prev / Next navigation
        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◄ Prev")
        self._prev_btn.setFixedWidth(90)
        self._prev_btn.clicked.connect(self.prev_requested)
        self._nav_label = QLabel("1 / 1")
        self._nav_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._next_btn = QPushButton("Next ►")
        self._next_btn.setFixedWidth(90)
        self._next_btn.clicked.connect(self.next_requested)
        nav_row.addStretch()
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._nav_label)
        nav_row.addWidget(self._next_btn)
        nav_row.addStretch()
        layout.addLayout(nav_row)

        # Status bar (pixel coords + intensity)
        self._status = QLabel("x=— y=— | intensity=—")
        self._status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._status)

        # ── drag / freehand state ─────────────────────────────────────────────
        self._tool: str = "none"
        self._drag_active: bool = False
        self._drag_start: Optional[tuple[float, float]] = None
        self._shift_held: bool = False
        self._preview_artists: list = []       # temporary matplotlib artists
        self._freehand_pts: list[tuple[float, float]] = []

        # ── rubber-band state (click-based tools) ─────────────────────────────
        self._rubber_band_pts: list[tuple[float, float]] = []
        self._rubber_band_artists: list = []

        # ── SAM box drag state ────────────────────────────────────────────────
        self._sam_press: Optional[tuple[float, float]] = None
        self._sam_box_anchor: Optional[tuple[float, float]] = None
        self._sam_box_artists: list = []

        # ── flat-region pick (one-shot rect for SEM calibration) ──────────────
        self._flat_pick_active: bool = False
        self._flat_pick_press: Optional[tuple[float, float]] = None

        # ── persistent region overlays ────────────────────────────────────────
        self._exclude_artists: list = []
        self._crop_artists: list = []

        # ── annotation selection / move / resize state ────────────────────────
        self._moving_ann = None
        self._move_start: Optional[tuple[float, float]] = None
        self._resizing_ann = None
        self._resize_handle: Optional[str] = None

        # ── connect matplotlib events ─────────────────────────────────────────
        self._mpl_canvas.mpl_connect("button_press_event",   self._on_press)
        self._mpl_canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self._mpl_canvas.mpl_connect("button_release_event", self._on_release)
        self._mpl_canvas.mpl_connect("key_press_event",      self._on_key_press)
        self._mpl_canvas.mpl_connect("key_release_event",    self._on_key_release)
        self._mpl_canvas.mpl_connect("scroll_event",         self._on_scroll)

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def current_tool(self) -> str:
        return self._tool

    def set_tool(self, tool: str) -> None:
        """Called by main_window whenever the active annotation tool changes."""
        self._tool = tool
        self._cancel_drag()
        self.clear_rubber_band()
        # Deactivate matplotlib toolbar zoom/pan so annotation clicks go through
        if tool != "none":
            from matplotlib.backend_bases import _Mode
            if self._toolbar.mode is _Mode.ZOOM:
                self._toolbar.zoom()
            elif self._toolbar.mode is _Mode.PAN:
                self._toolbar.pan()
        self._sync_cursor()

    def _sync_cursor(self) -> None:
        """Assert the correct cursor for the current tool.

        Called from set_tool and from _on_motion so matplotlib's internal
        set_cursor() calls cannot permanently override our choice.
        """
        if self._tool in _ANNOTATION_TOOLS or self._tool == "sam":
            self._mpl_canvas.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._mpl_canvas.unsetCursor()

    def set_rubber_band_pts(self, pts: list) -> None:
        """Set confirmed anchor points for click-based tool rubber-band preview."""
        self._rubber_band_pts = list(pts)

    def add_sam_point_marker(self, x: float, y: float, positive: bool, label: str = ""):
        """Draw a shaped marker at (x, y) for a SAM point prompt.  Returns a list of artists."""
        color = "#44ff77" if positive else "#ff4444"
        marker = "P" if positive else "X"   # filled + vs filled x
        dot, = self.canvas.ax.plot(
            [x], [y], marker,
            color=color, markeredgecolor="#ffffff", markeredgewidth=1.5,
            markersize=13, zorder=20,
        )
        artists = [dot]
        if label:
            txt = self.canvas.ax.text(
                x + 8, y - 8, label,
                color=color, fontsize=8, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#000000", alpha=0.6, edgecolor="none"),
                zorder=21,
            )
            artists.append(txt)
        for art in artists:
            self.canvas._overlay_artists.append(art)
        self.canvas.blit_annotations()
        return artists

    def remove_artist(self, artist) -> None:
        """Remove a single matplotlib artist (and its overlay registration) and redraw."""
        try:
            artist.remove()
        except (ValueError, AttributeError):
            pass
        try:
            self.canvas._overlay_artists.remove(artist)
        except ValueError:
            pass
        self.canvas.blit_annotations()

    def clear_rubber_band(self) -> None:
        """Remove rubber-band preview and clear anchors."""
        self._rubber_band_pts.clear()
        self._clear_rubber_band_artists()

    def start_flat_region_pick(self) -> None:
        """Activate one-shot flat-region rectangle pick mode.

        The next drag on the canvas emits flat_region_picked(x0, y0, x1, y1)
        and automatically cancels the mode.  Any ongoing SAM state is cleared.
        """
        self._flat_pick_active = True
        self._flat_pick_press  = None
        self.clear_sam_box_anchor()
        self._mpl_canvas.setCursor(
            Qt.CursorShape.CrossCursor
        )

    def cancel_flat_region_pick(self) -> None:
        """Cancel flat-region pick mode without emitting a signal."""
        self._flat_pick_active = False
        self._flat_pick_press  = None
        self._sync_cursor()

    def set_sam_box_anchor(self, x: float, y: float) -> None:
        """Record first corner of a SAM box and start showing a live preview rectangle."""
        self._sam_box_anchor = (x, y)
        self._clear_sam_box_artists()

    def clear_sam_box_anchor(self) -> None:
        """Cancel the SAM box preview (called on mode switch or after box fires)."""
        self._sam_box_anchor = None
        self._sam_press = None
        self._clear_sam_box_artists()

    def set_exclude_zone(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Draw a persistent red dashed rectangle marking the SAM exclude zone."""
        self.clear_exclude_zone()
        import matplotlib.patches as mpatches
        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        fill = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=0, edgecolor="none", facecolor="#cc2222", alpha=0.15, zorder=8,
        )
        border = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor="#cc2222", facecolor="none",
            linestyle="--", alpha=0.85, zorder=9,
        )
        self.canvas.ax.add_patch(fill)
        self.canvas.ax.add_patch(border)
        self._exclude_artists = [fill, border]
        self._mpl_canvas.draw_idle()

    def clear_exclude_zone(self) -> None:
        for a in self._exclude_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._exclude_artists.clear()
        self._mpl_canvas.draw_idle()

    def set_crop_region(self, x0: float, y0: float, x1: float, y1: float) -> None:
        """Draw a persistent cyan border marking the SAM crop region."""
        self.clear_crop_region()
        import matplotlib.patches as mpatches
        rx, ry = min(x0, x1), min(y0, y1)
        rw, rh = abs(x1 - x0), abs(y1 - y0)
        border = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=2.0, edgecolor="#00aacc", facecolor="none",
            linestyle="-", alpha=0.85, zorder=9,
        )
        self.canvas.ax.add_patch(border)
        self._crop_artists = [border]
        self._mpl_canvas.draw_idle()

    def clear_crop_region(self) -> None:
        for a in self._crop_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._crop_artists.clear()
        self._mpl_canvas.draw_idle()

    def _clear_sam_box_artists(self) -> None:
        for a in self._sam_box_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._sam_box_artists.clear()
        try:
            self.canvas.ax.figure.canvas.draw_idle()
        except Exception:
            pass

    def _update_sam_box_preview(self, x2: float, y2: float) -> None:
        """Redraw the dashed-rectangle preview from the box anchor to (x2, y2)."""
        if self._sam_box_anchor is None:
            return
        import matplotlib.patches as mpatches
        x0, y0 = self._sam_box_anchor
        self._clear_sam_box_artists()
        rx, ry = min(x0, x2), min(y0, y2)
        rw, rh = abs(x2 - x0), abs(y2 - y0)
        rect = mpatches.Rectangle(
            (rx, ry), rw, rh,
            linewidth=1.5, edgecolor="#bf7fff", facecolor="none",
            linestyle="--", alpha=0.9,
        )
        self.canvas.ax.add_patch(rect)
        self._sam_box_artists.append(rect)
        try:
            self.canvas.ax.figure.canvas.draw_idle()
        except Exception:
            pass
        self._mpl_canvas.draw_idle()

    def update_nav_label(self, current: int, total: int) -> None:
        self._nav_label.setText(f"{current} / {total}")
        self._prev_btn.setEnabled(total > 1)
        self._next_btn.setEnabled(total > 1)

    def set_nav_enabled(self, enabled: bool) -> None:
        self._prev_btn.setEnabled(enabled)
        self._next_btn.setEnabled(enabled)

    def reset_interaction(self) -> None:
        """Cancel any in-progress draw/drag/selection state. Call on image switch."""
        self._cancel_drag()
        self.clear_rubber_band()
        self._moving_ann = None
        self._move_start = None
        self._resizing_ann = None
        self._resize_handle = None

    def refresh(self) -> None:
        self._mpl_canvas.draw_idle()

    def force_redraw(self) -> None:
        """Schedule a redraw and flush pending Qt events so it executes immediately."""
        self._mpl_canvas.draw_idle()
        from PyQt6.QtCore import QCoreApplication, QEventLoop
        QCoreApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)

    # ── drag helpers ──────────────────────────────────────────────────────────

    def _cancel_drag(self) -> None:
        self._drag_active = False
        self._drag_start = None
        self._freehand_pts.clear()
        self._clear_preview()

    def _clear_preview(self) -> None:
        for artist in self._preview_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._preview_artists.clear()
        self._mpl_canvas.draw_idle()

    def _clear_rubber_band_artists(self) -> None:
        for artist in self._rubber_band_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._rubber_band_artists.clear()
        # no draw_idle here — caller decides when to redraw

    def _update_rubber_band_preview(self, x2: float, y2: float) -> None:
        """Draw live preview from confirmed anchor points to the current cursor position."""
        pts = self._rubber_band_pts
        if not pts:
            return
        ax = self.canvas.ax
        self._clear_rubber_band_artists()

        # Confirmed segments (solid line through all anchors)
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            line, = ax.plot(
                xs, ys,
                color=_PREVIEW_COLOR, linewidth=_PREVIEW_LW,
                linestyle="-", alpha=_PREVIEW_ALPHA, zorder=10,
            )
            self._rubber_band_artists.append(line)

        # Small dot at each confirmed anchor
        for p in pts:
            dot, = ax.plot(
                [p[0]], [p[1]], "o",
                color=_PREVIEW_COLOR, markersize=4,
                alpha=_PREVIEW_ALPHA, zorder=11,
            )
            self._rubber_band_artists.append(dot)

        # Dashed line from last anchor to cursor
        lx, ly = pts[-1]
        dash, = ax.plot(
            [lx, x2], [ly, y2],
            color=_PREVIEW_COLOR, linewidth=_PREVIEW_LW,
            linestyle="--", alpha=_PREVIEW_ALPHA, zorder=10,
        )
        self._rubber_band_artists.append(dash)
        self._mpl_canvas.draw_idle()

    @staticmethod
    def _apply_shift_constraint(
        tool: str, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float]:
        """Snap endpoint for line/arrow (45° steps) or square for rectangle."""
        if tool in ("line", "arrow"):
            dx, dy = x2 - x1, y2 - y1
            angle = math.atan2(dy, dx)
            snapped = round(angle / (math.pi / 4)) * (math.pi / 4)
            dist = math.hypot(dx, dy)
            return x1 + dist * math.cos(snapped), y1 + dist * math.sin(snapped)
        if tool == "rectangle":
            dx, dy = abs(x2 - x1), abs(y2 - y1)
            side = min(dx, dy)
            sx = 1 if x2 >= x1 else -1
            sy = 1 if y2 >= y1 else -1
            return x1 + side * sx, y1 + side * sy
        return x2, y2

    def _update_drag_preview(
        self, tool: str, x1: float, y1: float, x2: float, y2: float
    ) -> None:
        """Create or update the temporary preview artist for drag tools."""
        ax = self.canvas.ax

        if tool in ("line", "arrow"):
            if not self._preview_artists:
                line, = ax.plot(
                    [x1, x2], [y1, y2],
                    color=_PREVIEW_COLOR, linewidth=_PREVIEW_LW,
                    linestyle="--", alpha=_PREVIEW_ALPHA, zorder=10,
                )
                self._preview_artists.append(line)
            else:
                self._preview_artists[0].set_data([x1, x2], [y1, y2])

        elif tool == "circle":
            r = math.hypot(x2 - x1, y2 - y1)
            if not self._preview_artists:
                from matplotlib.patches import Circle
                patch = Circle(
                    (x1, y1), r,
                    color=_PREVIEW_COLOR, fill=False,
                    linewidth=_PREVIEW_LW, linestyle="--",
                    alpha=_PREVIEW_ALPHA, zorder=10,
                )
                ax.add_patch(patch)
                self._preview_artists.append(patch)
            else:
                self._preview_artists[0].set_radius(r)

        elif tool == "rectangle":
            x0, y0 = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            if not self._preview_artists:
                from matplotlib.patches import Rectangle
                patch = Rectangle(
                    (x0, y0), w, h,
                    color=_PREVIEW_COLOR, fill=False,
                    linewidth=_PREVIEW_LW, linestyle="--",
                    alpha=_PREVIEW_ALPHA, zorder=10,
                )
                ax.add_patch(patch)
                self._preview_artists.append(patch)
            else:
                self._preview_artists[0].set_xy((x0, y0))
                self._preview_artists[0].set_width(w)
                self._preview_artists[0].set_height(h)

        self._mpl_canvas.draw_idle()

    def _update_freehand_preview(self) -> None:
        """Update the polyline preview for freehand drawing."""
        if len(self._freehand_pts) < 2:
            return
        xs = [p[0] for p in self._freehand_pts]
        ys = [p[1] for p in self._freehand_pts]
        if not self._preview_artists:
            line, = self.canvas.ax.plot(
                xs, ys,
                color=_PREVIEW_COLOR, linewidth=_PREVIEW_LW,
                linestyle="--", alpha=_PREVIEW_ALPHA, zorder=10,
            )
            self._preview_artists.append(line)
        else:
            self._preview_artists[0].set_data(xs, ys)
        self._mpl_canvas.draw_idle()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_key_press(self, event) -> None:
        if event.key == "shift":
            self._shift_held = True
        elif event.key in ("delete", "backspace"):
            renderer = self.canvas.renderer
            if renderer is not None:
                ann = renderer.selected_annotation()
                if ann is not None:
                    self.annotation_delete_requested.emit(ann)
        elif event.key == "escape":
            renderer = self.canvas.renderer
            if renderer is not None:
                renderer.clear_selection_blit(self.canvas)
            self._moving_ann = None
            self._move_start = None
            self.annotation_selected.emit(None)

    def _on_key_release(self, event) -> None:
        if event.key == "shift":
            self._shift_held = False

    def _on_press(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        if self._toolbar.mode.name not in ("", "NONE"):
            return
        x, y, btn = float(event.xdata), float(event.ydata), event.button or 1

        if btn == 1 and self._tool == "none":
            renderer = self.canvas.renderer
            if renderer is not None:
                # Handle hit takes priority — start resize
                handle_id = renderer.hit_test_handle(event)
                if handle_id is not None:
                    self._resizing_ann = renderer.selected_annotation()
                    self._resize_handle = handle_id
                    return
                # Body hit — select + start move
                ann = renderer.hit_test(event)
                if ann is not None:
                    renderer.set_selected(ann, draw=False)
                    self.canvas.blit_annotations()
                    self._moving_ann = ann
                    self._move_start = (x, y)
                    self.annotation_selected.emit(ann)
                else:
                    renderer.clear_selection_blit(self.canvas)
                    self._moving_ann = None
                    self._move_start = None
                    self.annotation_selected.emit(None)
            return

        if btn == 1 and self._tool in _DRAG_TOOLS:
            # Start a drag stroke
            self._drag_active = True
            self._drag_start = (x, y)
            if self._tool == "freehand":
                self._freehand_pts = [(x, y)]
        else:
            # Single-click tools (text, scalebar, angle, roi, distance, …)
            # Also passes right-click (button=3) for ROI polygon closing
            if self._flat_pick_active and btn == 1:
                self._flat_pick_press = (x, y)
                return   # consume event — don't start any other interaction
            if btn == 1 and self._tool == "sam":
                self._sam_press = (x, y)
            self.click_event.emit(x, y, btn)

    def _on_motion(self, event) -> None:
        # Re-assert cursor — matplotlib's own set_cursor() may override ours
        self._sync_cursor()

        # ── hover status ──────────────────────────────────────────────────────
        if event.inaxes is None or event.xdata is None:
            self._status.setText("x=— y=— | intensity=—")
        else:
            x, y = int(round(event.xdata)), int(round(event.ydata))
            norm = self.canvas.norm_image
            dm4  = self.canvas.dm4
            if norm is not None and dm4 is not None:
                h, w = norm.shape[:2]
                if 0 <= y < h and 0 <= x < w:
                    val  = norm[y, x]
                    nm_x = x * dm4.pixel_size
                    nm_y = y * dm4.pixel_size
                    self._status.setText(
                        f"x={x} y={y} | {nm_x:.1f} nm, {nm_y:.1f} nm | intensity={val:.4f}"
                    )
                else:
                    self._status.setText(f"x={x} y={y}")
            else:
                self._status.setText(f"x={x} y={y}")

        # ── annotation resize drag ────────────────────────────────────────────
        if self._resizing_ann is not None and self._resize_handle is not None:
            if event.inaxes is not None and event.xdata is not None:
                renderer = self.canvas.renderer
                if renderer is not None:
                    self._apply_resize(
                        self._resizing_ann, self._resize_handle,
                        float(event.xdata), float(event.ydata), self.canvas,
                    )
                    renderer.update_inplace(self._resizing_ann)
                    renderer._update_selection_geometry(self._resizing_ann)
                    self.canvas.blit_annotations()
            return

        # ── annotation move drag ──────────────────────────────────────────────
        if self._moving_ann is not None and self._move_start is not None:
            if event.inaxes is not None and event.xdata is not None:
                x, y = float(event.xdata), float(event.ydata)
                dx = x - self._move_start[0]
                dy = y - self._move_start[1]
                renderer = self.canvas.renderer
                if renderer is not None:
                    self._translate_ann(self._moving_ann, dx, dy, self.canvas.dm4)
                    renderer.update_inplace(self._moving_ann)
                    renderer._update_selection_geometry(self._moving_ann)
                    self.canvas.blit_annotations()
                self._move_start = (x, y)
            return

        # ── rubber-band preview (click-based tools) ───────────────────────────
        if self._rubber_band_pts and event.inaxes is not None and event.xdata is not None:
            self._update_rubber_band_preview(float(event.xdata), float(event.ydata))

        # ── SAM box live preview ───────────────────────────────────────────────
        if self._sam_box_anchor is not None and event.inaxes is not None and event.xdata is not None:
            self._update_sam_box_preview(float(event.xdata), float(event.ydata))

        # ── drag preview update ───────────────────────────────────────────────
        if not self._drag_active or self._drag_start is None:
            return
        if event.inaxes is None or event.xdata is None:
            return

        x1, y1 = self._drag_start
        x2, y2 = float(event.xdata), float(event.ydata)

        if self._tool == "freehand":
            # Accumulate points at ~5 px spacing
            if self._freehand_pts:
                lx, ly = self._freehand_pts[-1]
                if (x2 - lx) ** 2 + (y2 - ly) ** 2 >= 25:
                    self._freehand_pts.append((x2, y2))
            self._update_freehand_preview()
        else:
            if self._shift_held:
                x2, y2 = self._apply_shift_constraint(self._tool, x1, y1, x2, y2)
            self._update_drag_preview(self._tool, x1, y1, x2, y2)

    def _on_release(self, event) -> None:
        # ── flat-region one-shot pick ─────────────────────────────────────────
        if (self._flat_pick_press is not None
                and self._flat_pick_active
                and event.inaxes is not None
                and event.xdata is not None):
            x0, y0 = self._flat_pick_press
            x1, y1 = float(event.xdata), float(event.ydata)
            self._flat_pick_press  = None
            self._flat_pick_active = False
            self._sync_cursor()
            if math.hypot(x1 - x0, y1 - y0) > 4.0:
                self.flat_region_picked.emit(
                    min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
                )
            return

        # ── SAM box drag-to-draw ───────────────────────────────────────────────
        if (self._sam_press is not None
                and self._tool == "sam"
                and event.inaxes is not None
                and event.xdata is not None):
            x0, y0 = self._sam_press
            x1, y1 = float(event.xdata), float(event.ydata)
            self._sam_press = None
            if math.hypot(x1 - x0, y1 - y0) > 8.0:
                # Meaningful drag — treat as box prompt
                self._clear_sam_box_artists()
                self._sam_box_anchor = None
                self.sam_box_commit.emit(
                    min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)
                )
                return
        self._sam_press = None

        # Clear annotation move/resize state on any release
        if self._resizing_ann is not None:
            self._resizing_ann = None
            self._resize_handle = None
        if self._moving_ann is not None:
            self._moving_ann = None
            self._move_start = None

        if not self._drag_active:
            return
        if self._toolbar.mode.name not in ("", "NONE"):
            self._cancel_drag()
            return

        # Remove preview before the committed annotation is rendered
        self._clear_preview()
        self._drag_active = False

        if event.inaxes is None or event.xdata is None:
            self._drag_start = None
            self._freehand_pts.clear()
            return

        x1, y1 = self._drag_start
        x2, y2 = float(event.xdata), float(event.ydata)
        self._drag_start = None

        if self._tool == "freehand":
            pts = list(self._freehand_pts)
            self._freehand_pts.clear()
            if len(pts) >= 3:
                self.freehand_commit.emit(pts)
        else:
            if self._shift_held:
                x2, y2 = self._apply_shift_constraint(self._tool, x1, y1, x2, y2)
            # Only commit if mouse actually moved (avoids accidental zero-size shapes)
            if math.hypot(x2 - x1, y2 - y1) > 1.0:
                self.drag_commit.emit(self._tool, x1, y1, x2, y2, self._shift_held)

    def _on_scroll(self, event) -> None:
        if event.inaxes is None:
            return
        factor = 1.25 if event.step < 0 else 1 / 1.25
        ax = event.inaxes
        xdata, ydata = event.xdata, event.ydata
        ax.set_xlim([xdata + (x - xdata) * factor for x in ax.get_xlim()])
        ax.set_ylim([ydata + (y - ydata) * factor for y in ax.get_ylim()])
        self._mpl_canvas.draw_idle()

    @staticmethod
    def _translate_ann(ann, dx: float, dy: float, dm4=None) -> None:
        """Translate annotation coordinates in-place by (dx, dy) pixels."""
        t = ann.type
        if t in ("arrow", "line", "distance"):
            ann.p1 = (ann.p1[0] + dx, ann.p1[1] + dy)
            ann.p2 = (ann.p2[0] + dx, ann.p2[1] + dy)
        elif t == "circle":
            ann.cx += dx
            ann.cy += dy
        elif t == "rectangle":
            ann.x0 += dx
            ann.y0 += dy
            ann.x1 += dx
            ann.y1 += dy
        elif t == "text":
            ann.x += dx
            ann.y += dy
        elif t == "scalebar":
            if dm4 is not None and dm4.shape:
                h, w = dm4.shape[:2]
                if w > 0:
                    ann.x_frac += dx / w
                if h > 0:
                    ann.y_frac += dy / h
        elif t == "angle":
            ann.p1 = (ann.p1[0] + dx, ann.p1[1] + dy)
            ann.vertex = (ann.vertex[0] + dx, ann.vertex[1] + dy)
            ann.p2 = (ann.p2[0] + dx, ann.p2[1] + dy)
        elif t == "roi":
            ann.vertices = [(x + dx, y + dy) for x, y in ann.vertices]

    @staticmethod
    def _apply_resize(ann, handle_id: str, x: float, y: float, canvas) -> None:
        """Update annotation geometry for a resize handle drag to (x, y)."""
        t = ann.type
        ps = canvas.renderer.pixel_size if canvas.renderer else 1.0

        if t in ("line", "arrow"):
            if handle_id == "p1":
                ann.p1 = (x, y)
            elif handle_id == "p2":
                ann.p2 = (x, y)

        elif t == "distance":
            if handle_id == "p1":
                ann.p1 = (x, y)
            elif handle_id == "p2":
                ann.p2 = (x, y)
            ann.distance_nm = math.hypot(
                ann.p2[0] - ann.p1[0], ann.p2[1] - ann.p1[1]
            ) * ps

        elif t == "circle":
            ann.r = max(1.0, math.hypot(x - ann.cx, y - ann.cy))

        elif t == "rectangle":
            if handle_id == "tl":
                ann.x0, ann.y0 = x, y
            elif handle_id == "tr":
                ann.x1, ann.y0 = x, y
            elif handle_id == "bl":
                ann.x0, ann.y1 = x, y
            elif handle_id == "br":
                ann.x1, ann.y1 = x, y

        elif t == "angle":
            if handle_id == "p1":
                ann.p1 = (x, y)
            elif handle_id == "vertex":
                ann.vertex = (x, y)
            elif handle_id == "p2":
                ann.p2 = (x, y)
            v = ann.vertex
            a1 = math.atan2(-(ann.p1[1] - v[1]), ann.p1[0] - v[0])
            a2 = math.atan2(-(ann.p2[1] - v[1]), ann.p2[0] - v[0])
            diff = abs(math.degrees(a1) - math.degrees(a2)) % 360
            ann.angle_deg = min(diff, 360.0 - diff)

        elif t == "scalebar":
            if handle_id == "right" and ps > 0:
                ax = canvas.ax
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                w_img = abs(xlim[1] - xlim[0])
                x0 = ann.x_frac * w_img + xlim[0]
                ann.nm = max(1.0, (x - x0) * ps)

        elif t == "roi":
            if handle_id.startswith("v"):
                idx = int(handle_id[1:])
                verts = list(ann.vertices)
                if 0 <= idx < len(verts):
                    verts[idx] = (x, y)
                    ann.vertices = verts
                    if ps > 0 and len(verts) >= 3:
                        import numpy as np
                        xy = np.array(verts)
                        n = len(xy)
                        area_px2 = 0.5 * abs(sum(
                            xy[i][0] * xy[(i+1) % n][1] - xy[(i+1) % n][0] * xy[i][1]
                            for i in range(n)
                        ))
                        ann.area_nm2 = area_px2 * ps * ps

