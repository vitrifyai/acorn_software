"""Renders annotation dataclasses onto a matplotlib Axes."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Polygon

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from acorn.core.annotations import AnyAnnotation, AnnotationStore


_SEL_COLOR = "#00FFFF"
_SEL_LW_EXTRA = 3.0
_SEL_ALPHA = 0.55


class AnnotationRenderer:
    """
    Stateful renderer: tracks a list of artists per annotation for incremental
    updates, hit testing, and selection highlighting.

    Parameters
    ----------
    ax          : matplotlib Axes to draw onto
    pixel_size  : nm/px (needed for distance / angle labels)
    """

    def __init__(self, ax: "Axes", pixel_size: float = 1.0) -> None:
        self.ax = ax
        self.pixel_size = pixel_size
        self._ann_to_artists: dict[int, list] = {}      # id(ann) → artists
        self._id_to_ann: dict[int, "AnyAnnotation"] = {}
        self._selected_id: Optional[int] = None
        self._selection_artists: list = []
        self._staging: list = []   # temporary buffer during _draw_one

    # ── public API ────────────────────────────────────────────────────────────

    def clear(self) -> None:
        """Remove all annotation artists from the Axes."""
        self._clear_selection_artists()
        self._selected_id = None
        for artists in self._ann_to_artists.values():
            for art in artists:
                try:
                    art.remove()
                except Exception:
                    pass
        self._ann_to_artists.clear()
        self._id_to_ann.clear()

    def render(self, store: "AnnotationStore") -> None:
        """Clear and redraw every annotation in *store*."""
        self.clear()
        for ann in store:
            self._draw_and_register(ann)
        self.ax.figure.canvas.draw_idle()

    def render_noblit(self, store: "AnnotationStore", canvas) -> None:
        """Full redraw optimised for blitting.

        Order: clear annotations → synchronous draw (image only) → save bg →
        draw annotations → blit.  This ensures the saved background never
        contains annotation artists, so subsequent blit calls don't double-paint.

        *canvas* is the CryoCanvas that owns this renderer.
        """
        self.clear()
        # Temporarily hide overlay artists (SAM point markers etc.) so they are
        # not baked into the background cache — they must only appear via blit.
        overlays = canvas._overlay_artists
        for art in overlays:
            art.set_visible(False)
        # draw() is synchronous: renders the image without any annotation artists
        self.ax.figure.canvas.draw()
        canvas._save_bg()
        for art in overlays:
            art.set_visible(True)
        # Now draw annotations on top of the saved background
        for ann in store:
            self._draw_and_register(ann)
        canvas.blit_annotations()

    def add_one(self, ann: "AnyAnnotation", draw: bool = True) -> None:
        """Add a single annotation without clearing existing ones."""
        self._draw_and_register(ann)
        if draw:
            self.ax.figure.canvas.draw_idle()

    def add_one_blit(self, ann: "AnyAnnotation", canvas) -> None:
        """Add a single annotation and update the display via blit."""
        self._draw_and_register(ann)
        canvas.blit_annotations()

    def update_inplace(self, ann: "AnyAnnotation") -> None:
        """Update artist geometry for a moved/resized annotation without recreating artists.

        Does NOT call draw_idle — caller is responsible.
        """
        t = ann.type
        ann_id = id(ann)
        artists = self._ann_to_artists.get(ann_id, [])

        if t == "line" and artists:
            artists[0].set_data([ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]])
            return
        if t == "circle" and artists:
            artists[0].set_center((ann.cx, ann.cy))
            artists[0].set_radius(ann.r)
            return
        if t == "rectangle" and artists:
            x0, y0 = min(ann.x0, ann.x1), min(ann.y0, ann.y1)
            artists[0].set_xy((x0, y0))
            artists[0].set_width(abs(ann.x1 - ann.x0))
            artists[0].set_height(abs(ann.y1 - ann.y0))
            return
        if t == "text" and artists:
            artists[0].set_position((ann.x, ann.y))
            return
        if t == "roi" and len(artists) >= 2:
            xy = np.array(ann.vertices)
            artists[0].set_xy(xy)
            cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
            artists[1].set_position((cx, cy))
            lbl = ann.label if ann.label else f"A={ann.area_nm2:.0f} nm²"
            if ann.stats:
                lbl += f"\nμ={ann.stats.get('mean', 0):.3f}"
            artists[1].set_text(lbl)
            return
        if t == "distance" and len(artists) >= 4:
            artists[0].set_data([ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]])
            mx = (ann.p1[0] + ann.p2[0]) / 2
            my = (ann.p1[1] + ann.p2[1]) / 2
            if not getattr(ann, "calibrated", True):
                label = f"{ann.distance_px:.1f} px"
            elif ann.distance_nm < 1000:
                label = f"{ann.distance_nm:.1f} nm"
            else:
                label = f"{ann.distance_nm/1000:.2g} µm"
            artists[1].set_position((mx, my))
            artists[1].set_text(label)
            artists[2].set_data([ann.p1[0]], [ann.p1[1]])
            artists[3].set_data([ann.p2[0]], [ann.p2[1]])
            return
        # Fallback for arrow, angle, scalebar: recreate artists
        self.remove_one(ann)
        self._draw_and_register(ann)

    def remove_one(self, ann: "AnyAnnotation") -> None:
        """Remove a single annotation's artists (no redraw)."""
        ann_id = id(ann)
        if ann_id == self._selected_id:
            self._clear_selection_artists()
            self._selected_id = None
        artists = self._ann_to_artists.pop(ann_id, [])
        for art in artists:
            try:
                art.remove()
            except Exception:
                pass
        self._id_to_ann.pop(ann_id, None)

    def hit_test(self, event) -> Optional["AnyAnnotation"]:
        """Return the topmost annotation under the mouse event, or None."""
        if event.xdata is None or event.ydata is None:
            return None
        # Reverse order so most recently added annotation wins on overlap
        for ann_id, artists in reversed(list(self._ann_to_artists.items())):
            for art in artists:
                try:
                    contains, _ = art.contains(event)
                    if contains:
                        return self._id_to_ann.get(ann_id)
                except Exception:
                    pass
        return None

    def set_selected(self, ann: "AnyAnnotation", draw: bool = True) -> None:
        """Highlight an annotation with a selection glow."""
        self._clear_selection_artists()
        ann_id = id(ann)
        if ann_id not in self._id_to_ann:
            return
        self._selected_id = ann_id
        self._draw_selection_glow(ann)
        if draw:
            self.ax.figure.canvas.draw_idle()

    def clear_selection(self) -> None:
        """Remove the selection highlight."""
        had = self._selected_id is not None
        self._clear_selection_artists()
        if had:
            self.ax.figure.canvas.draw_idle()

    def clear_selection_blit(self, canvas) -> None:
        """Remove the selection highlight using blit instead of draw_idle."""
        had = self._selected_id is not None
        self._clear_selection_artists()
        if had:
            canvas.blit_annotations()

    def selected_annotation(self) -> Optional["AnyAnnotation"]:
        """Return the currently selected annotation, or None."""
        if self._selected_id is None:
            return None
        return self._id_to_ann.get(self._selected_id)

    def hit_test_handle(self, event) -> Optional[str]:
        """Return the handle_id under the mouse for the selected annotation, or None."""
        if event.xdata is None or event.ydata is None or self._selected_id is None:
            return None
        ann = self._id_to_ann.get(self._selected_id)
        if ann is None:
            return None
        sz = self._get_handle_size() * 1.5
        for hx, hy, hid in self._get_handle_points(ann):
            if abs(event.xdata - hx) <= sz and abs(event.ydata - hy) <= sz:
                return hid
        return None

    # ── internal helpers ──────────────────────────────────────────────────────

    def _get_handle_size(self) -> float:
        """Return half-size of handle squares in data (pixel) coordinates."""
        xlim = self.ax.get_xlim()
        return max(2.0, abs(xlim[1] - xlim[0]) * 0.012)

    def _get_handle_points(self, ann) -> list:
        """Return [(x, y, handle_id), ...] for resize handles of *ann*."""
        t = ann.type
        if t in ("line", "arrow", "distance"):
            return [(ann.p1[0], ann.p1[1], "p1"), (ann.p2[0], ann.p2[1], "p2")]
        if t == "circle":
            return [(ann.cx + ann.r, ann.cy, "radius")]
        if t == "rectangle":
            return [
                (ann.x0, ann.y0, "tl"), (ann.x1, ann.y0, "tr"),
                (ann.x0, ann.y1, "bl"), (ann.x1, ann.y1, "br"),
            ]
        if t == "scalebar" and self.pixel_size > 0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            w_img = abs(xlim[1] - xlim[0])
            h_img = abs(ylim[0] - ylim[1])
            bar_px = ann.nm / self.pixel_size
            x0 = ann.x_frac * w_img + xlim[0]
            y0 = ann.y_frac * h_img + min(ylim)
            bar_h = max(h_img * 0.007, 3)
            return [(x0 + bar_px, y0 + bar_h / 2, "right")]
        if t == "angle":
            return [
                (ann.p1[0], ann.p1[1], "p1"),
                (ann.vertex[0], ann.vertex[1], "vertex"),
                (ann.p2[0], ann.p2[1], "p2"),
            ]
        if t == "roi":
            return [(x, y, f"v{i}") for i, (x, y) in enumerate(ann.vertices)]
        return []

    def _draw_and_register(self, ann: "AnyAnnotation") -> None:
        ann_id = id(ann)
        self._id_to_ann[ann_id] = ann
        self._staging = []
        self._draw_one(ann)
        self._ann_to_artists[ann_id] = self._staging
        self._staging = []

    def _update_selection_geometry(self, ann: "AnyAnnotation") -> None:
        """Update glow + handle square positions in-place during a drag.

        Avoids destroying and recreating all selection artists on every
        mouse-move event.  Falls back to a full rebuild if something is off.
        """
        if self._selected_id != id(ann) or not self._selection_artists:
            self._clear_selection_artists()
            self._draw_selection_glow(ann)
            return

        handles = self._get_handle_points(ann)
        n_h = len(handles)
        glow = self._selection_artists[:-n_h] if n_h else self._selection_artists
        t = ann.type

        try:
            if t in ("line", "arrow", "distance") and glow:
                glow[0].set_data([ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]])
            elif t == "circle" and glow:
                glow[0].set_center((ann.cx, ann.cy))
                glow[0].set_radius(ann.r)
            elif t == "rectangle" and glow:
                x0, y0 = min(ann.x0, ann.x1), min(ann.y0, ann.y1)
                glow[0].set_xy((x0, y0))
                glow[0].set_width(abs(ann.x1 - ann.x0))
                glow[0].set_height(abs(ann.y1 - ann.y0))
            elif t == "roi" and glow:
                glow[0].set_xy(np.array(ann.vertices))
            elif t == "angle" and len(glow) >= 2:
                v = ann.vertex
                glow[0].set_data([v[0], ann.p1[0]], [v[1], ann.p1[1]])
                glow[1].set_data([v[0], ann.p2[0]], [v[1], ann.p2[1]])
        except Exception:
            self._clear_selection_artists()
            self._draw_selection_glow(ann)
            return

        # Reposition handle squares
        sz = self._get_handle_size()
        handle_arts = self._selection_artists[-n_h:] if n_h else []
        try:
            for rect, (hx, hy, _) in zip(handle_arts, handles):
                rect.set_xy((hx - sz, hy - sz))
        except Exception:
            self._clear_selection_artists()
            self._draw_selection_glow(ann)

    def _clear_selection_artists(self) -> None:
        for art in self._selection_artists:
            try:
                art.remove()
            except Exception:
                pass
        self._selection_artists.clear()
        self._selected_id = None

    def _draw_selection_glow(self, ann: "AnyAnnotation") -> None:
        """Draw a cyan highlight overlay behind the annotation."""
        t = ann.type
        lw = getattr(ann, "linewidth", 2.0)
        glow_lw = lw + _SEL_LW_EXTRA

        if t in ("arrow", "line", "distance"):
            ln = Line2D(
                [ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]],
                color=_SEL_COLOR, lw=glow_lw, alpha=_SEL_ALPHA, zorder=4,
            )
            self.ax.add_line(ln)
            self._selection_artists.append(ln)

        elif t == "circle":
            c = Circle(
                (ann.cx, ann.cy), ann.r,
                fill=False, edgecolor=_SEL_COLOR, lw=glow_lw,
                alpha=_SEL_ALPHA, zorder=4,
            )
            self.ax.add_patch(c)
            self._selection_artists.append(c)

        elif t == "rectangle":
            x0, y0 = min(ann.x0, ann.x1), min(ann.y0, ann.y1)
            r = Rectangle(
                (x0, y0), abs(ann.x1 - ann.x0), abs(ann.y1 - ann.y0),
                fill=False, edgecolor=_SEL_COLOR, lw=glow_lw,
                alpha=_SEL_ALPHA, zorder=4,
            )
            self.ax.add_patch(r)
            self._selection_artists.append(r)

        elif t == "text":
            txt = self.ax.text(
                ann.x, ann.y, ann.label,
                color=_SEL_COLOR, fontsize=ann.fontsize, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=_SEL_COLOR, alpha=0.2),
                zorder=4,
            )
            self._selection_artists.append(txt)

        elif t == "roi":
            if len(ann.vertices) >= 3:
                xy = np.array(ann.vertices)
                poly = Polygon(
                    xy, closed=True,
                    fill=False, edgecolor=_SEL_COLOR,
                    lw=glow_lw, linestyle="-", alpha=_SEL_ALPHA, zorder=4,
                )
                self.ax.add_patch(poly)
                self._selection_artists.append(poly)

        elif t == "angle":
            v = ann.vertex
            for pt in (ann.p1, ann.p2):
                ln = Line2D(
                    [v[0], pt[0]], [v[1], pt[1]],
                    color=_SEL_COLOR, lw=glow_lw, alpha=_SEL_ALPHA, zorder=4,
                )
                self.ax.add_line(ln)
                self._selection_artists.append(ln)

        elif t == "scalebar":
            if self.pixel_size > 0:
                ax = self.ax
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                w_img = abs(xlim[1] - xlim[0])
                h_img = abs(ylim[0] - ylim[1])
                bar_px = ann.nm / self.pixel_size
                x0 = ann.x_frac * w_img + xlim[0]
                y0 = ann.y_frac * h_img + min(ylim)
                bar_h = max(h_img * 0.007, 3)
                from matplotlib.patches import Rectangle as _Rect
                rect = _Rect(
                    (x0 - 2, y0 - 2), bar_px + 4, bar_h + 4,
                    fill=False, edgecolor=_SEL_COLOR, lw=2,
                    alpha=_SEL_ALPHA, zorder=4,
                )
                ax.add_patch(rect)
                self._selection_artists.append(rect)

        # Draw resize handles as small squares at key points
        sz = self._get_handle_size()
        for hx, hy, _ in self._get_handle_points(ann):
            sq = Rectangle(
                (hx - sz, hy - sz), sz * 2, sz * 2,
                fill=True, facecolor=_SEL_COLOR, edgecolor="white",
                lw=1.0, alpha=0.9, zorder=8,
            )
            self.ax.add_patch(sq)
            self._selection_artists.append(sq)

    # ── internal dispatch ─────────────────────────────────────────────────────

    def _draw_one(self, ann: "AnyAnnotation") -> None:
        t = ann.type
        if t == "arrow":
            self._arrow(ann)
        elif t == "line":
            self._line(ann)
        elif t == "circle":
            self._circle(ann)
        elif t == "rectangle":
            self._rectangle(ann)
        elif t == "text":
            self._text(ann)
        elif t == "scalebar":
            self._scalebar(ann)
        elif t == "distance":
            self._distance(ann)
        elif t == "angle":
            self._angle(ann)
        elif t == "roi":
            self._roi(ann)

    # ── shape renderers ───────────────────────────────────────────────────────

    def _arrow(self, ann) -> None:
        a = self.ax.annotate(
            "",
            xy=ann.p2, xytext=ann.p1,
            arrowprops=dict(
                arrowstyle="->",
                color=ann.color,
                lw=ann.linewidth,
                mutation_scale=16,
            ),
            zorder=5,
        )
        self._staging.append(a)
        # invisible pick line so hit_test works on the arrow shaft
        ln = Line2D(
            [ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]],
            color="none", lw=0, alpha=0, zorder=5, pickradius=6,
        )
        self.ax.add_line(ln)
        self._staging.append(ln)

    def _line(self, ann) -> None:
        ln = Line2D(
            [ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]],
            color=ann.color, lw=ann.linewidth,
            linestyle=getattr(ann, "linestyle", "-"),
            zorder=5, pickradius=5,
        )
        self.ax.add_line(ln)
        self._staging.append(ln)

    def _circle(self, ann) -> None:
        c = Circle(
            (ann.cx, ann.cy), ann.r,
            fill=False, edgecolor=ann.color, lw=ann.linewidth,
            linestyle=getattr(ann, "linestyle", "-"), zorder=5,
        )
        self.ax.add_patch(c)
        self._staging.append(c)

    def _rectangle(self, ann) -> None:
        x0, y0 = min(ann.x0, ann.x1), min(ann.y0, ann.y1)
        r = Rectangle(
            (x0, y0), abs(ann.x1 - ann.x0), abs(ann.y1 - ann.y0),
            fill=False, edgecolor=ann.color, lw=ann.linewidth,
            linestyle=getattr(ann, "linestyle", "-"), zorder=5,
        )
        self.ax.add_patch(r)
        self._staging.append(r)

    def _text(self, ann) -> None:
        txt = self.ax.text(
            ann.x, ann.y, ann.label,
            color=ann.color,
            fontsize=ann.fontsize,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.45),
            zorder=6,
        )
        self._staging.append(txt)

    def _scalebar(self, ann) -> None:
        from acorn.render.scalebar import draw_scalebar
        artists = draw_scalebar(
            self.ax,
            length_nm=ann.nm,
            pixel_size=self.pixel_size,
            x_frac=ann.x_frac,
            y_frac=ann.y_frac,
            color=ann.color,
            linewidth=ann.linewidth,
            fontsize=ann.fontsize,
        )
        self._staging.extend(artists)

    def _distance(self, ann) -> None:
        """Dimension line with label at midpoint."""
        ln = Line2D(
            [ann.p1[0], ann.p2[0]], [ann.p1[1], ann.p2[1]],
            color=ann.color, lw=ann.linewidth,
            linestyle="--", zorder=5, pickradius=5,
        )
        self.ax.add_line(ln)
        self._staging.append(ln)
        mx = (ann.p1[0] + ann.p2[0]) / 2
        my = (ann.p1[1] + ann.p2[1]) / 2
        if not getattr(ann, "calibrated", True):
            label = f"{ann.distance_px:.1f} px"
        elif ann.distance_nm < 1000:
            label = f"{ann.distance_nm:.1f} nm"
        else:
            label = f"{ann.distance_nm/1000:.2g} µm"
        txt = self.ax.text(
            mx, my, label,
            color=ann.color, fontsize=10, fontweight="bold",
            ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
            zorder=7,
        )
        self._staging.append(txt)
        # endpoint markers
        for px, py in (ann.p1, ann.p2):
            dot = self.ax.plot(px, py, "o", color=ann.color, ms=4, zorder=6)[0]
            self._staging.append(dot)

    def _angle(self, ann) -> None:
        """Two rays from vertex with arc and label."""
        v = ann.vertex
        for pt in (ann.p1, ann.p2):
            ln = Line2D([v[0], pt[0]], [v[1], pt[1]],
                        color=ann.color, lw=ann.linewidth, zorder=5, pickradius=5)
            self.ax.add_line(ln)
            self._staging.append(ln)
        # arc
        r_arc = min(
            math.hypot(ann.p1[0]-v[0], ann.p1[1]-v[1]),
            math.hypot(ann.p2[0]-v[0], ann.p2[1]-v[1]),
        ) * 0.3
        a1 = math.degrees(math.atan2(-(ann.p1[1]-v[1]), ann.p1[0]-v[0]))
        a2 = math.degrees(math.atan2(-(ann.p2[1]-v[1]), ann.p2[0]-v[0]))
        from matplotlib.patches import Arc
        arc = Arc(v, 2*r_arc, 2*r_arc, angle=0,
                  theta1=min(a1, a2), theta2=max(a1, a2),
                  color=ann.color, lw=ann.linewidth, zorder=5)
        self.ax.add_patch(arc)
        self._staging.append(arc)
        # label
        mid_ang = math.radians((a1 + a2) / 2)
        lx = v[0] + r_arc * 1.5 * math.cos(mid_ang)
        ly = v[1] - r_arc * 1.5 * math.sin(mid_ang)
        txt = self.ax.text(
            lx, ly, f"{ann.angle_deg:.1f}°",
            color=ann.color, fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="black", alpha=0.55),
            zorder=7,
        )
        self._staging.append(txt)

    def _roi(self, ann) -> None:
        if len(ann.vertices) < 3:
            return
        xy = np.array(ann.vertices)
        fill = Polygon(
            xy, closed=True,
            fill=True, facecolor=ann.color, alpha=0.20,
            edgecolor="none", zorder=5,
        )
        self.ax.add_patch(fill)
        self._staging.append(fill)
        outline = Polygon(
            xy, closed=True,
            fill=False, edgecolor=ann.color, alpha=1.0,
            lw=ann.linewidth, linestyle="--", zorder=5,
        )
        self.ax.add_patch(outline)
        self._staging.append(outline)
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        if ann.label:
            centroid_text = ann.label
        else:
            centroid_text = f"A={ann.area_nm2:.0f} nm²"
            if ann.stats:
                centroid_text += f"\nμ={ann.stats.get('mean', 0):.3f}"
        txt = self.ax.text(
            cx, cy, centroid_text,
            color=ann.color, fontsize=9, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.55),
            zorder=7,
        )
        self._staging.append(txt)
