"""
CryoCanvas — backend-agnostic matplotlib Figure wrapper.

This module never imports PyQt6. The GUI backend ("QtAgg") must be set
*before* importing this module when running in GUI mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from acorn.core.contrast import apply_contrast, ContrastParams, _bp_cache
from acorn.core.annotations import AnnotationStore
from acorn.render.annotation_renderer import AnnotationRenderer
from acorn.render.scalebar import nice_scalebar_nm

# Maximum pixels per side passed to imshow.  Full-res is kept in self._norm for
# export / intensity readout; only the display copy is shrunk.
# The figure canvas is 800px at default 100 DPI; 1024 is a 1.28x oversample
# which is sharper than needed while cutting draw() time from ~470ms to ~150ms.
_DISPLAY_MAX_DIM = 1024


def _make_display_array(norm: np.ndarray) -> np.ndarray:
    """Stride-downsample *norm* so neither dimension exceeds _DISPLAY_MAX_DIM.

    The result is float32 (halves bandwidth vs float64).  set_extent() keeps
    the data-coordinate axes in full-image pixels, so annotations are unaffected.
    """
    h, w = norm.shape[:2]
    step = max(1, (max(h, w) + _DISPLAY_MAX_DIM - 1) // _DISPLAY_MAX_DIM)
    out = norm[::step, ::step]
    return out.astype(np.float32, copy=False)

if TYPE_CHECKING:
    from acorn.core.dm4_loader import DM4Image


class CryoCanvas:
    """
    Owns a matplotlib Figure and exposes a clean API for loading cryo-EM images,
    applying contrast, managing annotations, and saving.

    Works with *any* matplotlib backend — the GUI simply embeds ``self.fig``
    into a ``FigureCanvasQtAgg``; the CLI uses the Agg backend.

    Parameters
    ----------
    figsize : (width, height) in inches
    """

    def __init__(self, figsize: tuple[float, float] = (8, 8)) -> None:
        self.fig = Figure(figsize=figsize, facecolor="black")
        self.fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("black")
        self.ax.axis("off")

        self._dm4: Optional["DM4Image"] = None
        self._norm: Optional[np.ndarray] = None
        self._img_artist = self.ax.imshow(
            np.zeros((512, 512)), cmap="gray", origin="upper",
            interpolation="nearest", vmin=0, vmax=1,
        )

        self.store = AnnotationStore()
        self.renderer: Optional[AnnotationRenderer] = None
        self._last_store_len: int = 0
        self._loading: bool = False   # suppresses _on_store_change during image switch
        self._bg_cache = None         # blit background (image without annotations)
        self._overlay_artists: list = []  # persistent transient artists (e.g. SAM point markers)

        self._splash_artists: list = []   # cleared by load_image()

        # register store change callback
        self.store.on_change(self._on_store_change)
        # invalidate blit cache on resize so stale bbox coords don't cause bad blits
        self.fig.canvas.mpl_connect(
            "resize_event", lambda _e: setattr(self, "_bg_cache", None)
        )

        self._show_splash()

    # ── image loading ─────────────────────────────────────────────────────────

    def load_image(
        self,
        dm4img: "DM4Image",
        params: Optional[ContrastParams] = None,
        precomputed_norm=None,
    ) -> None:
        """
        Load a DM4Image into the canvas and apply contrast normalisation.

        Parameters
        ----------
        precomputed_norm : optional pre-normalised float32 array produced by
            apply_contrast() on a background thread.  When supplied, the
            (potentially expensive) apply_contrast call is skipped on the
            main thread, keeping the GUI responsive.
        """
        self._clear_splash()
        _bp_cache.clear()   # new image — invalidate bandpass background cache
        self._dm4 = dm4img
        if self.renderer is not None:
            self.renderer.clear()   # remove old artists from axes before replacing renderer
        self.renderer = AnnotationRenderer(self.ax, pixel_size=dm4img.pixel_size)

        # Fit axes to image
        h, w = dm4img.shape[:2]
        self._img_artist.set_extent([-0.5, w - 0.5, h - 0.5, -0.5])
        self.ax.set_xlim(-0.5, w - 0.5)
        self.ax.set_ylim(h - 0.5, -0.5)
        self.ax.set_title(dm4img.filename, color="white", fontsize=9, pad=3)

        if params is None:
            params = ContrastParams()
        self.update_contrast(params, precomputed_norm=precomputed_norm)

    def update_contrast(self, params: ContrastParams, precomputed_norm=None) -> None:
        """Re-apply contrast and colormap, then redraw annotations.

        Parameters
        ----------
        precomputed_norm : if supplied (e.g. computed on a background thread),
            apply_contrast is skipped and this array is used directly.
        """
        if self._dm4 is None or self._dm4.raw is None:
            return
        if self._dm4.is_color:
            # Color (H, W, 3) image — display as-is; derive grayscale luminance for _norm
            rgb = np.clip(self._dm4.raw, 0.0, 1.0)
            self._norm = (rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722)
            self._img_artist.set_data(_make_display_array(rgb))
        else:
            self._norm = precomputed_norm if precomputed_norm is not None else apply_contrast(self._dm4.raw, params)
            self._img_artist.set_data(_make_display_array(self._norm))
            self._img_artist.set_clim(0, 1)
            self._img_artist.set_cmap(params.colormap)
        self._bg_cache = None          # image changed — invalidate blit cache
        if self.renderer:
            self.renderer.render_noblit(self.store, self)   # full draw + cache bg
        else:
            self.fig.canvas.draw_idle()

    # ── annotation convenience ────────────────────────────────────────────────

    def add_default_scalebar(self, color: str = "#FFFFFF") -> None:
        """Add an auto-sized scale bar at the bottom-left corner."""
        if self._dm4 is None:
            return
        from acorn.core.annotations import ScalebarAnnotation
        ps = self._dm4.pixel_size
        w = self._dm4.shape[1] if self._dm4.shape else 512
        nm = nice_scalebar_nm(ps, w)
        self.store.add(ScalebarAnnotation(nm=nm, x_frac=0.03, y_frac=0.93, color=color))

    # ── pixel size ────────────────────────────────────────────────────────────

    def set_pixel_size(self, pixel_size_nm: float) -> None:
        """Update pixel size (nm/px) on the live image without reloading.

        Updates the DM4Image metadata, annotation renderer, and re-renders
        all annotations so measurements and scale bars reflect the new value.
        """
        if self._dm4 is None:
            return
        self._dm4.meta.pixel_size = pixel_size_nm
        if self.renderer is not None:
            self.renderer.pixel_size = pixel_size_nm
            self.renderer.render_noblit(self.store, self)

    # ── blit helpers ──────────────────────────────────────────────────────────

    def _save_bg(self) -> None:
        """Capture the canvas state (image, no annotations) for blitting."""
        try:
            self._bg_cache = self.fig.canvas.copy_from_bbox(self.ax.bbox)
        except Exception:
            self._bg_cache = None

    def blit_annotations(self) -> None:
        """Redraw annotation artists only, without re-rendering the image.

        Falls back to draw_idle() if the background cache is not available.
        """
        if self._bg_cache is None or self.renderer is None:
            self.fig.canvas.draw_idle()
            return
        try:
            self.fig.canvas.restore_region(self._bg_cache)
            for artists in self.renderer._ann_to_artists.values():
                for art in artists:
                    self.ax.draw_artist(art)
            for art in self.renderer._selection_artists:
                self.ax.draw_artist(art)
            for art in self._overlay_artists:
                self.ax.draw_artist(art)
            self.fig.canvas.blit(self.ax.bbox)
        except Exception:
            # Blit not supported by this backend (e.g. Agg headless) — fall back
            self._bg_cache = None
            self.fig.canvas.draw_idle()

    # ── internal callbacks ────────────────────────────────────────────────────

    def _on_store_change(self, items) -> None:
        n = len(items)
        if self._loading:
            self._last_store_len = n
            return
        if self.renderer is not None:
            if n == self._last_store_len + 1:
                # single item appended — incremental add, blit only
                self.renderer.add_one_blit(items[-1], self)
            elif n == self._last_store_len - 1 and self._bg_cache is not None:
                # single item removed (undo/delete) — fast path: remove artists + blit,
                # avoiding the 400 ms full draw() that render_noblit would trigger.
                current_ids = {id(ann) for ann in items}
                for ann_id in list(self.renderer._ann_to_artists):
                    if ann_id not in current_ids:
                        stale = self.renderer._id_to_ann.get(ann_id)
                        if stale is not None:
                            self.renderer.remove_one(stale)
                        else:
                            self.renderer._ann_to_artists.pop(ann_id, None)
                        break
                self.blit_annotations()
            else:
                self.renderer.render_noblit(self.store, self)
        else:
            self.fig.canvas.draw_idle()
        self._last_store_len = n

    # ── splash screen ─────────────────────────────────────────────────────────

    def _clear_splash(self) -> None:
        for a in self._splash_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._splash_artists.clear()
        self._img_artist.set_visible(True)

    def _show_splash(self) -> None:
        """Draw the ACORN branding placeholder in the empty canvas."""
        from matplotlib.patches import Ellipse, FancyBboxPatch
        import matplotlib.lines as mlines

        self._img_artist.set_visible(False)

        bg  = "#1a1a1a"
        ink = "#00703C"
        lw  = 1.6

        self.fig.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)

        def _add(artist):
            # For patches/lines not yet in any axes container
            self.ax.add_artist(artist)
            self._splash_artists.append(artist)
            return artist

        def _add_text(artist):
            # ax.text() already registers in ax.texts — just track for cleanup
            self._splash_artists.append(artist)
            return artist

        cx = 0.5

        # ── icon: clean outlines only, uniform stroke ─────────────────────────

        # Body — tall oval, bg fill so it occludes behind shapes
        _add(Ellipse((cx, 0.530), 0.142, 0.215,
                     facecolor=bg, edgecolor=ink, linewidth=lw, zorder=3))

        # Cap — wider, flatter ellipse sitting on body
        cap_cy = 0.643
        _add(Ellipse((cx, cap_cy), 0.196, 0.050,
                     facecolor=bg, edgecolor=ink, linewidth=lw, zorder=4))

        # Stem — thin upright rect outline
        stem_x, stem_y = cx - 0.008, cap_cy + 0.022
        _add(FancyBboxPatch((stem_x, stem_y), 0.016, 0.030,
                            boxstyle="square,pad=0",
                            facecolor=bg, edgecolor=ink,
                            linewidth=lw - 0.3, zorder=5))

        # Leaf — small rotated ellipse off the stem tip
        _add(Ellipse((cx + 0.040, stem_y + 0.022), 0.052, 0.018,
                     angle=28, facecolor=bg, edgecolor=ink,
                     linewidth=lw - 0.3, zorder=5))

        # ── typography ────────────────────────────────────────────────────────
        _add_text(self.ax.text(
            0.5, 0.380, "ACORN",
            ha="center", va="center", fontsize=34, fontweight="bold",
            color="#4dbb78", fontfamily="monospace",
            transform=self.ax.transAxes, zorder=10,
        ))

        rule = mlines.Line2D(
            [0.32, 0.68], [0.342, 0.342],
            transform=self.ax.transAxes,
            color="#363636", linewidth=0.8, zorder=9,
        )
        self.ax.add_artist(rule)
        self._splash_artists.append(rule)

        _add_text(self.ax.text(
            0.5, 0.319,
            "Annotate  \u00b7  Curate  \u00b7  Observe  \u00b7  Review  \u00b7  Navigate",
            ha="center", va="center", fontsize=9,
            color="#888888",
            transform=self.ax.transAxes, zorder=10,
        ))

        _add_text(self.ax.text(
            0.5, 0.260, "Open file  \u2014  Ctrl+O",
            ha="center", va="center", fontsize=8.5,
            color="#888888",
            transform=self.ax.transAxes, zorder=10,
        ))

        # Branding — bottom-right corner
        # e^- = electron symbol, \AA = angstrom symbol
        _add_text(self.ax.text(
            0.97, 0.032,
            r"an $e^{-}$MM$\AA$ designed software",
            ha="right", va="bottom", fontsize=7.5,
            color="#555555",
            transform=self.ax.transAxes, zorder=10,
        ))

        self.fig.canvas.draw_idle()

    # ── export ────────────────────────────────────────────────────────────────

    def save(
        self,
        output_path: str | Path,
        dpi: int = 300,
        fmt: Optional[str] = None,
    ) -> Path:
        """
        Save the current view (image + annotations) to *output_path*.

        Parameters
        ----------
        output_path : destination file; format inferred from extension unless *fmt* given
        dpi         : resolution in dots per inch (use 300–600 for publication)
        fmt         : override format string passed to fig.savefig (e.g. "png", "svg")

        Returns
        -------
        Resolved Path of the saved file.
        """
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        kwargs: dict = dict(
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0,
            facecolor="black",
        )
        if fmt:
            kwargs["format"] = fmt
        self.fig.savefig(str(out), **kwargs)
        return out

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def norm_image(self) -> Optional[np.ndarray]:
        """The currently displayed normalised float64 image array."""
        return self._norm

    @property
    def dm4(self) -> Optional["DM4Image"]:
        return self._dm4

    def close(self) -> None:
        plt.close(self.fig)
