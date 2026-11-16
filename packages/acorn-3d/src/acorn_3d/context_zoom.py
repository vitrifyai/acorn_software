"""
Context-zoom movie: whole cell on the low-magnification search map, zoom into the
tomogram footprint, dissolve into the tomogram slab, then draw the traced ROIs.

Contrast is re-stretched on every frame from the pixels currently in view and the
smoothing is scaled with the zoom, so the cell reads at the wide view and the
fibrils emerge as the view tightens.

CLI:
    python -m acorn_3d.context_zoom --search Position_2_Search.mrc --footprint 1623 1921 \
        --volume rec.mrc --z 275 --slab 15 --sidecar .rec.acorn3d.json --out zoom.mp4
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

from acorn_3d.render import STYLES, _nice_length, write_movie


# ── helpers ──────────────────────────────────────────────────────────────────

def stretch(a: np.ndarray, lo_p=0.5, hi_p=99.5) -> np.ndarray:
    lo, hi = np.percentile(a, [lo_p, hi_p])
    return np.clip((a - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)


def crop_view(img: np.ndarray, cx: float, cy: float, half: float, out: int, sigma_px: float) -> np.ndarray:
    """Square crop of half-size `half` (px) around (cx, cy), smoothed and stretched, resized to out x out."""
    h, w = img.shape
    x0, x1 = int(round(cx - half)), int(round(cx + half))
    y0, y1 = int(round(cy - half)), int(round(cy + half))
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    sub = img[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if any(pad):
        sub = cv2.copyMakeBorder(sub, *pad, cv2.BORDER_CONSTANT, value=float(np.median(sub)))
    # bin first when the crop is much larger than the output, then smooth in output pixels
    m = min(sub.shape); sub = sub[:m, :m]           # rounding can leave the crop one pixel off square
    f = max(1, int(m // (out * 1.5)))
    if f > 1:
        s = (m // f) * f
        sub = sub[:s, :s].reshape(s // f, f, s // f, f).mean((1, 3))
    sub = cv2.resize(sub.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA)
    if sigma_px > 0:
        sub = gaussian_filter(sub, sigma_px)
    return stretch(sub)


def letterbox(gray_or_bgr: np.ndarray, size: tuple[int, int], bg) -> tuple[np.ndarray, int, int]:
    W, H = size
    img = gray_or_bgr if gray_or_bgr.ndim == 3 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    s = min(W / w, H / h); nw, nh = int(w * s), int(h * s)
    frame = np.empty((H, W, 3), np.uint8); frame[:] = [int(255 * c) for c in bg[::-1]]
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    frame[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    return frame, x0, y0


def dashed_rect(img, x0, y0, x1, y1, col, thick=3, n=24):
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0)]
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        for i in range(0, n, 2):
            cv2.line(img, (int(ax + (bx - ax) * i / n), int(ay + (by - ay) * i / n)),
                     (int(ax + (bx - ax) * (i + 1) / n), int(ay + (by - ay) * (i + 1) / n)), col, thick, cv2.LINE_AA)


def overlay_text(frame, style, caption: Optional[str], scale_nm_per_px: float, field_nm: float, size):
    W, H = size; font = cv2.FONT_HERSHEY_DUPLEX; fs = 1.0 * W / 1920
    target = _nice_length(field_nm / 4); px = int(round(target / scale_nm_per_px))
    x0, y0 = W - px - int(0.03 * W), H - int(0.07 * H)
    cv2.rectangle(frame, (x0, y0), (x0 + px, y0 + max(4, int(0.008 * H))), style.text, -1)
    txt = f"{target:g} nm" if target < 1000 else f"{target / 1000:g} um"
    (tw, th), _ = cv2.getTextSize(txt, font, fs, 2)
    cv2.putText(frame, txt, (x0 + px // 2 - tw // 2, y0 - int(0.012 * H)), font, fs, style.text, 2, cv2.LINE_AA)
    if caption:
        (tw, th), _ = cv2.getTextSize(caption, font, fs * 0.95, 2)
        cv2.putText(frame, caption, (W // 2 - tw // 2, H - int(0.03 * H)), font, fs * 0.95, style.text, 2, cv2.LINE_AA)


def ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * min(1.0, max(0.0, t)))


# ── ROI overlays from the acorn_3d sidecar ───────────────────────────────────

def load_rois(sidecar: Optional[Path], z: int, z_tol: int = 0) -> list[tuple[str, np.ndarray]]:
    """(label, Nx2 vertices in volume px) for ROIs on slice z (or within z_tol slices)."""
    if sidecar is None or not Path(sidecar).exists():
        return []
    d = json.loads(Path(sidecar).read_text())
    out = []
    for zs, anns in d.get("slices", {}).items():
        if abs(int(zs) - z) > z_tol:
            continue
        for a in anns:
            if a.get("type", "roi") == "roi" and len(a.get("vertices", [])) >= 3:
                out.append((a.get("label", "") or "Unlabelled", np.array(a["vertices"], float)))
    return out


# ── movie ────────────────────────────────────────────────────────────────────

def context_zoom_frames(search: np.ndarray, search_nm_px: float, footprint_cxy: tuple[float, float],
                        field_nm: float, slab_img: np.ndarray, slab_nm_px: float,
                        rois: Sequence[tuple[str, np.ndarray]], class_colours: dict[str, tuple[int, int, int]],
                        style_name: str = "talk", size=(1920, 1080), fps: int = 30,
                        t_hold=3.5, t_zoom=9.0, t_fade=2.5, t_slab=3.5, t_trace=5.0, t_end=4.0,
                        caption_wide="E. coli cell on the grid (search map)", caption_slab="tomogram slab, 15 slices",
                        caption_trace="traced components"):
    style = STYLES[style_name]
    W, H = size; out = min(H, 1080)
    cx, cy = footprint_cxy
    half_wide = min(search.shape) / 2.0
    half_field = field_nm / 2.0 / search_nm_px
    col_box = (60, 255, 60)

    mcx, mcy = search.shape[1] / 2.0, search.shape[0] / 2.0     # map centre: the wide view starts here

    def search_frame(half, sigma, draw_box=True, caption=None, ccx=None, ccy=None):
        ccx = cx if ccx is None else ccx; ccy = cy if ccy is None else ccy
        view = crop_view(search, ccx, ccy, half, out, sigma)
        frame, ox, oy = letterbox(view, size, style.bg)
        s = out / (2 * half)                      # output px per search px
        if draw_box:
            fx, fy = ox + out / 2 + (cx - ccx) * s, oy + out / 2 + (cy - ccy) * s   # footprint centre on screen
            bx0 = int(fx - half_field * s); bx1 = int(fx + half_field * s)
            by0 = int(fy - half_field * s); by1 = int(fy + half_field * s)
            if bx1 - bx0 < out - 4:
                dashed_rect(frame, bx0, by0, bx1, by1, col_box)
        overlay_text(frame, style, caption, search_nm_px / s, 2 * half * search_nm_px, size)
        return frame

    # 1. hold on the wide view (binned + smoothed so the cell reads)
    n = int(t_hold * fps)
    for i in range(n):
        yield search_frame(half_wide, 1.2, caption=caption_wide, ccx=mcx, ccy=mcy)
    # 2. zoom: window shrinks from the whole map to 1.6x the footprint while the centre pans onto it;
    #    smoothing eases off as detail appears
    n = int(t_zoom * fps)
    half_end = half_field * 1.6
    for i in range(n):
        t = ease(i / max(1, n - 1))
        half = math.exp(math.log(half_wide) * (1 - t) + math.log(half_end) * t)
        sigma = 1.2 * (1 - t) + 0.6 * t
        yield search_frame(half, sigma, caption=None if t > 0.5 else caption_wide,
                           ccx=mcx + (cx - mcx) * t, ccy=mcy + (cy - mcy) * t)
    # 3. dissolve: search crop at the footprint -> tomogram slab
    slab_view = stretch(gaussian_filter(cv2.resize(slab_img.astype(np.float32), (out, out), interpolation=cv2.INTER_AREA), 0.8))
    end_search = crop_view(search, cx, cy, half_field, out, 0.6)
    n = int(t_fade * fps)
    for i in range(n):
        t = ease(i / max(1, n - 1))
        mix = cv2.addWeighted(end_search, 1 - t, slab_view, t, 0)
        frame, ox, oy = letterbox(mix, size, style.bg)
        overlay_text(frame, style, caption_slab if t > 0.5 else None, field_nm / out, field_nm, size)
        yield frame
    # 4. hold the slab, then draw the traced ROIs progressively
    def slab_frame(progress: float, caption):
        rgb = cv2.cvtColor(slab_view, cv2.COLOR_GRAY2BGR)
        s = out / slab_img.shape[1]
        for label, verts in rois:
            col = class_colours.get(label.lower(), (200, 200, 200))
            pts = (verts * s).astype(np.int32)
            k = max(2, int(len(pts) * progress)) if progress < 1 else len(pts)
            cv2.polylines(rgb, [pts[:k]], progress >= 1, col, 3, cv2.LINE_AA)
        frame, ox, oy = letterbox(rgb, size, style.bg)
        overlay_text(frame, style, caption, field_nm / out, field_nm, size)
        if rois:
            y = int(0.06 * H); x = int(0.025 * W); sw = int(0.03 * H); fs = 1.0 * W / 1920
            seen = []
            for label, _ in rois:
                if label.lower() in seen: continue
                seen.append(label.lower()); col = class_colours.get(label.lower(), (200, 200, 200))
                cv2.rectangle(frame, (x, y - sw // 2), (x + sw, y + sw // 2), col, -1)
                cv2.putText(frame, label, (x + sw + 12, y + sw // 3), cv2.FONT_HERSHEY_DUPLEX, fs, style.text, 2, cv2.LINE_AA)
                y += int(sw * 1.7)
        return frame
    for i in range(int(t_slab * fps)):
        yield slab_frame(0.0, caption_slab)
    n = int(t_trace * fps)
    for i in range(n):
        yield slab_frame(ease(i / max(1, n - 1)), caption_trace)
    for i in range(int(t_end * fps)):
        yield slab_frame(1.0, caption_trace)


def main(argv=None):
    import mrcfile
    from acorn_3d.volume_loader import VolumeImage
    from acorn.render.palette import color_for_label
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", required=True, help="Tomo5 Position_N_Search.mrc")
    ap.add_argument("--footprint", nargs=2, type=float, required=True, metavar=("CX", "CY"), help="tomogram field centre in search-map px")
    ap.add_argument("--volume", required=True); ap.add_argument("--z", type=int, required=True); ap.add_argument("--slab", type=int, default=15)
    ap.add_argument("--sidecar", default=None, help=".<stem>.acorn3d.json with the traced ROIs (default: next to the volume)")
    ap.add_argument("--z-tol", type=int, default=3, help="also draw ROIs from slices within this many of --z")
    ap.add_argument("--out", required=True); ap.add_argument("--style", choices=list(STYLES), default="talk")
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--speed", type=float, default=1.0, help="<1 slows every phase down (0.5 = twice as long)")
    a = ap.parse_args(argv)

    with mrcfile.open(a.search, permissive=True) as m:
        search = np.asarray(m.data, dtype=np.float32).squeeze(); s_nm = float(m.voxel_size.x) / 10
    v = VolumeImage.from_file(a.volume); v.slab = a.slab; v.set_slice(a.z); slab = np.asarray(v.raw, dtype=np.float32)
    field_nm = v.shape[1] * v.pixel_size
    sidecar = Path(a.sidecar) if a.sidecar else Path(a.volume).with_name(f".{Path(a.volume).stem}.acorn3d.json")
    rois = load_rois(sidecar, a.z, a.z_tol)
    colours = {}
    for label, _ in rois:
        hx = color_for_label(label).lstrip("#"); colours[label.lower()] = (int(hx[4:6], 16), int(hx[2:4], 16), int(hx[0:2], 16))
    W, Hh = (int(x) for x in a.size.lower().split("x"))
    frames = context_zoom_frames(search, s_nm, tuple(a.footprint), field_nm, slab, v.pixel_size, rois, colours,
                                 style_name=a.style, size=(W, Hh), fps=a.fps,
                                 t_hold=3.5 / a.speed, t_zoom=9.0 / a.speed, t_fade=2.5 / a.speed,
                                 t_slab=3.5 / a.speed, t_trace=5.0 / a.speed, t_end=4.0 / a.speed)
    write_movie(frames, Path(a.out), fps=a.fps, gif_path=None, progress=print)
    print("wrote", a.out, "| ROIs drawn:", len(rois), "| field nm:", round(field_nm))


if __name__ == "__main__":
    main()
