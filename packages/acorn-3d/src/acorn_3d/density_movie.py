"""
Movies of the tomogram itself: a z scroll through averaged slabs and a turntable
of the grayscale density, optionally with label outlines / tinted labels on top.

This is the conventional way to show cryo-ET data and does not depend on any
segmentation being right.  Rendering is done with torch (GPU if available):
the volume is resampled on a rotated grid and integrated along the view axis.

CLI:
    python -m acorn_3d.density_movie rec.mrc out_prefix [--labels labels.mrc --classes "1:cell,2:CNC"]
        [--scroll-seconds 8 --spin-seconds 10 --slab 15 --bin 2 --style talk]
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from acorn_3d.render import STYLES, TALK_COLOURS, PAPER_COLOURS, _nice_length, write_movie


# ── helpers ──────────────────────────────────────────────────────────────────

def _bin(vol: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return vol
    z, y, x = (s // b * b for s in vol.shape)
    v = vol[:z, :y, :x]
    return v.reshape(z // b, b, y // b, b, x // b, b).mean((1, 3, 5))


def _bin_labels(lab: np.ndarray, b: int) -> np.ndarray:
    if b <= 1:
        return lab
    z, y, x = (s // b * b for s in lab.shape)
    return lab[:z:b, :y:b, :x:b]


def normalise(vol: np.ndarray, zsmooth: int = 1, percentiles=(0.5, 99.5)) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    v = vol.astype(np.float32)
    if zsmooth > 1:
        v = uniform_filter1d(v, zsmooth, axis=0, mode="nearest")
    lo, hi = np.percentile(v[::4], percentiles)
    return np.clip((v - lo) / (hi - lo + 1e-6), 0, 1)


def _scale_bar(img: np.ndarray, nm_per_px: float, extent_nm: float, style, font_scale: float):
    h, w = img.shape[:2]
    target = _nice_length(extent_nm / 4)
    px = int(round(target / nm_per_px))
    x0, y0 = w - px - int(0.03 * w), h - int(0.07 * h)
    cv2.rectangle(img, (x0, y0), (x0 + px, y0 + max(4, int(0.008 * h))), style.text, -1)
    txt = f"{target:g} nm" if target < 1000 else f"{target / 1000:g} um"
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, font_scale, 2)
    cv2.putText(img, txt, (x0 + px // 2 - tw // 2, y0 - int(0.012 * h)), cv2.FONT_HERSHEY_DUPLEX, font_scale, style.text, 2, cv2.LINE_AA)


def _legend(img: np.ndarray, classes: dict[int, str], colours: dict, style, font_scale: float):
    h, w = img.shape[:2]
    y = int(0.06 * h); x = int(0.025 * w); sw = int(0.032 * h)
    for cid, name in sorted(classes.items()):
        col = tuple(int(255 * c) for c in colours.get(cid, (0.7, 0.7, 0.7))[::-1])
        cv2.rectangle(img, (x, y - sw // 2), (x + sw, y + sw // 2), col, -1)
        cv2.putText(img, name, (x + sw + int(0.008 * w), y + sw // 3), cv2.FONT_HERSHEY_DUPLEX, font_scale, style.text, 2, cv2.LINE_AA)
        y += int(sw * 1.7)


def _fit(img: np.ndarray, size: tuple[int, int], bg) -> np.ndarray:
    """Letterbox an image into size (w, h)."""
    W, H = size
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.empty((H, W, 3), np.uint8); out[:] = [int(255 * c) for c in bg[::-1]]
    y0, x0 = (H - nh) // 2, (W - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = r
    return out, s, (x0, y0)


def tint(gray: np.ndarray, lab: Optional[np.ndarray], colours: dict, alpha: float = 0.45, outline: bool = True) -> np.ndarray:
    """Grayscale [0,1] HxW + label HxW -> BGR uint8 with tinted labels and outlines."""
    g = (gray * 255).astype(np.uint8)
    rgb = np.stack([g, g, g], -1).astype(np.float32)
    if lab is not None:
        for cid in np.unique(lab):
            if cid == 0:
                continue
            col = np.array([255 * c for c in colours.get(int(cid), (0.7, 0.7, 0.7))[::-1]], np.float32)
            m = lab == cid
            rgb[m] = (1 - alpha) * rgb[m] + alpha * col
            if outline:
                edge = m ^ cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                rgb[edge] = col
    return np.clip(rgb, 0, 255).astype(np.uint8)


# ── z scroll ─────────────────────────────────────────────────────────────────

def scroll_frames(vol01: np.ndarray, voxel_nm: Sequence[float], seconds: float, fps: int, size,
                  labels: Optional[np.ndarray], classes: dict[int, str], style_name: str,
                  slab: int = 1, title: Optional[str] = None, z_range: Optional[tuple[int, int]] = None):
    style = STYLES[style_name]; colours = style.colours
    nz, ny, nx = vol01.shape
    z0, z1 = z_range if z_range else (0, nz - 1)
    n = int(seconds * fps)
    font = 1.0 * size[0] / 1920
    for i in range(n):
        z = int(round(z0 + (z1 - z0) * i / max(1, n - 1)))
        a, b = max(0, z - slab // 2), min(nz, z + slab // 2 + 1)
        g = vol01[a:b].mean(0)
        lab = labels[z] if labels is not None else None
        img = tint(g, lab, colours)
        frame, s, (ox, oy) = _fit(img, size, style.bg)
        _scale_bar(frame, voxel_nm[2] / s, nx * voxel_nm[2], style, font)
        if classes:
            _legend(frame, classes, colours, style, font)
        depth = f"z = {z * voxel_nm[0]:.0f} nm  (slice {z}/{nz - 1})"
        cv2.putText(frame, depth, (int(0.025 * size[0]), size[1] - int(0.03 * size[1])), cv2.FONT_HERSHEY_DUPLEX, font, style.text, 2, cv2.LINE_AA)
        if title:
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font * 0.9, 2)
            cv2.putText(frame, title, (size[0] // 2 - tw // 2, size[1] - int(0.03 * size[1])), cv2.FONT_HERSHEY_DUPLEX, font * 0.9, style.text, 2, cv2.LINE_AA)
        yield frame


# ── density turntable ────────────────────────────────────────────────────────

class DensitySpinner:
    """
    Resamples the volume on a camera-aligned grid for each azimuth and integrates
    along the view direction (mean of the densest fraction, so dark = dense material
    reads as bright structure on a dark background, or inverted for paper style).
    """

    def __init__(self, vol01: np.ndarray, voxel_nm: Sequence[float], labels: Optional[np.ndarray],
                 colours: dict, elevation: float = 35.0, device: Optional[str] = None,
                 out_px: int = 720, invert: bool = True, smooth_sigma: float = 2.0, threshold_pct: float = 93.0,
                 opacity: float = 0.08):
        self.opacity = opacity
        import torch
        self.torch = torch
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        vz, vy, vx = voxel_nm
        nz, ny, nx = vol01.shape
        # density: 1 = dense (dark in the tomogram). Smooth hard and keep only the dense tail so the
        # membranes and fibrils stand out of the back-projection noise instead of a uniform haze.
        from scipy.ndimage import gaussian_filter
        dens = 1.0 - vol01 if invert else vol01
        dens = gaussian_filter(dens.astype(np.float32), smooth_sigma)
        # the reconstruction border (shifted-stack edges, top/bottom of the slab) is bright noise: mask it
        bz, by, bx = int(0.05 * nz), int(0.04 * ny), int(0.04 * nx)
        mask = np.zeros_like(dens); mask[bz:nz - bz, by:ny - by, bx:nx - bx] = 1.0
        dens = dens * gaussian_filter(mask, 2.0)
        t = np.percentile(dens[::3], threshold_pct)
        hi = np.percentile(dens[::3], 99.95)
        dens = np.clip((dens - t) / (hi - t + 1e-6), 0, 1)
        self.vol = torch.from_numpy(np.ascontiguousarray(dens, dtype=np.float32)).to(self.dev)[None, None]   # 1,1,Z,Y,X
        self.lab = None if labels is None else torch.from_numpy(labels.astype(np.float32)).to(self.dev)[None, None]
        self.colours = colours
        self.ext = np.array([nx * vx, ny * vy, nz * vz], float)     # x, y, z in nm
        self.elev = math.radians(elevation)
        self.out_px = out_px
        diag = float(np.linalg.norm(self.ext)) * 1.02
        self.nm_per_px = diag / out_px
        # camera grid: u (right), v (up) in the image plane, w along the view ray
        n_w = int(diag / self.nm_per_px)
        u = torch.linspace(-diag / 2, diag / 2, out_px, device=self.dev)
        v = torch.linspace(-diag / 2, diag / 2, out_px, device=self.dev)
        w = torch.linspace(-diag / 2, diag / 2, n_w, device=self.dev)
        self.W, self.V, self.U = torch.meshgrid(w, v, u, indexing="ij")   # n_w, out, out

    def _sample(self, az_deg: float, tensor):
        t = self.torch
        az = math.radians(az_deg)
        # camera basis in world coordinates (x, y, z): view from elevation `elev` above the xy plane
        ce, se, ca, sa = math.cos(self.elev), math.sin(self.elev), math.cos(az), math.sin(az)
        right = t.tensor([ca, sa, 0.0], device=self.dev)
        up = t.tensor([-sa * se, ca * se, ce], device=self.dev)
        view = t.tensor([sa * ce, -ca * ce, se], device=self.dev)        # from camera into the volume
        P = self.U[..., None] * right + self.V[..., None] * up + self.W[..., None] * view   # n_w,out,out,3 (nm)
        ext = t.tensor(self.ext, device=self.dev, dtype=t.float32)
        grid = (P / (ext / 2)).to(t.float32)[None]                          # 1,n_w,out,out,3 normalised (x,y,z)
        out = t.nn.functional.grid_sample(tensor.to(t.float32), grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return out[0, 0]                                                    # n_w, out, out

    def frame(self, az_deg: float, style, classes: dict[int, str], title: Optional[str] = None,
              size=(1920, 1080), top_fraction: float = 0.08) -> np.ndarray:
        t = self.torch
        with t.no_grad():
            d = self._sample(az_deg, self.vol)                              # n_w, out, out (front to back)
            # front-to-back alpha compositing: dense voxels are emissive and partly opaque
            a = (d * self.opacity).clamp(0, 1)
            trans = t.cumprod(1 - a, dim=0)
            trans = t.cat([t.ones_like(trans[:1]), trans[:-1]], 0)
            img = (a * d * trans).sum(0)
            hi = t.quantile(img.flatten()[::7], 0.995)
            img = (img / (hi + 1e-6)).clamp(0, 1)
            g = (img.cpu().numpy() * 255).astype(np.uint8)
            if style.name == "paper":
                g = 255 - g
            rgb = np.stack([g, g, g], -1)
            if self.lab is not None:
                L = self._sample(az_deg, self.lab)
                front = L.max(0).values                                     # any label along the ray
                lab_img = front.round().cpu().numpy().astype(np.int32)
                rgb = tint(rgb[..., 0].astype(np.float32) / 255, lab_img, self.colours, alpha=0.55)
        frame, s, _ = _fit(rgb, size, style.bg)
        font = 1.0 * size[0] / 1920
        _scale_bar(frame, self.nm_per_px / s, self.ext[0], style, font)
        if classes:
            _legend(frame, classes, self.colours, style, font)
        if title:
            (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_DUPLEX, font * 0.9, 2)
            cv2.putText(frame, title, (size[0] // 2 - tw // 2, size[1] - int(0.03 * size[1])), cv2.FONT_HERSHEY_DUPLEX, font * 0.9, style.text, 2, cv2.LINE_AA)
        return frame


# ── driver ───────────────────────────────────────────────────────────────────

def make_density_movies(vol: np.ndarray, voxel_nm: Sequence[float], out_prefix: Path, labels: Optional[np.ndarray] = None,
                        classes: Optional[dict[int, str]] = None, style: str = "talk", binning: int = 2, slab: int = 15,
                        scroll_seconds: float = 15.0, spin_seconds: float = 20.0, fps: int = 30, size=(1920, 1080),
                        elevation: float = 35.0, title: Optional[str] = None, z_range: Optional[tuple[int, int]] = None,
                        device: Optional[str] = None, progress=None) -> dict:
    classes = classes or {}
    st = STYLES[style]
    vol_b = _bin(vol, binning)
    vox_b = tuple(v * binning for v in voxel_nm)
    v01 = normalise(vol_b, zsmooth=max(1, slab // binning))
    lab_b = _bin_labels(labels, binning) if labels is not None else None
    out_prefix = Path(out_prefix)
    outputs = {}
    if scroll_seconds > 0:
        zr = tuple(int(z / binning) for z in z_range) if z_range else None
        p = out_prefix.with_name(out_prefix.name + f"_scroll_{style}.mp4")
        write_movie(scroll_frames(normalise(vol_b), vox_b, scroll_seconds, fps, size, lab_b, classes, style,
                                  slab=max(1, slab // binning), title=title, z_range=zr), p, fps=fps, gif_path=None, progress=progress)
        outputs["scroll"] = p
    if spin_seconds > 0:
        sp = DensitySpinner(v01, vox_b, lab_b, st.colours, elevation=elevation, device=device, out_px=min(900, size[1]))
        n = int(spin_seconds * fps)
        p = out_prefix.with_name(out_prefix.name + f"_density_{style}.mp4")
        write_movie((sp.frame(360.0 * i / n, st, classes, title, size) for i in range(n)), p, fps=fps,
                    gif_path=p.with_suffix(".gif"), progress=progress)
        still = out_prefix.with_name(out_prefix.name + f"_density_{style}_030deg.png")
        cv2.imwrite(str(still), sp.frame(30.0, st, classes, title, size))
        outputs["spin"] = p; outputs["still"] = still
    return outputs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume"); ap.add_argument("out_prefix")
    ap.add_argument("--labels", default=None); ap.add_argument("--classes", default="")
    ap.add_argument("--style", choices=list(STYLES), default="talk")
    ap.add_argument("--bin", type=int, default=2); ap.add_argument("--slab", type=int, default=15)
    ap.add_argument("--scroll-seconds", type=float, default=15.0); ap.add_argument("--spin-seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30); ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--elevation", type=float, default=35.0); ap.add_argument("--title", default=None)
    ap.add_argument("--z", nargs=2, type=int, default=None); ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    import mrcfile
    from acorn_3d.volume_loader import open_mrc_memmap
    data, vox = open_mrc_memmap(a.volume)
    vol = np.asarray(data, dtype=np.float32)
    voxel_nm = (vox[2] / 10, vox[1] / 10, vox[0] / 10)
    labels = None
    if a.labels:
        with mrcfile.open(a.labels, permissive=True) as m:
            labels = np.asarray(m.data).astype(np.int32)
        if labels.shape != vol.shape:   # label volume made on a z sub-range: pad to the full volume
            full = np.zeros(vol.shape, np.int32); off = (vol.shape[0] - labels.shape[0]) // 2
            full[off:off + labels.shape[0]] = labels; labels = full
    classes = {}
    for part in [c for c in a.classes.split(",") if c.strip()]:
        k, v = part.split(":", 1); classes[int(k)] = v.strip()
    w, h = (int(x) for x in a.size.lower().split("x"))
    res = make_density_movies(vol, voxel_nm, Path(a.out_prefix), labels, classes, style=a.style, binning=a.bin, slab=a.slab,
                              scroll_seconds=a.scroll_seconds, spin_seconds=a.spin_seconds, fps=a.fps, size=(w, h),
                              elevation=a.elevation, title=a.title, z_range=tuple(a.z) if a.z else None, device=a.device,
                              progress=print)
    print(res)


if __name__ == "__main__":
    main()
