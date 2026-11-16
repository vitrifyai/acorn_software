"""
Render a labelled tomogram as a rotating 3D movie.

Input is a label volume (Z, Y, X int array, 0 = background, k = class k) plus the
voxel size.  Each class becomes a smoothed surface (marching cubes), the surfaces
are lit and rendered headlessly with pyrender (EGL), and the frames are written
as an H.264 mp4 (and optionally a GIF and still PNGs).

Two styles: "talk" (dark background, bold colours, large legend) and "paper"
(white background, scale bar, black text).

CLI:
    python -m acorn_3d.render labels.mrc out.mp4 --style talk --classes "1:CNC,2:nanoparticle"
"""
from __future__ import annotations

import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import ndimage
from skimage import measure

import trimesh
import pyrender
import cv2


# ── styles ───────────────────────────────────────────────────────────────────

TALK_COLOURS = {   # bold, distinguishable on a dark background
    1: (0.20, 0.80, 0.95),   # cyan
    2: (1.00, 0.55, 0.10),   # orange
    3: (0.65, 0.95, 0.30),   # lime
    4: (0.95, 0.35, 0.75),   # magenta
    5: (0.95, 0.90, 0.25),   # yellow
}
PAPER_COLOURS = {  # slightly deeper so they hold on white
    1: (0.00, 0.45, 0.70),
    2: (0.90, 0.40, 0.00),
    3: (0.00, 0.60, 0.50),
    4: (0.80, 0.30, 0.60),
    5: (0.60, 0.55, 0.00),
}


@dataclass
class ClassStyle:
    label: str
    colour: tuple[float, float, float]
    alpha: float = 1.0


@dataclass
class RenderStyle:
    name: str
    bg: tuple[float, float, float]
    text: tuple[int, int, int]           # BGR for cv2
    colours: dict[int, tuple[float, float, float]]
    ambient: float
    light: float
    scale_bar: bool
    legend: bool
    font_scale: float
    bbox: bool = False                    # draw the tomogram outline


STYLES = {
    "talk": RenderStyle("talk", (0.07, 0.07, 0.09), (235, 235, 235), TALK_COLOURS, 0.45, 3.0, True, True, 1.1, bbox=True),
    "paper": RenderStyle("paper", (1.0, 1.0, 1.0), (20, 20, 20), PAPER_COLOURS, 0.55, 2.6, True, True, 0.9, bbox=False),
}


# ── meshes ───────────────────────────────────────────────────────────────────

def remove_small(mask: np.ndarray, min_voxels: int) -> np.ndarray:
    if min_voxels <= 1:
        return mask
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_voxels
    return keep[lab]


def label_meshes(
    labels: np.ndarray,
    voxel_nm: Sequence[float] | float,
    class_ids: Optional[Sequence[int]] = None,
    smooth_sigma: float | Sequence[float] = (2.0, 1.0, 1.0),
    step: int = 1,
    min_voxels: int = 30,
    taubin_iters: int = 12,
    progress: Optional[Callable[[str], None]] = None,
) -> dict[int, trimesh.Trimesh]:
    """Marching-cubes surface per class. Vertices are returned in nm, axis order (x, y, z)."""
    if np.isscalar(voxel_nm):
        voxel_nm = (float(voxel_nm),) * 3
    vz, vy, vx = voxel_nm
    if class_ids is None:
        class_ids = [int(c) for c in np.unique(labels) if c != 0]
    meshes: dict[int, trimesh.Trimesh] = {}
    for cid in class_ids:
        if progress:
            progress(f"meshing class {cid}")
        mask = remove_small(labels == cid, min_voxels)
        if mask.sum() < 8:
            continue
        # pad so surfaces close at the volume border
        vol = np.pad(mask.astype(np.float32), 1)
        sig = tuple(smooth_sigma) if not np.isscalar(smooth_sigma) else (float(smooth_sigma),) * 3
        if max(sig) > 0:
            vol = ndimage.gaussian_filter(vol, sig)
        try:
            verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, spacing=(vz, vy, vx), step_size=step)
        except (ValueError, RuntimeError):
            continue
        verts -= np.array([vz, vy, vx])          # undo the pad
        verts = verts[:, ::-1]                    # (z,y,x) -> (x,y,z)
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        if taubin_iters > 0 and len(mesh.faces) > 0:
            trimesh.smoothing.filter_taubin(mesh, iterations=taubin_iters)
        mesh.fix_normals()
        meshes[cid] = mesh
    return meshes


def slice_plane(image: np.ndarray, voxel_nm: Sequence[float], z_nm: float,
                percentiles=(0.5, 99.5), opacity: float = 1.0) -> trimesh.Trimesh:
    """A textured plane showing one grayscale tomogram slice, for spatial context."""
    vz, vy, vx = voxel_nm
    ny, nx = image.shape
    lo, hi = np.percentile(image, percentiles)
    g = np.clip((image.astype(np.float32) - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    tex = np.dstack([g, g, g, np.full_like(g, int(255 * opacity))])
    from PIL import Image
    pil = Image.fromarray(np.flipud(tex), mode="RGBA")
    w, h = nx * vx, ny * vy
    verts = np.array([[0, 0, z_nm], [w, 0, z_nm], [w, h, z_nm], [0, h, z_nm]], np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    plane = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    plane.visual = trimesh.visual.TextureVisuals(uv=uv, image=pil)
    return plane


def bbox_lines(extent_nm: Sequence[float], colour=(0.6, 0.6, 0.6)) -> pyrender.Mesh:
    """Wireframe box outlining the tomogram volume."""
    sx, sy, sz = extent_nm
    c = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0], [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]], np.float32)
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    pos = np.array([c[i] for seg in e for i in seg], np.float32)
    prim = pyrender.Primitive(positions=pos, mode=pyrender.constants.GLTF.LINES,
                              material=pyrender.MetallicRoughnessMaterial(baseColorFactor=[*colour, 1.0], emissiveFactor=colour))
    return pyrender.Mesh(primitives=[prim])


# ── scene / rendering ───────────────────────────────────────────────────────

def _rot_z(deg: float) -> np.ndarray:
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]])


class SpinRenderer:
    """
    Turntable renderer: the tomogram slab lies in the XY plane, the camera looks
    down at `elevation` degrees and the model rotates about the slab normal (Z).
    """

    def __init__(self, meshes: dict[int, trimesh.Trimesh], extent_nm: Sequence[float],
                 style: RenderStyle, class_styles: dict[int, ClassStyle],
                 size=(1920, 1080), elevation: float = 35.0, zoom: float = 1.0,
                 context_plane: Optional[trimesh.Trimesh] = None, z_exaggeration: float = 1.0):
        self.style = style
        self.class_styles = class_styles
        self.size = size
        self.elevation = elevation
        self.extent = np.array(extent_nm, float)
        self.centre = self.extent / 2.0
        self.zexag = z_exaggeration

        self.scene = pyrender.Scene(bg_color=[*style.bg, 1.0], ambient_light=[style.ambient] * 3)
        self.model_node = pyrender.Node(matrix=np.eye(4))
        self.scene.add_node(self.model_node)

        for cid, mesh in meshes.items():
            cs = class_styles.get(cid, ClassStyle(f"class {cid}", style.colours.get(cid, (0.7, 0.7, 0.7))))
            mat = pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[*cs.colour, cs.alpha], metallicFactor=0.05, roughnessFactor=0.6,
                alphaMode="BLEND" if cs.alpha < 1 else "OPAQUE", doubleSided=True)
            m = mesh.copy()
            m.vertices = (m.vertices - self.centre) * np.array([1, 1, self.zexag])
            self.scene.add(pyrender.Mesh.from_trimesh(m, material=mat, smooth=True), parent_node=self.model_node)

        if context_plane is not None:
            p = context_plane.copy()
            p.vertices = (p.vertices - self.centre) * np.array([1, 1, self.zexag])
            pm = pyrender.Mesh.from_trimesh(p, smooth=False)
            for prim in pm.primitives:            # let the surfaces show through the slice
                prim.material.alphaMode = "BLEND"
                prim.material.doubleSided = True
            self.scene.add(pm, parent_node=self.model_node)

        if style.bbox:
            box = bbox_lines(self.extent * np.array([1, 1, self.zexag]), colour=(0.45, 0.45, 0.5))
            self.scene.add(box, parent_node=self.model_node, pose=np.eye(4) @ np.diag([1, 1, 1, 1]) @ _translate(-self.centre * np.array([1, 1, self.zexag])))

        # orthographic camera: fit the box's 3D diagonal (thick volumes need the z extent too)
        ext3 = self.extent * np.array([1, 1, self.zexag])
        diag = float(np.linalg.norm(ext3)) / 2.0 * 1.05
        self.xmag = diag / zoom
        self.ymag = self.xmag * size[1] / size[0]
        cam = pyrender.OrthographicCamera(xmag=self.xmag, ymag=self.ymag, znear=1.0, zfar=diag * 20)
        self.cam_pose = _rot_x(90 - elevation) @ _translate([0, 0, diag * 6])
        self.scene.add(cam, pose=self.cam_pose)
        key = pyrender.DirectionalLight(color=np.ones(3), intensity=style.light)
        self.scene.add(key, pose=self.cam_pose @ _rot_x(-25) @ _rot_z(30))
        fill = pyrender.DirectionalLight(color=np.ones(3), intensity=style.light * 0.35)
        self.scene.add(fill, pose=self.cam_pose @ _rot_x(40) @ _rot_z(-120))
        self.renderer = pyrender.OffscreenRenderer(*size)
        self.nm_per_px = 2 * self.xmag / size[0]

    def frame(self, azimuth_deg: float, label: Optional[str] = None) -> np.ndarray:
        self.scene.set_pose(self.model_node, _rot_z(azimuth_deg))
        rgb, _ = self.renderer.render(self.scene, flags=pyrender.RenderFlags.SKIP_CULL_FACES)
        bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2BGR)
        return self.overlay(bgr, label)

    def overlay(self, img: np.ndarray, label: Optional[str] = None) -> np.ndarray:
        h, w = img.shape[:2]; st = self.style; fs = st.font_scale * (w / 1920)
        font = cv2.FONT_HERSHEY_DUPLEX
        if st.scale_bar:
            target = _nice_length(self.extent[0] / 4)
            px = int(round(target / self.nm_per_px))
            x0, y0 = w - px - int(60 * w / 1920), h - int(70 * h / 1080)
            cv2.rectangle(img, (x0, y0), (x0 + px, y0 + max(4, int(8 * h / 1080))), st.text, -1)
            txt = f"{target:g} nm" if target < 1000 else f"{target/1000:g} um"
            (tw, th), _ = cv2.getTextSize(txt, font, fs, 2)
            cv2.putText(img, txt, (x0 + px // 2 - tw // 2, y0 - int(12 * h / 1080)), font, fs, st.text, 2, cv2.LINE_AA)
        if st.legend:
            y = int(60 * h / 1080); x = int(50 * w / 1920); sw = int(34 * h / 1080)
            for cid, cs in sorted(self.class_styles.items()):
                col = tuple(int(255 * c) for c in cs.colour[::-1])
                cv2.rectangle(img, (x, y - sw // 2), (x + sw, y + sw // 2), col, -1)
                cv2.putText(img, cs.label, (x + sw + int(16 * w / 1920), y + sw // 3), font, fs, st.text, 2, cv2.LINE_AA)
                y += int(sw * 1.7)
        if label:
            (tw, th), _ = cv2.getTextSize(label, font, fs * 0.9, 2)
            cv2.putText(img, label, (w // 2 - tw // 2, h - int(30 * h / 1080)), font, fs * 0.9, st.text, 2, cv2.LINE_AA)
        return img

    def close(self):
        self.renderer.delete()


def _translate(t) -> np.ndarray:
    m = np.eye(4); m[:3, 3] = t; return m


def _nice_length(x: float) -> float:
    e = 10 ** np.floor(np.log10(x)); f = x / e
    for n in (1, 2, 5, 10):
        if f <= n:
            return n * e
    return 10 * e


# ── movie writing ───────────────────────────────────────────────────────────

def write_movie(frames_iter, out_mp4: Path, fps: int = 30, gif_path: Optional[Path] = None,
                gif_width: int = 960, gif_fps: int = 15, progress: Optional[Callable[[str], None]] = None):
    import imageio.v2 as imageio
    out_mp4 = Path(out_mp4); out_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                macro_block_size=8)
    gif_frames = []
    for i, bgr in enumerate(frames_iter):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(rgb)
        if gif_path is not None and i % max(1, round(fps / gif_fps)) == 0:
            scale = gif_width / rgb.shape[1]
            gif_frames.append(cv2.resize(rgb, (gif_width, int(rgb.shape[0] * scale)), interpolation=cv2.INTER_AREA))
        if progress and i % 10 == 0:
            progress(f"frame {i}")
    writer.close()
    if gif_path is not None and gif_frames:
        imageio.mimsave(str(gif_path), gif_frames, duration=1.0 / gif_fps, loop=0)


def render_spin_movie(labels: np.ndarray, voxel_nm: Sequence[float] | float, out_mp4: Path,
                      class_labels: dict[int, str], style: str = "talk", seconds: float = 20.0, fps: int = 30,
                      size=(1920, 1080), elevation: float = 35.0, zoom: float = 1.0,
                      context_slice: Optional[np.ndarray] = None, context_z: Optional[int] = None,
                      stills_deg: Sequence[float] = (30, 120), gif: bool = True, title: Optional[str] = None,
                      z_exaggeration: float = 1.0, step: int = 1, smooth_sigma=(2.0, 1.0, 1.0), min_voxels: int = 30,
                      progress: Optional[Callable[[str], None]] = None) -> dict:
    """End-to-end: label volume -> meshes -> rotating movie (+gif, +stills). Returns output paths."""
    if np.isscalar(voxel_nm):
        voxel_nm = (float(voxel_nm),) * 3
    st = STYLES[style]
    meshes = label_meshes(labels, voxel_nm, list(class_labels), smooth_sigma=smooth_sigma, step=step,
                          min_voxels=min_voxels, progress=progress)
    nz, ny, nx = labels.shape
    extent = (nx * voxel_nm[2], ny * voxel_nm[1], nz * voxel_nm[0])
    cstyles = {cid: ClassStyle(name, st.colours.get(cid, (0.7, 0.7, 0.7))) for cid, name in class_labels.items()}
    plane = None
    if context_slice is not None:
        # default: lay the slice on the floor of the box so it does not hide the surfaces
        zc = context_z if context_z is not None else 0
        plane = slice_plane(context_slice, voxel_nm, zc * voxel_nm[0], opacity=0.85)
    r = SpinRenderer(meshes, extent, st, cstyles, size=size, elevation=elevation, zoom=zoom,
                     context_plane=plane, z_exaggeration=z_exaggeration)
    n = int(round(seconds * fps))
    out_mp4 = Path(out_mp4)
    gif_path = out_mp4.with_suffix(".gif") if gif else None
    write_movie((r.frame(360.0 * i / n, title) for i in range(n)), out_mp4, fps=fps, gif_path=gif_path, progress=progress)
    stills = []
    for deg in stills_deg:
        p = out_mp4.with_name(f"{out_mp4.stem}_{int(deg):03d}deg.png")
        cv2.imwrite(str(p), r.frame(deg, title)); stills.append(p)
    r.close()
    return {"mp4": out_mp4, "gif": gif_path, "stills": stills,
            "faces": {cid: len(m.faces) for cid, m in meshes.items()}}


# ── CLI ─────────────────────────────────────────────────────────────────────

def _parse_classes(s: str) -> dict[int, str]:
    out = {}
    for part in s.split(","):
        k, v = part.split(":", 1); out[int(k)] = v.strip()
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="label volume (.mrc or .npy), Z,Y,X")
    ap.add_argument("out_mp4")
    ap.add_argument("--classes", default="1:class 1", help='e.g. "1:CNC,2:nanoparticle"')
    ap.add_argument("--style", choices=list(STYLES), default="talk")
    ap.add_argument("--voxel-nm", type=float, default=None, help="override voxel size (nm)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--elevation", type=float, default=35.0)
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--zexag", type=float, default=1.0, help="stretch the thin Z axis for visibility")
    ap.add_argument("--context", default=None, help="tomogram .mrc whose central slice is shown as a plane")
    ap.add_argument("--context-z", type=int, default=None)
    ap.add_argument("--step", type=int, default=1, help="marching cubes step (2 = coarser, faster)")
    ap.add_argument("--smooth", type=float, default=None, help="isotropic smoothing sigma (default anisotropic 2,1,1)")
    ap.add_argument("--min-voxels", type=int, default=30)
    ap.add_argument("--title", default=None)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args(argv)

    import mrcfile
    if a.labels.endswith(".npy"):
        labels = np.load(a.labels); vox = (a.voxel_nm or 1.0,) * 3
    else:
        with mrcfile.open(a.labels, permissive=True) as m:
            labels = np.asarray(m.data).astype(np.int32)
            v = m.voxel_size; vox = (float(v.z) / 10, float(v.y) / 10, float(v.x) / 10)
        if a.voxel_nm:
            vox = (a.voxel_nm,) * 3
    ctx = None
    if a.context:
        with mrcfile.mmap(a.context, permissive=True) as m:
            zc = a.context_z if a.context_z is not None else m.data.shape[0] // 2
            ctx = np.asarray(m.data[zc])
    w, h = (int(x) for x in a.size.lower().split("x"))
    res = render_spin_movie(labels, vox, a.out_mp4, _parse_classes(a.classes), style=a.style, seconds=a.seconds,
                            fps=a.fps, size=(w, h), elevation=a.elevation, zoom=a.zoom, context_slice=ctx,
                            context_z=a.context_z, gif=not a.no_gif, title=a.title, z_exaggeration=a.zexag,
                            step=a.step, smooth_sigma=(a.smooth if a.smooth is not None else (2.0, 1.0, 1.0)),
                            min_voxels=a.min_voxels, progress=print)
    print(res)


if __name__ == "__main__":
    main()
