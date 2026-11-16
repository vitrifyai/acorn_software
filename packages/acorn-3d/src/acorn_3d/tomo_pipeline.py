"""
Command-line tomogram pipeline: SAM 3 text prompts propagated through z, a label
volume, and the spinning movies.  The same code the acorn_3d GUI panel uses,
runnable detached on a workstation.

Example:
    python -m acorn_3d.tomo_pipeline rec.mrc out_dir \
        --class "CNC=fiber" --class "nanoparticle=dark sphere" \
        --z 240 399 --bin 1 --denoise 1.5 --styles talk paper --seconds 12

Each --class is "label=text prompt".  Classes are tracked one after another on
the same GPU; later classes overwrite earlier ones where they overlap.
Outputs in out_dir: labels.mrc (+ labels.classes.txt), per-class instance
volumes (.npy), prompts.json, and <stem>_<style>.mp4/.gif/.png.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def specimen_z_range(v, frac: float = 0.70) -> tuple[int, int]:
    """
    Central `frac` of the z range.  Weighted back-projection noise makes per-slice contrast
    nearly flat through the volume, so a contrast profile cannot locate the specimen; the top
    and bottom 15 % of an AreTomo2 volume are mostly edge artefact and are dropped instead.
    """
    n = v.n_slices
    lo = int(round(n * (1 - frac) / 2))
    return lo, n - 1 - lo


def slicewise_text(u8: np.ndarray, word: str, cid: int, confidence: float, log) -> tuple[np.ndarray, int, str | None]:
    """SAM 3 image model, text prompt on every slice; returns (label volume, mask count, phrase)."""
    from acorn.core.sam_predictor import SAMPredictor
    pred = SAMPredictor(backend="sam3")
    pred.load_model()
    lab = np.zeros(u8.shape, np.uint8)
    n_masks = 0; phrase = None
    for z in range(u8.shape[0]):
        try:
            masks, ph = pred.predict_text(u8[z], word, confidence=confidence)
        except Exception as exc:
            log(f"  slice {z}: {exc}"); continue
        for m in masks:
            lab[z][m] = cid
        n_masks += len(masks); phrase = phrase or (ph if masks else None)
        if z % 50 == 0:
            log(f"  {word} 2D: slice {z}/{u8.shape[0]}, {n_masks} masks so far")
    return lab, n_masks, phrase


def load_stack(path: Path, z_from: int | None, z_to: int | None, binning: int, auto_z: bool = False):
    from acorn_3d.volume_loader import VolumeImage
    v = VolumeImage.from_file(path)
    n = v.n_slices
    if auto_z and z_from is None and z_to is None:
        z_from, z_to = specimen_z_range(v)
    z0 = 0 if z_from is None else max(0, z_from)
    z1 = n - 1 if z_to is None else min(n - 1, z_to)
    slices = []
    for z in range(z0, z1 + 1):
        v.set_slice(z)
        a = np.asarray(v.raw, dtype=np.float32)
        if binning > 1:
            h, w = (a.shape[0] // binning) * binning, (a.shape[1] // binning) * binning
            a = a[:h, :w].reshape(h // binning, binning, w // binning, binning).mean((1, 3))
        slices.append(a)
    stack = np.stack(slices)
    vox = (v.meta.voxel_depth_nm or v.pixel_size, v.pixel_size * binning, v.pixel_size * binning)
    return stack, vox, (z0, z1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("volume")
    ap.add_argument("out_dir")
    ap.add_argument("--class", dest="classes", action="append", required=True, help='"label=text prompt"')
    ap.add_argument("--z", nargs=2, type=int, default=(None, None), metavar=("FROM", "TO"))
    ap.add_argument("--zauto", action="store_true", help="use the central 70 %% of z (drops the artefact-prone top and bottom)")
    ap.add_argument("--z0", type=int, default=None, help="prompt slice (default: middle of the range)")
    ap.add_argument("--bin", type=int, default=1, help="extra in-plane binning before SAM 3")
    ap.add_argument("--denoise", type=float, default=1.5, help="Gaussian sigma (px) before contrast stretch")
    ap.add_argument("--zsmooth", type=int, default=5, help="average this many neighbouring slices before SAM 3")
    ap.add_argument("--det-thresh", type=float, default=0.5, help="SAM 3 video new-object threshold (default 0.7 in sam3)")
    ap.add_argument("--no-2d-fallback", action="store_true",
                    help="do not fall back to slice-by-slice SAM 3 image prompts when video tracking finds nothing")
    ap.add_argument("--fallback-conf", type=float, default=0.3, help="confidence for the slice-by-slice fallback")
    ap.add_argument("--border", type=float, default=0.03,
                    help="ignore labels within this fraction of the xy size from the volume edge (reconstruction border artefacts)")
    ap.add_argument("--min-slices", type=int, default=3, help="drop objects tracked on fewer slices")
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--styles", nargs="*", default=["talk", "paper"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--zexag", type=float, default=1.0)
    ap.add_argument("--no-context", action="store_true", help="do not show the central slice under the surfaces")
    ap.add_argument("--title", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--render-only", action="store_true", help="reuse out_dir/labels.mrc, just render")
    a = ap.parse_args(argv)

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("TQDM_DISABLE", "1")
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    vol_path = Path(a.volume)
    classes = []
    for c in a.classes:
        label, _, prompt = c.partition("=")
        classes.append((label.strip(), (prompt or label).strip()))
    names = {i + 1: lab for i, (lab, _) in enumerate(classes)}
    log = lambda *m: print(time.strftime("%H:%M:%S"), *m, flush=True)

    log("reading", vol_path)
    stack, vox, (z0, z1) = load_stack(vol_path, a.z[0], a.z[1], a.bin, auto_z=a.zauto)
    log(f"stack {stack.shape}, voxel nm {tuple(round(x, 3) for x in vox)}, z {z0}..{z1}")

    labels_path = out / "labels.mrc"
    if a.render_only and labels_path.exists():
        import mrcfile
        with mrcfile.open(str(labels_path), permissive=True) as m:
            labels = np.asarray(m.data).astype(np.uint8)
        log("loaded existing", labels_path)
    else:
        from acorn_3d.propagate import slices_to_uint8, shared_propagator, ZPrompt
        from acorn_3d.labels import write_label_mrc
        u8 = slices_to_uint8(stack, denoise_sigma=a.denoise, zsmooth=a.zsmooth)
        zp = shared_propagator(a.device)
        t0 = time.time(); zp.load(); log(f"SAM 3 video model loaded in {time.time() - t0:.0f}s")
        zp.set_detection_threshold(a.det_thresh)
        # ACORN's prompt vocabulary: alternative phrasings for each word, matched to the image polarity
        try:
            from acorn.core import vocabulary as _voc
            from acorn.core.sam_predictor import SAMPredictor as _SP
            polarity = _SP.image_polarity(u8[len(u8) // 2])
            def candidates(word):
                try:
                    return list(_voc.prompts_for(word, polarity))
                except Exception:
                    return []
        except Exception:
            polarity = None
            def candidates(word):
                return []
        log(f"image polarity: {polarity}")
        zprompt = (len(u8) // 2) if a.z0 is None else max(0, min(len(u8) - 1, a.z0 - z0))
        labels = np.zeros(u8.shape, np.uint8)
        record = {"volume": str(vol_path), "z_range": [z0, z1], "prompt_slice": z0 + zprompt, "bin": a.bin,
                  "denoise": a.denoise, "zsmooth": a.zsmooth, "det_thresh": a.det_thresh, "classes": []}
        for cid, (label, prompt) in enumerate(classes, start=1):
            t0 = time.time()
            res = zp.run(u8, [ZPrompt(z=zprompt, text=prompt)], score_min=a.score_min,
                         text_candidates=candidates(prompt),
                         progress=lambda d, n: log(f"  {label}: {d}/{n}") if d % 50 == 0 else None)
            log(f"  phrase used: {zp.phrase_used!r}")
            lab = res.to_label_volume(u8.shape[1:], class_id=cid, min_slices=a.min_slices)
            n_obj = len(res.object_ids())
            mode = "video"
            if n_obj == 0 and not a.no_2d_fallback:
                # Thin structures (CNC fibrils) rarely pass the video detector; segment every slice
                # with the SAM 3 image model instead (semantic, no object identity across z).
                log(f"  {label}: video tracking found nothing, falling back to slice-by-slice prompts")
                lab, n_obj, phrase = slicewise_text(u8, prompt, cid, a.fallback_conf, log)
                zp.phrase_used = phrase
                mode = "slicewise-2d"
            if a.border > 0:   # AreTomo2 volumes have shifted-stack borders that text prompts latch onto
                by, bx = int(round(lab.shape[1] * a.border)), int(round(lab.shape[2] * a.border))
                lab[:, :by, :] = 0; lab[:, -by:, :] = 0; lab[:, :, :bx] = 0; lab[:, :, -bx:] = 0
            labels[lab > 0] = cid
            if mode == "video":
                np.save(out / f"instances_{cid}_{label}.npy", res.to_instance_volume(u8.shape[1:]))
            log(f"{label} <- '{prompt}': {n_obj} objects, {int((lab > 0).sum())} voxels, {time.time() - t0:.0f}s")
            record["classes"].append({"id": cid, "label": label, "prompt": prompt, "phrase_used": zp.phrase_used, "mode": mode, "objects": n_obj,
                                      "voxels": int((lab > 0).sum()), "slices": len(res.masks)})
        write_label_mrc(labels, labels_path, vox)
        (out / "labels.classes.txt").write_text("".join(f"{k}\t{v}\n" for k, v in names.items()))
        (out / "prompts.json").write_text(json.dumps(record, indent=1))
        log("label volume written", labels_path)

    from acorn_3d.render import render_spin_movie
    w, h = (int(x) for x in a.size.lower().split("x"))
    ctx = None if a.no_context else stack[len(stack) // 2]
    for style in a.styles:
        t0 = time.time()
        res = render_spin_movie(labels, vox, out / f"{vol_path.stem}_{style}.mp4", names, style=style,
                                seconds=a.seconds, fps=a.fps, size=(w, h), context_slice=ctx,
                                context_z=0, z_exaggeration=a.zexag, title=a.title,
                                progress=lambda m: log(f"  {style}: {m}") if m.startswith("frame") and m.endswith("00") else None)
        log(f"{style}: {res['mp4']} ({time.time() - t0:.0f}s), faces {res['faces']}")


if __name__ == "__main__":
    main()
