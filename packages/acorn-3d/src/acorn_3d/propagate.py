"""
Propagate SAM 3 masks through the z slices of a tomogram.

The z stack is treated as a video (z = time) and handed to SAM 3's video
predictor.  A prompt (text, boxes or points) is placed on one slice and
propagated in both directions; every tracked object keeps its id across slices,
so the result is a per-slice set of instance masks that can be rasterised into
a label volume (see acorn_3d.labels).

The sam3 package and torch are imported lazily; the module can be imported
without them.
"""
from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np


@dataclass
class ZPrompt:
    z: int
    text: Optional[str] = None
    boxes_xyxy: Optional[list[tuple[float, float, float, float]]] = None   # image px
    points: Optional[list[tuple[float, float]]] = None                     # image px
    point_labels: Optional[list[int]] = None                               # 1 fg, 0 bg
    obj_id: Optional[int] = None


@dataclass
class PropagationResult:
    """masks[z][obj_id] -> bool (H, W) mask; scores[z][obj_id] -> float."""
    masks: dict[int, dict[int, np.ndarray]] = field(default_factory=dict)
    scores: dict[int, dict[int, float]] = field(default_factory=dict)
    n_slices: int = 0

    def object_ids(self) -> list[int]:
        ids = set()
        for d in self.masks.values():
            ids.update(d)
        return sorted(ids)

    def slices_for(self, obj_id: int) -> list[int]:
        return sorted(z for z, d in self.masks.items() if obj_id in d)

    def to_label_volume(self, shape_hw: tuple[int, int], class_id: int = 1, dtype=np.uint8,
                        min_slices: int = 1, score_min: float = 0.0) -> np.ndarray:
        """One class id for every tracked object (semantic volume)."""
        vol = np.zeros((self.n_slices, *shape_hw), dtype)
        keep = {o for o in self.object_ids() if len(self.slices_for(o)) >= min_slices}
        for z, d in self.masks.items():
            for o, m in d.items():
                if o in keep and self.scores.get(z, {}).get(o, 1.0) >= score_min:
                    vol[z][m] = class_id
        return vol

    def to_instance_volume(self, shape_hw: tuple[int, int], dtype=np.uint16) -> np.ndarray:
        vol = np.zeros((self.n_slices, *shape_hw), dtype)
        for z, d in self.masks.items():
            for o, m in d.items():
                vol[z][m] = o + 1
        return vol


def slices_to_uint8(vol: np.ndarray, percentiles=(0.5, 99.5), denoise_sigma: float = 0.0,
                    invert: bool = False, zsmooth: int = 1) -> np.ndarray:
    """
    Contrast-stretch a float (Z, H, W) volume to uint8 using global percentiles.
    zsmooth > 1 averages that many neighbouring slices (running mean along z), which lifts the
    signal of a noisy back-projection volume the way a slab view does in IMOD.
    """
    from scipy.ndimage import gaussian_filter, uniform_filter1d
    v = np.asarray(vol, dtype=np.float32)
    if zsmooth and zsmooth > 1 and v.shape[0] > 1:
        v = uniform_filter1d(v, size=int(zsmooth), axis=0, mode="nearest")
    if denoise_sigma > 0:
        v = np.stack([gaussian_filter(s, denoise_sigma) for s in v])
    lo, hi = np.percentile(v[:: max(1, v.shape[0] // 16)], percentiles)
    out = np.clip((v - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return 255 - out if invert else out


class ZPropagator:
    """Holds one SAM 3 video model and runs prompt + propagation sessions on z stacks."""

    def __init__(self, device: str = "cuda", checkpoint_path: Optional[str] = None,
                 offload_video_to_cpu: bool = False) -> None:
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.offload = offload_video_to_cpu
        self._predictor = None

    # ── model ───────────────────────────────────────────────────────────────
    def load(self) -> None:
        if self._predictor is not None:
            return
        try:
            from acorn.core import sam3_assets  # type: ignore
            sam3_assets.ensure_vocab()
        except Exception:
            pass
        import torch
        from sam3.model.sam3_video_predictor import Sam3VideoPredictor

        if self.device.startswith("cuda") and ":" in self.device:
            torch.cuda.set_device(int(self.device.split(":")[1]))
        kwargs = dict(load_from_HF=self.checkpoint_path is None)
        if self.checkpoint_path:
            kwargs["checkpoint_path"] = self.checkpoint_path
        sig = inspect.signature(Sam3VideoPredictor.__init__)
        if "device" in sig.parameters:
            kwargs["device"] = self.device
        if "offload_video_to_cpu" in sig.parameters:
            kwargs["offload_video_to_cpu"] = self.offload
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())}
        self._predictor = Sam3VideoPredictor(**kwargs)

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None

    # ── run ─────────────────────────────────────────────────────────────────
    def set_detection_threshold(self, new_det_thresh: Optional[float] = None) -> None:
        """Lower SAM 3 video's threshold for starting a new tracked object (default 0.7)."""
        self.load()
        model = self._predictor.model
        if new_det_thresh is not None and hasattr(model, "new_det_thresh"):
            model.new_det_thresh = float(new_det_thresh)

    def run(self, slices_u8: np.ndarray, prompts: Sequence[ZPrompt], direction: str = "both",
            max_slices: Optional[int] = None, score_min: float = 0.0,
            text_candidates: Optional[Sequence[str]] = None,
            progress: Optional[Callable[[int, int], None]] = None) -> PropagationResult:
        """
        slices_u8: (Z, H, W) or (Z, H, W, 3) uint8.  All prompts must share one z (one
        SAM 3 session tracks one prompt set); call run() once per class.
        text_candidates: alternative phrasings tried in order on the prompt slice until one
        detects something (the session is reset between attempts); the ZPrompt.text is tried first.
        """
        self.load()
        from PIL import Image
        import torch

        if slices_u8.ndim == 3:
            frames = [Image.fromarray(np.stack([s] * 3, -1)) for s in slices_u8]
        else:
            frames = [Image.fromarray(s) for s in slices_u8]
        n = len(frames); h, w = slices_u8.shape[1:3]
        res = PropagationResult(n_slices=n)
        if not prompts:
            return res
        z0 = int(prompts[0].z)
        pred = self._predictor
        session = pred.start_session(resource_path=frames)["session_id"]
        self.phrase_used: Optional[str] = None
        try:
            # text prompt: try the candidate phrasings on the prompt slice until one detects objects
            if len(prompts) == 1 and prompts[0].text and not prompts[0].boxes_xyxy and not prompts[0].points:
                cands = [prompts[0].text] + [c for c in (text_candidates or []) if c and c != prompts[0].text]
                chosen = None
                for phrase in cands:
                    r = pred.add_prompt(session_id=session, frame_idx=z0, text=phrase)
                    n_found = len(list(r["outputs"].get("out_obj_ids", [])))
                    if n_found > 0:
                        chosen = phrase
                        break
                    pred.reset_session(session_id=session)
                if chosen is None:
                    return res
                self.phrase_used = chosen
                prompts = []     # already added
            for p in prompts:
                # SAM 3 video takes boxes as normalised xywh and points as normalised xy
                boxes = None
                if p.boxes_xyxy:
                    boxes = [[max(0.0, x0 / w), max(0.0, y0 / h), min(1.0, (x1 - x0) / w), min(1.0, (y1 - y0) / h)]
                             for x0, y0, x1, y1 in p.boxes_xyxy]
                    boxes = [[bx, by, min(bw, 1.0 - bx), min(bh, 1.0 - by)] for bx, by, bw, bh in boxes]
                pts = [[float(x) / w, float(y) / h] for x, y in p.points] if p.points else None
                pred.add_prompt(session_id=session, frame_idx=int(p.z), text=p.text,
                                points=pts, point_labels=p.point_labels if pts else None,
                                bounding_boxes=boxes, bounding_box_labels=[1] * len(boxes) if boxes else None,
                                obj_id=p.obj_id)
            done = 0
            with torch.inference_mode():
                for ev in pred.propagate_in_video(session_id=session, propagation_direction=direction,
                                                  start_frame_idx=z0, max_frame_num_to_track=max_slices or n):
                    z = int(ev["frame_index"]); out = ev["outputs"]
                    ids = list(out.get("out_obj_ids", []))
                    masks = out.get("out_binary_masks")
                    probs = out.get("out_probs")
                    if masks is None or len(ids) == 0:
                        continue
                    masks = _to_numpy(masks).astype(bool)
                    probs = _to_numpy(probs) if probs is not None else np.ones(len(ids))
                    if masks.shape[-2:] != (h, w):
                        masks = _resize_masks(masks, (h, w))
                    res.masks.setdefault(z, {}); res.scores.setdefault(z, {})
                    for i, oid in enumerate(ids):
                        s = float(probs[i]) if i < len(probs) else 1.0
                        if s < score_min or not masks[i].any():
                            continue
                        res.masks[z][int(oid)] = masks[i]
                        res.scores[z][int(oid)] = s
                    done += 1
                    if progress:
                        progress(done, n)
        finally:
            try:
                pred.close_session(session)
            except Exception:
                pass
        return res

    def run_masks(self, slices_u8: np.ndarray, seeds: Sequence[tuple[int, int, np.ndarray]],
                  direction: str = "both", max_slices: Optional[int] = None, refine_rounds: int = 2,
                  warmup_text: str = "object",
                  progress: Optional[Callable[[int, int], None]] = None) -> PropagationResult:
        """
        Track user-drawn masks through z.  seeds are (z, obj_id, bool mask).

        SAM 3's public video API accepts point prompts tied to an object id, so each
        seed mask is turned into a few interior points (distance-transform peaks),
        the mask SAM returns on that slice is compared with the seed, and up to
        `refine_rounds` corrective points are added where they disagree.  The tracked
        objects are then propagated through the stack in `direction`.
        """
        self.load()
        import torch
        from PIL import Image

        if slices_u8.ndim == 3:
            frames = [Image.fromarray(np.stack([s] * 3, -1)) for s in slices_u8]
        else:
            frames = [Image.fromarray(s) for s in slices_u8]
        n = len(frames); h, w = slices_u8.shape[1:3]
        res = PropagationResult(n_slices=n)
        if not seeds:
            return res
        pred = self._predictor
        session = pred.start_session(resource_path=frames)["session_id"]
        seed_z = sorted({int(z) for z, _, _ in seeds})
        try:
            with torch.inference_mode():
                # SAM 3 only accepts point prompts on frames whose outputs are cached, which
                # requires one full propagation first: run a throwaway text prompt through
                # the stack, then drop whatever it found.
                warm = pred.add_prompt(session_id=session, frame_idx=seed_z[0], text=warmup_text)
                total = 2 * n   # warm-up pass + tracking pass, each visits every slice once
                done = 0
                for _ev in pred.propagate_in_video(session_id=session, propagation_direction=direction,
                                                   start_frame_idx=seed_z[0], max_frame_num_to_track=max_slices or n):
                    done += 1
                    if progress:
                        progress(done, total)
                for oid in sorted({int(i) for i in warm["outputs"].get("out_obj_ids", [])}):
                    try:
                        pred.remove_object(session_id=session, obj_id=oid)
                    except Exception:
                        pass
                for z, oid, m in seeds:
                    m = np.asarray(m, dtype=bool)
                    pts, labs = _seed_points(m, n_pos=3)
                    out = self._add_points(pred, session, int(z), int(oid), pts, labs, w, h)
                    for _ in range(refine_rounds):
                        got = _mask_for(out, int(oid), (h, w))
                        if got is None:
                            break
                        iou = (got & m).sum() / max(1, (got | m).sum())
                        if iou > 0.9:
                            break
                        extra_p, extra_l = _correction_points(m, got)
                        if not extra_p:
                            break
                        pts += extra_p; labs += extra_l
                        out = self._add_points(pred, session, int(z), int(oid), pts, labs, w, h)
                dirs = {"both": ("forward", "backward"), "forward": ("forward",), "backward": ("backward",)}[direction]
                for d in dirs:
                    start = seed_z[-1] if d == "backward" else seed_z[0]
                    for ev in pred.propagate_in_video(session_id=session, propagation_direction=d,
                                                      start_frame_idx=int(start),
                                                      max_frame_num_to_track=max_slices or n):
                        z = int(ev["frame_index"]); out = ev["outputs"]
                        ids = list(out.get("out_obj_ids", []))
                        masks = out.get("out_binary_masks")
                        if masks is None or len(ids) == 0:
                            done += 1
                            continue
                        masks = _to_numpy(masks).astype(bool)
                        if masks.shape[-2:] != (h, w):
                            masks = _resize_masks(masks, (h, w))
                        probs = _to_numpy(out.get("out_probs")) if out.get("out_probs") is not None else np.ones(len(ids))
                        res.masks.setdefault(z, {}); res.scores.setdefault(z, {})
                        for i, oid in enumerate(ids):
                            if masks[i].any():
                                res.masks[z][int(oid)] = masks[i]
                                res.scores[z][int(oid)] = float(probs[i]) if i < len(probs) else 1.0
                        done += 1
                        if progress:
                            progress(done, total)
        finally:
            try:
                pred.close_session(session)
            except Exception:
                pass
        return res

    @staticmethod
    def _add_points(pred, session, z, oid, pts, labs, w, h):
        r = pred.add_prompt(session_id=session, frame_idx=z, obj_id=oid,
                            points=[[x / w, y / h] for x, y in pts], point_labels=list(labs))
        return r["outputs"]

    def close(self) -> None:
        self._predictor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _seed_points(mask: np.ndarray, n_pos: int = 3) -> tuple[list[tuple[float, float]], list[int]]:
    """Interior points of a mask: the distance-transform maximum, then well separated peaks."""
    from scipy import ndimage
    dist = ndimage.distance_transform_edt(mask)
    pts: list[tuple[float, float]] = []
    d = dist.copy()
    for _ in range(n_pos):
        if d.max() <= 0:
            break
        y, x = np.unravel_index(int(np.argmax(d)), d.shape)
        pts.append((float(x), float(y)))
        r = max(3.0, float(dist[y, x]) * 1.5)
        yy, xx = np.ogrid[:d.shape[0], :d.shape[1]]
        d[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 0
    return pts, [1] * len(pts)


def _correction_points(target: np.ndarray, got: np.ndarray, max_each: int = 2):
    """Positive points in missed regions, negative points in spurious regions."""
    from scipy import ndimage
    pts, labs = [], []
    for region, lab in (((target & ~got), 1), ((got & ~target), 0)):
        if region.sum() < 25:
            continue
        dist = ndimage.distance_transform_edt(region)
        for _ in range(max_each):
            if dist.max() <= 1.5:
                break
            y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
            pts.append((float(x), float(y))); labs.append(lab)
            r = max(3.0, float(dist[y, x]) * 2)
            yy, xx = np.ogrid[:dist.shape[0], :dist.shape[1]]
            dist[(yy - y) ** 2 + (xx - x) ** 2 <= r * r] = 0
    return pts, labs


def _mask_for(outputs: dict, oid: int, hw: tuple[int, int]) -> Optional[np.ndarray]:
    ids = list(outputs.get("out_obj_ids", []))
    masks = outputs.get("out_binary_masks")
    if masks is None:
        return None
    for i, o in enumerate(ids):
        if int(o) == oid:
            m = _to_numpy(masks)[i].astype(bool)
            return m if m.shape == hw else _resize_masks(m[None], hw)[0]
    return None


def _to_numpy(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(x)


def _resize_masks(masks: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    import cv2
    return np.stack([cv2.resize(m.astype(np.uint8), (hw[1], hw[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in masks])


_SHARED: dict[str, ZPropagator] = {}


def shared_propagator(device: str = "cuda") -> ZPropagator:
    """One model per device, reused across calls (loading takes tens of seconds)."""
    if device not in _SHARED:
        _SHARED[device] = ZPropagator(device=device)
    return _SHARED[device]
