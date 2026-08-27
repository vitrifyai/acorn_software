"""Annotation provenance + opt-in study-mode instrumentation.

Two layers:

1. **Always on** (cheap, part of the data model): every annotation carries a
   ``Provenance`` block. The annotation store stamps it on ``add`` using the
   ambient provenance *context* (set by whoever is creating annotations —
   a predictor, CLU, an importer). With no context, origin defaults to
   ``manual`` and invocation to ``direct_gui``, so untagged canvas drawing is
   correctly attributed by absence. ``geometry_hash_at_creation`` lets analysis
   recover accepted-unchanged vs edited after the fact, even if an edit event
   was never captured.

2. **Study mode** (off by default): an append-only JSONL event log, a separate
   CLU conversation log, idle detection, and a session manifest. Started only
   when a study explicitly begins; shows a recording indicator (GUI layer).

This module is intentionally Qt-free so the offline replay/analysis script can
import it. The GUI-only pieces (idle event filter, recording indicator, study
dialog) live in ``acorn/gui/study.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── enums ─────────────────────────────────────────────────────────────────────

class Origin(str, Enum):
    MANUAL = "manual"
    SAM = "sam"
    YOLO = "yolo"
    UNET = "unet"
    CRYOBLOB = "cryoblob"
    SIMULATOR_PRETRAINED = "simulator_pretrained"
    IMPORTED_STAR = "imported_star"
    IMPORTED_SIDECAR = "imported_sidecar"
    UNKNOWN = "unknown"


#: Origins that represent (unreviewed-by-default) model output.
MODEL_ORIGINS = frozenset({
    Origin.SAM.value, Origin.YOLO.value, Origin.UNET.value,
    Origin.CRYOBLOB.value, Origin.SIMULATOR_PRETRAINED.value,
})


class Invocation(str, Enum):
    DIRECT_GUI = "direct_gui"
    CLU_NL = "clu_nl"
    BATCH = "batch"
    SYSTEM = "system"


# ── provenance block (nested on every annotation) ─────────────────────────────

@dataclass
class Provenance:
    """Per-annotation provenance. ``created_at is None`` ⇒ not yet stamped."""
    id: str = ""
    origin: str = Origin.UNKNOWN.value
    invocation: str = Invocation.DIRECT_GUI.value
    source_model: Optional[dict] = None          # {"path": ..., "sha256": ...}
    created_at: Optional[str] = None             # ISO-8601 UTC, ms
    modified_at: Optional[str] = None
    created_by: Optional[str] = None
    modification_count: int = 0
    parent_id: Optional[str] = None
    batch_id: Optional[str] = None
    turn_id: Optional[str] = None
    reviewed: bool = False
    geometry_hash_at_creation: Optional[str] = None


def provenance_from_dict(d: Any) -> Provenance:
    """Rebuild a Provenance from a serialized dict (sidecar round-trip)."""
    if not isinstance(d, dict):
        return Provenance()
    valid = {f.name for f in Provenance.__dataclass_fields__.values()}
    return Provenance(**{k: v for k, v in d.items() if k in valid})


# ── time / identity helpers ───────────────────────────────────────────────────

def now_iso() -> str:
    """ISO-8601 UTC timestamp with millisecond precision, e.g. ...T12:00:00.123Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def now_mono() -> float:
    """Monotonic clock — survives NTP steps; use for durations."""
    return time.monotonic()


def new_id() -> str:
    return uuid.uuid4().hex


_ANNOTATOR_CACHE: Optional[str] = None


def current_annotator() -> str:
    """Pseudonymous, per-install annotator id (study mode overrides this)."""
    if _RECORDER.active and _RECORDER.annotator_id:
        return _RECORDER.annotator_id
    global _ANNOTATOR_CACHE
    if _ANNOTATOR_CACHE:
        return _ANNOTATOR_CACHE
    try:
        p = Path.home() / ".acorn" / "annotator_id"
        if p.exists():
            _ANNOTATOR_CACHE = p.read_text().strip()
        else:
            _ANNOTATOR_CACHE = "anon-" + new_id()[:12]
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(_ANNOTATOR_CACHE)
    except OSError:
        _ANNOTATOR_CACHE = "anon-ephemeral"
    return _ANNOTATOR_CACHE


def hash_file(path: str) -> Optional[str]:
    """Fast checkpoint fingerprint: sha256 of (size + first MB + last MB).

    A full hash of a multi-GB checkpoint would stall the UI thread; this cheap
    fingerprint still reliably distinguishes one training round's model from
    another (different weights ⇒ different head/tail/size)."""
    try:
        size = os.path.getsize(path)
        h = hashlib.sha256()
        h.update(str(size).encode())
        with open(path, "rb") as f:
            h.update(f.read(1 << 20))
            if size > (1 << 20):
                f.seek(max(0, size - (1 << 20)))
                h.update(f.read(1 << 20))
        return h.hexdigest()[:16]
    except OSError:
        return None


def source_model(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    return {"path": str(path), "sha256": hash_file(path)}


# ── geometry hashing (the field that saves you) ───────────────────────────────

def geometry_signature(ann: Any) -> list:
    """Canonical, color/label-independent geometry descriptor for an annotation."""
    t = getattr(ann, "type", None)
    r = lambda v: round(float(v), 3)
    try:
        if t == "roi":
            return ["roi", [[r(x), r(y)] for x, y in ann.vertices]]
        if t == "rectangle":
            return ["rectangle", r(ann.x0), r(ann.y0), r(ann.x1), r(ann.y1)]
        if t == "circle":
            return ["circle", r(ann.cx), r(ann.cy), r(ann.r)]
        if t in ("line", "arrow", "distance"):
            return [t, [r(ann.p1[0]), r(ann.p1[1])], [r(ann.p2[0]), r(ann.p2[1])]]
        if t == "angle":
            return [t, [r(ann.p1[0]), r(ann.p1[1])],
                    [r(ann.vertex[0]), r(ann.vertex[1])],
                    [r(ann.p2[0]), r(ann.p2[1])]]
        if t == "text":
            return ["text", r(ann.x), r(ann.y), getattr(ann, "label", "")]
        if t == "scalebar":
            return ["scalebar", r(ann.nm), r(ann.x_frac), r(ann.y_frac)]
    except Exception:
        pass
    return [t]


def geometry_hash(ann: Any) -> str:
    return hashlib.sha256(
        json.dumps(geometry_signature(ann), sort_keys=True).encode()
    ).hexdigest()[:16]


# ── provenance context stack (thread-local) ───────────────────────────────────

@dataclass
class _Ctx:
    origin: str
    invocation: str = Invocation.DIRECT_GUI.value
    source_model: Optional[dict] = None
    parent_id: Optional[str] = None
    batch_id: Optional[str] = None
    turn_id: Optional[str] = None


_ctx = threading.local()


def _stack() -> list:
    if not hasattr(_ctx, "stack"):
        _ctx.stack = []
    return _ctx.stack


def current_context() -> Optional[_Ctx]:
    s = _stack()
    return s[-1] if s else None


@contextmanager
def provenance_context(origin: str,
                       invocation: str = Invocation.DIRECT_GUI.value,
                       source_model: Optional[dict] = None,
                       parent_id: Optional[str] = None,
                       batch_id: Optional[str] = None,
                       turn_id: Optional[str] = None):
    """Everything added to a store inside this block is stamped with these values."""
    if isinstance(origin, Origin):
        origin = origin.value
    if isinstance(invocation, Invocation):
        invocation = invocation.value
    _stack().append(_Ctx(origin, invocation, source_model, parent_id, batch_id, turn_id))
    try:
        yield
    finally:
        try:
            _stack().pop()
        except IndexError:
            pass


@contextmanager
def batch_transaction(origin: Optional[str] = None,
                      invocation: str = Invocation.BATCH.value,
                      human_interaction_count: int = 1,
                      source_model: Optional[dict] = None):
    """One human action producing many annotations. Stamps a shared ``batch_id``
    and logs a single ``human_interaction_count`` for the whole group, so a
    40-annotation batch does not read as 40 interactions."""
    parent = current_context()
    o = origin or (parent.origin if parent else Origin.MANUAL.value)
    if isinstance(o, Origin):
        o = o.value
    inv = invocation.value if isinstance(invocation, Invocation) else invocation
    sm = source_model or (parent.source_model if parent else None)
    bid = new_id()
    _RECORDER.log("batch_begin", batch_id=bid, origin=o, invocation=inv,
                  human_interaction_count=human_interaction_count)
    with provenance_context(o, invocation=inv, source_model=sm, batch_id=bid,
                            turn_id=parent.turn_id if parent else None):
        try:
            yield bid
        finally:
            _RECORDER.log("batch_end", batch_id=bid)


# ── stamping + store hooks ────────────────────────────────────────────────────

def stamp(ann: Any) -> None:
    """Fill an annotation's provenance from the ambient context, once."""
    prov = getattr(ann, "provenance", None)
    if prov is None or prov.created_at is not None:
        return
    ctx = current_context()
    now = now_iso()
    prov.id = prov.id or new_id()
    prov.origin = ctx.origin if ctx else Origin.MANUAL.value
    prov.invocation = ctx.invocation if ctx else Invocation.DIRECT_GUI.value
    prov.source_model = ctx.source_model if ctx else None
    prov.parent_id = ctx.parent_id if ctx else None
    prov.batch_id = ctx.batch_id if ctx else None
    prov.turn_id = ctx.turn_id if ctx else None
    prov.created_at = now
    prov.modified_at = now
    prov.created_by = current_annotator()
    prov.modification_count = 0
    prov.reviewed = (prov.origin == Origin.MANUAL.value)
    prov.geometry_hash_at_creation = geometry_hash(ann)


def _event_fields(ann: Any) -> dict:
    prov = getattr(ann, "provenance", None)
    return {
        "ann_id": prov.id if prov else None,
        "ann_type": getattr(ann, "type", None),
        "origin": prov.origin if prov else None,
        "invocation": prov.invocation if prov else None,
        "batch_id": prov.batch_id if prov else None,
        "turn_id": prov.turn_id if prov else None,
        "label": getattr(ann, "label", None),
        "geometry": geometry_signature(ann),
        "geometry_hash": geometry_hash(ann),
        "modification_count": prov.modification_count if prov else 0,
        "reviewed": prov.reviewed if prov else None,
    }


def on_add(ann: Any) -> None:
    stamp(ann)
    _RECORDER.log("annotation_add", **_event_fields(ann))


def on_update(ann: Any, event: str = "annotation_edit") -> None:
    prov = getattr(ann, "provenance", None)
    if prov is not None and prov.created_at is not None:
        prov.modified_at = now_iso()
        prov.modification_count += 1
        if (prov.geometry_hash_at_creation
                and geometry_hash(ann) != prov.geometry_hash_at_creation):
            prov.reviewed = True
    _RECORDER.log(event, **_event_fields(ann))


def on_remove(ann: Any) -> None:
    _RECORDER.log("annotation_delete", **_event_fields(ann))


def mark_reviewed(ann: Any, reviewed: bool = True) -> None:
    prov = getattr(ann, "provenance", None)
    if prov is not None:
        prov.reviewed = reviewed


# ── review / summary helpers (status indicator + finalize warning) ────────────

def is_model_output(ann: Any) -> bool:
    prov = getattr(ann, "provenance", None)
    return bool(prov and prov.origin in MODEL_ORIGINS)


def is_unreviewed(ann: Any) -> bool:
    """Model output that has not been touched or explicitly reviewed."""
    prov = getattr(ann, "provenance", None)
    if not prov or prov.origin not in MODEL_ORIGINS:
        return False
    if prov.reviewed:
        return False
    if (prov.geometry_hash_at_creation
            and geometry_hash(ann) != prov.geometry_hash_at_creation):
        return False  # edited ⇒ effectively reviewed
    return True


def summarize(annotations) -> dict:
    """Counts by origin + unreviewed model-output count for a status indicator."""
    by_origin: dict[str, int] = {}
    unreviewed = 0
    total = 0
    for a in annotations:
        total += 1
        prov = getattr(a, "provenance", None)
        o = prov.origin if prov else Origin.UNKNOWN.value
        by_origin[o] = by_origin.get(o, 0) + 1
        if is_unreviewed(a):
            unreviewed += 1
    return {"total": total, "by_origin": by_origin, "unreviewed": unreviewed}


def summary_text(annotations) -> str:
    s = summarize(annotations)
    if s["total"] == 0:
        return "0 annotations"
    parts = [f"{n} {o}" for o, n in sorted(s["by_origin"].items(), key=lambda kv: -kv[1])]
    txt = f"{s['total']} annotations ({', '.join(parts)})"
    if s["unreviewed"]:
        txt += f"  ⚠ {s['unreviewed']} unreviewed model prediction(s)"
    return txt


# ── study-mode recorder (event + conversation logs, manifest) ─────────────────

class Recorder:
    """Append-only JSONL event/conversation logger. Inactive until ``start``."""

    def __init__(self) -> None:
        self.active = False
        self.run_dir: Optional[Path] = None
        self.annotator_id: Optional[str] = None
        self.condition: Optional[str] = None
        self._lock = threading.Lock()
        self._events: Optional[Any] = None
        self._conv: Optional[Any] = None

    def start(self, run_dir, annotator_id: str, condition: str, manifest: dict) -> Path:
        self.stop()
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.annotator_id = annotator_id
        self.condition = condition
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._events = open(run_dir / "events.jsonl", "a", buffering=1)
        self._conv = open(run_dir / "conversation.jsonl", "a", buffering=1)
        self.active = True
        self.log("session_start", annotator_id=annotator_id, condition=condition)
        return run_dir

    def stop(self) -> None:
        if self.active:
            try:
                self.log("session_end")
            except Exception:
                pass
        self.active = False
        for h in (self._events, self._conv):
            try:
                if h:
                    h.close()
            except Exception:
                pass
        self._events = self._conv = None

    def log(self, event_type: str, **fields) -> None:
        if not self.active or self._events is None:
            return
        rec = {"event": event_type, "t_wall": now_iso(), "t_mono": now_mono(),
               "annotator_id": self.annotator_id, "condition": self.condition, **fields}
        with self._lock:
            try:
                self._events.write(json.dumps(rec) + "\n")
            except Exception:
                pass

    def log_conversation(self, **fields) -> None:
        if not self.active or self._conv is None:
            return
        rec = {"t_wall": now_iso(), "t_mono": now_mono(),
               "annotator_id": self.annotator_id, "condition": self.condition, **fields}
        with self._lock:
            try:
                self._conv.write(json.dumps(rec) + "\n")
            except Exception:
                pass


_RECORDER = Recorder()


def recorder() -> Recorder:
    return _RECORDER


def log_event(event_type: str, **fields) -> None:
    _RECORDER.log(event_type, **fields)


# ── study-mode preconditions ──────────────────────────────────────────────────

def find_sidecars(dataset_dir) -> list:
    """All ``.<stem>.acorn.json`` provenance sidecars under a dataset dir."""
    d = Path(dataset_dir)
    if not d.exists():
        return []
    return [p for p in d.rglob(".*.acorn.json")]


def build_manifest(annotator_id: str, condition: str, dataset_dir: str,
                   acorn_version: str = "", git_sha: str = "",
                   checkpoints=None, hardware=None,
                   screen_resolution: str = "") -> dict:
    return {
        "annotator_id": annotator_id,
        "condition": condition,
        "dataset_dir": str(dataset_dir),
        "acorn_version": acorn_version,
        "git_sha": git_sha,
        "checkpoints": list(checkpoints or []),
        "hardware": hardware or {},
        "screen_resolution": screen_resolution,
        "start_time": now_iso(),
        "start_mono": now_mono(),
    }
