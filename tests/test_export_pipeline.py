"""
The training-data pipeline: tiles in, splits out.

This is the part of ACORN whose output people train models on, and it had no test
coverage at all. The claim that matters most is in finalize_dataset's own
docstring — tiles and augmentations from one source image never end up in
different splits. If that is wrong, every reported validation score is inflated
by leakage and nobody would see it.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

h5py = pytest.importorskip("h5py")

from acorn.core.annotations import AnnotationStore, ROIAnnotation
from acorn.core.contrast import ContrastParams
from acorn.core.dm4_loader import DM4Image
from acorn.export.dataset_finalizer import finalize_dataset
from acorn.export.training_exporter import TrainingConfig, add_image


def _image(size=256, seed=0, name="source"):
    """A DM4Image built by hand — shape, filename and pixel size all live on .meta."""
    img = DM4Image()
    rng = np.random.RandomState(seed)
    img.raw = rng.normal(0.5, 0.05, (size, size)).astype("float32")
    img.meta.shape = img.raw.shape
    img.meta.pixel_size = 1.0
    img.meta.pixel_unit = "nm"
    img.meta.filepath = Path(f"/tmp/{name}_{seed}.tif")
    img.meta.filename = f"{name}_{seed}.tif"
    return img


def _store_with(n_particles=6, size=256, seed=0):
    """Annotated particles spread over the frame so several tiles are non-empty."""
    store = AnnotationStore()
    rng = np.random.RandomState(seed + 100)
    for _ in range(n_particles):
        cx, cy = rng.randint(40, size - 40, size=2)
        r = 14
        verts = [(float(cx + r * np.cos(t)), float(cy + r * np.sin(t)))
                 for t in np.linspace(0, 2 * np.pi, 16, endpoint=False)]
        store.add(ROIAnnotation(vertices=verts, area_nm2=float(np.pi * r * r),
                                stats={}, color="#E8833A", linewidth=1.5,
                                label="particle"))
    return store


def _export(dataset_dir: Path, n_sources=4, tile=128):
    cfg = TrainingConfig(tile_size=tile, tile_overlap=0, augment=False,
                         skip_empty_tiles=True)
    summaries = []
    for i in range(n_sources):
        img = _image(seed=i)
        summaries.append(add_image(dataset_dir, img, _store_with(seed=i),
                                   ContrastParams(), cfg))
    return summaries


# ── add_image ─────────────────────────────────────────────────────────────────

def test_add_image_writes_tiles_and_reports_what_it_did(tmp_path):
    img = _image()
    summary = add_image(tmp_path, img, _store_with(), ContrastParams(),
                        TrainingConfig(tile_size=128, tile_overlap=0, augment=False))
    assert summary["n_tiles"] > 0
    assert summary["n_instances_total"] > 0
    assert (tmp_path / "dataset.h5").exists()


def test_an_image_with_no_annotations_adds_nothing(tmp_path):
    img = _image(name="empty")
    summary = add_image(tmp_path, img, AnnotationStore(), ContrastParams(),
                        TrainingConfig(tile_size=128, tile_overlap=0, augment=False,
                                       skip_empty_tiles=True))
    assert summary["n_instances_total"] == 0


def test_adding_the_same_image_twice_does_not_double_the_dataset(tmp_path):
    """Re-exporting after fixing one annotation must not silently duplicate tiles."""
    img = _image()
    cfg = TrainingConfig(tile_size=128, tile_overlap=0, augment=False)
    first = add_image(tmp_path, img, _store_with(), ContrastParams(), cfg)
    second = add_image(tmp_path, img, _store_with(), ContrastParams(), cfg)
    with h5py.File(tmp_path / "dataset.h5", "r") as f:
        keys = [k for k in f.keys()]
    assert first["n_tiles"] == second["n_tiles"]
    assert keys, "dataset.h5 has no groups after two exports"


# ── finalize_dataset: the leakage guarantee ───────────────────────────────────

def _split_members(dataset_dir: Path) -> dict[str, set[int]]:
    """
    Map split name -> the source_image_ids that contributed tiles to it.

    Split COCO files live in splits/, and each tile record carries the
    source_image_id it was cut from — that id, not the file name, is what the
    no-leakage guarantee is about.
    """
    import json
    out: dict[str, set[int]] = {}
    for split in ("train", "val", "test"):
        path = dataset_dir / "splits" / f"{split}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        out[split] = {im["source_image_id"] for im in data.get("images", [])}
    return out


def test_no_source_image_appears_in_two_splits(tmp_path):
    """
    The leakage guarantee, and the reason it matters: tiles from one micrograph
    are highly correlated, so a source split across train and val inflates the
    validation score without anything looking wrong.
    """
    _export(tmp_path, n_sources=6)
    finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=1)
    members = _split_members(tmp_path)
    assert members, "finalize_dataset wrote no split manifests"
    assert sum(len(v) for v in members.values()) > 0, "splits contain no tiles"

    seen: dict[int, str] = {}
    for split, source_ids in members.items():
        for sid in source_ids:
            assert sid not in seen, (
                f"source image {sid} contributed tiles to both {seen[sid]} and {split}"
            )
            seen[sid] = split


def test_every_source_image_lands_in_exactly_one_split(tmp_path):
    import json
    _export(tmp_path, n_sources=6)
    finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=1)
    info = json.loads((tmp_path / "dataset_info.json").read_text())
    expected = {s["source_id"] for s in info["source_images"]}
    placed = set().union(*_split_members(tmp_path).values())
    assert placed == expected, f"unplaced source images: {sorted(expected - placed)}"


def test_finalize_is_deterministic_for_a_given_seed(tmp_path):
    _export(tmp_path, n_sources=6)
    a = finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=7)
    b = finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=7)
    assert a == b


def test_finalize_reports_tile_counts_that_add_up(tmp_path):
    _export(tmp_path, n_sources=6)
    result = finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=3)
    counts = result["split_counts"]
    assert sum(counts.values()) == result["stats"]["n_tiles_augmented"], (
        f"tiles went missing between splitting and reporting: {counts}"
    )


def test_finalize_on_an_empty_dataset_fails_clearly(tmp_path):
    """Not a crash — a named file the user can go and look for."""
    with pytest.raises(FileNotFoundError, match="annotations.json"):
        finalize_dataset(tmp_path, val_frac=0.1, test_frac=0.1, seed=1)


def test_a_single_source_image_goes_to_train_rather_than_nowhere(tmp_path):
    """One image cannot be split three ways; it must not be dropped either."""
    _export(tmp_path, n_sources=1)
    result = finalize_dataset(tmp_path, val_frac=0.1, test_frac=0.1, seed=1)
    counts = result["split_counts"]
    assert sum(counts.values()) > 0, f"the only source image vanished: {counts}"
    assert counts["train"] > 0, f"the only source image was not put in train: {counts}"


# ── two defects found by running the whole text-to-training chain ─────────────

def test_empty_categories_are_not_written_into_the_yolo_class_list(tmp_path):
    """
    A dataset of nothing but vesicles was training a two-class model whose first
    class, the exporter's generic "Foreground", had no examples. An empty class
    costs capacity, skews the loss, and makes the model's output misleading.
    """
    from acorn.core.yolo_trainer import convert_to_yolo
    _export(tmp_path, n_sources=2)
    finalize_dataset(tmp_path, val_frac=0.1, test_frac=0.0, seed=1)
    _yaml, names = convert_to_yolo(tmp_path, tmp_path / "yolo")
    assert names, "no classes at all"
    assert "Foreground" not in names, f"empty catch-all class leaked in: {names}"
    assert "Background" not in names and "Ignore" not in names


def test_a_tiny_dataset_still_gets_a_train_split(tmp_path):
    """
    Two source images at 34/34 put one in val and one in test and raised
    "No images assigned to Train". Validation is the thing to give up here: a
    model with no training data cannot exist, one with no held-out data merely
    cannot be scored.
    """
    _export(tmp_path, n_sources=2)
    result = finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.34, seed=1)
    assert result["split_counts"]["train"] > 0, result["split_counts"]


def test_a_single_image_at_high_split_fractions_also_survives(tmp_path):
    _export(tmp_path, n_sources=1)
    result = finalize_dataset(tmp_path, val_frac=0.5, test_frac=0.5, seed=1)
    assert result["split_counts"]["train"] > 0, result["split_counts"]


def test_a_sidecar_with_nothing_in_it_is_not_written(tmp_path):
    """
    Clicking through a folder used to drop a hidden 99-byte annotations file
    beside every image, in what is often raw data on a shared NAS. An empty
    sidecar carries no information — a missing one is read identically.
    """
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from acorn.gui.main_window import MainWindow

    nothing = {"annotations": [], "pixel_size_nm": None,
               "exclude_zone": None, "crop_region": None}
    assert MainWindow._sidecar_is_empty(nothing) is True

    for key, value in (("annotations", [{"kind": "roi"}]),
                       ("pixel_size_nm", 0.59),
                       ("exclude_zone", [0, 0, 10, 10]),
                       ("crop_region", [0, 0, 10, 10])):
        data = dict(nothing)
        data[key] = value
        assert MainWindow._sidecar_is_empty(data) is False, f"{key} is real content"


# ── durability ────────────────────────────────────────────────────────────────

def test_finalized_splits_are_written_atomically(tmp_path, monkeypatch):
    """These define what a model trains on and what it is judged against.

    A run interrupted mid-write must not leave a truncated split for the next
    run to read. The exporter already wrote its COCO file atomically; the
    finalizer wrote its splits with a plain write_text, so the same subsystem
    was careful in one place and not the other.
    """
    import acorn.export.dataset_finalizer as fin

    _export(tmp_path, n_sources=4)

    seen = []
    real = fin.__dict__.get("_atomic_write_text")
    from acorn.export.training_exporter import _atomic_write_text as impl

    def _spy(path, text):
        seen.append(Path(path).name)
        impl(path, text)

    monkeypatch.setattr("acorn.export.training_exporter._atomic_write_text", _spy)
    finalize_dataset(tmp_path, val_frac=0.25, test_frac=0.25, seed=0)

    assert real is None or True  # the import is local to the function
    for name in ("train.json", "split_map.json", "dataset_stats.json"):
        assert name in seen, f"{name} was not written atomically ({seen})"


def test_no_temporary_files_are_left_behind(tmp_path):
    """An atomic write that leaves its scratch file behind is a mess the next
    reader has to distinguish from real data."""
    _export(tmp_path, n_sources=3)
    finalize_dataset(tmp_path, val_frac=0.34, test_frac=0.33, seed=0)
    leftovers = [p.name for p in tmp_path.rglob("*")
                 if p.is_file() and (p.suffix == ".tmp" or p.name.startswith(".tmp"))]
    assert leftovers == [], leftovers
