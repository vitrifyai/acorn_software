"""
SAM 3's missing CLIP vocabulary.

The published sam3 wheel omits the assets/ directory holding
bpe_simple_vocab_16e6.txt.gz, and sam3 builds its text encoder whether or not a
typed prompt is used — so a stock install fails to load SAM 3 at all, with a bare
FileNotFoundError naming a path inside site-packages.
"""
from __future__ import annotations

import gzip
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from acorn.core import sam3_assets as A


def test_expected_path_is_where_sam3_actually_looks():
    pytest.importorskip("sam3")
    expected = A.expected_path()
    assert expected is not None
    assert expected.name == A.VOCAB_NAME
    from sam3 import model_builder
    import os.path as op
    sam3_path = op.normpath(op.join(op.dirname(model_builder.__file__), "..",
                                    "assets", A.VOCAB_NAME))
    assert str(expected) == sam3_path, "we would install it where sam3 does not look"


def test_the_installed_vocabulary_is_the_real_thing():
    """A truncated or wrong file would fail later and more confusingly."""
    pytest.importorskip("sam3")
    path = A.ensure_vocab()
    if path is None:
        pytest.skip("vocabulary not installed in this environment")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        lines = fh.read().split("\n")
    assert lines[0].startswith('"bpe_simple_vocab_16e6.txt#version')
    # CLIP consumes merges[1:49152-256-2+1]; fewer than that and the tokenizer breaks
    assert len(lines) - 1 >= 48894, f"only {len(lines) - 1} merge pairs"


def test_a_missing_vocabulary_is_reported_not_just_raised(monkeypatch, tmp_path):
    monkeypatch.setattr(A, "expected_path", lambda: tmp_path / A.VOCAB_NAME)
    monkeypatch.setattr(A, "_candidates", lambda: [])
    assert A.ensure_vocab() is None
    assert A.is_available() is False
    msg = A.missing_message()
    for expected in ("packaging bug", "SAM 2", A.VOCAB_NAME, A.SOURCE_URL):
        assert expected in msg, f"the message never mentions {expected!r}"


def test_a_copy_is_adopted_from_a_known_location(monkeypatch, tmp_path):
    """The self-heal: a copy in a shared directory gets put where sam3 reads."""
    source = tmp_path / "store" / A.VOCAB_NAME
    source.parent.mkdir()
    source.write_bytes(b"not really gzip, but this test is about the copy")
    target = tmp_path / "site-packages" / "assets" / A.VOCAB_NAME
    monkeypatch.setattr(A, "expected_path", lambda: target)
    monkeypatch.setattr(A, "_candidates", lambda: [source])
    assert A.ensure_vocab() == target
    assert target.read_bytes() == source.read_bytes()


def test_only_this_error_is_reinterpreted():
    """An unrelated OSError must not be blamed on the vocabulary."""
    assert A.looks_like_missing_vocab(
        FileNotFoundError(2, "No such file", f"/x/assets/{A.VOCAB_NAME}"))
    assert not A.looks_like_missing_vocab(FileNotFoundError(2, "No such file", "/x/model.pt"))
    assert not A.looks_like_missing_vocab(ValueError("something else entirely"))


def test_predictor_exposes_the_state_and_the_help():
    from acorn.core.sam_predictor import SAMPredictor
    assert isinstance(SAMPredictor(backend="sam3").text_prompts_available, bool)
    assert "packaging bug" in SAMPredictor.text_prompt_help()
