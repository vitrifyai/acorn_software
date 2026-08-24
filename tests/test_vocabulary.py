"""
The domain vocabulary: what scientists call things vs what the models recognise.

Measured on a simulated cryo-TEM micrograph with 30 dark round particles, SAM 3
returns 0 masks for "nanoparticle" and "vesicle" but 29 for "dark round blob" —
and 0 again for "dark blob". The mapping is not guessable, which is the whole
reason this module exists.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from acorn.core import vocabulary as V


# ── resolving what people type ────────────────────────────────────────────────

@pytest.mark.parametrize("word,expected", [
    ("vesicle", "vesicle"), ("vesicles", "vesicle"), ("VESICLES", "vesicle"),
    ("  vesicle  ", "vesicle"),
    ("nanoparticle", "nanoparticle"), ("nanoparticles", "nanoparticle"),
    ("np", "nanoparticle"), ("gold nanoparticles", "nanoparticle"),
    ("flagella", "filament"), ("microtubule", "filament"),
    ("grain boundaries", "grain boundary"),
    ("virion", "virus"), ("bacteria", "bacterium"),
    ("voids", "pore"),
])
def test_words_resolve_to_the_right_term(word, expected):
    term = V.resolve(word)
    assert term is not None, f"{word!r} did not resolve"
    assert term.canonical == expected


def test_an_unknown_word_resolves_to_nothing_but_is_still_tried():
    assert V.resolve("widget") is None
    # passing it through is deliberate: it may be everyday language the model knows
    assert V.prompts_for("widget") == ("widget",)
    assert V.prompts_for("") == ()


def test_every_synonym_is_unique_across_terms():
    seen: dict[str, str] = {}
    for term in V.TERMS:
        for name in (term.canonical, *term.synonyms):
            key = name.lower()
            assert key not in seen or seen[key] == term.canonical, (
                f"{name!r} claimed by both {seen.get(key)} and {term.canonical}"
            )
            seen[key] = term.canonical


# ── the phrases themselves ────────────────────────────────────────────────────

def test_every_term_has_phrasing_for_both_polarities():
    """The same object is dark in cryo-TEM and bright in HAADF-STEM."""
    for term in V.TERMS:
        assert term.prompts("dark"), f"{term.canonical} has no dark phrasing"
        assert term.prompts("light"), f"{term.canonical} has no light phrasing"


def test_phrases_avoid_the_jargon_that_returns_nothing():
    """
    The measured failure mode: the discipline's own word finds nothing. A phrase
    that merely repeats the term name would be a mapping that does no work.
    """
    for term in V.TERMS:
        for polarity in ("dark", "light"):
            for phrase in term.prompts(polarity):
                assert phrase.lower() != term.canonical.lower(), (
                    f"{term.canonical}: phrase is just the term again"
                )


def test_phrases_carry_a_tone_word_matching_the_polarity():
    dark_words = ("dark", "black")
    light_words = ("bright", "white")
    for term in V.TERMS:
        assert any(any(w in p.lower() for w in dark_words) for p in term.prompts("dark")), \
            f"{term.canonical}: no dark-toned phrase"
        assert any(any(w in p.lower() for w in light_words) for p in term.prompts("light")), \
            f"{term.canonical}: no light-toned phrase"


def test_the_measured_terms_use_the_measured_phrase_first():
    """'dark round blob' scored 29/30; it must stay the first thing tried."""
    for word in ("nanoparticle", "particle"):
        assert V.prompts_for(word, "dark")[0] == "dark round blob"


def test_both_disciplines_are_covered():
    bio = V.known_terms("biology")
    mat = V.known_terms("materials")
    assert len(bio) >= 10 and len(mat) >= 10
    names = {t.canonical for t in V.TERMS}
    for expected in ("vesicle", "membrane", "filament", "bacterium",
                     "nanoparticle", "grain", "pore", "crack"):
        assert expected in names


def test_autocomplete_suggests_canonical_names():
    assert "nanoparticle" in V.suggest("nano")
    assert "vesicle" in V.suggest("ves")
    assert V.suggest("") != []


# ── what each backend will actually do with the word ──────────────────────────

def test_only_text_capable_backends_claim_to_use_the_word():
    assert V.backend_uses_text("sam3")
    for backend in ("sam2", "sam", "usam", "yolo", "unet"):
        assert not V.backend_uses_text(backend), f"{backend} cannot take a text prompt"


def test_each_backend_explains_itself():
    for backend in ("sam2", "usam", "yolo", "unet"):
        note = V.backend_note(backend)
        assert note and ("label" in note.lower() or "train" in note.lower())
    assert "text prompt" in V.backend_note("sam3").lower()


# ── polarity detection, which picks the phrasing ──────────────────────────────

def _planted(polarity: str):
    yy, xx = np.mgrid[0:128, 0:128]
    base = np.full((128, 128), 128.0)
    blob = sum(60 * np.exp(-(((yy-cy)**2 + (xx-cx)**2) / (2*6.0**2)))
               for cy, cx in [(40, 40), (90, 90)])
    arr = base - blob if polarity == "dark" else base + blob
    return np.clip(arr, 0, 255).astype("uint8")


def test_polarity_is_read_from_the_image():
    from acorn.core.sam_predictor import SAMPredictor
    assert SAMPredictor.image_polarity(_planted("dark")) == "dark"
    assert SAMPredictor.image_polarity(_planted("light")) == "light"
    assert SAMPredictor.image_polarity(np.array([], dtype="uint8")) == "dark"


def test_text_prompt_on_a_backend_without_one_says_what_to_use_instead():
    from acorn.core.sam_predictor import SAMPredictor
    p = SAMPredictor(backend="sam2")
    p._active_backend = "sam2"
    p._predictor = object()          # bypass the load check
    with pytest.raises(RuntimeError, match="SAM 3|point, box"):
        p.predict_text(np.zeros((16, 16), "uint8"), "vesicles")
