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


# ── the text-to-training path ─────────────────────────────────────────────────
# The achievable route to "YOLO knows biology words": SAM 3 finds them from a
# description, the annotations are accepted and exported, and a YOLO trained on
# them knows the class by name. YOLO-World, the open-vocabulary alternative, was
# measured on cryo-TEM and FIB-SEM simulations and found 0-1 of 30 particles at
# confidence 0.05, so it is deliberately not offered.

def test_clu_can_ask_for_a_word_directly():
    from acorn_llm.agent import _TOOLS, _KNOWN_TOOLS
    from acorn_llm.tool_scope import scope_tools
    assert "run_sam_text" in _KNOWN_TOOLS
    tool = next(t for t in _TOOLS if t["name"] == "run_sam_text")
    assert "label" in tool["properties"] and tool["required"] == ["label"]
    # it belongs to Annotate, and nowhere else
    assert "run_sam_text" in {t["name"] for t in scope_tools(_TOOLS, "annotate")}
    assert "run_sam_text" not in {t["name"] for t in scope_tools(_TOOLS, "simulate")}


def test_the_tool_description_says_which_backends_can_do_this():
    from acorn_llm.agent import _TOOLS
    tool = next(t for t in _TOOLS if t["name"] == "run_sam_text")
    text = tool["description"].lower()
    assert "sam 3" in text
    for cannot in ("sam 2", "yolo", "unet"):
        assert cannot in text, f"the description never says {cannot} cannot do this"


def test_the_window_exposes_the_action():
    from acorn.gui.main_window import MainWindow
    assert callable(getattr(MainWindow, "run_sam_text", None))


# ── tone, not just shape ──────────────────────────────────────────────────────
# Measured on real cryo-TEM vesicles against 43 outlines drawn by the
# microscopist: "faint grey circle" recalled 0.88 and "grey round blob" 0.91,
# where "dark round blob" — the phrasing this term used to carry — managed 0.35.
# Every phrase naming the membrane or the ring returned nothing at all. These
# tests pin the ordering, because the win is entirely in which phrase goes first.

def test_vesicle_tries_a_grey_phrasing_before_a_dark_one():
    from acorn.core import vocabulary
    phrases = vocabulary.prompts_for("vesicle", "dark")
    assert phrases[0] == "faint grey circle"
    assert "grey round blob" in phrases[:2]
    # the old phrasing is kept as a fallback, not discarded
    assert "dark round blob" in phrases


def test_the_grey_phrasing_is_reachable_within_the_truncation_limit():
    # predict_text does prompts_for(...)[:max_phrases]. A fallback sitting beyond
    # that limit is dead code, so the tuples and the limit have to stay in step.
    import inspect
    from acorn.core import vocabulary
    from acorn.core.sam_predictor import SAMPredictor

    limit = inspect.signature(SAMPredictor.predict_text).parameters["max_phrases"].default
    for word in ("vesicle", "particle", "nanoparticle", "virus", "spore"):
        reachable = vocabulary.prompts_for(word, "dark")[:limit]
        assert any("grey" in p for p in reachable), (
            f"{word} has no grey fallback within the {limit} phrases that get used"
        )


def test_no_term_hides_phrases_beyond_the_truncation_limit_unreached():
    # Every phrase past the limit is never tried. Long tuples are fine, but the
    # limit must be big enough that the deliberate fallbacks are actually reached.
    import inspect
    from acorn.core import vocabulary
    from acorn.core.sam_predictor import SAMPredictor

    limit = inspect.signature(SAMPredictor.predict_text).parameters["max_phrases"].default
    assert limit >= 6, "the round-object tuples run to six or more; raise the limit"


def test_vesicle_puts_every_grey_phrase_before_every_dark_one():
    # The whole measured result is that grey beats dark on this sample type, and
    # predict_text stops at the first phrase that returns anything — so a dark
    # phrase sneaking in front of a grey one would silently undo it.
    from acorn.core import vocabulary
    phrases = vocabulary.prompts_for("vesicle", "dark")
    last_grey = max(i for i, p in enumerate(phrases) if "grey" in p)
    first_dark = min(i for i, p in enumerate(phrases)
                     if "dark" in p or "black" in p)
    assert last_grey < first_dark


def test_phrases_measured_to_return_nothing_are_not_padding_any_term():
    # Lengthening the lists is only useful if the additions work. These were
    # measured at zero masks and must not be used to pad a tuple.
    from acorn.core import vocabulary
    dead = {"dark blob", "round dark object", "dark grain", "dark ellipse",
            "round dark membrane", "faint round ring", "circle outline",
            "round vesicle membrane", "donut", "nanoparticle", "vesicle"}
    for term in vocabulary.TERMS:
        for phrase in (*term.dark, *term.light):
            assert phrase not in dead, f"{term.canonical} uses dead phrase {phrase!r}"


def test_membrane_and_ring_phrasings_are_not_used_for_vesicles():
    # These describe what a person sees and return 0 masks; measured.
    from acorn.core import vocabulary
    phrases = vocabulary.prompts_for("vesicle", "dark")
    for dead in ("round dark membrane", "faint round ring", "circle outline",
                 "round vesicle membrane", "donut"):
        assert dead not in phrases


def test_vesicle_synonyms_get_the_measured_phrasing_too():
    from acorn.core import vocabulary
    for word in ("vesicles", "liposome", "liposomes", "endosome"):
        assert vocabulary.prompts_for(word, "dark")[0] == "faint grey circle"


def test_bright_field_phrasing_is_untouched():
    # The grey result is about faint dark-on-bright cryo images; the light
    # polarity (HAADF-STEM, dark-field, SEM) was not measured and must not move.
    from acorn.core import vocabulary
    assert vocabulary.prompts_for("vesicle", "light")[0] == "bright round blob"


def test_the_vesicle_term_now_claims_to_be_validated():
    from acorn.core import vocabulary
    term = vocabulary.resolve("vesicle")
    assert term.validated is True
    assert "0.88" in term.note or "0.91" in term.note


# ── SAM regions carry their size ──────────────────────────────────────────────
# Every SAM path used to store area_nm2=0.0 and stats={}, so an accepted mask
# had an outline but no measurement: getting a diameter out meant recomputing it
# from the vertices somewhere else. These pin the size to the annotation itself.

def _sam_controller(pixel_size: float):
    import types
    from acorn.gui.sam_controller import SAMControllerMixin
    c = SAMControllerMixin.__new__(SAMControllerMixin)
    c._engine = types.SimpleNamespace(pixel_size=pixel_size)
    c._sam_color_for_label = lambda label: "#E8833A"
    return c


def _circle(radius_px: float, n: int = 60):
    import numpy as np
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [(100 + radius_px * np.cos(a), 100 + radius_px * np.sin(a)) for a in t]


def test_a_sam_region_knows_its_diameter():
    roi = _sam_controller(1.0)._roi_from_sam(_circle(20.0), "vesicle")
    assert roi.stats["ecd_nm"] == pytest.approx(40.0, abs=0.2)
    assert roi.area_nm2 == pytest.approx(3.14159 * 400, rel=0.01)


def test_the_diameter_scales_with_the_pixel_size():
    # same polygon, 0.5 nm/px -> half the diameter in nanometres
    roi = _sam_controller(0.5)._roi_from_sam(_circle(20.0), "vesicle")
    assert roi.stats["ecd_nm"] == pytest.approx(20.0, abs=0.2)


def test_a_sam_region_carries_shape_as_well_as_size():
    roi = _sam_controller(1.0)._roi_from_sam(_circle(20.0), "vesicle")
    for key in ("ecd_nm", "area_nm2", "perimeter_nm", "circularity",
                "aspect_ratio", "feret_nm"):
        assert key in roi.stats, f"{key} missing from a SAM region"
    assert roi.stats["circularity"] == pytest.approx(1.0, abs=0.02)


def test_the_size_survives_into_the_sidecar():
    import json
    from acorn.core.annotations import AnnotationStore
    store = AnnotationStore()
    store.add(_sam_controller(1.0)._roi_from_sam(_circle(20.0), "vesicle"))
    written = json.loads(store.to_json())
    assert written[0]["stats"]["ecd_nm"] == pytest.approx(40.0, abs=0.2)


def test_an_uncalibrated_image_reports_no_size_rather_than_a_wrong_one():
    # pixel size 0 means the scale is unknown; inventing a number in pixels and
    # calling it nanometres would be worse than reporting nothing.
    roi = _sam_controller(0.0)._roi_from_sam(_circle(20.0), "vesicle")
    assert roi.area_nm2 == 0.0
    assert roi.stats == {}
