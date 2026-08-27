"""
Translating what scientists call things into what the models recognise.

SAM 3 and the open-vocabulary YOLO variants accept a text prompt, but their
vocabulary is everyday visual language, not the terminology of a discipline.
Measured on a simulated cryo-TEM micrograph with 30 dark round particles:

    "dark round blob"      29 masks    "round dark object"     0 masks
    "black dot"            28 masks    "dark blob"             0 masks
    "round black particle" 28 masks    "dark grain"            0 masks
    "dark circle"          28 masks    "dark ellipse"          0 masks
    "black circle"         25 masks    "nanoparticle"          0 masks
    "dark spot"            15 masks    "vesicle"               0 masks

So the words a microscopist actually uses return nothing, while the right
everyday phrase finds almost everything — and the difference between a phrase
that works and one that does not ("dark round blob" against "dark blob") is not
guessable. This module holds the mapping so nobody has to guess.

A term also carries its appearance, because the same object is dark on a bright
field in cryo-TEM and bright on a dark field in HAADF-STEM. `prompts_for` takes
the polarity so the phrase matches the image in front of you.

HONESTY ABOUT COVERAGE: the numbers above are measured. The phrasing for the
other terms is reasoned from the same pattern — concrete shape and tone words,
avoiding jargon — and is NOT individually validated. Terms carry `validated` so
the interface can say which is which, and every term's phrasing can be
overridden by the user.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Polarity = Literal["dark", "light"]
Domain = Literal["biology", "materials", "shared"]


@dataclass(frozen=True)
class Term:
    """One thing people name, and the phrases a text-prompted model responds to."""

    canonical: str
    domain: Domain
    synonyms: tuple[str, ...] = ()
    # Phrases for features DARKER than the field (cryo-TEM, bright-field).
    dark: tuple[str, ...] = ()
    # Phrases for features BRIGHTER than the field (HAADF-STEM, dark-field, SEM).
    light: tuple[str, ...] = ()
    validated: bool = False
    note: str = ""

    def prompts(self, polarity: Polarity = "dark") -> tuple[str, ...]:
        primary = self.dark if polarity == "dark" else self.light
        return primary or self.dark or self.light


# Shape language that the models demonstrably respond to. Kept as building
# blocks so a new term is phrased like the ones that were measured.
_ROUND_DARK  = ("dark round blob", "black dot", "dark circle")
_ROUND_LIGHT = ("bright round blob", "white dot", "bright circle")
_LONG_DARK   = ("long thin dark line", "dark thin filament", "dark elongated shape")
_LONG_LIGHT  = ("long thin bright line", "bright thin filament", "bright elongated shape")
_BLOB_DARK   = ("dark irregular blob", "dark patch", "dark shape")
_BLOB_LIGHT  = ("bright irregular blob", "bright patch", "bright shape")


TERMS: tuple[Term, ...] = (
    # ── biology ───────────────────────────────────────────────────────────────
    Term("vesicle", "biology", ("vesicles", "liposome", "liposomes", "endosome"),
         dark=("dark round blob", "dark circle", "round dark membrane"),
         light=_ROUND_LIGHT,
         note="A membrane-bounded sphere; reads as a ring when the membrane is resolved."),
    Term("particle", "shared", ("particles", "protein particle", "single particle"),
         dark=_ROUND_DARK, light=_ROUND_LIGHT, validated=True,
         note="Measured: 28-29 of 30 on a simulated cryo-TEM micrograph."),
    Term("nanoparticle", "materials", ("nanoparticles", "np", "nps", "colloid"),
         dark=_ROUND_DARK, light=_ROUND_LIGHT, validated=True,
         note="Measured: 'dark round blob' found 29 of 30; the word itself found 0."),
    Term("virus", "biology", ("virion", "virions", "viruses", "capsid", "phage"),
         dark=("dark round blob", "dark hexagonal shape", "dark circle"),
         light=_ROUND_LIGHT),
    # "rod" alone is left to nanorod: in materials it is the usual shorthand,
    # while a biologist saying "rod" means rod-shaped, which is spelled out here.
    Term("bacterium", "biology",
         ("bacteria", "bacterial cell", "microbe", "rod-shaped bacterium", "bacillus"),
         dark=("dark rounded rectangle", "dark capsule shape", "large dark blob"),
         light=("bright rounded rectangle", "bright capsule shape", "large bright blob")),
    Term("cell", "biology", ("cells",),
         dark=("large dark blob", "dark rounded shape"),
         light=("large bright blob", "bright rounded shape")),
    Term("membrane", "biology", ("membranes", "bilayer", "lipid bilayer", "envelope"),
         dark=("dark thin curved line", "dark thin line", "dark outline"),
         light=("bright thin curved line", "bright thin line", "bright outline")),
    Term("ribosome", "biology", ("ribosomes",),
         dark=("small dark dot", "tiny dark speck"),
         light=("small bright dot", "tiny bright speck")),
    Term("filament", "biology", ("filaments", "fibril", "fibrils", "fiber", "fibre",
                                 "actin", "microtubule", "flagellum", "flagella", "pilus"),
         dark=_LONG_DARK, light=_LONG_LIGHT),
    Term("mitochondrion", "biology", ("mitochondria",),
         dark=("dark oval blob", "dark elongated blob"),
         light=("bright oval blob", "bright elongated blob")),
    Term("spore", "biology", ("spores", "endospore"),
         dark=("dark oval blob", "dark round blob"), light=_ROUND_LIGHT),
    Term("ice contamination", "biology", ("ice", "contamination", "crystalline ice"),
         dark=("dark angular shape", "dark hexagonal shape", "dark sharp-edged patch"),
         light=("bright angular shape", "bright hexagonal shape")),

    # ── materials ─────────────────────────────────────────────────────────────
    Term("nanorod", "materials", ("nanorods", "rod", "rods", "whisker"),
         dark=("dark elongated shape", "dark thick line", "dark rectangle"),
         light=("bright elongated shape", "bright thick line", "bright rectangle")),
    Term("nanowire", "materials", ("nanowires", "wire", "nanotube", "nanotubes", "cnt"),
         dark=_LONG_DARK, light=_LONG_LIGHT),
    Term("grain", "materials", ("grains", "crystallite", "crystallites"),
         dark=("dark angular patch", "dark polygon", "dark irregular blob"),
         light=("bright angular patch", "bright polygon", "bright irregular blob")),
    Term("grain boundary", "materials", ("grain boundaries", "boundary", "interface"),
         dark=("dark thin line between regions", "dark thin line"),
         light=("bright thin line between regions", "bright thin line")),
    Term("pore", "materials", ("pores", "void", "voids", "hole", "holes", "porosity"),
         dark=("dark round hole", "dark round blob"),
         light=("bright round hole", "bright round blob")),
    Term("precipitate", "materials", ("precipitates", "inclusion", "second phase"),
         dark=("dark round blob", "dark angular patch"),
         light=("bright round blob", "bright angular patch")),
    Term("crack", "materials", ("cracks", "fracture", "fissure"),
         dark=("dark jagged line", "dark thin line"),
         light=("bright jagged line", "bright thin line")),
    Term("dislocation", "materials", ("dislocations", "defect line"),
         dark=("dark curved line", "dark thin line"),
         light=("bright curved line", "bright thin line")),
    Term("thin film", "materials", ("film", "layer", "coating", "lamella"),
         dark=("dark horizontal band", "dark stripe"),
         light=("bright horizontal band", "bright stripe")),
    Term("catalyst particle", "materials", ("catalyst", "catalysts", "cluster", "clusters"),
         dark=_ROUND_DARK, light=_ROUND_LIGHT),
    Term("powder", "materials", ("powders", "agglomerate", "aggregate"),
         dark=_BLOB_DARK, light=_BLOB_LIGHT),
)


_INDEX: dict[str, Term] = {}
for _t in TERMS:
    for _name in (_t.canonical, *_t.synonyms):
        _INDEX[_name.strip().lower()] = _t


def resolve(word: str) -> Term | None:
    """
    The term a user's word refers to, or None if it is not one we know.

    Matches the canonical name and every synonym, ignoring case, surrounding
    space, and a trailing plural 's' that is not already covered.
    """
    key = (word or "").strip().lower()
    if not key:
        return None
    if key in _INDEX:
        return _INDEX[key]
    if key.endswith("s") and key[:-1] in _INDEX:
        return _INDEX[key[:-1]]
    # "gold nanoparticles" -> nanoparticle: try the last word, then the last two
    parts = key.split()
    for candidate in (parts[-1], " ".join(parts[-2:]) if len(parts) > 1 else ""):
        if candidate in _INDEX:
            return _INDEX[candidate]
        if candidate.endswith("s") and candidate[:-1] in _INDEX:
            return _INDEX[candidate[:-1]]
    return None


def prompts_for(word: str, polarity: Polarity = "dark") -> tuple[str, ...]:
    """
    Phrases to try for *word*, best first, or the word itself if unknown.

    Falling back to the raw word is deliberate: an unknown term may still be
    everyday language the model understands, and refusing to try would be worse
    than trying and finding nothing.
    """
    term = resolve(word)
    if term is None:
        return ((word or "").strip(),) if (word or "").strip() else ()
    return term.prompts(polarity)


def known_terms(domain: Domain | None = None) -> tuple[Term, ...]:
    """Every term, optionally just one discipline's."""
    if domain is None:
        return TERMS
    return tuple(t for t in TERMS if t.domain in (domain, "shared"))


def suggest(prefix: str, limit: int = 8) -> list[str]:
    """Canonical names starting with *prefix* — for autocomplete in a label box."""
    key = (prefix or "").strip().lower()
    if not key:
        return [t.canonical for t in TERMS][:limit]
    hits = [name for name in sorted(_INDEX) if name.startswith(key)]
    seen, out = set(), []
    for name in hits:
        canonical = _INDEX[name].canonical
        if canonical not in seen:
            seen.add(canonical)
            out.append(canonical)
    return out[:limit]


# ── which backends can use any of this ────────────────────────────────────────

TEXT_CAPABLE_BACKENDS = frozenset({"sam3", "yolo_world", "yoloe"})

_BACKEND_NOTE = {
    "sam3":   "SAM 3 takes a text prompt directly.",
    "sam2":   "SAM 2 has no text input; the word is used as the annotation label only. "
              "Use a point, box or scribble prompt, or switch to SAM 3.",
    "sam":    "SAM 1 has no text input; the word is used as the annotation label only.",
    "usam":   "micro-SAM has no text input; the word is used as the annotation label only.",
    "yolo":   "A trained YOLO detects only the classes it was trained on. The word is "
              "used as the annotation label; train a model on this class to detect it.",
    "unet":   "UNet segments the classes it was trained on. The word names the output, "
              "it does not steer the model.",
}


def backend_uses_text(backend: str) -> bool:
    """True when the word steers the model rather than just naming the result."""
    return (backend or "").strip().lower() in TEXT_CAPABLE_BACKENDS


def backend_note(backend: str) -> str:
    """One line explaining what this backend will do with the word."""
    key = (backend or "").strip().lower()
    return _BACKEND_NOTE.get(key, "This backend uses the word as the annotation label.")
