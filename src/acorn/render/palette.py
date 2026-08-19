"""
One palette for everything drawn on an image.

Annotation colours were being chosen in three unrelated places — a five-entry
list in the SAM controller, a different eight-entry list for spatial clusters,
and a hard-coded cyan in the simulator — so a single micrograph could carry three
colour schemes at once.

The colours here are picked for one job: reading clearly on top of a greyscale
micrograph. Grey images carry no chroma of their own, so saturated hues separate
from the data instead of competing with it, while staying dark enough not to
glare on a light field or wash out on a dark one. Ordered by how well they hold
up when there is only one label on screen, which is the common case.
"""
from __future__ import annotations

# Categorical annotation colours. Distinguishable in the common forms of colour
# blindness: no red/green pair carries meaning on its own, and the sequence
# alternates warm and cool so neighbouring labels never rely on hue alone.
ANNOTATION_PALETTE: tuple[str, ...] = (
    "#E8833A",   # amber      — first, because it reads on light and dark grey alike
    "#4C8DD9",   # blue
    "#C8577C",   # rose
    "#2FA39B",   # teal
    "#A579D6",   # violet
    "#8FAE3C",   # olive
    "#D2603C",   # rust
    "#6E8CA0",   # slate
)

# Meaning, not identity. These never come from the categorical palette, so a
# label can never accidentally look like a state.
PENDING   = "#E8B33A"   # proposed by a model, not yet accepted
ACCEPTED  = "#4CAF6A"   # committed to the store
SELECTED  = "#FFFFFF"   # current selection halo
EXCLUDED  = "#8A8A8A"   # inside an exclude zone

# Chrome that has to sit on the image itself.
SCALEBAR  = "#FFFFFF"
OVERLAY_BG = "#12181A"   # backing for text drawn over an image


def color_for_label(label: str) -> str:
    """
    Stable colour for a label name.

    The same label always gets the same colour within a session and across
    restarts, so a particle class keeps its colour between images.
    """
    key = (label or "").strip().lower()
    if not key:
        return ANNOTATION_PALETTE[0]
    # A fixed hash — Python's str hash is salted per process and would give the
    # same label a different colour on every launch.
    h = 0
    for ch in key:
        h = (h * 131 + ord(ch)) & 0xFFFFFFFF
    return ANNOTATION_PALETTE[h % len(ANNOTATION_PALETTE)]


def cluster_color(index: int) -> str:
    """Colour for the nth cluster / series in an overlay."""
    return ANNOTATION_PALETTE[index % len(ANNOTATION_PALETTE)]
