"""
Locating SAM 3's text-prompt vocabulary.

SAM 3 reads a CLIP byte-pair-encoding table to turn a phrase like "vesicles" into
tokens. `sam3/model_builder.py` looks for it at a path relative to its own file:

    <site-packages>/assets/bpe_simple_vocab_16e6.txt.gz

The published sam3 wheel does not contain that `assets/` directory — it is in the
source repository but was left out of the package, and the file is not in the
model checkpoint either. So a fresh install fails the first time anyone types a
text prompt, with a bare FileNotFoundError naming a path inside site-packages
that means nothing to the person reading it.

This module looks in the places a copy might reasonably live, puts one where sam3
expects it, and otherwise explains what is missing and what still works without
it. Box, point and scribble prompts need none of this.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

VOCAB_NAME = "bpe_simple_vocab_16e6.txt.gz"

# Where the file can be obtained, quoted verbatim in the error message so the
# person reading it does not have to go looking.
SOURCE_URL = (
    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"
)

_MISSING_MESSAGE = (
    "SAM 3 cannot load: its CLIP vocabulary file is missing.\n\n"
    "This is a packaging bug in the sam3 release, not a problem with your data or "
    "your checkpoint. The wheel does not ship the assets/ directory it reads, and "
    "sam3 builds its text encoder whether or not you use a typed prompt — so the "
    "model will not load at all until the file is in place.\n\n"
    "Right now: switch the SAM backend to SAM 2, which is unaffected.\n\n"
    "To fix it, put {name} here:\n    {expected}\n\n"
    "It is the standard CLIP vocabulary, about 1.3 MB:\n    {url}\n\n"
    "ACORN also picks it up automatically from any of:\n"
    "    $ACORN_SAM3_VOCAB\n"
    "    $ACORN_MODELS_DIR/sam3_assets/\n"
    "    /opt/models/acorn/models/sam3_assets/\n"
    "    ~/.cache/acorn/sam3_assets/"
)


def expected_path() -> Path | None:
    """Where sam3 will look, or None if sam3 is not installed."""
    try:
        from sam3 import model_builder
    except Exception:
        return None
    return (Path(model_builder.__file__).parent.parent / "assets" / VOCAB_NAME).resolve()


def _candidates() -> list[Path]:
    """Places a copy may already be sitting, most authoritative first."""
    out: list[Path] = []

    def add(value) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if path.is_dir():
            path = path / VOCAB_NAME
        if path.name == VOCAB_NAME and path.is_file():
            out.append(path)

    add(os.environ.get("ACORN_SAM3_VOCAB"))
    models_dir = os.environ.get("ACORN_MODELS_DIR")
    if models_dir:
        add(Path(models_dir) / "sam3_assets")
    for shared in ("/opt/models/acorn/models/sam3_assets", "/opt/acorn/models/sam3_assets"):
        add(shared)
    add(Path.home() / ".cache" / "acorn" / "sam3_assets")
    return out


def ensure_vocab() -> Path | None:
    """
    Make sure sam3 can find the vocabulary, copying a known good copy if needed.

    Returns the path sam3 will read, or None when no copy could be found. Never
    raises: a missing vocabulary must not stop SAM 3 loading for box and point
    prompts, which is the majority of what people do with it.
    """
    expected = expected_path()
    if expected is None:
        return None
    if expected.is_file():
        return expected
    for candidate in _candidates():
        try:
            expected.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, expected)
            return expected
        except OSError:
            continue      # read-only venv, or a race with another process
    return None


def is_available() -> bool:
    """True when SAM 3 has everything it needs to build."""
    return ensure_vocab() is not None


def looks_like_missing_vocab(exc: BaseException) -> bool:
    """True when *exc* is sam3 failing for want of this file, not something else."""
    return isinstance(exc, (FileNotFoundError, OSError)) and VOCAB_NAME in str(exc)


def missing_message() -> str:
    """What to tell the user, naming the exact path and where to get the file."""
    expected = expected_path()
    return _MISSING_MESSAGE.format(
        name=VOCAB_NAME,
        expected=expected or "<sam3 is not installed>",
        url=SOURCE_URL,
    )
