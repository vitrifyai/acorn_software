"""
Download AI model checkpoints for ACORN.

Run this any time to add more models.  Already-downloaded models are skipped.

Usage:
    python download_models.py              # interactive menu
    python download_models.py --preset recommended
    python download_models.py --preset em
    python download_models.py --preset all
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

# ── download locations ────────────────────────────────────────────────────────
# Shared /opt/acorn/models/ is used automatically when it exists.
# Individual env-var overrides take precedence if set.

import os as _os

_OPT_MODELS = Path("/opt/acorn/models")
_OPT_EXISTS = _OPT_MODELS.is_dir()

USAM_CACHE = Path(_os.environ.get(
    "MICROSAM_CACHEDIR",
    _OPT_MODELS / "micro_sam" if _OPT_EXISTS else Path.home() / ".cache" / "micro_sam"
))
YOLO_CACHE = Path(_os.environ.get(
    "ACORN_MODELS_DIR",
    str(_OPT_MODELS) if _OPT_EXISTS else str(Path.home() / ".acorn" / "models")
)) / "yolo"

# ── model catalogue ───────────────────────────────────────────────────────────

# (url, local_path, approximate_size_mb, description)
USAM_MODELS: dict[str, tuple[str, Path, int, str]] = {
    "vit_b_em_organelles": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt",
        USAM_CACHE / "vit_b_em_organelles" / "vit_b.pt",
        375,
        "SAM  vit_b  fine-tuned for electron microscopy organelles  [RECOMMENDED for cryo-EM]",
    ),
    "vit_l_em_organelles": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/humorous-crab/1.2/files/vit_l.pt",
        USAM_CACHE / "vit_l_em_organelles" / "vit_l.pt",
        760,
        "SAM  vit_l  fine-tuned for electron microscopy organelles  (larger, slower, more accurate)",
    ),
    "vit_b_lm": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
        USAM_CACHE / "vit_b_lm" / "vit_b.pt",
        375,
        "SAM  vit_b  fine-tuned for light microscopy",
    ),
    "vit_l_lm": (
        "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l.pt",
        USAM_CACHE / "vit_l_lm" / "vit_l.pt",
        760,
        "SAM  vit_l  fine-tuned for light microscopy  (larger)",
    ),
    "vit_b": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        USAM_CACHE / "vit_b" / "sam_vit_b_01ec64.pth",
        375,
        "SAM  vit_b  generic (Meta original, no microscopy fine-tuning)",
    ),
    "vit_l": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        USAM_CACHE / "vit_l" / "sam_vit_l_0b3195.pth",
        760,
        "SAM  vit_l  generic large",
    ),
    "vit_h": (
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        USAM_CACHE / "vit_h" / "sam_vit_h_4b8939.pth",
        2400,
        "SAM  vit_h  generic huge  (best accuracy, needs a lot of GPU memory)",
    ),
}

_GH = "https://github.com/ultralytics/assets/releases/download/v8.4.0"

YOLO_MODELS: dict[str, tuple[str, Path, int, str]] = {
    # Detection
    "yolo11n.pt":     (f"{_GH}/yolo11n.pt",     YOLO_CACHE / "yolo11n.pt",      6,   "YOLO11 nano    — detection only            (fastest, least accurate)"),
    "yolo11s.pt":     (f"{_GH}/yolo11s.pt",     YOLO_CACHE / "yolo11s.pt",      22,  "YOLO11 small   — detection only"),
    "yolo11m.pt":     (f"{_GH}/yolo11m.pt",     YOLO_CACHE / "yolo11m.pt",      64,  "YOLO11 medium  — detection only"),
    "yolo11l.pt":     (f"{_GH}/yolo11l.pt",     YOLO_CACHE / "yolo11l.pt",      100, "YOLO11 large   — detection only"),
    "yolo11x.pt":     (f"{_GH}/yolo11x.pt",     YOLO_CACHE / "yolo11x.pt",      130, "YOLO11 xlarge  — detection only            (slowest, most accurate)"),
    # Segmentation
    "yolo11n-seg.pt": (f"{_GH}/yolo11n-seg.pt", YOLO_CACHE / "yolo11n-seg.pt",  6,   "YOLO11 nano    — detection + segmentation  [RECOMMENDED starter]"),
    "yolo11s-seg.pt": (f"{_GH}/yolo11s-seg.pt", YOLO_CACHE / "yolo11s-seg.pt",  22,  "YOLO11 small   — detection + segmentation"),
    "yolo11m-seg.pt": (f"{_GH}/yolo11m-seg.pt", YOLO_CACHE / "yolo11m-seg.pt",  64,  "YOLO11 medium  — detection + segmentation"),
    "yolo11l-seg.pt": (f"{_GH}/yolo11l-seg.pt", YOLO_CACHE / "yolo11l-seg.pt",  100, "YOLO11 large   — detection + segmentation"),
    "yolo11x-seg.pt": (f"{_GH}/yolo11x-seg.pt", YOLO_CACHE / "yolo11x-seg.pt",  130, "YOLO11 xlarge  — detection + segmentation  (slowest, most accurate)"),
}

# ── presets ───────────────────────────────────────────────────────────────────

PRESETS: dict[str, dict] = {
    "recommended": {
        "label": "Recommended  —  best for cryo-EM, minimal download",
        "usam":  ["vit_b_em_organelles"],
        "yolo":  ["yolo11n-seg.pt"],
    },
    "em": {
        "label": "Full EM set  —  all EM-tuned SAM variants + all YOLO segmentation sizes",
        "usam":  ["vit_b_em_organelles", "vit_l_em_organelles"],
        "yolo":  ["yolo11n-seg.pt", "yolo11s-seg.pt", "yolo11m-seg.pt",
                  "yolo11l-seg.pt", "yolo11x-seg.pt"],
    },
    "lm": {
        "label": "Light microscopy  —  LM-tuned SAM vit_b + vit_l + YOLO nano seg",
        "usam":  ["vit_b_lm", "vit_l_lm"],
        "yolo":  ["yolo11n-seg.pt"],
    },
    "all": {
        "label": "Everything  —  every SAM variant + every YOLO size",
        "usam":  list(USAM_MODELS),
        "yolo":  list(YOLO_MODELS),
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

BOLD  = "\033[1m"
GREEN = "\033[1;32m"
CYAN  = "\033[1;36m"
DIM   = "\033[2m"
RESET = "\033[0m"


def _status(path: Path) -> str:
    if path.exists():
        mb = path.stat().st_size / 1_048_576
        return f"{GREEN}downloaded ({mb:.0f} MB){RESET}"
    return f"{DIM}not downloaded{RESET}"


def _progress_bar(label: str, width: int = 36):
    def hook(count, block_size, total_size):
        if total_size <= 0:
            sys.stdout.write(f"\r  Downloading {label}…")
            sys.stdout.flush()
            return
        done  = min(count * block_size, total_size)
        pct   = done / total_size
        filled = int(width * pct)
        bar    = "#" * filled + "-" * (width - filled)
        sys.stdout.write(
            f"\r  [{bar}]  {done/1_048_576:5.0f}/{total_size/1_048_576:.0f} MB  {label}"
        )
        sys.stdout.flush()
        if done >= total_size:
            print()
    return hook


def _download_file(url: str, dest: Path, name: str) -> bool:
    if dest.exists():
        print(f"  {name}  already present — skipping.")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    try:
        urllib.request.urlretrieve(url, tmp, reporthook=_progress_bar(name))
        tmp.rename(dest)
        return True
    except Exception as exc:
        if tmp.exists():
            tmp.unlink()
        print(f"\n  {BOLD}FAILED{RESET} {name}: {exc}")
        return False


def _total_mb(usam_keys: list, yolo_keys: list) -> int:
    total = 0
    for k in usam_keys:
        if not USAM_MODELS[k][1].exists():
            total += USAM_MODELS[k][2]
    for k in yolo_keys:
        if not YOLO_MODELS[k][1].exists():
            total += YOLO_MODELS[k][2]
    return total


def _run_downloads(usam_keys: list, yolo_keys: list) -> int:
    failed = 0
    if usam_keys:
        print(f"\n{BOLD}SAM checkpoints{RESET}  →  {USAM_CACHE}")
        for k in usam_keys:
            url, dest, _, _ = USAM_MODELS[k]
            if not _download_file(url, dest, k):
                failed += 1
    if yolo_keys:
        print(f"\n{BOLD}YOLO checkpoints{RESET}  →  {YOLO_CACHE}")
        for k in yolo_keys:
            url, dest, _, _ = YOLO_MODELS[k]
            if not _download_file(url, dest, k):
                failed += 1
    return failed


# ── interactive menu ──────────────────────────────────────────────────────────

def _show_catalogue() -> None:
    print(f"\n{BOLD}Available SAM checkpoints{RESET}  (stored in {USAM_CACHE})")
    for name, (_, path, mb, desc) in USAM_MODELS.items():
        print(f"  {CYAN}{name:<26}{RESET}  {mb:>5} MB  {desc}")
        print(f"  {' '*26}  status: {_status(path)}")

    print(f"\n{BOLD}Available YOLO checkpoints{RESET}  (stored in {YOLO_CACHE})")
    for name, (_, path, mb, desc) in YOLO_MODELS.items():
        print(f"  {CYAN}{name:<22}{RESET}  {mb:>5} MB  {desc}")
        print(f"  {' '*22}  status: {_status(path)}")


def _interactive_menu() -> tuple[list, list]:
    _show_catalogue()

    print(f"\n{BOLD}Choose what to download:{RESET}\n")
    for i, (key, preset) in enumerate(PRESETS.items(), 1):
        mb = _total_mb(preset["usam"], preset["yolo"])
        already = mb == 0
        note = "(already downloaded)" if already else f"~{mb} MB to download"
        print(f"  [{i}] {preset['label']}")
        print(f"      {DIM}{note}{RESET}")
    print(f"  [4] Custom — pick individual models")
    print(f"  [5] Show status and exit")
    print()

    choice = _prompt("Enter choice [1]: ", default="1")

    if choice == "1":
        p = PRESETS["recommended"]
        return p["usam"], p["yolo"]
    if choice == "2":
        p = PRESETS["em"]
        return p["usam"], p["yolo"]
    if choice == "3":
        p = PRESETS["all"]
        return p["usam"], p["yolo"]
    if choice == "4":
        return _custom_picker()
    if choice == "5":
        _show_catalogue()
        sys.exit(0)
    print("Invalid choice — defaulting to Recommended.")
    p = PRESETS["recommended"]
    return p["usam"], p["yolo"]


def _custom_picker() -> tuple[list, list]:
    usam_sel, yolo_sel = [], []

    print(f"\n{BOLD}SAM models{RESET}  (enter numbers separated by spaces, or press Enter to skip):")
    keys = list(USAM_MODELS)
    for i, (k, (_, path, mb, desc)) in enumerate(USAM_MODELS.items(), 1):
        status = "downloaded" if path.exists() else f"~{mb} MB"
        print(f"  [{i}] {k:<26}  {status}  —  {desc}")
    raw = _prompt("SAM selection: ", default="")
    for tok in raw.split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(keys):
                usam_sel.append(keys[idx])
        except ValueError:
            pass

    print(f"\n{BOLD}YOLO models{RESET}  (enter numbers separated by spaces, or press Enter to skip):")
    keys = list(YOLO_MODELS)
    for i, (k, (_, path, mb, desc)) in enumerate(YOLO_MODELS.items(), 1):
        status = "downloaded" if path.exists() else f"~{mb} MB"
        print(f"  [{i}] {k:<22}  {status}  —  {desc}")
    raw = _prompt("YOLO selection: ", default="")
    for tok in raw.split():
        try:
            idx = int(tok) - 1
            if 0 <= idx < len(keys):
                yolo_sel.append(keys[idx])
        except ValueError:
            pass

    return usam_sel, yolo_sel


def _prompt(msg: str, default: str = "") -> str:
    """Prompt for input; return default if stdin is not a terminal or empty."""
    if not sys.stdin.isatty():
        return default
    try:
        val = input(msg).strip()
        return val if val else default
    except (EOFError, KeyboardInterrupt):
        print()
        return default


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--preset",
        choices=["recommended", "em", "all"],
        help="Download a predefined set of models without showing the menu.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show all available models and their download status, then exit.",
    )
    args = parser.parse_args()

    if args.list:
        _show_catalogue()
        sys.exit(0)

    if args.preset:
        p = PRESETS[args.preset]
        usam_keys = p["usam"]
        yolo_keys = p["yolo"]
    else:
        usam_keys, yolo_keys = _interactive_menu()

    if not usam_keys and not yolo_keys:
        print("Nothing selected.")
        sys.exit(0)

    mb = _total_mb(usam_keys, yolo_keys)
    if mb == 0:
        print("\nAll selected models are already downloaded.")
        sys.exit(0)

    print(f"\nDownloading ~{mb} MB…")
    failed = _run_downloads(usam_keys, yolo_keys)

    print()
    if failed:
        print(f"WARNING: {failed} file(s) failed.  Check your connection and re-run.")
        sys.exit(1)
    else:
        print(f"{GREEN}All downloads complete.{RESET}")


if __name__ == "__main__":
    main()
