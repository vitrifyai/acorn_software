"""Export constants — single source of truth for output directory names."""
from pathlib import Path

ACORN_MEASUREMENTS_DIR = "acorn_measurements"  # folder created next to image files
MEASUREMENTS_CSV       = "measurements.csv"


def measurements_dir(image_dir: Path) -> Path:
    return image_dir / ACORN_MEASUREMENTS_DIR


def measurements_csv(image_dir: Path) -> Path:
    return measurements_dir(image_dir) / MEASUREMENTS_CSV
