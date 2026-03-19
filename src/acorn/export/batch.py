"""Batch image converter with parallel loading (DM4, TIFF, MRC, PNG, JPEG)."""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from acorn.core.dm4_loader import DM4Image, scan_folder
from acorn.core.contrast import ContrastParams
from acorn.export.image_exporter import export_image


@dataclass
class BatchResult:
    succeeded: list[Path] = field(default_factory=list)
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return len(self.succeeded) + len(self.failed)

    def summary(self) -> str:
        lines = [f"Batch complete: {len(self.succeeded)}/{self.n_total} succeeded"]
        for path, err in self.failed:
            lines.append(f"  FAILED {path.name}: {err}")
        return "\n".join(lines)


def batch_export(
    input_dir: str | Path,
    output_dir: str | Path,
    params: Optional[ContrastParams] = None,
    fmt: str = "tiff",
    dpi: int = 300,
    add_scalebar: bool = True,
    scalebar_color: str = "#FFFFFF",
    workers: int = 4,
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> BatchResult:
    """
    Convert all DM4 files in *input_dir* to images in *output_dir*.

    Parameters
    ----------
    input_dir     : folder containing *.dm4 files
    output_dir    : destination folder (created if absent)
    params        : contrast parameters (bandpass default)
    fmt           : output format: "png", "tiff", "jpeg", "svg", "eps", "pdf"
    dpi           : resolution (irrelevant for lossless TIFF of raw data)
    add_scalebar  : add calibrated scale bar to each image
    scalebar_color: scale bar colour
    workers       : number of parallel threads
    progress_cb   : optional callback(n_done, n_total, filename)

    Returns
    -------
    BatchResult with succeeded / failed lists.
    """
    import matplotlib
    matplotlib.use("Agg")

    if params is None:
        params = ContrastParams()

    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext_map = {"jpeg": "jpg", "tiff": "tif"}
    ext = ext_map.get(fmt, fmt)

    files = scan_folder(in_dir)   # scans all supported formats
    result = BatchResult()
    n_total = len(files)

    def _process(src: Path) -> tuple[Path, Optional[str]]:
        try:
            img = DM4Image.from_file(src)
            dst = out_dir / f"{src.stem}.{ext}"
            export_image(
                img, dst,
                params=params, fmt=fmt, dpi=dpi,
                add_scalebar=add_scalebar,
                scalebar_color=scalebar_color,
            )
            return dst, None
        except Exception as exc:
            return src, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, f): f for f in files}
        for n_done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            path, err = future.result()
            if err is None:
                result.succeeded.append(path)
            else:
                result.failed.append((futures[future], err))
            if progress_cb is not None:
                label = futures[future].name
                progress_cb(n_done, n_total, label)

    return result
