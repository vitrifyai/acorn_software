# ACORN — Microscopy Image Analysis Suite

**Annotate, Curate, Observe, Review, Navigate**

ACORN is an open-source desktop application for loading, annotating, and exporting cryo-EM and
other electron microscopy images.  It provides a PyQt6 GUI and a headless CLI, and integrates
SAM 3, micro-SAM, YOLO, and UNet for AI-assisted segmentation and detection.

Developed by **Alexis Williams** and **Chanda Harris** of the
[eMMA (electron Microscopy and Microanalysis) group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis),
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.

> For a plain-language introduction aimed at domain experts, see [QUICKSTART.md](QUICKSTART.md).

---

## Features

- Multi-format image loading: DM4, TIFF, MRC/MRCS, PNG, JPEG
- Interactive contrast adjustment with built-in presets (bandpass, CLAHE, percentile, sigma)
- Manual annotation: ROI polygons, rectangles, arrows, circles, scale bars, measurements
- AI-assisted segmentation:
  - **SAM 3 / SAM 2** (Meta) — point, box, scribble, and automatic mask generation
  - **micro-SAM** — SAM checkpoints fine-tuned for electron and light microscopy
  - **YOLO 11 / v8 / v9 / v10** (Ultralytics) — object detection and instance segmentation
  - **UNet / UNet++ / FPN / DeepLabV3+** (segmentation-models-pytorch) — semantic segmentation
- Export: annotated images, COCO JSON + RLE masks, training tiles, HuggingFace Hub push
- RELION .star file import (particle picks → circular ROIs)
- Training pipeline: tile extraction, augmentation, dataset splitting, YOLO and UNet training
- CLI: `acorn view`, `acorn export`, `acorn batch`, `acorn train-export`, `acorn finalize`

---

## Installation

### Personal machine (Linux / macOS)

```bash
bash install.sh
```

Uses [uv](https://github.com/astral-sh/uv) (installed automatically if not present).
Creates a self-contained `.venv`, installs all dependencies, downloads the recommended
model checkpoints (~380 MB), and creates a desktop shortcut.

### Shared workstation — all users (Linux, requires sudo)

```bash
sudo bash setup_system.sh
```

Installs ACORN to `/opt/acorn` with a shared model cache at `/opt/acorn/models/`.
Writes `/etc/profile.d/acorn.sh` (shared env vars), creates `/usr/local/bin/acorn` and
`/usr/local/bin/acorn-gui` wrappers, and registers a desktop entry for ThinLinc / GNOME / KDE.
No per-user install needed — any user on the machine can run `acorn-gui` immediately.

After the initial system setup, push updates from the dev copy without sudo:

```bash
bash deploy.sh
```

`deploy.sh` syncs source files and only reinstalls Python dependencies if `pyproject.toml`
has changed (sha256 hash check), keeping deploys fast.

### Migrating to a new machine

Model checkpoints are large.  To avoid re-downloading, copy the models directory:

```bash
# On the old machine — copy shared models to the new one
rsync -av /opt/acorn/models/ newmachine:/opt/acorn/models/
```

Run `sudo bash setup_system.sh` on the new machine first to create the directory structure,
then rsync the models over.  The setup script skips downloads for files that already exist.

---

## Python environment

| Deployment | Python | Venv location |
|------------|--------|---------------|
| Personal install | 3.10+ (uv-managed) | `<repo>/.venv` |
| Shared /opt | `/opt/conda/bin/python3` (3.13) | `/opt/acorn/.venv` |
| Dev (vnw) | 3.10 (uv-managed) | `/home/vnw/cryoem-tools/.venv` |

The editable dev install means changes to `.py` files are live immediately — no reinstall needed.

---

## AI models

### SAM 3 and micro-SAM

SAM 3 checkpoints are hosted on [HuggingFace Hub](https://huggingface.co/facebook/sam3) and
require accepting Meta's license agreement before the first download.  Log in at
[huggingface.co](https://huggingface.co) and accept the license, then authenticate:

```bash
huggingface-cli login
# or set the environment variable:
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

On a shared workstation, the admin can pre-download the checkpoint once to
`/opt/acorn/models/sam3/` so all users share it without needing individual HF accounts.

micro-SAM checkpoints (EM-organelle and light-microscopy fine-tunes) are downloaded
automatically on first use and cached to `$MICROSAM_CACHEDIR` (defaults to
`~/.cache/micro_sam`, or `/opt/acorn/models/micro_sam` on the shared install).
No login required.

### YOLO

YOLO model weights are fetched automatically by Ultralytics on first use and cached to
`$ACORN_MODELS_DIR/yolo` (shared install) or `~/.acorn/models/yolo` (personal install).
No login required.

### Pre-downloading models

```bash
python download_models.py              # interactive menu
python download_models.py --preset recommended   # ~381 MB, best for cryo-EM
python download_models.py --preset em            # all EM-tuned SAM + all YOLO-seg sizes
python download_models.py --preset all           # everything
python download_models.py --list                 # show status without downloading
```

| Model | Backend | Size | Notes |
|-------|---------|------|-------|
| vit_b_em_organelles | micro-SAM | 375 MB | EM organelles — recommended for cryo-EM |
| vit_l_em_organelles | micro-SAM | 760 MB | EM organelles, larger / more accurate |
| vit_b_lm | micro-SAM | 375 MB | Light microscopy |
| vit_l_lm | micro-SAM | 760 MB | Light microscopy, larger |
| vit_b / vit_l / vit_h | micro-SAM | 375 MB – 2.4 GB | Generic SAM (Meta original) |
| sam3.pt | SAM 3 | ~2.4 GB | SAM 3 (HF login required, see above) |
| yolo11n-seg.pt | YOLO | 6 MB | Nano segmentation — recommended starter |
| yolo11s/m/l/x-seg.pt | YOLO | 22–130 MB | Larger segmentation models |
| yolo11n/s/m/l/x.pt | YOLO | 6–130 MB | Detection-only variants |

---

## Source layout

```
src/acorn/
  core/       dm4_loader.py, contrast.py, annotations.py, measurements.py,
              sam_predictor.py, usam_predictor.py, yolo_predictor.py, unet_predictor.py,
              star_loader.py, quality.py, embedding_cache.py
  gui/        main_window.py, canvas_widget.py, contrast_panel.py, annotation_panel.py,
              sam_panel.py, yolo_panel.py, unet_panel.py, train_panel.py,
              export_panel.py, measurement_panel.py, analysis_panel.py
  export/     training_exporter.py, dataset_finalizer.py, mask_exporter.py,
              image_exporter.py, hub_exporter.py, batch.py
  render/     canvas.py, annotation_renderer.py, scalebar.py
  analysis/   surface_area.py, surface_area_stats.py
  cli/        main.py
```

---

## Optional dependencies

```toml
pip install "acorn[gui]"     # PyQt6, matplotlib (required for GUI)
pip install "acorn[mrc]"     # mrcfile (MRC/MRCS support)
pip install "acorn[sam]"     # sam3 + micro-sam
pip install "acorn[yolo]"    # ultralytics
pip install "acorn[unet]"    # segmentation-models-pytorch
pip install "acorn[hub]"     # huggingface datasets (push to Hub)
pip install "acorn[full]"    # all of the above
```

---

## CLI reference

```bash
acorn view image.dm4                              # open in GUI
acorn export image.dm4 -o out.png --dpi 300       # export image
acorn batch --input-dir ./images --output-dir ./out
acorn train-export --dataset-dir ./dataset        # extract training tiles
acorn finalize --dataset-dir ./dataset            # create train/val/test splits
acorn push-to-hub --dataset-dir ./dataset --repo-id user/name
```

---

## Supported file formats

| Format | Extension |
|--------|-----------|
| Gatan DM4 | .dm4 |
| TIFF | .tif, .tiff |
| MRC / MRCS | .mrc, .mrcs |
| PNG | .png |
| JPEG | .jpg, .jpeg |

---

## Contributing and feedback

Bug reports, feature requests, and pull requests are welcome via GitHub:
**[GitHub repository — link to be added]**

For direct correspondence: **williamsan@ornl.gov**

---

## About eMMA

The [electron Microscopy and Microanalysis (eMMA) group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis)
is part of the Center for Nanophase Materials Sciences (CNMS) at Oak Ridge National Laboratory.
The group develops and applies advanced multi-scale materials characterization techniques —
including aberration-corrected STEM, in-situ and cryo-EM, electron spectroscopy (EELS/EDX),
and atom probe tomography — to uncover structure-property-function relationships across
metals, ceramics, composites, nanomaterials, and biological specimens.

ACORN was developed to accelerate data analysis workflows for the group's cryo-EM and
analytical STEM programs, with a focus on ease of use for domain experts and compatibility
with standard ML training pipelines.

---

## License

See [LICENSE](LICENSE) for details.
