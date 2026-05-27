# ACORN — Microscopy Image Analysis Suite

**Annotate, Curate, Observe, Review, Navigate**

ACORN is an open-source desktop application for loading, annotating, analyzing, and exporting
cryo-EM and other electron microscopy images. It provides a PyQt6 GUI and a headless CLI,
integrates SAM 3, SAM 2, YOLO, and UNet for AI-assisted segmentation, ships a plugin architecture for
extending functionality, and includes **CLU** — a natural-language AI assistant that can drive
any feature in the application from a chat panel.

Developed by **Alexis Williams** and **Chanda Harris** of the
[eMMA (electron Microscopy and Microanalysis) group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis),
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.

> Expert user or hate instructions? See [QUICKSTART.md](QUICKSTART.md).

---

## Features

### Core image viewing and annotation
- Multi-format loading: DM4, TIFF (single and multi-frame), MRC/MRCS, PNG, JPEG
- Movie / multi-frame file support — automatic frame detection for DM4, TIFF, and MRC/MRCS stacks
- Interactive contrast: bandpass, percentile, sigma, CLAHE, manual, with saveable presets
- Manual annotation: ROI polygons, rectangles, arrows, circles, text, scale bars, distance and angle measurements
- RELION .star file import — particle picks loaded as circular ROI annotations

### AI-assisted segmentation
- **SAM 3 / SAM 2** (Meta) — point, box, scribble, automatic mask generation
- **micro-SAM** — SAM checkpoints fine-tuned for electron and light microscopy
- **YOLO 11 / v8 / v9 / v10** (Ultralytics) — object detection and instance segmentation
- **UNet / UNet++ / FPN / DeepLabV3+ / MAnet / PAN** (segmentation-models-pytorch) — semantic segmentation
- Pipe YOLO boxes directly into SAM for instance-accurate masks at detection speed

### Movie processing (multi-frame)
- **Frame averaging** — mean, motion-corrected, or dose-weighted (Grant & Grigorieff 2015)
- **Motion correction** — two-pass phase cross-correlation alignment (skimage); sub-pixel accurate
- **Drift trajectory plot** — MotionCor2-style per-frame displacement chart and drift path
- **Dose series analysis** — split movie into equal-dose bins, view averaged images and
  difference maps (`bin N − bin 1`) to visualise beam-induced structural changes
- Configurable frame range (start / end frame) — skip early high-motion frames or cap total dose
- Individual frame viewer — step through any frame in the movie

### Analysis
- **Surface area estimation** — 3D surface area from 2D ROI masks using four auto-selected
  methods (ellipsoid, Cauchy, Fourier, fractal for rough surfaces); uncertainty propagation,
  hollow/aggregate detection, multi-GPU batch
- **Particle tracking** — link annotations across image series using nearest-neighbour matching;
  configurable max displacement (nm), minimum track length, frame gap tolerance

### Training pipeline
- Tile extraction from annotated images, 8× augmentation, negative-prompt sampling
- Dataset splitting (train / val / test) with reproducible seeds
- In-app **YOLO training** — configure base model, epochs, batch size, image size, then start
- In-app **UNet training** — configure architecture (UNet, UNet++, FPN, DeepLabV3+…),
  encoder backbone (ResNet, EfficientNet, MiT…), and hyperparameters
- Export to COCO JSON + RLE masks
- One-click push to HuggingFace Hub

### Quality and export
- Image quality assessment (blur, contrast, saturation, low-frequency artifacts)
- Export annotated display image (PNG) for external annotation tools or publication
- Batch export over entire image folders

---

## Plugin architecture

ACORN uses a plugin system based on Python entry points (`acorn.plugins`). Plugins are
discovered automatically at startup — no configuration file needed. Each plugin provides a
tab panel in the main window and can respond to AI tool calls through the `action_requested`
signal on `AcornContext`.

### Built-in plugins

| Plugin package | Tab label | What it adds |
|----------------|-----------|--------------|
| `acorn_analysis` | **Analysis** | 3D surface area estimation from ROI masks; single or batch mode; multi-GPU support |
| `acorn_tracking` | **Track** | Particle / feature tracking across image sequences; configurable displacement and gap tolerance |
| `acorn_3d` | **3D** | Volume rendering and z-slice navigation for MRC tomograms |
| `acorn_llm` | **CLU** | Natural-language AI assistant chat panel (see below) |

### Writing a plugin

Create a package that subclasses `acorn.plugin_loader.AcornPlugin`:

```python
from acorn.plugin_loader import AcornPlugin

class MyPlugin(AcornPlugin):
    PLUGIN_ID = "my_plugin"
    TAB_LABEL = "My Tab"

    def create_panel(self):
        # return a QWidget to add as a tab, or None
        ...

    def teardown(self):
        # called on application exit
        ...
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."acorn.plugins"]
my_plugin = "my_plugin.plugin:MyPlugin"
```

Plugins receive an `AcornContext` object giving read/write access to the current image,
annotations, pixel size, contrast, and the `action_requested` signal so CLU can drive
plugin functionality.

---

## AI assistant — CLU

CLU is ACORN's built-in natural-language assistant. It lives in the **CLU** tab (added by the
`acorn_llm` plugin) and can perform any ACORN action from a chat message — segmentation,
contrast adjustment, training, dataset management, movie processing, and more.

### Supported providers

| Provider | How to connect |
|----------|---------------|
| Anthropic (Claude) | Set API key in the CLU tab settings, or `ANTHROPIC_API_KEY` env var |
| OpenAI / compatible | Set API key + Base URL; works with GPT-4o, Groq, and local Ollama models |
| Ollama (local) | Set provider to OpenAI-compatible, Base URL = `http://localhost:11434/v1`, model = `acorn-tools` |

### Two model modes

- **Vision model** — receives a thumbnail of the current image along with the message; good for
  questions about image content, contrast assessment, and open-ended analysis
- **Tool model** — uses function calling to drive ACORN actions; good for workflows ("segment
  and queue for training"), comparisons ("try different frame ranges"), and multi-step pipelines
- **Auto** (default) — uses the vision model when "Include image" is checked, the tool model
  otherwise

### What CLU can do

CLU has full knowledge of the current application state (loaded image, pixel size, annotations,
model configuration, export queue, training settings) and can:

- Segment structures: *"find the vesicles"*, *"detect all particles"*
- Fix contrast: *"I can't see anything in these movies"*, *"the contrast is off"*
- Drive the full training pipeline: *"prep this for training"*, *"train a YOLO model"*
- Process movies: *"average the frames"*, *"run motion correction on frames 3 to 50"*,
  *"show me how the membrane changes with dose"*
- Run analysis: *"measure surface areas of all vesicles"*, *"track particles across images"*
- Answer questions without tools: *"how many vesicles are annotated?"*, *"what's the pixel size?"*

### Ollama / local model setup

To run CLU entirely offline with a local model:

```bash
# Build the acorn-tools Ollama model (optimised system prompt + tool calling)
ollama create acorn-tools -f Modelfile.tools

# In the CLU tab settings:
#   Provider: OpenAI / compatible
#   Model: acorn-tools
#   Base URL: http://localhost:11434/v1
```

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

```bash
# On the old machine — copy shared models to the new one
rsync -av /opt/acorn/models/ newmachine:/opt/acorn/models/
```

Run `sudo bash setup_system.sh` on the new machine first to create the directory structure,
then rsync the models over. The setup script skips downloads for files that already exist.

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

### SAM 3, SAM 2, and micro-SAM

ACORN supports three SAM backends, selected automatically at load time:

| Backend | Package | Notes |
|---------|---------|-------|
| **SAM 3** | `pip install sam3` | Recommended — fastest, best cryo-EM accuracy |
| **SAM 2** | `pip install sam2` | Fallback if SAM 3 is not installed; same HF checkpoints |
| **micro-SAM** | bundled | Fine-tuned for EM/LM; no HF login required |

SAM 3 and SAM 2 checkpoints are hosted on HuggingFace and require accepting Meta's license
agreement before the first download. Log in at [huggingface.co](https://huggingface.co),
accept the license for [facebook/sam3](https://huggingface.co/facebook/sam3) or
[facebook/sam2](https://huggingface.co/facebook/sam2), then authenticate:

```bash
huggingface-cli login
# or:
export HUGGING_FACE_HUB_TOKEN=your_token_here
```

On a shared workstation, the admin can pre-download the checkpoint once to
`/opt/acorn/models/sam3/` (or `sam2/`) so all users share it without needing individual
HF accounts.

micro-SAM checkpoints are downloaded automatically on first use and cached to
`$MICROSAM_CACHEDIR` (defaults to `~/.cache/micro_sam`, or `/opt/acorn/models/micro_sam`
on the shared install). No login required.

### YOLO

YOLO model weights are fetched automatically by Ultralytics on first use and cached to
`$ACORN_MODELS_DIR/yolo` (shared install) or `~/.acorn/models/yolo` (personal install).

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
| sam3.pt | SAM 3 | ~2.4 GB | SAM 3 (HF login required) |
| sam2_hiera_large.pt | SAM 2 | ~2.4 GB | SAM 2 fallback (HF login required) |
| sam2_hiera_base_plus.pt | SAM 2 | ~320 MB | SAM 2 smaller variant |
| yolo11n-seg.pt | YOLO | 6 MB | Nano segmentation — recommended starter |
| yolo11s/m/l/x-seg.pt | YOLO | 22–130 MB | Larger segmentation models |
| yolo11n/s/m/l/x.pt | YOLO | 6–130 MB | Detection-only variants |

---

## Source layout

```
src/
  acorn/
    core/       dm4_loader.py        multi-format image loader; movie frame detection
                frame_processor.py   mean average, motion correction, dose weighting, dose series
                contrast.py          contrast normalisation methods and presets
                annotations.py       annotation store and types
                measurements.py      distance / angle / area engine
                sam_predictor.py     SAM 3 / SAM 2 wrapper
                usam_predictor.py    micro-SAM wrapper
                yolo_predictor.py    YOLO detection / segmentation wrapper
                unet_predictor.py    UNet inference wrapper
                star_loader.py       RELION .star file parser
                quality.py           image quality assessment
                embedding_cache.py   SAM embedding cache
                _train_worker.py     background training thread
                yolo_trainer.py      YOLO training wrapper
                unet_trainer.py      UNet training wrapper
    gui/        main_window.py       main application window
                canvas_widget.py     matplotlib-backed image canvas
                contrast_panel.py    contrast tab
                annotation_panel.py  manual annotation tab
                measurement_panel.py measurement tab
                sam_panel.py         SAM tab
                yolo_panel.py        YOLO tab
                unet_panel.py        UNet tab
                train_panel.py       training tab
                export_panel.py      export tab
                context.py           AcornContext — plugin interface
    export/     training_exporter.py tile extraction and COCO export
                dataset_finalizer.py train/val/test splitting
                mask_exporter.py     mask PNG export
                image_exporter.py    display image export
                hub_exporter.py      HuggingFace Hub push
                batch.py             headless batch export
    render/     canvas.py            image rendering (matplotlib)
                annotation_renderer.py
                scalebar.py
    analysis/   surface_area.py      3D surface area estimation
                surface_area_stats.py batch statistics
    cli/        main.py              CLI entry point
  acorn_analysis/  Analysis plugin — surface area tab
  acorn_tracking/  Track plugin — particle tracking tab
  acorn_3d/        3D plugin — volume rendering tab
  acorn_llm/       CLU plugin — AI assistant tab
    agent.py        LLM agent (Anthropic / OpenAI-compatible)
    panel.py        chat UI panel
    config.py       provider and model configuration
    plugin.py       plugin entry point
```

---

## Optional dependencies

```bash
pip install "acorn[gui]"       # PyQt6, matplotlib (required for GUI)
pip install "acorn[mrc]"       # mrcfile (MRC/MRCS support)
pip install "acorn[sam]"       # sam3 + micro-sam
pip install "acorn[yolo]"      # ultralytics
pip install "acorn[unet]"      # segmentation-models-pytorch + torch
pip install "acorn[hub]"       # huggingface datasets (push to Hub)
pip install "acorn[llm]"       # anthropic + openai (CLU with cloud providers)
pip install "acorn[analysis]"  # opencv-python + pandas (surface area analysis)
pip install "acorn[tracking]"  # pandas + scipy (particle tracking)
pip install "acorn[volume]"    # mrcfile + tifffile (3D plugin)
pip install "acorn[full]"      # all of the above (no dev tools)
pip install "acorn[all]"       # everything including dev tools
```

---

## CLI reference

```bash
acorn view image.dm4                               # open in GUI
acorn export image.dm4 -o out.png --dpi 300        # export image
acorn batch --input-dir ./images --output-dir ./out
acorn train-export --dataset-dir ./dataset         # extract training tiles
acorn finalize --dataset-dir ./dataset             # create train/val/test splits
acorn push-to-hub --dataset-dir ./dataset --repo-id user/name
```

---

## Supported file formats

| Format | Extension | Movie / multi-frame |
|--------|-----------|---------------------|
| Gatan DM4 | .dm4 | Yes — 3D stacks auto-detected |
| TIFF | .tif, .tiff | Yes — multi-page stacks |
| MRC / MRCS | .mrc, .mrcs | Yes — all 3D MRC data |
| PNG | .png | No |
| JPEG | .jpg, .jpeg | No |

When a multi-frame file is opened, ACORN automatically mean-averages the frames for display
and shows the **movie bar** below the canvas with controls for frame selection, compression
method, motion correction, and dose series analysis.

---

## Contributing and feedback

Bug reports, feature requests, and pull requests are welcome via GitHub:
**[https://github.com/vitrifyai/acorn_software](https://github.com/vitrifyai/acorn_software)**

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
