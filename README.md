# ACORN: Microscopy Image Analysis Suite

**Annotate, Curate, Observe, Review, Navigate**

ACORN is an open-source desktop application for loading, annotating, analyzing, and exporting
cryo-EM and other electron microscopy images. It provides a PyQt6 GUI and a headless CLI,
integrates SAM 3, SAM 2, YOLO, and UNet for AI-assisted segmentation, ships a plugin architecture for
extending functionality, and includes **CLU** (Cryo-EM Lab Utility), a natural-language AI assistant that can drive
any feature in the application from a chat panel.

Developed by **Alexis N. Williams** and **Chanda R. Harris** of the
[eMMA (electron Microscopy and Microanalysis) group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis),
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.

> Expert user or hate instructions? See [QUICKSTART.md](QUICKSTART.md).

---

## Features

### Core image viewing and annotation
- Multi-format loading: DM4, TIFF (single and multi-frame), MRC/MRCS, PNG, JPEG
- Movie / multi-frame file support: automatic frame detection for DM4, TIFF, and MRC/MRCS stacks
- Interactive contrast: bandpass, percentile, sigma, CLAHE, manual, with saveable presets
- Manual annotation: ROI polygons, rectangles, arrows, circles, text, scale bars, distance and angle measurements
- RELION .star file import: particle picks loaded as circular ROI annotations

### AI-assisted segmentation
- **SAM 3 / SAM 2** (Meta): point, box, scribble, automatic mask generation
- **micro-SAM**: SAM checkpoints fine-tuned for electron and light microscopy
- **YOLO 11 / v8 / v9 / v10** (Ultralytics): object detection and instance segmentation
- **UNet / UNet++ / FPN / DeepLabV3+ / MAnet / PAN** (segmentation-models-pytorch): semantic segmentation
- Pipe YOLO boxes directly into SAM for instance-accurate masks at detection speed

### Movie processing (multi-frame)
- **Frame averaging**: mean, motion-corrected, or dose-weighted (Grant & Grigorieff 2015)
- **Motion correction**: two-pass phase cross-correlation alignment (skimage); sub-pixel accurate
- **Drift trajectory plot**: MotionCor2-style per-frame displacement chart and drift path
- **Dose series analysis**: split movie into equal-dose bins, view averaged images and
  difference maps (`bin N - bin 1`) to visualise beam-induced structural changes
- Configurable frame range (start / end frame): skip early high-motion frames or cap total dose
- Individual frame viewer: step through any frame in the movie

### Analysis
- **Surface area estimation**: 3D surface area from 2D ROI masks using four auto-selected
  methods (ellipsoid, Cauchy, Fourier, fractal for rough surfaces); uncertainty propagation,
  hollow/aggregate detection, multi-GPU batch
- **Particle measurements**: ECD, Feret diameter, circularity, aspect ratio, area, and perimeter
  for every annotated particle; single-image or batch mode across all loaded images; results
  shown in a sortable table with a configurable histogram (count or density, adjustable bins,
  switchable x-axis metric); exported as CSV; uses each image's calibrated pixel size
- **Measurement export**: CLU's `export_measurements` command writes a combined summary CSV
  for all loaded images to `acorn_measurements/measurements.csv` inside the image folder;
  the file and histogram populate the Analysis tab automatically
- **Particle tracking**: link annotations across image series using nearest-neighbour matching;
  configurable max displacement (nm), minimum track length, frame gap tolerance

### Publication plotting and statistics (`acorn_plotting`)
- **Floating Plot window** — pops up automatically when CLU generates a figure; fully dockable,
  resizable, and closable; reopens on the next plot request
- **Plot types**: histogram, violin, box-and-whisker, waterfall/ridge (one row per label), scatter
- **Interactive figures**: hover over any data point to see a tooltip (image name, label, size
  metrics); click a point to navigate ACORN directly to that image
- **Multi-dataset overlay**: load additional CSV files and compare them on the same axes;
  groups are colour-coded and labelled automatically
- **Colour palette picker**: 6 built-in palettes (ACORN, Colorblind-safe, TEM greens, Warm,
  Cool, Grayscale); click any swatch to customise an individual colour with a colour picker
- **Reference markers**: drop vertical reference lines at any x-value by clicking the plot
- **Save PDF / SVG / PNG** directly from the window
- **Statistics tab** (same window): descriptive stats (mean, std, median, IQR), Shapiro-Wilk /
  D'Agostino normality test per group, auto-selected comparison test (Welch t-test or
  Mann-Whitney for 2 groups; one-way ANOVA or Kruskal-Wallis for 3+ groups), Tukey HSD
  or Bonferroni-corrected post-hoc pairwise comparisons; all using scipy, no extra dependencies
- **CLU integration**: `plot_measurements` and `run_statistics` tools; CLU explains p-values
  in plain English and recommends the appropriate test for your data

### Training pipeline
- Tile extraction from annotated images, 8x augmentation, negative-prompt sampling
- Dataset splitting (train / val / test) with reproducible seeds
- In-app **YOLO training**: configure base model, epochs, batch size, image size, then start
- In-app **UNet training**: configure architecture (UNet, UNet++, FPN, DeepLabV3+...),
  encoder backbone (ResNet, EfficientNet, MiT...), and hyperparameters
- Export to COCO JSON + RLE masks
- One-click push to HuggingFace Hub

### Quality and export
- Image quality assessment (blur, contrast, saturation, low-frequency artifacts)
- Export annotated display image (PNG) for external annotation tools or publication
- Batch export over entire image folders
- **NeXus HDF5 export** (`.nxs`): write images, annotations, and particle measurements
  to a NeXus-compatible HDF5 file for CNMS database ingestion and cross-instrument
  data sharing; structure follows `NXroot / NXentry / NXinstrument / NXsample / NXdata`
  with calibrated axes in nm, units attributes, and gzip-compressed image arrays;
  readable by `h5web`, `nexpy`, `pynxtools`, and any NeXus-aware tool;
  CLU command: *"export to NeXus"* / *"export HDF5"* / *"export for the database"*

---

## Plugin architecture

ACORN uses a plugin system based on Python entry points (`acorn.plugins`). Plugins are
discovered automatically at startup with no configuration file needed. Each plugin provides a
tab panel in the main window and can respond to AI tool calls through the `action_requested`
signal on `AcornContext`.

### Built-in plugins

| Plugin package | Where it appears | What it adds |
|----------------|-----------------|--------------|
| `acorn_analysis` | **Measure** tab | Surface area estimation and particle shape measurements (ECD, Feret, circularity) from ROI masks; single or batch; multi-GPU |
| `acorn_tracking` | Floating dock (View → Particle Tracking) | Particle / feature tracking across image sequences; configurable displacement and gap tolerance |
| `acorn_3d` | Floating dock (View → 3D Viewer) | Volume rendering and z-slice navigation for MRC tomograms |
| `acorn_llm` | Floating dock (View → AI Assistant) | Natural-language AI assistant; requires API key or Ollama base URL (see below) |
| `acorn_plotting` | Floating dock | Publication-quality interactive plots, statistical analysis, hover/click data linking (see below) |

The main right panel is organised into workflow tabs — **Annotate**, **Segment**, **Measure**, **Train**, **Export** — plus any plugin-defined tabs. The **Annotate** tab holds the manual annotation tools (and detectors like CryoBLOB that inject here). The **Segment** tab contains the unified SAM / YOLO / UNet panel with loaded-model indicators and shared Accept/Reject controls. Side tools that aren't part of the linear workflow — the 3D viewer, particle tracking, and the AI assistant — open as floating, dockable windows from the **View** menu rather than tabs.

### Writing a plugin

Subclass `acorn.plugin_base.AcornPlugin`, implement `create_panel()`, and register via
`pyproject.toml`. The `AcornContext` object passed at construction gives full access to
application state and signals.

#### Choosing where your plugin lives

By default, `create_panel()` opens a **new tab** in the right panel using `TAB_LABEL` as the
tab name. If your plugin logically belongs inside one of the built-in workflow stages
(**Annotate**, **Segment**, **Measure**, **Train**, **Export**), set `WORKFLOW_STAGE` to that
stage name instead — ACORN will inject your widget directly into that tab's layout rather than creating
a new tab.

```python
class MyMeasurementPlugin(AcornPlugin):
    PLUGIN_ID              = "my_measurement"
    TAB_LABEL              = "My Measurement"  # fallback only — not used when WORKFLOW_STAGE is set
    WORKFLOW_STAGE         = "Measure"          # inject into the Measure tab
    WORKFLOW_SECTION_LABEL = "My Measurements"  # optional QGroupBox label; omit for no header
    sort_order             = 60
```

If `WORKFLOW_STAGE` is set but does not match a known stage, the plugin falls back to its own
tab. To get your own tab unconditionally, leave `WORKFLOW_STAGE = None` (the default).

#### Floating dock instead of a tab

For side tools that aren't part of the main workflow (a 3D viewer, a chat assistant, a tracking
panel), set `FLOATING = True`. Your `create_panel()` widget is placed in a movable, floatable
dock (hidden on startup) with a toggle added to the **View** menu — no tab is created. The
built-in `acorn_3d`, `acorn_tracking`, and `acorn_llm` plugins all use this mode.

```python
class My3DViewerPlugin(AcornPlugin):
    PLUGIN_ID         = "my_viewer"
    FLOATING          = True
    FLOATING_TITLE    = "My Viewer"     # dock title + View-menu label (defaults to TAB_LABEL)
    FLOATING_SHORTCUT = "Ctrl+Shift+V"  # optional keyboard toggle
    FLOATING_AREA     = "right"         # initial dock area: left | right | top | bottom
    FLOATING_MIN_WIDTH = 300            # optional minimum dock width in px

    def create_panel(self) -> QWidget:
        return MyViewerWidget()
```

Returning `None` from `create_panel()` skips the dock entirely — useful for gating on
configuration (e.g. `acorn_llm` returns `None` and offers a Settings shortcut when no API key
is set).

```python
from acorn.plugin_base import AcornPlugin
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

class MyPlugin(AcornPlugin):
    PLUGIN_ID  = "my_plugin"
    TAB_LABEL  = "My Tab"
    sort_order = 50        # lower = further left in the tab bar

    def __init__(self, context):
        super().__init__(context)
        # Signals from core - connect to react to user actions
        context.image_loaded.connect(self._on_image_loaded)
        context.annotations_changed.connect(self._on_annotations_changed)
        context.pixel_size_changed.connect(self._on_pixel_size_changed)
        # CLU calls this whenever it dispatches a tool - filter by action name
        context.action_requested.connect(self._on_action_requested)

    def create_panel(self) -> QWidget:
        w = QWidget()
        QVBoxLayout(w).addWidget(QLabel("Hello from my plugin"))
        return w

    def _on_image_loaded(self, image):
        self._context.set_status(f"Loaded {image.filepath.name}", timeout_ms=3000)

    def _on_annotations_changed(self, store):
        print(f"{len(list(store))} annotations on current image")

    def _on_pixel_size_changed(self, px_nm: float):
        print(f"Pixel size set to {px_nm} nm/px")

    def _on_action_requested(self, action: str, params: dict):
        if action != "my_custom_action":
            return
        # params is whatever JSON CLU included in the tool call
        label = params.get("label", "")
        self._context.set_status(f"my_custom_action called with label={label}")
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."acorn.plugins"]
my_plugin = "my_plugin.plugin:MyPlugin"
```

Run `uv pip install -e .` once and the tab appears automatically on next launch.
See `acorn_tracking/plugin.py` for the simplest real-world example, and
`acorn_llm/plugin.py` for a full plugin that adds CLU tools, menu actions, and streaming.

---

## AI assistant: CLU (Cryo-EM Lab Utility)

CLU is ACORN's built-in natural-language assistant. It lives in the **CLU** tab (added by the
`acorn_llm` plugin) and can perform any ACORN action from a chat message: segmentation,
contrast adjustment, training, dataset management, movie processing, and more.

### Supported providers

| Provider | How to connect |
|----------|---------------|
| Anthropic (Claude) | Set API key in the CLU tab settings, or `ANTHROPIC_API_KEY` env var |
| OpenAI / compatible | Set API key + Base URL; works with GPT-4o, Groq, and local Ollama models |
| Ollama (local) | Set provider to OpenAI-compatible, Base URL = `http://localhost:11434/v1`, model = `acorn-tools` |

### Two model modes

- **Vision model**: receives a thumbnail of the current image along with the message; good for
  questions about image content, contrast assessment, and open-ended analysis
- **Tool model**: uses function calling to drive ACORN actions; good for workflows ("segment
  and queue for training"), comparisons ("try different frame ranges"), and multi-step pipelines
- **Auto** (default): uses the vision model when "Include image" is checked, the tool model
  otherwise

### What CLU can do

CLU has full knowledge of the current application state (loaded image, pixel size, annotations,
model configuration, export queue, training settings) and can:

- Segment structures: *"find the vesicles"*, *"detect all particles"*
- Fix contrast: *"I can't see anything in these movies"*, *"the contrast is off"*
- Drive the full training pipeline: *"prep this for training"*, *"train a YOLO model"*
- Process movies: *"average the frames"*, *"run motion correction on frames 3 to 50"*,
  *"show me how the membrane changes with dose"*
- Run analysis: *"measure the particle diameters"*, *"give me Feret lengths for all particles"*,
  *"measure surface areas of all vesicles"*, *"track particles across images"*
- Export for database / CNMS: *"export to NeXus"* / *"export HDF5"* — writes a `.nxs` file
  with images, annotations, and measurements in NeXus format for database ingestion
- Plot results: *"plot the size distribution"*, *"show a violin plot of Feret length"*,
  *"scatter ECD vs circularity"*, *"waterfall plot by label"* — opens the interactive Plot window
- Run statistics: *"are these two groups different?"*, *"compare the ECD between labels"*,
  *"run stats"* — CLU selects the right test, explains the p-value in plain English
- Control the histogram: *"show as raw counts"*, *"plot Feret length"*, *"use 50 bins"*
- Export measurements: *"export measurements"* — writes `acorn_measurements/measurements.csv`
  in the image folder and opens the results in the Analysis tab
- Scan the whole dataset: *"what are the pixel sizes for all images?"* — CLU navigates to each
  image to load its calibration, then reports a verified summary
- Answer questions without tools: *"how many vesicles are annotated?"*, *"what's the pixel size?"*,
  *"what's the mean ECD?"*

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

## Training models and bringing your own

### How to train your model inside ACORN

ACORN includes a full in-app training pipeline for both YOLO and UNet models.

**Recommended workflow:**

1. Annotate a representative set of images using SAM, YOLO, or manual tools
2. Accept annotations and queue each image for export in the **Export** tab
3. Set the dataset directory and click **Finalize Dataset** to create train/val/test splits
4. Go to the **Train** tab, choose YOLO or UNet, configure hyperparameters, and click **Start Training**
5. Training runs in the background; progress is shown in the Train tab

Use **YOLO** for countable, distinct objects (vesicles, particles, nanoparticles, cells).
Use **UNet** for continuous structures (membranes, filaments, surfaces, dense regions).

For large datasets (e.g., 20+ images @ 4K X 4K), annotate a sample of 5-15 images first, train a model,
then use that model to run batch inference on the remaining images and re-finalize the dataset.
CLU can guide you through this entire workflow from the chat panel.

### Bringing your own model

Got a trained model for your system? Cool,load it up and run it with ACORN. 

**Custom YOLO model:**
- Go to the **YOLO** tab
- Click the model path field and browse to your `.pt` file (any Ultralytics YOLO weight)
- Click **Load** and run detection or segmentation as normal

**Custom UNet model:**
- Go to the **UNet** tab
- Set the architecture and encoder to match how your model was trained
- Browse to your `.pt` checkpoint file and click **Load**

**Custom SAM checkpoint:**
- Go to the **SAM** tab
- Select the backend (SAM 3, SAM 2, or micro-SAM)
- Choose your checkpoint from the dropdown or browse to a local `.pt` file

**Exporting from ACORN for external training:**

ACORN exports datasets in COCO JSON format with RLE-encoded masks, compatible with most
standard training frameworks. Use the CLI for headless export:

```bash
acorn train-export --dataset-dir ./my_dataset   # extract tiles and annotations
acorn finalize --dataset-dir ./my_dataset        # create splits
```

The output directory will contain:
- `images/` - extracted image tiles (PNG)
- `masks/` - binary mask PNGs
- `annotations.json` - COCO-format annotations with RLE masks
- `splits/train.json`, `splits/val.json`, `splits/test.json`

You can then train any framework that accepts COCO-format data (Detectron2, MMDetection,
Ultralytics, custom PyTorch, etc.) and load the resulting checkpoint back into ACORN.

**Pushing your dataset to HuggingFace Hub:**

```bash
acorn push-to-hub --dataset-dir ./my_dataset --repo-id your-username/my-dataset
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

### Shared workstation (Linux, requires sudo)

```bash
sudo bash setup_system.sh
```

Installs ACORN to `/opt/acorn` with a shared model cache at `/opt/acorn/models/`.
Writes `/etc/profile.d/acorn.sh` (shared env vars), creates `/usr/local/bin/acorn` and
`/usr/local/bin/acorn-gui` wrappers, and registers a desktop entry for ThinLinc / GNOME / KDE.
No per-user install needed; any user on the machine can run `acorn-gui` immediately.

After the initial system setup, push updates from the dev copy without sudo:

```bash
bash deploy.sh
```

`deploy.sh` syncs source files and only reinstalls Python dependencies if `pyproject.toml`
has changed (sha256 hash check), keeping deploys fast.

### Migrating to a new machine

```bash
# On the old machine - copy shared models to the new one
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

The editable dev install means changes to `.py` files are live immediately with no reinstall needed.

---

## AI models

### SAM 3, SAM 2, and micro-SAM

ACORN supports three SAM backends, selected automatically at load time:

| Backend | Package | Notes |
|---------|---------|-------|
| **SAM 3** | `pip install sam3` | Recommended: fastest, best cryo-EM accuracy |
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
| vit_b_em_organelles | micro-SAM | 375 MB | EM organelles; recommended for cryo-EM |
| vit_l_em_organelles | micro-SAM | 760 MB | EM organelles, larger / more accurate |
| vit_b_lm | micro-SAM | 375 MB | Light microscopy |
| vit_l_lm | micro-SAM | 760 MB | Light microscopy, larger |
| vit_b / vit_l / vit_h | micro-SAM | 375 MB - 2.4 GB | Generic SAM (Meta original) |
| sam3.pt | SAM 3 | ~2.4 GB | SAM 3 (HF login required) |
| sam2_hiera_large.pt | SAM 2 | ~2.4 GB | SAM 2 fallback (HF login required) |
| sam2_hiera_base_plus.pt | SAM 2 | ~320 MB | SAM 2 smaller variant |
| yolo11n-seg.pt | YOLO | 6 MB | Nano segmentation; recommended starter |
| yolo11s/m/l/x-seg.pt | YOLO | 22-130 MB | Larger segmentation models |
| yolo11n/s/m/l/x.pt | YOLO | 6-130 MB | Detection-only variants |

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
                context.py           AcornContext plugin interface
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
  acorn_analysis/  Analysis plugin: surface area tab
  acorn_tracking/  Track plugin: particle tracking tab
  acorn_3d/        3D plugin: volume rendering tab
  acorn_llm/       CLU plugin: AI assistant tab
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
| Gatan DM4 | .dm4 | Yes: 3D stacks auto-detected |
| TIFF | .tif, .tiff | Yes: multi-page stacks |
| MRC / MRCS | .mrc, .mrcs | Yes: all 3D MRC data |
| PNG | .png | No |
| JPEG | .jpg, .jpeg | No |

When a multi-frame file is opened, ACORN automatically mean-averages the frames for display
and shows the **movie bar** below the canvas with controls for frame selection, compression
method, motion correction, and dose series analysis.

---

## Contributing and feedback

Bug reports, feature requests, and pull requests are welcome via GitHub:
**[https://github.com/vitrifyai/acorn_software](https://github.com/vitrifyai/acorn_software)**

For direct correspondence, to extend positive vibes, or tell us this sucks directly: **williamsan@ornl.gov**

---

## About eMMA

The [electron Microscopy and Microanalysis (eMMA) group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis)
is part of the Center for Nanophase Materials Sciences (CNMS) at Oak Ridge National Laboratory.
The group develops and applies advanced multi-scale materials characterization techniques (including aberration-corrected STEM, in-situ and cryo-EM, electron spectroscopy (EELS/EDX),
and atom probe tomography) to uncover structure-property-function relationships across
metals, ceramics, composites, nanomaterials, and biological specimens.

ACORN was developed to accelerate data analysis workflows for the group's low-dose cryo-EM and
analytical STEM programs, with a focus on ease of use for domain experts and compatibility
with standard ML training pipelines.

---

## License

See [LICENSE](LICENSE) for details.
