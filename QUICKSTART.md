# ACORN — Quick Start Guide

**For scientists who want to analyze microscopy images — no software background needed.**

---

## What is ACORN?

ACORN (Annotate, Curate, Observe, Review, Navigate) is a desktop application developed by the
[eMMA group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis) at Oak Ridge
National Laboratory. It was built to help microscopists answer scientific questions faster by
combining image viewing, manual annotation, AI-assisted segmentation, and a natural-language
assistant in one place.

If you have cryo-EM, STEM, TEM, or other microscopy images and you want to:

- view and compress movie stacks the same way you would in cryoSPARC
- measure features or draw regions of interest
- label structures for analysis or publication
- automatically detect and outline particles, organelles, or other objects
- build a labelled dataset for machine learning
- just type what you want to do in plain English and let the AI figure out the rest

...then ACORN is for you.

---

## What file types does it open?

| Format | Extension | Movie / multi-frame |
|--------|-----------|---------------------|
| Gatan DM4 | .dm4 | Yes — 3D stacks auto-detected |
| TIFF | .tif, .tiff | Yes — multi-page stacks |
| MRC / MRCS | .mrc, .mrcs | Yes — all 3D MRC data |
| PNG | .png | No |
| JPEG | .jpg, .jpeg | No |

When you open a movie file, ACORN automatically mean-averages all the frames for display so
you see a single clean image right away — nothing crashes or locks up. The original frames
are kept in memory and you can re-process them at any time from the **movie bar** that appears
below the image.

---

## Getting started in 3 steps

### Step 1 — Install

Open a terminal and run:

```
bash install.sh
```

This sets up everything automatically. It takes a few minutes and needs an internet connection.
You do not need to install Python or anything else first — the installer handles it.

### Step 2 — Open ACORN

After installation, double-click the **ACORN** icon on your Desktop.

Or open a terminal and type:

```
acorn-gui
```

### Step 3 — Open your image

Go to **File > Open** (or press **Ctrl+O**) and select your image file.

---

## The panels on the right

| Tab | What it does |
|-----|-------------|
| **Contrast** | Adjust brightness and contrast — pick a preset or dial it in manually |
| **Annotate** | Draw arrows, circles, rectangles, scale bars, ROIs, and measurements by hand |
| **SAM** | AI-assisted segmentation — click on an object and the AI outlines it for you |
| **YOLO** | Automatically detect and label every object of a given type in the image |
| **UNet** | Semantic segmentation using a custom-trained model |
| **Export** | Save annotated images, measurement data, or training datasets |
| **Train** | Prepare labelled data and train a custom YOLO or UNet model on your images |
| **Analysis** | Two sub-tabs: **Surface Area Analysis** — estimate 3D surface area of large objects (vesicles, cells) from 2D ROI masks; **Particle Measurements** — ECD, Feret diameter, circularity, aspect ratio, area, and perimeter for nanoparticles and small objects; histogram with count/density toggle and adjustable bins |
| **Track** | Link annotations across image series to track particle or feature motion |
| **3D** | Volume rendering and z-slice navigation for MRC tomograms |
| **CLU** | Natural-language AI assistant — type what you want to do, the AI does it |

The last four tabs (Analysis, Track, 3D, CLU) are added automatically by ACORN's plugin system.
If a plugin is not installed or its dependencies are missing, that tab simply will not appear.

---

## Working with movie files

When you open a DM4, TIFF, or MRC movie stack, ACORN:

1. Detects that the file is a stack (not a single image)
2. Mean-averages all frames and shows the result immediately
3. Shows the **movie bar** below the image with compression controls

### Movie bar controls

- **Method** — choose how to combine frames:
  - *mean* — fast simple average, good for a quick look
  - *motion corrected* — aligns frames before averaging (like MotionCor2 / cryoSPARC); best for
    cryo-EM data with beam-induced motion
  - *dose weighted* — applies the Grant & Grigorieff (2015) dose filter before averaging; gives
    the best resolution for high-dose movies
- **Start / End frame** — narrow the frame range before compressing; useful for K2/K3 datasets
  with thousands of frames where you want to skip the noisy early frames or cap total dose
- **Frame viewer** — step through individual frames one at a time; set to *avg* to go back to
  the compressed view
- **Apply** — recalculate the compressed image with the current settings
- **Motion plot** — appears after a motion-corrected run; shows the MotionCor2-style drift
  trajectory and per-frame displacement bar chart so you can see how much the sample moved
- **Dose series** — opens an interactive window that splits the movie into equal-dose bins,
  shows the per-bin averages, and optionally shows difference images (bin N minus bin 1)
  so you can watch structural changes accumulate with dose

### If you can't see anything in a movie

This is the most common problem with raw cryo-EM data. The individual frames are so noisy that
the image looks like static. Just compress the frames: select *motion corrected* in the movie
bar and click **Apply**. If you are using CLU, just type something like *"I can't see
anything"* or *"the movie looks terrible"* — CLU knows it is a movie and will compress and
fix the contrast for you automatically.

---

## Using CLU — the AI assistant

CLU lives in the **CLU** tab on the right. You type what you want to do in plain English and
CLU figures out how to do it.

### Connecting CLU to a model

Open the **CLU** tab and click the settings gear. You can connect CLU to:

| Provider | What to enter |
|----------|--------------|
| Anthropic (Claude) | Your API key from console.anthropic.com |
| OpenAI / compatible | API key + Base URL |
| Local Ollama model | Provider = OpenAI-compatible, Base URL = `http://localhost:11434/v1`, model = `acorn-tools` |

For a fully offline setup, see the Ollama section in the main README.

### What you can ask CLU

CLU understands microscopy intent, not just commands. You can say things like:

- *"find the vesicles"* — CLU loads SAM and segments them
- *"I can't see anything"* — CLU compresses the movie and fixes contrast
- *"the contrast is off"* — CLU picks the right contrast method for your image type
- *"detect all the particles"* — CLU runs YOLO detection
- *"prep this for training"* — CLU annotates, accepts, and queues the image for export
- *"train a YOLO model"* — CLU configures and starts training
- *"annotate all images at once"* — CLU runs SAM on every loaded image in one go
- *"measure particle diameters"* — CLU runs batch particle analysis and shows ECD/Feret in the Analysis tab
- *"show as raw counts"* / *"plot Feret length"* / *"use 50 bins"* — CLU adjusts the histogram live
- *"export measurements"* — saves `acorn_measurements/measurements.csv` in your image folder and opens the data in the Analysis tab
- *"average the frames"* — CLU compresses with mean averaging
- *"run motion correction on frames 3 to 50"* — CLU clips the frame range and motion-corrects
- *"show me how the membrane changes with dose"* — CLU opens the dose series tool
- *"how many vesicles are annotated across all images?"* — CLU reads the full dataset state
- *"which images still need annotations?"* — CLU lists unannotated images from the loaded set
- *"what's the pixel size?"* — CLU reads it from the loaded image metadata

CLU always knows the full state of every loaded image — annotation counts, labels, pixel sizes,
and export queue status — across the entire dataset, not just the current image. You do not
need to tell it anything twice.

### Two model modes

- **Tool model** (default) — CLU calls application functions to do things; use this for any
  action request
- **Vision model** — CLU receives a thumbnail of the image along with your message; use this
  for questions about image content or open-ended analysis
- **Auto** — uses the vision model when "Include image" is checked, tool model otherwise

---

## Typical workflow — cryo-EM movie

1. Open your movie file with **Ctrl+O** — ACORN shows a mean average immediately
2. In the movie bar, select *motion corrected* and click **Apply** to get a clean averaged image
3. Click **Motion plot** to verify the drift was small and the alignment succeeded
4. Switch to the **Contrast** tab and pick the *Default (Bandpass)* preset for cryo-EM
5. Go to the **SAM** tab, click **Load Model**, click a vesicle or particle to annotate it
6. Click **Commit & New**, repeat for more structures
7. When done, go to **Export** and save your annotations or queue images for training

Or: open the **CLU** tab and type *"motion correct the movie then find the vesicles and prep
for training"* — CLU will run the full workflow for you.

---

## Typical workflow — single image annotation

1. Open your image with **Ctrl+O**
2. Adjust contrast in the **Contrast** tab until features are clear
3. Go to the **SAM** tab, click **Load Model**, then click **+ Positive Point** and click on a
   structure — the AI will outline it
4. Click **Commit & New** to lock that annotation and move on to the next object
5. When done, go to **Export** and save your annotations as an image, CSV, or dataset

---

## Tips

- **Ctrl+Z** undoes the last annotation
- **Scroll wheel** zooms in and out on the canvas
- **Right-click drag** pans the image
- The **SAM** tab works best for irregular shapes (organelles, particles, membranes)
- The **YOLO** tab works best when you have many similar objects to detect all at once
- For large K2/K3 movies, use the Start/End frame controls to skip noisy early frames
- The **Dose series** button is useful for radiation-sensitive samples — you can see exactly
  when structures start to degrade
- In the **Motion plot**, a drift trajectory that curves or spikes late in the series often
  indicates beam-induced bubbling or charging — useful for deciding how many frames to keep
- If you are not sure which AI tool to use, open CLU and just describe what you see

---

## Working with large datasets

If you have 20 or more images to analyze, the recommended approach is a two-phase workflow
that avoids annotating every image by hand.

### Phase 1 — Annotate a representative sample and train a model

1. Load your images with **File > Open Folder**
2. Annotate 5–15 diverse images using SAM or manual tools
3. Accept the annotations and queue each image for export
4. Go to the **Export** tab and finalize the dataset (creates train/val/test splits)
5. Go to the **Train** tab, configure your model (YOLO for countable objects, UNet for
   continuous structures like membranes), and click **Start Training**

### Phase 2 — Run the trained model on the rest

6. After training finishes, go to the **YOLO** or **UNet** tab and load your new checkpoint
7. For each remaining image: run detection, accept the annotations, queue for export
8. Re-finalize the dataset — new images are added to the splits automatically
9. Optionally re-train on the expanded dataset for further improvement

### Letting CLU handle it

You can ask CLU to do this entire workflow in plain English:

- *"annotate all images with SAM, then prep for training"* — CLU batches SAM across every
  image, accepts, and queues automatically
- *"train a YOLO model on the queued images"* — CLU finalizes the dataset and starts training
- *"run detection on all images with the loaded YOLO model"* — CLU steps through each image

### Analysis across many images

The **Analysis** tab's batch mode automatically uses each image's own calibrated pixel size,
so you do not need to set a uniform pixel size when images were acquired at different
magnifications.

**Surface Area Analysis**: enable **Group same-label annotations across images** to combine all
*vesicle* annotations from all images into one distribution — or uncheck it to compare
image-by-image variation.

**Particle Measurements**: select labels and click **Run** to compute shape metrics (ECD,
Feret diameter, circularity, aspect ratio, area, perimeter) across all loaded images in one
pass. Results appear in a sortable table and can be exported as CSV.

---

## Writing a plugin

If you want to add your own tab to ACORN, you can write a plugin in about 20 lines of Python.
The full guide is in the main README, but here is the minimal example:

```python
from acorn.plugin_base import AcornPlugin
from PyQt6.QtWidgets import QLabel, QWidget, QVBoxLayout

class MyPlugin(AcornPlugin):
    PLUGIN_ID = "my_plugin"
    TAB_LABEL = "My Tab"
    sort_order = 50   # controls tab position

    def __init__(self, context):
        super().__init__(context)
        # context gives access to images, annotations, pixel size, status bar, menus
        context.image_loaded.connect(self._on_image_loaded)
        # respond to CLU tool calls:
        context.action_requested.connect(self._on_action)

    def create_panel(self):
        w = QWidget()
        QVBoxLayout(w).addWidget(QLabel("Hello from my plugin"))
        return w

    def _on_image_loaded(self, image):
        self._context.set_status(f"Loaded: {image.filepath.name}")

    def _on_action(self, action_name, params):
        if action_name == "my_custom_action":
            print("CLU called my action with", params)
```

Register it in `pyproject.toml`:

```toml
[project.entry-points."acorn.plugins"]
my_plugin = "my_plugin.plugin:MyPlugin"
```

Then `uv pip install -e .` and the tab appears automatically.

---

## Getting help

For questions or suggestions contact **williamsan@ornl.gov** or open an issue on GitHub
(link in the main README).

ACORN was developed by **Alexis Williams** and **Chanda Harris** of the eMMA group,
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.
