# ACORN — Quick Start Guide

**For scientists who want to annotate microscopy images — no software background needed.**

---

## What is ACORN?

ACORN (Annotate, Curate, Observe, Review, Navigate) is a desktop application developed by the
[eMMA group](https://www.ornl.gov/group/electron-microscopy-and-microanalysis) at Oak Ridge
National Laboratory.  It was built to help microscopists answer scientific questions faster by
combining image viewing, manual annotation, and AI-assisted segmentation in one place.

If you have cryo-EM, STEM, TEM, or other microscopy images and you want to:

- measure features or draw regions of interest
- label structures for analysis or publication
- automatically detect and outline particles, organelles, or other objects
- build a labelled dataset for machine learning

...then ACORN is for you.

---

## What file types does it open?

| Format | Extension |
|--------|-----------|
| Gatan DM4 | .dm4 |
| TIFF | .tif, .tiff |
| MRC / MRCS | .mrc, .mrcs |
| PNG | .png |
| JPEG | .jpg, .jpeg |

---

## Getting started in 3 steps

### Step 1 — Install

Open a terminal and run:

```
bash install.sh
```

This sets up everything automatically.  It takes a few minutes and needs an internet connection.
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

---

## Typical workflow for a domain expert

1. Open your image with **Ctrl+O**
2. Adjust contrast in the **Contrast** tab until features are clear
3. Go to the **SAM** tab, click **Load Model**, then click **+ Positive Point** and click on a
   structure you care about — the AI will outline it
4. Click **Commit & New** to lock that annotation and move on to the next object
5. When done, go to **Export** and save your annotations as an image, CSV, or dataset

---

## Tips

- **Ctrl+Z** undoes the last annotation
- **Scroll wheel** zooms in and out on the canvas
- **Right-click drag** pans the image
- The **SAM** tab works best for irregular shapes (organelles, particles, membranes)
- The **YOLO** tab works best when you have many similar objects to detect all at once
- If you are not sure which AI tool to use, start with SAM — it is the most flexible

---

## Getting help

For questions or suggestions contact **williamsan@ornl.gov** or open an issue on GitHub
(link in the main README).

ACORN was developed by **Alexis Williams** and **Chanda Harris** of the eMMA group,
Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.
