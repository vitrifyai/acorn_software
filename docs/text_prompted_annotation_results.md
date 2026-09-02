# Text-prompted annotation in ACORN — measurements

Recorded 2026-08-24. Kept for a future write-up: these are the numbers behind the
`run_sam_text` feature and behind the decision not to ship open-vocabulary YOLO.

Everything below is measured, not estimated. Reproduce with the scripts named at
the end.

---

## 1. Test data

A simulated cryo-TEM micrograph from ACORN's own physics engine
(`acorn_tem_sim.engine_io.generate_tem_advanced`), which is the reason the ground
truth is exact rather than hand-counted.

| | |
|---|---|
| Specimen | PLGA nanoparticles in vitreous ice |
| Image | 512 × 512 px, 1.8 nm/px |
| Microscope / detector | Krios, K3 |
| Dose | 40 e⁻/Å² |
| Defocus | −2.0 µm |
| Ice thickness | 60 nm |
| Particle diameter | 30 ± 6 nm (≈17 px) |
| **Ground truth** | **30 particles per image**, written as ROI polygons in the sidecar |
| Seed | 7 |

A FIB-SEM simulation (`acorn_fib_sim`, default `SimConfig`, 512 × 640) was used as
a second, deliberately different modality.

---

## 2. SAM 3 responds to appearance, not to terminology

SAM 3 (`facebook/sam3`), text prompt via `Sam3Processor.set_text_prompt`,
confidence 0.5, on the cryo-TEM image above.

| Prompt | Masks | Median score | Particle-sized |
|---|---:|---:|---:|
| `dark round blob` | **29** | 0.80 | 29 |
| `black dot` | 28 | 0.81 | 28 |
| `round black particle` | 28 | 0.81 | 28 |
| `dark circle` | 28 | 0.80 | 28 |
| `small dark circle` | 28 | 0.80 | 28 |
| `black circle` | 25 | 0.80 | 25 |
| `dark spot` | 15 | 0.53 | 15 |
| `round dark object` | 0 | — | — |
| `dark blob` | **0** | — | — |
| `dark grain` | 0 | — | — |
| `dark ellipse` | 0 | — | — |
| **`nanoparticle`** | **0** | — | — |
| **`vesicle`** | **0** | — | — |

Two findings:

1. **The discipline's own vocabulary returns nothing.** "nanoparticle" and
   "vesicle" find zero objects at every confidence tried (0.5, 0.2, 0.05), while
   an everyday visual description finds 29 of 30.
2. **The phrasing is not guessable.** `dark round blob` finds 29; `dark blob`
   finds 0. One word between a working prompt and a useless one.

This is what `acorn.core.vocabulary` exists to encode.

### Confidence sweep

| Prompt | 0.5 | 0.2 | 0.05 |
|---|---:|---:|---:|
| `black dot` | 28 | 31 | 76 |
| `dark circle` | 28 | 31 | 88 |
| `particle` | 0 | 0 | 32 |
| `cell` | 0 | 0 | 31 |
| `nanoparticle` | 0 | 0 | 0 |
| `vesicle` | 0 | 0 | 0 |

Lowering confidence eventually admits generic words, at the cost of large numbers
of false positives (76 and 88 against a ground truth of 30).

---

## 3. Open-vocabulary YOLO does not transfer to micrographs

YOLO-World v2 small (`yolov8s-worldv2.pt`), ultralytics 8.4.41, via
`set_classes()`. Confidence swept 0.25 → 0.01.

| Image | Prompt | Detections | Ground truth |
|---|---|---:|---:|
| cryo-TEM sim | `dark round blob` | 0 | 30 |
| cryo-TEM sim | `black dot` | 1 | 30 |
| cryo-TEM sim | `dark circle` | 1 | 30 |
| cryo-TEM sim | `nanoparticle` | 0 | 30 |
| cryo-TEM sim | `vesicle` | 1 | 30 |
| cryo-TEM sim, 2× upscaled | `dark round blob` | 0 | 30 |
| cryo-TEM sim, 4× upscaled | `dark round blob` | 0 | 30 |
| FIB-SEM sim | 5 phrases tried | 0 | — |

**Not an object-scale effect** — identical failure at 17, 34 and 68 px objects.
**Not one unrepresentative image** — the same on a second modality. YOLO-World's
vision–language alignment is trained on natural photographs and does not transfer
to greyscale micrographs. It is therefore not exposed in ACORN.

An implementation note: `set_classes()` raises a device mismatch when the model is
on CUDA (`Expected all tensors to be on the same device`); the measurements above
ran the text encoder on CPU.

---

## 4. Text prompt → annotations → trained detector

The full chain, run end to end.

| Step | Result |
|---|---|
| `predict_text("vesicles")` on image 1 | 29 masks via `dark round blob` |
| `predict_text("vesicles")` on image 2 | 28 masks via `dark round blob` |
| → ROI annotations | 64 polygons labelled `vesicle` |
| → `add_image` ×2 | 8 tiles (256 px), 64 instances |
| → `finalize_dataset` | 2 sources, 64 annotations, `{'vesicle': 64}` |
| → `convert_to_yolo` | 8 images, 8 polygon label files, `nc: 1`, `names: ['vesicle']` |
| → **YOLO11n-seg, 3 epochs** | **trained; `best.pt`, 5.95 MB** |
| trained model classes | `{0: 'vesicle'}` |

Validation after 3 epochs: box mAP50 0.067, mask mAP50 0.080, mask recall 0.83.
**These numbers demonstrate the pipeline, not model quality** — 3 epochs on 8
tiles from 2 source images. A usable detector needs tens of images and ~100
epochs.

The significance is the loop: SAM 3 converts a *description* into labelled data,
and the trained model then knows the class *by name* without SAM, at a few
milliseconds per image (2.2 ms inference here) instead of SAM 3's ~17 s load plus
per-image cost.

### Two defects this exposed, both fixed

- `convert_to_yolo` wrote the exporter's generic `Foreground` category into
  `data.yaml` even with zero annotations using it, training a two-class model
  whose first class had no examples.
- `finalize_dataset` raised `No images assigned to Train` for 2 source images at
  34 %/34 % splits, where rounding took every image away from train.

---

## 5. Comparison on the same image

| Method | Found | Ground truth | Notes |
|---|---:|---:|---|
| SAM 3, text `dark round blob` | 29 | 30 | confidence 0.5 |
| CryoBLOB, LoG detector | 27 | 30 | `min_sigma` 2, `max_sigma` 12 |
| SAM 3, automatic point grid | 15 | 30 | plus whole-image masks as false positives |
| YOLO-World, text | 0–1 | 30 | any phrase, any confidence |

---

## Reproducing

- Simulation: `acorn_tem_sim.engine_io.generate_tem_advanced`, seed 7, parameters in §1
- SAM 3 text: `acorn.core.sam_predictor.SAMPredictor.predict_text`
- Vocabulary: `acorn.core.vocabulary`
- Export chain: `acorn.export.training_exporter.add_image` →
  `acorn.export.dataset_finalizer.finalize_dataset` →
  `acorn.core.yolo_trainer.convert_to_yolo`
- Checkpoint: `facebook/sam3` at
  `3c879f39826c281e95690f02c7821c4de09afae7`
- SAM 3 requires `bpe_simple_vocab_16e6.txt.gz`, which its own wheel omits — see
  `acorn.core.sam3_assets`

Environment: Python 3.12.13, torch 2.6.0+cu126, ultralytics 8.4.41,
Tesla V100-SXM3-32GB.

---

## 6. Replication on a second specimen and modality — SEM bacterial spores

Recorded 2026-09-02. §2 established the vocabulary effect on one image, one
specimen, one modality. A single instance is an anecdote, so this repeats it on
simulated SEM spores: different physics engine (Monte Carlo electron transport,
not multislice), different specimen, different contrast mechanism, bright
objects rather than dark.

### 6.1 Test data

Four conditions from `acorn_sem_sim`, 512 x 512 at 20 nm/px, 30 spores placed
per field, seed 11. Ground truth is exact because the scene knows where it put
them.

| Case | Coating | kV | Charging | Clustering | Truth components |
|---|---|---:|---:|---:|---:|
| `coated` | 12 nm gold on silicon | 5.0 | 0.0 | 0.40 | 24 |
| `uncoated_clean` | none, on resin | 1.0 | 0.0 | 0.40 | 24 |
| `uncoated_charged` | none, on resin | 1.0 | 1.2 | 0.40 | 24 |
| `uncoated_dense` | none, on resin | 1.0 | 0.5 | 0.85 | 26 |

Note the gap between 30 placed and 24 components: touching spores merge under
connected-component labelling. This matters below.

### 6.2 The vocabulary effect replicates, and more sharply

Thirteen phrases, confidence 0.5. Two worked. Eleven returned nothing.

| Prompt | Masks (all four cases) |
|---|---:|
| `oval object` | **30** |
| `oval cell` | **30** |
| `bacterial spore` | 0 |
| `spore` | 0 |
| `bacteria` | 0 |
| `rod shaped cell` | 0 |
| `bright oval blob` | 0 |
| `bright blob` | 0 |
| `white oval object` | 0 |
| `bright ellipse` | 0 |
| `grain of rice` | 0 |
| `peanut` | 0 |
| `bean` | 0 |

The §2 findings hold, and the second is now stronger:

1. **The discipline's own vocabulary returns nothing.** "spore", "bacterial
   spore" and "bacteria" find zero objects; "oval object" finds every one.
2. **The phrasing is not guessable, and not even consistent with §2.** There the
   working phrase was `dark round blob` and `dark blob` failed. Here `bright
   oval blob` fails and the bare `oval object` succeeds — so the "<adjective>
   <shape> blob" pattern that worked on nanoparticles does not transfer. There
   is no rule to learn, only a lookup to build.

### 6.3 Where it works, it is close to exact

`oval object`, confidence 0.5, evaluated against the label map:

| Case | Masks | Recall @ IoU>=0.5 | Precision @ IoU>=0.5 |
|---|---:|---:|---:|
| `coated` | 30 | 1.00 | — |
| `uncoated_clean` | 30 | 1.00 | — |
| `uncoated_charged` | 30 | 0.96 | 0.77 |
| `uncoated_dense` | 30 | 1.00 | — |

Median IoU of the best-matching prediction per truth object: **0.90**.
At the stricter IoU >= 0.75 on `uncoated_charged`: TP 18, FP 12, FN 6.

**The apparent false positives are a ground-truth artefact, not model error.**
Measuring what fraction of each predicted mask lands on true spore material:

- 25 masks at > 90%
- 5 masks at 50-90%
- **0 masks below 50%**

SAM 3 returned exactly 30 masks for exactly 30 placed spores. The five partial
matches are touching spores it separated correctly and the connected-component
truth had merged. On instance boundaries the model was more correct than the
labels it was being scored against — which is worth remembering whenever a
segmentation metric is computed against component labelling.

### 6.4 Simulated charging does not perturb it

`uncoated_clean` and `uncoated_charged` returned identical mask counts and
near-identical recall, despite charging raising mean spore/substrate contrast
from 1.84 to 2.47 and blowing out individual spores (see
`acorn_sem_sim.charging`). The artefact that defeats intensity thresholding —
it changes measured object area by 2.2x, per `acorn.core.conditions` — does not
measurably affect SAM 3.

### 6.5 What this means for fine-tuning

There is no accuracy headroom to train for: recall is 0.96-1.00 with a median
IoU of 0.90, zero-shot, including on the hardest condition. Fine-tuning SAM 3
or micro-SAM to segment these better would be optimising a solved problem.

The reproducible defect is the vocabulary mapping, not the segmentation. That
is the case for fine-tuning SAM 3 specifically — grounding "spore" and
"vesicle" in real images so the model answers to the terms the field uses —
and it is a different objective from the usual reason for fine-tuning a
segmenter.

### 6.6 Limits of this measurement

Simulated data only: clean backgrounds, no debris, no contamination, no focus
variation, no stage drift. Whether recall near 1.00 survives a real micrograph
is untested and is the obvious next measurement. The ground truth merges
touching spores, which understated precision until §6.3 checked it directly.

### Reproducing §6

- Scenes: `acorn_sem_sim.scenes.build("spores", ...)`, seed 11, parameters in §6.1
- Imaging: `acorn_sem_sim.imaging.simulate`, TLD detector, 800 e-/px, 12000 trajectories
- Charging: `acorn_sem_sim.imaging.Detector(charging=...)`, off by default
- SAM 3: `Sam3Processor.set_text_prompt` directly, bypassing
  `acorn.core.vocabulary`, so the phrases above are the raw strings

---

## 7. Training on simulated data, and testing on real micrographs

Recorded 2026-09-02. Section 4 demonstrated the annotate-export-train chain on
8 tiles from 2 images for 3 epochs, which proved the plumbing and nothing else.
This is the same loop at a size that supports a claim, and then run against real
data of the same specimens.

Full archive: `/nas-158/Alexis/ACORN_methods_paper/`

### 7.1 Setup

400 simulated images per specimen at 640x640, split 280/60/60 by disjoint seed
range, YOLO11s-seg trained from an ImageNet-pretrained start. No real data in
training, no hand annotation.

### 7.2 The headline: simulated is near-perfect, real is not

| Dataset | Evaluated on | Box mAP@50 | Mask mAP@50 | Mask P | Mask R |
|---|---|---:|---:|---:|---:|
| Nanoparticles | simulated test | 0.985 | 0.943 | 0.950 | 0.932 |
| Nanoparticles | REAL PLGA | — | — | — | 16.0 det/image, no ground truth |
| Spores | simulated test | 0.987 | 0.990 | 0.990 | 0.986 |
| Spores | REAL micrographs | 0.091 | 0.079 | 0.213 | 0.127 |

Real cryo-TEM: 24 PLGA micrographs from `Atlas/ARM/PLGA_LA`, 4096 px at 2.23
A/px, binned 8x to the training scale. Real SEM: 26 micrographs with 1085
annotated spores from `SEM_Imaging`.

**Transfer held for cryo-TEM and failed for SEM.** That contrast is the result.

### 7.3 Three measured domain gaps in the SEM case

Each was measured from the real data before being closed, not guessed.

1. **Size.** Simulated spores were 900-1600 nm long; measured from 1087
   annotations at correct per-image scale, real spores are 942/1267/2413 nm
   (p10/median/p90).
2. **Magnification.** The real set spans 3.6-19.3 nm/px (5.8k-31k x) against a
   single simulated 20 nm/px. Spore count also had to become an areal density
   (0.35-0.95 per um^2), or a 2.5 um field holds a hundred 1.3 um spores.
3. **Substrate.** The largest gap. Simulated spores sat bright on near-black
   smooth substrate (background mean 20, std 24); real spores sit on rough
   organic debris BRIGHTER than they are (mean 100, std 62).

Closing all three moved real mask mAP@50 from 0.008 to 0.079 and recall from
0.13 to 0.16 -- five-fold, and still a failure. Where the detector fires on real
data it is correct; it simply misses most spores.

### 7.4 Why cryo-TEM works and SEM does not

Cryo-TEM contrast is dominated by physics the simulator models -- projected
potential, CTF, dose, detector DQE -- over uniform vitreous ice. SEM contrast is
dominated by specimen preparation: what the spores sit on, how they were
mounted, what debris came with them. The simulator models electron transport
correctly and the specimen mount not at all.

The reading: **simulation-trained models transfer when imaging physics dominates
appearance, and fail when preparation does.**

### 7.5 A distractor class that backfired

Adding the engine's `Contamination` component to teach the detector to ignore
debris made the cryo-TEM result much worse: mean detections on real micrographs
fell from 16.9 to 3.8 per image, and what remained fired on debris rather than
particles. The engine renders contamination as large flat polygonal slabs, which
overlay the particles while carrying no label, so training taught suppression in
exactly those regions. Removed. A distractor class only helps when it resembles
the real distractor.

### 7.6 A defect the metrics did not catch

The first nanoparticle dataset used `pixel_size_a` of 1.5-2.2 where the engine
expects angstroms: a 115 nm field with 30 nm particles packed into it. Ground
truth was exactly right for that scene and training reached box mAP@50 0.95 on
physically impossible images. No metric flagged it; plotting labels over images
did.

### 7.7 Data-quality findings in the real SEM set

- ACORN sidecars record `pixel_size_nm` 1.867 for every image; the true value is
  in the Zeiss `CZ_SEM` tag and ranges 3.6-19.3 nm/px. Any physical measurement
  taken from those sidecars is wrong by a per-image factor. Worth fixing at
  source.
- Of 1085 annotations, 793 (73%) are spore-like by size and convexity; the rest
  are debris contours. Scoring against the clean subset does not change the
  conclusion.

### Reproducing §7

`code/` in the archive: `gen_nano.py`, `gen_spores.py`, `train.py`,
`evaluate.py`, `build_real_sem.py`, `eval_real_sem.py`, `real_plga.py`,
`figures.py`, `tables.py`. Environment: Python 3.12, torch 2.6.0+cu126,
ultralytics, Tesla V100-SXM3-32GB.
