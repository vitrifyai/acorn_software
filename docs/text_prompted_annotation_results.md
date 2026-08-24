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
