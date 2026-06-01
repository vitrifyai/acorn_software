"""LLM agent — runs in a QThread, streams tokens and fires tool signals."""
from __future__ import annotations
import json
from typing import Optional

from PyQt6.QtCore import QThread, pyqtSignal

from acorn_llm.config import LLMConfig

# ---------------------------------------------------------------------------
# Tool definitions (provider-agnostic)
# ---------------------------------------------------------------------------

_TOOLS: list[dict] = [
    {
        "name": "run_sam_auto",
        "description": (
            "Run SAM automatic segmentation to find all objects on the current image. "
            "Use when the user asks to find, segment, or detect something and SAM is loaded."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label for found objects (e.g. 'lamella', 'vesicle', 'particle')"},
            "points_per_side": {"type": "integer", "description": "Grid density 4–128. Default 32. Use higher for small dense objects."},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "run_yolo_detect",
        "description": (
            "Run YOLO object detection on the current image. "
            "Use when YOLO is loaded and the user wants bounding-box detections."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label for detected objects"},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "run_yolo_segment",
        "description": (
            "Run YOLO instance segmentation (requires a YOLO-seg model). "
            "Produces precise polygon masks."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label"},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "accept_annotations",
        "description": "Accept all pending annotations, making them permanent ROIs.",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["sam", "yolo", "unet", "all"],
                "description": "Which model's pending annotations to accept. Default 'all'.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "queue_for_export",
        "description": "Add the current image and its annotations to the training export queue.",
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "start_training",
        "description": (
            "Start model training using the settings already configured on the Train tab. "
            "A confirmation dialog will be shown before anything executes."
        ),
        "properties": {
            "summary": {"type": "string", "description": "One-sentence description of what will be trained (shown to user in confirm dialog)"},
        },
        "required": ["summary"],
        "needs_confirm": True,
    },
    {
        "name": "finalize_dataset",
        "description": (
            "Finalize the training dataset — create train/val/test splits. "
            "Must be done before training. A confirmation dialog will be shown first."
        ),
        "properties": {
            "summary": {"type": "string", "description": "One-sentence description shown in confirm dialog"},
            "val_frac": {"type": "number", "description": "Fraction of data for validation (0.0–0.4). Default 0.1 (10%)."},
            "test_frac": {"type": "number", "description": "Fraction of data for test set (0.0–0.4). Default 0.1 (10%)."},
        },
        "required": ["summary"],
        "needs_confirm": True,
    },
    {
        "name": "load_sam",
        "description": (
            "Load the SAM model. Uses the checkpoint and backend already selected in the SAM tab. "
            "Optionally override the backend. Loading takes ~30 seconds — inform the user."
        ),
        "properties": {
            "backend": {
                "type": "string",
                "enum": ["auto", "sam3", "sam2", "usam"],
                "description": "Override backend: auto=prefer SAM3, sam3=SAM3 only, sam2=SAM2 only, usam=micro-SAM. Leave unset to use whatever is configured.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "set_pixel_size",
        "description": (
            "Set the pixel size calibration for the current image. "
            "Use when the user provides a pixel size value or asks to calibrate the image scale."
        ),
        "properties": {
            "pixel_size_nm": {
                "type": "number",
                "description": "Pixel size in nm/px (nanometres per pixel)",
            },
        },
        "required": ["pixel_size_nm"],
        "needs_confirm": False,
    },
    {
        "name": "next_image",
        "description": "Navigate to the next image in the queue.",
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "prev_image",
        "description": "Navigate to the previous image in the queue.",
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "go_to_image",
        "description": "Jump to a specific image by its position number (1-based).",
        "properties": {
            "index": {"type": "integer", "description": "Image number, 1-based"},
        },
        "required": ["index"],
        "needs_confirm": False,
    },
    {
        "name": "load_yolo",
        "description": (
            "Load the YOLO model using the path already configured in the YOLO tab. "
            "Use when YOLO is not loaded and the user wants detection or segmentation."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "load_unet",
        "description": (
            "Load the UNet model using settings already configured in the UNet tab. "
            "Use when UNet is not loaded and the user wants segmentation."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "run_unet",
        "description": (
            "Run UNet instance segmentation on the current image. "
            "Produces precise masks. UNet must be loaded first."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label for segmented regions"},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "reject_annotations",
        "description": "Discard all pending (not yet accepted) annotations from a model.",
        "properties": {
            "model": {
                "type": "string",
                "enum": ["sam", "yolo", "unet", "all"],
                "description": "Which model's pending annotations to reject. Default 'all'.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "undo_annotation",
        "description": "Undo the last annotation added to the current image.",
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "clear_annotations",
        "description": "Clear ALL accepted annotations from the current image.",
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "set_contrast",
        "description": (
            "Change the contrast/display method for the current image. "
            "Use to improve visibility of features."
        ),
        "properties": {
            "method": {
                "type": "string",
                "enum": ["bandpass", "percentile", "sigma", "adaptive", "fourier"],
                "description": "bandpass=Gaussian bandpass (best for cryo-EM), percentile=clip by percentile, sigma=clip by std dev, adaptive=CLAHE, fourier=Fourier bandpass",
            },
            "low": {
                "type": "number",
                "description": "Low parameter: percentile low % / sigma count / bandpass low sigma px / fourier HP px",
            },
            "high": {
                "type": "number",
                "description": "High parameter: percentile high % / bandpass high sigma px / fourier LP px",
            },
        },
        "required": ["method"],
        "needs_confirm": False,
    },
    {
        "name": "pipe_yolo_to_sam",
        "description": (
            "Run YOLO detection to get bounding boxes, then feed those boxes into SAM "
            "to produce precise polygon masks. Requires both YOLO and SAM to be loaded."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label for segmented objects"},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "check_quality",
        "description": (
            "Assess the quality of the current image — checks blur, contrast, "
            "and saturation. Results appear in the Export tab status."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "run_surface_area",
        "description": (
            "Estimate 3D projected surface area of large annotated structures (cells, organelles, vesicles, liposomes). "
            "NOT for nanoparticle size/diameter — use run_particle_analysis for that. "
            "Requires ROI annotations to be present."
        ),
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of annotation labels to analyse (e.g. ['vesicle', 'particle']). Leave empty for all labels.",
            },
            "mode": {
                "type": "string",
                "enum": ["single", "batch"],
                "description": "single=current image only, batch=all loaded images with annotations.",
            },
            "method": {
                "type": "string",
                "enum": ["auto", "ellipsoid", "cauchy", "fourier", "fourier_spiky", "capsule", "perimeter"],
                "description": "auto=best per particle (recommended). ellipsoid=smooth round particles. cauchy=irregular convex. fourier=rough/fractal surfaces. capsule=rod-shaped.",
            },
            "compound_mode": {
                "type": "string",
                "enum": ["separate", "auto", "subtract_inner", "union"],
                "description": "How to handle multiple same-label polygons on one image. separate=treat individually. auto=subtract if inner/outer pair detected. subtract_inner=hollow particles (liposomes, donuts). union=aggregate particles.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "run_particle_analysis",
        "description": (
            "Measure 2D shape metrics (ECD/diameter, Feret length, area, perimeter, circularity, aspect ratio) "
            "for every annotated ROI. Use for nanoparticles, quantum dots, nanostructures, or any case where "
            "the user asks for size, diameter, Feret length, or particle measurements. "
            "Results appear in the Analysis tab → Particles table and Figures histogram. "
            "Requires ROI annotations to be present."
        ),
        "properties": {
            "labels": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Annotation labels to measure (e.g. ['particle', 'vesicle']). Empty = all labels.",
            },
            "mode": {
                "type": "string",
                "enum": ["single", "batch"],
                "description": "single=current image, batch=all loaded images.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "configure_analysis_plot",
        "description": (
            "Control the histogram in the Analysis → Figures tab. "
            "Set x-axis metric, y-axis type (count or density), and number of bins. "
            "Use when the user asks to change the plot, show raw counts, switch to density, "
            "change what's on the x-axis, or adjust the number of bins."
        ),
        "properties": {
            "metric": {
                "type": "string",
                "enum": ["ecd_nm", "feret_nm", "area_nm2", "perimeter_nm",
                         "circularity", "aspect_ratio", "bbox_w_nm", "bbox_h_nm"],
                "description": "Metric to plot on x-axis. ecd_nm=diameter, feret_nm=Feret length, area_nm2=area.",
            },
            "plot_type": {
                "type": "string",
                "enum": ["count", "density"],
                "description": "count=raw bar histogram, density=KDE normalised density.",
            },
            "n_bins": {
                "type": "integer",
                "description": "Number of histogram bins (5–200). Default 30.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "track_particles",
        "description": (
            "Track particles or objects across multiple image frames (time series or z-stack). "
            "Uses Hungarian algorithm with gap-closing. "
            "Requires at least 2 images loaded, each with ROI annotations."
        ),
        "properties": {
            "max_displacement_nm": {
                "type": "number",
                "description": "Maximum allowed displacement between frames (nm). Default 500.",
            },
            "min_frames": {
                "type": "integer",
                "description": "Minimum number of frames a track must span to be kept. Default 2.",
            },
            "max_gap": {
                "type": "integer",
                "description": "Maximum number of frames a particle can disappear and still be linked. Default 1.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "apply_contrast_preset",
        "description": (
            "Apply a saved contrast preset by name. "
            "Use when the user asks for a specific preset, or to quickly set a known-good contrast."
        ),
        "properties": {
            "preset_name": {
                "type": "string",
                "description": "Name of the preset to apply. Must match exactly from the available presets list in current state.",
            },
        },
        "required": ["preset_name"],
        "needs_confirm": False,
    },
    {
        "name": "export_masks",
        "description": (
            "Export all ROI annotations on the current image as PNG mask + JSON label file. "
            "Useful for creating ground truth masks for external tools."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "export_display_image",
        "description": (
            "Export the current image as an 8-bit contrast-normalised PNG, saved next to the source file. "
            "Useful for external annotation tools (iPad apps, etc) or sharing previews."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "push_to_hub",
        "description": (
            "Push the finalized training dataset to Hugging Face Hub. "
            "Requires the dataset to be finalized first. "
            "A confirmation dialog will appear."
        ),
        "properties": {
            "repo_id": {
                "type": "string",
                "description": "HuggingFace repo ID in the form 'username/dataset-name'.",
            },
            "token": {
                "type": "string",
                "description": "HuggingFace API token (optional if already configured in environment).",
            },
        },
        "required": ["repo_id"],
        "needs_confirm": True,
    },
    {
        "name": "import_star_file",
        "description": (
            "Import particle picks from a RELION .star file and add them as circular ROI annotations. "
            "Opens a file picker dialog. Use when the user mentions RELION, star file, or particle picks."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "configure_training",
        "description": (
            "Configure the Train tab before starting training. "
            "Use to set which model type to train (YOLO or UNet), the dataset directory, "
            "base model / architecture, and hyperparameters. "
            "Call this before start_training when the user asks to train a model and you "
            "need to select or confirm the right model type."
        ),
        "properties": {
            "model_type": {
                "type": "string",
                "enum": ["yolo", "unet"],
                "description": (
                    "yolo = YOLO instance segmentation (fast, best for distinct countable objects "
                    "like vesicles, nanoparticles, cells). "
                    "unet = UNet semantic segmentation (best for continuous structures, "
                    "membranes, background classification, or when instance boundaries overlap)."
                ),
            },
            "dataset_dir": {
                "type": "string",
                "description": "Path to the exported/finalized dataset directory.",
            },
            "epochs": {
                "type": "integer",
                "description": "Number of training epochs. Typical: 50–200 for YOLO, 30–100 for UNet.",
            },
            "batch": {
                "type": "integer",
                "description": "Batch size. Reduce if GPU OOM. Default 8.",
            },
            "yolo_base_model": {
                "type": "string",
                "description": "YOLO base weights: yolo11n-seg.pt (nano/fastest) to yolo11x-seg.pt (xlarge/most accurate).",
            },
            "unet_arch": {
                "type": "string",
                "enum": ["Unet", "UnetPlusPlus", "FPN", "DeepLabV3Plus", "MAnet", "PAN"],
                "description": "UNet architecture. Unet/UnetPlusPlus for small objects; FPN/DeepLabV3Plus for multi-scale.",
            },
            "unet_encoder": {
                "type": "string",
                "description": "UNet encoder backbone. resnet34 (fast baseline), efficientnet-b3 (accurate), mit_b2 (transformer).",
            },
        },
        "required": ["model_type"],
        "needs_confirm": False,
    },
    {
        "name": "batch_run_sam",
        "description": (
            "Run SAM auto-segmentation on ALL loaded images sequentially: segment → accept → queue for export. "
            "This is the complete one-shot annotation pipeline for the whole dataset. "
            "Use when the user wants to process, annotate, or prep all images at once. "
            "SAM must be loaded first. Images already annotated are skipped by default."
        ),
        "properties": {
            "label": {"type": "string", "description": "Annotation label for all found objects (e.g. 'vesicle', 'particle')"},
            "points_per_side": {"type": "integer", "description": "SAM grid density 4–128. Default 32."},
            "skip_annotated": {"type": "boolean", "description": "Skip images that already have annotations. Default true."},
        },
        "required": ["label"],
        "needs_confirm": False,
    },
    {
        "name": "add_scalebar",
        "description": (
            "Add a scale bar to the current image at the bottom-left corner. "
            "Size is auto-calculated from the pixel size. Use after setting pixel_size_nm."
        ),
        "properties": {
            "color": {"type": "string", "description": "Scale bar color as hex (e.g. '#FFFFFF') or CSS name. Default white."},
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "rename_label",
        "description": (
            "Rename annotation labels on the current image — replace every occurrence "
            "of old_label with new_label. Useful for correcting typos or reclassifying annotations."
        ),
        "properties": {
            "old_label": {"type": "string", "description": "Label to replace"},
            "new_label": {"type": "string", "description": "New label name"},
        },
        "required": ["old_label", "new_label"],
        "needs_confirm": False,
    },
    {
        "name": "save_contrast_preset",
        "description": (
            "Save the current contrast settings as a named preset for future use. "
            "After saving, the preset appears in the contrast panel dropdown and can be recalled by name."
        ),
        "properties": {
            "name": {"type": "string", "description": "Name for the new preset (must not duplicate a built-in preset name)"},
        },
        "required": ["name"],
        "needs_confirm": False,
    },
    {
        "name": "export_measurements",
        "description": (
            "Export measurements from ALL loaded images into a single summary CSV. "
            "Saved as acorn_measurements/measurements.csv inside the same folder the images were opened from. "
            "Overwrites the file each time so it always reflects the full current dataset. "
            "Includes image name, pixel size, ROI areas, distances, and label counts. "
            "Results are also shown in the Analysis tab."
        ),
        "properties": {},
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "compress_frames",
        "description": (
            "Average the frames of the currently loaded movie/multi-frame file to produce "
            "a single high-SNR image. Three methods are available: "
            "'mean' — simple average (fast, good for visual inspection); "
            "'motion_corrected' — phase cross-correlation alignment before averaging "
            "(corrects beam-induced motion, ~30s for large frames); "
            "'dose_weighted' — Fourier-space dose filter using the Grant & Grigorieff (2015) "
            "critical exposure formula, attenuating high-frequency content at high dose. "
            "Use mean for a quick look; motion_corrected for annotation-quality data; "
            "dose_weighted when the user asks for dose filtering or asks to match cryoSPARC output."
        ),
        "properties": {
            "method": {
                "type": "string",
                "enum": ["mean", "motion_corrected", "dose_weighted"],
                "description": "mean (default) | motion_corrected | dose_weighted",
            },
            "dose_per_frame": {
                "type": "number",
                "description": (
                    "Electron dose per frame in e/A² (used only for dose_weighted). "
                    "Typical cryo-EM: 1–3 e/A²/frame. Default 1.0."
                ),
            },
            "start_frame": {
                "type": "integer",
                "description": (
                    "First frame to include (1-indexed). Default 1. "
                    "Skip early frames to avoid high-motion or beam-induced damage at the start of the movie. "
                    "Typical: skip first 2–5 frames (set start_frame=3) for K2/K3 acquisitions."
                ),
            },
            "end_frame": {
                "type": "integer",
                "description": (
                    "Last frame to include (1-indexed, inclusive). Default 0 = all remaining frames. "
                    "Reduce to limit cumulative dose — e.g. end_frame=20 uses only the first 20 frames "
                    "to maximise high-resolution signal before critical dose is exceeded."
                ),
            },
        },
        "required": [],
        "needs_confirm": False,
    },
    {
        "name": "dose_comparison",
        "description": (
            "Open the Dose Series dialog, which splits the current movie into equal-dose bins "
            "and shows per-bin averaged images alongside difference images (bin N − bin 1) "
            "to visualise dose-dependent structural changes — e.g. membrane recession, "
            "particle damage, bubble formation, or contrast evolution under the beam. "
            "Use when the user asks about structural changes with dose, beam damage, "
            "how something looks at different doses, or wants to compare early vs late frames."
        ),
        "properties": {
            "n_bins": {
                "type": "integer",
                "description": "Number of equal-dose bins to divide the movie into. Default 4.",
            },
            "dose_per_frame": {
                "type": "number",
                "description": "Electron dose per frame in e/Å². Used to label cumulative dose on each bin.",
            },
        },
        "required": [],
        "needs_confirm": False,
    },
]

_NEEDS_CONFIRM = {t["name"] for t in _TOOLS if t["needs_confirm"]}


def _to_anthropic_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        out.append({
            "name": t["name"],
            "description": t["description"],
            "input_schema": {
                "type": "object",
                "properties": t.get("properties", {}),
                "required": t.get("required", []),
            },
        })
    return out


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": {
                    "type": "object",
                    "properties": t.get("properties", {}),
                    "required": t.get("required", []),
                },
            },
        })
    return out


def build_system_prompt(state: dict) -> str:
    img_info = "No image loaded"
    if state.get("image_name"):
        shape = state.get("image_shape") or []
        px = state.get("pixel_size_nm", 0)
        img_info = state["image_name"]
        if len(shape) >= 2:
            img_info += f" ({shape[-1]}×{shape[-2]} px)"
        if px and px > 0:
            img_info += f", {px:.4f} nm/px"

    loaded = [m for m, k in [("SAM", "sam_loaded"), ("YOLO", "yolo_loaded"), ("UNet", "unet_loaded")] if state.get(k)]
    model_info = ", ".join(loaded) if loaded else "none loaded"

    # Model configuration (what's selected but not necessarily loaded)
    sam_cfg = ""
    if state.get("sam_backend"):
        sam_cfg = f"backend={state['sam_backend']}, checkpoint={state.get('sam_checkpoint', 'auto')}"
        ckpts = state.get("sam_checkpoints_available", [])
        if len(ckpts) > 1:
            sam_cfg += f" (available: {', '.join(ckpts[:4])})"
    yolo_cfg  = state.get("yolo_model_path", "") or "not configured"
    unet_cfg  = f"{state.get('unet_arch','')} / {state.get('unet_encoder','')} / {state.get('unet_ckpt','')}" if state.get("unet_arch") else "not configured"

    pending = []
    if state.get("pending_sam"):  pending.append(f"SAM: {state['pending_sam']}")
    if state.get("pending_yolo"): pending.append(f"YOLO: {state['pending_yolo']}")
    if state.get("pending_unet"): pending.append(f"UNet: {state['pending_unet']}")
    pending_info = ", ".join(pending) if pending else "none"
    contrast_info = state.get("contrast_method", "unknown")
    presets_list  = state.get("contrast_presets") or []
    presets_info  = ", ".join(f'"{p}"' for p in presets_list) if presets_list else "none"
    sam_detail = ""
    if state.get("sam_points_per_side") is not None:
        sam_detail = (
            f" [points_per_side={state['sam_points_per_side']}, "
            f"iou_thresh={state.get('sam_iou_thresh','?')}, "
            f"stability={state.get('sam_stability_thresh','?')}]"
        )

    ann = state.get("annotation_labels") or {}
    ann_types = state.get("annotation_types") or {}
    ann_info = f"{state.get('annotation_count', 0)} annotations"
    if ann_types:
        ann_info += " by type: " + ", ".join(f"{k}={v}" for k, v in ann_types.items())
    if ann:
        ann_info += " | by label: " + ", ".join(f"{k}: {v}" for k, v in ann.items())

    measurements_lines = []
    for d in state.get("distance_measurements") or []:
        cal = "calibrated" if d["calibrated"] else "pixels only"
        measurements_lines.append(f"  {d['distance_nm']} nm  ({d['distance_px']} px)  [{cal}]")
    for r in state.get("roi_areas") or []:
        measurements_lines.append(f"  {r['label']} area: {r['area_nm2']} nm²")
    meas_info = "\n".join(measurements_lines) if measurements_lines else "none"

    # Export / dataset state
    export_dir        = state.get("export_dataset_dir", "") or "not set"
    export_queue      = state.get("export_queue_count", 0)
    val_frac          = state.get("export_val_frac", 0.1)
    test_frac         = state.get("export_test_frac", 0.1)
    train_pct         = int(round((1.0 - val_frac - test_frac) * 100))
    val_pct           = int(round(val_frac * 100))
    test_pct          = int(round(test_frac * 100))
    finalized         = state.get("dataset_finalized", False)
    split_status      = "finalized (splits exist)" if finalized else "not yet finalized"

    # Training configuration
    train_type = state.get("train_model_type", "")
    train_dir  = state.get("train_dataset_dir", "") or "not set"
    if train_type == "yolo":
        train_cfg = (
            f"YOLO, base={state.get('train_yolo_base','?')}, "
            f"epochs={state.get('train_epochs','?')}, batch={state.get('train_batch','?')}, "
            f"imgsz={state.get('train_yolo_imgsz','?')}"
        )
    elif train_type == "unet":
        train_cfg = (
            f"UNet ({state.get('train_unet_arch','?')} / {state.get('train_unet_encoder','?')}), "
            f"epochs={state.get('train_epochs','?')}, batch={state.get('train_batch','?')}, "
            f"imgsz={state.get('train_unet_imgsz','?')}"
        )
    else:
        train_cfg = "not configured"

    n_imgs = state.get("image_count", 0)
    idx = state.get("current_image_index", -1)

    # Full image list (compact — one line per image; priority images shown first)
    image_list = state.get("image_list") or []
    if image_list:
        def _img_line(entry):
            i       = entry["index"]
            fname   = entry["filename"]
            n_ann   = entry["annotation_count"]
            lbl_str = ", ".join(f"{k}:{v}" for k, v in entry.get("label_counts", {}).items()) or "none"
            px      = entry.get("pixel_size_nm", 0)
            queued  = "queued" if entry.get("in_export_queue") else ""
            current = " <-- CURRENT" if i == idx else ""
            return (
                f"  [{i+1:3d}] {fname}  |  {n_ann} ann ({lbl_str})"
                + (f"  |  {px:.4f}nm/px" if px and px > 0 else "")
                + (f"  |  {queued}" if queued else "")
                + current
            )

        _MAX_LINES = 60
        priority = [
            e for e in image_list
            if e["annotation_count"] > 0 or e.get("in_export_queue") or e["index"] == idx
        ]
        unannotated = [
            e for e in image_list
            if e not in priority
        ]
        if len(image_list) <= _MAX_LINES:
            shown = image_list
            omitted = 0
        else:
            shown = (priority + unannotated)[: _MAX_LINES]
            omitted = len(image_list) - len(shown)

        img_lines = [_img_line(e) for e in shown]
        if omitted:
            img_lines.append(
                f"  ... {omitted} additional unannotated image(s) not shown"
                f" (use navigate_to_image by index to reach them)"
            )
        image_list_section = "## Loaded images\n" + "\n".join(img_lines)
    else:
        image_list_section = ""

    # Dataset stats from finalized dataset
    ds_stats = state.get("dataset_stats") or {}
    if ds_stats:
        ds_stats_lines = []
        for k, v in ds_stats.items():
            ds_stats_lines.append(f"  {k}: {v}")
        ds_stats_section = "Dataset stats (finalized):\n" + "\n".join(ds_stats_lines)
    else:
        ds_stats_section = ""

    return f"""You are CLU, the AI assistant built into ACORN — a cryo-EM and electron microscopy image analysis platform at Oak Ridge National Laboratory (ORNL). Your name is CLU. If asked who you are, say you are CLU, ACORN's microscopy analysis assistant. Never bold or format your own name — write it as plain text: CLU, not **CLU**. You think and communicate like a senior microscopist and data scientist.

## Current application state
- Image: {img_info}{f"  [MOVIE: {state['n_frames']} frames — call compress_frames to average]" if state.get('is_movie') else ""}
- Queue: {n_imgs} images  (viewing #{idx + 1})
- Models loaded: {model_info}
- SAM config: {sam_cfg}{sam_detail}
- YOLO config: {yolo_cfg}
- UNet config: {unet_cfg}
- Pending (unaccepted) annotations: {pending_info}
- Contrast method: {contrast_info}  |  Available presets: {presets_info}
- Accepted annotations: {ann_info}
- Measurements: {meas_info}
- Export dataset dir: {export_dir}  ({split_status})
- Export queue: {export_queue} images pending export ({", ".join(state.get("export_queue_filenames") or []) or "none"})
- Split ratios configured: {train_pct}% train / {val_pct}% val / {test_pct}% test
- Train tab: {train_cfg}, dataset: {train_dir}
- Dataset-wide: {state.get("dataset_total_annotations", 0)} total annotations across {state.get("dataset_images_annotated", 0)}/{n_imgs} images  |  labels: {", ".join(f"{k}:{v}" for k, v in (state.get("dataset_label_counts") or {}).items()) or "none"}
{ds_stats_section}
{image_list_section}

## Your role
Your purpose is to help researchers with their microscopy analysis and model training needs. You annotate, segment, analyze images, run surface area and tracking analysis, manage training datasets, and guide users through the full pipeline — all through natural language. You understand scientific intent — "find the vesicles", "how dense is the sample?", "prep this for training", "the contrast looks off" — and translate it into the right sequence of actions without the user needing to know tool names.

You are proactive: if a prerequisite is missing (model not loaded, no image open), handle it in the same response rather than just reporting the problem. Chain multiple tools when a request implies a full workflow.

## Scientific domain knowledge
- **CryoEM**: Images are typically 2048–4096 px, low signal-to-noise, Gaussian bandpass contrast is best. Structures: vesicles (50–500 nm), liposomes, membranes, particles, lamella, ice contamination.
- **SEM/TEM**: Higher contrast, sharper edges. Percentile or sigma contrast works well. Structures: nanoparticles, bacteria, cells, pores, surfaces.
- **Pixel size matters**: Always note nm/px when reporting measurements. If not calibrated, note that measurements are in pixels only.
- **SAM** (Segment Anything Model): Best for precise polygon masks of arbitrary shapes. Use `run_sam_auto` to find everything, or prompt with points/boxes. SAM 3 > SAM 2 > micro-SAM for general use.
- **YOLO**: Fast detection/segmentation for objects it was trained on. `pipe_yolo_to_sam` gives YOLO's speed with SAM's precision.
- **UNet**: Semantic or instance segmentation for specific trained classes.
- **Bandpass contrast**: Removes low-frequency background and high-frequency noise — ideal for cryo-EM. Increase `bp_low_sigma` (e.g. 200) to suppress larger background variations.
- **Dose series analysis**: Splitting a movie into equal-dose bins and comparing averages reveals dose-dependent changes — membrane recession, bubble formation, organic contrast loss, or electrode delamination in materials science. The difference image (bin N − bin 1) highlights WHERE change occurred: blue = signal decreased (material receded/dissolved), red = signal increased (contamination, swelling, or charging). More bins = finer dose resolution but noisier images per bin.
- **Movie frame selection**: K2/K3 cameras collect 20–1000+ frames per exposure. Early frames have the most beam-induced motion; later frames accumulate more dose. Useful strategies: skip first 2–5 frames (start_frame=3) to avoid motion blur; cap at a dose-limited subset (end_frame=20–40 for typical 1–2 e/A²/frame) to preserve high-resolution signal; or use all frames for maximum SNR on low-resolution structures.
- **YOLO vs UNet for training**:
  - Choose **YOLO** when: objects are distinct, countable, and separable (vesicles, liposomes, nanoparticles, cells, bacteria); you want instance masks with individual IDs; dataset has varied backgrounds.
  - Choose **UNet** when: structures are continuous or overlapping (membranes, filaments, ice contamination); you want pixel-level class maps rather than instance objects; or you need a custom architecture.
  - Default recommendation: YOLO for most cryo-EM particle and organelle tasks. UNet for membrane/surface segmentation.

## Reasoning guidelines

### Act first, report once
- **For any multi-step workflow, execute all tools in sequence first. Give ONE summary message at the end. Do not narrate between steps, do not ask permission mid-workflow, do not pause to say "I'll now do X" before doing X.**
- If a prerequisite is missing (model not loaded, no image), handle it silently as part of the same response.
- Only ask a clarifying question if the user's intent is genuinely ambiguous and the wrong choice would be destructive. Otherwise, make a reasonable assumption and proceed.
- Do not ask "should I keep existing annotations?" — if the task implies segmenting, just do it. Only clear annotations if the user explicitly says to.

### Multi-image / batch work
- When the user says "do all", "whole folder", "all images", "process everything", or any instruction implying the full dataset: use `batch_run_sam` (not `next_image` loops) then chain with `run_particle_analysis(mode=batch)` and `export_measurements`. Execute everything without stopping for confirmation.
- **Multi-image pixel size**: Images not yet navigated to may show 1.0 nm/px (default). Navigate to each before reporting per-image pixel sizes. For batch operations, pixel size is read automatically on each load.

### Measurement tool selection
- **Nanoparticles, quantum dots, small discrete objects** → `run_particle_analysis` (ECD/diameter, Feret length, area, circularity). This is almost always the right choice.
- **3D surface area of large hollow objects (vesicles, liposomes, cells)** → `run_surface_area`. Only use this when the user explicitly asks for surface area.
- When unsure, use `run_particle_analysis`.

### Scientific reporting
- After measurements complete, always report: n, mean ± std, median, min–max for the primary metric (ECD or Feret). Pull values from shape_measurements or roi_areas in state.
- Then set the plot to count histogram by default: configure_analysis_plot(plot_type="count", metric="feret_nm") or metric="ecd_nm" as appropriate.
- If pixel size is not calibrated, note measurements are in pixels only.
- Flag scientifically odd results (FFT image loaded by mistake, pixel size looks wrong, all particles same size = segmentation artefact).

### Contrast / visibility
- **"I can't see anything" on a movie**: compress_frames(method="mean") → set_contrast(method="bandpass")
- **"I can't see anything" on a still image**: reason about type (cryo-EM → bandpass; SEM/TEM → percentile) → set_contrast. Don't ask, act.
- **Never assume the user knows microscopy jargon**. One plain-language sentence when doing something technical.

### Training pipeline
- Before training: dataset must be finalized. If not, call finalize_dataset first.
- Export dataset dir and train dataset dir must be the same path.

## Common multi-step workflows
**"Find / segment X"**: load_sam if needed → run_sam_auto(label=X) → accept_annotations → report count
**"Detect X"**: load_yolo if needed → run_yolo_detect(label=X) → accept_annotations → report count
**"High-quality segmentation of X"**: load_yolo + load_sam → pipe_yolo_to_sam(label=X) → accept_annotations
**"Measure / analyze X"**: ensure annotations exist (segment first if not) → run_particle_analysis(labels=[X], mode=batch) → export_measurements
**"Prep for training"**: load_sam if needed → batch_run_sam(label=X, skip_annotated=true) — one call handles all images
**Full training pipeline**:
  1. Annotate each image: segment → accept_annotations → queue_for_export → next_image → repeat
  2. When all images queued: finalize_dataset(val_frac=0.1, test_frac=0.1) — creates 80/10/10 splits
  3. configure_training(model_type=yolo/unet, dataset_dir=<export_dataset_dir>, epochs=100)
  4. start_training(summary=...)
  Note: The export dataset dir and the train dataset dir must be the same path.
  If the dataset is already finalized (splits exist), skip step 2.
**"Fix the contrast" / "I can't see anything" / "image is blank/dark/washed out"**: reason about image type → if cryo-EM or movie: set_contrast(method="bandpass") → if SEM/TEM: set_contrast(method="percentile", low=0.5, high=99.5)
**"I can't see these movies" / "fix contrast on movie" / any visibility complaint on a movie**: compress_frames(method="mean") → set_contrast(method="bandpass") → explain in plain language
**"How many X are there?"**: read annotation_labels from state — no tool needed
**"What are the sizes?"**: read roi_areas from state — no tool needed
**"Go to image N"**: go_to_image(N)
**"Load SAM"**: load_sam → inform user loading takes ~30s
**"Nanoparticle / particle size (diameter, Feret, ECD)"**: ensure ROI annotations exist → run_particle_analysis(labels=[...], mode=single/batch) — gives ECD, Feret length, area, circularity in Analysis tab
**"Show raw count histogram"**: configure_analysis_plot(plot_type="count")
**"Show density / distribution"**: configure_analysis_plot(plot_type="density")
**"Plot Feret length / diameter / area / circularity"**: configure_analysis_plot(metric="feret_nm") etc.
**"Change bins / more bins / fewer bins"**: configure_analysis_plot(n_bins=N)
**"What are the stats / mean / std / distribution"**: read shape_measurements or roi_areas from state — report mean, median, std, min, max, n. No tool needed.
**"Surface area of large objects (vesicles, cells, organelles)"**: ensure ROI annotations exist → run_surface_area(labels=[...], mode=single/batch, method=auto)
**"Track particles"**: ensure multiple images with annotations → track_particles(max_displacement_nm=500)
**"Apply [preset name]"**: apply_contrast_preset(preset_name=...) — use exact name from available presets list
**"Export masks / ground truth"**: export_masks()
**"Share / export for annotation"**: export_display_image()
**"Push to HuggingFace"**: push_to_hub(repo_id=...) — dataset must be finalized first
**"Import RELION picks / star file"**: import_star_file()
**"Dose series" / "compare bins" / "beam damage" / "membrane recession" / "how does it look at different doses"**: dose_comparison(n_bins=4, dose_per_frame=1.0) — opens dialog showing averaged bins + difference images vs first bin; blue = signal lost, red = signal gained
**"Compress / average movie frames"**: compress_frames(method=mean/motion_corrected/dose_weighted, start_frame=1, end_frame=0) — use mean for quick look, motion_corrected for annotation, dose_weighted when the user mentions dose filtering or cryoSPARC-style output
**"Try different frame ranges"** / "play with compression" / "use only early frames" / "skip first N frames": adjust start_frame and end_frame — e.g. skip early high-motion frames (start_frame=3), limit cumulative dose (end_frame=20), or compare subsets
**"Annotate / process all images at once"**: load_sam if needed → batch_run_sam(label=..., skip_annotated=true) — runs SAM on every image automatically, no per-image confirmation needed. ALWAYS use this instead of next_image loops when user says "do all", "whole folder", "all images", or "process everything".
**Large-dataset two-phase workflow** (recommended when dataset has 20+ images):
  Phase 1 — Build training set from a representative sample:
    1. Annotate 5–15 diverse images (SAM or manual) → accept_annotations → queue_for_export each
    2. finalize_dataset → configure_training → start_training
  Phase 2 — Scale to full dataset using the trained model:
    3. After training: load_yolo (or load_unet) with the new checkpoint
    4. Run batch detection on remaining images: for each un-annotated image → go_to_image → run_yolo_detect / run_yolo_segment / run_unet → accept_annotations → queue_for_export
    5. Re-finalize dataset (adds new images to splits) → optionally re-train for improvement
  This approach avoids annotating all images by hand — annotate a sample, train, run inference on the rest.
**"Add a scale bar"**: add_scalebar() — auto-sizes to pixel size; set pixel size first if needed
**"Rename label X to Y"**: rename_label(old_label=X, new_label=Y) — fixes typos or reclassifies annotations on current image
**"Save this contrast as a preset"**: save_contrast_preset(name=...) — saves current settings to the preset dropdown
**"Export measurements / data"**: export_measurements() — writes ALL loaded images' measurements to acorn_measurements/measurements.csv in the same folder as the images; shown in Analysis tab
**"Which images have annotations?"**: read image_list from state — no tool needed
**"Which images are queued?"**: read export_queue_filenames or image_list.in_export_queue from state — no tool needed
**"How many vesicles across all images?"**: read dataset_label_counts from state — no tool needed
**"Train a model"**: assess target structures → configure_training(model_type=yolo/unet, ...) → start_training(summary=...)
**"Train YOLO on X"**: configure_training(model_type=yolo, epochs=100) → start_training
**"Train UNet for membranes"**: configure_training(model_type=unet, unet_arch=Unet, unet_encoder=resnet34) → start_training

## Tool calling format (Ollama/local models only)
To call a tool, output ONLY a JSON object on its own line:
{{"name": "run_sam_auto", "label": "vesicle"}}
{{"name": "run_yolo_detect", "label": "particle"}}
{{"name": "run_yolo_segment", "label": "membrane"}}
{{"name": "run_unet", "label": "membrane"}}
{{"name": "pipe_yolo_to_sam", "label": "vesicle"}}
{{"name": "accept_annotations", "model": "all"}}
{{"name": "reject_annotations", "model": "all"}}
{{"name": "undo_annotation"}}
{{"name": "clear_annotations"}}
{{"name": "load_sam"}}
{{"name": "load_sam", "backend": "sam3"}}
{{"name": "load_yolo"}}
{{"name": "load_unet"}}
{{"name": "set_contrast", "method": "bandpass"}}
{{"name": "set_contrast", "method": "percentile", "low": 0.5, "high": 99.5}}
{{"name": "set_contrast", "method": "sigma", "low": 3}}
{{"name": "next_image"}}
{{"name": "prev_image"}}
{{"name": "go_to_image", "index": 3}}
{{"name": "set_pixel_size", "pixel_size_nm": 0.123}}
{{"name": "check_quality"}}
{{"name": "queue_for_export"}}
{{"name": "start_training", "summary": "Train YOLO on vesicle dataset, 100 epochs"}}
{{"name": "finalize_dataset", "summary": "Create 80/10/10 train/val/test splits"}}
{{"name": "configure_training", "model_type": "yolo", "epochs": 100, "yolo_base_model": "yolo11m-seg.pt"}}
{{"name": "configure_training", "model_type": "unet", "unet_arch": "Unet", "unet_encoder": "resnet34", "epochs": 50}}
{{"name": "run_surface_area", "labels": ["vesicle"], "mode": "batch", "method": "auto"}}
{{"name": "run_surface_area", "labels": ["membrane"], "mode": "single", "method": "fourier", "compound_mode": "subtract_inner"}}
{{"name": "track_particles", "max_displacement_nm": 500, "min_frames": 2, "max_gap": 1}}
{{"name": "apply_contrast_preset", "preset_name": "Default (Bandpass)"}}
{{"name": "apply_contrast_preset", "preset_name": "Percentile 0.5/99.5 (standard annotation for low dose)"}}
{{"name": "export_masks"}}
{{"name": "export_display_image"}}
{{"name": "push_to_hub", "repo_id": "username/my-dataset"}}
{{"name": "import_star_file"}}
{{"name": "dose_comparison", "n_bins": 4}}
{{"name": "dose_comparison", "n_bins": 8, "dose_per_frame": 1.5}}
{{"name": "compress_frames", "method": "mean"}}
{{"name": "compress_frames", "method": "mean", "start_frame": 3}}
{{"name": "compress_frames", "method": "mean", "start_frame": 1, "end_frame": 20}}
{{"name": "compress_frames", "method": "motion_corrected", "start_frame": 3, "end_frame": 50}}
{{"name": "compress_frames", "method": "dose_weighted", "dose_per_frame": 1.5}}
{{"name": "batch_run_sam", "label": "vesicle", "skip_annotated": true}}
{{"name": "batch_run_sam", "label": "particle", "points_per_side": 64, "skip_annotated": false}}
{{"name": "add_scalebar"}}
{{"name": "add_scalebar", "color": "#FFFF00"}}
{{"name": "rename_label", "old_label": "vesicle", "new_label": "liposome"}}
{{"name": "save_contrast_preset", "name": "My Cryo Settings"}}
{{"name": "export_measurements"}}
For conversation, respond in plain text — no JSON."""


# ---------------------------------------------------------------------------
# Text-based tool call parser (for Ollama/local models that output JSON as text)
# ---------------------------------------------------------------------------

_KNOWN_TOOLS = {t["name"] for t in _TOOLS}


def _parse_text_tool_calls(text: str) -> list[tuple[str, dict]]:
    """Extract tool calls from model text output.

    Looks for JSON objects (possibly in code fences) containing a "name" key
    that matches a known tool name.
    """
    import re
    results = []

    # Strip code fences if present
    stripped = re.sub(r"```(?:json)?\s*|\s*```", " ", text)

    # Find all {...} blocks
    for m in re.finditer(r"\{[^{}]*\}", stripped):
        try:
            obj = json.loads(m.group())
        except json.JSONDecodeError:
            continue
        name = obj.get("name") or obj.get("tool") or obj.get("function")
        if not name or name not in _KNOWN_TOOLS:
            continue
        # Params can be under "parameters", "arguments", "input", or top-level keys
        params = (
            obj.get("parameters")
            or obj.get("arguments")
            or obj.get("input")
            or {k: v for k, v in obj.items() if k not in ("name", "tool", "function")}
        )
        if not isinstance(params, dict):
            params = {}
        results.append((name, params))

    return results


# ---------------------------------------------------------------------------
# Agent thread
# ---------------------------------------------------------------------------

class LLMAgent(QThread):
    """Runs the LLM call + tool-use loop in a background thread."""

    token_emitted   = pyqtSignal(str)        # streaming text chunk
    tool_called     = pyqtSignal(str, dict)  # (tool_name, params) — immediate action
    confirm_needed  = pyqtSignal(str, str, dict)  # (tool_name, summary, params) — needs confirm
    done            = pyqtSignal()
    error           = pyqtSignal(str)

    def __init__(
        self,
        config: LLMConfig,
        messages: list[dict],
        state: dict,
        image_b64: Optional[str] = None,
        context=None,
    ) -> None:
        super().__init__()
        self._config    = config
        self._messages  = list(messages)
        self._state     = state
        self._image_b64 = image_b64
        self._context   = context  # AcornContext — used to refresh state after navigation

    # ------------------------------------------------------------------

    def run(self) -> None:
        try:
            print(f"[LLMAgent] provider={self._config.provider} model={self._config.model}", flush=True)
            if self._config.provider == "anthropic":
                self._run_anthropic()
            else:
                self._run_openai()
            print("[LLMAgent] done", flush=True)
        except Exception as exc:
            print(f"[LLMAgent] error: {exc}", flush=True)
            self.error.emit(str(exc))

    # ------------------------------------------------------------------
    # Anthropic path
    # ------------------------------------------------------------------

    def _run_anthropic(self) -> None:
        try:
            import anthropic
        except ImportError:
            self.error.emit("anthropic package not installed. Run: uv pip install anthropic")
            return

        client = anthropic.Anthropic(api_key=self._config.api_key or None)
        system = build_system_prompt(self._state)
        tools  = _to_anthropic_tools(_TOOLS)
        msgs   = self._build_anthropic_messages()

        while True:
            thinking_cfg = {"type": "adaptive"} if "claude" in self._config.model.lower() else None

            kwargs: dict = dict(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                system=system,
                tools=tools,
                messages=msgs,
            )
            if thinking_cfg:
                kwargs["thinking"] = thinking_cfg

            with client.messages.stream(**kwargs) as stream:
                for event in stream:
                    if (
                        hasattr(event, "type")
                        and event.type == "content_block_delta"
                        and hasattr(event, "delta")
                        and getattr(event.delta, "type", None) == "text_delta"
                    ):
                        self.token_emitted.emit(event.delta.text)
                response = stream.get_final_message()

            msgs.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                name   = block.name
                params = dict(block.input) if block.input else {}
                result = self._dispatch_tool(name, params)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            msgs.append({"role": "user", "content": tool_results})

        self.done.emit()

    def _build_anthropic_messages(self) -> list[dict]:
        """Convert internal message list; inject image into first user message."""
        out = []
        image_injected = False
        for msg in self._messages:
            if msg["role"] == "user" and not image_injected and self._image_b64:
                content: list = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._image_b64,
                        },
                    },
                    {"type": "text", "text": msg["content"]},
                ]
                out.append({"role": "user", "content": content})
                image_injected = True
            else:
                out.append(msg)
        return out

    # ------------------------------------------------------------------
    # OpenAI-compatible path (OpenAI, Ollama, Groq, Together, etc.)
    # ------------------------------------------------------------------

    def _run_openai(self) -> None:
        try:
            from openai import OpenAI, BadRequestError
        except ImportError:
            self.error.emit("openai package not installed. Run: uv pip install openai")
            return

        kwargs: dict = {"api_key": self._config.api_key or "ollama"}
        if self._config.base_url:
            kwargs["base_url"] = self._config.base_url

        client = OpenAI(**kwargs)
        system = build_system_prompt(self._state)
        tools  = _to_openai_tools(_TOOLS)
        msgs: list[dict] = [{"role": "system", "content": system}]
        msgs  += self._build_openai_messages()
        # use_api_tools: True = OpenAI function-calling format; False = text JSON fallback (Ollama)
        use_api_tools = True
        # newer OpenAI models (gpt-5.5+) use max_completion_tokens instead of max_tokens
        tokens_key = "max_tokens"

        while True:
            create_kwargs: dict = dict(
                model=self._config.model,
                messages=msgs,
                stream=True,
            )
            create_kwargs[tokens_key] = self._config.max_tokens
            if use_api_tools:
                create_kwargs["tools"] = tools

            try:
                stream = client.chat.completions.create(**create_kwargs)
            except BadRequestError as exc:
                err = str(exc).lower()
                if "max_tokens" in err and "max_completion_tokens" in err:
                    tokens_key = "max_completion_tokens"
                    continue
                if use_api_tools and "tools" in err:
                    use_api_tools = False
                    continue
                raise

            collected_text     = ""
            tool_calls_acc: dict[int, dict] = {}
            finish = None

            for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue
                delta = choice.delta
                if delta.content:
                    self.token_emitted.emit(delta.content)
                    collected_text += delta.content
                if use_api_tools and delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index
                        if idx not in tool_calls_acc:
                            tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                        if tc.id:
                            tool_calls_acc[idx]["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                tool_calls_acc[idx]["name"] = tc.function.name
                            if tc.function.arguments:
                                tool_calls_acc[idx]["arguments"] += tc.function.arguments
                finish = choice.finish_reason

            if use_api_tools:
                # Filter malformed / unrecognised tool calls
                known = {t["name"] for t in _TOOLS}
                tool_calls_acc = {
                    k: v for k, v in tool_calls_acc.items()
                    if v.get("name") and v["name"] != "None" and v["name"] in known
                }

                print(f"[LLMAgent] finish_reason={finish!r} tool_calls={list(tool_calls_acc.keys())} text={repr(collected_text[:120])}", flush=True)

                if tool_calls_acc:
                    # Build assistant message with tool_calls list
                    tc_list = []
                    for idx in sorted(tool_calls_acc):
                        tc = tool_calls_acc[idx]
                        tc_list.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": tc["arguments"]},
                        })
                    msgs.append({"role": "assistant", "content": collected_text or None, "tool_calls": tc_list})

                    for tc in tc_list:
                        name = tc["function"]["name"]
                        try:
                            params = json.loads(tc["function"]["arguments"] or "{}")
                        except json.JSONDecodeError:
                            params = {}
                        print(f"[LLMAgent] dispatching {name} params={params}", flush=True)
                        result = self._dispatch_tool(name, params)
                        msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                    continue  # loop for follow-up response

                # No API tool calls — check if model output text JSON (Ollama-style fallback)
                if not tool_calls_acc:
                    text_tools = _parse_text_tool_calls(collected_text)
                    if text_tools:
                        # Model is outputting JSON as text (Ollama) — switch to text mode
                        use_api_tools = False
                        msgs.append({"role": "assistant", "content": collected_text})
                        results = [f"[{n}]: {self._dispatch_tool(n, p)}" for n, p in text_tools]
                        msgs.append({"role": "user", "content": "\n".join(results)})
                        continue

                msgs.append({"role": "assistant", "content": collected_text})
                break

            else:
                # Text-based tool call parsing (Ollama fallback)
                msgs.append({"role": "assistant", "content": collected_text})
                text_tools = _parse_text_tool_calls(collected_text)
                if not text_tools:
                    break
                results = [f"[{n}]: {self._dispatch_tool(n, p)}" for n, p in text_tools]
                msgs.append({"role": "user", "content": "\n".join(results)})

        self.done.emit()

    def _build_openai_messages(self) -> list[dict]:
        out = []
        image_injected = False
        for msg in self._messages:
            if msg["role"] == "user" and not image_injected and self._image_b64:
                content: list = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{self._image_b64}"},
                    },
                    {"type": "text", "text": msg["content"]},
                ]
                out.append({"role": "user", "content": content})
                image_injected = True
            else:
                out.append(msg)
        return out

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    def _dispatch_tool(self, name: str, params: dict) -> str:
        # Check prerequisites before dispatching
        needs_image = name not in (
            "start_training", "finalize_dataset", "configure_training",
            "load_sam", "load_yolo", "load_unet",
            "import_star_file", "push_to_hub", "track_particles",
            "batch_run_sam",
        )
        if needs_image and not self._state.get("image_name"):
            return "Error: No image is loaded. Tell the user to open an image first."
        if name in ("run_sam_auto", "batch_run_sam") and not self._state.get("sam_loaded"):
            return "Error: SAM model is not loaded. Call load_sam first, then retry."
        if name in ("run_yolo_detect", "run_yolo_segment", "pipe_yolo_to_sam") and not self._state.get("yolo_loaded"):
            return "Error: YOLO model is not loaded. Call load_yolo first, then retry."
        if name == "pipe_yolo_to_sam" and not self._state.get("sam_loaded"):
            return "Error: SAM model is not loaded. Both YOLO and SAM must be loaded for pipe_yolo_to_sam."
        if name == "run_unet" and not self._state.get("unet_loaded"):
            return "Error: UNet model is not loaded. Call load_unet first, then retry."

        if name in _NEEDS_CONFIRM:
            summary = params.get("summary", name)
            self.confirm_needed.emit(name, summary, params)
            return "Confirmation dialog shown to user."
        else:
            is_nav  = name in ("next_image", "prev_image", "go_to_image")
            is_meas = name == "export_measurements"
            if is_nav and self._context is not None:
                self._context.arm_nav_wait()   # clear event BEFORE signal fires
            self.tool_called.emit(name, params)
            if is_meas:
                return (
                    "Measurements exported and displayed. "
                    "Tell the user: the results are now visible in the Analysis tab on the right — "
                    "the Particles sub-tab shows the full data table and the Figures sub-tab shows "
                    "the histogram. The CSV is saved to acorn_measurements/measurements.csv "
                    "in the same folder as the images."
                )
            if is_nav:
                if self._context is not None:
                    self._context.wait_for_image_load(timeout=10.0)
                    try:
                        fresh = self._context.get_nav_state()
                        self._state.update(fresh)
                    except Exception:
                        pass
                new_name  = self._state.get("image_name", "unknown")
                new_px    = self._state.get("pixel_size_nm", 0)
                px_str    = f"{new_px:.4f} nm/px" if new_px and new_px != 1.0 else "not yet calibrated (1.0 nm default)"
                ann_count = self._state.get("annotation_count", 0)
                labels    = self._state.get("annotation_labels", {})
                ann_str   = (
                    f"{ann_count} existing annotations" +
                    (f" ({', '.join(f'{v} {k}' for k, v in labels.items())})" if labels else "")
                ) if ann_count else "no existing annotations"
                return f"Moved to {new_name}. Pixel size: {px_str}. {ann_str}."
            return "Action dispatched — results will appear on the canvas as pending annotations. The user must click Accept All to keep them."
