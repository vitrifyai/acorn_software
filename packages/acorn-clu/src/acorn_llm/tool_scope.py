"""
Scoping CLU's tools to the active workspace.

CLU knows 50-odd commands. Offering all of them on every request is what makes it
pick a plausible neighbour instead of the right tool — "run cryoblob" once landed
on `load_sam` for exactly this reason. A workspace narrows the menu to the dozen
commands that belong to the job in front of the user, which is both more accurate
and cheaper per request.

Nothing is lost: a tool the model still names is dispatched normally, and the
"Show Every Panel" escape hatch (or an unknown workspace) restores the full set.
"""
from __future__ import annotations

# Always offered — moving around, looking at the image, and calibration are part
# of every job.
GLOBAL_TOOLS: frozenset[str] = frozenset({
    "next_image", "prev_image", "go_to_image", "scan_all_images",
    "set_pixel_size", "set_contrast", "apply_contrast_preset", "save_contrast_preset",
    "check_quality",
    "switch_workspace",     # the way out of a workspace that lacks the right tool
})

WORKSPACE_TOOLS: dict[str, frozenset[str]] = {
    "explore": frozenset({
        "add_scalebar", "compress_frames", "dose_comparison",
        "export_display_image", "import_star_file",
    }),
    "annotate": frozenset({
        "load_sam", "run_sam_auto", "run_sam_text", "batch_run_sam",
        "load_yolo", "run_yolo_detect", "run_yolo_segment", "batch_run_yolo",
        "load_unet", "run_unet", "batch_run_unet",
        "run_cryoblob", "detect_atoms",
        "accept_annotations", "reject_annotations", "undo_annotation",
        "clear_annotations", "rename_label",
        "import_star_file", "add_scalebar",
    }),
    "dataset": frozenset({
        "queue_for_export", "finalize_dataset",
        "configure_training", "start_training", "train_atom_model",
        "export_masks", "export_nexus", "export_display_image", "push_to_hub",
    }),
    "analyze": frozenset({
        "run_surface_area", "run_particle_analysis", "analyze_stem_atoms",
        "atom_statistics", "spatial_analysis", "track_particles",
        "configure_analysis_plot", "plot_measurements", "run_statistics",
        "export_measurements", "export_nexus",
    }),
    "simulate": frozenset({
        "generate_tem_simulation", "generate_tem_advanced", "generate_4dstem",
        "generate_fib_simulation", "simulate_from_reference",
    }),
}


def tool_names_for(workspace: str | None) -> frozenset[str] | None:
    """
    Names CLU should be offered in `workspace`, or None meaning "offer everything".

    None is returned for an unknown or missing workspace so that a caller which
    does not track workspaces keeps the old, unfiltered behaviour.
    """
    if not workspace or workspace not in WORKSPACE_TOOLS:
        return None
    return GLOBAL_TOOLS | WORKSPACE_TOOLS[workspace]


def scope_tools(tools: list[dict], workspace: str | None) -> list[dict]:
    """Filter a tool-schema list to `workspace`, preserving order."""
    allowed = tool_names_for(workspace)
    if allowed is None:
        return tools
    scoped = [t for t in tools if t["name"] in allowed]
    return scoped or tools     # never hand the model an empty toolbox
