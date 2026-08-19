"""
Workspaces — the five jobs ACORN is used for.

A workspace is a named task ("Explore", "Annotate", …). Choosing one decides
which control tabs are shown and which tool docks open; it never unloads the
image, the annotations, or a model that is already in memory. Everything hidden
by a workspace is one click away again via View ▸ Show Every Panel.

Adding a workspace means adding a WORKSPACES entry — nothing else in the GUI
hard-codes the list.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Control-panel tabs owned by the core, in canonical left-to-right order.
# Plugin tabs are appended after these in the order plugins are discovered.
CORE_TAB_ORDER = ("Contrast", "Annotate", "Segment", "Measure", "Export", "Train")


@dataclass(frozen=True)
class Workspace:
    """One named job. `tabs` and `docks` are what the window shows when it is active."""

    wid:      str            # stable id used in config files and plugin metadata
    label:    str            # what the switcher button says
    tagline:  str            # one line, shown on the welcome screen and as a tooltip
    blurb:    str            # a sentence for the welcome screen
    tabs:     tuple[str, ...]        # core/plugin tab labels to show, in order
    docks:    tuple[str, ...] = ()   # plugin ids to open automatically
    shortcut: str = ""               # e.g. "Ctrl+1"
    # Starting width of the control panel, in px. Workspaces whose tools live in
    # docks want a narrow panel so the image keeps the room; workspaces whose
    # tools are all tabs want a wide one. The user can drag it either way, and
    # the panel's own 320 px minimum still wins on a narrow window.
    panel_width: int = 400


WORKSPACES: tuple[Workspace, ...] = (
    Workspace(
        wid="explore",
        label="Explore",
        tagline="Look at images",
        blurb="Open files, adjust contrast, step through frames, and take a quick "
              "measurement. Start here if you are new.",
        tabs=("Contrast", "Measure"),
        shortcut="Ctrl+1",
        panel_width=320,
    ),
    Workspace(
        wid="annotate",
        label="Annotate",
        tagline="Mark things up",
        blurb="Draw regions by hand, or let SAM, YOLO, or UNet propose them and "
              "accept the ones you want.",
        tabs=("Annotate", "Segment", "Contrast"),
        shortcut="Ctrl+2",
        panel_width=420,
    ),
    Workspace(
        wid="dataset",
        label="Dataset",
        tagline="Build training data",
        blurb="Queue up annotated images, check their quality, train a model, and "
              "export in the format you need.",
        tabs=("Export", "Train"),
        shortcut="Ctrl+3",
        panel_width=400,
    ),
    Workspace(
        wid="analyze",
        label="Analyze",
        tagline="Get numbers out",
        blurb="Particle measurements, spatial statistics, tracking across a series, "
              "publication figures, and the 3D viewer.",
        tabs=("Measure",),
        docks=("acorn_spatial", "acorn_tracking", "acorn_3d"),
        shortcut="Ctrl+4",
        panel_width=320,
    ),
    Workspace(
        wid="simulate",
        label="Simulate",
        tagline="Make synthetic data",
        blurb="Generate realistic TEM, FIB-SEM, and 4D-STEM images with labels "
              "already attached, ready to feed back into Dataset.",
        tabs=("Contrast", "Annotate"),
        docks=("acorn_tem_sim", "acorn_fib_sim"),
        shortcut="Ctrl+5",
        panel_width=320,
    ),
)

DEFAULT_WORKSPACE = "explore"

# Docks that stay available no matter which workspace is active. They are not
# force-opened — the user's last choice is respected.
ALWAYS_AVAILABLE_DOCKS = ("acorn_llm",)


def by_id(wid: str) -> Optional[Workspace]:
    for ws in WORKSPACES:
        if ws.wid == wid:
            return ws
    return None


def workspace_for_tab(tab_label: str) -> Optional[Workspace]:
    """First workspace that shows `tab_label`, or None if no workspace claims it."""
    for ws in WORKSPACES:
        if tab_label in ws.tabs:
            return ws
    return None


# ── preferences ───────────────────────────────────────────────────────────────
# Stored under $XDG_CONFIG_HOME (the launchers point this at each install's own
# runtime dir), so two ACORN installs on one machine never share window state.

def _config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / "acorn" / "workspace.json"


@dataclass
class WorkspacePrefs:
    last_workspace: str = DEFAULT_WORKSPACE
    welcome_seen:   bool = False
    show_all:       bool = False   # user pinned every panel open
    extra:          dict = field(default_factory=dict)


def load_prefs() -> WorkspacePrefs:
    path = _config_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            known = WorkspacePrefs.__dataclass_fields__
            return WorkspacePrefs(**{k: v for k, v in data.items() if k in known})
        except Exception:
            pass
    return WorkspacePrefs()


def save_prefs(prefs: WorkspacePrefs) -> None:
    path = _config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({
            "last_workspace": prefs.last_workspace,
            "welcome_seen":   prefs.welcome_seen,
            "show_all":       prefs.show_all,
            "extra":          prefs.extra,
        }, indent=2))
        tmp.replace(path)
    except Exception:
        pass   # preferences are a convenience; never let them break startup
