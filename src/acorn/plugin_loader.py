"""Plugin discovery via Python entry points."""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from acorn.plugin_base import AcornPlugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "acorn.plugins"


def discover_plugins(context: "AcornContext") -> list["AcornPlugin"]:
    """
    Load all installed packages that declare an 'acorn.plugins' entry point.

    Entry point format in a plugin's pyproject.toml:
        [project.entry-points."acorn.plugins"]
        acorn_analysis = "acorn_analysis.plugin:AnalysisPlugin"

    Failed plugins are logged as warnings and skipped; they must never crash
    the application.
    """
    from importlib.metadata import entry_points
    eps = entry_points(group=ENTRY_POINT_GROUP)
    plugins: list["AcornPlugin"] = []
    for ep in eps:
        try:
            cls = ep.load()
            instance = cls(context)
            plugins.append(instance)
            logger.debug("Loaded plugin: %s (%s)", ep.name, cls)
        except Exception as exc:
            logger.warning("Failed to load plugin %r: %s", ep.name, exc)
    plugins.sort(key=lambda p: (p.sort_order, p.PLUGIN_ID))
    return plugins
