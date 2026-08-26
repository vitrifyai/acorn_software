"""Plugin discovery via Python entry points."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from acorn.gui.context import AcornContext
    from acorn.plugin_base import AcornPlugin

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "acorn.plugins"


@dataclass(frozen=True)
class PluginFailure:
    """A plugin that could not be loaded, and whether that is news.

    `missing_dependency` separates the two very different reasons a plugin is
    absent. Several are meant to be optional -- the 3-D viewer needs mrcfile,
    CLU needs an LLM client -- and reporting those as faults on every launch
    teaches people to ignore the report. Anything else is a bug.
    """

    name: str
    error: BaseException
    missing_dependency: bool


def is_missing_dependency(exc: BaseException) -> bool:
    """True when `exc` means "an optional package is not installed".

    ImportError covers ModuleNotFoundError, and also the `from x import y` case
    where the module exists but is too old to have the name.
    """
    return isinstance(exc, ImportError)


_failures: list[PluginFailure] = []


def load_failures() -> list[PluginFailure]:
    """Plugins that failed during the most recent `discover_plugins` call."""
    return list(_failures)


def discover_plugins(context: "AcornContext") -> list["AcornPlugin"]:
    """
    Load all installed packages that declare an 'acorn.plugins' entry point.

    Entry point format in a plugin's pyproject.toml:
        [project.entry-points."acorn.plugins"]
        acorn_analysis = "acorn_analysis.plugin:AnalysisPlugin"

    Failed plugins are recorded and skipped; they must never crash the
    application. `load_failures()` returns what went wrong, so a caller can show
    the ones that are genuine faults rather than leaving a hole where a panel
    should be.
    """
    from importlib.metadata import entry_points
    eps = entry_points(group=ENTRY_POINT_GROUP)
    plugins: list["AcornPlugin"] = []
    _failures.clear()
    for ep in eps:
        try:
            cls = ep.load()
            instance = cls(context)
            plugins.append(instance)
            logger.debug("Loaded plugin: %s (%s)", ep.name, cls)
        except Exception as exc:
            optional = is_missing_dependency(exc)
            _failures.append(PluginFailure(ep.name, exc, optional))
            if optional:
                # An uninstalled extra is a configuration, not a fault.
                logger.info("Plugin %r unavailable (optional dependency): %s",
                            ep.name, exc)
            else:
                logger.warning("Failed to load plugin %r: %s", ep.name, exc,
                               exc_info=True)
    plugins.sort(key=lambda p: (p.sort_order, p.PLUGIN_ID))
    return plugins
