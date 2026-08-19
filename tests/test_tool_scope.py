"""CLU's tool menu must shrink with the workspace without anything becoming unreachable."""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from acorn_llm.agent import _TOOLS, _KNOWN_TOOLS, build_system_prompt
from acorn_llm.tool_scope import GLOBAL_TOOLS, WORKSPACE_TOOLS, scope_tools, tool_names_for


def _all_mapped() -> set[str]:
    return set(GLOBAL_TOOLS) | {n for s in WORKSPACE_TOOLS.values() for n in s}


def test_every_tool_is_reachable_from_some_workspace():
    """A tool no workspace offers is a tool CLU can never be asked to use."""
    unreachable = {t["name"] for t in _TOOLS} - _all_mapped()
    assert not unreachable, f"unreachable tools: {sorted(unreachable)}"


def test_scope_map_has_no_phantom_tools():
    phantom = _all_mapped() - {t["name"] for t in _TOOLS}
    assert not phantom, f"scoped names that do not exist: {sorted(phantom)}"


def test_workspaces_match_the_gui():
    from acorn.gui.workspaces import WORKSPACES
    assert {w.wid for w in WORKSPACES} == set(WORKSPACE_TOOLS)


def test_scoping_actually_narrows_the_menu():
    full = len(_TOOLS)
    for wid in WORKSPACE_TOOLS:
        assert len(scope_tools(_TOOLS, wid)) < full, f"{wid} did not narrow anything"


def test_switch_workspace_is_always_offered():
    """Without it, a scoped CLU could not reach the tool the user actually wants."""
    assert "switch_workspace" in _KNOWN_TOOLS
    for wid in WORKSPACE_TOOLS:
        names = {t["name"] for t in scope_tools(_TOOLS, wid)}
        assert "switch_workspace" in names


def test_unknown_or_missing_workspace_offers_everything():
    assert tool_names_for(None) is None
    assert tool_names_for("nonsense") is None
    assert len(scope_tools(_TOOLS, None)) == len(_TOOLS)
    assert len(scope_tools(_TOOLS, "nonsense")) == len(_TOOLS)


def test_scoping_never_returns_an_empty_toolbox():
    assert scope_tools(_TOOLS, "explore")
    assert scope_tools([], "explore") == []


def test_cryoblob_and_sam_are_not_confusable_across_workspaces():
    """The original misroute: 'run cryoblob' answered by load_sam."""
    sim = {t["name"] for t in scope_tools(_TOOLS, "simulate")}
    assert "load_sam" not in sim and "run_cryoblob" not in sim
    ann = {t["name"] for t in scope_tools(_TOOLS, "annotate")}
    assert {"run_cryoblob", "load_sam"} <= ann     # both live here, by design


def test_prompt_names_the_workspace_and_the_way_out():
    prompt = build_system_prompt({"workspace": "explore"})
    assert "WORKSPACE: explore" in prompt
    assert "switch_workspace" in prompt


def test_prompt_without_a_workspace_makes_no_claims():
    prompt = build_system_prompt({})
    assert "WORKSPACE:" not in prompt
