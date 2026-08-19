"""
The simulator panel used to be one 45-field scroll covering three unrelated jobs.
It is now three tabs — but every widget the plugin and CLU read must survive that.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def panel():
    app = QApplication.instance() or QApplication([])
    from acorn_tem_sim.panel import TemSimPanel
    p = TemSimPanel()
    yield p
    p.deleteLater()
    app.processEvents()


def test_three_tabs_one_per_workflow(panel):
    labels = [panel._tabs.tabText(i) for i in range(panel._tabs.count())]
    assert labels == ["Micrograph", "4D-STEM", "Match a real image"]


def test_every_widget_the_plugin_reads_still_exists(panel):
    for attr in (
        "_microscope", "_species", "_add_np", "_add_contam", "_energy_filter",
        "_conv_mrad", "_scan_size", "_reference_path", "_modality", "_calibrate",
        "_generate_btn", "_fourd_btn", "_ref_btn", "_status", "_output_dir",
    ):
        assert hasattr(panel, attr), f"regrouping lost {attr}"


def test_params_contract_is_unchanged(panel):
    p = panel.params()
    assert len(p) == 34
    for key in ("output_dir", "count", "simulation_path", "preset",
                "image_size_px", "pixel_size_a", "total_dose_e_per_a2"):
        assert key in p


def test_params_round_trip(panel):
    panel.apply_params({"total_dose_e_per_a2": 55.0, "image_size_px": 256})
    p = panel.params()
    assert p["total_dose_e_per_a2"] == 55.0
    assert p["image_size_px"] == 256


def test_4dstem_action_still_carries_probe_settings(panel):
    got = {}
    panel.action_requested.connect(lambda a, d: got.update({"action": a, **d}))
    panel._output_dir.setText("/tmp/out")
    panel._scan_size.setValue(96)
    panel._conv_mrad.setValue(31.0)
    panel._emit_action("generate_4dstem")
    assert got["action"] == "generate_4dstem"
    assert got["scan_size"] == 96
    assert got["conv_mrad"] == 31.0
    assert "BF" in got["detectors"]


def test_reference_action_still_carries_reference_settings(panel):
    got = {}
    panel.action_requested.connect(lambda a, d: got.update({"action": a, **d}))
    panel._output_dir.setText("/tmp/out")
    panel._reference_path.setText("/tmp/ref.mrc")
    panel._calibrate.setChecked(True)
    panel._emit_action("simulate_from_reference")
    assert got["action"] == "simulate_from_reference"
    assert got["reference_path"] == "/tmp/ref.mrc"
    assert got["calibrate"] is True
    assert got["modality"] == "cryoem"


def test_each_tab_ends_with_its_own_run_button(panel):
    """A tab you cannot run from is a tab that sends people hunting."""
    for btn in (panel._generate_btn, panel._fourd_btn, panel._ref_btn):
        assert btn.parent() is not None
        assert btn.text()
