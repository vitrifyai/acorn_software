"""Numbers that end up in a manuscript must be reportable as they stand.

These are not tests of the statistics -- scipy does those. They test how results
are PRESENTED, which is where the defects were: a p-value rounded into
meaninglessness, groups dropped without saying so, and a figure disagreeing with
its own table.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import pandas as pd
from acorn_plotting.figures import _stats
from acorn_plotting.stats import _format_p, _p_phrase, _stars, run_statistics


def _frame(groups: dict) -> pd.DataFrame:
    labels, values = [], []
    for name, vals in groups.items():
        labels.extend([name] * len(vals))
        values.extend(vals)
    return pd.DataFrame({"label": labels, "ecd_nm": values})


# --- p-value reporting ------------------------------------------------------

def test_a_tiny_p_value_is_not_reported_as_zero():
    """Rounding to four places turned anything below 0.00005 into "0.0", and the
    recommendation said so verbatim. No test yields p = 0."""
    rng = np.random.default_rng(0)
    df = _frame({"a": rng.normal(10, 1, 40), "b": rng.normal(40, 1, 40)})
    result = run_statistics(df, "ecd_nm")

    assert result["comparison"]["p_text"] == "< 0.0001"
    assert result["comparison"]["p"] > 0, "the numeric p must survive intact"
    assert "p = 0.0 " not in result["recommendation"]
    assert "p < 0.0001" in result["recommendation"]


def test_an_ordinary_p_value_is_reported_with_an_equals_sign():
    rng = np.random.default_rng(3)
    df = _frame({"a": rng.normal(10, 1, 40), "b": rng.normal(10.2, 1, 40)})
    rec = run_statistics(df, "ecd_nm")["recommendation"]
    assert "p = 0." in rec and "p < " not in rec


@pytest.mark.parametrize("p,expected", [
    (1e-90, "< 0.0001"), (0.00001, "< 0.0001"),
    (0.0005, "0.00050"), (0.03, "0.0300"), (0.5, "0.5000"),
])
def test_p_formatting(p, expected):
    assert _format_p(p) == expected


def test_p_phrase_reads_correctly_either_way():
    assert _p_phrase("< 0.0001") == "p < 0.0001"
    assert _p_phrase("0.0300") == "p = 0.0300"


# --- silent exclusion -------------------------------------------------------

def test_groups_too_small_to_test_are_named_not_dropped_in_silence():
    """Four conditions in, a two-group comparison out, and nothing saying which
    two were left behind."""
    rng = np.random.default_rng(1)
    df = _frame({"a": rng.normal(10, 1, 20), "b": rng.normal(12, 1, 20),
                 "c": rng.normal(30, 1, 2), "d": rng.normal(50, 1, 1)})
    result = run_statistics(df, "ecd_nm")

    excluded = {e["group"]: e["n"] for e in result["excluded"]}
    assert excluded == {"c": 2, "d": 1}
    assert "c (n=2)" in result["recommendation"]
    assert "d (n=1)" in result["recommendation"]
    assert [d["group"] for d in result["descriptive"]] == ["a", "b"]


def test_nothing_excluded_means_nothing_said():
    rng = np.random.default_rng(2)
    df = _frame({"a": rng.normal(10, 1, 20), "b": rng.normal(12, 1, 20)})
    result = run_statistics(df, "ecd_nm")
    assert result["excluded"] == []
    assert "Excluded" not in result["recommendation"]


def test_all_groups_too_small_still_reports_which():
    df = _frame({"a": [1.0, 2.0], "b": [3.0]})
    result = run_statistics(df, "ecd_nm")
    assert "error" in result
    assert {e["group"] for e in result["excluded"]} == {"a", "b"}


# --- figure and table must agree --------------------------------------------

@pytest.mark.parametrize("n", [3, 5, 12, 40, 200])
def test_figure_standard_deviation_matches_the_statistics_table(n):
    """The figure title used population sd while the table used sample sd, so
    the same data read differently in two places."""
    rng = np.random.default_rng(7)
    vals = rng.normal(30, 4, n)
    from_figure = _stats(vals)[2]
    from_table = run_statistics(_frame({"a": vals}), "ecd_nm")["descriptive"][0]["std"]
    assert from_figure == pytest.approx(from_table, rel=1e-12)


def test_a_single_value_has_no_standard_deviation_rather_than_a_crash():
    assert _stats(np.array([5.0])) == (1, 5.0, 0.0, 5.0)


# --- one definition of significance -----------------------------------------

def test_the_figure_and_the_text_use_the_same_thresholds():
    """A figure disagreeing with its own caption is the failure mode here."""
    from acorn_analysis.panel import _p_stars
    for p in (0.0009, 0.0011, 0.009, 0.011, 0.049, 0.051, 0.5):
        assert _stars(p) == _p_stars(p), p


@pytest.mark.parametrize("p,expected", [
    (0.0009, "***"), (0.005, "**"), (0.02, "*"), (0.05, "ns"), (0.9, "ns"),
])
def test_significance_thresholds(p, expected):
    assert _stars(p) == expected
