"""Tests for src/repro.py.

Unit tests for the individual Check helpers plus a gentle smoke test
that the full runner doesn't crash. We don't invoke the subprocesses
inside the check runs here — those are expensive — we just verify the
helper logic.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import repro  # noqa: E402


def test_compare_dataframes_identical_ok():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.2, 0.3]})
    ok, _ = repro._compare_dataframes(df, df.copy())
    assert ok


def test_compare_dataframes_shape_mismatch():
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})
    ok, detail = repro._compare_dataframes(df1, df2)
    assert not ok
    assert "shape" in detail


def test_compare_dataframes_within_rtol():
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({"a": [1.000001, 2.000001, 3.000001]})
    ok, _ = repro._compare_dataframes(df1, df2, rtol=1e-4)
    assert ok


def test_compare_dataframes_beyond_rtol():
    df1 = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.5, 3.0]})
    ok, detail = repro._compare_dataframes(df1, df2, rtol=1e-4)
    assert not ok
    assert "column" in detail


def test_compare_dataframes_string_columns():
    """String-valued cells (e.g. "0.78 ± 0.01" from evaluate.py) compare exactly."""
    df1 = pd.DataFrame({"model": ["Transformer", "RF"], "score": ["0.78", "0.79"]})
    df2 = df1.copy()
    ok, _ = repro._compare_dataframes(df1, df2)
    assert ok
    df3 = pd.DataFrame({"model": ["Transformer", "TR"], "score": ["0.78", "0.79"]})
    ok, _ = repro._compare_dataframes(df1, df3)
    assert not ok


def test_check_artefacts_exist_returns_check():
    """On the committed repo this should pass; at minimum it returns a Check."""
    c = repro.check_artefacts_exist(REPO)
    assert isinstance(c, repro.Check)
    assert c.name == "artefacts_exist"


def test_check_python_pins_passes():
    c = repro.check_python_pins(REPO)
    assert c.passed, c.detail


def test_report_summarisation():
    r = repro.Report()
    r.add(repro.Check(name="a", passed=True))
    r.add(repro.Check(name="b", passed=False, detail="oops"))
    d = r.as_dict()
    assert d["n_checks"] == 2
    assert d["n_passed"] == 1
    assert d["all_passed"] is False


def test_run_all_does_not_crash(tmp_path):
    """End-to-end: the runner should return a Report, even if some
    checks fail (e.g. we're in a dirty working tree)."""
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    rep = repro.run_all(REPO, scratch)
    assert isinstance(rep, repro.Report)
    assert len(rep.checks) >= 4
