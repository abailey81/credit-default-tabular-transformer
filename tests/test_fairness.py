"""fairness.py — subgroup metrics, disparity table, e2e."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

from src.evaluation import fairness as fa  # noqa: E402


@pytest.fixture
def tiny_data():
    rng = np.random.default_rng(0)
    n = 2000
    attr = rng.integers(1, 3, size=n)
    pos_prob = np.where(attr == 1, 0.4, 0.1)
    y = rng.binomial(1, pos_prob)
    p = np.clip(pos_prob + 0.1 * rng.standard_normal(n), 0.01, 0.99)
    return y, p, attr


def test_subgroup_metrics_smoke(tiny_data):
    y, p, attr = tiny_data
    results = fa.audit_attribute(y, p, attr, "SEX", "run", "identity")
    assert len(results) == 2
    g1 = next(r for r in results if r.subgroup_code == 1)
    g2 = next(r for r in results if r.subgroup_code == 2)
    assert g1.base_rate > g2.base_rate
    assert np.isfinite(g1.auc_roc) and np.isfinite(g2.auc_roc)


def test_disparity_table_uses_largest_as_ref(tiny_data):
    y, p, attr = tiny_data
    results = fa.audit_attribute(y, p, attr, "SEX", "run", "identity")
    df = fa.disparity_table(results)
    largest_n = max(r.n for r in results)
    ref = df[df["n"] == largest_n].iloc[0]
    assert ref["demographic_parity_diff"] == 0
    assert ref["equal_opportunity_violation"] == 0


def test_subgroup_too_small_marked_underpowered():
    rng = np.random.default_rng(1)
    n = 110
    attr = np.concatenate([np.ones(100), 2 * np.ones(10)]).astype(int)
    y = rng.binomial(1, 0.22, size=n)
    p = rng.uniform(0, 1, size=n)
    results = fa.audit_attribute(y, p, attr, "SEX", "run", "identity")
    small = next(r for r in results if r.subgroup_code == 2)
    assert small.underpowered is True
    big = next(r for r in results if r.subgroup_code == 1)
    assert big.underpowered is False


def test_audit_attribute_handles_single_class_subgroup():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    attr = np.array([1, 1, 1, 2, 2, 2])
    results = fa.audit_attribute(y, p, attr, "SEX", "run", "identity")
    for r in results:
        assert not np.isfinite(r.auc_roc)


def test_attribute_labels_exposed():
    assert "SEX" in fa.ATTRIBUTE_LABELS
    assert "EDUCATION" in fa.ATTRIBUTE_LABELS
    assert "MARRIAGE" in fa.ATTRIBUTE_LABELS
    assert 1 in fa.ATTRIBUTE_LABELS["SEX"]


def test_main_end_to_end(tmp_path):
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    test_raw = REPO / "data" / "processed" / "test_raw.csv"
    if not (seed_42 / "test_predictions.npz").is_file() or not test_raw.is_file():
        pytest.skip("no committed transformer runs or preprocessing")
    rc = fa.main(
        [
            "--runs",
            str(seed_42),
            "--rf-dir",
            str(REPO / "results" / "baseline" / "rf"),
            "--test-raw",
            str(test_raw),
            "--output-dir",
            str(tmp_path / "out"),
            "--figures-dir",
            str(tmp_path / "fig"),
        ]
    )
    assert rc == 0
    assert (tmp_path / "out" / "subgroup_metrics.csv").is_file()
    df = pd.read_csv(tmp_path / "out" / "subgroup_metrics.csv")
    assert {"SEX", "EDUCATION", "MARRIAGE"}.issubset(set(df["attribute"]))
