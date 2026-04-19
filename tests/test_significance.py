"""significance.py — McNemar / DeLong / paired bootstrap / BH-FDR / power."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent

from src.evaluation import significance as sig  # noqa: E402
from sklearn.metrics import roc_auc_score


def test_mcnemar_identical_predictions_p_is_one():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500)
    y_pred = rng.integers(0, 2, size=500)
    r = sig.mcnemar_test(y, y_pred, y_pred)
    assert r.p_value == 1.0
    assert r.effect == 0.0
    assert r.extra["b"] == 0 and r.extra["c"] == 0


def test_mcnemar_clearly_different_predictions_significant():
    y = np.tile([0, 1], 100)
    r = sig.mcnemar_test(y, y, np.zeros_like(y))
    assert r.p_value < 0.001


def test_mcnemar_small_discordant_uses_exact_binomial():
    y = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    a = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
    b = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    r = sig.mcnemar_test(y, a, b)
    assert r.extra["method"] == "exact_binomial"


def test_delong_identical_scores_p_is_one():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=300)
    p = rng.uniform(0, 1, size=300)
    r = sig.delong_auc_test(y, p, p)
    assert r.p_value == 1.0
    assert abs(r.effect) < 1e-12


def test_delong_stat_matches_sign_of_auc_diff():
    rng = np.random.default_rng(2)
    n = 1000
    y = rng.integers(0, 2, size=n)
    p_a = np.where(y == 1, rng.uniform(0.5, 1.0, size=n), rng.uniform(0.0, 0.5, size=n))
    p_b = np.where(y == 1, rng.uniform(0.4, 0.9, size=n), rng.uniform(0.1, 0.6, size=n))
    r = sig.delong_auc_test(y, p_a, p_b)
    auc_a, auc_b = roc_auc_score(y, p_a), roc_auc_score(y, p_b)
    assert (r.effect > 0) == (auc_a > auc_b)
    assert r.ci_low <= r.effect <= r.ci_high


def test_delong_requires_both_classes():
    y = np.zeros(10, dtype=int)
    p = np.linspace(0.1, 0.9, 10)
    with pytest.raises(ValueError):
        sig.delong_auc_test(y, p, p)


def test_paired_bootstrap_ci_brackets_observed():
    rng = np.random.default_rng(3)
    n = 500
    y = rng.integers(0, 2, size=n)
    p_a = rng.uniform(0, 1, size=n)
    p_b = rng.uniform(0, 1, size=n)
    r = sig.paired_bootstrap(
        y, lambda y, p: float(roc_auc_score(y, p)),
        p_a, p_b, n_resamples=200, seed=0, metric_name="auc_roc",
    )
    assert r.ci_low <= r.effect <= r.ci_high
    assert 0.0 <= r.p_value <= 1.0


def test_paired_bootstrap_zero_effect_large_p():
    rng = np.random.default_rng(4)
    n = 500
    y = rng.integers(0, 2, size=n)
    p = rng.uniform(0, 1, size=n)
    r = sig.paired_bootstrap(
        y, lambda y, p: float(roc_auc_score(y, p)),
        p, p, n_resamples=200, seed=0, metric_name="auc_roc",
    )
    assert r.p_value > 0.5
    assert abs(r.effect) < 1e-12


def test_bh_fdr_preserves_nothing_when_all_ones():
    p = [1.0, 1.0, 1.0, 1.0]
    out = sig.bh_fdr(p, q=0.05)
    assert not out["rejected"].any()


def test_bh_fdr_rejects_all_when_all_zero():
    p = [1e-9, 1e-8, 1e-9, 1e-7]
    out = sig.bh_fdr(p, q=0.05)
    assert out["rejected"].all()


def test_bh_fdr_q_values_monotone_in_p_rank():
    p = [0.001, 0.01, 0.02, 0.04, 0.5, 0.8]
    out = sig.bh_fdr(p, q=0.05)
    order = np.argsort(p)
    q_sorted = out["q_values"][order]
    assert (np.diff(q_sorted) >= -1e-9).all()


def test_min_n_for_auc_difference_scales_with_gap():
    n_small = sig.min_n_for_auc_difference(0.785, 0.78, prevalence=0.22)
    n_big = sig.min_n_for_auc_difference(0.80, 0.78, prevalence=0.22)
    assert n_small > n_big, (
        f"Smaller effect should need more N: n_small={n_small}, n_big={n_big}"
    )


def test_min_n_for_auc_difference_infinite_on_identical():
    n = sig.min_n_for_auc_difference(0.78, 0.78, prevalence=0.22)
    assert n >= 1e8


def test_run_all_pairs_on_committed(tmp_path):
    seed_42 = REPO / "results" / "transformer" / "seed_42"
    seed_1 = REPO / "results" / "transformer" / "seed_1"
    rf = REPO / "results" / "baseline" / "rf"
    if not all((seed_42 / "test_predictions.npz").is_file()
               for p in [seed_42, seed_1, rf]
               for _ in [p]) or not (rf / "test_predictions.npz").is_file():
        pytest.skip("committed predictions missing")
    loaded = [sig._load_run(seed_42), sig._load_run(seed_1), sig._load_run(rf)]
    loaded = [r for r in loaded if r is not None]
    df = sig.run_all_pairs(loaded, n_resamples=200, seed=0)
    assert not df.empty
    assert "q_value" in df.columns
    finite = df["p_value"].dropna()
    assert ((finite >= 0) & (finite <= 1)).all()
