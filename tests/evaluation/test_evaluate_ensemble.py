"""evaluate.py — ensemble_run + the rf-full-predictions branch."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent.parent

from src.evaluation import evaluate as ev  # noqa: E402


@pytest.fixture
def synthetic_runs():
    rng = np.random.default_rng(0)
    n = 300
    y = rng.integers(0, 2, size=n)
    runs = []
    for i in range(3):
        p = np.clip(0.3 + 0.4 * y + 0.15 * rng.standard_normal(n), 0.0, 1.0)
        runs.append(
            {
                "run_name": f"seed_{i}",
                "metrics": {
                    "auc_roc": 0.8,
                    "auc_pr": 0.5,
                    "f1": 0.5,
                    "accuracy": 0.75,
                    "precision": 0.5,
                    "recall": 0.5,
                    "brier": 0.2,
                    "ece": 0.1,
                },
                "threshold": 0.5,
                "y_true": y,
                "y_prob": p,
                "y_pred": (p >= 0.5).astype(int),
            }
        )
    return runs


def test_ensemble_run_arithmetic_matches_plain_mean(synthetic_runs):
    result = ev.ensemble_run(synthetic_runs, mode="arithmetic", display_name="ens")
    expected_prob = np.mean([r["y_prob"] for r in synthetic_runs], axis=0)
    assert np.allclose(result["y_prob"], expected_prob)
    assert result["run_name"] == "ens"
    assert result["component_runs"] == ["seed_0", "seed_1", "seed_2"]


def test_ensemble_run_geometric_sane(synthetic_runs):
    result = ev.ensemble_run(synthetic_runs, mode="geometric", display_name="ens_geo")
    assert ((result["y_prob"] >= 0) & (result["y_prob"] <= 1)).all()
    arith = ev.ensemble_run(synthetic_runs, mode="arithmetic")
    assert not np.allclose(result["y_prob"], arith["y_prob"])


def test_ensemble_run_returns_none_for_single_run(synthetic_runs):
    assert ev.ensemble_run(synthetic_runs[:1]) is None


def test_ensemble_run_rejects_mismatched_y_true(synthetic_runs):
    bad = [dict(synthetic_runs[0])]
    mismatch = dict(synthetic_runs[1])
    mismatch["y_true"] = synthetic_runs[1]["y_true"].copy()
    mismatch["y_true"][0] ^= 1
    bad.append(mismatch)
    with pytest.raises(ValueError):
        ev.ensemble_run(bad)


def test_build_comparison_table_includes_ensemble_row(synthetic_runs):
    from_scratch = ev.aggregate_runs(synthetic_runs)
    ensemble = ev.ensemble_run(synthetic_runs, mode="arithmetic", display_name="ens")
    table = ev.build_comparison_table(
        from_scratch,
        transformer_mtlm=None,
        rf={},
        ensemble=ensemble,
    )
    assert any("ensemble" in str(x).lower() for x in table["model"])


def test_load_rf_from_predictions_returns_none_when_missing(tmp_path):
    assert ev.load_rf_from_predictions(tmp_path / "no_such") is None


def test_aggregate_runs_warns_and_propagates_on_nan(synthetic_runs, caplog):
    # NaN metrics must propagate AND warn — a silent nanmean would hide
    # which run and which metric went bad
    runs = [dict(r) for r in synthetic_runs]
    runs[0] = dict(runs[0])
    runs[0]["metrics"] = dict(runs[0]["metrics"])
    runs[0]["metrics"]["auc_roc"] = float("nan")

    with caplog.at_level(logging.WARNING, logger="evaluate"):
        agg = ev.aggregate_runs(runs)

    assert np.isnan(agg["mean"]["auc_roc"])
    messages = [rec.getMessage() for rec in caplog.records]
    assert any("NaN auc_roc" in m and runs[0]["run_name"] in m for m in messages), (
        f"Expected a NaN-auc_roc warning mentioning {runs[0]['run_name']!r}; " f"got: {messages}"
    )


def test_load_rf_from_predictions_loads_committed_rf():
    rf = REPO / "results" / "baseline" / "rf"
    if not (rf / "test_predictions.npz").is_file():
        pytest.skip("no rf prediction artefact")
    out = ev.load_rf_from_predictions(rf)
    assert out is not None
    assert out["y_prob"].shape == out["y_true"].shape
    assert "metrics" in out and "auc_roc" in out["metrics"]
