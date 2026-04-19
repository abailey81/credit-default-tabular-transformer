"""rf_predictions.py — regenerates RF per-row probabilities from rf_config.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent.parent

from src.baselines import rf_predictions as rf  # noqa: E402


def test_coerce_best_params_handles_stringified_types():
    raw = {
        "n_estimators": "200",
        "max_depth": "None",
        "min_samples_split": "2",
        "max_features": "sqrt",
        "ccp_alpha": "0.01",
        "class_weight": "balanced",
    }
    out = rf._coerce_best_params(raw)
    assert out["n_estimators"] == 200
    assert out["max_depth"] is None
    assert out["min_samples_split"] == 2
    assert out["max_features"] == "sqrt"
    assert out["ccp_alpha"] == pytest.approx(0.01)
    assert out["class_weight"] == "balanced"


def test_coerce_best_params_passes_through_non_strings():
    raw = {"n_estimators": 200, "max_depth": None}
    out = rf._coerce_best_params(raw)
    assert out == raw


def test_fit_and_predict_tiny(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    cols = ["LIMIT_BAL", "AGE", "PAY_0", "PAY_AMT1", "BILL_AMT1", "DEFAULT"]
    df = pd.DataFrame({c: rng.normal(size=n) for c in cols[:-1]})
    df["DEFAULT"] = rng.binomial(1, 0.25, size=n)

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    df.iloc[:200].to_csv(data_dir / "train.csv", index=False)
    df.iloc[200:300].to_csv(data_dir / "val.csv", index=False)
    df.iloc[300:].to_csv(data_dir / "test.csv", index=False)

    cfg = {
        "best_params": {
            "n_estimators": "20",
            "max_depth": "4",
            "min_samples_split": "2",
            "min_samples_leaf": "1",
        },
        "best_threshold": 0.5,
    }
    cfg_path = tmp_path / "rf_config.json"
    cfg_path.write_text(json.dumps(cfg))

    result = rf.fit_and_predict(
        cfg_path,
        data_dir / "train.csv",
        data_dir / "val.csv",
        data_dir / "test.csv",
        random_state=0,
    )
    assert result["y_true"].shape == (100,)
    assert result["y_prob"].shape == (100,)
    assert ((result["y_prob"] >= 0) & (result["y_prob"] <= 1)).all()
    assert set(result["metrics"].keys()) >= {"auc_roc", "brier", "ece"}


def test_save_predictions_writes_files(tmp_path):
    out_dir = tmp_path / "rf_out"
    result = {
        "y_true": np.zeros(10, dtype=int),
        "y_prob": np.linspace(0, 1, 10),
        "y_pred": np.zeros(10, dtype=int),
        "threshold": 0.5,
        "metrics": {"auc_roc": 0.5},
        "best_params": {"n_estimators": 10},
        "random_state": 42,
    }
    paths = rf.save_predictions(result, out_dir)
    assert paths["predictions"].is_file()
    assert paths["metrics"].is_file()
    preds = np.load(paths["predictions"])
    assert "y_prob" in preds.files and preds["y_prob"].shape == (10,)


def test_main_against_committed_data():
    cfg = REPO / "results" / "baseline" / "rf_config.json"
    proc = REPO / "data" / "processed"
    if not (cfg.is_file() and (proc / "splits" / "train_engineered.csv").is_file()):
        pytest.skip("no committed rf_config or engineered data")
    # scratch dir — leave committed results/rf alone
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        rc = rf.main(
            [
                "--config",
                str(cfg),
                "--output-dir",
                str(Path(tmp) / "rf"),
            ]
        )
        assert rc == 0
