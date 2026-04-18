"""
rf_predictions.py — Regenerate the tuned Random-Forest's per-row test
probabilities using the hyperparameters already committed in
``results/rf_config.json``, and persist them in the same
``test_predictions.npz`` layout ``train.py`` uses for the transformer.

Motivation
----------
The original ``random_forest.py`` pipeline writes aggregate metrics
(``results/rf_metrics.csv``) and tuned hyperparameters
(``results/rf_config.json``) but not the raw test-set probability vector.
Several downstream analyses in Phase 11/12 *need* the probability vector:

    * Phase 11  (calibration)   → post-hoc ECE/Brier on RF predictions.
    * Phase 12  (significance)  → DeLong AUC-difference, McNemar on paired
                                  binary predictions, paired-bootstrap ΔAUC.
    * Phase 11  (reliability)   → the transformer-vs-RF reliability plot
                                  in the report's §4 needs RF bin rates.

Rather than re-running the expensive 200-iter RandomizedSearchCV from
``random_forest.py``, we fit the *already-tuned* final estimator on the
train+val concatenation (matching the original pipeline's final-fit) and
score the committed test split. Output is written in the same shape as
``train.py`` so every Phase 11/12 module can consume both models through
a single loader.

Outputs
-------
    results/rf/test_predictions.npz   — arrays y_true, y_prob, y_pred.
    results/rf/test_metrics.json      — Brier + ECE + kappa + specificity +
                                        every metric ``train.py`` produces,
                                        so ``evaluate.py`` can read it
                                        with the same loader.

CLI
---
    poetry run python src/rf_predictions.py
    poetry run python src/rf_predictions.py --config results/rf_config.json \\
        --output-dir results/rf

References: Plan §9 (RF benchmark), §10 (evaluation).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from train import compute_classification_metrics  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42
TARGET = "DEFAULT"


def _coerce_best_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """``rf_config.json`` stringifies every hyperparameter. Map back to the
    scalars sklearn expects — ints, floats, None, or unchanged strings."""
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            out[k] = v
            continue
        if v == "None":
            out[k] = None
            continue
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v  # leave as string (e.g. "sqrt", "entropy")
    return out


def fit_and_predict(
    config_path: Path,
    train_csv: Path,
    val_csv: Path,
    test_csv: Path,
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    cfg = json.loads(Path(config_path).read_text())
    best = _coerce_best_params(cfg["best_params"])
    threshold = float(cfg.get("best_threshold", threshold))

    logger.info("Loading train/val/test splits from %s", train_csv.parent)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Match the original pipeline: final refit on train+val.
    df_fit = pd.concat([df_train, df_val], ignore_index=True)
    X_fit = df_fit.drop(columns=[TARGET]).values
    y_fit = df_fit[TARGET].astype(int).values

    X_test = df_test.drop(columns=[TARGET]).values
    y_test = df_test[TARGET].astype(int).values

    model = RandomForestClassifier(
        **best,
        random_state=random_state,
        n_jobs=-1,
    )
    logger.info("Fitting tuned RandomForest with params %s", best)
    model.fit(X_fit, y_fit)

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_classification_metrics(y_test, y_prob, threshold=threshold)

    return {
        "y_true": y_test,
        "y_prob": y_prob,
        "y_pred": y_pred,
        "threshold": threshold,
        "metrics": metrics,
        "best_params": best,
        "random_state": random_state,
    }


def save_predictions(result: Dict[str, Any], output_dir: Path) -> Dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / "test_predictions.npz"
    np.savez(
        npz_path,
        y_true=result["y_true"].astype(np.int64),
        y_prob=result["y_prob"].astype(np.float32),
        y_pred=result["y_pred"].astype(np.int64),
    )

    json_path = output_dir / "test_metrics.json"
    json_path.write_text(json.dumps({
        "threshold": result["threshold"],
        "metrics": result["metrics"],
        "best_params": result["best_params"],
        "random_state": result["random_state"],
    }, indent=2, default=str))

    return {"predictions": npz_path, "metrics": json_path}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit the already-tuned RF using hyperparameters from rf_config.json, "
            "score the test split, and persist per-row probabilities + metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("results/rf_config.json"))
    p.add_argument("--train-csv", type=Path,
                   default=Path("data/processed/train_engineered.csv"))
    p.add_argument("--val-csv", type=Path,
                   default=Path("data/processed/val_engineered.csv"))
    p.add_argument("--test-csv", type=Path,
                   default=Path("data/processed/test_engineered.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/rf"))
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return p


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    result = fit_and_predict(
        args.config, args.train_csv, args.val_csv, args.test_csv,
        random_state=args.random_state,
    )
    paths = save_predictions(result, args.output_dir)
    logger.info("AUC %.4f, Brier %.4f, ECE %.4f",
                result["metrics"]["auc_roc"],
                result["metrics"]["brier"],
                result["metrics"]["ece"])
    logger.info("Wrote %s and %s", paths["predictions"], paths["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
