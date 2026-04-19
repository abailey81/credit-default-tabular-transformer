"""Refit the tuned RF from its saved config and emit per-row test predictions.

Why this module exists separately from :mod:`random_forest`
-----------------------------------------------------------
Two reasons, in order of importance:

1. **The downstream phases (calibration §11, statistical significance §12,
   SHAP §13, bootstrapped confidence intervals) need per-row probabilities in
   the same ``test_predictions.npz`` layout that ``train.py`` produces for
   every transformer variant.** :mod:`random_forest` only writes
   aggregate metrics (``rf_metrics.csv``) and per-feature importance — it
   never persists ``predict_proba`` output. Rebuilding the downstream
   interface on top of the aggregate metrics is fragile; it's cleaner to
   materialise the per-row probs once and share the npz contract.

2. **Regenerating the predictions is cheap and bit-stable.** The full
   RandomizedSearchCV in :func:`run_rf_benchmark` is a ~1 000-fit search that
   takes minutes even on 8 cores. But once ``rf_config.json`` records the
   winning hyperparameters we can refit the *single* tuned RF on
   ``train + val`` in seconds and reproduce the exact per-row probability
   vector every time (sklearn's RF with a fixed ``random_state`` is
   deterministic). So this module is the "fast path" for downstream
   pipelines that don't care about the tuning grid but need the numbers.

Public surface
--------------
:func:`fit_and_predict` refits and scores; :func:`save_predictions` writes
the two artefacts (``test_predictions.npz`` + ``test_metrics.json``) in the
layout ``train.py`` uses.

Non-obvious dependency: we re-use
:func:`src.training.train.compute_classification_metrics` for the metric
dict so RF and transformer metrics are computed by exactly the same code
path — important for like-for-like comparison in the CHANGELOG tables."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..training.train import compute_classification_metrics

logger = logging.getLogger(__name__)

__all__ = [
    "fit_and_predict",
    "save_predictions",
    "main",
    "DEFAULT_RANDOM_STATE",
    "TARGET",
]

DEFAULT_RANDOM_STATE = 42
TARGET = "DEFAULT"


def _coerce_best_params(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce string-serialised RF params back to their sklearn-native types.

    :func:`export_results` stringifies the best-params dict so the JSON stays
    clean through None, class-weight dicts, and the odd tuple. Going the
    other way we have to reverse that:

    * ``"None"`` → Python ``None`` (for ``max_depth=None``, ``class_weight=None``).
    * Numeric-looking strings → ``float`` if they contain ``.``, else ``int``.
    * Anything else (``"sqrt"``, ``"entropy"``, ``"balanced_subsample"``)
      passes through unchanged — sklearn accepts these as literal strings.

    Non-string values (already the right type) are preserved as-is.
    """
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if not isinstance(v, str):
            out[k] = v
            continue
        if v == "None":
            out[k] = None
            continue
        try:
            # Heuristic: '.' means float, otherwise int. max_features=0.5 is
            # the canonical float case; n_estimators=200 the int case.
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError:
            out[k] = v  # "sqrt", "entropy", "log2", "balanced_subsample" …
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
    """Refit the tuned RF on ``train+val`` and score the test split.

    Parameters
    ----------
    config_path
        Path to ``rf_config.json`` written by :func:`export_results`.
        Supplies both the best hyperparameters and the tuned τ.
    train_csv, val_csv, test_csv
        Engineered-feature CSVs (the ``*_engineered.csv`` outputs of the
        preprocessing pipeline, NOT the ``*_scaled.csv`` files the
        transformer consumes — RF doesn't benefit from scaling).
    random_state
        Seed for the underlying :class:`RandomForestClassifier`. Keep fixed
        (default 42) for bit-stable reproduction of the per-row probs.
    threshold
        Fallback decision threshold if ``rf_config.json`` doesn't carry one.
        Normally overridden by ``cfg["best_threshold"]``.

    Returns
    -------
    dict
        ``y_true / y_prob / y_pred / threshold / metrics / best_params /
        random_state``. ``metrics`` is computed by the same
        :func:`compute_classification_metrics` the transformer loop uses.
    """
    cfg = json.loads(Path(config_path).read_text())
    best = _coerce_best_params(cfg["best_params"])
    # Prefer the config's tuned τ so npz probabilities align with the tuned
    # deployment decision boundary. Falls back to 0.5 if the older config
    # schema didn't record one.
    threshold = float(cfg.get("best_threshold", threshold))

    logger.info("Loading train/val/test splits from %s", train_csv.parent)
    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    # Fit on train+val: matches the pipeline's "tune on train/val, then
    # retrain on union before scoring test" convention. This is what gives
    # bit-stable per-row probs — otherwise tiny CV-fold order differences
    # would propagate through RF's per-tree bootstrapping.
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
    """Persist per-row predictions and a metrics sidecar.

    Layout mirrors :func:`src.training.train.main` exactly:

    * ``test_predictions.npz`` — arrays ``y_true`` (int64), ``y_prob``
      (float32), ``y_pred`` (int64). Consumed by calibration (§11) and
      significance (§12) modules as a drop-in replacement for any
      transformer variant's predictions.
    * ``test_metrics.json`` — threshold, metrics dict, tuned hyperparameters,
      and ``random_state`` for traceability.

    Returns the two paths as a dict for the caller to log / surface in its
    own CLI.
    """
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
    json_path.write_text(
        json.dumps(
            {
                "threshold": result["threshold"],
                "metrics": result["metrics"],
                "best_params": result["best_params"],
                "random_state": result["random_state"],
            },
            indent=2,
            default=str,
        )
    )

    return {"predictions": npz_path, "metrics": json_path}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Fit the already-tuned RF using hyperparameters from rf_config.json, "
            "score the test split, and persist per-row probabilities + metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("results/baseline/rf_config.json"))
    p.add_argument("--train-csv", type=Path, default=Path("data/processed/train_engineered.csv"))
    p.add_argument("--val-csv", type=Path, default=Path("data/processed/val_engineered.csv"))
    p.add_argument("--test-csv", type=Path, default=Path("data/processed/test_engineered.csv"))
    p.add_argument("--output-dir", type=Path, default=Path("results/baseline/rf"))
    p.add_argument("--random-state", type=int, default=DEFAULT_RANDOM_STATE)
    return p


def main(argv=None) -> int:
    """CLI entry point: refit the tuned RF and write the two prediction artefacts."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)
    result = fit_and_predict(
        args.config,
        args.train_csv,
        args.val_csv,
        args.test_csv,
        random_state=args.random_state,
    )
    paths = save_predictions(result, args.output_dir)
    logger.info(
        "AUC %.4f, Brier %.4f, ECE %.4f",
        result["metrics"]["auc_roc"],
        result["metrics"]["brier"],
        result["metrics"]["ece"],
    )
    logger.info("Wrote %s and %s", paths["predictions"], paths["metrics"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
