# results/baseline/rf/

Bit-stable Random Forest test-set predictions + metrics. Regenerated in
seconds without retuning — used as the reference output for the
reproducibility check.

## Files

| File | Shows |
|---|---|
| `test_predictions.npz` | Arrays `y_true` (N,) + `y_proba` (N,) + `y_pred` (N,) on the held-out test set. |
| `test_metrics.json` | Test AUC/PR-AUC/F1/accuracy/precision/recall/specificity/kappa/ECE/Brier at the tuned threshold. |

## Produced by

[`src/baselines/rf_predictions.py`](../../../src/baselines/rf_predictions.py)
— loads the tuned RF (fitted inside `random_forest.py`), scores the
canonical test split, and writes these two files.

## Consumed by

- [`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) for
  the head-to-head comparison table.
- [`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py)
  and [`significance.py`](../../../src/evaluation/significance.py) as the
  RF-side probability array.
- Report **Section 3** figures pull from the same NPZ.

## Regenerate

```bash
poetry run python -m src.baselines.rf_predictions
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
