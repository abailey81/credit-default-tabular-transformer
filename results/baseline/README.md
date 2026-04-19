# results/baseline/

Random Forest baseline artefacts — Section 3 of the report.

## Files

| File | Shows |
|---|---|
| `rf_config.json` | Best hyperparameters after `RandomizedSearchCV` + the tuned decision threshold. |
| `rf_cross_validation.csv` | Per-fold CV scores (AUC, F1, accuracy, ...) for the search. |
| `rf_feature_importance.csv` | Gini importance per feature, sorted descending. |
| `rf_metrics.csv` | Aggregate train / val / test metrics at the tuned threshold. |

## Subfolder

| Folder | Contents |
|---|---|
| [`rf/`](rf/) | Per-sample test predictions (`test_predictions.npz`) + replicate metrics JSON regenerated cheaply by `rf_predictions.py`. |

## Produced by

[`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) —
the full tune + fit + evaluate pipeline (slow, ~minutes).

The cheap predictions-only replay lives in
[`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py)
and writes to `rf/` only.

## Consumed by

- Report **Section 3** (tables + text).
- Evaluation module loads `rf/test_predictions.npz` for the head-to-head
  comparison table in `results/evaluation/comparison/`.

## Regenerate

```bash
poetry run python -m src.baselines.random_forest        # full retune
poetry run python -m src.baselines.rf_predictions       # cheap replay only
```

Deterministic under fixed seed (`rf_config.json::seed`); regenerates
bit-stably via `python -m src.infra.repro`.
