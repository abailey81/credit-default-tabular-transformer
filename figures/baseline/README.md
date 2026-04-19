# figures/baseline/

Random Forest baseline diagnostic plots — Section 3 of the report.

## Files

| File | Shows |
|---|---|
| `rf_confusion_matrix.png` | Test-set confusion matrix at the tuned threshold. |
| `rf_roc_pr_curves.png` | ROC and precision–recall curves (test set). |
| `rf_feature_importance.png` | Gini feature importances, top-20. |
| `rf_threshold_analysis.png` | F1 / precision / recall vs decision threshold sweep. |
| `rf_tuning_analysis.png` | CV score surface for the RandomizedSearch grid. |

## Produced by

[`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) —
full tune + fit + evaluate pipeline (writes both the figures and
`results/baseline/rf_*`).

Predictions only (no retune) can be regenerated via
[`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py).

## Consumed by

- Report **Section 3** (Baseline).
- `results/evaluation/comparison/` loads the RF test predictions for the
  head-to-head table.

## Regenerate

```bash
poetry run python -m src.baselines.random_forest
```

Deterministic under fixed seed (see `results/baseline/rf_config.json`);
regenerates bit-stably via `python -m src.infra.repro`.
