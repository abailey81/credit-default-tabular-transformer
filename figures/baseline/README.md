# figures/baseline/

> **Breadcrumb**: [↑ repo root](../../) > [↑ figures](../) > **baseline/**

**Random Forest diagnostic plots** — 5 PNGs consumed by Section 3 (Model build-up) of the report, alongside the RF configuration and metrics under [`../../results/baseline/`](../../results/baseline/).

The RF is the tuned benchmark (200-iter `RandomizedSearchCV`) that the transformer is compared against. These figures are the visual side of Section 3 — the numeric side lives under `results/baseline/`. The "head-to-head" plot comparing RF with the transformer lives separately under [`../evaluation/comparison/`](../evaluation/comparison/).

## What's here

| File | Contents |
|---|---|
| [`rf_confusion_matrix.png`](rf_confusion_matrix.png) | Test-set confusion matrix at the tuned threshold. |
| [`rf_roc_pr_curves.png`](rf_roc_pr_curves.png) | ROC and precision-recall curves (test set). |
| [`rf_feature_importance.png`](rf_feature_importance.png) | Gini feature importances, top-20. |
| [`rf_threshold_analysis.png`](rf_threshold_analysis.png) | F1 / precision / recall vs decision threshold sweep. |
| [`rf_tuning_analysis.png`](rf_tuning_analysis.png) | CV score surface for the RandomizedSearch grid. |

## How it was produced

[`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) — full tune + fit + evaluate (writes both figures and `results/baseline/rf_*`). Deterministic under fixed seed (see [`../../results/baseline/`](../../results/baseline/)); regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.baselines.random_forest
```

Predictions-only replay (no retune) via [`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py).

## How it's consumed

- Report **Section 3** (Baseline).
- [`../../results/evaluation/comparison/`](../../results/evaluation/comparison/) loads the RF test predictions for the head-to-head table.

## How to regenerate

```bash
poetry run python -m src.baselines.random_forest
```

## Neighbours

- **↑ Parent**: [`../`](../) — figures/ index
- **↔ Siblings**: [`../eda/`](../eda/), [`../evaluation/`](../evaluation/)
- **↓ Children**: none
