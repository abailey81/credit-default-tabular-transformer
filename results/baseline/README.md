# results/baseline/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **baseline/**

**Random Forest baseline artefacts** — consumed by Section 3 (Model build-up) of the report for the tuned RF hyperparameters, CV log, feature importances, and aggregate metrics.

The RF is the main benchmark that the transformer is compared against. The full tune-plus-fit pipeline lives in [`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) (slow, ~minutes) and writes the four files at this directory root; the cheap predictions-only replay lives in [`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py) and writes to [`rf/`](rf/). Both paths are deterministic under fixed seed (`rf_config.json::seed`).

## What's here

| File / Subfolder | Contents |
|---|---|
| [`rf_config.json`](rf_config.json) | Best hyperparameters after `RandomizedSearchCV` + the tuned decision threshold. |
| [`rf_cross_validation.csv`](rf_cross_validation.csv) | Per-fold CV scores (AUC, F1, accuracy, ...) for the search. |
| [`rf_feature_importance.csv`](rf_feature_importance.csv) | Gini importance per feature, sorted descending. |
| [`rf_metrics.csv`](rf_metrics.csv) | Aggregate train / val / test metrics at the tuned threshold. |
| [`rf/`](rf/) | Per-sample test predictions (`test_predictions.npz`) + replicate metrics JSON regenerated cheaply by `rf_predictions.py`. |

## How it was produced

- Root files: [`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) — full tune + fit + evaluate (slow).
- [`rf/`](rf/) contents: [`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py) — cheap predictions-only replay.

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.baselines.random_forest        # full retune
poetry run python -m src.baselines.rf_predictions       # cheap replay only
```

## How it's consumed

- Report **Section 3** (tables + text).
- [`../evaluation/comparison/`](../evaluation/comparison/) loads `rf/test_predictions.npz` for the head-to-head table.
- [`../../figures/baseline/`](../../figures/baseline/) renders the RF diagnostic plots from these files.

## How to regenerate

```bash
poetry run python -m src.baselines.random_forest
poetry run python -m src.baselines.rf_predictions
```

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../transformer/`](../transformer/), [`../mtlm/`](../mtlm/), [`../evaluation/`](../evaluation/), [`../repro/`](../repro/), [`../pipeline/`](../pipeline/)
- **↓ Children**: [`rf/`](rf/)
