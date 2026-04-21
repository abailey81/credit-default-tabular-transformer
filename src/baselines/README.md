# src/baselines/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **baselines/**

**Random Forest benchmark** — implements the RF benchmark the transformer is compared against in Section 4 (Experiments) of the report. Keeps training and per-row prediction emission in separate modules so calibration / significance / fairness consumers can refit without rerunning the 200-iteration search.

Reads the engineered CSV (`train_engineered.csv`) from [`../../data/processed/splits/`](../../data/processed/splits/), not the scaled CSV — the transformer and RF consume different feature views deliberately (tree models do not need scaled features). Writes `rf_config.json` that `rf_predictions.py` refits from, so the two modules are tightly coupled by the JSON schema.

## What's here

| File | Contents |
|---|---|
| [`random_forest.py`](random_forest.py) | Hyperparameter-tuned RF benchmark: 200-iter randomised search across a 7-dimensional grid, class-balanced, stratified 5-fold inner CV. Persists `rf_config.json` + CV diagnostics under `results/baseline/`. |
| [`rf_predictions.py`](rf_predictions.py) | Refits the tuned RF from `rf_config.json` and emits deterministic per-row probabilities + metrics. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written; deterministic under fixed seed. Refit tolerance `max|Δp| < 1e-6` pinned by `src.infra.repro`.

```bash
# Tune + fit (slow; ~10 min on a laptop):
python -m src.baselines.random_forest --output-dir results/baseline/

# Refit from a saved config + emit per-row probabilities (fast):
python -m src.baselines.rf_predictions \
    --config results/baseline/rf_config.json \
    --output-dir results/baseline/rf
```

## How it's consumed

- [`../../results/baseline/`](../../results/baseline/) — RF config, CV log, feature importances, metrics.
- [`../../results/baseline/rf/`](../../results/baseline/rf/) — per-row predictions.
- [`../evaluation/`](../evaluation/) — consumes `rf/test_predictions.npz` for head-to-head comparison, calibration, and significance tests.
- Report **Section 3** (RF hyperparameters, feature-importance figure), **Section 4** (head-to-head comparison). Appendix 8 — `rf_predictions_regenerate` check in `src.infra.repro`.

## How to regenerate

```bash
python -m src.baselines.random_forest
python -m src.baselines.rf_predictions
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
