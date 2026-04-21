# results/baseline/rf/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ baseline](../) > **rf/**

**Bit-stable Random Forest test-set predictions + metrics** — regenerated in seconds without retuning. Used as the reference output for the reproducibility check and consumed by Section 3 (Model) of the report plus Section 4 (Experiments) evaluation consumers.

These two files are the deterministic RF side of every head-to-head comparison. The tune is expensive (~minutes); the predictions replay is cheap (~seconds). The repro harness diffs these bytes against the committed copy with tolerance `max|Δp| < 1e-6`.

## What's here

| File | Contents |
|---|---|
| [`test_predictions.npz`](test_predictions.npz) | Arrays `y_true` (N,) + `y_proba` (N,) + `y_pred` (N,) on the held-out test set. |
| [`test_metrics.json`](test_metrics.json) | Test AUC / PR-AUC / F1 / accuracy / precision / recall / specificity / kappa / ECE / Brier at the tuned threshold. |

## How it was produced

[`src/baselines/rf_predictions.py`](../../../src/baselines/rf_predictions.py) — loads the tuned RF (fitted inside `random_forest.py`), scores the canonical test split. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.baselines.rf_predictions
```

## How it's consumed

- [`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) — head-to-head comparison table.
- [`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py) — RF-side probability array for calibration.
- [`src/evaluation/significance.py`](../../../src/evaluation/significance.py) — RF-side array for paired-bootstrap tests.
- Report **Section 3** figures in [`../../figures/baseline/`](../../../figures/baseline/) pull from the same NPZ.

## How to regenerate

```bash
poetry run python -m src.baselines.rf_predictions
```

## Neighbours

- **↑ Parent**: [`../`](../) — baseline/ index
- **↔ Siblings**: none (root baseline/ contains only files alongside this subfolder)
- **↓ Children**: none
