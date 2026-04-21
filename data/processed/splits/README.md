# data/processed/splits/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ data](../../) > [↑ processed](../) > **splits/**

**Train / val / test CSVs** — three stages × three splits = nine files. Feeds Section 2 (Data), Section 3 (Model), and Section 4 (Experiments) of the report; gated by Appendix 8 (Reproducibility) via `SPLIT_HASHES.md`.

Split methodology: 70 / 15 / 15 train / val / test, stratified on `DEFAULT`, which preserves the 22.12 % base rate in each split. Determinism comes from a fixed seed inside `preprocessing.py` — identical raw data in ⇒ identical splits out. `RobustScaler` is **fit on train only** and then applied to val + test (no leakage). All nine CSVs are git-ignored; `../SPLIT_HASHES.md` pins the exact bytes.

## What's here

| File | Contents |
|---|---|
| [`train_raw.csv`](train_raw.csv) / [`val_raw.csv`](val_raw.csv) / [`test_raw.csv`](test_raw.csv) | 24 cols: schema + categorical cleanup, pre-scaling. Consumed by fairness audit (SEX, EDUCATION, MARRIAGE, AGE_BIN) and EDA notebooks. |
| [`train_scaled.csv`](train_scaled.csv) / [`val_scaled.csv`](val_scaled.csv) / [`test_scaled.csv`](test_scaled.csv) | 24 cols: raw → `RobustScaler` on numerics, categoricals unchanged. Transformer input. |
| [`train_engineered.csv`](train_engineered.csv) / [`val_engineered.csv`](val_engineered.csv) / [`test_engineered.csv`](test_engineered.csv) | 46 cols: scaled + 22 derived features (utilisation, repayment, delinquency, trend). RF input. |

## How it was produced

[`src/data/preprocessing.py`](../../../src/data/preprocessing.py) (`run_preprocessing_pipeline`). Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
```

## How it's consumed

- `*_scaled.csv` → [`src/training/train.py`](../../../src/training/train.py), [`src/training/train_mtlm.py`](../../../src/training/train_mtlm.py), [`src/evaluation/uncertainty.py`](../../../src/evaluation/uncertainty.py).
- `*_engineered.csv` → [`src/baselines/random_forest.py`](../../../src/baselines/random_forest.py), [`src/baselines/rf_predictions.py`](../../../src/baselines/rf_predictions.py).
- `*_raw.csv` → [`src/evaluation/fairness.py`](../../../src/evaluation/fairness.py), [`src/analysis/eda.py`](../../../src/analysis/eda.py).

## How to regenerate

```bash
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
poetry run python -m src.infra.repro  # runs check_split_hashes_match
```

Any drift fails CI.

## Neighbours

- **↑ Parent**: [`../`](../) — processed/ index
- **↔ Siblings**: none (only file sibling is `../SPLIT_HASHES.md`)
- **↓ Children**: none
