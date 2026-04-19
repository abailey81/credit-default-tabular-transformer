# `data/processed/splits/` — train / val / test CSVs

Nine files: three split stages × three splits. All produced in one pass by
`src/data/preprocessing.py::run_preprocessing_pipeline`.

## Split methodology

- **Ratio.** 70 / 15 / 15 train / val / test, stratified on `DEFAULT`.
  Preserves the 22.12% base rate in each split.
- **Determinism.** Fixed seed inside `preprocessing.py`; identical raw data
  in ⇒ identical splits out. Hashes in `../SPLIT_HASHES.md` pin the exact
  bytes.
- **Scaling.** `RobustScaler` is **fit on train only** and then applied to
  val + test. No leakage.

## The three stages

| Stage | Columns | Produced how | Consumers |
| --- | --- | --- | --- |
| `_raw` | 24 | schema + categorical cleanup, pre-scaling | `src/evaluation/fairness.py` (reads `test_raw.csv` for protected-attribute codes: SEX, EDUCATION, MARRIAGE, AGE_BIN); EDA notebooks |
| `_scaled` | 24 | raw → `RobustScaler` on numerics, categoricals unchanged | `src/training/train.py`, `src/training/train_mtlm.py`, `src/evaluation/uncertainty.py` — the transformer's input |
| `_engineered` | 46 | scaled + 22 derived features (utilisation, repayment, delinquency, trend) | `src/baselines/random_forest.py`, `src/baselines/rf_predictions.py` — the RF input |

## Files

```
train_raw.csv        val_raw.csv        test_raw.csv
train_scaled.csv     val_scaled.csv     test_scaled.csv
train_engineered.csv val_engineered.csv test_engineered.csv
```

All are git-ignored (see `.gitignore`); regenerate with
`python scripts/run_pipeline.py --preprocess-only --source auto`.

## Verification

`python -m src.infra.repro` runs `check_split_hashes_match`, which reads
the SHA-256 table in `../SPLIT_HASHES.md` and confirms every CSV here
matches byte-for-byte. Any drift fails CI.
