# `data/` — dataset root

UCI "Default of Credit Card Clients" (dataset id 350, Taiwan 2005). 30,000
clients, 23 predictors, binary target `DEFAULT` with 22.12% base rate. Every
number in the paper flows from these files.

## Layout

```
data/
├── raw/
│   ├── README.md
│   └── default_of_credit_card_clients.xls   # tracked local fallback
└── processed/
    ├── README.md
    ├── SPLIT_HASHES.md                       # SHA-256 of every split
    ├── feature_metadata.json                 # tokeniser / categorical stats
    ├── validation_report.json                # data-quality summary
    └── splits/
        ├── README.md
        ├── {train,val,test}_raw.csv          # pre-scaling, 24 cols
        ├── {train,val,test}_scaled.csv       # Robust-scaled numerics, 24 cols
        └── {train,val,test}_engineered.csv   # + 22 derived features, 46 cols
```

Everything under `processed/splits/` is git-ignored (large, regenerable);
`SPLIT_HASHES.md`, `feature_metadata.json`, and `validation_report.json` are
tracked so reviewers can verify provenance without regenerating.

## Provenance

- **Raw.** `data/raw/default_of_credit_card_clients.xls` is the tracked
  local fallback fetched from the UCI ML Repository. The canonical loader is
  `src/data/sources.py::build_default_data_source`, which tries the live UCI
  API first and falls back to this file on network failure.
- **Processed.** Produced by `scripts/run_pipeline.py --preprocess-only`,
  which drives `src/data/preprocessing.py`. That pipeline applies the schema
  / categorical cleanup, does a 70 / 15 / 15 stratified split on `DEFAULT`,
  fits a `RobustScaler` on train only and applies it to val + test, and
  finally derives 22 engineering features for the RF baseline.

## Reproducibility

- `docs/REPRODUCIBILITY.md` — full recipe + taxonomy of the seven repro
  checks.
- `data/processed/SPLIT_HASHES.md` — committed SHA-256 of every split CSV
  and the feature metadata. `python -m src.infra.repro` verifies.
