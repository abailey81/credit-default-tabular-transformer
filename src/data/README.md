# src/data/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **data/**

**Data ingestion + preprocessing** — owns everything between the raw UCI "Default of Credit Card Clients" dataset and the committed train/val/test CSVs under [`../../data/processed/splits/`](../../data/processed/splits/). Consumed by Section 2 (Data) and Section 3 (Model) of the report, and pinned by Appendix 8 (Reproducibility).

Handles resilient fetching (UCI API → local `.xls` fallback), schema normalisation, 22-feature engineering for the RF baseline, stratified 70/15/15 splitting, and leak-free `RobustScaler` fit on train only. Writes `feature_metadata.json` — the hard contract the tokenizer and baselines depend on.

## What's here

| File | Contents |
|---|---|
| [`sources.py`](sources.py) | Resilient loader with provenance. `build_default_data_source` tries the live UCI API first and falls back to the local `.xls`; every consumer must route through it so provenance is written to `data/processed/source_provenance.json`. |
| [`preprocessing.py`](preprocessing.py) | Clean, engineer, split, scale. Writes `train_{raw,engineered,scaled}.csv`, `val_*`, `test_*`, plus `feature_metadata.json` and `validation_report.json`. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written module; deterministic output — regenerates bit-stably via `python -m src.infra.repro`. `feature_metadata.json` is the schema contract: do not add fields without updating the tokenizer (`src.tokenization`).

```bash
python -m src.data.preprocessing --output-dir data/processed
python -m src.data.sources --probe   # provenance probe only
```

Also wrapped by `scripts/run_pipeline.py --preprocess-only` and the Stage-1 driver `scripts/run_all.py`.

## How it's consumed

- [`../tokenization/`](../tokenization/) reads `feature_metadata.json` for categorical vocabs.
- [`../baselines/`](../baselines/) consumes `feature_metadata.json` + the engineered CSVs.
- [`../evaluation/fairness.py`](../evaluation/fairness.py) reads `test_raw.csv` for protected attributes.
- [`../../tests/data/`](../../tests/data/) — currently covered transitively by `tests/infra/test_repro.py`.
- Report **Section 2** (Data) and **Section 3** (Model) both cite these modules.

## How to regenerate

```bash
python -m src.data.preprocessing --output-dir data/processed
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
