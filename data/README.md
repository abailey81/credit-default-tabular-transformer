# data/

> **Breadcrumb**: [↑ repo root](../) > **data/**

**Dataset root** — holds the raw UCI "Default of Credit Card Clients" file and every preprocessing artefact. Feeds every downstream module and is referenced from Section 2 (Data) and Appendix 8 (Reproducibility) of the report.

UCI dataset id 350 (Taiwan 2005): 30,000 clients, 23 predictors, binary target `DEFAULT` with a 22.12 % base rate. Every number in the report ultimately flows from the files under this root. Everything below `processed/splits/` is git-ignored (large, deterministic-given-seed), while `SPLIT_HASHES.md`, `feature_metadata.json`, and `validation_report.json` are tracked so reviewers can verify provenance without regenerating anything.

## What's here

| File / Subfolder | Contents |
|---|---|
| [`raw/`](raw/) | Tracked local fallback of the UCI `.xls` source file. |
| [`processed/`](processed/) | Schema-clean artefacts: hashes, metadata, validation report, and split CSVs. |

## How it was produced

- **Raw.** `data/raw/default_of_credit_card_clients.xls` was fetched from the UCI ML Repository and committed as a local fallback. The canonical loader [`src/data/sources.py`](../src/data/sources.py) (`build_default_data_source`) tries the live UCI API first and only drops to this file on network failure.
- **Processed.** Produced by `scripts/run_pipeline.py --preprocess-only`, which drives [`src/data/preprocessing.py`](../src/data/preprocessing.py): schema + categorical cleanup, 70 / 15 / 15 stratified split on `DEFAULT`, `RobustScaler` fit on train only, then 22 engineered features for the RF baseline. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

## How it's consumed

- [`src/tokenization/`](../src/tokenization/) and [`src/training/`](../src/training/) load `processed/splits/*_scaled.csv` + `feature_metadata.json`.
- [`src/baselines/`](../src/baselines/) loads `processed/splits/*_engineered.csv`.
- [`src/evaluation/fairness.py`](../src/evaluation/fairness.py) reads `processed/splits/test_raw.csv` for protected attributes.
- [`docs/DATA_SHEET.md`](../docs/DATA_SHEET.md) documents provenance for Section 2.

## How to regenerate

```bash
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
poetry run python -m src.infra.repro  # verifies hashes match SPLIT_HASHES.md
```

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../src/`](../src/), [`../tests/`](../tests/), [`../scripts/`](../scripts/), [`../results/`](../results/), [`../figures/`](../figures/), [`../notebooks/`](../notebooks/), [`../docs/`](../docs/)
- **↓ Children**: [`raw/`](raw/), [`processed/`](processed/)
