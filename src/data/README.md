# `src/data/` — Data ingestion & preprocessing

Owns everything between the raw UCI "Default of Credit Card Clients"
dataset and the committed train/val/test CSVs under
`data/processed/splits/`. Handles resilient fetching, schema
normalisation, 22-feature engineering, stratified 70/15/15 splitting,
and leak-free scaling.

## Key modules

| Module | Purpose |
|---|---|
| `sources.py`       | Resilient loader with provenance: UCI ML Repository API → local manual `.xls` fallback. Every consumer must route through `build_default_data_source` so the provenance record is written to `data/processed/source_provenance.json`. |
| `preprocessing.py` | Clean, engineer, split, scale. Writes `train_{raw,engineered,scaled}.csv`, `val_*`, `test_*`, plus `feature_metadata.json` and `validation_report.json`. |

## Non-obvious dependencies

This subpackage writes `feature_metadata.json`, which every downstream
consumer (`src.tokenization`, `src.baselines`, `src.evaluation`) reads
to discover categorical vocabularies and the 23-feature
`TOKEN_ORDER`. Do not add fields to that JSON without updating the
tokenizer contract.

## Invocation

```bash
python -m src.data.preprocessing --output-dir data/processed
# provenance only — no preprocessing:
python -m src.data.sources --probe
```

Both modules are also wrapped by `scripts/run_pipeline.py --preprocess-only`
and the Stage-1 driver `scripts/run_all.py`.

## Tests

Tests live at `tests/data/`; there are no unit tests here yet — the
preprocessing contract is covered transitively by `tests/infra/test_repro.py`
(which diffs the regenerated splits against `SPLIT_HASHES.md`). A
`tests/data/README.md` records the intended future coverage.

## Report section

- Section 3 (Dataset & Preprocessing) references `sources.py` and
  `preprocessing.py`.
- Section 4 (EDA) reads the CSVs this subpackage writes.
- Appendix (Reproducibility) uses `SPLIT_HASHES.md` entries keyed by
  the filenames this module produces.
