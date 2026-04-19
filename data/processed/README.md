# `data/processed/` — preprocessed artefacts

Everything downstream of `scripts/run_pipeline.py --preprocess-only` lives
here. The three artefact families below are the hard contract every model,
evaluator, and test consumes.

## Layout

```
processed/
├── SPLIT_HASHES.md           # hash ledger (tracked)
├── feature_metadata.json     # tokeniser / categorical stats (tracked)
├── validation_report.json    # data-quality summary (tracked)
└── splits/                   # see splits/README.md
    ├── {train,val,test}_raw.csv
    ├── {train,val,test}_scaled.csv
    └── {train,val,test}_engineered.csv
```

The CSVs are git-ignored (large, deterministic-given-seed, regenerable);
the three files at this directory root are tracked.

## Artefact families

### 1. Metadata — `feature_metadata.json`

Per-feature descriptors the transformer tokeniser and the RF baseline
consume. The schema has two blocks:

- `numerical_features` — for each column: `mean`, `std`, `min`, `max`,
  `median`, `q25`, `q75`, scaler parameters. Used by the tokeniser to
  z-score new batches and by EDA for distribution checks.
- `categorical_features` — for each column: `n_categories`,
  `values` list, `value_to_index` map. Used to build the embedding tables
  (`src/tokenization/tokenizer.py::build_categorical_vocab`).

### 2. Data-quality — `validation_report.json`

Machine-readable record of the validation sweep `run_preprocessing_pipeline`
performs before it writes any split: row count, target balance,
per-column null / out-of-range counts, and the fetch timestamp + source
URL. A mismatch here against the expected schema blocks the pipeline.

### 3. Splits — `splits/`

The nine CSVs (three stages × three splits). See `splits/README.md` for
the split methodology and which module consumes which stage.

## Regenerating

```bash
python scripts/run_pipeline.py --preprocess-only --source auto
python -m src.infra.repro          # verifies hashes match SPLIT_HASHES.md
```

A clean rebuild is deterministic — the committed hashes should match to
the byte. If they don't, something upstream drifted (scikit-learn RNG,
raw data, preprocessing logic); `docs/REPRODUCIBILITY.md` is the
runbook.
