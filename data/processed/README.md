# data/processed/

> **Breadcrumb**: [↑ repo root](../../) > [↑ data](../) > **processed/**

**Preprocessed artefacts** — the hard contract every model, evaluator, and test consumes. Feeds Section 2 (Data), Section 3 (Model), Section 4 (Experiments), and Appendix 8 (Reproducibility).

Everything downstream of `scripts/run_pipeline.py --preprocess-only` lives here. The CSVs under `splits/` are git-ignored (large, deterministic-given-seed, regenerable); the three files at this directory root are tracked so reviewers can verify provenance without rerunning the pipeline. A clean rebuild is deterministic — committed hashes must match byte-for-byte or `python -m src.infra.repro` fails.

## What's here

| File / Subfolder | Contents |
|---|---|
| [`SPLIT_HASHES.md`](SPLIT_HASHES.md) | SHA-256 ledger pinning every split CSV + `feature_metadata.json`. |
| [`feature_metadata.json`](feature_metadata.json) | Per-feature descriptors: numerical stats (mean/std/q25/q75/...) + categorical `value_to_index` maps. Tokenizer + embedding + RF all consume this. |
| [`validation_report.json`](validation_report.json) | Data-quality summary: row count, target balance, per-column null / out-of-range counts, fetch timestamp, source URL. |
| [`splits/`](splits/) | Nine CSVs (three stages × three splits). |

## How it was produced

[`src/data/preprocessing.py`](../../src/data/preprocessing.py) (`run_preprocessing_pipeline`). Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
```

## How it's consumed

- [`src/tokenization/tokenizer.py`](../../src/tokenization/tokenizer.py) reads `feature_metadata.json` to build the categorical vocabulary.
- [`src/training/train.py`](../../src/training/train.py) + [`src/training/train_mtlm.py`](../../src/training/train_mtlm.py) load `splits/*_scaled.csv`.
- [`src/baselines/random_forest.py`](../../src/baselines/random_forest.py) loads `splits/*_engineered.csv`.
- [`src/evaluation/fairness.py`](../../src/evaluation/fairness.py) reads `splits/test_raw.csv` for protected attributes.
- [`src/infra/repro.py`](../../src/infra/repro.py) verifies hashes against `SPLIT_HASHES.md`.

## How to regenerate

```bash
poetry run python scripts/run_pipeline.py --preprocess-only --source auto
poetry run python -m src.infra.repro  # verifies SPLIT_HASHES.md
```

If the hashes drift something upstream changed (scikit-learn RNG, raw data, preprocessing logic); [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) is the runbook.

## Neighbours

- **↑ Parent**: [`../`](../) — data/ root
- **↔ Siblings**: [`../raw/`](../raw/)
- **↓ Children**: [`splits/`](splits/)
