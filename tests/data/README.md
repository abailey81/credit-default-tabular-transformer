# `tests/data/` — Data-subpackage tests

Covers (will cover) `src/data/` — the resilient UCI loader and the
preprocessing / split / scaling pipeline.

## Current state

No dedicated unit tests yet. Coverage is transitive:

- `tests/infra/test_repro.py::test_split_hashes_check_passes_on_committed_repo`
  diffs every byte of the regenerated splits against the committed
  SHA-256 ledger at `data/processed/SPLIT_HASHES.md`, which pins
  `src/data/preprocessing.py`'s output.
- `tests/scripts/test_run_all.py` exercises the Stage-1 dispatch that
  invokes `src/data/preprocessing.py`.

## Intended future coverage

| Future file         | Would cover |
|---|---|
| `test_sources.py`         | `build_default_data_source` API → local fallback ordering, provenance JSON schema. |
| `test_preprocessing.py`   | Schema normalisation idempotence, stratified split class-balance tolerance, scaling leak-safety, `feature_metadata.json` schema round-trip. |

## Running

Once tests land here they run with:

```bash
python -m pytest tests/data/ -q
```

## Gotcha

Do not re-run preprocessing in a test unless you scope to `tmp_path`.
The committed splits under `data/processed/splits/` are golden —
overwriting them is how you break the reproducibility gate.
