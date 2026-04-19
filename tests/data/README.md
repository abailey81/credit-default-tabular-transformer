# tests/data/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **data/**

**Data-subpackage tests** — covers (will cover) [`src/data/`](../../src/data/), the resilient UCI loader and the preprocessing / split / scaling pipeline. Gated by Appendix 8 (Reproducibility) of the report.

No dedicated unit tests yet. Coverage is transitive through two existing tests. This directory exists with a README to document the intended future coverage and the contract that preprocessing already satisfies via the hash ledger.

## What's here

No test files yet — only this README.

Intended future coverage:

| Future file | Would cover |
|---|---|
| `test_sources.py` | `build_default_data_source` API → local fallback ordering, provenance JSON schema. |
| `test_preprocessing.py` | Schema normalisation idempotence, stratified-split class-balance tolerance, scaling leak-safety, `feature_metadata.json` schema round-trip. |

## How it was produced

Directory scaffolded during the 2026 restructure. Transitive coverage today:

- `tests/infra/test_repro.py::test_split_hashes_check_passes_on_committed_repo` diffs every byte of the regenerated splits against [`data/processed/SPLIT_HASHES.md`](../../data/processed/SPLIT_HASHES.md), pinning `src/data/preprocessing.py` output.
- `tests/scripts/test_run_all.py` exercises the Stage-1 dispatch that invokes `src/data/preprocessing.py`.

## How it's consumed

- CI runs whatever lands here via `pytest tests/data/ -q`.
- Report **Appendix 8** will cite these tests once they land.

## How to regenerate

Once tests land here:

```bash
python -m pytest tests/data/ -q
```

Gotcha: do not re-run preprocessing in a test unless you scope to `tmp_path`. The committed splits under `data/processed/splits/` are golden — overwriting them breaks the reproducibility gate.

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
