# tests/infra/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **infra/**

**Reproducibility-gate tests** — covers [`src/infra/repro.py`](../../src/infra/repro.py), the "regenerate-and-diff" gate that backs the Appendix-8 claim that every number in the report is reproducible from a clean checkout.

Uses `repo_root` from `conftest.py`; e2e tests use `tmp_path` when a synthetic repo layout is needed. `test_split_hashes_check_passes_on_committed_repo` skips on branches where `data/processed/SPLIT_HASHES.md` has not been committed. On `main` and `feature/restructure-and-polish` it must pass. The test has no side effects — it reads committed bytes and hashes in-memory; it never regenerates data.

## What's here

| File | Contents |
|---|---|
| [`test_repro.py`](test_repro.py) | Unit tests for each `Check` helper (`check_python_pins`, `check_git_clean`, `check_artefacts_exist`, `check_split_hashes_match`, `check_transformer_run_files`, `check_rf_predictions_regenerate`, `check_evaluate_regenerates`), `Report.as_dict()` aggregation, committed-data e2e asserting `split_hashes_match` passes on `main`. |

## How it was produced

Hand-written pytest.

```bash
python -m pytest tests/infra/ -q

# Or the gate directly (outside pytest) for human-readable ledger:
python -m src.infra.repro
```

## How it's consumed

- CI runs this subpackage + runs the gate.
- Pinned by Report **Appendix 8** as part of the 320-test suite and the 7/7 PASS claim.

## How to regenerate

```bash
python -m pytest tests/infra/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../scripts/`](../scripts/)
- **↓ Children**: none
