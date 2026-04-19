# `tests/infra/` — Reproducibility-gate tests

Covers `src/infra/repro.py` — the "regenerate-and-diff" reproducibility
gate that backs the Appendix-level claim that every number in the
report is reproducible from a clean checkout.

## What's covered

| File             | Subject |
|---|---|
| `test_repro.py`  | Unit tests for each `Check` helper (`check_python_pins`, `check_git_clean`, `check_artefacts_exist`, `check_split_hashes_match`, `check_transformer_run_files`, `check_rf_predictions_regenerate`, `check_evaluate_regenerates`), `Report.as_dict()` aggregation, and a committed-data e2e that asserts `split_hashes_match` passes on the committed `main`. |

## Fixtures used

`repo_root` from `conftest.py`. E2E tests use `tmp_path` when a
synthetic repo layout is needed.

## Running

```bash
python -m pytest tests/infra/ -q
```

Or run the gate directly (outside pytest) to see the human-readable
pass/fail ledger:

```bash
python -m src.infra.repro
```

## Gotchas

- `test_split_hashes_check_passes_on_committed_repo` skips on branches
  where `data/processed/SPLIT_HASHES.md` has not been committed.
  On `main` / `feature/restructure-and-polish` it must pass.
- The test has no side effects — it reads committed bytes and hashes
  in-memory; it never regenerates data.
