# tests/scripts/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **scripts/**

**End-to-end orchestrator tests** — covers [`scripts/run_all.py`](../../scripts/run_all.py), the one-command Option A driver. Goal is CLI plumbing coverage (argument parsing, stage ordering, `--only` / `--skip-*` / `--force` gating, log-file creation) without paying the cost of any real pipeline stage. Gated by Appendix 8 (Reproducibility) of the report.

All subprocess invocations are mocked — tests never shell out to the real pipeline. When you add a new pipeline stage to `run_all.py`, also add a dispatch test here — the stage table is the single source of truth and this test is the lock.

## What's here

| File | Contents |
|---|---|
| [`test_run_all.py`](test_run_all.py) | `subprocess.Popen` stubbed so every stage sees a zero-exit fake child; the orchestrator walks its state machine in milliseconds. Catches regressions in `--only eda`, `--only rf`, `--skip-preprocess`, `--skip-eda`, `--force`, log-file rotation, and exit-code propagation. |

## How it was produced

Hand-written pytest. Uses `repo_root` from `conftest.py` to resolve `scripts/run_all.py`. Tests modify `sys.path` to make `import run_all` work; they do not leak state because each test restores `sys.path` in a finalizer.

```bash
python -m pytest tests/scripts/ -q
```

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
python -m pytest tests/scripts/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/)
- **↓ Children**: none
