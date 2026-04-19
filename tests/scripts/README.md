# `tests/scripts/` — End-to-end orchestrator tests

Covers `scripts/run_all.py` — the one-command Option A driver. The
goal is CLI plumbing coverage (argument parsing, stage ordering,
`--only` / `--skip-*` / `--force` gating, log-file creation) without
paying the cost of any real pipeline stage.

## What's covered

| File                 | Subject |
|---|---|
| `test_run_all.py`    | Subprocess.Popen is stubbed so every stage sees a zero-exit fake child; the orchestrator walks its state machine in milliseconds. Catches regressions in: `--only eda`, `--only rf`, `--skip-preprocess`, `--skip-eda`, `--force`, log-file rotation, and exit-code propagation. |

## Fixtures used

`repo_root` (to resolve `scripts/run_all.py`). All subprocess
invocations are mocked — tests never shell out to the real pipeline.

## Running

```bash
python -m pytest tests/scripts/ -q
```

## Gotchas

- When you add a new pipeline stage to `run_all.py`, also add a
  dispatch test here. The stage table is the single source of truth
  and this test is the lock.
- Tests modify `sys.path` to make `import run_all` work; they do not
  leak state into other test files because each test restores
  `sys.path` in a finalizer.
