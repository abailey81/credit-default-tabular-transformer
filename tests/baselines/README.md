# `tests/baselines/` — Random Forest tests

Covers `src/baselines/rf_predictions.py` (the deterministic refit
path). `random_forest.py` itself (the 200-iter tuning search) is not
unit-tested directly — it is too slow and is covered transitively by
`tests/infra/test_repro.py`, which regenerates RF predictions from
`rf_config.json` and diffs the committed copy.

## What's covered

| File                          | Subject |
|---|---|
| `test_rf_predictions.py`      | JSON schema round-trip, deterministic refit, per-row probability shape + bounds, e2e against committed `rf_config.json`. |

## Fixtures used

`repo_root` from `conftest.py` to locate `results/baseline/rf/`. No
other fixtures needed — the RF does not consume the tokenizer.

## Running

```bash
python -m pytest tests/baselines/ -q
```

## Gotchas

- The committed-data e2e test skips if `rf_config.json` is missing
  (typical on a fresh clone before Stage 3 has run). Run
  `python scripts/run_all.py --only rf` first if you want full
  coverage.
- The refit is deterministic to `max|Δp| < 1e-6` — if that tolerance
  ever tightens, check `np.random` seeding in `rf_predictions.py`
  before relaxing the test.
