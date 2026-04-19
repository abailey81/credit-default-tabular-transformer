# tests/baselines/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **baselines/**

**Random Forest tests** — covers [`src/baselines/rf_predictions.py`](../../src/baselines/rf_predictions.py) (the deterministic refit path). `random_forest.py` (the 200-iter tuning search) is not unit-tested directly — too slow — and is covered transitively by `tests/infra/test_repro.py` regenerating RF predictions from `rf_config.json` and diffing the committed copy. Gated by Appendix 8 (Reproducibility) of the report.

Uses only `repo_root` from `conftest.py` to locate `results/baseline/rf/`; no tokenizer fixtures needed (the RF consumes engineered CSVs directly).

## What's here

| File | Contents |
|---|---|
| [`test_rf_predictions.py`](test_rf_predictions.py) | JSON schema round-trip, deterministic refit, per-row probability shape + bounds, e2e against committed `rf_config.json`. |

## How it was produced

Hand-written pytest. The committed-data e2e test skips if `rf_config.json` is missing (typical on a fresh clone before Stage 3 has run).

```bash
python -m pytest tests/baselines/ -q
# For full e2e, first:
python scripts/run_all.py --only rf
```

The refit is deterministic to `max|Δp| < 1e-6` — if that tolerance ever tightens, check `np.random` seeding in `rf_predictions.py` before relaxing the test.

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
python -m pytest tests/baselines/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../evaluation/`](../evaluation/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
