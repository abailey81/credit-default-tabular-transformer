# tests/evaluation/

> **Breadcrumb**: [↑ repo root](../../) > [↑ tests](../) > **evaluation/**

**Evaluation subpackage tests** — covers all seven modules under [`src/evaluation/`](../../src/evaluation/). Each file has a synthetic unit path (always runs) and a committed-artefact e2e path (skips gracefully when `results/transformer/seed_42/` or `results/baseline/rf/` is missing). Gated by Appendix 8 (Reproducibility) of the report.

Set `MPLBACKEND=Agg` or `test_visualise.py` will try to open a GUI. E2E tests for `uncertainty` and `interpret` require `results/transformer/seed_42/best.pt` — they skip cleanly if missing.

## What's here

| File | Contents |
|---|---|
| [`test_calibration.py`](test_calibration.py) | Temperature / Platt / isotonic fits on synthetic data; ECE / MCE / Brier decomposition; reliability bin edges; e2e. |
| [`test_fairness.py`](test_fairness.py) | Subgroup metrics (AUC / TPR / FPR / selection rate) on synthetic data; disparity table schema; e2e SEX / EDUCATION / MARRIAGE audit. |
| [`test_uncertainty.py`](test_uncertainty.py) | `enable_dropout` flips only dropout modules; `mc_dropout_predict` restores eval mode; refuse-curve monotonicity; e2e. |
| [`test_significance.py`](test_significance.py) | McNemar / DeLong / paired bootstrap numerics; BH-FDR correction; Hanley-McNeil power; e2e transformer vs RF. |
| [`test_interpret.py`](test_interpret.py) | Attention rollout on synthetic tensors; per-feature importance ranking; e2e with committed `test_attn_weights.npz`. |
| [`test_visualise.py`](test_visualise.py) | PNG output, reliability-bin edges, CLI argparse. |
| [`test_evaluate_ensemble.py`](test_evaluate_ensemble.py) | `ensemble_run` aggregation, RF-full-predictions branch, comparison table schema. |

## How it was produced

Hand-written pytest. Uses `repo_root` + `metadata` fixtures.

```bash
MPLBACKEND=Agg python -m pytest tests/evaluation/ -q
# Skip e2e paths:
python -m pytest tests/evaluation/ -q -k 'not main_end_to_end and not against_committed'
```

## How it's consumed

- CI runs this subpackage.
- Pinned by Report **Appendix 8** as part of the 320-test suite.

## How to regenerate

```bash
MPLBACKEND=Agg python -m pytest tests/evaluation/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — tests/ index
- **↔ Siblings**: [`../data/`](../data/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../infra/`](../infra/), [`../scripts/`](../scripts/)
- **↓ Children**: none
