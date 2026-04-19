# `tests/evaluation/` — Evaluation subpackage tests

Covers all seven modules under `src/evaluation/`. Each file has a
synthetic unit path (always runs) and a committed-artefact e2e path
(skips gracefully when `results/transformer/seed_42/` or
`results/baseline/rf/` is missing).

## What's covered

| File                          | Subject |
|---|---|
| `test_calibration.py`         | Temperature / Platt / isotonic fits on synthetic data; ECE / MCE / Brier decomposition; reliability bin edges; e2e against committed run. |
| `test_fairness.py`            | Subgroup metrics (AUC / TPR / FPR / selection rate) on synthetic data; disparity table schema; e2e SEX / EDUCATION / MARRIAGE audit. |
| `test_uncertainty.py`         | `enable_dropout` flips only dropout modules; `mc_dropout_predict` restores eval mode; refuse curve monotonicity; e2e MC-dropout on a committed checkpoint. |
| `test_significance.py`        | McNemar / DeLong / paired bootstrap numerics; BH-FDR correction; Hanley-McNeil power; e2e paired test between transformer and RF. |
| `test_interpret.py`           | Attention rollout on synthetic tensors; per-feature importance ranking; e2e with committed `test_attn_weights.npz`. |
| `test_visualise.py`           | PNG output, reliability-bin edges, CLI argparse. |
| `test_evaluate_ensemble.py`   | `ensemble_run` aggregation, RF-full-predictions branch, comparison table schema. |

## Fixtures used

- `repo_root` — locate `results/`.
- `metadata` — where tests build categorical labels inline.

## Running

```bash
python -m pytest tests/evaluation/ -q
# Skip e2e paths:
python -m pytest tests/evaluation/ -q -k 'not main_end_to_end and not against_committed'
```

## Gotchas

- Set `MPLBACKEND=Agg` or `test_visualise.py` will try to open a GUI.
- E2E tests for `uncertainty` and `interpret` require
  `results/transformer/seed_42/best.pt`. They skip cleanly if missing.
