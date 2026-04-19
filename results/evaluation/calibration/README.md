# results/evaluation/calibration/

Post-hoc calibration metrics (temperature, Platt, isotonic) — Section 4 /
"Calibration".

## Files

| File | Shows |
|---|---|
| `calibration_metrics.csv` | Long-format table: one row per (run, calibrator) with ECE, MCE, Brier, log-loss, and the fitted scalar(s). |
| `calibration_summary.json` | Structured best-calibrator-per-run selection + reliability-bin counts for the plot backend. |

## Produced by

[`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py) —
fits calibrators on the held-out val split (val_predictions.npz) and scores
test (test_predictions.npz) for every transformer run + RF.

## Consumed by

- [`figures/evaluation/calibration/`](../../../figures/evaluation/calibration/) for the plots.
- Report **Section 4** / "Calibration".
- `docs/MODEL_CARD.md` § Calibration paragraph.

## Regenerate

```bash
poetry run python -m src.evaluation.calibration
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
