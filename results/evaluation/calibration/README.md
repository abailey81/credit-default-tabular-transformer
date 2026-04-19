# results/evaluation/calibration/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ evaluation](../) > **calibration/**

**Post-hoc calibrator metrics** — ECE / MCE / Brier / log-loss produced by the four-way calibrator sweep. Consumed by Section 4 / "Calibration" of the report and by the Model Card's calibration paragraph.

Calibration is the single biggest quality gap between the raw transformer (ECE ~0.26) and the tuned RF (ECE ~0.010). Platt scaling fit on the validation split brings transformer ECE to 0.011 ± 0.003 across seeds — indistinguishable from RF — without touching AUC. These CSVs are the ground truth for every calibration claim in the report.

## What's here

| File | Contents |
|---|---|
| [`calibration_metrics.csv`](calibration_metrics.csv) | Long-format: one row per (run, calibrator) with ECE, MCE, Brier, log-loss, and the fitted scalar(s). |
| [`calibration_summary.json`](calibration_summary.json) | Structured best-calibrator-per-run selection + reliability-bin counts for the plot backend. |

## How it was produced

[`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py) — fits calibrators on the held-out val split (`val_predictions.npz`) and scores test (`test_predictions.npz`) for every transformer run + RF. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.calibration
```

~15 s. Part of the 7-check reproducibility harness.

## How it's consumed

- Report **Section 4** / "Calibration".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Calibration.
- [`../../../figures/evaluation/calibration/`](../../../figures/evaluation/calibration/) — reliability diagrams rendered from this data.

## How to regenerate

```bash
poetry run python -m src.evaluation.calibration
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/)
- **↓ Children**: none
