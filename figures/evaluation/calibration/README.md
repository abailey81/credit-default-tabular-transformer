# figures/evaluation/calibration/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **calibration/**

**Reliability diagrams and calibration-error bar charts** — compares the raw transformer, post-temperature, post-Platt, post-isotonic, and the tuned Random Forest. Consumed by Section 4 / "Calibration" of the report and by the Model Card's calibration paragraph.

Calibration is the single biggest quality gap between the raw transformer (ECE ~0.26) and the tuned RF (ECE ~0.010). Platt scaling fit on val brings transformer ECE to 0.011 ± 0.003 across seeds — indistinguishable from RF — without touching AUC. These PNGs are the visual proof; the CSV backing them lives at [`../../../results/evaluation/calibration/`](../../../results/evaluation/calibration/).

## What's here

| File | Contents |
|---|---|
| [`calibration_reliability.png`](calibration_reliability.png) | Four reliability curves (one panel per calibrator) on the seed_42 test set. Diagonal = perfect calibration. |
| [`calibration_ece_bar.png`](calibration_ece_bar.png) | Bar chart of ECE across (run, calibrator) pairs — lowest bar = best calibrated. |

## How it was produced

[`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py) — fits calibrators on val-set logits, scores test set, emits both figures plus `results/evaluation/calibration/*`. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.calibration
```

## How it's consumed

- Report **Section 4** / "Calibration".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Calibration.
- [`../../../results/evaluation/calibration/`](../../../results/evaluation/calibration/) — numeric ground truth.

## How to regenerate

```bash
poetry run python -m src.evaluation.calibration
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/), [`../interpret/`](../interpret/)
- **↓ Children**: none
