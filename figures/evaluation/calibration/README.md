# figures/evaluation/calibration/

Reliability diagrams and calibration-error bar charts comparing the raw
transformer, post-temperature, post-Platt, post-isotonic, and the tuned
Random Forest — Section 4 / "Calibration" subsection.

## Files

| File | Shows |
|---|---|
| `calibration_reliability.png` | Four reliability curves (one panel per calibrator) on the seed_42 test set. Diagonal = perfect calibration. |
| `calibration_ece_bar.png` | Bar chart of ECE across (run, calibrator) pairs — lowest bar = best calibrated. |

## Produced by

[`src/evaluation/calibration.py`](../../../src/evaluation/calibration.py) —
fits calibrators on val-set logits, scores test set, and emits both figures
plus `results/evaluation/calibration/*`.

## Consumed by

- Report **Section 4** / "Calibration".
- `docs/MODEL_CARD.md` § Calibration.

## Regenerate

```bash
poetry run python -m src.evaluation.calibration
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
