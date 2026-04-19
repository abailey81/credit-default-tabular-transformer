# figures/evaluation/comparison/

Side-by-side Transformer-vs-RF visual comparison and multi-seed training
diagnostics — Section 4 / "Comparison" subsection.

## Files

| File | Shows |
|---|---|
| `roc_curves_transformer.png` | ROC curves for all transformer runs + the RF baseline on the test set. |
| `pr_curves_transformer.png` | Precision–recall curves on the test set (same runs). |
| `confusion_matrices_transformer.png` | Confusion matrix panel per seeded run at the tuned threshold. |
| `reliability_diagrams.png` | Pre-calibration reliability curves per run — motivates Section 4 calibration. |
| `training_curves.png` | Per-epoch train / val loss and AUC across the three supervised seeds + the MTLM fine-tune. |

## Produced by

- [`src/evaluation/visualise.py`](../../../src/evaluation/visualise.py) —
  training curves and reliability diagrams.
- [`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) — ROC,
  PR and confusion-matrix panels.

## Consumed by

- Report **Section 4** ("Head-to-head evaluation").
- README cover image pulls `training_curves.png`.

## Regenerate

```bash
poetry run python -m src.evaluation.evaluate
poetry run python -m src.evaluation.visualise
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
