# figures/evaluation/comparison/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **comparison/**

**Side-by-side Transformer-vs-RF visual comparison** and multi-seed training diagnostics. Consumed by Section 4 / "Head-to-head evaluation" of the report and by the top-level README cover image.

These are the plots the reader sees first when they reach Section 4. Training curves show convergence behaviour across the three supervised seeds + the MTLM fine-tune; ROC/PR/confusion matrices show that the transformer matches or beats the tuned RF on every headline metric. Reliability diagrams motivate the Section 4 calibration subsection.

## What's here

| File | Contents |
|---|---|
| [`roc_curves_transformer.png`](roc_curves_transformer.png) | ROC curves for all transformer runs + the RF baseline on the test set. |
| [`pr_curves_transformer.png`](pr_curves_transformer.png) | Precision-recall curves on the test set (same runs). |
| [`confusion_matrices_transformer.png`](confusion_matrices_transformer.png) | Confusion matrix panel per seeded run at the tuned threshold. |
| [`reliability_diagrams.png`](reliability_diagrams.png) | Pre-calibration reliability curves per run — motivates the calibration subsection. |
| [`training_curves.png`](training_curves.png) | Per-epoch train / val loss + AUC across the three supervised seeds + MTLM fine-tune. |

## How it was produced

- [`src/evaluation/visualise.py`](../../../src/evaluation/visualise.py) — training curves, reliability diagrams.
- [`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) — ROC, PR, confusion matrices.

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.evaluate
poetry run python -m src.evaluation.visualise
```

## How it's consumed

- Report **Section 4** / "Head-to-head evaluation".
- Top-level [`README.md`](../../../README.md) pulls `training_curves.png` as the cover image.
- [`../../../results/evaluation/comparison/`](../../../results/evaluation/comparison/) contains the numeric table backing these plots.

## How to regenerate

```bash
poetry run python -m src.evaluation.evaluate
poetry run python -m src.evaluation.visualise
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/), [`../interpret/`](../interpret/)
- **↓ Children**: none
