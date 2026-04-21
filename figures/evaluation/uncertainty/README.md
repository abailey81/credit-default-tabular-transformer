# figures/evaluation/uncertainty/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **uncertainty/**

**MC-dropout uncertainty figures** — Novelty **N11**. Consumed by Section 4 / "Uncertainty-aware refusal" of the report and by the Model Card's Limitations section.

The entropy histogram shows predictive uncertainty is bimodal — confident correct predictions are very low-entropy, confident-but-wrong is the long right tail. The refusal curve shows accuracy rises monotonically as we abstain on the highest-entropy samples, which is the headline claim of N11. Raw probability tensors and bin counts live at [`../../../results/evaluation/uncertainty/`](../../../results/evaluation/uncertainty/).

## What's here

| File | Contents |
|---|---|
| [`uncertainty_entropy_hist.png`](uncertainty_entropy_hist.png) | Histogram of predictive entropy over the test set, split by correct / incorrect. |
| [`uncertainty_refuse_curve.png`](uncertainty_refuse_curve.png) | Accuracy vs fraction refused when abstaining on the highest-entropy samples. |

## How it was produced

[`src/evaluation/uncertainty.py`](../../../src/evaluation/uncertainty.py) — runs T=20 stochastic forward passes with dropout active, saves the per-sample probability tensor (`results/evaluation/uncertainty/mc_dropout.npz`) and emits both figures. Stochastic, seeded with `torch` global seed **42** — regenerates bit-stably under the committed seed.

```bash
poetry run python -m src.evaluation.uncertainty
```

## How it's consumed

- Report **Section 4** / "Uncertainty-aware refusal".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Limitations (refusal policy).
- [`../../../results/evaluation/uncertainty/`](../../../results/evaluation/uncertainty/) — the NPZ + CSV backing these plots.

## How to regenerate

```bash
poetry run python -m src.evaluation.uncertainty
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../significance/`](../significance/), [`../interpret/`](../interpret/)
- **↓ Children**: none
