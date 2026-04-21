# results/evaluation/uncertainty/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ evaluation](../) > **uncertainty/**

**MC-dropout uncertainty artefacts** — Novelty **N11**. Consumed by Section 4 / "Uncertainty-aware refusal" of the report and by the Model Card's Limitations section.

T=20 stochastic forward passes with dropout active are saved per sample, along with the refusal curve (accuracy as fraction-refused rises from 0 → 1). The MC-dropout NPZ is the raw probability tensor that every downstream uncertainty claim derives from.

## What's here

| File | Contents |
|---|---|
| [`mc_dropout.npz`](mc_dropout.npz) | `probs_mc` shape (T=20, N) stochastic forward passes + `y_true` + per-sample mean / std. |
| [`refuse_curve.csv`](refuse_curve.csv) | Accuracy, coverage, AUC at each refusal threshold (top-k % highest entropy abstained). |
| [`uncertainty_summary.json`](uncertainty_summary.json) | Headline accuracy-at-coverage checkpoints + entropy histogram bins. |

## How it was produced

[`src/evaluation/uncertainty.py`](../../../src/evaluation/uncertainty.py) — loads the seed_42 transformer checkpoint, enables dropout at inference, runs T forward passes. Stochastic, seeded with `torch` global seed **42** (matches the seed_42 supervised run) — regenerates bit-stably under the committed seed.

```bash
poetry run python -m src.evaluation.uncertainty
```

## How it's consumed

- Report **Section 4** / "Uncertainty-aware refusal".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Limitations.
- [`../../../figures/evaluation/uncertainty/`](../../../figures/evaluation/uncertainty/) — plots rendered from this data.

## How to regenerate

```bash
poetry run python -m src.evaluation.uncertainty
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../significance/`](../significance/)
- **↓ Children**: none
