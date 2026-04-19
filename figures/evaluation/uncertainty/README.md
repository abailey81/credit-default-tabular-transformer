# figures/evaluation/uncertainty/

MC-dropout uncertainty figures — Section 4 / "Uncertainty" subsection
(Novelty **N11**).

## Files

| File | Shows |
|---|---|
| `uncertainty_entropy_hist.png` | Histogram of predictive entropy over the test set, split by correct / incorrect. |
| `uncertainty_refuse_curve.png` | Accuracy vs fraction refused when abstaining on the highest-entropy samples. |

## Produced by

[`src/evaluation/uncertainty.py`](../../../src/evaluation/uncertainty.py) —
runs T=20 stochastic forward passes with dropout active, saves the
per-sample probability tensor (`results/evaluation/uncertainty/mc_dropout.npz`)
and emits both figures.

**Stochastic**: results depend on the dropout RNG. Seeded with `torch`
global seed **42** (the seed_42 transformer run). Re-running yields identical
arrays because the sampler is seeded per call.

## Consumed by

- Report **Section 4** / "Uncertainty-aware refusal".
- `docs/MODEL_CARD.md` § Limitations (refusal policy).

## Regenerate

```bash
poetry run python -m src.evaluation.uncertainty
```

Deterministic given the fixed seed — regenerates bit-stably via
`python -m src.infra.repro`.
