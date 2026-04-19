# results/evaluation/uncertainty/

MC-dropout uncertainty artefacts — Section 4 / "Uncertainty" (Novelty **N11**).

## Files

| File | Shows |
|---|---|
| `mc_dropout.npz` | Arrays `probs_mc` of shape (T, N) — T=20 stochastic forward passes — plus `y_true` and the mean / std per sample. |
| `refuse_curve.csv` | Accuracy, coverage, AUC at each refusal threshold (top-k% highest entropy abstained). |
| `uncertainty_summary.json` | Structured summary: headline accuracy-at-coverage checkpoints + the entropy histogram bins. |

## Produced by

[`src/evaluation/uncertainty.py`](../../../src/evaluation/uncertainty.py) —
loads the seed_42 transformer checkpoint, enables dropout at inference,
runs T forward passes.

**Stochastic** outputs. Seeded with `torch` global seed **42** (matches the
seed_42 supervised run) so the committed NPZ regenerates bit-stably.

## Consumed by

- [`figures/evaluation/uncertainty/`](../../../figures/evaluation/uncertainty/) for the plots.
- Report **Section 4** / "Uncertainty-aware refusal".
- `docs/MODEL_CARD.md` § Limitations.

## Regenerate

```bash
poetry run python -m src.evaluation.uncertainty
```

Deterministic given the fixed seed — regenerates bit-stably via
`python -m src.infra.repro`.
