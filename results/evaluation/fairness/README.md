# results/evaluation/fairness/

Subgroup fairness audit artefacts — Section 4 / "Fairness" (Novelty **N10**).

Protected attributes: SEX, EDUCATION, MARRIAGE (per the original UCI encoding
plus small regrouping described in `src/data/preprocessing.py`).

## Files

| File | Shows |
|---|---|
| `subgroup_metrics.csv` | Per-(attribute, subgroup) AUC / F1 / FPR / FNR / ECE / support count on the test set. |
| `disparity_metrics.csv` | Max–min and ratio disparities per attribute for every metric (headline numbers). |
| `fairness_summary.json` | Structured summary used by the plot backend and the model card. |

## Produced by

[`src/evaluation/fairness.py`](../../../src/evaluation/fairness.py) — reads
`results/transformer/seed_*/test_predictions.npz` and the raw test-split
attributes.

## Consumed by

- [`figures/evaluation/fairness/`](../../../figures/evaluation/fairness/) for the plots.
- Report **Section 4** / "Fairness audit".
- `docs/MODEL_CARD.md` § Intended-use / limitations.

## Regenerate

```bash
poetry run python -m src.evaluation.fairness
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
