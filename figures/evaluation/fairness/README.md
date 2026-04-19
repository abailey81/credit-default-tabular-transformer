# figures/evaluation/fairness/

Subgroup fairness audit artefacts — Section 4 / "Fairness" subsection
(Novelty **N10**). Protected attributes: SEX, EDUCATION, MARRIAGE.

## Files

| File | Shows |
|---|---|
| `fairness_disparity.png` | Max–min disparity in AUC, F1, FPR, FNR across subgroups per attribute. |
| `fairness_reliability_sex.png` | Reliability curve per SEX subgroup. |
| `fairness_reliability_education.png` | Reliability curve per EDUCATION bucket. |
| `fairness_reliability_marriage.png` | Reliability curve per MARRIAGE bucket. |

## Produced by

[`src/evaluation/fairness.py`](../../../src/evaluation/fairness.py) — consumes
`results/transformer/seed_*/test_predictions.npz` and emits both figures plus
`results/evaluation/fairness/*`.

## Consumed by

- Report **Section 4** / "Fairness audit".
- `docs/MODEL_CARD.md` § Intended-use / limitations.

## Regenerate

```bash
poetry run python -m src.evaluation.fairness
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
