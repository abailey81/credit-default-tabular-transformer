# figures/evaluation/fairness/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **fairness/**

**Subgroup fairness audit artefacts** — Novelty **N10**. Consumed by Section 4 / "Fairness audit" of the report and by the Model Card's Intended-use / Limitations section. Protected attributes: SEX, EDUCATION, MARRIAGE.

The audit surfaces whether headline metrics hide subgroup-level quality gaps. The disparity bar plot is the headline visual; the three per-attribute reliability plots expose where calibration drifts subgroup-by-subgroup (key call-out: male vs female reliability differ by less than 1 % ECE after calibration).

## What's here

| File | Contents |
|---|---|
| [`fairness_disparity.png`](fairness_disparity.png) | Max-min disparity in AUC, F1, FPR, FNR across subgroups per attribute. |
| [`fairness_reliability_sex.png`](fairness_reliability_sex.png) | Reliability curve per SEX subgroup. |
| [`fairness_reliability_education.png`](fairness_reliability_education.png) | Reliability curve per EDUCATION bucket. |
| [`fairness_reliability_marriage.png`](fairness_reliability_marriage.png) | Reliability curve per MARRIAGE bucket. |

## How it was produced

[`src/evaluation/fairness.py`](../../../src/evaluation/fairness.py) — consumes `results/transformer/seed_*/test_predictions.npz` and emits both figures plus `results/evaluation/fairness/*`. Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.evaluation.fairness
```

## How it's consumed

- Report **Section 4** / "Fairness audit".
- [`docs/MODEL_CARD.md`](../../../docs/MODEL_CARD.md) §Intended-use / Limitations.
- [`../../../results/evaluation/fairness/`](../../../results/evaluation/fairness/) — numeric disparity tables.

## How to regenerate

```bash
poetry run python -m src.evaluation.fairness
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../uncertainty/`](../uncertainty/), [`../significance/`](../significance/), [`../interpret/`](../interpret/)
- **↓ Children**: none
