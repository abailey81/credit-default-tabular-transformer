# figures/evaluation/significance/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ figures](../../) > [↑ evaluation](../) > **significance/**

**Multi-seed statistical-significance figure** — Novelty **N12**. Consumed by Section 4 / "Are the gains significant?" of the report and by Appendix Table A.3.

Without this plot the claim "transformer beats RF" cannot distinguish real signal from seed noise. The heatmap reports paired-bootstrap p-values (B=1000 resamples by default) across the three supervised seeds, the MTLM fine-tune, and the RF baseline — darker cells correspond to smaller p-values. The supporting CSVs + power analysis live at [`../../../results/evaluation/significance/`](../../../results/evaluation/significance/).

## What's here

| File | Contents |
|---|---|
| [`significance_pvalue_heatmap.png`](significance_pvalue_heatmap.png) | Pairwise p-value matrix (paired bootstrap on AUC) across the three supervised seeds, the MTLM fine-tune, and the RF baseline. Darker = smaller p. |

## How it was produced

[`src/evaluation/significance.py`](../../../src/evaluation/significance.py) — paired bootstrap on the per-sample test probabilities, emits the heatmap plus `results/evaluation/significance/*`. Stochastic (bootstrap resamples), seeded — regenerates bit-stably under the committed seed.

```bash
poetry run python -m src.evaluation.significance
```

## How it's consumed

- Report **Section 4** / "Are the gains significant?".
- Report **Appendix 8** Table A.3.
- [`../../../results/evaluation/significance/`](../../../results/evaluation/significance/) — numeric p-values + power analysis.

## How to regenerate

```bash
poetry run python -m src.evaluation.significance
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/), [`../interpret/`](../interpret/)
- **↓ Children**: none
