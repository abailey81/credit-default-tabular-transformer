# results/evaluation/significance/

> **Breadcrumb**: [↑ repo root](../../../) > [↑ results](../../) > [↑ evaluation](../) > **significance/**

**Multi-seed statistical-significance artefacts** — Novelty **N12**. Consumed by Section 4 / "Are the gains significant?" of the report and by Appendix 8 Table A.3.

Paired-bootstrap tests (B=1000 resamples by default) across the three supervised seeds, the MTLM fine-tune, and the RF baseline — the load-bearing "is this beyond seed noise?" evidence. Power analysis confirms the sample size (n_test) actually detects the observed effect sizes.

## What's here

| File | Contents |
|---|---|
| [`pairwise_tests.csv`](pairwise_tests.csv) | Pairwise paired-bootstrap results (ΔAUC, p-value, 95 % CI) for every (run_a, run_b) pair. |
| [`power_analysis.csv`](power_analysis.csv) | Estimated statistical power at the observed effect sizes, for the selected sample size (n_test) and B=1000 bootstrap resamples. |

## How it was produced

[`src/evaluation/significance.py`](../../../src/evaluation/significance.py) — consumes every `test_predictions.npz` under `results/transformer/` plus the RF baseline. Stochastic (bootstrap resamples), seeded — regenerates bit-stably under the committed seed.

```bash
poetry run python -m src.evaluation.significance
```

## How it's consumed

- Report **Section 4** / "Are the gains significant?".
- Report **Appendix 8** Table A.3.
- [`../../../figures/evaluation/significance/`](../../../figures/evaluation/significance/) — heatmap rendered from `pairwise_tests.csv`.

## How to regenerate

```bash
poetry run python -m src.evaluation.significance
```

## Neighbours

- **↑ Parent**: [`../`](../) — evaluation/ index
- **↔ Siblings**: [`../comparison/`](../comparison/), [`../calibration/`](../calibration/), [`../fairness/`](../fairness/), [`../uncertainty/`](../uncertainty/)
- **↓ Children**: none
