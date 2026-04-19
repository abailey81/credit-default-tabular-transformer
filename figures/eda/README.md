# figures/eda/

> **Breadcrumb**: [↑ repo root](../../) > [↑ figures](../) > **eda/**

**Exploratory Data Analysis figures** — 12 PNGs for the UCI Default of Credit Card Clients dataset, consumed by Section 2 (Data Exploration) of the report.

The gap at fig12 is intentional — that plot was a duplicate of fig11 and was dropped during Phase 3; the numbering was preserved to keep earlier write-ups' cross-references valid. All figures are rendered from the cleaned dataset under [`../../data/processed/`](../../data/processed/), so they are leak-free views of the train split (never touches raw UCI Excel).

## What's here

| File | Contents |
|---|---|
| [`fig01_class_distribution.png`](fig01_class_distribution.png) | Target prior: ~22 % defaulters vs 78 % non-defaulters. |
| [`fig02_categorical_by_target.png`](fig02_categorical_by_target.png) | SEX / EDUCATION / MARRIAGE default rates. |
| [`fig03_numerical_distributions.png`](fig03_numerical_distributions.png) | LIMIT_BAL, AGE, and BILL/PAY_AMT amounts. |
| [`fig04_pay_status_analysis.png`](fig04_pay_status_analysis.png) | Delinquency code distribution per PAY_0..PAY_6. |
| [`fig05_temporal_trajectories.png`](fig05_temporal_trajectories.png) | Per-customer BILL_AMT and PAY_AMT across 6 months. |
| [`fig06_utilisation_analysis.png`](fig06_utilisation_analysis.png) | BILL_AMT / LIMIT_BAL ratio vs default rate. |
| [`fig07_correlation_heatmap.png`](fig07_correlation_heatmap.png) | Pairwise Pearson correlations of the 23 raw features. |
| [`fig08_feature_target_association.png`](fig08_feature_target_association.png) | Mutual-information / point-biserial score per feature. |
| [`fig09_bill_amt_autocorrelation.png`](fig09_bill_amt_autocorrelation.png) | Lag-1..lag-5 autocorrelation of BILL_AMT (motivates N3 temporal encoding). |
| [`fig10_feature_interactions.png`](fig10_feature_interactions.png) | 2D heatmaps of top interaction pairs. |
| [`fig11_pay_transitions.png`](fig11_pay_transitions.png) | Markov transition matrix between PAY_m and PAY_{m+1}. |
| [`fig13_repayment_ratio.png`](fig13_repayment_ratio.png) | PAY_AMT / BILL_AMT ratio, capped and log-transformed. |

## How it was produced

[`src/analysis/eda.py`](../../src/analysis/eda.py). Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.analysis.eda
```

## How it's consumed

- Report **Section 2** (all 12 figures).
- [`notebooks/01_exploratory_data_analysis.ipynb`](../../notebooks/01_exploratory_data_analysis.ipynb) mirrors these plots inline.

## How to regenerate

```bash
poetry run python -m src.analysis.eda
```

## Neighbours

- **↑ Parent**: [`../`](../) — figures/ index
- **↔ Siblings**: [`../baseline/`](../baseline/), [`../evaluation/`](../evaluation/)
- **↓ Children**: none
