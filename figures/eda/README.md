# figures/eda/

Exploratory Data Analysis figures for the UCI Default of Credit Card Clients
dataset — Section 2 of the report.

## Files

| File | Shows |
|---|---|
| `fig01_class_distribution.png` | Target prior: ~22 % defaulters vs 78 % non-defaulters. |
| `fig02_categorical_by_target.png` | SEX / EDUCATION / MARRIAGE default rates. |
| `fig03_numerical_distributions.png` | LIMIT_BAL, AGE and the BILL/PAY_AMT amounts. |
| `fig04_pay_status_analysis.png` | Delinquency code distribution per PAY_0..PAY_6. |
| `fig05_temporal_trajectories.png` | Per-customer BILL_AMT and PAY_AMT across the 6 months. |
| `fig06_utilisation_analysis.png` | BILL_AMT / LIMIT_BAL ratio vs default rate. |
| `fig07_correlation_heatmap.png` | Pairwise Pearson correlations of the 23 raw features. |
| `fig08_feature_target_association.png` | Mutual-information / point-biserial score per feature. |
| `fig09_bill_amt_autocorrelation.png` | Lag-1..lag-5 autocorrelation of BILL_AMT (motivates temporal encoding, N3). |
| `fig10_feature_interactions.png` | 2D heatmaps of top interaction pairs. |
| `fig11_pay_transitions.png` | Markov transition matrix between PAY_m and PAY_{m+1}. |
| `fig13_repayment_ratio.png` | PAY_AMT / BILL_AMT ratio, capped and log-transformed. |

**Note the gap at fig12** — it was a duplicate of fig11 removed in Phase 3; the
numbering was kept to preserve cross-references in earlier write-ups.

## Produced by

[`src/analysis/eda.py`](../../src/analysis/eda.py) — runs from the cleaned
dataset in `data/processed/`.

## Consumed by

- Report **Section 2** (all 12 figures).
- [`notebooks/credit_default_eda.ipynb`](../../notebooks/credit_default_eda.ipynb) mirrors these plots inline.

## Regenerate

```bash
poetry run python -m src.analysis.eda
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
