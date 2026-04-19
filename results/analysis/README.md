# results/analysis/

Aggregate EDA summary statistics — Section 2 of the report.

## Files

| File | Shows |
|---|---|
| `summary_statistics.csv` | Per-feature count / mean / std / min / quartiles / max for the cleaned dataset. |
| `summary_statistics.tex` | Same table typeset as LaTeX `\begin{tabular}` — drop-in for the report. |

## Produced by

[`src/analysis/eda.py`](../../src/analysis/eda.py) — the EDA pipeline that
also writes `figures/eda/fig01..fig13`.

## Consumed by

- Report **Section 2**, Table 1.
- [`notebooks/credit_default_eda.ipynb`](../../notebooks/credit_default_eda.ipynb) renders the CSV inline.

## Regenerate

```bash
poetry run python -m src.analysis.eda
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
