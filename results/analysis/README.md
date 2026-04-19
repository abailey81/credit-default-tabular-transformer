# results/analysis/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **analysis/**

**Aggregate EDA summary statistics** — consumed by Section 2 (Data Exploration), Table 1 of the report and by [`notebooks/01_exploratory_data_analysis.ipynb`](../../notebooks/01_exploratory_data_analysis.ipynb).

Per-feature summary statistics backing every distributional claim in Section 2: count, mean, std, min, quartiles, and max across the cleaned training split. The LaTeX version is a drop-in `\begin{tabular}` for the report PDF.

## What's here

| File | Contents |
|---|---|
| [`summary_statistics.csv`](summary_statistics.csv) | Per-feature count / mean / std / min / quartiles / max for the cleaned dataset. |
| [`summary_statistics.tex`](summary_statistics.tex) | Same table typeset as LaTeX — drop-in for the report. |

## How it was produced

[`src/analysis/eda.py`](../../src/analysis/eda.py) — the EDA pipeline that also writes [`../../figures/eda/`](../../figures/eda/). Deterministic — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.analysis.eda
```

## How it's consumed

- Report **Section 2**, Table 1.
- [`notebooks/01_exploratory_data_analysis.ipynb`](../../notebooks/01_exploratory_data_analysis.ipynb) renders the CSV inline.
- [`../../figures/eda/`](../../figures/eda/) — the visual counterpart (same EDA module writes both).

## How to regenerate

```bash
poetry run python -m src.analysis.eda
```

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../baseline/`](../baseline/), [`../transformer/`](../transformer/), [`../mtlm/`](../mtlm/), [`../evaluation/`](../evaluation/), [`../repro/`](../repro/), [`../pipeline/`](../pipeline/)
- **↓ Children**: none
