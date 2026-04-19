# figures/evaluation/significance/

Multi-seed statistical-significance figures — Section 4 / "Significance"
subsection (Novelty **N12**).

## Files

| File | Shows |
|---|---|
| `significance_pvalue_heatmap.png` | Pairwise p-value matrix (paired bootstrap on AUC) across the three supervised seeds, the MTLM fine-tune, and the RF baseline. Darker = smaller p. |

## Produced by

[`src/evaluation/significance.py`](../../../src/evaluation/significance.py) —
runs paired bootstrap (B=1000 resamples by default) on the per-sample test
probabilities and emits the heatmap plus `results/evaluation/significance/*`.

## Consumed by

- Report **Section 4** / "Are the gains significant?".
- Table in the Appendix referenced as "Tab. A.3".

## Regenerate

```bash
poetry run python -m src.evaluation.significance
```

**Stochastic** (bootstrap resamples) but seeded — regenerates bit-stably via
`python -m src.infra.repro`.
