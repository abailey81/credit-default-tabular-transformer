# results/evaluation/significance/

Multi-seed statistical-significance artefacts — Section 4 / "Significance"
(Novelty **N12**).

## Files

| File | Shows |
|---|---|
| `pairwise_tests.csv` | Pairwise paired-bootstrap results (ΔAUC, p-value, 95 % CI) for every (run_a, run_b) pair among the transformer seeds, the MTLM fine-tune, and RF. |
| `power_analysis.csv` | Estimated statistical power at the observed effect sizes, for the selected sample size (n_test) and B=1000 bootstrap resamples. |

## Produced by

[`src/evaluation/significance.py`](../../../src/evaluation/significance.py) —
consumes every `test_predictions.npz` under `results/transformer/` plus the
RF baseline.

**Stochastic** (bootstrap resamples) but seeded — bit-stable regeneration.

## Consumed by

- [`figures/evaluation/significance/`](../../../figures/evaluation/significance/) for the heatmap.
- Report **Section 4** / "Are the gains significant?".
- Appendix Table A.3.

## Regenerate

```bash
poetry run python -m src.evaluation.significance
```

Deterministic given the fixed seed — regenerates bit-stably via
`python -m src.infra.repro`.
