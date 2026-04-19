# results/evaluation/comparison/

Head-to-head table comparing every transformer run against the RF baseline
— Section 4 / "Head-to-head evaluation".

## Files

| File | Shows |
|---|---|
| `comparison_table.csv` | Machine-readable comparison table: one row per (run, metric). Columns include auc_roc / auc_pr / f1 / accuracy / precision / recall / specificity / cohen_kappa / ece / brier, with `mean ± std` aggregates for the multi-seed transformer. |
| `comparison_table.md` | Same table as GitHub-flavoured Markdown, drop-in for the report. |
| `evaluate_summary.json` | Structured summary: per-run metrics, aggregates, and the winner-per-metric record. |
| `head_to_head_summary.txt` | Plain-text narrative ("Transformer beats RF on AUC by +X …") used in README / CHANGELOG. |

## Produced by

[`src/evaluation/evaluate.py`](../../../src/evaluation/evaluate.py) — loads
every `results/transformer/seed_*/test_predictions.npz` plus
`results/baseline/rf/test_predictions.npz`.

## Consumed by

- Report **Section 4** (table 2 + narrative).
- Top-level [`README.md`](../../../README.md) pulls numbers from the `.md`
  version.

## Regenerate

```bash
poetry run python -m src.evaluation.evaluate
```

Deterministic — regenerates bit-stably via `python -m src.infra.repro`.
