# results/evaluation/

All downstream evaluation artefacts for the Transformer runs in
[`../transformer/`](../transformer/) plus the RF baseline in
[`../baseline/rf/`](../baseline/rf/).

## Subfolders

| Folder | Contents | Produced by | Novelty |
|---|---|---|---|
| [`comparison/`](comparison/) | `comparison_table.{csv,md}`, `evaluate_summary.json`, `head_to_head_summary.txt`. | [`src/evaluation/evaluate.py`](../../src/evaluation/evaluate.py) | — |
| [`calibration/`](calibration/) | `calibration_metrics.csv`, `calibration_summary.json`. | [`src/evaluation/calibration.py`](../../src/evaluation/calibration.py) | — |
| [`fairness/`](fairness/) | `subgroup_metrics.csv`, `disparity_metrics.csv`, `fairness_summary.json`. | [`src/evaluation/fairness.py`](../../src/evaluation/fairness.py) | **N10** |
| [`uncertainty/`](uncertainty/) | `mc_dropout.npz`, `refuse_curve.csv`, `uncertainty_summary.json`. | [`src/evaluation/uncertainty.py`](../../src/evaluation/uncertainty.py) | **N11** |
| [`significance/`](significance/) | `pairwise_tests.csv`, `power_analysis.csv`. | [`src/evaluation/significance.py`](../../src/evaluation/significance.py) | **N12** |

## Loose file

| File | Shows | Produced by |
|---|---|---|
| `interpret.json` | Attention-interpretability numeric summary (CLS feature importances, rollout stats, RF-vs-attn correlation). | [`src/evaluation/interpret.py`](../../src/evaluation/interpret.py) |

**Note the layout inconsistency**: `interpret.json` lives at the
`results/evaluation/` root rather than in an `interpret/` subfolder, while
its PNGs live under `figures/evaluation/interpret/`. This is intentional —
too many existing cross-references to move it — but flagged here so browsers
aren't surprised.

## Consumed by

- Report **Section 4** (all subfolders) + Appendix interpretability.
- `docs/MODEL_CARD.md` pulls calibration + fairness numbers.

## Regenerate

```bash
poetry run python -m src.infra.repro
```

Deterministic where seeded (uncertainty + significance use fixed seeds);
regenerates bit-stably.
