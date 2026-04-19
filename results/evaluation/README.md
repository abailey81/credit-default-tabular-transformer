# results/evaluation/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **evaluation/**

**Downstream evaluation artefacts** — for the Transformer runs in [`../transformer/`](../transformer/) plus the RF baseline in [`../baseline/rf/`](../baseline/rf/). Consumed by Section 4 (Experiments) and Appendix 8 (interpretability) of the report.

Five subfolders mirror Section 4's subsections (one per `src/evaluation/*.py` module) plus a loose `interpret.json` at the root. That one layout inconsistency — interpretability's numeric summary lives here rather than in an `interpret/` subfolder, while its PNGs live under `figures/evaluation/interpret/` — is intentional (too many existing cross-references to move) but flagged so browsers aren't surprised.

## What's here

| File / Subfolder | Contents |
|---|---|
| [`comparison/`](comparison/) | `comparison_table.{csv,md}`, `evaluate_summary.json`, `head_to_head_summary.txt`. |
| [`calibration/`](calibration/) | `calibration_metrics.csv`, `calibration_summary.json`. |
| [`fairness/`](fairness/) | `subgroup_metrics.csv`, `disparity_metrics.csv`, `fairness_summary.json` (**N10**). |
| [`uncertainty/`](uncertainty/) | `mc_dropout.npz`, `refuse_curve.csv`, `uncertainty_summary.json` (**N11**). |
| [`significance/`](significance/) | `pairwise_tests.csv`, `power_analysis.csv` (**N12**). |
| [`interpret.json`](interpret.json) | Attention-interpretability numeric summary (CLS feature importances, rollout stats, RF-vs-attn correlation). |

## How it was produced

Each subfolder is written by the corresponding `src/evaluation/*.py` module — see each child README. `interpret.json` comes from [`src/evaluation/interpret.py`](../../src/evaluation/interpret.py).

Deterministic where seeded — the uncertainty NPZ and significance bootstrap are stochastic, seeded; everything regenerates bit-stably via `python -m src.infra.repro`.

## How it's consumed

- [`../../figures/evaluation/`](../../figures/evaluation/) — plots rendered from these CSVs / JSONs / NPZs.
- Report **Section 4** (all subfolders) + **Appendix 8** (interpretability).
- [`docs/MODEL_CARD.md`](../../docs/MODEL_CARD.md) pulls calibration + fairness numbers.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../baseline/`](../baseline/), [`../transformer/`](../transformer/), [`../mtlm/`](../mtlm/), [`../repro/`](../repro/), [`../pipeline/`](../pipeline/)
- **↓ Children**: [`comparison/`](comparison/), [`calibration/`](calibration/), [`fairness/`](fairness/), [`uncertainty/`](uncertainty/), [`significance/`](significance/)
