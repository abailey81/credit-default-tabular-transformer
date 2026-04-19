# figures/evaluation/

> **Breadcrumb**: [↑ repo root](../../) > [↑ figures](../) > **evaluation/**

**Transformer evaluation figures** — six subfolders, one per `src/evaluation/*.py` module. Consumed by Section 4 (Experiments/Results) of the report and by Appendix 8 (interpretability).

Each subfolder mirrors a Section 4 subsection of the report: comparison, calibration, fairness audit (N10), MC-dropout uncertainty (N11), multi-seed significance (N12), and attention interpretability. The load-bearing narrative is that the transformer matches or beats the tuned RF on every headline metric after post-hoc calibration, and the fairness / uncertainty / significance audits corroborate the delta.

## What's here

| Subfolder | Contents |
|---|---|
| [`comparison/`](comparison/) | ROC / PR / reliability / training curves / confusion matrices. |
| [`calibration/`](calibration/) | Reliability diagrams and ECE bar chart for the four calibrators. |
| [`fairness/`](fairness/) | Subgroup disparity bar + per-attribute reliability (**N10**). |
| [`uncertainty/`](uncertainty/) | MC-dropout entropy histogram + refusal curve (**N11**). |
| [`significance/`](significance/) | Pairwise p-value heatmap across the multi-seed runs (**N12**). |
| [`interpret/`](interpret/) | Attention rollout, per-head, class-conditional, RF-vs-attention importance (Appendix). |

## How it was produced

- `comparison/` from [`src/evaluation/visualise.py`](../../src/evaluation/visualise.py) + [`src/evaluation/evaluate.py`](../../src/evaluation/evaluate.py).
- `calibration/` from [`src/evaluation/calibration.py`](../../src/evaluation/calibration.py).
- `fairness/` from [`src/evaluation/fairness.py`](../../src/evaluation/fairness.py).
- `uncertainty/` from [`src/evaluation/uncertainty.py`](../../src/evaluation/uncertainty.py) — stochastic, seeded.
- `significance/` from [`src/evaluation/significance.py`](../../src/evaluation/significance.py) — stochastic, seeded.
- `interpret/` from [`src/evaluation/interpret.py`](../../src/evaluation/interpret.py).

All outputs are deterministic or stochastic-and-seeded; regenerate bit-stably via `python -m src.infra.repro`.

## How it's consumed

- Report **Section 4** (all six subfolders) + **Appendix 8** (interpretability).
- [`docs/MODEL_CARD.md`](../../docs/MODEL_CARD.md) pulls calibration + fairness figures.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

## Neighbours

- **↑ Parent**: [`../`](../) — figures/ index
- **↔ Siblings**: [`../eda/`](../eda/), [`../baseline/`](../baseline/)
- **↓ Children**: [`comparison/`](comparison/), [`calibration/`](calibration/), [`fairness/`](fairness/), [`uncertainty/`](uncertainty/), [`significance/`](significance/), [`interpret/`](interpret/)
