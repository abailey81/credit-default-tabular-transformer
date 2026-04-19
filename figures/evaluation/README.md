# figures/evaluation/

Transformer evaluation figures — Section 4 of the report. Each subfolder
corresponds to one `src/evaluation/*.py` module.

## Subfolders

| Folder | Contents | Produced by | Novelty |
|---|---|---|---|
| [`comparison/`](comparison/) | ROC / PR / reliability / training curves / confusion matrices. | [`src/evaluation/visualise.py`](../../src/evaluation/visualise.py) + [`src/evaluation/evaluate.py`](../../src/evaluation/evaluate.py) | — |
| [`calibration/`](calibration/) | Reliability diagrams and ECE bar chart for the four calibrators. | [`src/evaluation/calibration.py`](../../src/evaluation/calibration.py) | — |
| [`fairness/`](fairness/) | Subgroup disparity bar + per-attribute reliability. | [`src/evaluation/fairness.py`](../../src/evaluation/fairness.py) | **N10** |
| [`uncertainty/`](uncertainty/) | MC-dropout entropy histogram + refusal curve. | [`src/evaluation/uncertainty.py`](../../src/evaluation/uncertainty.py) | **N11** |
| [`significance/`](significance/) | Pairwise p-value heatmap across the multi-seed runs. | [`src/evaluation/significance.py`](../../src/evaluation/significance.py) | **N12** |
| [`interpret/`](interpret/) | Attention rollout, per-head, class-conditional and RF-vs-attention importance. | [`src/evaluation/interpret.py`](../../src/evaluation/interpret.py) | Appendix (interpretability) |

## Consumed by

- Report **Section 4** (all six subfolders).
- `docs/MODEL_CARD.md` pulls the calibration and fairness figures.

## Regenerate

```bash
poetry run python -m src.infra.repro
```

All outputs are deterministic except `uncertainty/` (stochastic MC-dropout,
seed documented in its README).
