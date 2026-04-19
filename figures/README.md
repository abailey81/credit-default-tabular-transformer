# figures/

Committed PNG artefacts for the report and README. Every PNG here is
regenerable from the CSV/JSON data in `results/` via a single
`python -m src.<module>` invocation, and all are **deterministic — regenerate
bit-stably via `python -m src.infra.repro`**.

## Subfolders

| Folder | Contents | Produced by | Report section |
|---|---|---|---|
| [`eda/`](eda/) | 12 exploratory data figures (`fig01`–`fig13`, no fig12). | [`src/analysis/eda.py`](../src/analysis/eda.py) | Section 2 (Data understanding). |
| [`baseline/`](baseline/) | 5 Random Forest diagnostic plots. | [`src/baselines/random_forest.py`](../src/baselines/random_forest.py) | Section 3 (Baseline). |
| [`evaluation/`](evaluation/) | Transformer evaluation figures: comparison / calibration / fairness / uncertainty / significance / interpret. | `src/evaluation/*.py` | Section 4 + Appendix. |

## Regenerate everything

```bash
poetry run python -m src.infra.repro
```

Re-runs EDA, RF predictions, and the full evaluation suite (calibration,
fairness, uncertainty, significance, interpret, visualise) and rewrites every
PNG here. Expect all hashes to match the committed versions.

## Novelty cross-reference

- `evaluation/fairness/` → Novelty **N10** (subgroup fairness audit).
- `evaluation/uncertainty/` → Novelty **N11** (MC-dropout refusal curves).
- `evaluation/significance/` → Novelty **N12** (multi-seed significance).
- `evaluation/interpret/` → attention interpretability (not on the N1–N12
  register but referenced in the Appendix).
- Feature-group bias (N2) + temporal decay (N3) have no standalone figure —
  they appear as annotations on training / evaluation plots.
