# results/

All numeric artefacts — CSVs, JSON summaries, NPZ prediction arrays, and
trained model checkpoints. Every file here is either produced by
`src/infra/repro.py` (most of them) or checked in as "ground truth" for
downstream consumers.

## Subfolders

| Folder | Contents | Produced by |
|---|---|---|
| [`analysis/`](analysis/) | EDA summary table (CSV + LaTeX). | [`src/analysis/eda.py`](../src/analysis/eda.py) |
| [`baseline/`](baseline/) | RF config, CV log, feature importances, aggregate metrics. | [`src/baselines/random_forest.py`](../src/baselines/random_forest.py) |
| [`baseline/rf/`](baseline/rf/) | RF test predictions + metrics (bit-stable regen). | [`src/baselines/rf_predictions.py`](../src/baselines/rf_predictions.py) |
| [`transformer/`](transformer/) | One `seed_X/` per supervised run; aggregate summary CSV. | [`src/training/train.py`](../src/training/train.py) |
| [`mtlm/`](mtlm/) | Pretrained MTLM encoder + training log (Novelty **N4**). | [`src/training/train_mtlm.py`](../src/training/train_mtlm.py) |
| [`evaluation/`](evaluation/) | Calibration / fairness / uncertainty / significance / comparison + `interpret.json`. | `src/evaluation/*.py` |
| [`repro/`](repro/) | Reproducibility report; `_scratch/` for bit-stability side-by-side. | [`src/infra/repro.py`](../src/infra/repro.py) |
| [`pipeline/`](pipeline/) | Per-stage logs from `scripts/run_all.*` invocations. | `scripts/run_all.{sh,ps1}` (driven by `src.infra.repro`). |

## Regenerate everything

```bash
poetry run python -m src.infra.repro
```

All committed outputs are deterministic except the MC-dropout NPZ in
`evaluation/uncertainty/` (stochastic, seeded) and the bootstrap resamples
under `evaluation/significance/` (stochastic, seeded). Everything
regenerates bit-stably under the committed seeds.
