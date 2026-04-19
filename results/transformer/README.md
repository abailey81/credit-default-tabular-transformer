# results/transformer/

One directory per supervised transformer run plus an aggregate summary.

## Per-run directories

| Directory | Seed | Notes |
|---|---|---|
| `seed_1/` | 1 | Primary replicate. |
| `seed_2/` | 2 | Second replicate. |
| `seed_42/` | 42 | Canonical / reference run — loaded by `uncertainty.py` and `interpret.py`. Only this run saves `test_attn_weights.npz` (gitignored, ~76 MB). |
| `seed_42_mtlm_finetune/` | 42 | Same seed but fine-tuned from the MTLM-pretrained encoder in [`../mtlm/run_42/`](../mtlm/run_42/) — Novelty **N4**. |

Each directory contains the same layout:

| File | Shows |
|---|---|
| `config.json` | Resolved hyperparameters, seed, git SHA, data split hashes. |
| `train_log.csv` | Per-epoch train / val loss + AUC + LR. |
| `train_metrics.json` / `val_metrics.json` / `test_metrics.json` | Final split metrics at the best checkpoint. |
| `train_predictions.npz` / `val_predictions.npz` / `test_predictions.npz` | `y_true`, `y_proba`, `y_pred` per split. |
| `best.pt` + `best.pt.weights` + `best.pt.meta.json` | Best checkpoint (full state dict + weights-only shard + metadata). |
| `test_attn_weights.npz` | (seed_1, seed_42 only) Raw attention weights for interpretability. Gitignored. |

## Aggregate

| File | Shows |
|---|---|
| `train_val_test_summary.csv` | One row per run with headline metrics across all three splits. Header: `run,train_acc,train_auc,train_f1,val_acc,val_auc,val_f1,test_acc,test_auc,test_f1`. |

## Produced by

- [`src/training/train.py`](../../src/training/train.py) — writes the
  per-seed `seed_X/` directories.
- [`src/evaluation/evaluate.py`](../../src/evaluation/evaluate.py) — writes
  the aggregate `train_val_test_summary.csv`.

## Consumed by

- Everything under [`../evaluation/`](../evaluation/) loads the per-seed
  `test_predictions.npz` + `val_predictions.npz`.
- Report **Section 4** (all tables + figures).

## Regenerate

```bash
poetry run python -m src.training.train --seed 1
poetry run python -m src.training.train --seed 2
poetry run python -m src.training.train --seed 42
poetry run python -m src.training.train --seed 42 --from-mtlm results/mtlm/run_42/encoder_pretrained.pt \
    --run-name seed_42_mtlm_finetune
poetry run python -m src.evaluation.evaluate   # writes train_val_test_summary.csv
```

Deterministic under fixed seeds — regenerates bit-stably via
`python -m src.infra.repro`.
