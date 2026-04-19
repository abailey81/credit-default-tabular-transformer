# results/transformer/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **transformer/**

**Supervised transformer run artefacts** — one directory per seed plus an aggregate summary. Consumed by Section 4 (Experiments) of the report and by every module under [`../evaluation/`](../evaluation/).

Three supervised seeds (1, 2, 42) + one MTLM fine-tuned seed (`seed_42_mtlm_finetune`, **N4**). seed_42 is the canonical reference run: it is the one loaded by `uncertainty.py` and `interpret.py`, and the only run where `test_attn_weights.npz` is saved (gitignored, ~76 MB). Every per-seed directory has the same file layout.

## What's here

| Directory / File | Contents |
|---|---|
| [`seed_1/`](seed_1/) | Seed 1 run — primary replicate. Contains `config.json`, `train_log.csv`, `{train,val,test}_metrics.json`, `{train,val,test}_predictions.npz`, `best.pt` + `best.pt.weights` + `best.pt.meta.json`. |
| [`seed_2/`](seed_2/) | Seed 2 run — second replicate. Same layout as `seed_1/`. |
| [`seed_42/`](seed_42/) | Seed 42 — canonical reference run; also emits `test_attn_weights.npz` (gitignored). |
| [`seed_42_mtlm_finetune/`](seed_42_mtlm_finetune/) | Seed 42 fine-tuned from the MTLM-pretrained encoder at [`../mtlm/run_42/`](../mtlm/run_42/) — **N4**. |
| [`train_val_test_summary.csv`](train_val_test_summary.csv) | One row per run with headline metrics across all three splits. Columns: `run,train_acc,train_auc,train_f1,val_acc,val_auc,val_f1,test_acc,test_auc,test_f1`. |

## How it was produced

- [`src/training/train.py`](../../src/training/train.py) writes the per-seed `seed_X/` directories.
- [`src/evaluation/evaluate.py`](../../src/evaluation/evaluate.py) writes the aggregate `train_val_test_summary.csv`.

Deterministic under fixed seeds — regenerates bit-stably via `python -m src.infra.repro`.

```bash
poetry run python -m src.training.train --seed 1
poetry run python -m src.training.train --seed 2
poetry run python -m src.training.train --seed 42
poetry run python -m src.training.train --seed 42 --from-mtlm results/mtlm/run_42/encoder_pretrained.pt \
    --run-name seed_42_mtlm_finetune
poetry run python -m src.evaluation.evaluate   # writes train_val_test_summary.csv
```

## How it's consumed

- Everything under [`../evaluation/`](../evaluation/) loads per-seed `test_predictions.npz` + `val_predictions.npz`.
- Report **Section 4** (all tables + figures).
- [`../../figures/evaluation/`](../../figures/evaluation/) renders plots from these predictions.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../baseline/`](../baseline/), [`../mtlm/`](../mtlm/), [`../evaluation/`](../evaluation/), [`../repro/`](../repro/), [`../pipeline/`](../pipeline/)
- **↓ Children**: [`seed_1/`](seed_1/), [`seed_2/`](seed_2/), [`seed_42/`](seed_42/), [`seed_42_mtlm_finetune/`](seed_42_mtlm_finetune/)
