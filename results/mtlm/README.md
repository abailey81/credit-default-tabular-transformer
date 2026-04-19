# results/mtlm/

> **Breadcrumb**: [↑ repo root](../../) > [↑ results](../) > **mtlm/**

**Masked Tabular Language Model pretraining artefacts** — Novelty **N4**. Consumed by [`../transformer/seed_42_mtlm_finetune/`](../transformer/seed_42_mtlm_finetune/) (the fine-tuned classifier) and by Section 4 / "MTLM pretraining" of the report.

MTLM pretraining masks a random subset of feature tokens per step and trains the encoder to reconstruct them. The pretrained encoder-only state dict is then consumed by `train.py --from-mtlm` for fine-tuning. The headline N4 result is the AUC delta between seed_42 from scratch and seed_42_mtlm_finetune.

## What's here

| Directory | Contents |
|---|---|
| [`run_42/`](run_42/) | Seed 42 pretraining run. Contains `config.json`, `pretrain_log.csv`, `encoder_pretrained.pt` (encoder-only state dict — the artefact `train.py --from-mtlm` consumes), `best.pt` + `best.pt.weights` + `best.pt.meta.json` (full best checkpoint including reconstruction heads, kept for ablations). |

## How it was produced

[`src/training/train_mtlm.py`](../../src/training/train_mtlm.py). Deterministic under the fixed seed — regenerates bit-stably via `python -m src.infra.repro` (the repro harness invokes this if `encoder_pretrained.pt` is missing).

```bash
poetry run python -m src.training.train_mtlm --seed 42 --run-name run_42
```

## How it's consumed

- [`src/training/train.py`](../../src/training/train.py) via `--from-mtlm <path to encoder_pretrained.pt>` — produces `../transformer/seed_42_mtlm_finetune/`.
- Report **Section 4** / "MTLM pretraining" — the AUC delta vs seed_42 from scratch is the headline N4 result.
- CHANGELOG Phase 6A entry.

## How to regenerate

```bash
poetry run python -m src.training.train_mtlm --seed 42 --run-name run_42
```

## Neighbours

- **↑ Parent**: [`../`](../) — results/ index
- **↔ Siblings**: [`../analysis/`](../analysis/), [`../baseline/`](../baseline/), [`../transformer/`](../transformer/), [`../evaluation/`](../evaluation/), [`../repro/`](../repro/), [`../pipeline/`](../pipeline/)
- **↓ Children**: [`run_42/`](run_42/)
