# results/mtlm/

Masked Tabular Language Model pretraining artefacts — Novelty **N4**.

## Per-run directories

| Directory | Seed | Notes |
|---|---|---|
| `run_42/` | 42 | The one pretraining run used by the committed MTLM fine-tune (`../transformer/seed_42_mtlm_finetune/`). |

Each directory contains:

| File | Shows |
|---|---|
| `config.json` | Resolved MTLM hyperparameters (mask ratio, loss weights, optimizer), seed, git SHA. |
| `pretrain_log.csv` | Per-epoch pretraining loss (numeric + categorical reconstruction heads). |
| `encoder_pretrained.pt` | **Encoder-only state dict** — the artefact `train.py --from-mtlm` consumes to initialise fine-tuning. |
| `best.pt` + `best.pt.weights` + `best.pt.meta.json` | Full best-checkpoint including the reconstruction heads (kept for ablations / inspection). |

## Produced by

[`src/training/train_mtlm.py`](../../src/training/train_mtlm.py) — masks a
random subset of feature tokens at each step and learns to reconstruct them;
the resulting encoder is then fine-tuned for classification.

## Consumed by

- [`src/training/train.py`](../../src/training/train.py) via
  `--from-mtlm <path to encoder_pretrained.pt>` — produces the
  `seed_42_mtlm_finetune/` run.
- Report **Section 4** / "MTLM pretraining" — the AUC delta vs seed_42 from
  scratch is the headline N4 result.
- CHANGELOG Phase 6A entry.

## Regenerate

```bash
poetry run python -m src.training.train_mtlm --seed 42 --run-name run_42
```

Deterministic under the fixed seed — regenerates bit-stably via
`python -m src.infra.repro` (the repro harness invokes this if the
`encoder_pretrained.pt` is missing).
