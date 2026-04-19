# src/infra/

> **Breadcrumb**: [↑ repo root](../../) > [↑ src](../) > **infra/**

**Reproducibility gate** — the "trust-but-verify" layer. Regenerates every committed derivative artefact and diffs it byte-for-byte (or with a tight numerical tolerance) against what is on disk. If any check fails the repo is not trustworthy for the claims in the report. Consumed by Appendix 8 (Reproducibility) and gated in CI.

`repro.py` imports from every other subpackage — it reads everything, writes only its own JSON. Keep that direction; nothing else should import `src.infra`. The `split_hashes_match` check probes `data/processed/splits/` first and falls back to `data/processed/` (pre-2026 layout) so it round-trips regardless of which layout is on disk.

## What's here

| File | Contents |
|---|---|
| [`repro.py`](repro.py) | Runs 7 independent checks: `artefacts_exist`, `transformer_run_files`, `split_hashes_match`, `python_pins`, `git_clean`, `rf_predictions_regenerate`, `evaluate_regenerates`. Writes a JSON report under `results/repro/` and an ASCII summary to stdout. |
| [`__init__.py`](__init__.py) | Package marker. |

## How it was produced

Hand-written. Deterministic by construction — that is the point.

```bash
python -m src.infra.repro
# → "All reproducibility checks passed." on a clean repo
```

Also invoked from `scripts/run_all.py` as the final stage.

## How it's consumed

- CI (`.github/workflows/*`) and local pre-merge checks gate on 7/7 PASS.
- [`../../results/repro/`](../../results/repro/) — JSON output + `_scratch/` side-by-side regen.
- [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md) is the human-readable companion — every bullet maps to a `Check` here.
- Report **Appendix 8** cites the 7/7 PASS status per phase.

## How to regenerate

```bash
python -m src.infra.repro
```

Tests:

```bash
python -m pytest tests/infra/ -q
```

## Neighbours

- **↑ Parent**: [`../`](../) — src/ index
- **↔ Siblings**: [`../data/`](../data/), [`../analysis/`](../analysis/), [`../tokenization/`](../tokenization/), [`../models/`](../models/), [`../training/`](../training/), [`../baselines/`](../baselines/), [`../evaluation/`](../evaluation/)
- **↓ Children**: none
