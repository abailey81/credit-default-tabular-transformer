# results/

> **Breadcrumb**: [↑ repo root](../) > **results/**

**Numeric artefacts root** — CSVs, JSON summaries, NPZ prediction arrays, trained model checkpoints, and reproducibility logs. Feeds Section 2 (Data), Section 3 (Model), Section 4 (Experiments), and Appendix 8 (Reproducibility) of the report.

Every file here is either produced by [`src/infra/repro.py`](../src/infra/repro.py) (most of them) or checked in as "ground truth" for downstream consumers. Committed outputs are deterministic — bit-stable regen via `python -m src.infra.repro` — except the MC-dropout NPZ in `evaluation/uncertainty/` and the bootstrap resamples under `evaluation/significance/`, which are stochastic-and-seeded. All regenerate bit-stably under the committed seeds.

## What's here

| Subfolder | Contents |
|---|---|
| [`analysis/`](analysis/) | EDA summary table (CSV + LaTeX). |
| [`baseline/`](baseline/) | RF config, CV log, feature importances, aggregate metrics. |
| [`transformer/`](transformer/) | One `seed_X/` per supervised run + aggregate summary CSV. |
| [`mtlm/`](mtlm/) | Pretrained MTLM encoder + training log (**N4**). |
| [`evaluation/`](evaluation/) | Calibration / fairness / uncertainty / significance / comparison + `interpret.json`. |
| [`repro/`](repro/) | Reproducibility report; `_scratch/` for bit-stability side-by-side. |
| [`pipeline/`](pipeline/) | Per-stage logs from `scripts/run_all.*` invocations. |

## How it was produced

Each subfolder is written by the corresponding `src/` module — see each child README. The reproducibility harness [`src/infra/repro.py`](../src/infra/repro.py) re-invokes every CLI and hashes outputs to detect drift.

## How it's consumed

- [`figures/`](../figures/) loads CSV / JSON data from here to render PNGs.
- Report **Section 2-4** + **Appendix 8** cite tables and numbers directly from these files.
- [`docs/MODEL_CARD.md`](../docs/MODEL_CARD.md) pulls calibration + fairness + uncertainty numbers.
- CI (`.github/workflows/*`) asserts 7/7 PASS via `results/repro/reproducibility_report.json`.

## How to regenerate

```bash
poetry run python -m src.infra.repro
```

## Neighbours

- **↑ Parent**: [`../`](../) — repo root
- **↔ Siblings**: [`../data/`](../data/), [`../figures/`](../figures/), [`../src/`](../src/), [`../tests/`](../tests/), [`../scripts/`](../scripts/), [`../notebooks/`](../notebooks/), [`../docs/`](../docs/)
- **↓ Children**: [`analysis/`](analysis/), [`baseline/`](baseline/), [`transformer/`](transformer/), [`mtlm/`](mtlm/), [`evaluation/`](evaluation/), [`repro/`](repro/), [`pipeline/`](pipeline/)
