# `src/` — Source Packages

Each subpackage has its own `README.md` — click into the folder for details.

This directory holds every first-party module of the credit-card default
prediction pipeline, organised bottom-up so that data layers never
depend on modelling layers and modelling layers never depend on
training/evaluation. The single cross-cutting contract is
`src.tokenization.tokenizer.TOKEN_ORDER`, which every sequence consumer
routes through to stay drift-safe.

## Subpackage tree

```
src/
├── data/            # UCI loader + preprocessing + split + scaling (Section 2)
├── analysis/        # EDA figures and statistical tests (Section 2)
├── tokenization/    # Hybrid tokenizer + FeatureEmbedding (Section 3 / Novelty N1)
├── models/          # Attention / Transformer / TabularTransformer / MTLM (Section 3)
├── training/        # Losses, dataset sampler, supervised + MTLM training loops (Section 3)
├── baselines/       # Random Forest benchmark + per-row prediction refit (Section 4)
├── evaluation/      # calibration / fairness / uncertainty / significance / interpret / visualise (Section 4 + Appendix)
└── infra/           # Reproducibility gate (regenerate-and-diff) (Appendix)
```

## Import-direction invariant (what depends on what)

```
data, analysis        ──► tokenization, models        ──► training, evaluation, baselines
                                        │
                                        └─────────►    infra (reads everything, writes nothing)
```

Breaking this ordering would cause circular imports and is caught by
`tests/` subfolder structure, which mirrors the layering.

## Invoking the CLIs

Every subpackage exposes at least one CLI. Run from the repository root:

- `python -m src.data.preprocessing`      — build `data/processed/splits/*.csv`
- `python -m src.analysis.eda`            — emit 12 EDA figures to `figures/eda/`
- `python -m src.baselines.random_forest` — tune and persist `results/baseline/rf/`
- `python -m src.baselines.rf_predictions`— refit RF and dump per-row probabilities
- `python -m src.training.train_mtlm`     — Phase 6A pretraining (Novelty N4)
- `python -m src.training.train`          — Phase 6/7/8 supervised training
- `python -m src.evaluation.evaluate`     — ensemble metrics + comparison table
- `python -m src.evaluation.calibration`  — temperature/Platt/isotonic + ECE
- `python -m src.evaluation.fairness`     — SEX/EDUCATION/MARRIAGE disparity audit
- `python -m src.evaluation.uncertainty`  — MC-dropout + refuse curve
- `python -m src.evaluation.significance` — McNemar / DeLong / bootstrap + BH-FDR
- `python -m src.evaluation.interpret`    — attention rollout + feature importance
- `python -m src.evaluation.visualise`    — ROC/PR/CM/training-curve/reliability PNGs
- `python -m src.infra.repro`             — regenerate every artefact and diff

## Tests

Every subpackage has a mirrored folder in `tests/`, e.g. `src/models/`
is covered by `tests/models/`. See `tests/README.md` for layout details.

## Report sections

The coursework report has 8 sections total (1-Intro, 2-Data Exploration,
3-Model Build-up, 4-Experiments/Results, 5-Conclusions,
6-Acknowledgements, 7-References, 8-Appendices). Code maps to them as:

| Report section                    | Code location                                              |
|-----------------------------------|------------------------------------------------------------|
| Section 2 — Data Exploration      | `src/data/`, `src/analysis/`                               |
| Section 3 — Model Build-up (40%)  | `src/tokenization/`, `src/models/`, `src/training/`        |
| Section 4 — Experiments & Results (30%) | `src/baselines/`, `src/evaluation/`                  |
| Appendix                          | `src/infra/`                                               |

Section 3 covers PDF requirements i–vi (tokenisation, embedding,
attention, encoder stack, training loop, regularisation).
