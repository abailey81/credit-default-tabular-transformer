# `docs/` — Project documentation index

Human-readable companions to the report. Each file has a narrow scope
and lives here (rather than in the repo root) so the top-level
`README.md` stays short.

## Documents

| File                  | Purpose |
|---|---|
| `ARCHITECTURE.md`     | System diagram of the pipeline: stages, artefact dependencies, subprocess boundaries. Read this before touching `scripts/run_all.py` or reordering stages. |
| `MODEL_CARD.md`       | Model card for the `TabularTransformer` — intended use, training data, metrics, limitations, fairness audit summary. Written against Mitchell et al. (2019). |
| `DATA_SHEET.md`       | Data sheet for the UCI "Default of Credit Card Clients" dataset — motivation, composition, collection process, preprocessing, uses, distribution. Written against Gebru et al. (2018). |
| `REPRODUCIBILITY.md`  | The human-readable companion to `src/infra/repro.py`. Every bullet corresponds to one `Check` name and explains what it guards against. |
| `coursework_spec.md`  | The original coursework specification that this project fulfils — kept as-is for auditability. |

## Cross-reference

- `src/README.md`                — package-wide overview + subpackage tree.
- `tests/README.md`              — test layout + fixtures.
- `scripts/README.md`            — Option A / Option B pipeline drivers.
- `notebooks/README.md`          — exploration notebooks.
- `data/README.md`               — data layout (managed by Agent A).
- `figures/README.md`            — figure index (managed by Agent B).
- `results/README.md`            — artefact index (managed by Agent B).
- Repo-root `README.md`          — Quick Start + headline results.
- Repo-root `PROJECT_PLAN.md`    — the authoritative plan all work maps against.
- Repo-root `CHANGELOG.md`       — phase-by-phase history.
- Repo-root `SECURITY_AUDIT.md`  — weights-only checkpoint + CVE-tracking.

## Conventions

- Every `src/X/` subpackage has a `README.md` mirroring its layout.
- Every `tests/X/` subfolder has a `README.md` describing coverage.
- Report section numbers refer to the final report; ablation / novelty
  tags (`A1..A16`, `N1..N5`) refer to `PROJECT_PLAN.md`.
