# Architecture

A folder-by-folder, subpackage-by-subpackage map of the
`credit-default-tabular-transformer` repository. The goal: anyone landing
on the repo for the first time (reviewer, marker, collaborator) can find
the exact file that implements every claim in `README.md`, `PROJECT_PLAN.md`
or the coursework report, without grepping.

This document is the canonical folder guide. `README.md` has a short
`Repository Structure` tree for the GitHub front page; this file is the
full version with rationale, data flow, and the novelty-register /
report-section cross references.

---

## 1. Folder-by-folder purpose

| Folder | Purpose |
|:---|:---|
| `src/` | All production Python code, organised into thin domain subpackages (see §2). Every module is covered by a test under `tests/` and is importable as `from src.<subpackage>.<module> import ...`. |
| `tests/` | 316 pytest cases: one `test_<module>.py` per `src/` module, plus `conftest.py` which adds the repo root to `sys.path`. Covers unit tests, numerical-property tests, and a handful of end-to-end integration tests that touch the committed artefacts. |
| `scripts/` | Thin CLI entry points. `run_pipeline.py` (EDA / preprocessing / RF benchmark); `run_all.py` (one-command end-to-end). No logic — all real work is in `src/`. |
| `figures/` | PNG/SVG figures grouped by the section of the report that cites them: `eda/`, `baseline/`, `evaluation/{comparison,calibration,fairness,uncertainty,significance,interpret}/`. All regenerate deterministically from the CSVs under `results/`. |
| `results/` | Numeric artefacts (CSV / JSON / NPZ) grouped by section: `analysis/`, `baseline/`, `evaluation/...`, plus `transformer/seed_*/`, `mtlm/run_*/`, `repro/`. Every CSV and JSON is checked into git; large NPZ arrays are committed where the reproducibility gate consumes them. |
| `data/` | `raw/` holds the tracked UCI fallback `.xls` (30K rows, ~600 KB); `processed/` holds the stratified 70/15/15 splits plus `feature_metadata.json` and `validation_report.json`. `SPLIT_HASHES.md` commits a SHA-256 for every processed file so `src.infra.repro` can detect drift. |
| `notebooks/` | Four Jupyter notebooks, one per coursework phase: `01_exploratory_data_analysis`, `02_data_preprocessing`, `03_random_forest_benchmark`, `04_train_transformer`. They re-use the code in `src/` — they do not duplicate logic. |
| `docs/` | `ARCHITECTURE.md` (this file), `MODEL_CARD.md` (Mitchell-style), `DATA_SHEET.md` (Gebru-style), `REPRODUCIBILITY.md` (artefact regeneration classes), `coursework_spec.md` (markdown transcription of the assignment PDF). |

Top-level files worth knowing about: `pyproject.toml` (Poetry dependencies
+ black / isort / flake8 config), `poetry.lock` (pinned), `CHANGELOG.md`
(per-PR log + contributor roster), `SECURITY_AUDIT.md` (C-1 weights-only
finding closed), `PROJECT_PLAN.md` (14-phase execution blueprint with the
novelty register).

---

## 2. Subpackage-by-subpackage purpose

### `src/data/` — ingestion + preprocessing

Two modules. `sources.py` is the layered, provenance-aware loader:
`UCIRepoSource` tries the UCI API (three retries with exponential backoff);
`LocalExcelSource` reads the tracked `.xls` fallback; `ChainedDataSource`
wires them together and `build_default_data_source` is the public factory
every consumer uses. Every load returns a `DataSourceResult` with the
source name, origin URI, wall-clock duration, and any fallback attempts —
so the provenance is auditable. `preprocessing.py` normalises column
names, folds undocumented codes (`EDUCATION {0,5,6} → 4`, `MARRIAGE {0} →
3`), engineers 22 derived features, does the stratified 70/15/15 split,
fits the scaler on train-only, and exports the processed CSVs plus
`feature_metadata.json` for the tokeniser.

### `src/analysis/` — EDA

One module, `eda.py`, generates the twelve EDA figures (`figures/eda/fig01_*`
through `fig13_*`) with statistical tests embedded (Wilson CIs, KS,
Mann-Whitney U, Cohen's d, Cramer's V, VIF, mutual information). Each
figure is tied to a paragraph of the report — see §5 below.

### `src/tokenization/` — tokeniser + embedding

`tokenizer.py` implements the hybrid PAY state + severity encoding
(Novelty **N1**) as a vectorised `tokenize_dataframe`, plus `MTLMCollator`
for the BERT-style random-masking protocol used by Novelty **N4**.
`embedding.py` is `FeatureEmbedding`: per-feature linear projections,
a `[CLS]` prepend, an optional temporal positional encoding (Ablation
A7), and an optional `[MASK]` token for MTLM pretraining. It also exposes
`build_temporal_layout` and `build_group_assignment` helpers used by the
attention biases in `src/models/transformer.py`.

### `src/models/` — attention, transformer, top-level model, MTLM head

Four modules. `attention.py` is `ScaledDotProductAttention` and
`MultiHeadAttention` written from scratch with an `attn_bias` hook that
the two novelty biases plug into. `transformer.py` stacks
`FeedForward` + PreNorm `TransformerBlock` into `TransformerEncoder`, and
defines `TemporalDecayBias` (Novelty **N3**) and `FeatureGroupBias`
(Novelty **N2**). `model.py` is the end-to-end wrapper: tokeniser →
embedding → encoder → pool (`[CLS]` / mean / max) → 2-layer MLP head →
logit (~28,600 trainable params at plan defaults). It also provides
`predict_logits`, `predict_proba`, `ensemble_probabilities`, and
`load_pretrained_encoder` (which accepts both full checkpoint bundles and
raw MTLM state dicts). `mtlm.py` is the self-supervised head — 3
categorical + 6 PAY + 14 numerical per-feature heads — plus `mtlm_loss`
(entropy-normalised CE + variance-normalised MSE) and `MTLMModel`.

### `src/training/` — dataset loader, losses, training loops, utils

Five modules. `dataset.py` has `StratifiedBatchSampler` (every batch
contains exactly `round(batch_size * positive_rate)` positives) and
`make_loader` (train / val / test / mtlm modes). `losses.py` has
`WeightedBCELoss`, `FocalLoss` (γ-configurable, α ∈ {scalar, tuple,
"balanced", None}), and `LabelSmoothingBCELoss`. `utils.py` has the
determinism protocol, device selection, weights-only checkpoint save/load
(per SECURITY_AUDIT C-1), `EarlyStopping`, parameter accounting, and
UTF-8-safe logging. `train.py` is the supervised loop — AdamW + cosine
warmup + gradient clipping + early stopping on val AUC-ROC, with optional
two-stage LR for §8.5.5 MTLM fine-tune and optional multi-task PAY_0 aux
head (Novelty **N5**). `train_mtlm.py` is the self-supervised
pretraining loop; it emits a ~130 KB `encoder_pretrained.pt` artefact
that `train.py --pretrained-encoder` consumes.

### `src/baselines/` — RF benchmark + RF prediction regenerator

`random_forest.py` is the full 200-iteration × 7-parameter
RandomizedSearchCV (1,000 fits) on engineered features, with 5-fold CV,
dual importance (Gini + permutation), threshold optimisation, and five
publication figures. `rf_predictions.py` refits the tuned RF from
`rf_config.json` to produce `results/baseline/rf/test_predictions.npz` +
`test_metrics.json` — this is what the reproducibility harness
regenerates bit-stably.

### `src/evaluation/` — Section 4 report pack

Seven modules, one per Section-4 deliverable.
`evaluate.py` aggregates the per-seed JSONs into the head-to-head
`comparison_table.csv`. `visualise.py` draws the ROC / PR / confusion /
reliability / training-curve figures under
`figures/evaluation/comparison/`. `calibration.py` is
`TemperatureScaling`, `PlattScaling`, `IsotonicCalibrator`, ECE / MCE /
Brier decomposition — drops raw ECE from 0.26 to 0.011 ± 0.003.
`fairness.py` (Novelty **N10**) audits SEX / EDUCATION / MARRIAGE
subgroups with demographic-parity / equal-opportunity / equalised-odds /
AUC-disparity / ECE-disparity tables. `uncertainty.py` (Novelty **N11**)
implements MC-dropout predictive entropy and the refuse curve —
`enable_dropout()` flips only `nn.Dropout*` submodules. `significance.py`
is McNemar (exact + chi-sq) / DeLong (Sun & Xu 2014 mid-ranks) / paired
bootstrap / BH-FDR / Hanley-McNeil power. `interpret.py` is the attention
rollout + per-feature importance comparison against RF Gini.

### `src/infra/` — reproducibility harness

One module. `repro.py` runs seven regeneration checks: artefact presence,
transformer run-files, Python/torch pins, git-tree cleanliness,
split-file SHA-256 hashes, RF-predictions bit-parity, and
`evaluate.py` bit-parity. `python -m src.infra.repro` exits 0 when every
derivative artefact matches the committed copy at
`rtol=1e-4, atol=1e-6`. CI gates on this.

---

## 3. Data flow

A single run of the full pipeline (as `scripts/run_all.py` drives it):

```
                      data/raw/default_of_credit_card_clients.xls
                      (tracked UCI fallback; .xls bytes reproducible)
                                          │
                             UCIRepoSource (API, 3 retries)
                                          │
                     ┌────────────────────┴────────────────────┐
                     │           src.data.sources              │
                     │  ChainedDataSource → DataSourceResult   │
                     └────────────────────┬────────────────────┘
                                          ▼
                     src.data.preprocessing
                     ├─ schema normalise (PAY_1→PAY_0, drop ID)
                     ├─ clean (EDUCATION {0,5,6}→4, MARRIAGE {0}→3)
                     ├─ engineer (22 derived features)
                     ├─ stratified 70/15/15 split
                     ├─ fit scaler on train only
                     └─ export data/processed/*.csv + metadata JSON
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
             src.analysis.eda                      src.baselines.random_forest
             (12 figures, stat tests)              (RandomizedSearchCV → tuned RF)
                                                                │
                                                    src.baselines.rf_predictions
                                                    (refit from rf_config.json)
                                                                │
                                          ▼                     ▼
                     ┌──────────────────────────────────────────┴────┐
                     │ src.tokenization.tokenizer  (hybrid PAY N1)   │
                     │ src.tokenization.embedding  ([CLS] + temporal │
                     │                              pos + MASK)      │
                     └────────────────────┬──────────────────────────┘
                                          ▼
                     src.models (attention → transformer → model)
                     TemporalDecayBias (N3), FeatureGroupBias (N2)
                                          │
                     ┌────────────────────┴──────────────────────────┐
                     ▼                                               ▼
             src.training.train_mtlm                          src.training.train
             (N4 MTLM pretraining)                            (supervised: 3 seeds)
             → encoder_pretrained.pt                          → seed_{42,1,2}/
                                          │                    best.pt + logs + preds
                                          ▼
                          src.training.train --pretrained-encoder
                          (§8.5.5 two-stage LR fine-tune)
                          → seed_42_mtlm_finetune/
                                          │
                                          ▼
                     ┌─────────────────────────────────────────────┐
                     │             src.evaluation                  │
                     │  evaluate   → comparison/comparison_table   │
                     │  calibration → ECE / Brier / Platt / iso    │
                     │  fairness    → subgroup / disparity (N10)   │
                     │  uncertainty → MC-dropout / refuse (N11)    │
                     │  significance→ McNemar / DeLong / BH-FDR    │
                     │  interpret   → attention rollout (N13)      │
                     │  visualise   → §4 report figures            │
                     └────────────────────┬────────────────────────┘
                                          ▼
                     src.infra.repro
                     (7 checks, exit 0 iff every artefact regenerates
                      bit-stably or within float tolerance)
                                          │
                                          ▼
                     results/repro/reproducibility_report.json
                     figures/evaluation/*/*.png + results/evaluation/*/*.csv
```

The key invariant: every leaf artefact under `results/` and
`figures/` is produced by exactly one source module, and
`src.infra.repro` regenerates it on every CI run to catch drift.

---

## 4. Where each novelty (N1-N12) lives

| Novelty | Name | File | One-line description |
|:---:|:---|:---|:---|
| **N1** | Hybrid PAY state + severity | `src/tokenization/tokenizer.py` | Encode `{-2,-1,0}` categorically and `{1..8}` as a normalised severity in one tokeniser pass. |
| **N2** | Feature-group attention bias | `FeatureGroupBias` in `src/models/transformer.py` | Learnable 5×5 bias matrix (CLS / demographic / PAY / BILL_AMT / PAY_AMT); zero-init. |
| **N3** | Temporal-decay bias | `TemporalDecayBias` in `src/models/transformer.py` | ALiBi-inspired learnable prior over within-group month distance (EDA Fig 9). |
| **N4** | Masked Tabular Language Modelling | `src/models/mtlm.py` + `src/training/train_mtlm.py` | BERT-style self-supervised pretraining with per-feature heads. |
| **N5** | Multi-task PAY_0 aux head | `aux_pay0=True` in `src/models/model.py`, `--aux-pay0-lambda` in `src/training/train.py` | Auxiliary 11-class CE on PAY_0 for extra gradient signal. |
| **N6** | Hybrid numerical/categorical MTLM loss | `mtlm_loss` in `src/models/mtlm.py` | Entropy-normalised CE + variance-normalised MSE in one objective. |
| **N7** | Stratified batch sampler | `StratifiedBatchSampler` in `src/training/dataset.py` | Every batch has exactly `round(batch_size * base_rate)` positives. |
| **N8** | Two-stage fine-tune LR | `--encoder-lr-ratio` in `src/training/train.py` | Lower LR on pretrained encoder vs. fresh head (§8.5.5). |
| **N9** | Hardened weights-only checkpoints | `src/training/utils.py` | `weights_only=True` default; `trust_source` flag per SECURITY_AUDIT C-1. |
| **N10** | Subgroup fairness audit | `src/evaluation/fairness.py` | Demographic parity / equal opportunity / equalised odds / AUC + ECE disparity. |
| **N11** | MC-dropout uncertainty | `src/evaluation/uncertainty.py` | Predictive entropy / mutual info / std + refuse curve. |
| **N12** | Model Card + Data Sheet | `docs/MODEL_CARD.md` + `docs/DATA_SHEET.md` | Mitchell (2019) / Gebru (2021) documentation conventions. |
| N13 | Attention rollout (interpretability) | `src/evaluation/interpret.py` | Abnar & Zuidema rollout + per-feature importance vs. RF Gini. |

---

## 5. Report-section-to-file map

Each Section 3 / Section 4 claim in the report maps to one source module
plus one backing artefact. The reproducibility harness regenerates
everything in the third column.

| Report section | Source module | Backing artefact |
|:---|:---|:---|
| §3.1 Dataset / EDA | `src/analysis/eda.py` | `figures/eda/fig01_*`–`fig13_*` |
| §3.2 Preprocessing pipeline | `src/data/preprocessing.py` | `data/processed/*.csv` + `feature_metadata.json` + `SPLIT_HASHES.md` |
| §3.3 Tokenisation (N1) | `src/tokenization/tokenizer.py` | `data/processed/feature_metadata.json` |
| §3.4 Embedding + temporal pos | `src/tokenization/embedding.py` | (inline: no committed artefact) |
| §3.5 Attention + transformer | `src/models/attention.py` + `src/models/transformer.py` | `results/transformer/seed_*/test_attn_weights.npz` |
| §3.6 Supervised training | `src/training/train.py` + `src/training/losses.py` | `results/transformer/seed_{42,1,2}/` |
| §3.7 MTLM pretraining (N4) | `src/models/mtlm.py` + `src/training/train_mtlm.py` | `results/mtlm/run_42/encoder_pretrained.pt` |
| §3.8 RF benchmark | `src/baselines/random_forest.py` | `results/baseline/rf_*.{csv,json}` + `figures/baseline/rf_*.png` |
| §4.1 Head-to-head comparison | `src/evaluation/evaluate.py` + `src/evaluation/visualise.py` | `results/evaluation/comparison/comparison_table.{csv,md}` + `figures/evaluation/comparison/*` |
| §4.2 Calibration | `src/evaluation/calibration.py` | `results/evaluation/calibration/calibration_metrics.csv` + reliability diagrams |
| §4.3 Fairness (N10) | `src/evaluation/fairness.py` | `results/evaluation/fairness/subgroup_metrics.csv` + disparity figures |
| §4.4 Uncertainty (N11) | `src/evaluation/uncertainty.py` | `results/evaluation/uncertainty/refuse_curve.csv` + MC-dropout NPZ |
| §4.5 Significance | `src/evaluation/significance.py` | `results/evaluation/significance/pairwise_tests.csv` + `power_analysis.csv` |
| §4.6 Interpretability | `src/evaluation/interpret.py` | `results/evaluation/interpret.json` + `figures/evaluation/interpret/*` |
| §4.7 Reproducibility | `src/infra/repro.py` | `results/repro/reproducibility_report.json` |
| Appendix: Model Card / Data Sheet (N12) | — | `docs/MODEL_CARD.md` + `docs/DATA_SHEET.md` |

---

## 6. How to add a new feature

**Pick the subpackage that owns the concern.** A new attention variant
goes in `src/models/`. A new tabular encoding goes in
`src/tokenization/`. A new evaluator (say, threshold sweep across
multiple cost ratios) goes in `src/evaluation/`. If it is genuinely new
territory — a cost-sensitive learning layer, or a non-attention
baseline — create a new subpackage and give it an `__init__.py` that
re-exports the public API. Keep domain concerns separate; a new model
module should never import directly from `src/evaluation/` and vice
versa.

**Register, test, and wire up the CLI.** Add the new module's public
symbols to the subpackage `__init__.py` so downstream code can do
`from src.<pkg> import NewThing`. Write a `tests/test_<module>.py` with
unit tests (numerical invariants, shape checks) and at least one end-to-
end case — the repo holds itself to 85%+ coverage per module. If the
feature produces a committed artefact, add it to
`src/infra/repro.py::run_all()` as a `Check`. If it needs a user-facing
entry point, add a CLI at the bottom of the module
(`if __name__ == "__main__": main()`) that `argparse`s whatever flags it
needs, with sane defaults pointing at the correct
`results/<section>/<subsection>/` directory. Finally, add a one-line
bullet in `CHANGELOG.md` under the current `[Unreleased]` block, update
the novelty register in `PROJECT_PLAN.md` if applicable, and cross-
reference it in §2 or §4/§5 of this file.

---

## 7. Where to go next

- `README.md` — fastest orientation; headline numbers, Quick Start,
  repository tree.
- `docs/MODEL_CARD.md` — Mitchell-style model card, intended use,
  deploy-with-Platt recommendation.
- `docs/DATA_SHEET.md` — Gebru-style datasheet for the UCI dataset.
- `docs/REPRODUCIBILITY.md` — deterministic / approximately deterministic
  / stochastic taxonomy, seed register, known Docker gap.
- `PROJECT_PLAN.md` — 14-phase execution blueprint with the full novelty
  register and the §21 coursework-PDF requirements audit.
- `CHANGELOG.md` — per-PR / per-commit log with the contributor roster.
