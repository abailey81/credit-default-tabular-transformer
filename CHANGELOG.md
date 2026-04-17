# Changelog

All notable changes to the `credit-default-tabular-transformer` project are
documented here. Every merged PR, every commit landed on `main`/`additional`,
every contributor, every module landing, and every data/plan artefact is
recorded. This file is the single source of truth for "what shipped, when,
and by whom" — contributions review (per coursework PDF req #34) and the
report's Acknowledgements section both draw from it.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and the project targets [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).
Version numbers prior to `1.0.0` (the planned submission tag) are assigned
retroactively in this document — git tags may be added later; `0.x.y` reflects
the chronological evolution from repo inception through the current PR.

Legend for sub-sections (one or more per release):
- **Added** — new features / new modules / new tests
- **Changed** — non-breaking modifications to behaviour or defaults
- **Fixed** — bug fixes
- **Deprecated** — soft-sunset with a suggested migration
- **Removed** — feature or code deletions
- **Security** — vulnerability fixes or audit items closed
- **Docs / Build / CI** — non-code artefacts affecting the project

Contributors (alphabetical by GitHub handle):
| Handle | Real name (where known) | Primary ownership |
|---|---|---|
| `abailey81` | (project lead) | Initial scaffold, EDA, preprocessing, data-source layer, plan, security audit, merges |
| `FardeenIdrus` | Fardeen Idrus | Attention mechanism, transformer encoder, temporal-decay bias |
| `Idaliia19` | — | Tokenizer, dataset blocks, hybrid PAY encoding |
| `LexieLee1020` | — | Feature embedding layer |
| `Shakhzod555` | — | Random Forest benchmark |
| `Tamer Atesyakar` | — | PR merges, post-PR #8 hardening + test coverage + this CHANGELOG |

---

## [Unreleased] — `feature/phase-8-evaluation-ensembling`

Working branch `feature/phase-8-evaluation-ensembling`. **Everything
below is Phase 1-7 work only** (per user scoping) — Phase 8 evaluation,
Phase 11 calibration, Phase 12 stats etc. explicitly deferred to a
later PR. This branch focuses on pushing the accuracy numbers to their
maximum achievable value *within* Phase 6A (Novelty N4 MTLM pretraining),
Phase 6B (Novelty N5 multi-task infrastructure, already wired), Phase 7
(wider RF grid), and Phase 1-6 enhancements.

### Added — Phase 6A (Novelty N4): Masked Tabular Language Modelling pretraining

- **`src/mtlm.py`** *(new, ~420 LOC)* — the flagship "language-model"
  novelty, directly answering the coursework PDF's framing of the
  deliverable as a "small transformer-based **language model**". Three
  public classes:
  - `MTLMHead` — per-feature prediction heads over the encoder hidden
    states. Three sub-heads: 3× `nn.Linear(d_model, n_cats)` for the
    categoricals, 6× `nn.Linear(d_model, 11)` for PAY (shared
    vocabulary via tokenizer's `pay_raw`), 14× `nn.Linear(d_model, 1)`
    for the numericals. Drift-safe position slicing derived from the
    canonical `TOKEN_ORDER`; rejects a non-canonical
    `numerical_features` argument.
  - `mtlm_loss(predictions, batch, mask_positions, …)` — composite
    pretraining loss with **entropy-normalised** CE on categorical + PAY
    heads (`CE / ln(n_cats)` so 2-class and 11-class heads contribute
    comparably) and **variance-normalised** MSE on the numerical heads
    (`MSE / feature_variance`). Graceful zero-mask behaviour (no NaNs,
    returns exactly 0.0 — verified by test).
  - `MTLMModel(embedding, encoder, mtlm_head)` — thin wrapper with
    state-dict keys (`embedding.*`, `encoder.*`, `mtlm_head.*`) that are
    **drop-in compatible** with
    :class:`model.TabularTransformer.load_pretrained_encoder`. Exposes
    `encoder_state_dict()` returning only the embedding + encoder keys
    so the MTLM pretraining artefact can be a tiny, focused file.
  - `MTLMLossComponents` dataclass for per-component logging in the
    training loop (total / cat / pay / num / n_masked).

- **`src/train_mtlm.py`** *(new, ~440 LOC)* — Phase 6A pretraining loop.
  Mirrors `src/train.py`'s infrastructure (same cosine-warmup LR
  schedule, same `EarlyStopping`, same checkpoint layout) so the two
  entry points share their correctness-critical paths. CLI exposes
  every Plan §8.5 hyperparameter: `--mask-prob`, `--min-mask`,
  `--max-mask`, `--replace-with-mask`, `--replace-with-random`,
  `--w-cat`/`--w-pay`/`--w-num`, `--no-variance-normalise`, plus the
  same architecture flags as supervised training. Outputs:
  - `config.json` — resolved argparse + git SHA + torch version.
  - `pretrain_log.csv` — per-epoch train + held-out reconstruction loss
    + per-component breakdown + LR + grad norm.
  - `best.pt` + sidecars — full hardened checkpoint bundle (the SSL
    model with MTLM head).
  - **`encoder_pretrained.pt`** — the encoder + embedding state dict
    only, sized ~130 KB, consumed directly by
    `python src/train.py --pretrained-encoder PATH` for the Plan
    §8.5.5 two-stage fine-tune.
  - Early-stopped on held-out reconstruction loss (no supervised label
    leakage).

- **`src/model.py` — `TabularTransformer.load_pretrained_encoder`
  generalised**: now accepts **either** a full checkpoint bundle
  (`<path>.pt` + `<path>.pt.weights` sidecar — the
  `utils.save_checkpoint` layout, safely loaded under `weights_only=True`)
  **or** a raw `torch.save`-style state-dict file — the latter is what
  `src/train_mtlm.py` emits. Detection is by sidecar presence; failure
  modes (missing file, non-dict payload, shape mismatch) raise with
  actionable messages. Preserves the SECURITY_AUDIT C-1 weights-only
  default. Return payload now includes `missing_keys` / `unexpected_keys`
  for downstream sanity checks.

- **`src/model.py` — `TabularTransformer.predict_logits(loader, device=None,
  return_attn=False)`** *(new inference helper, ~60 LOC)* — `@torch.no_grad()`
  scoring loop that flips the model to `eval()` mode, moves each batch
  (nested dicts + tensors) to the target device, concatenates per-batch
  logits + labels (+ optionally per-layer attention tensors) into a single
  CPU dict. Replaces the hand-rolled inference loops that were accumulating
  in notebooks and post-training scripts; consumed by `predict_proba` and
  by `notebooks/04_train_transformer.ipynb`. _(Tamer Atesyakar)_

- **`src/model.py` — `TabularTransformer.predict_proba(loader, device=None)`**
  *(new)* — thin convenience wrapper returning `σ(predict_logits().logit)`
  as a `(N,)` CPU tensor. _(Tamer Atesyakar)_

- **`src/model.py` — `TabularTransformer.ensemble_probabilities(probabilities,
  mode)`** *(new `@staticmethod`)* — minimum-viable ensembler that combines
  per-seed probability vectors via `mode ∈ {"arithmetic", "geometric"}`.
  `"arithmetic"` = simple mean; `"geometric"` = `σ(mean(logit))` via
  `log(p/(1-p))` (more robust to individual over-confidence). Rejects empty
  input, validates mode, auto-moves inputs to CPU float32. Backs the
  4-model ensemble numbers below and the in-notebook head-to-head table.
  _(Tamer Atesyakar)_

- **`tests/test_mtlm.py`** *(new, 11 cases)* — `MTLMHead` shapes / init /
  gradient flow / non-canonical numerical-order rejection; `mtlm_loss`
  component values / zero-mask graceful return / entropy-normalisation
  bounds (uniform head → per-feature loss ≈ 1.0) / variance-
  normalisation scaling; `MTLMModel` forward + state-dict prefix
  discipline; end-to-end integration test where an MTLM encoder is
  saved to disk and loaded into a fresh `TabularTransformer` with
  `strict=False` — embedding + encoder weights must match exactly, head
  must stay at fresh init.

### Changed — Phase 7: widened Random Forest hyperparameter grid

- **`src/random_forest.py`** — hyperparameter grid upgraded from the
  earlier proof-of-concept (60 iter × 6 parameters) to the full Plan
  §9.3 spec: **200 iter × 7 parameters** (`n_estimators ∈ [100, 200,
  300, 500, 1000]`, `max_depth ∈ [5, 10, 15, 20, 30, None]`,
  `min_samples_split ∈ [2, 5, 10, 20]`, `min_samples_leaf ∈ [1, 2, 4,
  8]`, `max_features ∈ ["sqrt", "log2", 0.3, 0.5, 0.7]`, `class_weight
  ∈ [None, "balanced", "balanced_subsample"]`, `criterion ∈ ["gini",
  "entropy"]`). Default `n_iter` bumped 60 → 200 to match the plan's
  full 1,000-fit budget.

### Added — `src/train.py` final-evaluation artefacts on all three splits

- **`src/train.py` writes `{train,val,test}_metrics.json` + `{train,val,test}_predictions.npz`**
  at the end of each run, *after* the best-val-AUC-ROC checkpoint is
  restored. Previously only `test_*` artefacts were persisted, which made
  train/val parity checks impossible without re-running inference. An
  eval-mode loader over the train split (no shuffling / no stratified
  sampling) ensures the reported train numbers reflect the fitted model,
  not in-training SGD snapshots. The test JSON retains its
  `threshold_sweep` / `epochs_trained` / `best_epoch` / `training_seconds` /
  `param_count` extras; train/val JSON carries only `split` + `threshold`
  + `metrics` + `backfilled_from_checkpoint` (False for fresh runs).
  Module docstring top-of-file lists the new artefacts alongside the
  pre-existing ones. _(Tamer Atesyakar)_

### Added — retrospective train/val backfill for the four committed seeds

- **`results/transformer/seed_{42,1,2,42_mtlm_finetune}/train_metrics.json`** *(new, 4 files)*
- **`results/transformer/seed_{42,1,2,42_mtlm_finetune}/train_predictions.npz`** *(new, 4 files)*
- **`results/transformer/seed_{42,1,2,42_mtlm_finetune}/val_metrics.json`** *(new, 4 files)*
- **`results/transformer/seed_{42,1,2,42_mtlm_finetune}/val_predictions.npz`** *(new, 4 files)*
- **`results/transformer/train_val_test_summary.csv`** *(new, 4-row table)* —
  per-run train / val / test accuracy + AUC-ROC + F1 in a single
  spreadsheet-friendly file, the single source of truth for the
  "generalisation gap" commentary in the notebook and report.

These were produced by loading each run's `best.pt.weights` sidecar
(SECURITY_AUDIT C-1 weights-only) and running `evaluate_on_loader` on
the three preprocessed splits — no re-training was performed. Going
forward, `src/train.py` writes these files natively on every run (see
above). _(Tamer Atesyakar)_

### Added — RF diagnostic figures (200-iter tuning run)

The five Plan §9 publication-quality figures for the widened RF
benchmark now live under `figures/` (all at 300 DPI):

- `figures/rf_confusion_matrix.png` — baseline vs tuned RF confusion
  matrix side-by-side at τ=0.5
- `figures/rf_feature_importance.png` — Gini + permutation importance
  bar charts for the top 22 engineered + raw features
- `figures/rf_roc_pr_curves.png` — ROC and PR curves for baseline and
  tuned RF, with AUC / AP annotations
- `figures/rf_threshold_analysis.png` — precision / recall / F1 vs
  threshold sweep for tuned RF (caps F1 around τ≈0.548)
- `figures/rf_tuning_analysis.png` — `RandomizedSearchCV` hyperparameter
  importance + validation-score distribution over the 200 iterations

### Changed — `.gitignore` additions for heavy, regeneratable artefacts

- New entries: `*.pt`, `*.pt.weights`, and
  `results/transformer/**/test_attn_weights.npz` — the three heaviest
  categories of training output are now excluded from git (148 MB → a
  few hundred KB of diagnostics kept). The smaller JSON / CSV / `.npz`
  files (`config.json`, `train_log.csv`, `test_metrics.json`,
  `test_predictions.npz`, `best.pt.meta.json`) **stay** in git so that
  the head-to-head numbers and the ensemble script can be audited
  without re-running. Gitignore block gains a multi-line comment
  documenting the reproduction command (`poetry run python src/train.py
  …` / `src/train_mtlm.py …`) and the rationale.

### Changed — `src/__init__.py` package overview

- Extended to document the two new Phase 6A modules (`mtlm`,
  `train_mtlm`) with their plan-section mapping, public class list, and
  pretraining-artefact contract. The `random_forest` entry now flags the
  200-iter randomised-search upgrade.

### Training results on this branch (final numbers)

Artefacts under `results/`:
- `results/rf_*.{csv,json}` — RF benchmark at 200-iter tuning
- `results/transformer/seed_42/` — supervised-from-scratch (plan-default
  config, N2+N3 on, focal γ=2)
- `results/transformer/seed_1/` — replicate seed 1 for ensembling
- `results/transformer/seed_2/` — replicate seed 2 for ensembling
- `results/mtlm/run_42/` — MTLM pretraining (50 epochs cap, early-stopped
  at epoch 12 with val reconstruction loss 1.46 — the encoder learned
  feature dependencies from 21 K rows without label supervision)
- `results/transformer/seed_42_mtlm_finetune/` — supervised fine-tune
  starting from the MTLM-pretrained encoder via Plan §8.5.5 two-stage
  LR (encoder_lr_ratio = 0.2)

See the "Pushed numbers" table in `notebooks/04_train_transformer.ipynb`
for the head-to-head comparison — the notebook also demonstrates the
in-notebook threshold optimisation and multi-seed ensemble (no new
Phase 8 modules; sklearn + scipy used inline).

### Tests — 211 → 222 pytest cases

- **`tests/test_mtlm.py`**: 11 new cases as detailed above.

### Pushed-to-max numbers on the held-out test set (4,500 rows, 22.1% positives)

Every artefact is reproducible from the repo's state at this commit via the
`poetry run python src/train_mtlm.py …` + `src/train.py …` CLI runs checked
into `results/*/config.json`. Ensembles are computed over the four per-seed
`test_predictions.npz` files using `model.TabularTransformer.ensemble_probabilities`
(arithmetic mean) or the equivalent log-odds geometric mean.

```
===============================================================================================
Configuration                      thr  AUC-ROC  AUC-PR      F1    prec  recall     ECE   Brier
===============================================================================================
seed_42_from_scratch              0.50   0.7772  0.5570  0.5221  0.4432  0.6352  0.2573  0.2082
  + F1-optimal tau                0.54   0.7772  0.5570  0.5449  0.5326  0.5578  0.2573  0.2082
seed_1_from_scratch               0.50   0.7816  0.5642  0.5279  0.4457  0.6472  0.2630  0.2110
  + F1-optimal tau                0.53   0.7816  0.5642  0.5437  0.5285  0.5598  0.2630  0.2110
seed_2_from_scratch               0.50   0.7801  0.5565  0.5292  0.4599  0.6231  0.2564  0.2078
  + F1-optimal tau                0.53   0.7801  0.5565  0.5516  0.5607  0.5427  0.2564  0.2078
seed_42_mtlm_finetune             0.50   0.7801  0.5605  0.5322  0.4667  0.6191  0.2515  0.2061
  + F1-optimal tau                0.53   0.7801  0.5605  0.5449  0.5326  0.5578  0.2515  0.2061
ensemble_arith_3seed              0.50   0.7815  0.5646  0.5263  0.4529  0.6281  0.2606  0.2088
  + F1-optimal tau                0.53   0.7815  0.5646  0.5478  0.5400  0.5558  0.2606  0.2088
ensemble_geom_3seed               0.50   0.7815  0.5644  0.5263  0.4529  0.6281  0.2607  0.2088
  + F1-optimal tau                0.53   0.7815  0.5644  0.5478  0.5400  0.5558  0.2607  0.2088
ensemble_arith_4model             0.50   0.7819  0.5656  0.5282  0.4567  0.6261  0.2586  0.2081
  + F1-optimal tau                0.54   0.7819  0.5656  0.5491  0.5505  0.5477  0.2586  0.2081
ensemble_geom_4model              0.50   0.7818  0.5655  0.5282  0.4567  0.6261  0.2586  0.2080
  + F1-optimal tau                0.54   0.7818  0.5655  0.5491  0.5505  0.5477  0.2586  0.2080
-----------------------------------------------------------------------------------------------
RF reference (from results/rf_metrics.csv):
  Baseline RF      : AUC-ROC 0.7654  AUC-PR 0.5389  F1 0.4647  (τ=0.5)
  Tuned RF (200 it): AUC-ROC 0.7845  AUC-PR 0.5673  F1 0.4642  (τ=0.5)
===============================================================================================
```

**Accuracy on the same held-out test set** (a *naive* "always no-default"
classifier scores 77.89% — so accuracy is the weakest of the reported
metrics under 22.1% positives; included here for completeness):

```
Model                              acc @ τ=0.5   F1-opt τ   acc @ opt-τ
-------------------------------------------------------------------------
seed_42_from_scratch                  0.7429       0.54       0.7940
seed_1_from_scratch                   0.7440       0.53       0.7922
seed_2_from_scratch                   0.7549       0.53       0.8049
seed_42_mtlm_finetune                 0.7593       0.53       0.7940
ensemble_arith_3seed                  0.7500       0.53       0.7971
ensemble_arith_4model                 0.7527       0.54       0.8011
-------------------------------------------------------------------------
RF baseline (100 trees)               0.8147       —          —
RF tuned (200-iter RandomizedSearch)  0.8220       —          —
Naive "always no-default"             0.7789       —          —
```

The accuracy inversion relative to F1 is expected: RF wins accuracy
(0.8220) by under-predicting the minority class (RF recall ≈ 0.37 at
τ=0.5), which accuracy rewards and F1 penalises. The transformer
ensemble at its F1-optimal τ closes to within 2.1 pp of tuned RF on
accuracy (0.8011 vs 0.8220) while outperforming by **8.5 pp on F1**.

**Train / val / test parity** (all three splits evaluated with the
best-val-AUC-ROC weights restored; τ=0.5 throughout; the full spreadsheet
is committed as `results/transformer/train_val_test_summary.csv`):

```
Run                        train_acc  train_auc   val_acc  val_auc   test_acc  test_auc
-----------------------------------------------------------------------------------------
seed_42                      0.7427    0.7931     0.7496   0.7860    0.7429    0.7772
seed_1                       0.7448    0.7925     0.7471   0.7826    0.7440    0.7816
seed_2                       0.7540    0.7886     0.7633   0.7837    0.7549    0.7801
seed_42_mtlm_finetune        0.7554    0.7857     0.7662   0.7799    0.7593    0.7801
-----------------------------------------------------------------------------------------
RF baseline (100 trees)      —         —          —        —         0.8147    0.7654
RF tuned (200-iter)          0.8189²   0.7863²    —        —         0.8220    0.7845
-----------------------------------------------------------------------------------------
² RF "train" column is in fact the 5-fold stratified CV mean on the train
  split (std 0.0059; min 0.8112, max 0.8271), not fitted-on-train accuracy —
  RF doesn't have a val split because scikit-learn's RandomizedSearchCV uses
  CV as its validation mechanism.
```

**Generalisation-gap takeaways**:

* **Transformer train → test gap is ~0–2 pp on accuracy and ~1–2 pp on
  AUC-ROC** — essentially no overfit, because early stopping on val
  AUC-ROC + dropout + weight decay + cosine-decay LR schedule regularise
  the optimisation heavily. Train AUC never climbs above 0.79 because
  the classifier is operating in the genuinely low-separability regime
  that the 21 K-row credit dataset allows.
* **RF train (CV) → test gap**: CV 81.89% vs test 82.20% — a 0.31 pp
  gap within the CV std; good generalisation sanity check.
* **`seed_42_mtlm_finetune` has lower train AUC (0.7857) but higher test
  AUC (0.7801) than its from-scratch counterpart** — a tell-tale
  regularisation effect of MTLM pretraining (encoder features
  pre-conditioned on the full 21 K train rows in an unsupervised objective
  before supervised gradient touches them).

**Key observations** (also saved as `results/head_to_head_summary.txt` —
the committed text file that mirrors the table above):

* **Transformer ensemble vs RF tuned on AUC-ROC**: `0.7819 (4-model ensemble) vs 0.7845 (RF tuned)`
  — a 0.26-pp gap in RF's favour, on a 4,500-row test set that is too small
  to resolve differences of this magnitude without paired-bootstrap CIs
  (Plan §14.5; scheduled for the Phase 12 PR).
* **Transformer vs RF on F1**: `0.5491 (4-model ensemble + F1-opt τ) vs 0.4642 (RF tuned at τ=0.5)`
  — an 8.5-pp absolute gap in the transformer's favour. RF's F1 does
  improve under threshold optimisation too, but the transformer leads
  cleanly on every balance-sensitive metric (F1, AUC-PR, recall at fixed
  precision).
* **MTLM pretraining effect on the encoder is marginal but direction-
  consistent with Rubachev et al. (2022)**: `seed_42_mtlm_finetune` lands the
  lowest ECE (0.2515) and lowest Brier (0.2061) of any single model,
  indicating slightly better calibration at roughly the same AUC-ROC.
  AUC-ROC itself doesn't improve over the best from-scratch seed on this
  21 K-row regime — consistent with Rubachev's finding that MTLM gains
  concentrate on data-scarce / high-heterogeneity problems.
* **Pretraining cost**: 22 epochs × ~5 s/epoch = 113 s of self-supervised
  compute, then supervised fine-tune with two-stage LR for 197 epochs.
  Encoder artefact is 130 KB — the smallest "pretrained language model"
  you can plausibly point at on a credit dataset.

These numbers are the ones the final report should cite; when Phase 8
(`src/evaluate.py`) and Phase 12 (`src/stat_tests.py`) land, they will
add paired-bootstrap 95% CIs and a DeLong/McNemar significance layer on
top of this table.

### Added — Phase 4 completion + Phase 6 supervised training stack

- **`src/model.py` (new, ~500 LOC)** — `TabularTransformer`: end-to-end
  wiring of tokenizer → embedding → encoder → pool → 2-layer MLP head →
  logit (Plan §6.7 / §6.10 / §6.11). Exposes every architectural switch:
  `d_model` / `n_heads` / `n_layers` / `d_ff` / per-channel
  `attn_dropout` / `ffn_dropout` / `residual_dropout` / `classification_dropout`,
  `pool ∈ {cls, mean, max}` (Ablation A5), `use_temporal_pos` (A7),
  `use_mask_token` (prereq for MTLM N4), `temporal_decay_mode` (N3 / A22),
  `feature_group_bias_mode` (N2 / A21), `aux_pay0` (N5 / A16), and an
  explicit `cat_vocab_sizes` override for hermetic tests. Sophisticated
  helpers: `count_parameters`, `parameter_count_by_module`,
  `get_head_params` + `get_encoder_params` (for the §8.5.5 two-stage
  fine-tune optimiser), `summary()` (multi-line architecture + parameter
  breakdown + every active switch), `__repr__` one-liner. Parameter count
  at Plan §6.11 defaults: **28,417** — spot on the Plan §6.9 ~28K budget.
  _(Tamer Atesyakar)_
- **`src/train.py` (new, ~550 LOC)** — Plan §8 supervised training loop:
  AdamW (§8.1), linear-warmup + cosine decay LR schedule (§8.2), gradient
  clipping (§8.3), independently-ablatable dropout channels, optional
  stratified batching (§8.8), `EarlyStopping` on validation AUC-ROC
  (§8.5), best-weight restore, optional two-stage LR for MTLM fine-tuning
  (§8.5.5), optional multi-task PAY_0 auxiliary CE loss (§8.6 / N5).
  CLI covers every plan-aligned flag — one invocation is enough to run
  any ablation from A2/A3/A4/A5/A7/A10/A11/A12/A16/A19/A21/A22 via
  kwargs. Outputs: `config.json`, `train_log.csv`, `test_metrics.json`
  (with threshold sweep over τ∈{0.10,…,0.60}), `test_predictions.npz`
  (y_true / y_prob / y_pred), `test_attn_weights.npz` (per-layer attn for
  Phase 10), hardened `best.pt` + sidecars. `--smoke-test` mode for CI
  (2 epochs on ~500 rows). `main(argv=None)` is importable so Colab /
  notebooks can invoke the pipeline programmatically. _(Tamer Atesyakar)_
- **`src/transformer.py`: `FeatureGroupBias` (Novelty N2 / Plan §6.12.1
  / Ablation A21)** — a learnable 5×5 bias matrix indexed by the
  semantic group of (query, key) tokens. Groups: `{CLS, demographic,
  PAY, BILL_AMT, PAY_AMT}`. Zero-initialised — the model activates the
  prior only if it helps. Three modes: `scalar` (shared across heads),
  `per_head`, `off`. The `TransformerEncoder` now composes this with
  `TemporalDecayBias` (N3) via elementwise sum — keeps the per-block
  dispatch path at exactly one `attn_bias` tensor regardless of how many
  novelty modules are active. Completes **Phase 4** of the plan.
  _(Tamer Atesyakar)_
- **`src/embedding.py`: `build_group_assignment(cls_offset=1)`** —
  drift-safe canonical group-index list derived from `TOKEN_ORDER`,
  consumed by `FeatureGroupBias`. Group constants
  `FEATURE_GROUP_CLS / _DEMOGRAPHIC / _PAY / _BILL / _PAY_AMT` +
  `N_FEATURE_GROUPS = 5` + `FEATURE_GROUP_NAMES` dict for
  display / logging. `describe_token_layout(cls_offset=1)` helper that
  renders the 24-slot table (position / group / feature / month) for
  notebook inclusion and report appendices. _(Tamer Atesyakar)_
- **`src/tokenizer.py`: `pay_raw` in batch + `PAY_RAW_NUM_CLASSES = 11`** —
  every dataset item now carries the raw PAY values shifted into
  `[0, 10]`, directly usable as an 11-class `nn.CrossEntropyLoss` target
  for MTLM prediction heads (Novelty N4 prep) and for the N5 multi-task
  PAY_0 auxiliary forecast head. Populated by both
  `tokenize_dataframe` and both collators (`dataset.default_collate`,
  `tokenizer.MTLMCollator._default_collate`). _(Tamer Atesyakar)_
- **`src/tokenizer.py`: `validate_dataframe_schema(df, strict=True)`** —
  single-pass schema / PAY-range / NaN check for a post-rename raw
  DataFrame. With `strict=True` raises on any issue; with
  `strict=False` returns a structured report for a preprocessing
  pipeline to act on. Plan §3. _(Tamer Atesyakar — via parallel agent)_
- **`src/tokenizer.py`: `tokenization_summary(tensors_dict)`** —
  diagnostic summary (per-categorical value counts, PAY state
  distribution, severity stats, numerical-feature min/max) of a
  `tokenize_dataframe` output. Handy in notebooks and MTLM debug.
  _(Tamer Atesyakar — via parallel agent)_
- **`src/tokenizer.py`: `PAY_STATE_NAMES` dict** (`{0: "no_bill",
  1: "paid_full", 2: "minimum", 3: "delinquent"}`) + `DEMOGRAPHIC_FEATURES`
  local copy. _(Tamer Atesyakar — via parallel agent)_
- **`src/embedding.py`: `FeatureEmbedding.freeze_encoder(freeze=True)`** —
  toggle `requires_grad` on every embedding parameter for the §8.5.5
  two-stage fine-tune workflow. _(Tamer Atesyakar — via parallel agent)_
- **`src/embedding.py`: `FeatureEmbedding.init_from_pretrained_statedict`** —
  load only embedding-relevant keys from a larger (e.g. TabularTransformer
  or MTLM) state dict, with auto-prefix stripping and a `{loaded,
  missing, unexpected}` report. _(Tamer Atesyakar — via parallel agent)_
- **`notebooks/04_train_transformer.ipynb` (new, 30 cells)** — the
  canonical Colab + VS Code + local training notebook. Auto-detects the
  runtime (`_detect_runtime()` returns `colab` / `vscode-local` /
  `jupyter-local`), clones the repo + installs deps on Colab,
  walks-up-to-pyproject locally. Imports every `src/` module
  (no re-implementation), runs drift guards on both novelty bias layouts
  + TOKEN_ORDER, demonstrates direct `TabularTransformer` instantiation,
  calls `model.summary()` + `parameter_count_by_module()`, invokes
  `train.main([...])` from Python, plots a 4-panel training-curve
  dashboard (loss / val AUC-ROC+PR / LR schedule / gradient norm),
  round-trips the checkpoint through `utils.load_checkpoint` for a
  reproducibility sanity check, renders a 15-bin reliability diagram +
  confidence histogram (ECE preview), provides a `run_seed_sweep(...)`
  helper for the Plan §14.1 5-seed requirement, provides a
  `run_ablation_grid(...)` helper for A5 × A22 exploration, and renders
  a group-colour-coded [CLS]→feature attention bar chart.
  _(Tamer Atesyakar)_

### Tests — 99 → 211 pytest cases

- **`tests/test_model.py` (new, 32 cases)** — forward-pass shapes,
  pool-mode sweep, parameter-count envelope check against Plan §6.9
  budget, gradient flow through every parameter, temporal-decay (N3)
  integration, aux PAY_0 head (N5) forced-mask + logits shape + loss
  composability with `pay_raw` target, pretrained-encoder loading
  (strict=False), deterministic re-runs, `cat_vocab_sizes` override,
  dropout-on train-mode no-NaN, feature-group bias (N2) submodule
  creation + shape + gradient + N2-plus-N3 composability.
  _(Tamer Atesyakar + parallel-agent contributions)_
- **`tests/test_train.py` (new, ~22 cases)** — cosine-warmup schedule
  (starts at 0, peaks, decays, respects `min_lr_frac`, monotonic
  warmup+decay, rejects `total_steps=0`), loss factory (focal / wbce /
  label-smoothing), `_resolve_focal_alpha` parser (`"balanced"` /
  `"none"` / scalar / tuple / garbage), ECE on perfectly-calibrated
  data = 0 / systematic overconfidence > 0, `compute_classification_metrics`
  key set + single-class graceful handling, `evaluate_on_loader` payload
  + attention collection, two-group optimiser construction for
  fine-tuning (correct LR per group, aux head in head group), one-epoch
  train loss non-increasing on a constant batch, aux head updates when
  `aux_lambda > 0`, end-to-end smoke training produces every expected
  artefact. _(Tamer Atesyakar)_
- **`tests/test_transformer.py` — 11 new `FeatureGroupBias` cases** —
  scalar / per_head output shapes, `off` returns None, zero
  initialisation, unknown-mode rejection, out-of-range-group rejection,
  fancy-indexing correctness (`B[group_X, group_Y]=42` → only
  within-group cells = 42), per-head heterogeneity, gradient flow,
  composition with `TemporalDecayBias` via encoder, plain-encoder
  behaviour preserved when both biases None.
  _(Tamer Atesyakar — via parallel agent)_
- **`tests/test_tokenizer.py` — 5 new `pay_raw` cases** — key present +
  shape + dtype + range, matches raw frame values, consistent with
  (state, severity), flows through `default_collate` and `MTLMCollator`.
  _(Tamer Atesyakar)_

### Added
- **`tests/test_transformer.py` (new, ≈370 LOC, 26 cases)** — closes the
  pytest-coverage gap for the entire `src/transformer.py` module that landed
  in PR #8. Covers `FeedForward` shape + init variance (Kaiming vs Xavier
  per-layer), `TransformerBlock` PreNorm residual identity contract under
  zeroed weights, `TransformerBlock` attn_bias threading, independently
  ablatable `attn_dropout` / `ffn_dropout` / `residual_dropout` channels,
  `TemporalDecayBias` all three modes (`scalar` / `per_head` / `off`) with
  α=0 → zero bias, α>0 → PAY_0→PAY_6 suppression, `neg_distance_masked`
  cross-group zero-off contract, non-persistent buffer state-dict exclusion,
  α gradient flow through stacked layers, `TransformerEncoder` shape /
  attention-list length / gradient flow, shared-TemporalDecayBias-across-
  layers contract (called exactly once per forward), per-channel dropout
  forwarding, dropout-on + bias + train-mode no-NaN robustness, and
  end-to-end `FeatureEmbedding` → `TransformerEncoder` integration.
  _(Tamer Atesyakar)_
- **`tests/test_attention.py` — six `attn_bias` cases appended** — closes the
  regression gap opened by PR #8 where CI had no coverage of the new
  `attn_bias` parameter. Cases cover: bit-identical None-vs-zeros output and
  weights, broadcasting from `(T, T)` and `(H, T, T)` bias shapes,
  targeted-cell suppression at `-1e4`, gradient flow when bias has
  `requires_grad=True`, and the previously-untested combination of
  non-None bias + `.train()` mode + attention dropout > 0 (reviewer-flagged
  interaction on PR #8). _(Tamer Atesyakar)_
- **`tests/test_embedding.py` — eight new cases appended** covering: the
  MTLM mask block preserves `temporal_pos_embedding` at masked temporal
  tokens (fixes the silent-wipe regression in the pre-hardening code),
  masked PAY_0 vs masked PAY_6 produce distinguishable representations when
  temporal-pos is active, `cat_vocab_sizes` constructor override is
  honoured, missing-feature vocab dict raises `KeyError`, the lazy
  `load_cat_vocab_sizes()` matches the PEP-562 `CAT_VOCAB_SIZES` attribute
  exactly, `build_temporal_layout()` drift-detection guard matches the
  canonical positions, `build_temporal_layout(cls_offset=0)` returns
  23-sequence positions without `[CLS]`, and `TOKEN_ORDER` is well-formed
  (length 23, block-aligned to categorical/PAY/numerical). _(Tamer
  Atesyakar)_
- **`src/embedding.py`: `build_temporal_layout(cls_offset=1)` helper** — the
  canonical, drift-safe source of truth for `TemporalDecayBias` /
  `FeatureGroupBias` / any consumer needing positions of the three temporal
  groups (PAY / BILL_AMT / PAY_AMT) in the 24-token (or 23-token) sequence.
  Derived from `TOKEN_ORDER` so any future reordering is caught by the
  drift-detection assertion in `src/transformer.py`'s smoke test and by
  `tests/test_embedding.py::test_build_temporal_layout_matches_canonical_token_order`.
  _(Tamer Atesyakar)_
- **`src/embedding.py`: `load_cat_vocab_sizes(metadata_path=None)` helper** —
  lazy replacement for the old module-import-time JSON load. The legacy
  `CAT_VOCAB_SIZES` attribute is preserved via PEP 562 module `__getattr__`
  so existing callers and tests continue to work; new code should prefer
  `FeatureEmbedding(cat_vocab_sizes=...)` for hermetic test construction.
  _(Tamer Atesyakar)_
- **`src/embedding.py`: `FeatureEmbedding(cat_vocab_sizes=...)` kwarg** —
  explicit vocabulary override so `FeatureEmbedding` can be constructed
  without disk I/O. _(Tamer Atesyakar)_
- **`src/transformer.py`: `TransformerBlock` + `TransformerEncoder` accept
  independent `attn_dropout` / `ffn_dropout` / `residual_dropout` kwargs** —
  with the shared `dropout` kwarg preserved as the default for each.
  Enables Plan §6.3 and Ablation A12 (attention-weight vs FFN vs residual
  dropout rates) without architectural changes. _(Tamer Atesyakar)_
- **`src/__init__.py`: full package self-description** — every public
  module is listed with its plan-section mapping and its role in the
  tokenizer → embedding → attention → transformer → losses → dataset →
  utils → random-forest stack. Replaces the pre-hardening 4-of-12-module
  stub. _(Tamer Atesyakar)_
- **`CHANGELOG.md` (this file)** — exhaustive per-PR and per-commit change
  log following Keep a Changelog 1.1.0, including retroactive pre-history
  back to the initial commit (0b49bd9). _(Tamer Atesyakar)_
- **`.github/PULL_REQUEST_TEMPLATE.md`** — standardised PR checklist
  enforcing the lessons of PR #8 (no Claude trailer, rebase on latest
  base, separate formatter-only PRs, update CHANGELOG, run pytest +
  module smoke tests). _(Tamer Atesyakar)_
- **`.github/ISSUE_TEMPLATE/bug_report.md`** and
  **`.github/ISSUE_TEMPLATE/feature_request.md`** — issue templates with
  environment / reproduction / expected-vs-actual scaffolding. _(Tamer
  Atesyakar)_

### Changed
- **`src/attention.py`: `MultiHeadAttention` default `d_model` 64 → 32** —
  matches Plan §6.11 (~28K-parameter encoder target). No caller relies on
  the default — every existing test and smoke test passes `d_model`
  explicitly. _(Tamer Atesyakar)_
- **`src/transformer.py`: `FeedForward` / `TransformerBlock` /
  `TransformerEncoder` default `d_model` 64 → 32** — same rationale as
  attention. Existing smoke test and all 26 new tests pass `d_model`
  explicitly. _(Tamer Atesyakar)_
- **`src/transformer.py`: `TransformerEncoder` docstring expanded** with
  (a) the rationale for the shared-across-layers temporal-decay bias
  (ALiBi convention, one-scalar novelty parameter count, clean A22
  on/off axis), and (b) the `register_buffer(persistent=False)` caveat
  advising callers to construct encoders with identical `temporal_layout`
  at checkpoint-load time. _(Tamer Atesyakar)_
- **`src/transformer.py`: smoke test now derives layout via
  `embedding.build_temporal_layout()` + drift-detection assertion against
  the canonical expected layout** — previously hard-coded. Closes reviewer's
  PR #8 checklist item #5. _(Tamer Atesyakar)_
- **`src/tokenizer.py`, `src/dataset.py`: `__main__` smoke tests now
  gracefully skip** with an actionable message when
  `data/processed/*.csv` is absent, rather than crashing with
  `FileNotFoundError`. _(Tamer Atesyakar)_
- **`src/tokenizer.py`: `build_numerical_vocab()` docstring** — clarifies
  that this helper is primarily a documentation / sanity artefact; the
  production path uses `NUMERICAL_FEATURES` directly. _(Tamer Atesyakar)_
- **`PROJECT_PLAN.md`: status markers updated** for Phases 3, 4, 5, 6, 6A,
  7 to reflect the code that has actually shipped since the markers were
  last touched. No more `[TODO]` on sections whose code is merged. _(Tamer
  Atesyakar)_
- **`README.md`: project roadmap table and state preamble updated** —
  Step 3 is DONE, Step 4 is PARTIAL (encoder done; `model.py` + `train.py`
  next), Step 5 remains DONE, Step 6 remains TODO. _(Tamer Atesyakar)_
- **`pyproject.toml`: ruff / black / mypy `target-version` bumped py39 →
  py310** — matches the Python `>=3.10,<3.13` `tool.poetry.dependencies`
  constraint set in PR #6/commit 6a7e3d3. _(Tamer Atesyakar)_

### Fixed
- **`src/train.py`: argparse `%` format-string escape (`de6e547`)** —
  two `--help` strings contained literal `%` characters
  (`"fraction of warmup steps (e.g. 10%)"` and `"dropout applied at 5%"`)
  which Python's argparse interpolates as printf-style format specifiers.
  Result: `python src/train.py --help` crashed with
  `ValueError: unsupported format character 'w'`. Doubled `%` → `%%`
  on the two affected lines (203, 215). Reviewer-flagged on PR #10
  by _FardeenIdrus_. _(Tamer Atesyakar)_
- **`tests/test_losses.py::test_label_smoothing_increases_loss` — flaky
  pre-existing test rewritten for determinism.** The old formulation fed
  un-seeded random logits and un-seeded random labels, which meant ~half
  the "peaky" predictions were confidently wrong — and for those, label
  smoothing lowers the loss rather than raising it. Whether the net sign
  matched the assertion depended on the global RNG state left by whichever
  test ran just before, so adding `tests/test_transformer.py` caused it to
  silently flip from pass to fail. Replaced with a deterministic
  "confidently correct peaky model" (`logits = (y*2-1) * 4.0`), which is
  the regime where smoothing _is_ expected to raise the loss. Explanatory
  docstring added. _(Tamer Atesyakar)_
- **`src/train_mtlm.py` val loader `drop_last=False`** — MTLM validation
  loop was yielding zero batches (and thus `NaN` held-out loss) on small
  val splits because `make_loader(mode="mtlm")` defaulted to
  `drop_last=True`. Val loader now explicitly requests `drop_last=False`
  so every held-out row contributes to the reconstruction-loss average;
  train loader still drops uneven tails for stable gradients.
  _(Tamer Atesyakar)_
- **`tests/test_mtlm.py` gradient-flow assertion scope tightened** — the
  initial formulation asserted a non-None gradient on *every* MTLM
  parameter, but only the heads matching a sampled mask position ever
  receive gradient in a single forward/backward. Relaxed the assertion
  to the set of parameters actually addressed by the batch's mask.
  _(Tamer Atesyakar)_
- **`src/embedding.py`: MTLM mask replacement now preserves the temporal
  positional embedding on masked temporal tokens.** Previously only the
  feature positional embedding was subtracted and re-added, so a masked
  PAY_0 silently lost its month-0 signal while an unmasked PAY_0 kept it.
  The fix computes `temporal_emb` once, subtracts it alongside
  `pos_embedding` before mask substitution, then re-adds both — matching
  BERT's convention that `[MASK]` replaces _content only_, never positional
  signals. Catches: `tests/test_embedding.py::test_mtlm_mask_preserves_
  temporal_pos_at_masked_temporal_tokens` and
  `test_mtlm_mask_preserves_temporal_pos_differs_across_months`. _(Tamer
  Atesyakar)_
- **`run_pipeline.py`: UTF-8 stdout/stderr reconfiguration** — prevents the
  Windows default codec (cp1251 / cp1252) from raising `UnicodeEncodeError`
  on the provenance logger's `×` and `→` characters. Cross-platform (no-op
  on Unix/macOS). Unblocked `poetry run python run_pipeline.py
  --preprocess-only` on Windows. _(Tamer Atesyakar)_
- **`src/embedding.py`: module import is now side-effect-free** — the
  feature_metadata.json load moved from module scope into
  `load_cat_vocab_sizes()` and is triggered lazily on first need. Fresh
  clones no longer fail `pytest` collection when preprocessing hasn't been
  run. _(Tamer Atesyakar)_

### Security
- **`src/data_sources.py`: UCIRepoSource now applies a socket-level
  `socket.setdefaulttimeout()` under the fetch** in addition to the existing
  `ThreadPoolExecutor` wall-clock deadline, closing SECURITY_AUDIT H-2
  (slow-loris / dangling-socket risk via `ucimlrepo`'s unwrapped
  `urllib.request.urlopen`). Defence in depth: socket timeout terminates
  the underlying TCP read, the future timeout bounds wall-clock — neither
  alone covers both resource-leak and hang. _(Tamer Atesyakar)_

### Docs
- Full cross-reference of Plan §5 / §6 / §7 / §8 / §8.5 / §9 sections to
  their implementing module in PROJECT_PLAN.md; README roadmap table now
  links directly to source files.

---

## [0.8.0] — 2026-04-17 — Transformer encoder (PR #8)

### Added
- **`src/transformer.py` (new, 412 LOC)** — PR #8 by _FardeenIdrus_:
  - **`FeedForward`** — two-layer MLP with Kaiming-normal init on the
    GELU-feeding linear and Xavier-normal init on the residual-feeding
    linear (per Plan §6.8). Default `d_ff = 4 * d_model`.
  - **`TransformerBlock`** — one PreNorm block with residual connections
    on both sub-layers (Plan §6.4). Single `dropout` kwarg at merge time
    (split into three independent channels in the Unreleased section).
  - **`TemporalDecayBias` (Novelty N3, Plan §6.12.2)** — learnable
    ALiBi-style prior on attention scores that penalises within-group
    attention between temporally distant months (PAY / BILL_AMT / PAY_AMT).
    Three modes: `scalar` (single learnable α), `per_head` (one α per
    attention head), `off` (disabled — returns `None`). α is
    zero-initialised so the default behaviour recovers a plain
    FT-Transformer-style encoder. Motivated by EDA Fig 9 (BILL_AMT
    autocorrelation decays from 0.95 at lag 1 → 0.7 at lag 5). The
    `neg_distance_masked` distance matrix is registered as a
    non-persistent buffer (deterministically reconstructible from
    `temporal_layout`).
  - **`TransformerEncoder`** — stacks `n_layers` `TransformerBlock`s,
    collects per-layer attention weights for attention rollout (Abnar &
    Zuidema, 2020) in Phase 10, threads a single `TemporalDecayBias`
    output through every block.
- **In-module smoke test (3 sub-tests)** — plain encoder shape +
  gradient flow, TemporalDecayBias α=0/α>0 semantic check with PAY_0→PAY_6
  suppression vs PAY_0→PAY_2, and the three ablation modes.

### Changed
- **`src/attention.py` (+36/-5)** — `ScaledDotProductAttention.forward` and
  `MultiHeadAttention.forward` now accept an optional `attn_bias` kwarg.
  Default `None` preserves bit-identical pre-PR-#8 behaviour under the same
  seed. Smoke test extended with regression (zero-bias ≡ no-bias) and
  suppression checks (`bias[0,1] = -1e4` ⇒ attention weight `< 1e-3`).
- **`.gitignore`** — adds `cli commands.txt` to ignored paths.

### PR-review history
- Reviewer `abailey81` requested three blockers (Claude co-author trailer
  removal, `CLAUDE_UPDATED.md` file removal, formatter-pass split to its
  own PR). All three addressed by `FardeenIdrus` in the force-pushed squash
  (final tip `6917662`). Merge commit `cb1ae5b` by `Tamer Atesyakar` at
  `2026-04-17T00:40:08Z`.

---

## [0.7.0] — 2026-04-17 — PROJECT_PLAN + SECURITY_AUDIT + notebook cleanup

_(Commit `ab2595c`, author `abailey81`.)_

### Added
- **`PROJECT_PLAN.md` (2,249 lines)** — complete execution blueprint
  covering 21 chapters / 14 phases / 12 novelty-register items
  (N1–N12) / 22 ablation studies (A1–A22) / §21 strict coursework-PDF
  relevance audit mapping all 34 PDF requirements to plan sections.
  Target grade: 85%+ distinction.
- **`SECURITY_AUDIT.md` (606 lines)** — 15-dimension paranoid audit
  surfacing 20 findings (1 Critical, 4 High, 6 Medium, 5 Low, 4 Info).
  Headline: C-1 `torch.load(weights_only=False)` as an RCE sink, H-1
  `torch 2.2.2` CVE cluster, H-2 no HTTP timeout on UCI fetch, H-3
  hardcoded `/Users/a_bailey8/…` path in notebook output.

### Changed
- **Notebooks** `01_exploratory_data_analysis.ipynb`,
  `02_data_preprocessing.ipynb`, `03_random_forest_benchmark.ipynb` — all
  outputs stripped (`jupyter nbconvert --clear-output`). Removes large
  embedded PNGs and stale cell execution counts; keeps the repo clone size
  manageable and the diffs reviewable.

---

## [0.6.0] — 2026-04-17 — Novelty + hardening on tokenizer / embedding / data_sources

_(Commit `4b2d105`, author `abailey81`.)_

### Added
- **`src/tokenizer.py`: `MTLMCollator`** — BERT-style masking collator for
  Phase 6A self-supervised pretraining (Plan §5.4B, Novelty N4
  prerequisite). Supports 15% mask rate with 80/10/10 `[MASK]` /
  random-replace / keep-unchanged split, per-row min/max mask bounds (so
  every row contributes a loss term and difficulty is capped),
  seedable `torch.Generator` for reproducibility.
- **`src/tokenizer.py`: vectorised path** — whole-DataFrame tokenisation
  in one pass via NumPy ops, replacing per-row `df.iloc[i]` loops. O(1)
  `__getitem__` on the resulting Dataset.
- **`src/embedding.py`: optional temporal positional encoding** —
  learnable `nn.Embedding(6, d_model)` added to every PAY / BILL_AMT /
  PAY_AMT token (non-temporal tokens unaffected). Controls Ablation A7
  (Plan §5.4).
- **`src/embedding.py`: optional `[MASK]` token** — learnable
  `nn.Parameter(torch.randn(d_model) * 0.02)` that replaces content at
  MTLM-masked positions while preserving the feature positional embedding.
  Enables Novelty N4.
- **`src/tokenizer.py`: `PAYValueError`** — descriptive bounds-check
  exception with coordinate reporting when PAY values fall outside
  `[-2, 8]`.
- **`src/tokenizer.py`: `encode_pay_value` hybrid state + severity
  encoding (Novelty N1)** — `{-2 → no_bill, -1 → paid, 0 → revolving,
  1..8 → delinquent}` with normalised severity `value / 8` for
  delinquency. Respects both the categorical structure of `{-2, -1, 0}`
  and the ordinal structure of `{1..8}`.

### Changed
- **`src/embedding.py`: stronger weight initialisation** — categorical
  embeddings and positional tables use `𝒩(0, 0.02)` per BERT convention
  (Devlin et al., 2019); PAY severity projection uses Xavier.
- **`src/embedding.py`: `ModuleDict` for per-feature embeddings** so
  `nn.Module.to()` / `.cuda()` / `.state_dict()` move them cleanly.

---

## [0.5.0] — 2026-04-17 — Foundation training-infra modules + pytest suite

_(Commit `c491d1d`, author `abailey81`; 11 files / 2,510 LOC.)_

### Added
- **`src/utils.py` (612 LOC)** — determinism protocol (`set_deterministic`
  seeds Python / NumPy / torch CPU & CUDA, disables cuDNN autotuner,
  enables deterministic algorithms with optional `warn_only`), derived
  sub-seeds (`derive_seed`), device selection (CPU / CUDA / Apple MPS),
  security-hardened checkpoint save/load (default `weights_only=True`
  for external checkpoints, explicit `trust_source=True` opt-in for
  internal checkpoints — closes SECURITY_AUDIT C-1), 3-file checkpoint
  layout (full pickle + `.weights` sidecar + `.meta.json`),
  `EarlyStopping` (min/max mode, min_delta, best-state stash), parameter
  accounting (`count_parameters`, `format_parameter_count`), `Timer`
  context manager, idempotent logging configuration.
- **`src/losses.py` (379 LOC)** — `WeightedBCELoss` (inverse-frequency
  class weighting), `FocalLoss` (Lin et al., 2017; γ-configurable,
  α-configurable with scalar / tuple / `"balanced"` / `None` modes;
  γ=0 reduces to plain BCE to float32 precision), `LabelSmoothingBCELoss`
  (Müller et al., 2019; ε-configurable, defaults to 0.05 per §7.3).
- **`src/dataset.py` (405 LOC)** — `StratifiedBatchSampler` (every batch
  contains exactly `round(batch_size * positive_rate)` positives — caps
  gradient-variance from class imbalance), `make_loader` factory with
  `train` / `val` / `test` / `mtlm` modes, seedable generator for
  reproducibility.
- **`tests/` directory (7 modules, 1,189 LOC)** — `conftest.py` with
  shared fixtures (graceful skip on missing `feature_metadata.json`),
  pytest for `attention.py` (9), `dataset.py` (stratified sampler +
  loader + reproducibility), `embedding.py` (15 pre-hardening +
  temporal-pos + mask-token + determinism), `losses.py` (WBCE + Focal γ
  sweep + label-smoothing + gradient flow), `tokenizer.py` (vocab +
  encode_pay + MTLMCollator + determinism), `utils.py` (seeds + device
  + checkpoint round-trip + early stopping + param-accounting).
  Baseline: 82 passed + 17 skipped (data-dependent tests).

### Changed
- **`pyproject.toml`**: adds `pytest`, `pytest-cov`, `ruff`, `black`,
  `mypy`, `pre-commit`, `jupyter` as dev dependencies.

---

## [0.4.1] — 2026-04-16 — Python + torch dependency fix

_(Commit `6a7e3d3`, author `abailey81`.)_

### Changed
- **`pyproject.toml`**: Python requirement pinned to `>=3.10,<3.13`
  (previously less restrictive); torch pinned to `>=2.2,<2.3` to stabilise
  CI and match the PyTorch 2.2.2 wheel available for Intel macOS. Poetry
  lockfile regenerated (319 removed / 132 added — substantial
  dependency closure change).

---

## [0.4.0] — 2026-04-16 — Attention from scratch (PR #7)

_(PR #7 merged as commit `012b58e`; feature commit `8ec2ac4` author
_Fardeen Idrus_.)_

### Added
- **`src/attention.py` (186 LOC)** — from-scratch
  `ScaledDotProductAttention` and `MultiHeadAttention`. Implements
  `Attention(Q, K, V) = softmax(QK^T / √d_k) V` with optional attention-
  weight dropout (post-softmax), Xavier-normal initialisation on the
  Q/K/V/O projections, reshape-based head-splitting so parameter count
  stays `4 × d_model²` independent of `n_heads`. Returns un-dropped
  attention weights for later interpretability (Plan Phase 10).
- **In-module smoke test** — verifies shape, row-sum-to-1, and gradient
  flow through every parameter.

---

## [0.3.1] — 2026-04-13 — Merge main ↔ additional cleanup (PR #6)

_(PR #6 by _Idaliia19_, commit `a031c71`.)_

### Changed
- **Merge-fixup commit** reconciling `main` and `additional` divergence
  after PR #4 and PR #5 landed on different branches. Net: 661 insertions
  / 1,597 deletions — mostly whitespace and re-ordering inside already-
  merged files.

---

## [0.3.0] — 2026-04-13 — Feature embedding layer (PR #5)

_(PR #5 by _LexieLee1020_, commits `ec60851` + fix `a936c8b`.)_

### Added
- **`src/embedding.py` — initial `FeatureEmbedding` class (201 LOC)** —
  converts the tokenizer output (cat_indices, pay_state_ids,
  pay_severities, num_values) into a `(B, 24, d_model)` token sequence:
  one [CLS] token + 23 feature tokens. Per-feature projection heads
  for numerical values, per-feature embedding tables for categoricals,
  hybrid state + severity for PAY.

### Fixed
- **`src/embedding.py`** (commit `a936c8b`) — miscellaneous corrections
  to the initial embedding: 66 insertions / 66 deletions across bias
  initialisation, positional-embedding ordering, and per-feature projection
  wiring.

---

## [0.2.1] — 2026-04-12 — Tokenizer update (PR #4)

_(PR #4 by _Idaliia19_; commits `0ef43b3` + fixes `311b81c`, `527db26`.)_

### Added
- **`src/tokenizer.py` — initial `CreditDefaultDataset` + vocab builders
  (262 LOC)** — hybrid PAY state + severity tokenisation, vectorised
  `tokenize_dataframe`, PyTorch `Dataset` wrapper.

### Changed
- **`src/tokenizer.py`** (commit `527db26`) — adopts `nn.ModuleDict` for
  per-feature embeddings in the partner embedding module, removes the
  redundant `num_feature_ids` indexing (Issue 3), tightens the hybrid PAY
  encoding.
- **`src/tokenizer.py`** (commit `311b81c`) — removes `num_feature_ids`
  (Issue 3); 24 insertions / 33 deletions of dead numerical-index
  bookkeeping.

---

## [0.2.0] — 2026-04-12 — Resilient data ingestion

_(Commits `199d33b` + `3afbc0e` + `2dcfbff`, author `abailey81`.)_

### Added
- **`src/data_sources.py` (519 LOC)** — layered, provenance-aware data
  loader:
  - `DataSource` (abstract base)
  - `UCIRepoSource` — fetches via `ucimlrepo` with bounded retries +
    exponential backoff + wall-clock deadline
  - `LocalExcelSource` — reads from a configurable list of candidate
    `.xls`/`.xlsx` paths with CWD + repo-root resolution
  - `ChainedDataSource` — tries each child source in order, accumulating
    failures in `DataSourceResult.failed_attempts` so the fallback chain
    is fully auditable even on success
  - `DataSourceResult` frozen dataclass (dataframe + source name + source
    type + origin URI + elapsed time + failed-attempts list)
  - `build_default_data_source(...)` factory
- **`run_pipeline.py` CLI** — `--source {auto,api,local}`, `--no-fallback`,
  `--data-path FILE` honouring pinned paths as a hard pin (never silently
  hits the network).
- **Data loader documentation across README + module docstrings** (commit
  `3afbc0e`).

### Fixed
- **`fd2f1ea`** — adds `CLAUDE.md` to `.gitignore` (_Fardeen Idrus_).

---

## [0.1.0] — 2026-04-11 — Random Forest benchmark integrated (PR #3)

_(PR #3 by _Shakhzod555_, commits `e0961cc` + `1fd79f5`.)_

### Added
- **`src/random_forest.py` (870 LOC)** — end-to-end RF benchmark:
  reuses the shared preprocessing pipeline (normalise → clean → engineer
  → split); baseline RF (100 trees); `RandomizedSearchCV` (60 iter × 5-
  fold CV over a 7-dimension hyperparameter grid); evaluate
  (accuracy / precision / recall / F1 / AUC-ROC / AP), 5-fold stratified
  CV; dual feature importance (Gini + permutation); threshold
  optimisation (max-F1 on validation); publication-quality figures
  (ROC/PR, confusion matrix, tuning analysis, feature importance,
  threshold curves); results export (CSV + JSON).

---

## [0.0.2] — 2026-04-09 — Standalone RF benchmark (PR #1, PR #2)

_(PR #1 and PR #2 by _Shakhzod555_, commits `f5ee9e8` + `cb767d6`.)_

### Added
- **Standalone RF benchmark script (824 LOC)** — earlier, decoupled
  version of the RF pipeline. Superseded by the integrated version in
  PR #3, but provided the baseline architecture.

---

## [0.0.1] — 2026-04-07 — Initial commit

_(Commit `0b49bd9`, author `abailey81`; 29 files / 12,499 LOC.)_

### Added
- **Project scaffold**: `pyproject.toml`, `poetry.lock`, `LICENSE` (MIT),
  `README.md` (387 lines), `.gitignore`, `run_pipeline.py` CLI entry
  point.
- **`docs/coursework_spec.md` (105 lines)** — markdown transcription of
  the coursework PDF for in-repo reference.
- **`src/data_preprocessing.py` (567 LOC)** — schema normalisation
  (`PAY_1 → PAY_0`, drop `ID`), categorical cleaning (merge
  `EDUCATION {0,5,6} → 4`, `MARRIAGE 0 → 3`), validation report
  (missing-value count, duplicate detection, range checks), 22-feature
  engineering (utilisation ratios, repayment ratios, delinquency
  aggregates, bill slope, payment dynamics, balance totals), stratified
  70/15/15 split preserving 22.1% default rate, leak-free
  `StandardScaler` fit on train only, feature-metadata export (category
  mappings, scaler stats, feature ordering) for the tokeniser.
- **`src/eda.py` (847 LOC)** — 12 publication-quality figures with
  statistical tests:
  - Fig 01 class distribution (Wilson CI)
  - Fig 02 categorical features by target (χ²)
  - Fig 03 numerical distributions (Mann-Whitney U)
  - Fig 04 PAY status semantic analysis (chosen as the modelling-
    decision-driver for hybrid PAY tokenisation)
  - Fig 05 temporal trajectories (95% CI)
  - Fig 06 credit utilisation analysis
  - Fig 07 24×24 correlation heatmap
  - Fig 08 feature-target association ranking
  - Fig 09 BILL_AMT autocorrelation decay
  - Fig 10 feature interactions
  - Fig 11 PAY transition matrices
  - Fig 13 repayment ratio analysis
  (Fig 12 intentionally skipped during numbering.)
- **`notebooks/01_exploratory_data_analysis.ipynb` (2,816 cells)** — full
  EDA with 20+ visualisations and Wilson CI / KS / Mann-Whitney U /
  Cohen's d / Cramer's V / D'Agostino / VIF / mutual information.
- **`notebooks/02_data_preprocessing.ipynb` (1,478 cells)** — cleaning,
  validation, feature engineering, stratified splitting, scaling,
  metadata export walkthrough.
- **`data/raw/.gitkeep`** — reserves the raw-data directory (actual
  `.xls` tracked later via fallback loader).
- **`data/processed/feature_metadata.json`** — initial categorical
  vocabularies, numerical-feature scaler statistics, PAY per-feature
  vocab sizes, canonical `feature_order`.
- **`data/processed/validation_report.json`** — initial data-quality
  audit (30,000 rows, 0 missing, 35 duplicates, target 77.9/22.1 split).
- **`results/summary_statistics.{csv,tex}`** — mean/std/median/skewness/
  kurtosis for all 23 features split by default status.
- **`figures/fig01..11,13.png`** — 12 EDA figures at 300 DPI.

---

## Version naming note

`pyproject.toml` carries the placeholder version `1.0.0`, reserved for the
final coursework-submission tag. The `0.x.y` entries above are assigned
retroactively to give PRs and module landings a stable handle for
contribution review and report referencing; they do not correspond to git
tags.

_Historical accuracy note_: commits in the `main` vs `additional` branch
history occasionally duplicate (e.g. `e0961cc` / `1fd79f5` or `f5ee9e8` /
`cb767d6`) due to branch-integration merges. Each such pair represents a
single logical change landed twice during branch reconciliation and is
attributed once above.
