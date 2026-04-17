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

## [Unreleased]

Working branch `feature/hardening-and-test-coverage`. Not yet merged to
`additional` at the time of writing.

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
