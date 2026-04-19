"""Credit Card Default Prediction — from-scratch tabular transformer + RF benchmark.

This package is organised bottom-up, mirroring Plan §§3-8. The layering is
enforced by import direction: data layers never depend on modelling layers,
and modelling layers never depend on training/evaluation. The only shared
contract is :mod:`src.tokenization.tokenizer.TOKEN_ORDER` — every consumer
of the 23-feature sequence routes through it to stay drift-safe.

Architecture layers (bottom-up, matching Plan §§3-8):

Data & preprocessing (:mod:`src.data`, :mod:`src.analysis`)
-----------------------------------------------------------
* :mod:`src.data.sources`        — resilient multi-source loader (UCI ML
                                   Repository API -> local manual ``.xls``
                                   fallback, with provenance tracking).
* :mod:`src.data.preprocessing`  — schema normalisation, cleaning,
                                   22-feature engineering, stratified
                                   70/15/15 split, leak-free scaling,
                                   metadata export.
* :mod:`src.analysis.eda`        — 12 publication-quality EDA figures
                                   with statistical tests (Plan §4).

Transformer stack (:mod:`src.tokenization`, :mod:`src.models`)
--------------------------------------------------------------
* :mod:`src.tokenization.tokenizer`  — hybrid PAY state+severity
                                       tokenisation (Novelty N1),
                                       ``CreditDefaultDataset`` (exposes
                                       ``pay_raw`` — 11-class shifted PAY
                                       values for MTLM / N5 targets),
                                       ``MTLMCollator`` (supports Novelty N4).
* :mod:`src.tokenization.embedding`  — ``FeatureEmbedding`` with
                                       per-feature projections, [CLS]
                                       token, optional temporal positional
                                       encoding (Ablation A7), optional
                                       [MASK] token for MTLM pretraining;
                                       ``build_temporal_layout`` helper
                                       (drift-safe canonical layout).
* :mod:`src.models.attention`        — from-scratch
                                       ``ScaledDotProductAttention`` and
                                       ``MultiHeadAttention`` with an
                                       ``attn_bias`` hook for architectural
                                       novelties.
* :mod:`src.models.transformer`      — ``FeedForward``,
                                       ``TransformerBlock`` (PreNorm,
                                       independently ablatable attention /
                                       FFN / residual dropout),
                                       ``TemporalDecayBias`` (Novelty N3;
                                       ALiBi-style learnable decay), and
                                       ``TransformerEncoder``.
* :mod:`src.models.model`            — ``TabularTransformer``, the
                                       top-level end-to-end model
                                       (Plan §6.7 / §6.10 / §6.11).
                                       Wires tokenizer -> embedding ->
                                       encoder -> pool -> classification
                                       head -> logit. Exposes every
                                       architectural switch (``pool``,
                                       ``use_temporal_pos``,
                                       ``temporal_decay_mode``,
                                       independently-ablatable dropout
                                       channels, ``aux_pay0`` for
                                       Novelty N5 / Ablation A16), plus a
                                       ``load_pretrained_encoder`` helper
                                       for the §8.5.5 MTLM-fine-tune
                                       transition.

Training, evaluation, baseline (:mod:`src.training`, :mod:`src.evaluation`, :mod:`src.baselines`)
------------------------------------------------------------------------------------------------
* :mod:`src.training.losses`     — ``WeightedBCELoss``, ``FocalLoss``
                                   (Plan §7.2, Ablation A11),
                                   ``LabelSmoothingBCELoss``.
* :mod:`src.training.dataset`    — ``StratifiedBatchSampler`` +
                                   ``make_loader`` factory (supports
                                   MTLM mode).
* :mod:`src.training.utils`      — determinism protocol, device
                                   selection, hardened (weights-only by
                                   default) checkpoint save/load,
                                   ``EarlyStopping``, parameter
                                   accounting, UTF-8-safe logging setup.
* :mod:`src.training.train`      — Plan §8 supervised training loop:
                                   AdamW + cosine warmup (§8.2) +
                                   gradient clipping (§8.3) + early
                                   stopping on val AUC-ROC (§8.5) +
                                   optional two-stage LR fine-tuning
                                   (§8.5.5) + optional multi-task PAY_0
                                   auxiliary head (§8.6 / N5). Invoke via
                                   ``python -m src.training.train``.
                                   Produces per-run ``config.json``,
                                   ``train_log.csv``,
                                   ``test_metrics.json``,
                                   ``test_predictions.npz``,
                                   ``test_attn_weights.npz``, and a
                                   hardened checkpoint under
                                   ``--output-dir``.
* :mod:`src.models.mtlm`         — Plan §8.5 / Novelty N4: Masked
                                   Tabular Language Modelling.
                                   ``MTLMHead`` with per-feature
                                   prediction heads (3 categorical + 6
                                   PAY + 14 numerical), ``mtlm_loss``
                                   with entropy-normalised CE +
                                   variance-normalised MSE, and
                                   ``MTLMModel`` whose state-dict
                                   prefixes are drop-in for
                                   ``TabularTransformer.load_pretrained_encoder``.
* :mod:`src.training.train_mtlm` — Phase 6A pretraining loop. Produces a
                                   tiny encoder-only state-dict artefact
                                   (``encoder_pretrained.pt``) that the
                                   supervised ``train.py`` picks up via
                                   ``--pretrained-encoder`` for §8.5.5
                                   two-stage fine-tuning.
* :mod:`src.baselines.random_forest` — hyperparameter-tuned Random
                                       Forest benchmark (Plan §9 —
                                       200-iter randomised search across
                                       a 7-dimensional grid).
* :mod:`src.baselines.rf_predictions` — refits the tuned RF and emits
                                        per-row predictions for
                                        calibration / significance /
                                        fairness consumers.
* :mod:`src.evaluation.evaluate`      — aggregates per-seed metrics into
                                        a transformer-vs-RF comparison
                                        table.
* :mod:`src.evaluation.visualise`     — §4 report figures (ROC / PR /
                                        confusion matrices / training
                                        curves / reliability).
* :mod:`src.evaluation.calibration`   — post-hoc calibration
                                        (temperature / Platt / isotonic)
                                        with ECE / MCE / Brier
                                        decomposition.
* :mod:`src.evaluation.fairness`      — subgroup fairness audit across
                                        SEX / EDUCATION / MARRIAGE.
* :mod:`src.evaluation.uncertainty`   — MC-dropout predictive entropy +
                                        BALD mutual info + refuse curve.
* :mod:`src.evaluation.significance`  — paired tests (McNemar / DeLong /
                                        bootstrap) with BH-FDR and
                                        Hanley-McNeil power.
* :mod:`src.evaluation.interpret`     — attention rollout and per-feature
                                        importance vs RF Gini.
* :mod:`src.infra.repro`              — reproducibility gate: regenerates
                                        every derivative artefact and
                                        diffs vs the committed copy.

Every consumer of the raw dataset routes through
``src.data.sources.build_default_data_source`` so that the API -> local
fallback semantics apply uniformly across the entire pipeline. If you
reach for ``pd.read_excel`` or ``ucimlrepo.fetch_ucirepo`` directly, you
are bypassing the provenance tracking the reproducibility gate expects.

Originally scaffolded across PRs #1-#8 (phases 1-11); the subpackage
restructure landed on ``feature/restructure-and-polish``.
"""
