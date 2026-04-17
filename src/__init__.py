"""
Credit Card Default Prediction — from-scratch tabular transformer + RF benchmark.

Architecture layers (bottom-up, matching Plan §§3–8):

Data & preprocessing
--------------------
* :mod:`data_sources`        — resilient multi-source loader (UCI ML
                               Repository API → local manual ``.xls``
                               fallback, with provenance tracking).
* :mod:`data_preprocessing`  — schema normalisation, cleaning, 22-feature
                               engineering, stratified 70/15/15 split,
                               leak-free scaling, metadata export.
* :mod:`eda`                 — 12 publication-quality EDA figures with
                               statistical tests (Plan §4).

Transformer stack (Plan §§5–6 / §§8–8.5)
----------------------------------------
* :mod:`tokenizer`  — hybrid PAY state+severity tokenisation (Novelty N1),
                      :class:`tokenizer.CreditDefaultDataset` (now exposes
                      ``pay_raw`` — 11-class shifted PAY values for MTLM /
                      N5 targets), :class:`tokenizer.MTLMCollator`
                      (supports Novelty N4).
* :mod:`embedding`  — :class:`embedding.FeatureEmbedding` with per-feature
                      projections, [CLS] token, optional temporal positional
                      encoding (Ablation A7), optional [MASK] token for MTLM
                      pretraining; :func:`embedding.build_temporal_layout`
                      helper (drift-safe canonical layout).
* :mod:`attention`  — from-scratch
                      :class:`attention.ScaledDotProductAttention` and
                      :class:`attention.MultiHeadAttention` with an
                      ``attn_bias`` hook for architectural novelties.
* :mod:`transformer` — :class:`transformer.FeedForward`,
                      :class:`transformer.TransformerBlock` (PreNorm,
                      independently ablatable attention / FFN / residual
                      dropout), :class:`transformer.TemporalDecayBias`
                      (Novelty N3; ALiBi-style learnable decay), and
                      :class:`transformer.TransformerEncoder`.
* :mod:`model`      — :class:`model.TabularTransformer`, the top-level
                      end-to-end model (Plan §6.7 / §6.10 / §6.11). Wires
                      tokenizer → embedding → encoder → pool → classification
                      head → logit. Exposes every architectural switch
                      (``pool``, ``use_temporal_pos``, ``temporal_decay_mode``,
                      independently-ablatable dropout channels,
                      ``aux_pay0`` for Novelty N5 / Ablation A16), plus a
                      ``load_pretrained_encoder`` helper for the §8.5.5
                      MTLM-fine-tune transition.

Training, evaluation, baseline
------------------------------
* :mod:`losses`   — :class:`losses.WeightedBCELoss`,
                    :class:`losses.FocalLoss` (Plan §7.2, Ablation A11),
                    :class:`losses.LabelSmoothingBCELoss`.
* :mod:`dataset`  — :class:`dataset.StratifiedBatchSampler` +
                    :func:`dataset.make_loader` factory (supports MTLM mode).
* :mod:`utils`    — determinism protocol, device selection, hardened
                    (weights-only by default) checkpoint save/load,
                    :class:`utils.EarlyStopping`, parameter accounting,
                    UTF-8-safe logging setup.
* :mod:`train`    — Plan §8 supervised training loop: AdamW + cosine warmup
                    (§8.2) + gradient clipping (§8.3) + early stopping on
                    val AUC-ROC (§8.5) + optional two-stage LR fine-tuning
                    (§8.5.5) + optional multi-task PAY_0 auxiliary head
                    (§8.6 / N5). Invoke via ``python src/train.py`` or
                    ``from train import main; main([...])``. Produces per-run
                    ``config.json``, ``train_log.csv``, ``test_metrics.json``,
                    ``test_predictions.npz``, ``test_attn_weights.npz``,
                    and a hardened checkpoint under ``--output-dir``.
* :mod:`mtlm`     — Plan §8.5 / Novelty N4: Masked Tabular Language
                    Modelling. :class:`mtlm.MTLMHead` with per-feature
                    prediction heads (3 categorical + 6 PAY + 14 numerical),
                    :func:`mtlm.mtlm_loss` with entropy-normalised CE +
                    variance-normalised MSE, and
                    :class:`mtlm.MTLMModel` whose state-dict prefixes are
                    drop-in for
                    :meth:`model.TabularTransformer.load_pretrained_encoder`.
* :mod:`train_mtlm` — Phase 6A pretraining loop. Produces a tiny
                    encoder-only state-dict artefact
                    (``encoder_pretrained.pt``) that the supervised
                    ``train.py`` picks up via ``--pretrained-encoder`` for
                    §8.5.5 two-stage fine-tuning.
* :mod:`random_forest` — hyperparameter-tuned Random Forest benchmark
                         (Plan §9 — 200-iter randomised search across a
                         7-dimensional grid).

Every consumer of the raw dataset routes through
:func:`data_sources.build_default_data_source` so that the API → local
fallback semantics apply uniformly across the entire pipeline.
"""
