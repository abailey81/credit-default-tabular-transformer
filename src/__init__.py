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
                      :class:`tokenizer.CreditDefaultDataset`,
                      :class:`tokenizer.MTLMCollator` (supports Novelty N4).
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
* :mod:`random_forest` — hyperparameter-tuned Random Forest benchmark
                         (Plan §9).

Every consumer of the raw dataset routes through
:func:`data_sources.build_default_data_source` so that the API → local
fallback semantics apply uniformly across the entire pipeline.
"""
