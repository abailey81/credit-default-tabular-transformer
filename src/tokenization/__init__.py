"""Tokenisation and feature-embedding subpackage.

Two modules with a tight contract:

* :mod:`src.tokenization.tokenizer` — turns a preprocessed DataFrame into
  the tensor layout the embedding layer consumes: categorical indices,
  hybrid PAY state+severity tensors (Novelty N1), shifted PAY_raw for
  MTLM/N5 targets, numerical values, and labels. Also owns the
  ``MTLMCollator`` used during Phase-6A pretraining (Novelty N4).

* :mod:`src.tokenization.embedding` — turns those tensors into a
  ``(B, 24, d_model)`` sequence with a learnable [CLS] at position 0,
  per-feature-type embeddings, optional temporal positional encoding
  (Ablation A7), and an optional [MASK] vector for MTLM.

Both modules share ``TOKEN_ORDER`` as the canonical token layout. Every
downstream consumer of the 23-feature sequence — attention rollout,
feature-group bias (N2), temporal-decay bias (N3), per-token
interpretability — looks up positions through ``TOKEN_ORDER`` rather than
hard-coded slices, so adding / reordering a feature is a single-line
change here instead of a hunt across the whole codebase.
"""
