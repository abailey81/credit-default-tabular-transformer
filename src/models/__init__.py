"""Transformer architecture subpackage.

Houses every learnable module that sits between the tokeniser and the loss.
The split is deliberate: ``attention`` is the hand-rolled SDPA + multi-head
primitive (coursework requires a from-scratch Q/K/V path); ``transformer``
layers PreNorm residual blocks and the two additive-bias priors (N2 feature
group, N3 temporal decay); ``model`` is the end-to-end ``TabularTransformer``
including the optional auxiliary PAY_0 head (N5); and ``mtlm`` is the BERT-
style masked-token pretraining head + loss (N4).

Key public symbols (re-exported via submodules, not here, to keep the import
graph shallow — ``from src.models.attention import MultiHeadAttention`` etc.):

- ``attention``: ``ScaledDotProductAttention``, ``MultiHeadAttention``
- ``transformer``: ``FeedForward``, ``TransformerBlock``, ``TransformerEncoder``,
  ``FeatureGroupBias``, ``TemporalDecayBias``
- ``model``: ``TabularTransformer``
- ``mtlm``: ``MTLMHead``, ``MTLMModel``, ``mtlm_loss``, ``MTLMLossComponents``

Design choice
-------------
We do NOT re-export symbols at package level. Downstream callers always go
through the submodule (e.g. ``from src.models.model import TabularTransformer``)
so the cost of touching one submodule is visible in imports elsewhere and
circular-import risk stays low — ``model`` imports from ``transformer`` which
imports from ``attention``; keeping that DAG explicit pays off when debugging
checkpoint-loading issues.

Critical invariant
------------------
The 24-token sequence layout (``[CLS, 3 cat, 6 PAY, 14 num]``) is defined by
``src.tokenization.embedding.TOKEN_ORDER``; every module here takes that as
ground truth via ``build_group_assignment`` / ``build_temporal_layout``. Any
reshuffling must happen there, never here.
"""
