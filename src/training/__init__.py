"""Training pipeline: supervised + MTLM loops, loaders, losses, utilities.

This sub-package gathers every piece of machinery that runs *after* feature
engineering and tokenisation are done: PyTorch :class:`~torch.utils.data.DataLoader`
wrappers, binary-classification losses tuned for the 22.1 %-positive regime,
AdamW + cosine-warmup plumbing, early stopping / checkpointing / seeding
helpers, and the two top-level entry points ``train.py`` (supervised) and
``train_mtlm.py`` (self-supervised pretraining described in plan §7.5 / N4).

Sub-modules
-----------
``dataset``
    DataLoader factory (``make_loader``) and :class:`StratifiedBatchSampler`
    for fixed per-batch class ratios.
``losses``
    :class:`WeightedBCELoss`, :class:`FocalLoss`, :class:`LabelSmoothingBCELoss`
    plus the ``compute_pos_weight`` / ``balanced_alpha`` helpers used by all
    three.
``utils``
    Determinism, device resolution, checkpoint save/load (security-critical —
    see :func:`load_checkpoint`), :class:`EarlyStopping`, param counting.
``train``
    Supervised training loop with optional fine-tuning from an MTLM encoder.
``train_mtlm``
    Masked tabular language model pretraining (BERT 80/10/10 on our tokens).

Design note
-----------
The two training loops deliberately share ``build_cosine_warmup_schedule`` via
copy rather than import — they were authored as standalone entry points and
keeping the schedules independent avoids cross-coupling when one loop needs to
diverge (e.g. layer-wise LR decay at fine-tune time)."""
