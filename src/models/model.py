"""End-to-end ``TabularTransformer`` — the supervised classification model.

Purpose
-------
Wire ``FeatureEmbedding`` -> ``TransformerEncoder`` -> pooling -> MLP head
into a single ``nn.Module`` with the knobs needed for every ablation in the
coursework and the pretrain-then-finetune pipeline.

Key public symbols
------------------
- ``TabularTransformer`` — the top-level model. Defaults (``d_model=32``,
  ``n_heads=4``, ``n_layers=2``) land at ~28K parameters, inside the plan
  budget of [20K, 40K] for the 21K-row training set.
- ``PAY_0_FEATURE_POSITION_23`` / ``PAY_0_OUTPUT_POSITION_24`` /
  ``FULL_SEQ_LEN`` — layout constants pinned to ``TOKEN_ORDER``; isolating
  them here makes it obvious which call sites break if the tokeniser
  reshuffles.

Design choices
--------------
- Pooling defaults to ``"cls"`` (classic BERT-style). ``"mean"`` and
  ``"max"`` drop the CLS token before pooling so a drifted CLS norm
  cannot swamp the reduction — this fell out of ablation A06.
- The auxiliary PAY_0 head (N5) force-masks the PAY_0 token on every
  forward so the head is always predicting a genuinely unseen label,
  rather than memorising the input slot.
- Weight loading supports two on-disk shapes: the full bundle
  (``utils.save_checkpoint``) and a raw encoder state-dict (as written by
  MTLM pretraining). The raw path tolerates missing head keys so a
  pretrained encoder drops into a fresh classifier cleanly.

Dependencies
------------
Pulls ``FeatureEmbedding`` and the layout helpers from
``src.tokenization.embedding``; pulls ``TransformerEncoder``,
``TemporalDecayBias``, ``FeatureGroupBias`` from ``.transformer``; and
``PAY_RAW_NUM_CLASSES`` from ``src.tokenization.tokenizer`` (for the aux
head output size).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Optional

import torch
import torch.nn as nn

from ..tokenization.embedding import (
    N_FEATURE_GROUPS,
    FeatureEmbedding,
    build_group_assignment,
    build_temporal_layout,
)
from ..tokenization.tokenizer import PAY_RAW_NUM_CLASSES
from .transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder

logger = logging.getLogger(__name__)

# Token layout constants — pinned to TOKEN_ORDER in embedding.py so the
# aux head cannot silently latch onto the wrong slot after a reshuffle.
# Two positions exist for the same feature: before CLS is prepended
# (23-slot feature-only layout, used for mask arrays in the dataset) and
# after (24-slot layout, used by the encoder).
PAY_0_FEATURE_POSITION_23 = 3  # 23-slot, pre-CLS
PAY_0_OUTPUT_POSITION_24 = PAY_0_FEATURE_POSITION_23 + 1  # 24-slot, CLS at 0
FULL_SEQ_LEN = 24  # CLS + 23 features


class TabularTransformer(nn.Module):
    """Embedding -> encoder -> pool -> classification head.

    Architecture
    ------------
    ``FeatureEmbedding`` tokenises the input into 24 slots (CLS + 3
    categorical + 6 PAY status + 14 numerical). The encoder runs
    ``n_layers`` PreNorm transformer blocks with optional N2 / N3
    additive-bias priors. Pooling reduces ``(B, 24, d)`` to ``(B, d)``,
    which the 2-layer MLP head projects to a single default-vs-not logit.

    At the repo defaults (d_model=32, n_heads=4, n_layers=2) the module
    lands at ~28K parameters — inside the plan envelope of [20K, 40K]
    for our 21K-row training set.

    Invariants
    ----------
    - When ``aux_pay0=True`` the embedding's mask-token path is forced on
      so the force-mask trick in ``_apply_aux_force_mask`` has a
      ``[MASK]`` embedding to substitute in.
    - ``use_mask_token`` can be toggled without aux; MTLM pretraining
      needs it but supervised fine-tuning does not (masks are all False).

    Forward returns ``{logit: (B,), [aux_pay0_logits: (B, 11)],
    [attn_weights: list of (B, H, 24, 24)]}``. ``aux_pay0_logits`` is
    only present when ``aux_pay0=True``; ``attn_weights`` only when
    ``return_attn=True`` on the forward call.
    """

    def __init__(
        self,
        *,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        attn_dropout: Optional[float] = None,
        ffn_dropout: Optional[float] = None,
        residual_dropout: Optional[float] = None,
        classification_dropout: float = 0.1,
        pool: Literal["cls", "mean", "max"] = "cls",
        use_temporal_pos: bool = False,
        use_mask_token: bool = False,
        temporal_decay_mode: Literal["off", "scalar", "per_head"] = "off",
        feature_group_bias_mode: Literal["off", "scalar", "per_head"] = "off",
        aux_pay0: bool = False,
        cat_vocab_sizes: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        if pool not in ("cls", "mean", "max"):
            raise ValueError(f"pool must be one of 'cls'/'mean'/'max', got {pool!r}")

        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.pool = pool
        self.use_temporal_pos = use_temporal_pos
        self.aux_pay0 = aux_pay0

        # When aux_pay0 is on, we always force-mask PAY_0 — the embedding
        # must therefore provision a [MASK] token regardless of whether
        # the caller passed use_mask_token. The OR here means aux users
        # don't have to remember to flip both flags.
        self.embedding = FeatureEmbedding(
            d_model=d_model,
            dropout=dropout,
            cat_vocab_sizes=cat_vocab_sizes,
            use_temporal_pos=use_temporal_pos,
            use_mask_token=use_mask_token or aux_pay0,
        )

        # N3 TemporalDecayBias: constructed with cls_offset=1 because the
        # bias operates in the 24-slot space (CLS at 0). Only instantiated
        # when the ablation mode actually needs it — "off" is cheap.
        if temporal_decay_mode != "off":
            temporal_decay: Optional[TemporalDecayBias] = TemporalDecayBias(
                temporal_layout=build_temporal_layout(cls_offset=1),
                seq_len=FULL_SEQ_LEN,
                mode=temporal_decay_mode,
                n_heads=n_heads,
            )
        else:
            temporal_decay = None

        # N2 FeatureGroupBias: same offset convention. N_FEATURE_GROUPS
        # comes from embedding.py so the count matches whatever the
        # tokeniser considers a "group".
        if feature_group_bias_mode != "off":
            feature_group_bias: Optional[FeatureGroupBias] = FeatureGroupBias(
                group_assignment=build_group_assignment(cls_offset=1),
                n_groups=N_FEATURE_GROUPS,
                mode=feature_group_bias_mode,
                n_heads=n_heads,
            )
        else:
            feature_group_bias = None

        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            residual_dropout=residual_dropout,
            temporal_decay=temporal_decay,
            feature_group_bias=feature_group_bias,
        )

        # Classification head: LN -> 2-layer MLP -> single logit. The LN
        # before the MLP is important when pooling is "mean" or "max" —
        # those reductions can drift the activation scale away from the
        # encoder-internal LayerNorms.
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(classification_dropout),
            nn.Linear(d_model, 1),
        )

        # Auxiliary PAY_0 head (N5): same shape as the main classifier but
        # with its own LayerNorm so the two tasks don't share a norm whose
        # affine parameters would have to compromise between them.
        if aux_pay0:
            self.aux_pay0_head: Optional[nn.Sequential] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(classification_dropout),
                nn.Linear(d_model, PAY_RAW_NUM_CLASSES),
            )
        else:
            self.aux_pay0_head = None

        self._init_heads()

    def _init_heads(self) -> None:
        """Kaiming on each head's GELU-feeding linears, Xavier on the last.

        Same convention as ``FeedForward``: variance-preserving (Xavier)
        on the layer that feeds a residual or logit sink, half-linear
        (Kaiming/ReLU) on the layer whose output passes through GELU.
        """
        for container in (self.classifier, self.aux_pay0_head):
            if container is None:
                continue
            linears = [m for m in container if isinstance(m, nn.Linear)]
            for idx, lin in enumerate(linears):
                if idx < len(linears) - 1:
                    nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(lin.weight)
                nn.init.zeros_(lin.bias)

    def _pool(self, hidden: torch.Tensor) -> torch.Tensor:
        """Reduce ``(B, 24, d) -> (B, d)`` per the configured pooling mode.

        For ``"mean"`` / ``"max"`` we drop CLS before reducing because CLS
        is trained as a global summary and its norm tends to be larger
        than the feature tokens' — including it would let it dominate the
        reduction and make the "mean/max" ablations effectively equivalent
        to CLS pooling.
        """
        if self.pool == "cls":
            return hidden[:, 0, :]
        feature_hidden = hidden[:, 1:, :]
        if self.pool == "mean":
            return feature_hidden.mean(dim=1)
        return feature_hidden.max(dim=1).values

    def _apply_aux_force_mask(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Force PAY_0 into ``[MASK]`` on every row.

        The auxiliary PAY_0 head must predict a genuinely unseen label, or
        else it degenerates to an identity mapping on the input slot. By
        OR-ing a forced mask into any existing mask_positions we preserve
        MTLM-style random masking while guaranteeing PAY_0 is always
        hidden from the encoder.
        """
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device
        force = torch.zeros(B, 23, dtype=torch.bool, device=device)
        force[:, PAY_0_FEATURE_POSITION_23] = True
        existing = batch.get("mask_positions")
        if existing is not None:
            # Preserve any MTLM-style random masking the dataloader added.
            force = existing.to(device) | force
        return {**batch, "mask_positions": force}

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_attn: bool = False,
    ) -> dict[str, Any]:
        """Run embedding -> encoder -> pool -> head.

        Parameters
        ----------
        batch
            Dict with tokeniser tensors (``num_values``, ``pay_raw``,
            ``cat_indices``, optional ``mask_positions``) on the right
            device. Anything else is ignored.
        return_attn
            If ``True`` include ``attn_weights`` (list of per-layer
            tensors) in the output dict — only needed for Phase 10
            interpretability work; costs memory at large batch size.

        Returns
        -------
        dict
            Always contains ``logit: (B,)``. Contains
            ``aux_pay0_logits: (B, PAY_RAW_NUM_CLASSES)`` iff
            ``aux_pay0=True``. Contains ``attn_weights: list of
            (B, H, 24, 24)`` iff ``return_attn=True``.
        """
        # Force-mask PAY_0 first so the encoder genuinely does not see the
        # label the aux head is about to predict.
        if self.aux_pay0:
            batch = self._apply_aux_force_mask(batch)

        tokens = self.embedding(batch)
        hidden, attn_weights = self.encoder(tokens)
        pooled = self._pool(hidden)
        # squeeze(-1): head outputs (B, 1); downstream BCEWithLogitsLoss
        # expects (B,).
        logit = self.classifier(self.head_norm(pooled)).squeeze(-1)

        out: dict[str, Any] = {"logit": logit}

        if self.aux_pay0:
            assert self.aux_pay0_head is not None
            # Read the PAY_0 slot's final hidden state — the aux head
            # then predicts the original (now-masked) PAY_0 category.
            pay0_hidden = hidden[:, PAY_0_OUTPUT_POSITION_24, :]
            out["aux_pay0_logits"] = self.aux_pay0_head(pay0_hidden)

        if return_attn:
            out["attn_weights"] = attn_weights

        return out

    def count_parameters(self, trainable_only: bool = True) -> int:
        """Total parameter count (defaults to trainable-only)."""
        return sum(p.numel() for p in self.parameters() if (not trainable_only) or p.requires_grad)

    def parameter_count_by_module(
        self,
        trainable_only: bool = True,
    ) -> dict[str, int]:
        """Parameter count per top-level submodule.

        Stray leaf parameters (directly on the ``TabularTransformer``, not
        inside a submodule) are aggregated under the ``_leaf`` key. Used
        by ``summary()`` for the percentage breakdown.
        """
        counts: dict[str, int] = {}
        # Track param ids we've seen inside submodules so we can detect
        # leaf parameters (belonging to self, not to any child module).
        seen_param_ids: set[int] = set()
        for name, module in self.named_children():
            if module is None:
                continue
            total = 0
            for p in module.parameters():
                if (not trainable_only) or p.requires_grad:
                    total += p.numel()
                    seen_param_ids.add(id(p))
            if total:
                counts[name] = total
        leaf = 0
        for p in self.parameters(recurse=False):
            if id(p) in seen_param_ids:
                continue
            if (not trainable_only) or p.requires_grad:
                leaf += p.numel()
        if leaf:
            counts["_leaf"] = leaf
        return counts

    def get_head_params(self) -> list[torch.nn.Parameter]:
        """Parameters belonging to the classification + aux heads.

        ``train.build_optimizer`` puts these into the "head" parameter
        group so during fine-tuning they can stay at peak learning rate
        while the (pretrained) encoder uses a lower LR / warmup.
        """
        params: list[torch.nn.Parameter] = []
        for module in (self.head_norm, self.classifier, self.aux_pay0_head):
            if module is None:
                continue
            params.extend(module.parameters())
        return params

    def get_encoder_params(self) -> list[torch.nn.Parameter]:
        """Everything not returned by ``get_head_params()``.

        The two lists partition ``self.parameters()`` exactly, so the
        optimiser groups always add up to the full param set.
        """
        head_ids = {id(p) for p in self.get_head_params()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def summary(self) -> str:
        """Formatted multi-line architecture + param-count table.

        Notebook-friendly: the return value is a ready-to-print string
        with a header, parameter breakdown by submodule, total count, and
        a pass/fail vs the plan envelope (20K-40K).
        """
        n_total = self.count_parameters()
        breakdown = self.parameter_count_by_module()
        lines: list[str] = []
        lines.append("TabularTransformer - architecture summary")
        lines.append("-" * 56)
        lines.append(f"  d_model           : {self.d_model}")
        lines.append(f"  n_heads           : {self.n_heads}")
        lines.append(f"  n_layers          : {self.n_layers}")
        lines.append(f"  pool              : {self.pool}")
        td = self.encoder.temporal_decay
        fg = self.encoder.feature_group_bias
        lines.append(f"  temporal_decay    : {td.mode if td is not None else 'off'}")
        lines.append(f"  feature_group_bias: {fg.mode if fg is not None else 'off'}")
        lines.append(f"  use_temporal_pos  : {self.use_temporal_pos}")
        lines.append(f"  aux_pay0 head     : {self.aux_pay0}")
        lines.append("")
        lines.append("Parameter breakdown (trainable):")
        # Sorted descending so the heavyweight submodules (usually encoder)
        # are on top; pct column is relative to total.
        for name, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
            pct = 100.0 * count / max(1, n_total)
            lines.append(f"  {name:18s} {count:>8,d}  ({pct:5.1f}%)")
        lines.append("-" * 56)
        lines.append(f"  Total             : {n_total:,d} parameters")
        plan_lo, plan_hi = 20_000, 40_000
        in_plan = plan_lo <= n_total <= plan_hi
        status = "within plan envelope" if in_plan else "outside plan envelope"
        lines.append(f"  Plan budget       : ~28,000  ->  {status}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TabularTransformer(d_model={self.d_model}, n_heads={self.n_heads}, "
            f"n_layers={self.n_layers}, pool={self.pool!r}, "
            f"temporal_decay={self.encoder.temporal_decay.mode if self.encoder.temporal_decay else 'off'!r}, "
            f"feature_group_bias={self.encoder.feature_group_bias.mode if self.encoder.feature_group_bias else 'off'!r}, "
            f"aux_pay0={self.aux_pay0}, "
            f"params={self.count_parameters():,})"
        )

    @torch.no_grad()
    def predict_logits(
        self,
        loader,
        device: Optional[torch.device] = None,
        *,
        return_attn: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run inference over a dataloader and concat the outputs.

        Parameters
        ----------
        loader
            Any iterable of batches (DataLoader, list, etc.) whose batch
            dicts match ``forward()``'s contract.
        device
            Target device. ``None`` defaults to the module's current
            device (``next(self.parameters()).device``).
        return_attn
            Propagated to ``forward()``. If set, the returned dict also
            contains ``attn_weights`` — a list of per-layer concatenated
            tensors of shape ``(N, H, 24, 24)``. Memory-heavy at scale.

        Returns
        -------
        dict
            ``logit: (N,)``, optionally ``label: (N,)`` if the batches
            carry labels, optionally ``attn_weights`` as above. Everything
            is moved to CPU before returning — the caller owns the
            concatenated tensors.
        """
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        logits_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []
        # attn_chunks stays None until we see the first batch's return
        # structure — that's when we find out how many layers there are.
        attn_chunks: Optional[list[list[torch.Tensor]]] = None

        for batch in loader:
            # Move every tensor leaf to `device`. Dicts of tensors (e.g.
            # cat_indices) are recursed one level; deeper nesting is not
            # produced by our dataset collate_fn, so we don't handle it.
            moved: dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, dict):
                    moved[k] = {kk: vv.to(device, non_blocking=True) for kk, vv in v.items()}
                elif isinstance(v, torch.Tensor):
                    moved[k] = v.to(device, non_blocking=True)
                else:
                    moved[k] = v
            out = self(moved, return_attn=return_attn)
            logits_chunks.append(out["logit"].detach().cpu())
            if "label" in batch:
                label_chunks.append(batch["label"].detach().cpu())
            if return_attn:
                if attn_chunks is None:
                    attn_chunks = [[] for _ in range(len(out["attn_weights"]))]
                for i, w in enumerate(out["attn_weights"]):
                    attn_chunks[i].append(w.detach().cpu())

        result: dict[str, torch.Tensor] = {
            "logit": torch.cat(logits_chunks, dim=0),
        }
        if label_chunks:
            result["label"] = torch.cat(label_chunks, dim=0)
        if attn_chunks is not None:
            # Concatenate along the batch dim within each layer; the outer
            # list still has len == n_layers.
            result["attn_weights"] = [torch.cat(layer, dim=0) for layer in attn_chunks]
        return result

    @torch.no_grad()
    def predict_proba(
        self,
        loader,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """``sigmoid(predict_logits(...))`` returned as a ``(N,)`` CPU tensor."""
        logits = self.predict_logits(loader, device=device)["logit"]
        return torch.sigmoid(logits)

    @staticmethod
    def ensemble_probabilities(
        probabilities: Sequence[torch.Tensor],
        mode: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> torch.Tensor:
        """Combine per-model probability vectors into a single ensemble.

        Two aggregation modes:

        - ``"arithmetic"``: ``mean(p)``. Standard equal-weight average.
        - ``"geometric"``: ``sigmoid(mean(logit(p)))`` — mean in logit
          space, then sigmoid. More robust when one seed is heavily
          over-confident, because logit-space averaging dampens the
          pull of a near-0 or near-1 outlier (a probability of 0.999
          maps to logit ~6.9, which averages cleanly with logits in the
          normal range).

        Parameters
        ----------
        probabilities
            Non-empty sequence of ``(N,)`` tensors on any device/dtype.
        mode
            Aggregation strategy, see above.

        Returns
        -------
        torch.Tensor
            ``(N,)`` aggregated probabilities on CPU in float32.
        """
        if not probabilities:
            raise ValueError("probabilities cannot be empty")
        # Stack on CPU and float32 so the ensemble output is always in a
        # predictable dtype/device regardless of where the inputs came
        # from.
        stacked = torch.stack([p.detach().cpu().float() for p in probabilities], dim=0)
        if mode == "arithmetic":
            return stacked.mean(dim=0)
        if mode == "geometric":
            # Clamp away from the exact endpoints — logit(0)=-inf would
            # poison the mean.
            eps = 1e-7
            p = stacked.clamp(eps, 1.0 - eps)
            mean_logit = torch.log(p / (1.0 - p)).mean(dim=0)
            return torch.sigmoid(mean_logit)
        raise ValueError(f"mode must be 'arithmetic' or 'geometric', got {mode!r}")

    def load_pretrained_encoder(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        strict: bool = False,
        trust_source: bool = False,
        map_location: Optional[torch.device | str] = None,
    ) -> dict[str, Any]:
        """Load encoder weights from one of two on-disk shapes.

        1. Full bundle: the three-file layout written by
           ``utils.save_checkpoint`` (``path``, ``path.weights``,
           ``path.meta.json``). Routed through ``utils.load_checkpoint``
           which handles metadata and weights-only loading by default.
        2. Raw state-dict: a single ``torch.save``'d file containing only
           ``embedding.*`` / ``encoder.*`` keys — as written by the MTLM
           pretraining script as ``encoder_pretrained.pt``.

        Parameters
        ----------
        checkpoint_path
            Path to the checkpoint file. For bundles this is the
            metadata-holding file; the ``.weights`` sidecar is discovered
            automatically.
        strict
            Forwarded to ``load_state_dict``. Default ``False`` so the
            freshly-initialised classification head is left untouched
            when loading an MTLM checkpoint (which has no head).
        trust_source
            If ``True``, disables ``weights_only`` on the raw path. Only
            set this for files you produced yourself — untrusted
            checkpoints can execute arbitrary pickle payloads otherwise.
        map_location
            Device mapping argument forwarded to ``torch.load``.

        Returns
        -------
        dict
            Metadata describing the load (keys applied, missing, etc.).
            Exact shape differs between the bundle and raw paths.
        """
        from pathlib import Path as _Path

        from ..training.utils import load_checkpoint

        path = _Path(checkpoint_path)
        # The bundle layout uses a ".weights" sidecar for the tensors —
        # probing for that file is how we distinguish "bundle" from "raw
        # state-dict".
        sidecar = path.with_suffix(path.suffix + ".weights")

        if sidecar.is_file():
            checkpoint = load_checkpoint(
                path,
                self,
                strict=strict,
                trust_source=trust_source,
                map_location=map_location,
            )
            logger.info(
                "TabularTransformer: loaded pretrained encoder (bundle) from %s "
                "(classification head left at fresh init)",
                path,
            )
            return checkpoint

        # Raw state-dict branch — typically the encoder_pretrained.pt
        # artefact from the MTLM script.
        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint found at: {path}")
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=not trust_source,
        )
        # Guard against someone passing a full-bundle metadata file here
        # by accident — torch.load on a pickled dict would still succeed
        # shape-wise but load_state_dict would explode unhelpfully.
        if not isinstance(state, dict):
            raise TypeError(
                f"Expected a state-dict (mapping of tensor names to Tensors) at "
                f"{path}, got {type(state).__name__}. Pass a full checkpoint "
                "bundle if you meant load_checkpoint's optimiser-state path."
            )
        incompatible = self.load_state_dict(state, strict=strict)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        logger.info(
            "TabularTransformer: loaded raw encoder state from %s - "
            "%d tensors applied (missing=%d, unexpected=%d)",
            path,
            len(state),
            len(missing),
            len(unexpected),
        )
        if missing:
            # Expected when coming from MTLM: head_norm.*, classifier.*
            # will be missing because MTLM has no classification head.
            logger.debug("Missing keys (OK for MTLM->supervised): %s", missing[:6])
        if unexpected:
            logger.debug("Unexpected keys (dropped): %s", unexpected[:6])
        return {
            "metadata": {},
            "missing_keys": missing,
            "unexpected_keys": unexpected,
            "artefact_type": "raw_state_dict",
            "path": str(path),
        }


if __name__ == "__main__":
    # Smoke test — uses real preprocessing output if available, otherwise
    # skips (the CI path runs tests/test_model.py which fakes the data).
    # Sections:
    #   A: default-shape forward
    #   B: grad flow through every learnable tensor
    #   C: all three pool modes produce (B,) logits
    #   D: temporal_pos + N3 decay + N5 aux head together
    #   E: aux head gets finite gradients
    #   F: same seed -> identical outputs (torch determinism sanity)
    import json
    import sys

    for _stream in (sys.stdout, sys.stderr):
        if hasattr(_stream, "reconfigure"):
            try:
                _stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):
                pass

    import pandas as pd

    from ..tokenization.tokenizer import CreditDefaultDataset, build_categorical_vocab
    from ..training.dataset import make_loader

    root = Path(__file__).resolve().parent.parent.parent
    meta_path = root / "data/processed/feature_metadata.json"
    csv_path = root / "data/processed/splits/train_scaled.csv"
    if not (meta_path.is_file() and csv_path.is_file()):
        print(
            "[SKIP] need preprocessing output - "
            "run `poetry run python scripts/run_pipeline.py --preprocess-only` first."
        )
        sys.exit(0)

    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path).head(256)
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df, cat_vocab, verbose=False)
    loader = make_loader(ds, batch_size=32, mode="val")

    # A: defaults -------------------------------------------------------------
    torch.manual_seed(0)
    model = TabularTransformer()
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch, return_attn=True)
    n_params = model.count_parameters()
    print("-- A: defaults d=32 h=4 L=2 --")
    print(f"  logit:    {tuple(out['logit'].shape)}")
    print(f"  attn:     {len(out['attn_weights'])} layers, {tuple(out['attn_weights'][0].shape)}")
    print(f"  params:   {n_params:,} (~28K)")
    assert out["logit"].shape == (32,)
    assert len(out["attn_weights"]) == 2
    # Plan-envelope guard: if someone changes defaults and the param
    # count jumps outside [20K, 40K], this catches it.
    assert (
        20_000 <= n_params <= 40_000
    ), f"param count {n_params:,} outside the [20K, 40K] plan envelope"

    # B: grad flow ------------------------------------------------------------
    model.train()
    grad_out = model(batch)
    grad_out["logit"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"no grad on {no_grad}"
    print("\n-- B: grad flow --")
    print(f"  all {sum(1 for _ in model.parameters())} param groups got grads")

    # C: pool modes -----------------------------------------------------------
    print("\n-- C: pool modes --")
    for pool in ("cls", "mean", "max"):
        m = TabularTransformer(pool=pool)
        m.eval()
        with torch.no_grad():
            logit = m(batch)["logit"]
        assert logit.shape == (32,), pool
        print(f"  pool={pool}: {tuple(logit.shape)}")

    # D: temporal decay + aux_pay0 -------------------------------------------
    torch.manual_seed(0)
    m = TabularTransformer(
        d_model=32,
        n_heads=4,
        n_layers=2,
        use_temporal_pos=True,
        temporal_decay_mode="scalar",
        aux_pay0=True,
    )
    m.train()
    out = m(batch, return_attn=True)
    assert "aux_pay0_logits" in out
    assert out["aux_pay0_logits"].shape == (32, PAY_RAW_NUM_CLASSES)
    print("\n-- D: temporal_pos + decay + aux_pay0 --")
    print(f"  logit:       {tuple(out['logit'].shape)}")
    print(f"  aux logits:  {tuple(out['aux_pay0_logits'].shape)}")
    print(f"  alpha (decay):   {m.encoder.temporal_decay.alpha.item():.4f}")

    # E: aux grads ------------------------------------------------------------
    loss = out["aux_pay0_logits"].sum()
    loss.backward()
    assert m.aux_pay0_head is not None
    for name, p in m.aux_pay0_head.named_parameters():
        assert p.grad is not None, f"no grad on aux head {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on aux head {name}"
    print(f"  aux grads OK ({sum(1 for _ in m.aux_pay0_head.parameters())} groups)")

    # F: determinism ----------------------------------------------------------
    torch.manual_seed(1)
    m1 = TabularTransformer(d_model=32, n_heads=4, n_layers=2)
    torch.manual_seed(1)
    m2 = TabularTransformer(d_model=32, n_heads=4, n_layers=2)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        l1 = m1(batch)["logit"]
        l2 = m2(batch)["logit"]
    assert torch.allclose(l1, l2, atol=1e-6)
    print("\n-- F: same seed -> same output")

    print("\nall smoke checks passed.")
