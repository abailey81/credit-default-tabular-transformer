"""TabularTransformer: embedding → encoder → pool → MLP head → logit.
Optional aux PAY_0 head (N5), MTLM [MASK] path, temporal-decay (N3),
feature-group attention bias (N2)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
import torch.nn as nn

from ..tokenization.embedding import (
    FeatureEmbedding,
    N_FEATURE_GROUPS,
    build_group_assignment,
    build_temporal_layout,
)
from ..tokenization.tokenizer import PAY_RAW_NUM_CLASSES
from .transformer import FeatureGroupBias, TemporalDecayBias, TransformerEncoder

logger = logging.getLogger(__name__)

# pinned to TOKEN_ORDER in embedding.py — keep aux head drift-safe
PAY_0_FEATURE_POSITION_23 = 3                               # 23-slot, pre-CLS
PAY_0_OUTPUT_POSITION_24 = PAY_0_FEATURE_POSITION_23 + 1    # 24-slot, CLS at 0
FULL_SEQ_LEN = 24                                           # CLS + 23 features


class TabularTransformer(nn.Module):
    """embedding → encoder → pool → head. Defaults land at ~28K params.

    mean/max pool drop CLS so it can't dominate. use_mask_token is forced on
    when aux_pay0=True (aux head force-masks PAY_0). forward returns
    {logit: (B,), [aux_pay0_logits: (B, 11)], [attn_weights: list of
    (B, H, 24, 24)]}.
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
        cat_vocab_sizes: Optional[Dict[str, int]] = None,
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

        # aux_pay0 force-masks PAY_0 → [MASK] must exist
        self.embedding = FeatureEmbedding(
            d_model=d_model,
            dropout=dropout,
            cat_vocab_sizes=cat_vocab_sizes,
            use_temporal_pos=use_temporal_pos,
            use_mask_token=use_mask_token or aux_pay0,
        )

        if temporal_decay_mode != "off":
            temporal_decay: Optional[TemporalDecayBias] = TemporalDecayBias(
                temporal_layout=build_temporal_layout(cls_offset=1),
                seq_len=FULL_SEQ_LEN,
                mode=temporal_decay_mode,
                n_heads=n_heads,
            )
        else:
            temporal_decay = None

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

        # LN → 2-layer MLP → logit
        self.head_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(classification_dropout),
            nn.Linear(d_model, 1),
        )

        # aux PAY_0 head: same shape, own LN (decoupled from main)
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
        """Kaiming on GELU-feeding linears, Xavier on each head's last linear."""
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
        """(B, 24, d) → (B, d) by CLS / mean / max."""
        if self.pool == "cls":
            return hidden[:, 0, :]
        # drop CLS so a drifted CLS norm can't swamp the mean/max
        feature_hidden = hidden[:, 1:, :]
        if self.pool == "mean":
            return feature_hidden.mean(dim=1)
        return feature_hidden.max(dim=1).values

    def _apply_aux_force_mask(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """force-mask PAY_0 (OR with any existing mask)."""
        B = batch["num_values"].shape[0]
        device = batch["num_values"].device
        force = torch.zeros(B, 23, dtype=torch.bool, device=device)
        force[:, PAY_0_FEATURE_POSITION_23] = True
        existing = batch.get("mask_positions")
        if existing is not None:
            force = existing.to(device) | force
        return {**batch, "mask_positions": force}

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        return_attn: bool = False,
    ) -> Dict[str, Any]:
        """batch → {logit, [aux_pay0_logits], [attn_weights]}."""
        if self.aux_pay0:
            batch = self._apply_aux_force_mask(batch)

        tokens = self.embedding(batch)
        hidden, attn_weights = self.encoder(tokens)
        pooled = self._pool(hidden)
        logit = self.classifier(self.head_norm(pooled)).squeeze(-1)

        out: Dict[str, Any] = {"logit": logit}

        if self.aux_pay0:
            assert self.aux_pay0_head is not None
            pay0_hidden = hidden[:, PAY_0_OUTPUT_POSITION_24, :]
            out["aux_pay0_logits"] = self.aux_pay0_head(pay0_hidden)

        if return_attn:
            out["attn_weights"] = attn_weights

        return out

    def count_parameters(self, trainable_only: bool = True) -> int:
        """total (trainable) param count."""
        return sum(
            p.numel()
            for p in self.parameters()
            if (not trainable_only) or p.requires_grad
        )

    def parameter_count_by_module(
        self, trainable_only: bool = True,
    ) -> Dict[str, int]:
        """param count per top-level submodule; stray leaf params go under _leaf."""
        counts: Dict[str, int] = {}
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

    def get_head_params(self) -> List[torch.nn.Parameter]:
        """head (+ aux head) params. train.build_optimizer uses these
        for the head group at peak lr during fine-tuning."""
        params: List[torch.nn.Parameter] = []
        for module in (self.head_norm, self.classifier, self.aux_pay0_head):
            if module is None:
                continue
            params.extend(module.parameters())
        return params

    def get_encoder_params(self) -> List[torch.nn.Parameter]:
        """everything not in get_head_params()."""
        head_ids = {id(p) for p in self.get_head_params()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def summary(self) -> str:
        """multi-line arch + param breakdown (for notebooks)."""
        n_total = self.count_parameters()
        breakdown = self.parameter_count_by_module()
        lines: List[str] = []
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
    ) -> Dict[str, torch.Tensor]:
        """loader → {logit: (N,), [label], [attn_weights]}. All on CPU."""
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        logits_chunks: List[torch.Tensor] = []
        label_chunks: List[torch.Tensor] = []
        attn_chunks: Optional[List[List[torch.Tensor]]] = None

        for batch in loader:
            moved: Dict[str, Any] = {}
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

        result: Dict[str, torch.Tensor] = {
            "logit": torch.cat(logits_chunks, dim=0),
        }
        if label_chunks:
            result["label"] = torch.cat(label_chunks, dim=0)
        if attn_chunks is not None:
            result["attn_weights"] = [
                torch.cat(layer, dim=0) for layer in attn_chunks
            ]
        return result

    @torch.no_grad()
    def predict_proba(
        self,
        loader,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """σ(predict_logits(...)) on CPU."""
        logits = self.predict_logits(loader, device=device)["logit"]
        return torch.sigmoid(logits)

    @staticmethod
    def ensemble_probabilities(
        probabilities: Sequence[torch.Tensor],
        mode: Literal["arithmetic", "geometric"] = "arithmetic",
    ) -> torch.Tensor:
        """arithmetic = mean(p); geometric = σ(mean(logit(p))) — the latter
        shrugs off one over-confident seed."""
        if not probabilities:
            raise ValueError("probabilities cannot be empty")
        stacked = torch.stack([p.detach().cpu().float() for p in probabilities], dim=0)
        if mode == "arithmetic":
            return stacked.mean(dim=0)
        if mode == "geometric":
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
    ) -> Dict[str, Any]:
        """Load weights from one of two on-disk shapes:

        1. full bundle — path + path.weights + path.meta.json from
           utils.save_checkpoint; routed through utils.load_checkpoint
           (weights-only by default).
        2. raw state dict — a single torch.save'd file with only
           embedding.*/encoder.* keys, as written by train_mtlm as
           encoder_pretrained.pt.

        strict=False leaves the fresh classification head alone (MTLM has no
        head). trust_source=True disables weights_only — only use on files
        you produced yourself.
        """
        from pathlib import Path as _Path
        from ..training.utils import load_checkpoint

        path = _Path(checkpoint_path)
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

        if not path.is_file():
            raise FileNotFoundError(f"No checkpoint found at: {path}")
        state = torch.load(
            path,
            map_location=map_location,
            weights_only=not trust_source,
        )
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
            path, len(state), len(missing), len(unexpected),
        )
        if missing:
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
    csv_path = root / "data/processed/train_scaled.csv"
    if not (meta_path.is_file() and csv_path.is_file()):
        print(
            "[SKIP] need preprocessing output — "
            "run `poetry run python scripts/run_pipeline.py --preprocess-only` first."
        )
        sys.exit(0)

    meta = json.loads(meta_path.read_text())
    df = pd.read_csv(csv_path).head(256)
    cat_vocab = build_categorical_vocab(meta)
    ds = CreditDefaultDataset(df, cat_vocab, verbose=False)
    loader = make_loader(ds, batch_size=32, mode="val")

    # A: defaults
    torch.manual_seed(0)
    model = TabularTransformer()
    model.eval()
    batch = next(iter(loader))
    with torch.no_grad():
        out = model(batch, return_attn=True)
    n_params = model.count_parameters()
    print(f"-- A: defaults d=32 h=4 L=2 --")
    print(f"  logit:    {tuple(out['logit'].shape)}")
    print(f"  attn:     {len(out['attn_weights'])} layers, {tuple(out['attn_weights'][0].shape)}")
    print(f"  params:   {n_params:,} (~28K)")
    assert out["logit"].shape == (32,)
    assert len(out["attn_weights"]) == 2
    assert 20_000 <= n_params <= 40_000, (
        f"param count {n_params:,} outside the [20K, 40K] plan envelope"
    )

    # B: grad flow
    model.train()
    grad_out = model(batch)
    grad_out["logit"].sum().backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    assert not no_grad, f"no grad on {no_grad}"
    print("\n-- B: grad flow --")
    print(f"  all {sum(1 for _ in model.parameters())} param groups got grads")

    # C: pool modes
    print("\n-- C: pool modes --")
    for pool in ("cls", "mean", "max"):
        m = TabularTransformer(pool=pool)
        m.eval()
        with torch.no_grad():
            logit = m(batch)["logit"]
        assert logit.shape == (32,), pool
        print(f"  pool={pool}: {tuple(logit.shape)}")

    # D: temporal decay + aux_pay0
    torch.manual_seed(0)
    m = TabularTransformer(
        d_model=32, n_heads=4, n_layers=2,
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
    print(f"  α (decay):   {m.encoder.temporal_decay.alpha.item():.4f}")

    # E: aux grads
    loss = out["aux_pay0_logits"].sum()
    loss.backward()
    assert m.aux_pay0_head is not None
    for name, p in m.aux_pay0_head.named_parameters():
        assert p.grad is not None, f"no grad on aux head {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on aux head {name}"
    print(f"  aux grads OK ({sum(1 for _ in m.aux_pay0_head.parameters())} groups)")

    # F: same seed → same output
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
    print("\n-- F: same seed → same output")

    print("\nall smoke checks passed.")
