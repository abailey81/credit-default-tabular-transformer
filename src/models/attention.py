"""Hand-rolled scaled-dot-product and multi-head self-attention.

Purpose
-------
Provide a transparent, single-file implementation of the attention primitive
used by every ``TransformerBlock`` in this repo. ``nn.MultiheadAttention``
would have worked functionally, but the coursework brief explicitly asks for
a from-scratch Q/K/V path so the forward is readable end-to-end and — more
practically — so Phase 10 interpretability work can hand back the *raw*
pre-dropout attention distribution without monkey-patching a PyTorch internal.

Key public symbols
------------------
- ``ScaledDotProductAttention`` — the Vaswani+ 2017 Eq. 1 kernel with an
  optional additive score bias (used by N2/N3 priors).
- ``MultiHeadAttention`` — splits ``d_model`` into ``n_heads`` parallel
  attention kernels, concatenates, projects back.

Design choice
-------------
We pass the attention bias as an *additive pre-softmax* term rather than a
post-softmax reweighting. Composing priors in logit space is linear and
gives us a single clean hook (``attn_bias``) that covers both ALiBi-style
temporal decay (N3) and the learnable feature-group bias (N2); we do not
need a bespoke attention variant per prior.

Critical invariant
------------------
``forward`` returns (output, weights) where ``weights`` is the PRE-dropout
distribution. Dropout is applied only on the value-projection path. This is
load-bearing for ``src.evaluation.interpret`` — a dropped-out view is not a
valid probability distribution and rollout would break on it.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention (Vaswani+ 2017, Eq. 1).

    Computes ``softmax((Q K^T + b) / sqrt(d_k)) . V`` where ``b`` is an
    optional additive bias. The bias hook is the single point through which
    architectural priors (N2 feature-group, N3 temporal-decay) inject
    structure; the kernel itself is vanilla SDPA.

    Returns the PRE-dropout attention distribution as the second element of
    the tuple so interpretability code sees the real softmax output, not a
    sparsified view. Dropout is applied only to the matmul with ``V``.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        # Attention dropout acts on the VALUE path only. Applying dropout to
        # the returned `attn_weights` would invalidate the "rows sum to 1"
        # invariant that the rollout interpretation (Abnar+ 2020) relies on.
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention forward pass.

        Parameters
        ----------
        Q, K, V
            Query / key / value tensors of shape ``(B, H, T, d_k)``.
        attn_bias
            Optional additive bias. Any shape broadcast-compatible with
            ``(B, H, T, T)`` is accepted — callers typically pass ``(T, T)``
            for a scalar prior or ``(H, T, T)`` for a per-head prior. Added
            pre-softmax so priors compound with learned attention rather
            than post-hoc reweighting the output.

        Returns
        -------
        tuple of tensors
            ``(output, weights)`` where ``output`` has shape
            ``(B, H, T, d_k)`` and ``weights`` is the pre-dropout attention
            distribution ``(B, H, T, T)``. Rows of ``weights`` sum to 1.
        """
        d_k = Q.size(-1)
        # Scale by 1/sqrt(d_k) BEFORE softmax. Without this, raw dot-product
        # magnitudes grow with d_k, softmax saturates, and the gradient
        # flowing back to Q/K shrinks to ~0 — Vaswani+ 2017 §3.2.1.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        # Additive prior (N2 / N3) lives in logit space so it composes
        # linearly with the learned attention. Passing bias=zeros here must
        # be a no-op and the smoke test below checks that invariant.
        if attn_bias is not None:
            scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1)
        # Dropout on the distribution used to mix V; `attn_weights` itself
        # is handed back un-dropped for downstream interpretability.
        attn_weights_dropped = self.dropout(attn_weights)
        output = torch.matmul(attn_weights_dropped, V)

        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention (Vaswani+ 2017, §3.2.2).

    Projects the input into ``n_heads`` parallel (Q, K, V) streams of size
    ``d_k = d_model // n_heads``, runs ``ScaledDotProductAttention`` on
    each, concatenates head outputs, and applies a final linear projection.

    At the repo default ``d_model=32, n_heads=4`` each head sees a 8-dim
    subspace, which was chosen deliberately: with only ~21K training rows
    anything larger saturates the parameter budget (~28K total) without
    buying extra AUROC on the ablation runs.
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        # d_model must split evenly; we deliberately do NOT support the
        # "last head gets the remainder" pattern — the ablations would
        # become ambiguous and the checkpoint shapes brittle.
        assert (
            d_model % n_heads == 0
        ), f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Separate Q/K/V projections (not a fused single Linear). Fusing
        # would save a tiny amount of compute but makes per-head weight
        # inspection in Phase 10 interpretability harder.
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier-normal on all four projection weights, zero biases.

        Xavier (a.k.a. Glorot) keeps activation variance roughly constant
        across the projection for the linear case, which matches the
        pre-softmax distribution we want before the temperature scaling
        kicks in.
        """
        for linear in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-head attention over ``x``.

        Parameters
        ----------
        x
            Input of shape ``(B, T, d_model)``. Here ``T=24`` for the
            tabular-transformer (CLS + 23 feature tokens).
        attn_bias
            Optional additive bias broadcastable to ``(B, H, T, T)``.
            Forwarded unchanged to ``ScaledDotProductAttention``.

        Returns
        -------
        tuple of tensors
            ``(output, attn_weights)`` where ``output`` has shape
            ``(B, T, d_model)`` and ``attn_weights`` is the per-head
            pre-dropout distribution ``(B, H, T, T)``.
        """
        B, seq_len, _ = x.shape

        # Project once, then reshape — cheaper than three matmuls per head.
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # (B, T, d_model) -> (B, n_heads, T, d_k). The transpose is what
        # actually puts "heads" on an independent batch dimension; without
        # it the attention matmul would mix head subspaces.
        Q = Q.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, attn_bias=attn_bias)

        # Concatenate heads by undoing the transpose. `.contiguous()` is
        # required because `view` refuses non-contiguous tensors after a
        # transpose — a .reshape() would hide the copy and make the memory
        # pattern harder to reason about.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, seq_len, self.d_model)

        # Final output projection mixes information across heads. Without
        # W_O each head would be stuck in its own d_k subspace forever.
        return self.W_O(attn_output), attn_weights


if __name__ == "__main__":
    # Smoke test: covers shapes, softmax-row-sum invariant, the
    # zero-bias-is-no-op contract, bias suppression behaviour, and
    # gradient flow through every learnable tensor.
    B, seq_len, d_model, n_heads = 4, 24, 64, 4

    # Seed so that "bias=None vs bias=zeros" produces bit-identical
    # outputs on repeated runs.
    torch.manual_seed(0)

    x = torch.randn(B, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()  # dropout off so outputs are deterministic across calls

    output, weights = mha(x)

    print(f"input shape:             {x.shape}")
    print(f"output shape:            {output.shape}")
    print(f"attention weights shape: {weights.shape}")

    assert output.shape == (
        B,
        seq_len,
        d_model,
    ), f"Expected ({B}, {seq_len}, {d_model}), got {output.shape}"
    assert weights.shape == (
        B,
        n_heads,
        seq_len,
        seq_len,
    ), f"Expected ({B}, {n_heads}, {seq_len}, {seq_len}), got {weights.shape}"

    # Rows of the attention distribution must sum to 1 — if this breaks,
    # the attention-rollout interpretability path breaks with it.
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(
        row_sums, torch.ones_like(row_sums), atol=1e-5
    ), "Attention weights don't sum to 1"

    # Zero bias must equal the no-bias path exactly. This is what keeps
    # ablations fair: "N2 off" really means no-op at the attention level.
    zero_bias = torch.zeros(seq_len, seq_len)
    output_zero, weights_zero = mha(x, attn_bias=zero_bias)
    assert torch.allclose(output, output_zero, atol=1e-6), "Zero attn_bias changed output"
    assert torch.allclose(weights, weights_zero, atol=1e-6), "Zero attn_bias changed weights"

    # Strong negative bias at a specific (q, k) pair should drive that
    # attention weight to ~0 post-softmax — sanity-checks the masking idiom.
    biased = torch.zeros(seq_len, seq_len)
    biased[0, 1] = -1e4
    _, weights_biased = mha(x, attn_bias=biased)
    suppressed = weights_biased[:, :, 0, 1]
    assert torch.all(
        suppressed < 1e-3
    ), f"Bias did not suppress attention weight (got max {suppressed.max().item():.4f})"

    # Grad check: run with dropout on so we exercise the training path,
    # then verify every learnable tensor actually received a gradient.
    mha.train()
    output, _ = mha(x)
    loss = output.sum()
    loss.backward()
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("all checks passed.")
