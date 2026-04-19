"""Scaled-dot-product + multi-head self-attention, rolled by hand instead of
nn.MultiheadAttention so the forward is fully visible and we can hand the raw
attention weights back for the phase-10 interpretability work."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """softmax(QK^T / √d_k) · V with optional additive bias on the scores.
    Returns (output, pre-dropout weights) so interpretability code sees the
    real distribution."""

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if attn_bias is not None:
            scores = scores + attn_bias

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)
        output = torch.matmul(attn_weights_dropped, V)

        # hand back the un-dropped weights, not the sparsified view
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """Split d_model into n_heads of size d_k = d_model // n_heads, run SDPA on
    each, concat, project back. d_model=32 hits the ~28K-param encoder target
    for our 21K-row training set."""

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """xavier-normal W, zero b."""
        for linear in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, seq_len, _ = x.shape

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # (B, T, d) -> (B, h, T, d_k)
        Q = Q.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        attn_output, attn_weights = self.attention(Q, K, V, attn_bias=attn_bias)

        # back to (B, T, d)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, seq_len, self.d_model)
        )

        return self.W_O(attn_output), attn_weights


if __name__ == "__main__":
    B, seq_len, d_model, n_heads = 4, 24, 64, 4

    # seeded so the bias=None vs bias=zeros check is reproducible
    torch.manual_seed(0)

    x = torch.randn(B, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.0)
    mha.eval()  # dropout off so outputs match across calls

    output, weights = mha(x)

    print(f"input shape:             {x.shape}")
    print(f"output shape:            {output.shape}")
    print(f"attention weights shape: {weights.shape}")

    assert output.shape == (B, seq_len, d_model), f"Expected ({B}, {seq_len}, {d_model}), got {output.shape}"
    assert weights.shape == (B, n_heads, seq_len, seq_len), f"Expected ({B}, {n_heads}, {seq_len}, {seq_len}), got {weights.shape}"

    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention weights don't sum to 1"

    # zero bias must equal the no-bias path exactly
    zero_bias = torch.zeros(seq_len, seq_len)
    output_zero, weights_zero = mha(x, attn_bias=zero_bias)
    assert torch.allclose(output, output_zero, atol=1e-6), "Zero attn_bias changed output"
    assert torch.allclose(weights, weights_zero, atol=1e-6), "Zero attn_bias changed weights"

    # big -ve bias at (0, 1) should kill that attention weight
    biased = torch.zeros(seq_len, seq_len)
    biased[0, 1] = -1e4
    _, weights_biased = mha(x, attn_bias=biased)
    suppressed = weights_biased[:, :, 0, 1]
    assert torch.all(suppressed < 1e-3), f"Bias did not suppress attention weight (got max {suppressed.max().item():.4f})"

    # grad check with dropout on so we exercise the training path
    mha.train()
    output, _ = mha(x)
    loss = output.sum()
    loss.backward()
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("all checks passed.")
