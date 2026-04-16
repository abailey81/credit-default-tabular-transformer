"""
attention.py — Scaled dot-product and multi-head self-attention from scratch.

This is the core of the transformer, built without nn.MultiheadAttention.
Two classes:
    1. ScaledDotProductAttention — the fundamental attention operation.
    2. MultiHeadAttention — runs multiple attention heads in parallel.

Both return attention weights alongside outputs for interpretability
(Phase 10: attention visualisation and rollout analysis).

Reference: Vaswani et al. (2017), "Attention Is All You Need", NeurIPS.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    """
    Compute scaled dot-product attention.

        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V

    Args:
        dropout: Dropout rate applied to attention weights (default 0.1).

    Input shapes:
        Q: (B, n_heads, seq_len, d_k)
        K: (B, n_heads, seq_len, d_k)
        V: (B, n_heads, seq_len, d_k)

    Returns:
        output:       (B, n_heads, seq_len, d_k) — weighted sum of values.
        attn_weights: (B, n_heads, seq_len, seq_len) — attention probabilities.
    """

    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = Q.size(-1)

        # 1. Dot product between queries and keys → (B, n_heads, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # 2. Scale by sqrt(d_k) to prevent softmax saturation
        scores = scores / math.sqrt(d_k)

        # 3. Softmax → attention weights (each row sums to 1)
        attn_weights = F.softmax(scores, dim=-1)

        # 4. Dropout on attention weights (regularisation)
        attn_weights_dropped = self.dropout(attn_weights)

        # 5. Weighted sum of values
        output = torch.matmul(attn_weights_dropped, V)

        # Return un-dropped weights for interpretability
        return output, attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention.

    Splits d_model into n_heads parallel attention heads, each operating
    on d_k = d_model // n_heads dimensions. Concatenates and projects back.

        MultiHead(X) = Concat(head_1, ..., head_h) · W_O
        where head_i = Attention(X·W_Qi, X·W_Ki, X·W_Vi)

    Args:
        d_model: Model dimension (default 64).
        n_heads: Number of attention heads (default 4).
        dropout: Dropout rate for attention weights (default 0.1).

    Input:
        x: (B, seq_len, d_model) — token embeddings.

    Returns:
        output:       (B, seq_len, d_model) — updated token representations.
        attn_weights: (B, n_heads, seq_len, seq_len) — per-head attention maps.
    """

    def __init__(
        self,
        d_model: int = 64,
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

        # Projection matrices — one nn.Linear each, split into heads via reshape
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout=dropout)

        self._init_weights()

    def _init_weights(self):
        """Xavier normal for projections, zeros for biases."""
        for linear in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, seq_len, _ = x.shape

        # 1. Project to Q, K, V → (B, seq_len, d_model)
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # 2. Reshape into heads: (B, seq_len, d_model) → (B, n_heads, seq_len, d_k)
        Q = Q.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(B, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 3. Scaled dot-product attention per head
        attn_output, attn_weights = self.attention(Q, K, V)

        # 4. Concatenate heads: (B, n_heads, seq_len, d_k) → (B, seq_len, d_model)
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(B, seq_len, self.d_model)
        )

        # 5. Output projection
        output = self.W_O(attn_output)

        return output, attn_weights


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, seq_len, d_model, n_heads = 4, 24, 64, 4

    x = torch.randn(B, seq_len, d_model)
    mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, dropout=0.1)
    output, weights = mha(x)

    print(f"Input shape:            {x.shape}")
    print(f"Output shape:           {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Shape checks
    assert output.shape == (B, seq_len, d_model), f"Expected ({B}, {seq_len}, {d_model}), got {output.shape}"
    assert weights.shape == (B, n_heads, seq_len, seq_len), f"Expected ({B}, {n_heads}, {seq_len}, {seq_len}), got {weights.shape}"

    # Attention weights should sum to 1 along last dim
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), "Attention weights don't sum to 1"

    # Gradient check
    loss = output.sum()
    loss.backward()
    for name, param in mha.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"

    print("All checks passed.")
