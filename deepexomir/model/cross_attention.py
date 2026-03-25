"""Bidirectional cross-attention Transformer for miRNA-target interaction.

Each ``CrossAttentionBlock`` performs **bidirectional** information exchange:

    1. miRNA self-attention  -> cross-attention (target attends to miRNA)
    2. Target self-attention -> cross-attention (miRNA attends to target)

Both branches use **pre-norm** (LayerNorm before attention) and are followed
by a position-wise feed-forward network (FFN) with residual connections.

``CrossAttentionEncoder`` stacks *N* such blocks with optional DropPath
(stochastic depth) regularisation and SwiGLU feed-forward networks.

Model ⑥ enhancements (v6):
    - DropPath (stochastic depth) with linearly increasing drop rates
    - SwiGLU gated feed-forward networks
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DropPath (Stochastic Depth)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    During training, randomly drops the entire residual branch with
    probability ``drop_prob``, forcing the network to rely on skip
    connections.  Linearly increasing drop rates across layers
    regularise deeper layers more aggressively.

    Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask: shape (B, 1, 1, ...) broadcasts over other dims
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor_(random_tensor + keep_prob)
        return x / keep_prob * random_tensor

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


# ---------------------------------------------------------------------------
# Multi-Head Attention (shared primitive)
# ---------------------------------------------------------------------------

class _MultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention.

    This is a minimal, self-contained implementation so that the cross-
    attention module does not depend on ``torch.nn.MultiheadAttention``
    batch-first conventions that vary across PyTorch versions.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute multi-head attention.

        Parameters
        ----------
        query : Tensor [B, Lq, D]
        key   : Tensor [B, Lk, D]
        value : Tensor [B, Lk, D]
        key_padding_mask : Tensor [B, Lk], optional
            ``True`` at positions to **mask** (pad tokens).

        Returns
        -------
        Tensor [B, Lq, D]
        """
        B, Lq, _ = query.shape
        Lk = key.shape[1]

        # Project & reshape -> [B, n_heads, L, d_k]
        Q = self.w_q(query).view(B, Lq, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, Lk, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if key_padding_mask is not None:
            # key_padding_mask: [B, Lk] -> [B, 1, 1, Lk]
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum -> [B, n_heads, Lq, d_k]
        context = torch.matmul(attn_weights, V)

        # Concat heads -> [B, Lq, D]
        context = context.transpose(1, 2).contiguous().view(B, Lq, self.d_model)
        return self.w_o(context)


# ---------------------------------------------------------------------------
# Feed-Forward Networks
# ---------------------------------------------------------------------------

class _FeedForward(nn.Module):
    """Position-wise feed-forward network: Linear -> GELU -> Dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class _SwiGLUFeedForward(nn.Module):
    """SwiGLU gated feed-forward network.

    Uses a gated linear unit with SiLU (Swish) activation::

        SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    The intermediate dimension is scaled to ``2/3 * d_ff`` to keep the
    total parameter count comparable to a standard 2-layer FFN.

    Reference: Shazeer, "GLU Variants Improve Transformer", 2020.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Keep param count ~same as standard FFN:
        # Standard: 2 * d_model * d_ff
        # SwiGLU:   3 * d_model * d_ff_reduced → d_ff_reduced ≈ 2/3 * d_ff
        d_ff_reduced = int(2 * d_ff / 3)
        # Round to nearest multiple of 8 for GPU efficiency
        d_ff_reduced = ((d_ff_reduced + 7) // 8) * 8

        self.w_gate = nn.Linear(d_model, d_ff_reduced)
        self.w_up = nn.Linear(d_model, d_ff_reduced)
        self.w_down = nn.Linear(d_ff_reduced, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(self.dropout(gate * up))


# ---------------------------------------------------------------------------
# Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """Single bidirectional cross-attention block with optional DropPath.

    Processing order for **each** of the two branches (miRNA / target):

        1. Pre-norm self-attention  + residual  (+ DropPath)
        2. Pre-norm cross-attention + residual  (+ DropPath)
        3. Pre-norm FFN             + residual  (+ DropPath)

    Parameters
    ----------
    d_model : int
        Hidden dimensionality (default: 256).
    n_heads : int
        Number of attention heads (default: 8).
    d_ff : int
        Inner dimensionality of the FFN (default: 1024).
    dropout : float
        Dropout rate applied in attention and FFN (default: 0.1).
    drop_path_rate : float
        DropPath (stochastic depth) rate for this block (default: 0.0).
    use_swiglu : bool
        If True, use SwiGLU FFN instead of standard GELU FFN (default: False).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()

        # Select FFN type
        FFN = _SwiGLUFeedForward if use_swiglu else _FeedForward

        # ---- miRNA branch ----
        self.mirna_self_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.mirna_cross_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.mirna_ffn = FFN(d_model, d_ff, dropout)

        self.mirna_norm_sa = nn.LayerNorm(d_model)
        self.mirna_norm_ca = nn.LayerNorm(d_model)
        self.mirna_norm_ff = nn.LayerNorm(d_model)

        # DropPath on residual connections (replaces standard Dropout)
        self.mirna_drop_sa = DropPath(drop_path_rate)
        self.mirna_drop_ca = DropPath(drop_path_rate)
        self.mirna_drop_ff = DropPath(drop_path_rate)

        # ---- target branch ----
        self.target_self_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.target_cross_attn = _MultiHeadAttention(d_model, n_heads, dropout)
        self.target_ffn = FFN(d_model, d_ff, dropout)

        self.target_norm_sa = nn.LayerNorm(d_model)
        self.target_norm_ca = nn.LayerNorm(d_model)
        self.target_norm_ff = nn.LayerNorm(d_model)

        self.target_drop_sa = DropPath(drop_path_rate)
        self.target_drop_ca = DropPath(drop_path_rate)
        self.target_drop_ff = DropPath(drop_path_rate)

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional cross-attention forward pass.

        Parameters
        ----------
        mirna_emb : Tensor [B, Lm, D]
            miRNA embeddings.
        target_emb : Tensor [B, Lt, D]
            Target-site embeddings.
        mirna_mask : Tensor [B, Lm], optional
            ``True`` for **padding** positions in miRNA.
        target_mask : Tensor [B, Lt], optional
            ``True`` for **padding** positions in target.

        Returns
        -------
        (mirna_emb, target_emb) : tuple[Tensor, Tensor]
            Updated embeddings, both ``[B, L, D]``.
        """
        # ---- miRNA branch: self-attention ----
        m_normed = self.mirna_norm_sa(mirna_emb)
        mirna_emb = mirna_emb + self.mirna_drop_sa(
            self.mirna_self_attn(m_normed, m_normed, m_normed, mirna_mask)
        )

        # ---- target branch: self-attention ----
        t_normed = self.target_norm_sa(target_emb)
        target_emb = target_emb + self.target_drop_sa(
            self.target_self_attn(t_normed, t_normed, t_normed, target_mask)
        )

        # ---- cross-attention (truly bidirectional) ----
        # Compute both cross-attention outputs from the SAME pre-update state
        # so neither branch sees the other's update within this block.
        t_normed = self.target_norm_ca(target_emb)
        m_normed = self.mirna_norm_ca(mirna_emb)

        # target attends to miRNA
        target_ca_out = self.target_drop_ca(
            self.target_cross_attn(t_normed, m_normed, m_normed, mirna_mask)
        )
        # miRNA attends to target (from same pre-update state)
        mirna_ca_out = self.mirna_drop_ca(
            self.mirna_cross_attn(m_normed, t_normed, t_normed, target_mask)
        )

        # Apply residuals simultaneously
        target_emb = target_emb + target_ca_out
        mirna_emb = mirna_emb + mirna_ca_out

        # ---- FFN ----
        mirna_emb = mirna_emb + self.mirna_drop_ff(
            self.mirna_ffn(self.mirna_norm_ff(mirna_emb))
        )
        target_emb = target_emb + self.target_drop_ff(
            self.target_ffn(self.target_norm_ff(target_emb))
        )

        return mirna_emb, target_emb


# ---------------------------------------------------------------------------
# Cross-Attention Encoder (stack of N blocks)
# ---------------------------------------------------------------------------

class CrossAttentionEncoder(nn.Module):
    """Stack of ``N`` :class:`CrossAttentionBlock` layers with stochastic depth.

    Parameters
    ----------
    n_layers : int
        Number of cross-attention blocks (default: 4).
    d_model : int
        Hidden dimensionality (default: 256).
    n_heads : int
        Number of attention heads (default: 8).
    d_ff : int
        FFN inner dimensionality (default: 1024).
    dropout : float
        Dropout probability (default: 0.1).
    drop_path_rate : float
        Maximum DropPath rate (applied linearly from 0 to this value
        across layers).  Default: 0.0 (no stochastic depth).
    use_swiglu : bool
        If True, use SwiGLU FFN instead of GELU FFN (default: False).
    """

    def __init__(
        self,
        n_layers: int = 4,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()

        # Linearly increasing DropPath rates from 0 to drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        self.layers = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model, n_heads, d_ff, dropout,
                    drop_path_rate=dpr[i],
                    use_swiglu=use_swiglu,
                )
                for i in range(n_layers)
            ]
        )
        # Final layer-norm on each branch
        self.mirna_final_norm = nn.LayerNorm(d_model)
        self.target_final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all cross-attention blocks.

        Parameters
        ----------
        mirna_emb : Tensor [B, Lm, D]
        target_emb : Tensor [B, Lt, D]
        mirna_mask : Tensor [B, Lm], optional
            ``True`` at padding positions.
        target_mask : Tensor [B, Lt], optional
            ``True`` at padding positions.

        Returns
        -------
        (mirna_emb, target_emb) : tuple[Tensor, Tensor]
        """
        for layer in self.layers:
            mirna_emb, target_emb = layer(
                mirna_emb, target_emb, mirna_mask, target_mask
            )

        mirna_emb = self.mirna_final_norm(mirna_emb)
        target_emb = self.target_final_norm(target_emb)

        return mirna_emb, target_emb
