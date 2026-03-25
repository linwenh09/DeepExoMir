"""Interaction-aware attention pooling for miRNA-target prediction.

Replaces simple mean pooling with a cross-sequence attention mechanism
that produces interaction-aware sequence summaries:

1. Self-attended miRNA summary (what the miRNA "is")
2. Cross-attended miRNA summary (how the miRNA "sees" the target)
3. Self-attended target summary (what the target "is")
4. Cross-attended target summary (how the target "sees" the miRNA)

Output: [B, 4 * d_model] instead of [B, 2 * d_model]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InteractionPooling(nn.Module):
    """Cross-sequence attention pooling.

    Uses learnable query tokens to attend to each sequence, producing
    both self-summary and cross-summary representations.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    n_heads : int
        Number of attention heads (default: 4).
    dropout : float
        Attention dropout (default: 0.1).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        # Learnable query tokens for pooling
        self.mirna_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.target_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Self-attention pooling: query attends to same sequence
        self.self_pool = _PoolingAttention(d_model, n_heads, dropout)

        # Cross-attention pooling: query attends to OTHER sequence
        self.cross_pool = _PoolingAttention(d_model, n_heads, dropout)

        # Layer norms
        self.mirna_ln = nn.LayerNorm(d_model)
        self.target_ln = nn.LayerNorm(d_model)

        # Output projection to fuse self + cross
        self.out_proj = nn.Sequential(
            nn.Linear(d_model * 4, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        mirna_emb : [B, Lm, D]
        target_emb : [B, Lt, D]
        mirna_mask : [B, Lm] or None, True = padding
        target_mask : [B, Lt] or None, True = padding

        Returns
        -------
        pooled : [B, 2*D]
            Interaction-aware pooled representation.
        """
        B = mirna_emb.shape[0]

        # Normalize
        m = self.mirna_ln(mirna_emb)
        t = self.target_ln(target_emb)

        # Expand queries: [1, 1, D] -> [B, 1, D]
        m_q = self.mirna_query.expand(B, -1, -1)
        t_q = self.target_query.expand(B, -1, -1)

        # Self-attention pooling: each query attends to its own sequence
        m_self = self.self_pool(m_q, m, m, mirna_mask)   # [B, 1, D]
        t_self = self.self_pool(t_q, t, t, target_mask)  # [B, 1, D]

        # Cross-attention pooling: each query attends to the OTHER sequence
        m_cross = self.cross_pool(m_q, t, t, target_mask)  # miRNA query -> target
        t_cross = self.cross_pool(t_q, m, m, mirna_mask)   # target query -> miRNA

        # Squeeze the query dimension
        m_self = m_self.squeeze(1)   # [B, D]
        t_self = t_self.squeeze(1)   # [B, D]
        m_cross = m_cross.squeeze(1) # [B, D]
        t_cross = t_cross.squeeze(1) # [B, D]

        # Concatenate all 4 views and project
        combined = torch.cat([m_self, m_cross, t_self, t_cross], dim=-1)  # [B, 4D]
        pooled = self.out_proj(combined)  # [B, 2D]

        return pooled


class _PoolingAttention(nn.Module):
    """Multi-head attention for pooling (query attends to key/value)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,     # [B, 1, D]
        key: torch.Tensor,       # [B, L, D]
        value: torch.Tensor,     # [B, L, D]
        key_mask: Optional[torch.Tensor] = None,  # [B, L], True = padding
    ) -> torch.Tensor:
        B, Lq, D = query.shape
        Lk = key.shape[1]

        q = self.q_proj(query).view(B, Lq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: [B, H, 1, Lk]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_mask is not None:
            # key_mask: [B, Lk] -> [B, 1, 1, Lk]
            attn = attn.masked_fill(key_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # [B, H, 1, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Lq, D)
        out = self.out_proj(out)

        return out
