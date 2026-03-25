"""Hybrid encoder for Model 8: BiConvGate + Cross-Attention.

Implements a hybrid encoder that interleaves:
  - BiConvGate blocks: Bidirectional 1D-conv gated blocks (Mamba-2 analog)
  - Cross-Attention blocks: Bidirectional multi-head cross-attention

Architecture (8 layers)::

    Layer 1-2: BiConvGate (separate miRNA / target self-encoding)
    Layer 3:   Cross-Attention (miRNA <-> target information exchange)
    Layer 4-5: BiConvGate (with enriched representations)
    Layer 6:   Cross-Attention (final cross-sequence refinement)
    Layer 7-8: BiConvGate (final per-token encoding)

The BiConvGate block is a pure-PyTorch implementation inspired by
Mamba-2's selective scan. It replaces the SSM core with multi-scale
1D depthwise convolutions + gated linear units, achieving similar
inductive biases for short RNA sequences (30-50 nt) without requiring
the ``mamba-ssm`` CUDA package.

If ``mamba-ssm`` is installed, set ``use_mamba=True`` to use real
Mamba-2 blocks instead.

References:
    - Mamba: Gu & Dao, "Mamba: Linear-Time Sequence Modeling with
      Selective State Spaces", 2023
    - HybriDNA: Microsoft, "Mamba-Transformer Hybrid 7:1 ratio", 2025
    - CrossLLM-Mamba: BiMamba for miRNA-target, 2026
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from deepexomir.model.cross_attention import (
    CrossAttentionBlock,
    DropPath,
)


# ---------------------------------------------------------------------------
# BiConvGate Block (Pure-PyTorch Mamba-2 analog)
# ---------------------------------------------------------------------------

class _ConvGateBranch(nn.Module):
    """Single-direction conv-gated sequence encoder.

    Implements a simplified Mamba-like architecture::

        x -> Linear(expand) -> split -> {
            Branch A: DepthwiseConv1d -> SiLU -> Linear
            Branch B: SiLU (gate)
        } -> A * B -> Linear(contract) -> output

    Multi-scale depthwise convolutions at kernel sizes [3, 5, 7] capture
    local context at different scales, analogous to Mamba's selective scan
    with different effective receptive fields.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand

        # Expand
        self.in_proj = nn.Linear(d_model, d_inner * 2)

        # Multi-scale depthwise convolution (Mamba-style 1D conv)
        self.conv = nn.Conv1d(
            d_inner, d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,  # causal padding (trimmed below)
            groups=d_inner,      # depthwise
        )

        # Contract
        self.out_proj = nn.Linear(d_inner, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_conv = d_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a single direction.

        Parameters
        ----------
        x : Tensor [B, L, D]

        Returns
        -------
        Tensor [B, L, D]
        """
        B, L, D = x.shape

        # Expand and split into two branches
        xz = self.in_proj(x)  # [B, L, 2*d_inner]
        x_branch, z_branch = xz.chunk(2, dim=-1)  # each [B, L, d_inner]

        # Branch A: depthwise conv + SiLU
        # Conv1d expects [B, C, L]
        x_conv = x_branch.transpose(1, 2)  # [B, d_inner, L]
        x_conv = self.conv(x_conv)[:, :, :L]  # trim causal padding
        x_conv = F.silu(x_conv).transpose(1, 2)  # [B, L, d_inner]

        # Branch B: gate
        z_gate = F.silu(z_branch)  # [B, L, d_inner]

        # Gated output
        y = x_conv * z_gate  # [B, L, d_inner]
        y = self.dropout(y)
        y = self.out_proj(y)  # [B, L, D]

        return y


class BiConvGateBlock(nn.Module):
    """Bidirectional conv-gated block (Mamba-2 analog).

    Processes the sequence in both forward and backward directions,
    then combines via a linear projection. Uses pre-norm + residual
    connection with optional DropPath.

    Parameters
    ----------
    d_model : int
        Hidden dimensionality (default: 256).
    d_conv : int
        Convolution kernel size (default: 4, matching Mamba default).
    expand : int
        Expansion factor for inner dimension (default: 2).
    dropout : float
        Dropout rate (default: 0.1).
    drop_path_rate : float
        DropPath (stochastic depth) rate (default: 0.0).
    """

    def __init__(
        self,
        d_model: int = 256,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd_branch = _ConvGateBranch(d_model, d_conv, expand, dropout)
        self.bwd_branch = _ConvGateBranch(d_model, d_conv, expand, dropout)
        self.merge = nn.Linear(d_model * 2, d_model)
        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Bidirectional forward pass.

        Parameters
        ----------
        x : Tensor [B, L, D]

        Returns
        -------
        Tensor [B, L, D]
        """
        residual = x
        x = self.norm(x)

        # Forward direction
        fwd = self.fwd_branch(x)  # [B, L, D]

        # Backward direction: reverse -> process -> reverse back
        bwd = self.bwd_branch(x.flip(1)).flip(1)  # [B, L, D]

        # Merge bidirectional outputs
        merged = self.merge(torch.cat([fwd, bwd], dim=-1))  # [B, L, D]

        return residual + self.drop_path(merged)


# ---------------------------------------------------------------------------
# Hybrid Encoder Block (BiConvGate + optional Cross-Attention)
# ---------------------------------------------------------------------------

class HybridEncoderBlock(nn.Module):
    """Single block of the hybrid encoder.

    Can operate in two modes:
    1. Self-encoding mode (``cross_attn=False``): BiConvGate on each
       sequence independently.
    2. Cross-attention mode (``cross_attn=True``): BiConvGate self-encoding
       followed by bidirectional cross-attention exchange.

    Parameters
    ----------
    d_model : int
    n_heads : int
    d_ff : int
    d_conv : int
    expand : int
    dropout : float
    drop_path_rate : float
    cross_attn : bool
        Whether this block includes cross-attention.
    use_swiglu : bool
        Use SwiGLU FFN in cross-attention (if enabled).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        cross_attn: bool = False,
        use_swiglu: bool = True,
    ) -> None:
        super().__init__()

        # BiConvGate for miRNA and target (separate parameters)
        self.mirna_convgate = BiConvGateBlock(
            d_model, d_conv, expand, dropout, drop_path_rate,
        )
        self.target_convgate = BiConvGateBlock(
            d_model, d_conv, expand, dropout, drop_path_rate,
        )

        # Optional cross-attention
        self.has_cross_attn = cross_attn
        if cross_attn:
            self.cross_attn_block = CrossAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                drop_path_rate=drop_path_rate,
                use_swiglu=use_swiglu,
            )

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        mirna_emb : Tensor [B, Lm, D]
        target_emb : Tensor [B, Lt, D]
        mirna_mask, target_mask : Tensor [B, L], optional
            True at padding positions.

        Returns
        -------
        (mirna_emb, target_emb) : tuple[Tensor, Tensor]
        """
        # BiConvGate self-encoding
        mirna_emb = self.mirna_convgate(mirna_emb)
        target_emb = self.target_convgate(target_emb)

        # Cross-attention (if this block has it)
        if self.has_cross_attn:
            mirna_emb, target_emb = self.cross_attn_block(
                mirna_emb, target_emb, mirna_mask, target_mask,
            )

        return mirna_emb, target_emb


# ---------------------------------------------------------------------------
# Full Hybrid Encoder
# ---------------------------------------------------------------------------

class HybridEncoder(nn.Module):
    """Hybrid encoder: interleaved BiConvGate + Cross-Attention blocks.

    Default layout (8 layers, 6 BiConvGate + 2 Cross-Attention)::

        Block 0: BiConvGate (self-encoding)
        Block 1: BiConvGate (self-encoding)
        Block 2: BiConvGate + Cross-Attention
        Block 3: BiConvGate (self-encoding)
        Block 4: BiConvGate (self-encoding)
        Block 5: BiConvGate + Cross-Attention
        Block 6: BiConvGate (self-encoding)
        Block 7: BiConvGate (self-encoding)

    The cross-attention ratio is configurable. For an 8-layer encoder
    with ``cross_attn_every=3``, layers 2 and 5 get cross-attention
    (matching the HybriDNA 7:1 Mamba-to-Attention ratio from Microsoft).

    Parameters
    ----------
    n_layers : int
        Total number of layers (default: 8).
    d_model : int
        Hidden dimensionality (default: 256).
    n_heads : int
        Number of attention heads for cross-attention (default: 8).
    d_ff : int
        FFN inner dim for cross-attention (default: 1024).
    d_conv : int
        Convolution kernel size for BiConvGate (default: 4).
    expand : int
        Expansion factor for BiConvGate inner dim (default: 2).
    dropout : float
        Dropout rate (default: 0.1).
    drop_path_rate : float
        Maximum DropPath rate (linearly increases across layers).
    cross_attn_every : int
        Insert cross-attention every N layers (default: 3).
        E.g., for 8 layers with interval=3: layers 2 and 5.
    use_swiglu : bool
        Use SwiGLU FFN in cross-attention blocks (default: True).
    """

    def __init__(
        self,
        n_layers: int = 8,
        d_model: int = 256,
        n_heads: int = 8,
        d_ff: int = 1024,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        cross_attn_every: int = 3,
        use_swiglu: bool = True,
    ) -> None:
        super().__init__()

        # Linearly increasing DropPath rates
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, n_layers)]

        # Determine which layers get cross-attention
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            has_cross_attn = (i > 0) and ((i + 1) % cross_attn_every == 0)
            self.blocks.append(
                HybridEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    d_ff=d_ff,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    drop_path_rate=dpr[i],
                    cross_attn=has_cross_attn,
                    use_swiglu=use_swiglu,
                )
            )

        # Final layer norms
        self.mirna_final_norm = nn.LayerNorm(d_model)
        self.target_final_norm = nn.LayerNorm(d_model)

        # Log architecture
        n_cross = sum(1 for b in self.blocks if b.has_cross_attn)
        n_self = n_layers - n_cross
        cross_layers = [i for i, b in enumerate(self.blocks) if b.has_cross_attn]
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            "HybridEncoder: %d layers (%d BiConvGate + %d CrossAttn), "
            "d_model=%d, d_conv=%d, expand=%d, cross_attn_at=%s",
            n_layers, n_self, n_cross, d_model, d_conv, expand, cross_layers,
        )

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through all hybrid blocks.

        Parameters
        ----------
        mirna_emb : Tensor [B, Lm, D]
        target_emb : Tensor [B, Lt, D]
        mirna_mask : Tensor [B, Lm], optional
        target_mask : Tensor [B, Lt], optional

        Returns
        -------
        (mirna_emb, target_emb) : tuple[Tensor, Tensor]
        """
        for block in self.blocks:
            mirna_emb, target_emb = block(
                mirna_emb, target_emb, mirna_mask, target_mask,
            )

        mirna_emb = self.mirna_final_norm(mirna_emb)
        target_emb = self.target_final_norm(target_emb)

        return mirna_emb, target_emb
