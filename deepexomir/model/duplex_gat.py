"""Duplex Graph Attention Network for miRNA-target interaction.

Builds a graph from the miRNA-target duplex where:
- Nodes = nucleotides (miRNA + target), features from HybridEncoder output
- Edges:
    - Backbone: sequential within miRNA and within target
    - Base-pair: Watson-Crick/wobble pairs between miRNA and target
    - Proximity: within distance-2 backbone hops (captures stacking)
- Uses GATv2Conv for message passing, then pools to a fixed-dim vector.

Inspired by AdarEdit (2026) structure-aware graph attention framework.
"""
from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool

logger = logging.getLogger(__name__)


class DuplexGAT(nn.Module):
    """Graph Attention Network on the miRNA-target duplex structure.

    Parameters
    ----------
    node_dim : int
        Input node feature dimension (d_model from encoder).
    hidden_dim : int
        Hidden dimension for GAT layers.
    out_dim : int
        Output dimension after graph pooling.
    n_heads : int
        Number of attention heads per GAT layer.
    n_layers : int
        Number of GAT layers.
    dropout : float
        Dropout rate.
    max_mirna_len : int
        Maximum miRNA length.
    max_target_len : int
        Maximum target length.
    """

    def __init__(
        self,
        node_dim: int = 256,
        hidden_dim: int = 128,
        out_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        max_mirna_len: int = 30,
        max_target_len: int = 50,
    ) -> None:
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len
        self.n_layers = n_layers

        # Node type embedding (0=miRNA, 1=target)
        self.node_type_emb = nn.Embedding(2, node_dim)

        # Edge type embedding (0=backbone, 1=base-pair, 2=proximity)
        self.edge_type_emb = nn.Embedding(3, n_heads)

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(node_dim, hidden_dim * n_heads),
            nn.LayerNorm(hidden_dim * n_heads),
            nn.GELU(),
        )

        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(n_layers):
            in_ch = hidden_dim * n_heads
            self.gat_layers.append(
                GATv2Conv(
                    in_channels=in_ch,
                    out_channels=hidden_dim,
                    heads=n_heads,
                    concat=True,  # output = hidden_dim * n_heads
                    dropout=dropout,
                    add_self_loops=True,
                    edge_dim=n_heads,  # edge features = edge type embedding
                )
            )
            self.layer_norms.append(nn.LayerNorm(hidden_dim * n_heads))

        # Output projection: concat mean + max pool
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * n_heads * 2, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Precompute backbone edges template (reused for every sample)
        self._register_backbone_edges()

        logger.info(
            "DuplexGAT: %d layers, %d heads, hidden=%d, out=%d",
            n_layers, n_heads, hidden_dim, out_dim,
        )

    def _register_backbone_edges(self) -> None:
        """Pre-compute backbone edge indices for miRNA and target."""
        mirna_len = self.max_mirna_len
        target_len = self.max_target_len

        # miRNA backbone: 0-1, 1-2, ..., (L-2)-(L-1)
        mirna_src = list(range(mirna_len - 1))
        mirna_dst = list(range(1, mirna_len))

        # Target backbone: L_m - (L_m+1), ..., (L_m+L_t-2)-(L_m+L_t-1)
        offset = mirna_len
        target_src = [offset + i for i in range(target_len - 1)]
        target_dst = [offset + i for i in range(1, target_len)]

        # Bidirectional
        src = mirna_src + mirna_dst + target_src + target_dst
        dst = mirna_dst + mirna_src + target_dst + target_src

        backbone_edges = torch.tensor([src, dst], dtype=torch.long)
        self.register_buffer("_backbone_edges", backbone_edges, persistent=False)

        # Also precompute proximity edges (distance-2 hops within sequence)
        mirna_src2 = list(range(mirna_len - 2))
        mirna_dst2 = list(range(2, mirna_len))
        target_src2 = [offset + i for i in range(target_len - 2)]
        target_dst2 = [offset + i for i in range(2, target_len)]

        src2 = mirna_src2 + mirna_dst2 + target_src2 + target_dst2
        dst2 = mirna_dst2 + mirna_src2 + target_dst2 + target_src2

        prox_edges = torch.tensor([src2, dst2], dtype=torch.long)
        self.register_buffer("_proximity_edges", prox_edges, persistent=False)

    def _build_bp_edges(
        self,
        mirna_seqs: list[str],
        target_seqs: list[str],
        batch_size: int,
    ) -> list[torch.Tensor]:
        """Build base-pairing edges for each sample in the batch.

        Uses simple Watson-Crick + wobble matching on aligned positions.
        miRNA 3'->5' aligns with target 5'->3', so miRNA[i] pairs with
        target[len-1-i] in the canonical seed alignment.
        """
        bp_edges_list = []
        offset = self.max_mirna_len
        complement = {
            ("A", "U"), ("U", "A"),
            ("G", "C"), ("C", "G"),
            ("G", "U"), ("U", "G"),  # wobble
        }

        for b in range(batch_size):
            src, dst = [], []
            m_seq = mirna_seqs[b].upper().replace("T", "U") if b < len(mirna_seqs) else ""
            t_seq = target_seqs[b].upper().replace("T", "U") if b < len(target_seqs) else ""

            # Reverse complement alignment: miRNA 3'->5' vs target 5'->3'
            m_len = min(len(m_seq), self.max_mirna_len)
            t_len = min(len(t_seq), self.max_target_len)

            # Align from miRNA 3' end (position m_len-1) with target 5' end (position 0)
            # miRNA position i aligns with target position (t_len - 1 - i) for seed region
            align_len = min(m_len, t_len)
            for i in range(align_len):
                m_pos = i
                t_pos = t_len - 1 - i  # reverse alignment
                if m_pos < m_len and t_pos >= 0 and t_pos < t_len:
                    m_nt = m_seq[m_pos] if m_pos < len(m_seq) else "N"
                    t_nt = t_seq[t_pos] if t_pos < len(t_seq) else "N"
                    if (m_nt, t_nt) in complement:
                        # Bidirectional edges
                        src.extend([m_pos, offset + t_pos])
                        dst.extend([offset + t_pos, m_pos])

            if src:
                bp_edges_list.append(
                    torch.tensor([src, dst], dtype=torch.long)
                )
            else:
                bp_edges_list.append(
                    torch.zeros(2, 0, dtype=torch.long)
                )

        return bp_edges_list

    def forward(
        self,
        mirna_enc: torch.Tensor,      # [B, Lm, D] from encoder
        target_enc: torch.Tensor,      # [B, Lt, D] from encoder
        mirna_seqs: Optional[list[str]] = None,
        target_seqs: Optional[list[str]] = None,
        mirna_mask: Optional[torch.Tensor] = None,   # [B, Lm] True=pad
        target_mask: Optional[torch.Tensor] = None,   # [B, Lt] True=pad
    ) -> torch.Tensor:
        """Forward pass.

        Returns
        -------
        Tensor [B, out_dim]
            Graph-level representation of the duplex.
        """
        B = mirna_enc.shape[0]
        device = mirna_enc.device
        Lm = self.max_mirna_len
        Lt = self.max_target_len
        total_nodes = Lm + Lt

        # Add node type embeddings
        mirna_type = self.node_type_emb(torch.zeros(Lm, dtype=torch.long, device=device))
        target_type = self.node_type_emb(torch.ones(Lt, dtype=torch.long, device=device))

        mirna_enc = mirna_enc[:, :Lm, :] + mirna_type.unsqueeze(0)
        target_enc = target_enc[:, :Lt, :] + target_type.unsqueeze(0)

        # Build base-pair edges per sample
        if mirna_seqs is not None and target_seqs is not None:
            bp_edges_list = self._build_bp_edges(mirna_seqs, target_seqs, B)
        else:
            bp_edges_list = [torch.zeros(2, 0, dtype=torch.long) for _ in range(B)]

        # Construct PyG Batch
        data_list = []
        backbone = self._backbone_edges.to(device)
        proximity = self._proximity_edges.to(device)

        for b in range(B):
            # Node features: concat miRNA + target
            node_feat = torch.cat([mirna_enc[b], target_enc[b]], dim=0)  # [Lm+Lt, D]

            # Build edge index and edge types
            bp_edges = bp_edges_list[b].to(device)
            n_bb = backbone.shape[1]
            n_bp = bp_edges.shape[1]
            n_px = proximity.shape[1]

            edge_index = torch.cat([backbone, bp_edges, proximity], dim=1)

            # Edge type: 0=backbone, 1=base-pair, 2=proximity
            edge_type = torch.cat([
                torch.zeros(n_bb, dtype=torch.long, device=device),
                torch.ones(n_bp, dtype=torch.long, device=device),
                torch.full((n_px,), 2, dtype=torch.long, device=device),
            ])

            data_list.append(Data(
                x=node_feat,
                edge_index=edge_index,
                edge_type=edge_type,
            ))

        batch = Batch.from_data_list(data_list)

        # Project node features
        x = self.input_proj(batch.x)

        # Edge features from type embedding
        edge_attr = self.edge_type_emb(batch.edge_type)

        # GAT layers with residual connections
        for i in range(self.n_layers):
            residual = x
            x = self.gat_layers[i](x, batch.edge_index, edge_attr=edge_attr)
            x = self.layer_norms[i](x + residual)  # residual connection
            if i < self.n_layers - 1:
                x = F.gelu(x)

        # Pool: mean + max
        mean_pool = global_mean_pool(x, batch.batch)  # [B, hidden*heads]
        max_pool = global_max_pool(x, batch.batch)     # [B, hidden*heads]
        pooled = torch.cat([mean_pool, max_pool], dim=-1)  # [B, hidden*heads*2]

        # Output projection
        out = self.output_proj(pooled)  # [B, out_dim]

        return out
