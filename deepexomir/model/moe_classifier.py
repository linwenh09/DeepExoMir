"""Mixture of Experts (MoE) classification head for Model 8.

Replaces the simple MLP classification head with 4 specialized experts:
  - Expert 1: Canonical seed-site interactions (8mer, 7mer)
  - Expert 2: 3' compensatory binding sites
  - Expert 3: Non-canonical / bulge-mediated sites
  - Expert 4: Structure-dependent interactions

A learned gating network routes each sample to the top-K experts
(default K=2) based on the input feature vector. A load-balancing
auxiliary loss encourages uniform expert utilization.

References:
    - Shazeer et al., "Outrageously Large Neural Networks", 2017
    - Fedus et al., "Switch Transformers", 2022
    - MoEFold2D (RNA structure MoE), Biol Methods, 2025
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """Single expert: a 3-layer MLP with GELU and dropout."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 2,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MoEClassifier(nn.Module):
    """Mixture of Experts classification head.

    Parameters
    ----------
    in_dim : int
        Input feature dimensionality.
    hidden_dim : int
        Hidden dimension per expert (default: 256).
    n_experts : int
        Number of experts (default: 4).
    top_k : int
        Number of experts selected per sample (default: 2).
    n_classes : int
        Number of output classes (default: 2).
    dropout : float
        Dropout rate inside experts (default: 0.15).
    lb_weight : float
        Load-balancing loss weight (default: 0.01).
    platt_scaling : bool
        Whether to include a learnable temperature parameter.
    """

    def __init__(
        self,
        in_dim: int = 1088,
        hidden_dim: int = 256,
        n_experts: int = 4,
        top_k: int = 2,
        n_classes: int = 2,
        dropout: float = 0.15,
        lb_weight: float = 0.01,
        platt_scaling: bool = True,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.n_classes = n_classes
        self.lb_weight = lb_weight

        # Expert networks
        self.experts = nn.ModuleList([
            Expert(in_dim, hidden_dim, n_classes, dropout)
            for _ in range(n_experts)
        ])

        # Gating network
        gate_hidden = max(64, in_dim // 8)
        self.gate = nn.Sequential(
            nn.Linear(in_dim, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, n_experts),
        )

        # Temperature scaling
        self.platt_scaling = platt_scaling
        if platt_scaling:
            self.temperature = nn.Parameter(
                torch.tensor(1.0, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "temperature",
                torch.tensor(1.0, dtype=torch.float32),
            )

        # Store load balance loss (computed during forward)
        self.load_balance_loss = torch.tensor(0.0)

    def _compute_load_balance_loss(
        self,
        gate_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the load-balancing auxiliary loss.

        Encourages uniform expert utilization by penalizing the
        variance of expert selection frequencies.

        Parameters
        ----------
        gate_logits : Tensor [B, n_experts]
            Raw gating logits.

        Returns
        -------
        Tensor (scalar)
            Load-balancing loss.
        """
        # Fraction of tokens routed to each expert
        gate_probs = F.softmax(gate_logits, dim=-1)  # [B, n_experts]
        # Average routing probability per expert
        mean_probs = gate_probs.mean(dim=0)  # [n_experts]

        # One-hot expert selection
        _, indices = gate_logits.topk(self.top_k, dim=-1)
        expert_mask = torch.zeros_like(gate_logits)
        expert_mask.scatter_(1, indices, 1.0)
        # Fraction of tokens assigned to each expert
        mean_selected = expert_mask.mean(dim=0)  # [n_experts]

        # Loss = n_experts * sum(f_i * P_i) where f_i is fraction selected,
        # P_i is mean probability. Minimized when uniform.
        lb_loss = self.n_experts * (mean_probs * mean_selected).sum()

        return lb_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MoE forward pass.

        Parameters
        ----------
        x : Tensor [B, in_dim]
            Input feature vector.

        Returns
        -------
        Tensor [B, n_classes]
            Temperature-scaled logits.
        """
        B = x.shape[0]

        # Compute gating weights
        gate_logits = self.gate(x)  # [B, n_experts]
        top_k_logits, top_k_indices = gate_logits.topk(self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1)  # [B, top_k]

        # Compute all expert outputs (more efficient than sparse routing
        # for small n_experts=4)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=1
        )  # [B, n_experts, n_classes]

        # Select top-k expert outputs
        selected = expert_outputs.gather(
            1,
            top_k_indices.unsqueeze(-1).expand(-1, -1, self.n_classes),
        )  # [B, top_k, n_classes]

        # Weighted combination
        logits = (top_k_weights.unsqueeze(-1) * selected).sum(dim=1)  # [B, n_classes]

        # Temperature scaling
        logits = logits / self.temperature.clamp(min=1e-6)

        # Compute load-balancing loss (stored as attribute)
        if self.training:
            self.load_balance_loss = (
                self.lb_weight * self._compute_load_balance_loss(gate_logits)
            )
        else:
            self.load_balance_loss = torch.tensor(
                0.0, device=x.device, dtype=x.dtype,
            )

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return calibrated class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)

    def get_expert_assignments(self, x: torch.Tensor) -> torch.Tensor:
        """Return which experts are selected for each sample.

        Returns
        -------
        Tensor [B, top_k]
            Expert indices for each sample.
        """
        gate_logits = self.gate(x)
        _, indices = gate_logits.topk(self.top_k, dim=-1)
        return indices
