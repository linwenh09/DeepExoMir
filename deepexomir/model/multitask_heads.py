"""Multi-task learning auxiliary heads for Model 8.

Auxiliary tasks provide additional supervision signal and act as
regularizers for the shared encoder. Each head operates on the
encoder's output representations.

Tasks:
    1. Primary: miRNA-target interaction prediction (binary, handled by
       MoE classifier, not in this module)
    2. Seed match type classification (8 classes)
    3. Duplex MFE regression (continuous)
    4. Binding position regression (which target position is bound)

The multi-task loss is a weighted sum::

    L = L_primary
      + w_seed * L_seed_type
      + w_mfe * L_mfe
      + w_pos * L_position
      + w_lb * L_load_balance

References:
    - Kendall et al., "Multi-Task Learning Using Uncertainty to Weigh
      Losses for Scene Geometry and Semantics", CVPR 2018
    - CrossLLM-Mamba: multi-task miRNA-target, 2026
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Seed match type categories
SEED_MATCH_TYPES = [
    "8mer",         # Perfect seed + position 1 A
    "7mer-m8",      # Perfect seed, no position 1 match
    "7mer-A1",      # Seed 2-7 match + position 1 A
    "6mer",         # Seed 2-7 match only
    "6mer-A1",      # Seed 2-6 match + position 1 A
    "offset-6mer",  # Seed 3-8 match
    "non-canonical", # Non-canonical / compensatory
    "no_site",      # No recognizable site
]


class MultiTaskHeads(nn.Module):
    """Auxiliary prediction heads for multi-task learning.

    Each head is a small MLP that takes the encoder's sequence
    representation as input and produces a task-specific prediction.

    Parameters
    ----------
    encoder_dim : int
        Dimension of the encoder output per sequence (after mean pooling).
        Default: 512 (= d_model * 2 from concatenated miRNA + target).
    n_seed_types : int
        Number of seed match type classes (default: 8).
    dropout : float
        Dropout rate in head MLPs (default: 0.1).
    """

    def __init__(
        self,
        encoder_dim: int = 512,
        n_seed_types: int = 8,
        dropout: float = 0.1,
        contrastive_dim: int = 0,
    ) -> None:
        super().__init__()

        # Task 2: Seed match type classification (8 classes)
        self.seed_type_head = nn.Sequential(
            nn.Linear(encoder_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_seed_types),
        )

        # Task 3: Duplex MFE regression
        self.mfe_head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

        # Task 4: Binding position regression
        # Predicts which target position (0 to max_target_len-1)
        # is the center of the binding site
        self.position_head = nn.Sequential(
            nn.Linear(encoder_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output in [0, 1], scale to target length
        )

        # Task 5: Contrastive projection (optional)
        if contrastive_dim > 0:
            self.contrastive_head = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(encoder_dim // 2, contrastive_dim),
            )

    def forward(
        self,
        seq_repr: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute auxiliary task predictions.

        Parameters
        ----------
        seq_repr : Tensor [B, encoder_dim]
            Concatenated mean-pooled miRNA + target representation
            from the encoder.

        Returns
        -------
        dict
            - ``seed_type_logits``: Tensor [B, n_seed_types]
            - ``mfe_pred``: Tensor [B, 1]
            - ``position_pred``: Tensor [B, 1] (in [0, 1])
            - ``contrastive_proj``: Tensor [B, proj_dim] (if head exists)
        """
        result = {
            "seed_type_logits": self.seed_type_head(seq_repr),
            "mfe_pred": self.mfe_head(seq_repr),
            "position_pred": self.position_head(seq_repr),
        }
        if hasattr(self, "contrastive_head"):
            proj = self.contrastive_head(seq_repr)
            result["contrastive_proj"] = F.normalize(proj.float(), dim=-1, eps=1e-6)
        return result


class MultiTaskLoss(nn.Module):
    """Compute weighted multi-task loss.

    Combines:
        - Primary focal loss (from existing losses.py)
        - Seed type cross-entropy
        - MFE mean squared error
        - Binding position MSE
        - MoE load-balancing loss

    Supports optional learnable uncertainty-based weighting
    (Kendall et al., 2018).

    Parameters
    ----------
    w_seed : float
        Weight for seed type classification loss (default: 0.3).
    w_mfe : float
        Weight for MFE regression loss (default: 0.2).
    w_position : float
        Weight for binding position loss (default: 0.2).
    w_load_balance : float
        Weight for MoE load-balancing loss (default: 0.01).
    learnable_weights : bool
        If True, use learnable log-variance weights (default: False).
    """

    def __init__(
        self,
        w_seed: float = 0.3,
        w_mfe: float = 0.2,
        w_position: float = 0.2,
        w_load_balance: float = 0.01,
        w_contrastive: float = 0.0,
        contrastive_temperature: float = 0.07,
        learnable_weights: bool = False,
    ) -> None:
        super().__init__()
        self.w_seed = w_seed
        self.w_mfe = w_mfe
        self.w_position = w_position
        self.w_load_balance = w_load_balance
        self.w_contrastive = w_contrastive
        self.contrastive_temperature = contrastive_temperature
        self.learnable_weights = learnable_weights

        if learnable_weights:
            # Log-variance parameters (Kendall et al.)
            # Initialize such that initial weight ≈ configured weight
            self.log_var_seed = nn.Parameter(torch.tensor(0.0))
            self.log_var_mfe = nn.Parameter(torch.tensor(0.0))
            self.log_var_position = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        primary_loss: torch.Tensor,
        aux_preds: Dict[str, torch.Tensor],
        seed_type_labels: Optional[torch.Tensor] = None,
        mfe_labels: Optional[torch.Tensor] = None,
        position_labels: Optional[torch.Tensor] = None,
        load_balance_loss: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total multi-task loss.

        Parameters
        ----------
        primary_loss : Tensor (scalar)
            Primary task loss (e.g., focal loss for interaction prediction).
        aux_preds : dict
            Output from MultiTaskHeads.forward().
        seed_type_labels : Tensor [B], optional
            Ground-truth seed match type indices (0-7).
        mfe_labels : Tensor [B, 1], optional
            Ground-truth duplex MFE values.
        position_labels : Tensor [B, 1], optional
            Ground-truth normalized binding positions (in [0, 1]).
        load_balance_loss : Tensor (scalar), optional
            MoE load-balancing loss.

        Returns
        -------
        total_loss : Tensor (scalar)
        loss_dict : dict
            Individual loss components for logging.
        """
        loss_dict = {"primary": primary_loss.detach()}
        total = primary_loss

        # Seed type classification
        if seed_type_labels is not None and "seed_type_logits" in aux_preds:
            seed_loss = F.cross_entropy(
                aux_preds["seed_type_logits"],
                seed_type_labels.long(),
            )
            if self.learnable_weights:
                w = torch.exp(-self.log_var_seed)
                total = total + w * seed_loss + self.log_var_seed
            else:
                total = total + self.w_seed * seed_loss
            loss_dict["seed_type"] = seed_loss.detach()

        # MFE regression
        if mfe_labels is not None and "mfe_pred" in aux_preds:
            mfe_loss = F.mse_loss(
                aux_preds["mfe_pred"].squeeze(-1),
                mfe_labels.float(),
            )
            if self.learnable_weights:
                w = torch.exp(-self.log_var_mfe)
                total = total + w * mfe_loss + self.log_var_mfe
            else:
                total = total + self.w_mfe * mfe_loss
            loss_dict["mfe"] = mfe_loss.detach()

        # Binding position regression
        if position_labels is not None and "position_pred" in aux_preds:
            pos_loss = F.mse_loss(
                aux_preds["position_pred"].squeeze(-1),
                position_labels.float(),
            )
            if self.learnable_weights:
                w = torch.exp(-self.log_var_position)
                total = total + w * pos_loss + self.log_var_position
            else:
                total = total + self.w_position * pos_loss
            loss_dict["position"] = pos_loss.detach()

        # MoE load balancing
        if load_balance_loss is not None:
            total = total + self.w_load_balance * load_balance_loss
            loss_dict["load_balance"] = load_balance_loss.detach()

        # Supervised contrastive loss
        if (self.w_contrastive > 0
                and "contrastive_proj" in aux_preds
                and labels is not None):
            con_loss = self._supcon_loss(
                aux_preds["contrastive_proj"],
                labels,
                temperature=self.contrastive_temperature,
            )
            total = total + self.w_contrastive * con_loss
            loss_dict["contrastive"] = con_loss.detach()

        loss_dict["total"] = total.detach()

        return total, loss_dict

    @staticmethod
    def _supcon_loss(
        projections: torch.Tensor,
        labels: torch.Tensor,
        temperature: float = 0.07,
    ) -> torch.Tensor:
        """Supervised contrastive loss (SupCon).

        Pulls same-label samples together, pushes different-label
        samples apart in the projection space.

        Parameters
        ----------
        projections : Tensor [B, D]
            L2-normalized projection embeddings.
        labels : Tensor [B]
            Binary labels (0 or 1).
        temperature : float
            Temperature scaling (lower = sharper).

        Returns
        -------
        Tensor (scalar)
        """
        B = projections.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=projections.device)

        # Cosine similarity matrix: [B, B]
        sim = torch.matmul(projections, projections.T) / temperature

        # Mask: same label = positive pair
        labels = labels.view(-1)
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()

        # Remove self-pairs from positives
        eye = torch.eye(B, device=projections.device)
        mask_pos = mask_pos - eye

        # Count positive pairs per sample
        n_pos = mask_pos.sum(dim=1)

        # Log-sum-exp trick for numerical stability
        logits_max, _ = sim.max(dim=1, keepdim=True)
        logits = sim - logits_max.detach()

        # Exclude self from denominator
        exp_logits = torch.exp(logits) * (1.0 - eye)
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

        # Mean of log-likelihood over positive pairs
        valid = n_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=projections.device)

        mean_log_prob = (mask_pos * log_prob).sum(dim=1) / (n_pos + 1e-8)
        loss = -mean_log_prob[valid].mean()

        return loss
