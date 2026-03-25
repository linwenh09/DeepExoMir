"""Loss functions for DeepExoMir training.

The primary loss is **Focal Loss** (Lin et al., 2017) with optional label
smoothing.  Focal Loss down-weights well-classified examples so that the
model focuses on hard, misclassified cases -- particularly useful for the
imbalanced miRNA-target interaction datasets where true positives are
significantly rarer than negatives.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss with optional label smoothing.

    Focal Loss modulates the standard cross-entropy by a factor
    ``(1 - p_t)^gamma`` so that easy examples receive a lower loss
    contribution:

    .. math::

        FL(p_t) = -\\alpha_t \\, (1 - p_t)^{\\gamma} \\, \\log(p_t)

    When ``gamma = 0`` this reduces to standard weighted cross-entropy.

    Parameters
    ----------
    gamma : float
        Focusing parameter.  Higher values down-weight easy examples more
        aggressively (default: 2.0).
    alpha : float or None
        Weighting factor for the **positive** class.  The negative class
        receives weight ``1 - alpha``.  Set to ``None`` for no class
        weighting (default: 0.75).
    label_smoothing : float
        Label smoothing factor in ``[0, 1)``.  Replaces hard targets
        ``{0, 1}`` with ``{eps/K, 1 - eps + eps/K}`` where ``K`` is the
        number of classes (default: 0.05).
    reduction : str
        ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[float] = 0.75,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Parameters
        ----------
        logits : Tensor [B, C]
            Raw (un-normalised) class scores.  ``C`` is the number of classes.
        targets : Tensor [B]
            Ground-truth class indices (dtype ``long``).

        Returns
        -------
        Tensor
            Scalar loss (if ``reduction`` is ``"mean"`` or ``"sum"``) or
            per-sample losses ``[B]`` (if ``"none"``).
        """
        n_classes = logits.shape[-1]

        # ---- label smoothing ------------------------------------------------
        # Convert integer targets to soft targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(logits)
            smooth_targets.fill_(self.label_smoothing / n_classes)
            smooth_targets.scatter_(
                1,
                targets.unsqueeze(1),
                1.0 - self.label_smoothing + self.label_smoothing / n_classes,
            )

        # ---- per-class probabilities ----------------------------------------
        log_probs = F.log_softmax(logits, dim=-1)  # [B, C]
        probs = log_probs.exp()                     # [B, C]

        # ---- focal modulating factor ----------------------------------------
        # p_t for each class weighted by the soft target
        focal_weight = (1.0 - probs).pow(self.gamma)  # [B, C]

        # ---- alpha weighting ------------------------------------------------
        if self.alpha is not None:
            alpha_tensor = torch.full(
                (n_classes,), 1.0 - self.alpha,
                dtype=logits.dtype, device=logits.device,
            )
            alpha_tensor[1] = self.alpha  # positive class = index 1
            alpha_weight = alpha_tensor.unsqueeze(0)  # [1, C]
        else:
            alpha_weight = 1.0

        # ---- loss computation -----------------------------------------------
        # Element-wise: -alpha * (1-p)^gamma * log(p) * soft_target
        loss = -alpha_weight * focal_weight * log_probs * smooth_targets  # [B, C]
        loss = loss.sum(dim=-1)  # [B]

        # ---- reduction ------------------------------------------------------
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, alpha={self.alpha}, "
            f"label_smoothing={self.label_smoothing}, reduction={self.reduction!r}"
        )
