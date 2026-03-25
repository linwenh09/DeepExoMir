"""Main training loop for DeepExoMir miRNA-target interaction models.

Implements mixed-precision training with gradient accumulation, cosine
learning-rate scheduling with linear warmup, TensorBoard logging, early
stopping, top-*k* checkpoint management, and optional EMA (Exponential
Moving Average) parameter smoothing.

Configuration is read from ``configs/train_config.yaml`` and injected
as a plain dictionary at construction time.

Model v6 enhancements:
    - EMA (Exponential Moving Average) for smoother generalisation
"""

from __future__ import annotations

import copy
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from deepexomir.training.callbacks import EarlyStopping, ModelCheckpoint
from deepexomir.training.evaluator import Evaluator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average)
# ---------------------------------------------------------------------------

class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of trainable parameters that is updated
    each optimiser step as::

        shadow_param = decay * shadow_param + (1 - decay) * param

    During validation and checkpointing the shadow parameters are
    swapped into the model, producing a smoother and typically
    better-generalising set of weights.

    Parameters
    ----------
    model : nn.Module
        The model whose parameters to track.
    decay : float
        EMA decay factor (default: 0.9995).  Higher values produce
        smoother but more lagging averages.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9995) -> None:
        self.model = model
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        self._init_shadow()

    def _init_shadow(self) -> None:
        """Initialise shadow parameters as copies of current params."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply_shadow(self) -> None:
        """Replace model params with EMA shadow params for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model params after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return EMA shadow parameters for serialisation."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load EMA shadow parameters from a checkpoint."""
        for k, v in state_dict.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)


# ---------------------------------------------------------------------------
# Focal loss
# ---------------------------------------------------------------------------

def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = 0.75,
    label_smoothing: float = 0.05,
) -> torch.Tensor:
    """Focal loss for binary classification with optional label smoothing.

    Parameters
    ----------
    logits : torch.Tensor
        Raw logits of shape ``(B, 2)``.
    targets : torch.Tensor
        Ground-truth labels of shape ``(B,)``, values in ``{0, 1}``.
    gamma : float
        Focusing parameter.  ``gamma=0`` recovers standard cross-entropy.
    alpha : float
        Weighting factor for the positive class.
    label_smoothing : float
        Label smoothing epsilon.

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    num_classes = logits.size(1)
    # Apply label smoothing to one-hot targets
    with torch.no_grad():
        smooth_targets = torch.zeros_like(logits)
        smooth_targets.fill_(label_smoothing / (num_classes - 1))
        smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - label_smoothing)

    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)

    # Per-class alpha weighting
    alpha_weight = torch.tensor(
        [1.0 - alpha, alpha], device=logits.device, dtype=logits.dtype
    )
    alpha_t = alpha_weight[targets]  # (B,)

    # Focal modulation: (1 - p_t)^gamma
    p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
    focal_weight = (1.0 - p_t) ** gamma  # (B,)

    # Cross-entropy with smoothed targets
    ce = -(smooth_targets * log_probs).sum(dim=1)  # (B,)

    loss = alpha_t * focal_weight * ce
    return loss.mean()


# ---------------------------------------------------------------------------
# Learning-rate schedule with linear warmup + cosine decay
# ---------------------------------------------------------------------------

def _cosine_warmup_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    """Create a LambdaLR scheduler with linear warmup then cosine decay.

    During the first ``warmup_steps`` optimiser steps the learning rate
    increases linearly from 0 to the base LR.  After that it follows a
    cosine decay to 0.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(current_step / max(warmup_steps, 1), 0.0)
        progress = (current_step - warmup_steps) / max(
            total_steps - warmup_steps, 1
        )
        return max(0.5 * (1.0 + math.cos(math.pi * progress)), 0.0)

    return LambdaLR(optimizer, lr_lambda)


def _cosine_warmup_restarts_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    steps_per_epoch: int,
    t0_epochs: int = 10,
    t_mult: int = 2,
    min_lr_ratio: float = 0.01,
) -> LambdaLR:
    """Cosine annealing with warm restarts and LR floor (v10).

    After linear warmup, the LR follows a cosine schedule that restarts
    periodically.  Each restart cycle is ``t_mult`` times longer than the
    previous one.  The LR never drops below ``min_lr_ratio * base_lr``.

    Parameters
    ----------
    warmup_steps : int
        Number of linear warmup steps.
    steps_per_epoch : int
        Optimiser steps per epoch (after gradient accumulation).
    t0_epochs : int
        Length of the first cosine cycle in epochs (default: 10).
    t_mult : int
        Multiplicative factor for cycle length (default: 2).
        Cycle lengths: T0, T0*t_mult, T0*t_mult^2, ...
    min_lr_ratio : float
        Minimum LR as a fraction of base LR (default: 0.01 = 1%).
    """
    t0_steps = t0_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(current_step / max(warmup_steps, 1), 0.0)

        # Find which restart cycle we're in
        step = current_step - warmup_steps
        cycle_steps = t0_steps
        cycle_start = 0
        while step >= cycle_start + cycle_steps:
            cycle_start += cycle_steps
            cycle_steps = int(cycle_steps * t_mult)

        # Progress within current cycle [0, 1)
        progress = (step - cycle_start) / max(cycle_steps, 1)
        cosine_val = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Scale between min_lr_ratio and 1.0
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_val

    return LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class Trainer:
    """End-to-end training driver for DeepExoMir models.

    Parameters
    ----------
    model : nn.Module
        Model that accepts a batch dictionary and returns logits ``(B, 2)``.
    train_loader : DataLoader
        Training data loader yielding ``(batch_dict, labels)`` tuples.
    val_loader : DataLoader
        Validation data loader with the same contract.
    config : dict[str, Any]
        Nested configuration dictionary (mirrors ``train_config.yaml``).
        Expected top-level keys: ``training``, ``loss``, ``early_stopping``,
        ``mixed_precision``, ``checkpointing``, ``logging``.
    device : str or torch.device, optional
        Device to train on.  Defaults to ``"cuda"`` if available.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict[str, Any],
        device: str | torch.device | None = None,
    ) -> None:
        # ---- Device ----------------------------------------------------
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # ---- Data ------------------------------------------------------
        self.train_loader = train_loader
        self.val_loader = val_loader

        # ---- Config sub-dicts ------------------------------------------
        train_cfg = config.get("training", {})
        loss_cfg = config.get("loss", {})
        es_cfg = config.get("early_stopping", {})
        amp_cfg = config.get("mixed_precision", {})
        ckpt_cfg = config.get("checkpointing", {})
        log_cfg = config.get("logging", {})

        # ---- Training hyper-params -------------------------------------
        self.epochs: int = train_cfg.get("epochs", 50)
        self.accumulation_steps: int = train_cfg.get("accumulation_steps", 2)
        self.gradient_clip_norm: float = train_cfg.get("gradient_clip_norm", 1.0)
        self.log_every_n_steps: int = log_cfg.get("log_every_n_steps", 50)

        # ---- Loss params -----------------------------------------------
        self.focal_gamma: float = loss_cfg.get("gamma", 2.0)
        self.focal_alpha: float = loss_cfg.get("alpha", 0.75)
        self.label_smoothing: float = loss_cfg.get("label_smoothing", 0.05)

        # ---- Mixup augmentation (v10) -----------------------------------
        self.mixup_alpha: float = train_cfg.get("mixup_alpha", 0.0)
        self.mixup_prob: float = train_cfg.get("mixup_prob", 0.0)
        if self.mixup_alpha > 0:
            logger.info(
                "Embedding-level Mixup enabled: alpha=%.2f, prob=%.2f",
                self.mixup_alpha, self.mixup_prob,
            )

        # ---- Optimiser (with differential learning rates) ----------------
        lr: float = train_cfg.get("learning_rate", 3e-4)
        wd: float = train_cfg.get("weight_decay", 1e-4)
        backbone_lr_scale: float = train_cfg.get("backbone_lr_scale", 0.1)

        # Group parameters for differential learning rates:
        # - backbone_feature_mlp + projections (process backbone output): low LR
        # - everything else (cross-attention, lightweight encoder, classifier): base LR
        # - temperature parameter: FROZEN (calibrate post-hoc, conflicts with focal loss gamma)
        backbone_params = []
        base_params = []
        backbone_module_names = {"backbone_feature_mlp", "mirna_projection", "target_projection"}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Freeze Platt temperature during training — calibrate post-hoc
            if "temperature" in name:
                param.requires_grad = False
                logger.info("Freezing temperature parameter: %s", name)
                continue
            module_name = name.split(".")[0]
            if module_name in backbone_module_names:
                backbone_params.append(param)
            else:
                base_params.append(param)

        param_groups = [
            {"params": base_params, "lr": lr},
        ]
        if backbone_params:
            backbone_lr = lr * backbone_lr_scale
            param_groups.append(
                {"params": backbone_params, "lr": backbone_lr},
            )
            logger.info(
                "Differential LR: base=%.1e, backbone=%.1e (scale=%.2f)",
                lr, backbone_lr, backbone_lr_scale,
            )

        self.optimizer = AdamW(param_groups, weight_decay=wd)

        # ---- Scheduler -------------------------------------------------
        steps_per_epoch = len(self.train_loader) // self.accumulation_steps
        total_steps = steps_per_epoch * self.epochs
        warmup_steps: int = train_cfg.get("warmup_steps", 500)
        scheduler_type: str = train_cfg.get("scheduler", "cosine_with_warmup")

        if scheduler_type == "cosine_with_warmup_restarts":
            t0_epochs = train_cfg.get("restart_t0_epochs", 10)
            t_mult = train_cfg.get("restart_t_mult", 2)
            min_lr_ratio = train_cfg.get("min_lr_ratio", 0.01)
            self.scheduler = _cosine_warmup_restarts_schedule(
                self.optimizer, warmup_steps, steps_per_epoch,
                t0_epochs=t0_epochs, t_mult=t_mult,
                min_lr_ratio=min_lr_ratio,
            )
            logger.info(
                "LR scheduler: cosine_with_warmup_restarts "
                "(T0=%d epochs, T_mult=%d, min_lr_ratio=%.4f)",
                t0_epochs, t_mult, min_lr_ratio,
            )
        else:
            self.scheduler = _cosine_warmup_schedule(
                self.optimizer, warmup_steps, total_steps
            )

        # ---- R-Drop consistency regularization (v10) --------------------
        self.rdrop_alpha: float = train_cfg.get("rdrop_alpha", 0.0)
        if self.rdrop_alpha > 0:
            logger.info(
                "R-Drop consistency regularization enabled: alpha=%.2f",
                self.rdrop_alpha,
            )

        # ---- Mixed precision -------------------------------------------
        self.use_amp: bool = amp_cfg.get("enabled", True) and self.device.type == "cuda"
        self.scaler = GradScaler(device=self.device.type, enabled=self.use_amp)

        # ---- EMA (Exponential Moving Average) --------------------------
        ema_decay: float = train_cfg.get("ema_decay", 0.0)
        self.ema: EMA | None = None
        if ema_decay > 0:
            self.ema = EMA(self.model, decay=ema_decay)
            logger.info("EMA enabled with decay=%.5f", ema_decay)

        # ---- v8: Multi-task loss ----------------------------------------
        self.is_v8 = False
        self.multitask_loss_fn = None
        try:
            from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
            if isinstance(self.model, DeepExoMirModelV8):
                self.is_v8 = True
                from deepexomir.model.multitask_heads import MultiTaskLoss
                mt_cfg = {}
                if hasattr(self.model, "config"):
                    mt_cfg = self.model.config.get("multitask", {})
                w_con = mt_cfg.get("w_contrastive", 0.0)
                con_temp = mt_cfg.get("contrastive_temperature", 0.07)
                self.multitask_loss_fn = MultiTaskLoss(
                    w_seed=mt_cfg.get("w_seed", 0.3),
                    w_mfe=mt_cfg.get("w_mfe", 0.2),
                    w_position=mt_cfg.get("w_position", 0.2),
                    w_load_balance=mt_cfg.get("w_load_balance", 0.01),
                    w_contrastive=w_con,
                    contrastive_temperature=con_temp,
                ).to(self.device)
                logger.info(
                    "v8 multi-task loss: w_seed=%.2f, w_mfe=%.2f, "
                    "w_pos=%.2f, w_lb=%.3f, w_con=%.2f",
                    mt_cfg.get("w_seed", 0.3), mt_cfg.get("w_mfe", 0.2),
                    mt_cfg.get("w_position", 0.2),
                    mt_cfg.get("w_load_balance", 0.01), w_con,
                )
        except ImportError:
            pass

        # ---- Callbacks -------------------------------------------------
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 7),
            min_delta=es_cfg.get("min_delta", 0.001),
            mode=es_cfg.get("mode", "max"),
        )
        self.checkpoint = ModelCheckpoint(
            save_dir=ckpt_cfg.get("checkpoint_dir", "checkpoints"),
            save_top_k=ckpt_cfg.get("save_top_k", 3),
            monitor=es_cfg.get("monitor", "val_auc"),
            mode=es_cfg.get("mode", "max"),
        )

        # ---- TensorBoard -----------------------------------------------
        tb_dir = log_cfg.get("tensorboard_dir", "logs")
        self.writer = SummaryWriter(log_dir=tb_dir)

        # ---- Book-keeping ----------------------------------------------
        self.global_step: int = 0
        self.current_epoch: int = 0

    # ------------------------------------------------------------------
    # Forward helper
    # ------------------------------------------------------------------

    def _forward_batch(self, batch: dict):
        """Unpack batch dict and call model.forward with proper kwargs.

        Parameters
        ----------
        batch : dict
            Batch dictionary from the DataLoader collate function.
            Expected keys: ``mirna_seq``, ``target_seq``,
            ``base_pairing_matrix``, ``structural_features``.
            Optional keys (pre-computed embeddings): ``mirna_emb``,
            ``target_emb``, ``mirna_mask``, ``target_mask``.

        Returns
        -------
        Tensor [B, 2] (v7) or dict (v8)
            Model output. v7 returns logits tensor, v8 returns dict
            with ``logits``, ``aux_preds``, ``load_balance_loss``.
        """
        kwargs = dict(
            mirna_seqs=batch.get("mirna_seq"),
            target_seqs=batch.get("target_seq"),
            bp_matrix=batch.get("base_pairing_matrix"),
            struct_features=batch.get("structural_features"),
            mirna_emb=batch.get("mirna_emb"),
            target_emb=batch.get("target_emb"),
            mirna_mask=batch.get("mirna_mask"),
            target_mask=batch.get("target_mask"),
            mirna_pooled_emb=batch.get("mirna_pooled_emb"),
            target_pooled_emb=batch.get("target_pooled_emb"),
            mirna_pertoken_emb=batch.get("mirna_pertoken_emb"),
            mirna_pertoken_mask=batch.get("mirna_pertoken_mask"),
        )
        # v8: pass per-token target backbone embeddings
        if self.is_v8:
            kwargs["target_pertoken_emb"] = batch.get("target_pertoken_emb")
            kwargs["target_pertoken_mask"] = batch.get("target_pertoken_mask")
        return self.model(**kwargs)

    # ------------------------------------------------------------------
    # Training epoch
    # ------------------------------------------------------------------

    def train_epoch(self) -> dict[str, float]:
        """Run a single training epoch.

        Returns
        -------
        dict[str, float]
            Metrics dictionary with at least ``train_loss``.
        """
        self.model.train()
        running_loss = 0.0
        num_samples = 0

        all_labels: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        self.optimizer.zero_grad(set_to_none=True)
        total_batches = len(self.train_loader)
        running_aux_losses: dict[str, float] = {}  # v8: multi-task loss components

        for batch_idx, (batch, labels) in enumerate(self.train_loader):
            # Move data to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                batch = batch.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # ---- Mixup augmentation (v10) ----
            do_mixup = (
                self.mixup_alpha > 0
                and np.random.random() < self.mixup_prob
            )
            labels_b = None
            lam = 1.0
            if do_mixup:
                lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                lam = max(lam, 1.0 - lam)  # keep dominant sample >= 0.5
                perm = torch.randperm(labels.size(0), device=self.device)
                labels_b = labels[perm]
                # Mix all floating-point tensor features in batch
                if isinstance(batch, dict):
                    for k, v in batch.items():
                        if isinstance(v, torch.Tensor) and v.is_floating_point():
                            batch[k] = lam * v + (1.0 - lam) * v[perm]

            # Forward pass with AMP
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self._forward_batch(batch)

                # Parse output: v8 returns dict, v7 returns tensor
                if self.is_v8 and isinstance(output, dict):
                    logits = output["logits"]
                    aux_preds = output.get("aux_preds", {})
                    lb_loss = output.get("load_balance_loss")
                else:
                    logits = output
                    aux_preds = {}
                    lb_loss = None

                # Primary focal loss (with Mixup support, v10)
                if do_mixup and labels_b is not None:
                    loss_a = focal_loss(
                        logits, labels,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha,
                        label_smoothing=self.label_smoothing,
                    )
                    loss_b = focal_loss(
                        logits, labels_b,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha,
                        label_smoothing=self.label_smoothing,
                    )
                    primary_loss = lam * loss_a + (1.0 - lam) * loss_b
                else:
                    primary_loss = focal_loss(
                        logits,
                        labels,
                        gamma=self.focal_gamma,
                        alpha=self.focal_alpha,
                        label_smoothing=self.label_smoothing,
                    )

                # Multi-task loss (v8)
                loss_components = None
                if self.is_v8 and self.multitask_loss_fn is not None and not do_mixup:
                    # Skip auxiliary task losses during mixup
                    # (categorical labels can't be interpolated)
                    struct_feat = batch.get("structural_features")
                    seed_labels = None
                    mfe_labels = None
                    if struct_feat is not None:
                        seed_labels = struct_feat[:, 5].long()  # seed_match_type
                        mfe_labels = struct_feat[:, 0]  # duplex_mfe
                    loss, loss_components = self.multitask_loss_fn(
                        primary_loss, aux_preds,
                        seed_type_labels=seed_labels,
                        mfe_labels=mfe_labels,
                        load_balance_loss=lb_loss,
                        labels=labels,
                    )
                elif do_mixup and lb_loss is not None:
                    # During mixup: only add load-balance loss
                    loss = primary_loss + 0.01 * lb_loss
                else:
                    loss = primary_loss

                # R-Drop consistency regularization (v10)
                if self.rdrop_alpha > 0 and not do_mixup:
                    output2 = self._forward_batch(batch)
                    if self.is_v8 and isinstance(output2, dict):
                        logits2 = output2["logits"]
                    else:
                        logits2 = output2
                    # Symmetric KL divergence
                    p_log = F.log_softmax(logits, dim=-1)
                    q_log = F.log_softmax(logits2, dim=-1)
                    p_prob = F.softmax(logits.detach(), dim=-1)
                    q_prob = F.softmax(logits2.detach(), dim=-1)
                    kl_pq = F.kl_div(p_log, q_prob, reduction="batchmean")
                    kl_qp = F.kl_div(q_log, p_prob, reduction="batchmean")
                    rdrop_loss = (kl_pq + kl_qp) / 2.0
                    loss = loss + self.rdrop_alpha * rdrop_loss

                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Accumulate metrics (use unscaled loss)
            batch_size = labels.size(0)
            running_loss += loss.item() * self.accumulation_steps * batch_size
            num_samples += batch_size

            # Track auxiliary loss components (v8)
            if loss_components is not None:
                for k, v in loss_components.items():
                    running_aux_losses[k] = running_aux_losses.get(k, 0.0) + v.item() * batch_size

            with torch.no_grad():
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs >= 0.5).astype(np.int64)
                all_labels.append(labels.cpu().numpy())
                all_preds.append(preds)
                all_probs.append(probs)

            # Optimiser step every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Unscale before gradient clipping
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # EMA update after each optimiser step
                if self.ema is not None:
                    self.ema.update()

                # TensorBoard: step-level loss + console progress
                if self.global_step % self.log_every_n_steps == 0:
                    step_loss = loss.item() * self.accumulation_steps
                    lr = self.scheduler.get_last_lr()[0]
                    self.writer.add_scalar(
                        "train/step_loss", step_loss, self.global_step
                    )
                    self.writer.add_scalar(
                        "train/lr", lr, self.global_step,
                    )
                    pct = 100.0 * (batch_idx + 1) / total_batches
                    logger.info(
                        "  [%5.1f%%] batch %d/%d | loss=%.4f | lr=%.2e",
                        pct, batch_idx + 1, total_batches,
                        step_loss, lr,
                    )

        # Handle leftover gradients when len(loader) is not divisible
        remainder = len(self.train_loader) % self.accumulation_steps
        if remainder != 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
            if self.ema is not None:
                self.ema.update()

        # Epoch-level metrics
        avg_loss = running_loss / max(num_samples, 1)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
        metrics["train_loss"] = avg_loss

        # TensorBoard: epoch-level
        self.writer.add_scalar("train/epoch_loss", avg_loss, self.current_epoch)
        self.writer.add_scalar("train/auc_roc", metrics["auc_roc"], self.current_epoch)
        self.writer.add_scalar("train/f1", metrics["f1"], self.current_epoch)

        # TensorBoard: epoch-level auxiliary losses (v8)
        if running_aux_losses:
            for k, v in running_aux_losses.items():
                avg_v = v / max(num_samples, 1)
                self.writer.add_scalar(
                    f"train/{k}_loss", avg_v, self.current_epoch,
                )
                metrics[f"train_{k}_loss"] = avg_v

        return metrics

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate the model on the validation set.

        When EMA is enabled, validation uses the EMA shadow parameters
        for evaluation, then restores the original training parameters.

        Returns
        -------
        dict[str, float]
            Metrics dictionary including ``val_loss``, ``val_auc``, etc.
        """
        # Apply EMA shadow params for validation
        if self.ema is not None:
            self.ema.apply_shadow()

        self.model.eval()
        running_loss = 0.0
        num_samples = 0

        all_labels: list[np.ndarray] = []
        all_preds: list[np.ndarray] = []
        all_probs: list[np.ndarray] = []

        for batch, labels in self.val_loader:
            if isinstance(batch, dict):
                batch = {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            else:
                batch = batch.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                output = self._forward_batch(batch)

                # Parse output: v8 returns dict, v7 returns tensor
                if self.is_v8 and isinstance(output, dict):
                    logits = output["logits"]
                else:
                    logits = output

                loss = focal_loss(
                    logits,
                    labels,
                    gamma=self.focal_gamma,
                    alpha=self.focal_alpha,
                    label_smoothing=self.label_smoothing,
                )

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            num_samples += batch_size

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(np.int64)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

        # Restore original params after EMA validation
        if self.ema is not None:
            self.ema.restore()

        avg_loss = running_loss / max(num_samples, 1)
        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        y_prob = np.concatenate(all_probs)
        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        # Prefix with "val_"
        val_metrics: dict[str, float] = {"val_loss": avg_loss}
        for k, v in metrics.items():
            val_metrics[f"val_{k}"] = v

        # TensorBoard
        self.writer.add_scalar("val/loss", avg_loss, self.current_epoch)
        self.writer.add_scalar("val/auc_roc", metrics["auc_roc"], self.current_epoch)
        self.writer.add_scalar("val/auc_pr", metrics["auc_pr"], self.current_epoch)
        self.writer.add_scalar("val/f1", metrics["f1"], self.current_epoch)
        self.writer.add_scalar("val/mcc", metrics["mcc"], self.current_epoch)

        return val_metrics

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(self) -> dict[str, Any]:
        """Run the full training loop with early stopping.

        Returns
        -------
        dict[str, Any]
            Dictionary containing:
            - ``best_val_metrics``: best validation metrics observed.
            - ``best_epoch``: epoch at which best val_auc was achieved.
            - ``best_checkpoint``: path to the best checkpoint file.
            - ``total_epochs``: number of epochs actually trained.
            - ``stopped_early``: whether early stopping triggered.
        """
        logger.info(
            "Starting training for up to %d epochs on %s (AMP=%s, EMA=%s)",
            self.epochs,
            self.device,
            self.use_amp,
            f"decay={self.ema.decay}" if self.ema else "off",
        )

        best_val_metrics: dict[str, float] = {}
        best_epoch: int = 0
        stopped_early = False
        training_start = time.time()
        epoch_times: list[float] = []

        for epoch in range(self.epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # --- Train ---
            train_metrics = self.train_epoch()

            # --- Validate (uses EMA params if enabled) ---
            val_metrics = self.validate()
            epoch_time = time.time() - epoch_start
            epoch_times.append(epoch_time)

            # --- ETA calculation ---
            avg_epoch_time = sum(epoch_times) / len(epoch_times)
            remaining_epochs = self.epochs - (epoch + 1)
            eta_seconds = remaining_epochs * avg_epoch_time
            eta_min = eta_seconds / 60
            elapsed_total = time.time() - training_start

            # Combined logging with ETA
            val_auc = val_metrics.get("val_auc_roc", 0.0)
            logger.info(
                "Epoch %3d/%d | train_loss=%.4f | val_loss=%.4f | "
                "val_auc=%.4f | val_f1=%.4f | time=%.1fs | "
                "ETA=%.1fmin (elapsed=%.1fmin)",
                epoch + 1,
                self.epochs,
                train_metrics["train_loss"],
                val_metrics["val_loss"],
                val_auc,
                val_metrics.get("val_f1", 0.0),
                epoch_time,
                eta_min,
                elapsed_total / 60,
            )

            # --- Checkpointing (save EMA params if enabled) ---
            if self.ema is not None:
                self.ema.apply_shadow()
            self.checkpoint(val_auc, self.model, epoch)
            if self.ema is not None:
                self.ema.restore()

            # Track best
            if not best_val_metrics or val_auc > best_val_metrics.get("val_auc_roc", 0.0):
                best_val_metrics = val_metrics.copy()
                best_epoch = epoch

            # --- Early stopping ---
            if self.early_stopping(val_auc):
                logger.info(
                    "Early stopping triggered at epoch %d (patience=%d)",
                    epoch + 1,
                    self.early_stopping.patience,
                )
                stopped_early = True
                break

        self.writer.close()

        result = {
            "best_val_metrics": best_val_metrics,
            "best_epoch": best_epoch,
            "best_checkpoint": self.checkpoint.best_checkpoint,
            "total_epochs": self.current_epoch + 1,
            "stopped_early": stopped_early,
        }

        logger.info("Training complete. Best epoch: %d", best_epoch + 1)
        logger.info("Best val metrics:\n%s", Evaluator.format_metrics(
            {k.replace("val_", ""): v for k, v in best_val_metrics.items() if k != "val_loss"}
        ))

        return result
