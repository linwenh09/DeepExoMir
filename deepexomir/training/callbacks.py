"""Training callbacks for early stopping and model checkpointing.

Provides lightweight callback classes that decouple stopping / saving
logic from the main training loop in :mod:`deepexomir.training.trainer`.
"""

from __future__ import annotations

import heapq
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement after which training is stopped.
    min_delta : float
        Minimum change in the monitored metric to qualify as an improvement.
    mode : ``"max"`` or ``"min"``
        Whether the monitored metric should be maximised or minimised.
    """

    def __init__(
        self,
        patience: int = 7,
        min_delta: float = 0.001,
        mode: Literal["max", "min"] = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

        self._best: float | None = None
        self._counter: int = 0
        self._is_better = self._max_better if mode == "max" else self._min_better

    # ------------------------------------------------------------------
    # Comparison helpers
    # ------------------------------------------------------------------
    def _max_better(self, current: float, best: float) -> bool:
        return current > best + self.min_delta

    def _min_better(self, current: float, best: float) -> bool:
        return current < best - self.min_delta

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(self, metric: float) -> bool:
        """Check whether training should stop.

        Parameters
        ----------
        metric : float
            Current value of the monitored metric.

        Returns
        -------
        bool
            ``True`` if training should stop (patience exhausted).
        """
        if self._best is None or self._is_better(metric, self._best):
            self._best = metric
            self._counter = 0
            return False

        self._counter += 1
        logger.info(
            "EarlyStopping: no improvement for %d/%d epochs (best=%.5f, current=%.5f)",
            self._counter,
            self.patience,
            self._best,
            metric,
        )
        return self._counter >= self.patience

    def reset(self) -> None:
        """Reset internal state."""
        self._best = None
        self._counter = 0

    @property
    def best(self) -> float | None:
        """Best metric value observed so far."""
        return self._best

    @property
    def counter(self) -> int:
        """Number of epochs since last improvement."""
        return self._counter


@dataclass(order=True)
class _CheckpointEntry:
    """Priority-queue entry for checkpoint management.

    For ``mode='max'``, we want the *smallest* metric to be evicted first,
    so we store the metric directly.  For ``mode='min'``, we negate the metric
    so the heap still evicts the "worst" checkpoint first.
    """

    priority: float
    epoch: int = field(compare=False)
    path: Path = field(compare=False)


class ModelCheckpoint:
    """Save the top-*k* model checkpoints ranked by a monitored metric.

    Parameters
    ----------
    save_dir : str or Path
        Directory where checkpoint files are written.
    save_top_k : int
        Maximum number of checkpoint files to keep on disk.
    monitor : str
        Name of the metric to monitor (used only for logging / file names).
    mode : ``"max"`` or ``"min"``
        Whether higher or lower metric values are better.
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_top_k: int = 3,
        monitor: str = "val_auc",
        mode: Literal["max", "min"] = "max",
    ) -> None:
        if mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got '{mode}'")

        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_top_k = save_top_k
        self.monitor = monitor
        self.mode = mode

        # Min-heap of (_CheckpointEntry).  For mode='max', priority = metric
        # (so the smallest metric is popped first).  For mode='min',
        # priority = -metric (so the largest metric -- worst -- is popped).
        self._heap: list[_CheckpointEntry] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _metric_to_priority(self, metric: float) -> float:
        return metric if self.mode == "max" else -metric

    def _save_checkpoint(
        self,
        model: nn.Module,
        epoch: int,
        metric: float,
    ) -> Path:
        """Serialize model state dict to disk."""
        filename = f"checkpoint_epoch{epoch:03d}_{self.monitor}_{metric:.4f}.pt"
        path = self.save_dir / filename
        torch.save(
            {
                "epoch": epoch,
                self.monitor: metric,
                "model_state_dict": model.state_dict(),
            },
            path,
        )
        logger.info("Saved checkpoint: %s", path)
        return path

    def _remove_checkpoint(self, entry: _CheckpointEntry) -> None:
        """Delete a checkpoint file from disk."""
        if entry.path.exists():
            entry.path.unlink()
            logger.info(
                "Removed checkpoint (epoch %d, %s=%.4f): %s",
                entry.epoch,
                self.monitor,
                entry.priority if self.mode == "max" else -entry.priority,
                entry.path,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def __call__(
        self,
        metric: float,
        model: nn.Module,
        epoch: int,
    ) -> None:
        """Conditionally save a checkpoint if the metric is in the top-*k*.

        Parameters
        ----------
        metric : float
            Current value of the monitored metric.
        model : nn.Module
            Model whose ``state_dict`` will be saved.
        epoch : int
            Current epoch number (used in the checkpoint filename).
        """
        priority = self._metric_to_priority(metric)

        if len(self._heap) < self.save_top_k:
            path = self._save_checkpoint(model, epoch, metric)
            heapq.heappush(
                self._heap,
                _CheckpointEntry(priority=priority, epoch=epoch, path=path),
            )
        elif priority > self._heap[0].priority:
            # New checkpoint is better than the current worst in the heap
            worst = heapq.heapreplace(
                self._heap,
                _CheckpointEntry(priority=priority, epoch=epoch, path=self._save_checkpoint(model, epoch, metric)),
            )
            self._remove_checkpoint(worst)
        else:
            logger.debug(
                "Checkpoint not saved: %s=%.4f is not in top-%d",
                self.monitor,
                metric,
                self.save_top_k,
            )

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best checkpoint observed so far."""
        if not self._heap:
            return None
        best = max(self._heap, key=lambda e: e.priority)
        return best.path

    @property
    def best_metric(self) -> float | None:
        """Best metric value observed so far."""
        if not self._heap:
            return None
        best = max(self._heap, key=lambda e: e.priority)
        return best.priority if self.mode == "max" else -best.priority
