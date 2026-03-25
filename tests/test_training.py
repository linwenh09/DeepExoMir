"""Tests for training pipeline.

Tests the Evaluator, EarlyStopping, and ModelCheckpoint components
independently of the full training loop.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from deepexomir.training.callbacks import EarlyStopping, ModelCheckpoint
from deepexomir.training.evaluator import Evaluator


# ====================================================================
# test_evaluator_metrics
# ====================================================================


class TestEvaluatorMetrics:
    """Verify metrics computation on known data."""

    def test_perfect_predictions(self):
        """All correct predictions should yield AUC=1.0, ACC=1.0."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.95])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 1.0
        assert metrics["f1"] == 1.0
        assert metrics["mcc"] == 1.0
        assert metrics["sensitivity"] == 1.0
        assert metrics["specificity"] == 1.0

    def test_all_wrong_predictions(self):
        """All wrong predictions should have low accuracy."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        y_prob = np.array([0.9, 0.8, 0.1, 0.2])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        assert metrics["accuracy"] == 0.0
        assert metrics["mcc"] == -1.0

    def test_mixed_predictions(self):
        """Mixed predictions should have intermediate metrics."""
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1])
        y_prob = np.array([0.2, 0.6, 0.8, 0.3, 0.1, 0.9, 0.2, 0.7])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        # Should have intermediate values
        assert 0.0 < metrics["accuracy"] < 1.0
        assert 0.0 < metrics["f1"] < 1.0
        assert "auc_roc" in metrics
        assert "auc_pr" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "precision" in metrics
        assert "recall" in metrics

    def test_metric_keys_complete(self):
        """All expected metric keys should be present."""
        y_true = np.array([0, 1])
        y_pred = np.array([0, 1])
        y_prob = np.array([0.1, 0.9])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        expected_keys = {
            "auc_roc", "auc_pr", "accuracy", "f1", "mcc",
            "sensitivity", "specificity", "precision", "recall",
        }
        assert set(metrics.keys()) == expected_keys

    def test_single_class_handling(self):
        """Should handle degenerate case where only one class is present."""
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.7, 0.95])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)

        # AUC is undefined with one class, should be 0.0
        assert metrics["auc_roc"] == 0.0
        assert metrics["accuracy"] == 1.0

    def test_format_metrics(self):
        """format_metrics should return a formatted string."""
        metrics = {
            "auc_roc": 0.95,
            "accuracy": 0.90,
            "f1": 0.88,
        }
        formatted = Evaluator.format_metrics(metrics)
        assert isinstance(formatted, str)
        assert "AUC-ROC" in formatted
        assert "Accuracy" in formatted


# ====================================================================
# test_evaluator_perfect_predictions
# ====================================================================


class TestEvaluatorPerfectPredictions:
    """Specifically test that perfect predictions yield ideal scores."""

    def test_auc_is_one(self):
        """Perfect separation should give AUC-ROC = 1.0."""
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert metrics["auc_roc"] == 1.0

    def test_accuracy_is_one(self):
        """Perfect predictions should give accuracy = 1.0."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])
        y_prob = np.array([0.1, 0.9, 0.2, 0.8])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert metrics["accuracy"] == 1.0

    def test_mcc_is_one(self):
        """Perfect predictions should give MCC = 1.0."""
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8])

        metrics = Evaluator.compute_metrics(y_true, y_pred, y_prob)
        assert abs(metrics["mcc"] - 1.0) < 1e-6


# ====================================================================
# test_early_stopping_triggers
# ====================================================================


class TestEarlyStoppingTriggers:
    """Verify the patience mechanism of EarlyStopping."""

    def test_triggers_after_patience(self):
        """Should return True after patience epochs with no improvement."""
        es = EarlyStopping(patience=3, min_delta=0.001, mode="max")

        # First call sets baseline
        assert es(0.80) is False  # best=0.80
        # No improvement
        assert es(0.79) is False  # counter=1
        assert es(0.78) is False  # counter=2
        # Patience exhausted at counter=3
        assert es(0.77) is True  # counter=3 -> triggers

    def test_patience_one(self):
        """With patience=1, should trigger after one non-improving epoch."""
        es = EarlyStopping(patience=1, mode="max")

        assert es(0.90) is False  # best=0.90
        assert es(0.89) is True   # counter=1 -> triggers

    def test_min_delta_respected(self):
        """Improvement below min_delta should not count."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="max")

        assert es(0.80) is False  # best=0.80
        # Improvement of 0.005 < min_delta=0.01
        assert es(0.805) is False  # counter=1
        assert es(0.806) is True   # counter=2 -> triggers

    def test_min_mode(self):
        """mode='min' should trigger when metric stops decreasing."""
        es = EarlyStopping(patience=2, min_delta=0.001, mode="min")

        assert es(0.50) is False  # best=0.50
        assert es(0.60) is False  # counter=1 (worse)
        assert es(0.55) is True   # counter=2 -> triggers

    def test_reset(self):
        """reset() should clear internal state."""
        es = EarlyStopping(patience=2, mode="max")
        es(0.80)
        es(0.70)

        assert es.counter == 1
        es.reset()
        assert es.counter == 0
        assert es.best is None

    def test_best_property(self):
        """best should track the best observed metric value."""
        es = EarlyStopping(patience=5, mode="max")
        es(0.70)
        es(0.80)
        es(0.75)

        assert es.best == 0.80


# ====================================================================
# test_early_stopping_no_trigger
# ====================================================================


class TestEarlyStoppingNoTrigger:
    """Verify that early stopping does not trigger when metric keeps improving."""

    def test_continuous_improvement(self):
        """Monotonically improving metric should never trigger."""
        es = EarlyStopping(patience=3, mode="max")

        for val in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
            assert es(val) is False

        assert es.counter == 0
        assert es.best == 0.95

    def test_improvement_resets_counter(self):
        """An improving value should reset the counter."""
        es = EarlyStopping(patience=3, mode="max")

        assert es(0.80) is False  # best=0.80
        assert es(0.79) is False  # counter=1
        assert es(0.78) is False  # counter=2
        # Now improve
        assert es(0.85) is False  # counter=0, best=0.85
        assert es.counter == 0
        assert es.best == 0.85


# ====================================================================
# test_model_checkpoint_saves
# ====================================================================


class TestModelCheckpointSaves:
    """Verify that checkpoint files are created correctly."""

    def test_checkpoint_files_created(self):
        """Checkpoint files should be created in the save directory."""
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                save_dir=tmpdir,
                save_top_k=3,
                monitor="val_auc",
                mode="max",
            )

            # Create a simple model
            model = nn.Linear(10, 2)

            # Save some checkpoints
            ckpt(0.80, model, epoch=0)
            ckpt(0.85, model, epoch=1)
            ckpt(0.82, model, epoch=2)

            # Check files exist
            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(checkpoint_files) == 3

    def test_top_k_eviction(self):
        """Only top-k checkpoints should be kept."""
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                save_dir=tmpdir,
                save_top_k=2,
                monitor="val_auc",
                mode="max",
            )

            model = nn.Linear(10, 2)

            ckpt(0.70, model, epoch=0)
            ckpt(0.80, model, epoch=1)
            ckpt(0.90, model, epoch=2)  # Should evict epoch 0 (0.70)

            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(checkpoint_files) == 2

            # The worst checkpoint (0.70) should have been removed
            filenames = [f.name for f in checkpoint_files]
            assert not any("epoch000" in fn for fn in filenames)

    def test_best_checkpoint_property(self):
        """best_checkpoint should return path to the best checkpoint."""
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                save_dir=tmpdir,
                save_top_k=3,
                monitor="val_auc",
                mode="max",
            )

            model = nn.Linear(10, 2)

            ckpt(0.80, model, epoch=0)
            ckpt(0.90, model, epoch=1)
            ckpt(0.85, model, epoch=2)

            best = ckpt.best_checkpoint
            assert best is not None
            assert best.exists()
            assert "epoch001" in best.name  # epoch 1 had highest metric

    def test_best_metric_property(self):
        """best_metric should track the best monitored metric value."""
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                save_dir=tmpdir,
                save_top_k=3,
                monitor="val_auc",
                mode="max",
            )

            model = nn.Linear(10, 2)

            ckpt(0.80, model, epoch=0)
            ckpt(0.90, model, epoch=1)

            assert abs(ckpt.best_metric - 0.90) < 1e-6

    def test_checkpoint_contains_state_dict(self):
        """Checkpoint files should contain the model state dict."""
        import torch
        import torch.nn as nn

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(
                save_dir=tmpdir,
                save_top_k=1,
                monitor="val_auc",
                mode="max",
            )

            model = nn.Linear(10, 2)
            ckpt(0.90, model, epoch=0)

            checkpoint_files = list(Path(tmpdir).glob("checkpoint_*.pt"))
            assert len(checkpoint_files) == 1

            data = torch.load(checkpoint_files[0], map_location="cpu", weights_only=False)
            assert "model_state_dict" in data
            assert "epoch" in data
            assert "val_auc" in data

    def test_no_checkpoints_initially(self):
        """best_checkpoint should be None before any saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = ModelCheckpoint(save_dir=tmpdir)
            assert ckpt.best_checkpoint is None
            assert ckpt.best_metric is None
