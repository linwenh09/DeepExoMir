"""Tests for model architecture.

All tests use pre-computed embeddings to avoid downloading the RNA
backbone model. The backbone is either skipped (load_backbone=False)
or mocked where needed.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from deepexomir.model.classifier import ClassificationHead
from deepexomir.model.cross_attention import CrossAttentionBlock, CrossAttentionEncoder
from deepexomir.model.losses import FocalLoss
from deepexomir.model.structural_encoder import BasePairingCNN, StructuralMLP


# ====================================================================
# Fixtures
# ====================================================================


@pytest.fixture
def default_model_config():
    """Return a model config dict with default values (no backbone loading)."""
    return {
        "backbone": {
            "name": "multimolecule/rinalmo-giga",
            "embed_dim": 1280,
            "freeze": True,
        },
        "model": {
            "d_model": 256,
            "n_heads": 8,
            "d_ff": 1024,
            "n_cross_layers": 4,
            "dropout": 0.2,
            "attention_dropout": 0.1,
            "max_mirna_len": 30,
            "max_target_len": 40,
        },
        "structural": {
            "bp_cnn_out": 128,
            "struct_mlp_in": 6,
            "struct_mlp_out": 64,
        },
        "classifier": {
            "hidden_dims": [256, 128],
            "n_classes": 2,
            "platt_scaling": True,
        },
    }


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def d_model():
    return 256


@pytest.fixture
def backbone_dim():
    return 1280


# ====================================================================
# test_cross_attention_block_shapes
# ====================================================================


class TestCrossAttentionBlock:
    """Verify input/output shapes of a single CrossAttentionBlock."""

    def test_output_shapes(self, batch_size, d_model):
        """Output shapes should match input shapes."""
        block = CrossAttentionBlock(d_model=d_model, n_heads=8, d_ff=1024, dropout=0.1)
        mirna = torch.randn(batch_size, 30, d_model)
        target = torch.randn(batch_size, 40, d_model)

        m_out, t_out = block(mirna, target)
        assert m_out.shape == (batch_size, 30, d_model)
        assert t_out.shape == (batch_size, 40, d_model)

    def test_with_masks(self, batch_size, d_model):
        """Block should handle padding masks correctly."""
        block = CrossAttentionBlock(d_model=d_model, n_heads=8)
        mirna = torch.randn(batch_size, 30, d_model)
        target = torch.randn(batch_size, 40, d_model)

        # Create masks (True = padding position)
        mirna_mask = torch.zeros(batch_size, 30, dtype=torch.bool)
        mirna_mask[:, 20:] = True
        target_mask = torch.zeros(batch_size, 40, dtype=torch.bool)
        target_mask[:, 30:] = True

        m_out, t_out = block(mirna, target, mirna_mask, target_mask)
        assert m_out.shape == (batch_size, 30, d_model)
        assert t_out.shape == (batch_size, 40, d_model)

    def test_gradients_flow(self, d_model):
        """Gradients should flow through the block."""
        block = CrossAttentionBlock(d_model=d_model, n_heads=8)
        mirna = torch.randn(2, 10, d_model, requires_grad=True)
        target = torch.randn(2, 15, d_model, requires_grad=True)

        m_out, t_out = block(mirna, target)
        loss = m_out.sum() + t_out.sum()
        loss.backward()

        assert mirna.grad is not None
        assert target.grad is not None


# ====================================================================
# test_cross_attention_encoder_shapes
# ====================================================================


class TestCrossAttentionEncoder:
    """Verify stacked cross-attention blocks."""

    def test_output_shapes(self, batch_size, d_model):
        """Encoder output should match input shape."""
        encoder = CrossAttentionEncoder(
            n_layers=4, d_model=d_model, n_heads=8, d_ff=1024, dropout=0.1
        )
        mirna = torch.randn(batch_size, 30, d_model)
        target = torch.randn(batch_size, 40, d_model)

        m_out, t_out = encoder(mirna, target)
        assert m_out.shape == (batch_size, 30, d_model)
        assert t_out.shape == (batch_size, 40, d_model)

    def test_single_layer(self, batch_size, d_model):
        """Encoder with a single layer should still work."""
        encoder = CrossAttentionEncoder(n_layers=1, d_model=d_model, n_heads=8)
        mirna = torch.randn(batch_size, 5, d_model)
        target = torch.randn(batch_size, 8, d_model)

        m_out, t_out = encoder(mirna, target)
        assert m_out.shape == (batch_size, 5, d_model)
        assert t_out.shape == (batch_size, 8, d_model)

    def test_has_final_layer_norms(self, d_model):
        """Encoder should contain final LayerNorm layers."""
        encoder = CrossAttentionEncoder(n_layers=2, d_model=d_model, n_heads=8)
        assert hasattr(encoder, "mirna_final_norm")
        assert hasattr(encoder, "target_final_norm")
        assert isinstance(encoder.mirna_final_norm, nn.LayerNorm)
        assert isinstance(encoder.target_final_norm, nn.LayerNorm)


# ====================================================================
# test_bp_cnn_shapes
# ====================================================================


class TestBasePairingCNN:
    """Verify BasePairingCNN: [B, 1, 30, 40] -> [B, 128]."""

    def test_output_shape(self, batch_size):
        """Default output should be [B, 128]."""
        cnn = BasePairingCNN(out_dim=128)
        bp_matrix = torch.randn(batch_size, 1, 30, 40)
        output = cnn(bp_matrix)
        assert output.shape == (batch_size, 128)

    def test_custom_output_dim(self, batch_size):
        """Custom out_dim should be respected."""
        cnn = BasePairingCNN(out_dim=64)
        bp_matrix = torch.randn(batch_size, 1, 30, 40)
        output = cnn(bp_matrix)
        assert output.shape == (batch_size, 64)

    def test_different_spatial_sizes(self, batch_size):
        """CNN should handle different spatial dimensions via adaptive pooling."""
        cnn = BasePairingCNN(out_dim=128)
        # Non-standard spatial dimensions
        bp_matrix = torch.randn(batch_size, 1, 20, 50)
        output = cnn(bp_matrix)
        assert output.shape == (batch_size, 128)

    def test_gradients_flow(self):
        """Gradients should flow through the CNN."""
        cnn = BasePairingCNN(out_dim=128)
        bp_matrix = torch.randn(2, 1, 30, 40, requires_grad=True)
        output = cnn(bp_matrix)
        output.sum().backward()
        assert bp_matrix.grad is not None


# ====================================================================
# test_structural_mlp_shapes
# ====================================================================


class TestStructuralMLP:
    """Verify StructuralMLP: [B, 6] -> [B, 64]."""

    def test_output_shape(self, batch_size):
        """Default output should be [B, 64]."""
        mlp = StructuralMLP(in_dim=6, out_dim=64)
        features = torch.randn(batch_size, 6)
        output = mlp(features)
        assert output.shape == (batch_size, 64)

    def test_custom_dims(self, batch_size):
        """Custom input and output dimensions should work."""
        mlp = StructuralMLP(in_dim=10, out_dim=32)
        features = torch.randn(batch_size, 10)
        output = mlp(features)
        assert output.shape == (batch_size, 32)

    def test_all_outputs_non_negative(self, batch_size):
        """Output should be non-negative due to final ReLU."""
        mlp = StructuralMLP(in_dim=6, out_dim=64)
        features = torch.randn(batch_size, 6)
        output = mlp(features)
        assert (output >= 0).all()


# ====================================================================
# test_classification_head_shapes
# ====================================================================


class TestClassificationHead:
    """Verify ClassificationHead: [B, 704] -> [B, 2]."""

    def test_output_shape(self, batch_size):
        """Default output should be [B, 2]."""
        head = ClassificationHead(in_dim=704, hidden_dims=[256, 128], n_classes=2)
        x = torch.randn(batch_size, 704)
        output = head(x)
        assert output.shape == (batch_size, 2)

    def test_predict_proba_shape(self, batch_size):
        """predict_proba should return probabilities summing to 1."""
        head = ClassificationHead(in_dim=704)
        x = torch.randn(batch_size, 704)
        probs = head.predict_proba(x)
        assert probs.shape == (batch_size, 2)
        # Probabilities should sum to ~1.0 for each sample
        row_sums = probs.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(batch_size), atol=1e-5)

    def test_temperature_scaling(self, batch_size):
        """Temperature parameter should exist when platt_scaling=True."""
        head = ClassificationHead(in_dim=704, platt_scaling=True)
        assert hasattr(head, "temperature")
        assert isinstance(head.temperature, nn.Parameter)

    def test_set_temperature(self):
        """set_temperature should update the temperature parameter."""
        head = ClassificationHead(in_dim=704, platt_scaling=True)
        head.set_temperature(2.0)
        assert abs(head.temperature.item() - 2.0) < 1e-6


# ====================================================================
# test_focal_loss_computation
# ====================================================================


class TestFocalLoss:
    """Verify FocalLoss is scalar and gradients flow."""

    def test_loss_is_scalar(self, batch_size):
        """Loss should be a scalar tensor."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75, label_smoothing=0.05)
        logits = torch.randn(batch_size, 2, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,))
        loss = loss_fn(logits, targets)
        assert loss.dim() == 0  # scalar
        assert loss.item() > 0  # should be positive

    def test_gradients_flow(self, batch_size):
        """Gradients should flow through the loss."""
        loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
        logits = torch.randn(batch_size, 2, requires_grad=True)
        targets = torch.randint(0, 2, (batch_size,))
        loss = loss_fn(logits, targets)
        loss.backward()
        assert logits.grad is not None

    def test_gamma_zero_reduces_to_ce(self, batch_size):
        """With gamma=0 and no label smoothing, should approximate weighted CE."""
        loss_fn = FocalLoss(gamma=0.0, alpha=None, label_smoothing=0.0)
        logits = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))

        focal_loss_val = loss_fn(logits, targets)
        ce_loss_val = torch.nn.functional.cross_entropy(logits, targets)

        # Should be close (both are standard cross-entropy)
        assert abs(focal_loss_val.item() - ce_loss_val.item()) < 0.1

    def test_no_reduction(self, batch_size):
        """reduction='none' should return per-sample losses."""
        loss_fn = FocalLoss(gamma=2.0, reduction="none")
        logits = torch.randn(batch_size, 2)
        targets = torch.randint(0, 2, (batch_size,))
        loss = loss_fn(logits, targets)
        assert loss.shape == (batch_size,)


# ====================================================================
# test_full_model_forward_precomputed
# ====================================================================


class TestFullModelForward:
    """Full model forward pass with pre-computed embeddings (no backbone)."""

    def test_forward_with_precomputed_embeddings(self, default_model_config, batch_size, backbone_dim):
        """Model should accept pre-computed embeddings and produce logits."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)
        model.eval()

        # Create synthetic pre-computed embeddings
        mirna_emb = torch.randn(batch_size, 30, backbone_dim)
        target_emb = torch.randn(batch_size, 40, backbone_dim)
        bp_matrix = torch.randn(batch_size, 1, 30, 40)
        struct_features = torch.randn(batch_size, 6)

        with torch.no_grad():
            logits = model(
                mirna_emb=mirna_emb,
                target_emb=target_emb,
                bp_matrix=bp_matrix,
                struct_features=struct_features,
            )

        assert logits.shape == (batch_size, 2)

    def test_forward_without_structural_features_raises(self, default_model_config, batch_size, backbone_dim):
        """Model should raise when structural features are omitted (dimension mismatch).

        The classifier head expects 704-dim input (512 from cross-attention + 128
        from BP-CNN + 64 from structural MLP). When bp_matrix and struct_features
        are both None, only 512 dims are provided, causing a RuntimeError.
        """
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)
        model.eval()

        mirna_emb = torch.randn(batch_size, 30, backbone_dim)
        target_emb = torch.randn(batch_size, 40, backbone_dim)

        with pytest.raises(RuntimeError):
            with torch.no_grad():
                model(
                    mirna_emb=mirna_emb,
                    target_emb=target_emb,
                )

    def test_raises_without_sequences_or_embeddings(self, default_model_config):
        """Model should raise ValueError when neither sequences nor embeddings are given."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)
        model.eval()

        with pytest.raises(ValueError, match="Must provide either"):
            model()

    def test_raises_without_backbone_for_raw_sequences(self, default_model_config):
        """Model should raise RuntimeError when raw sequences are given without backbone."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)

        with pytest.raises(RuntimeError, match="RNABackbone is not loaded"):
            model(mirna_seqs=["AUGCUA"], target_seqs=["GCUAAU"])


# ====================================================================
# test_model_predict
# ====================================================================


class TestModelPredict:
    """Verify predict returns dict with expected keys."""

    def test_predict_returns_expected_keys(self, default_model_config, batch_size, backbone_dim):
        """predict() should return a dict with logits, probabilities, predictions, confidence."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)

        mirna_emb = torch.randn(batch_size, 30, backbone_dim)
        target_emb = torch.randn(batch_size, 40, backbone_dim)
        bp_matrix = torch.randn(batch_size, 1, 30, 40)
        struct_features = torch.randn(batch_size, 6)

        result = model.predict(
            mirna_emb=mirna_emb,
            target_emb=target_emb,
            bp_matrix=bp_matrix,
            struct_features=struct_features,
        )

        assert "logits" in result
        assert "probabilities" in result
        assert "predictions" in result
        assert "confidence" in result

        assert result["logits"].shape == (batch_size, 2)
        assert result["probabilities"].shape == (batch_size, 2)
        assert result["predictions"].shape == (batch_size,)
        assert result["confidence"].shape == (batch_size,)

    def test_predictions_are_binary(self, default_model_config, backbone_dim):
        """Predictions should be 0 or 1."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)

        result = model.predict(
            mirna_emb=torch.randn(8, 30, backbone_dim),
            target_emb=torch.randn(8, 40, backbone_dim),
            bp_matrix=torch.randn(8, 1, 30, 40),
            struct_features=torch.randn(8, 6),
        )

        preds = result["predictions"]
        assert ((preds == 0) | (preds == 1)).all()

    def test_probabilities_sum_to_one(self, default_model_config, backbone_dim):
        """Probabilities for each sample should sum to 1.0."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)

        result = model.predict(
            mirna_emb=torch.randn(4, 30, backbone_dim),
            target_emb=torch.randn(4, 40, backbone_dim),
            bp_matrix=torch.randn(4, 1, 30, 40),
            struct_features=torch.randn(4, 6),
        )

        prob_sums = result["probabilities"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(4), atol=1e-5)


# ====================================================================
# test_trainable_params
# ====================================================================


class TestTrainableParams:
    """Verify backbone is frozen and other parameters require grad."""

    def test_non_backbone_params_require_grad(self, default_model_config):
        """All non-backbone parameters should require gradients."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)

        # All parameters in this model should require grad (no backbone loaded)
        trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
        total_count = sum(1 for p in model.parameters())

        assert trainable_count > 0
        assert trainable_count == total_count  # all trainable since no backbone

    def test_trainable_parameter_count_is_positive(self, default_model_config):
        """trainable_parameters() should return a positive number."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)
        assert model.trainable_parameters() > 0

    def test_total_equals_trainable_without_backbone(self, default_model_config):
        """Without backbone, total should equal trainable params."""
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        model = DeepExoMirModel(config=default_model_config, load_backbone=False)
        assert model.total_parameters() == model.trainable_parameters()
