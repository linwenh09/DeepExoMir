"""DeepExoMir model architecture modules.

Public API
----------
DeepExoMirModel      -- Complete end-to-end model (v7 cross-attention)
DeepExoMirModelV8    -- Model v8: Hybrid encoder + MoE + multi-task
RNABackbone          -- Frozen RNA foundation model embedding extractor
CrossAttentionBlock  -- Single bidirectional cross-attention block
CrossAttentionEncoder-- Stack of cross-attention blocks
HybridEncoder        -- BiConvGate + Cross-Attention hybrid encoder (v8)
BasePairingCNN       -- 2-D CNN for base-pairing matrices
ContactMapCNN        -- 2-D CNN for contact maps (v7)
StructuralMLP        -- MLP for scalar structural features
ClassificationHead   -- Classifier with temperature scaling (v7)
MoEClassifier        -- Mixture of Experts classifier (v8)
MultiTaskHeads       -- Auxiliary task prediction heads (v8)
MultiTaskLoss        -- Multi-task loss computation (v8)
EvoAug               -- RNA-specific data augmentation (v8)
FocalLoss            -- Focal loss with label smoothing
"""

from deepexomir.model.backbone import RNABackbone
from deepexomir.model.classifier import ClassificationHead
from deepexomir.model.cross_attention import CrossAttentionBlock, CrossAttentionEncoder
from deepexomir.model.deepexomir_model import DeepExoMirModel
from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
from deepexomir.model.evoaug import EvoAug
from deepexomir.model.hybrid_encoder import HybridEncoder
from deepexomir.model.losses import FocalLoss
from deepexomir.model.moe_classifier import MoEClassifier
from deepexomir.model.multitask_heads import MultiTaskHeads, MultiTaskLoss
from deepexomir.model.structural_encoder import (
    BasePairingCNN,
    ContactMapCNN,
    StructuralMLP,
)

__all__ = [
    "DeepExoMirModel",
    "DeepExoMirModelV8",
    "RNABackbone",
    "CrossAttentionBlock",
    "CrossAttentionEncoder",
    "HybridEncoder",
    "BasePairingCNN",
    "ContactMapCNN",
    "StructuralMLP",
    "ClassificationHead",
    "MoEClassifier",
    "MultiTaskHeads",
    "MultiTaskLoss",
    "EvoAug",
    "FocalLoss",
]
