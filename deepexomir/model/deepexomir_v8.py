"""DeepExoMir Model v8 -- Unified architecture assembly.

Brings together all v8 sub-modules into a single ``nn.Module``.

Model v8 pipeline::

    miRNA sequences -+
                     +-> RiNALMo-giga (FROZEN, pre-computed embeddings)
    target sequences +
                                |
                                v
            miRNA: [B, 30, 1280]     Target: [B, 50, 1280]
                |                         |
                v                         v
        Linear(1280,256)+LN+PE     Linear(1280,256)+LN+PE
                |                         |
                v                         v
        +--------------------------------------------------+
        | HYBRID ENCODER (8 layers)                        |
        |   L0-1: BiConvGate (self-encoding)               |
        |   L2:   BiConvGate + Cross-Attention             |
        |   L3-4: BiConvGate (self-encoding)               |
        |   L5:   BiConvGate + Cross-Attention             |
        |   L6-7: BiConvGate (self-encoding)               |
        +--------------------------------------------------+
                |[B,30,256]        |[B,50,256]
                |                  |
        +-------+--+------+-------+--------+
        |          |      |                |
        v          v      v                v
    Mean Pool  Contact  BP CNN       Backbone
     [512]     Map CNN  [128]        Feat MLP
               [128]                  [256]
        |          |      |                |
        +-----+----+------+--------+------+
              |                    |
       Struct MLP              (concat)
        [64]                      |
              |                    |
              +----+---------+----+
                   |
           [512+128+128+64+256 = 1088]
                   |
                   v
        +----------------------------------+
        | MoE CLASSIFIER (4 experts, top-2) |
        +----------------------------------+
                   |
              Primary: P(target) logits [B, 2]
              +
              Aux: seed_type [B, 8]
              Aux: mfe_pred [B, 1]
              Aux: position_pred [B, 1]

Key differences from v7:
    - HybridEncoder replaces CrossAttentionEncoder (BiConvGate + CrossAttn)
    - MoEClassifier replaces ClassificationHead (4 experts, top-2)
    - MultiTaskHeads for auxiliary supervision
    - EvoAug for RNA-specific data augmentation
    - Support for per-token target embeddings (not just mean-pooled)
    - max_target_len increased from 40 to 50
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from deepexomir.model.backbone import RNABackbone
from deepexomir.model.evoaug import EvoAug
from deepexomir.model.hybrid_encoder import HybridEncoder
from deepexomir.model.interaction_pooling import InteractionPooling
from deepexomir.model.moe_classifier import MoEClassifier
from deepexomir.model.multitask_heads import MultiTaskHeads
from deepexomir.model.structural_encoder import (
    BasePairingCNN,
    ContactMapCNN,
    StructuralMLP,
)

logger = logging.getLogger(__name__)

# Fast character-to-RNA-ID lookup (C-level numpy, no Python loops)
_CHAR_TO_RNA_ID = np.full(256, 4, dtype=np.int8)  # default = N (4)
_CHAR_TO_RNA_ID[ord("A")] = 0
_CHAR_TO_RNA_ID[ord("U")] = 1
_CHAR_TO_RNA_ID[ord("G")] = 2
_CHAR_TO_RNA_ID[ord("C")] = 3


def _get(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


class DeepExoMirModelV8(nn.Module):
    """Model v8: Hybrid encoder + MoE classifier + multi-task heads.

    Parameters
    ----------
    config : dict
        Model configuration. Expected keys:

        .. code-block:: yaml

            backbone:
              name: multimolecule/rinalmo-giga
              embed_dim: 1280
              freeze: true
            model:
              d_model: 256
              n_heads: 8
              d_ff: 1024
              n_layers: 8
              d_conv: 4
              expand: 2
              cross_attn_every: 3
              dropout: 0.2
              attention_dropout: 0.1
              drop_path_rate: 0.1
              use_swiglu: true
              max_mirna_len: 30
              max_target_len: 50
              backbone_feat_dim: 256
            structural:
              bp_cnn_out: 128
              struct_mlp_in: 8
              struct_mlp_out: 64
            contact_map:
              enabled: true
              proj_dim: 32
              out_dim: 128
            classifier:
              type: moe
              n_experts: 4
              top_k: 2
              hidden_dim: 256
              n_classes: 2
              platt_scaling: true
            multitask:
              enabled: true
              w_seed: 0.3
              w_mfe: 0.2
              w_position: 0.2
            augmentation:
              enabled: true
              p_augment: 0.3
              mutation_rate: 0.05
              struct_noise_std: 0.1

    load_backbone : bool
        Whether to load the RiNALMo backbone (default: False for v8,
        since we use pre-computed embeddings).
    precomputed_embeddings : bool
        Whether to use pre-computed embeddings (default: True).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        load_backbone: bool = False,
        precomputed_embeddings: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # ---- Unpack config --------------------------------------------------
        backbone_cfg = _get(config, "backbone", {})
        model_cfg = _get(config, "model", {})
        struct_cfg = _get(config, "structural", {})
        contact_cfg = _get(config, "contact_map", {})
        cls_cfg = _get(config, "classifier", {})
        mt_cfg = _get(config, "multitask", {})
        aug_cfg = _get(config, "augmentation", {})

        backbone_embed_dim: int = _get(backbone_cfg, "embed_dim", 1280)
        d_model: int = _get(model_cfg, "d_model", 256)
        n_heads: int = _get(model_cfg, "n_heads", 8)
        d_ff: int = _get(model_cfg, "d_ff", 1024)
        n_layers: int = _get(model_cfg, "n_layers", 8)
        d_conv: int = _get(model_cfg, "d_conv", 4)
        expand: int = _get(model_cfg, "expand", 2)
        cross_attn_every: int = _get(model_cfg, "cross_attn_every", 3)
        dropout: float = _get(model_cfg, "dropout", 0.2)
        attn_dropout: float = _get(model_cfg, "attention_dropout", 0.1)
        drop_path_rate: float = _get(model_cfg, "drop_path_rate", 0.1)
        use_swiglu: bool = _get(model_cfg, "use_swiglu", True)
        self.max_mirna_len: int = _get(model_cfg, "max_mirna_len", 30)
        self.max_target_len: int = _get(model_cfg, "max_target_len", 50)

        bp_cnn_out: int = _get(struct_cfg, "bp_cnn_out", 128)
        bp_channels: int = _get(struct_cfg, "bp_channels", 1)
        self._bp_channels = bp_channels
        struct_mlp_in: int = _get(struct_cfg, "struct_mlp_in", 8)
        struct_mlp_out: int = _get(struct_cfg, "struct_mlp_out", 64)
        struct_mlp_hidden: int | None = _get(struct_cfg, "struct_mlp_hidden", None)
        struct_mlp_layers: int = _get(struct_cfg, "struct_mlp_layers", 2)
        struct_feat_dropout: float = _get(struct_cfg, "struct_feat_dropout", 0.0)

        use_contact_map: bool = _get(contact_cfg, "enabled", True)
        contact_proj_dim: int = _get(contact_cfg, "proj_dim", 32)
        contact_out_dim: int = _get(contact_cfg, "out_dim", 128)
        self.use_contact_map = use_contact_map

        backbone_feat_dim: int = _get(model_cfg, "backbone_feat_dim", 256)

        uses_backbone_features = load_backbone or precomputed_embeddings
        proj_in_dim = backbone_embed_dim if uses_backbone_features else d_model
        backbone_feat_dim_val = backbone_feat_dim if uses_backbone_features else 0
        contact_dim_val = contact_out_dim if use_contact_map else 0

        # Classifier input dimension
        cls_in_dim = (
            d_model * 2          # mean-pooled encoder output
            + bp_cnn_out         # base-pairing CNN
            + struct_mlp_out     # structural features
            + backbone_feat_dim_val  # backbone features
            + contact_dim_val    # contact map CNN
        )

        self.load_backbone = load_backbone
        self.use_precomputed = precomputed_embeddings
        self.backbone_embed_dim = backbone_embed_dim

        # ---- 1. Backbone (optional) ----------------------------------------
        self.backbone: Optional[RNABackbone] = None
        if load_backbone:
            backbone_name = _get(backbone_cfg, "name", "multimolecule/rinalmo-giga")
            backbone_freeze = _get(backbone_cfg, "freeze", True)
            self.backbone = RNABackbone(
                model_name=backbone_name,
                freeze=backbone_freeze,
                cache_dir=cache_dir,
            )

        # ---- 1b. Backbone Feature MLP (mean-pooled) ------------------------
        self._backbone_feat_dim = backbone_feat_dim_val
        if uses_backbone_features:
            self.backbone_feature_mlp = nn.Sequential(
                nn.Linear(backbone_embed_dim * 2, backbone_feat_dim),
                nn.LayerNorm(backbone_feat_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(backbone_feat_dim, backbone_feat_dim),
                nn.LayerNorm(backbone_feat_dim),
            )
        else:
            self.backbone_feature_mlp = None

        # ---- 2. Linear projections + LayerNorm ------------------------------
        self.mirna_projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ---- 3. Positional embeddings ---------------------------------------
        self.mirna_pos_emb = nn.Embedding(self.max_mirna_len, d_model)
        self.target_pos_emb = nn.Embedding(self.max_target_len, d_model)

        # ---- 4. Hybrid Encoder (v8: BiConvGate + CrossAttention) ------------
        self.encoder = HybridEncoder(
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            d_conv=d_conv,
            expand=expand,
            dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
            cross_attn_every=cross_attn_every,
            use_swiglu=use_swiglu,
        )

        # ---- 4b. Interaction Pooling (optional, replaces mean pool) ----------
        pooling_cfg = _get(config, "pooling", {})
        self.use_interaction_pooling = _get(pooling_cfg, "type", "mean") == "interaction"
        if self.use_interaction_pooling:
            pool_heads = _get(pooling_cfg, "n_heads", 4)
            self.interaction_pooling = InteractionPooling(
                d_model=d_model,
                n_heads=pool_heads,
                dropout=dropout,
            )
            logger.info(
                "Interaction pooling enabled: %d heads, output=%d",
                pool_heads, d_model * 2,
            )

        # ---- 5. Structural Encoders -----------------------------------------
        self.bp_cnn = BasePairingCNN(out_dim=bp_cnn_out, in_channels=bp_channels)
        self.struct_mlp = StructuralMLP(
            in_dim=struct_mlp_in, out_dim=struct_mlp_out,
            hidden_dim=struct_mlp_hidden, n_layers=struct_mlp_layers,
            feat_dropout=struct_feat_dropout,
        )

        # ---- 5b. Contact Map CNN --------------------------------------------
        self.contact_map_cnn: Optional[ContactMapCNN] = None
        if use_contact_map:
            self.contact_map_cnn = ContactMapCNN(
                d_model=d_model,
                proj_dim=contact_proj_dim,
                out_dim=contact_out_dim,
            )

        # ---- 5c. Duplex Graph Attention (v22) --------------------------------
        gat_cfg = _get(config, "duplex_gat", {})
        self.use_duplex_gat = _get(gat_cfg, "enabled", False)
        self._duplex_gat_out = 0
        if self.use_duplex_gat:
            from deepexomir.model.duplex_gat import DuplexGAT
            gat_out = _get(gat_cfg, "out_dim", 128)
            self._duplex_gat_out = gat_out
            self.duplex_gat = DuplexGAT(
                node_dim=d_model,
                hidden_dim=_get(gat_cfg, "hidden_dim", 128),
                out_dim=gat_out,
                n_heads=_get(gat_cfg, "n_heads", 4),
                n_layers=_get(gat_cfg, "n_layers", 2),
                dropout=dropout,
                max_mirna_len=self.max_mirna_len,
                max_target_len=self.max_target_len,
            )
            cls_in_dim += gat_out

        # ---- 6. MoE Classifier (v8) ----------------------------------------
        cls_type = _get(cls_cfg, "type", "moe")
        if cls_type == "moe":
            self.classifier = MoEClassifier(
                in_dim=cls_in_dim,
                hidden_dim=_get(cls_cfg, "hidden_dim", 256),
                n_experts=_get(cls_cfg, "n_experts", 4),
                top_k=_get(cls_cfg, "top_k", 2),
                n_classes=_get(cls_cfg, "n_classes", 2),
                dropout=dropout,
                platt_scaling=_get(cls_cfg, "platt_scaling", True),
            )
        else:
            # Fallback to simple MLP classifier (for ablation)
            from deepexomir.model.classifier import ClassificationHead
            self.classifier = ClassificationHead(
                in_dim=cls_in_dim,
                n_classes=_get(cls_cfg, "n_classes", 2),
                platt_scaling=_get(cls_cfg, "platt_scaling", True),
            )

        # ---- 7. Multi-task Heads (v8) ---------------------------------------
        self.use_multitask = _get(mt_cfg, "enabled", True)
        contrastive_dim = _get(mt_cfg, "contrastive_dim", 0)
        if self.use_multitask:
            self.multitask_heads = MultiTaskHeads(
                encoder_dim=d_model * 2,  # concatenated miRNA + target
                dropout=dropout,
                contrastive_dim=contrastive_dim,
            )
            if contrastive_dim > 0:
                logger.info("Contrastive head enabled: proj_dim=%d", contrastive_dim)

        # ---- 8. EvoAug (v8) -------------------------------------------------
        self.use_augmentation = _get(aug_cfg, "enabled", True)
        self._use_emb_augment = _get(aug_cfg, "emb_augment", False)
        if self.use_augmentation:
            self.evoaug = EvoAug(
                p_augment=_get(aug_cfg, "p_augment", 0.3),
                mutation_rate=_get(aug_cfg, "mutation_rate", 0.05),
                struct_noise_std=_get(aug_cfg, "struct_noise_std", 0.1),
            )
            if self._use_emb_augment:
                logger.info("Per-token embedding augmentation enabled")

        # ---- Dropout before classifier --------------------------------------
        self.pre_cls_dropout = nn.Dropout(dropout)

        # ---- GPU BP lookup tables ----------------------------------------
        # Original combined score table (v7/v8 compatible)
        bp_table = torch.full((6, 6), -1.0)
        bp_table[0, 1] = 1.0; bp_table[1, 0] = 1.0  # A-U
        bp_table[2, 3] = 1.0; bp_table[3, 2] = 1.0  # G-C
        bp_table[2, 1] = 0.5; bp_table[1, 2] = 0.5  # G-U wobble
        bp_table[4, :] = -0.5; bp_table[:, 4] = -0.5  # N
        bp_table[5, :] = -0.5; bp_table[:, 5] = -0.5  # PAD
        self.register_buffer("_bp_score_table", bp_table, persistent=False)

        if bp_channels > 1:
            # v10: Multi-channel BP matrix lookup tables
            # Ch0: A-U pairing (binary)
            au = torch.zeros(6, 6)
            au[0, 1] = 1.0; au[1, 0] = 1.0
            # Ch1: G-C pairing (binary)
            gc = torch.zeros(6, 6)
            gc[2, 3] = 1.0; gc[3, 2] = 1.0
            # Ch2: G-U wobble (binary)
            gu = torch.zeros(6, 6)
            gu[2, 1] = 1.0; gu[1, 2] = 1.0
            # Ch3: Any canonical pairing (binary union)
            any_pair = au + gc + gu
            # Ch4: Mismatch between valid nucleotides (binary)
            mismatch = torch.zeros(6, 6)
            for i in range(4):
                for j in range(4):
                    if any_pair[i, j] == 0.0:
                        mismatch[i, j] = 1.0
            # Ch5: Combined score (original continuous, backward compat)
            combined = bp_table.clone()
            tables = torch.stack(
                [au, gc, gu, any_pair, mismatch, combined], dim=0,
            )  # [6, 6, 6]
            self.register_buffer("_bp_score_tables", tables, persistent=False)
            logger.info("6-channel BP matrix enabled")

        # ---- RNA vocabulary --------------------------------------------------
        self._rna_pad_idx = 5

        logger.info(
            "DeepExoMirModelV8 initialised: d_model=%d, n_layers=%d, "
            "cls_in=%d, cls_type=%s, multitask=%s, evoaug=%s, bp_ch=%d",
            d_model, n_layers, cls_in_dim, cls_type,
            self.use_multitask, self.use_augmentation, bp_channels,
        )

    # -- helpers ---------------------------------------------------------------

    def _add_positional_embeddings(
        self, emb: torch.Tensor, pos_emb_layer: nn.Embedding, max_len: int,
    ) -> torch.Tensor:
        seq_len = min(emb.shape[1], max_len)
        positions = torch.arange(seq_len, device=emb.device)
        pos_emb = pos_emb_layer(positions)
        emb[:, :seq_len, :] = emb[:, :seq_len, :] + pos_emb.unsqueeze(0)
        return emb

    @staticmethod
    def _mean_pool(
        emb: torch.Tensor, mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
        else:
            valid_mask = torch.ones(
                emb.shape[0], emb.shape[1], 1,
                dtype=emb.dtype, device=emb.device,
            )
        sum_emb = (emb * valid_mask).sum(dim=1)
        lengths = valid_mask.sum(dim=1).clamp(min=1.0)
        return sum_emb / lengths

    def _pad_or_truncate(
        self, emb: torch.Tensor, max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = emb.shape
        device = emb.device
        if L >= max_len:
            return emb[:, :max_len, :], torch.zeros(
                B, max_len, dtype=torch.bool, device=device,
            )
        pad = torch.zeros(B, max_len - L, D, dtype=emb.dtype, device=device)
        emb_padded = torch.cat([emb, pad], dim=1)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        mask[:, L:] = True
        return emb_padded, mask

    def _tokenize_batch(
        self, sequences: List[str], max_len: int, reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = len(sequences)
        ids_np = np.full((B, max_len), self._rna_pad_idx, dtype=np.int64)
        mask_np = np.ones((B, max_len), dtype=np.bool_)
        for i, seq in enumerate(sequences):
            s = seq.upper().replace("T", "U")
            length = min(len(s), max_len)
            if length == 0:
                continue
            raw = np.frombuffer(s[:length].encode("ascii"), dtype=np.uint8)
            mapped = _CHAR_TO_RNA_ID[raw]
            if reverse:
                ids_np[i, :length] = mapped[::-1]
            else:
                ids_np[i, :length] = mapped
            mask_np[i, :length] = False
        device = self._bp_score_table.device
        token_ids = torch.from_numpy(ids_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        return token_ids, mask

    def _compute_bp_matrix_gpu(
        self, mirna_seqs: List[str], target_seqs: List[str],
    ) -> torch.Tensor:
        m_ids, _ = self._tokenize_batch(mirna_seqs, self.max_mirna_len, reverse=False)
        t_ids, _ = self._tokenize_batch(target_seqs, self.max_target_len, reverse=True)
        if self._bp_channels > 1:
            # v10: multi-channel [C,6,6] x [B,Lm,1] x [B,1,Lt] -> [C,B,Lm,Lt]
            bp = self._bp_score_tables[:, m_ids[:, :, None], t_ids[:, None, :]]
            return bp.permute(1, 0, 2, 3)  # [B, C, Lm, Lt]
        bp_matrix = self._bp_score_table[m_ids[:, :, None], t_ids[:, None, :]]
        return bp_matrix.unsqueeze(1)  # [B, 1, Lm, Lt]

    # -- forward ---------------------------------------------------------------

    def forward(
        self,
        mirna_seqs: Optional[List[str]] = None,
        target_seqs: Optional[List[str]] = None,
        bp_matrix: Optional[torch.Tensor] = None,
        struct_features: Optional[torch.Tensor] = None,
        *,
        mirna_emb: Optional[torch.Tensor] = None,
        target_emb: Optional[torch.Tensor] = None,
        mirna_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        mirna_pooled_emb: Optional[torch.Tensor] = None,
        target_pooled_emb: Optional[torch.Tensor] = None,
        mirna_pertoken_emb: Optional[torch.Tensor] = None,
        mirna_pertoken_mask: Optional[torch.Tensor] = None,
        target_pertoken_emb: Optional[torch.Tensor] = None,
        target_pertoken_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass.

        Unlike v7, this returns a dictionary containing:
        - ``logits``: primary classification logits [B, 2]
        - ``aux_preds``: auxiliary task predictions (if multitask enabled)
        - ``load_balance_loss``: MoE load balancing loss

        Parameters
        ----------
        mirna_seqs, target_seqs : list[str], optional
        bp_matrix : Tensor [B, 1, Lm, Lt], optional
        struct_features : Tensor [B, n_feat], optional
        mirna_pertoken_emb : Tensor [B, Lm, D], optional
            Per-token miRNA backbone embeddings.
        target_pertoken_emb : Tensor [B, Lt, D], optional
            Per-token target backbone embeddings (v8: NEW).
        mirna_pooled_emb, target_pooled_emb : Tensor [B, D], optional
            Mean-pooled backbone embeddings for classifier.
        *_mask : Tensor [B, L], optional
            True = valid token for pertoken masks.
            True = padding for raw masks.

        Returns
        -------
        dict
            - ``logits``: Tensor [B, n_classes]
            - ``aux_preds``: dict (if multitask enabled)
            - ``load_balance_loss``: Tensor (scalar)
        """
        # ---- Step 1: Resolve per-token embeddings ---------------------------

        # miRNA: prefer per-token backbone embeddings
        if mirna_pertoken_emb is not None:
            m_emb = mirna_pertoken_emb  # [B, 30, 1280]
            if mirna_pertoken_mask is not None:
                mirna_mask = ~mirna_pertoken_mask  # validity -> padding mask
        elif mirna_emb is not None:
            m_emb = mirna_emb
        else:
            raise ValueError("Must provide mirna_pertoken_emb or mirna_emb")

        # Target: prefer per-token backbone embeddings (v8 NEW)
        if target_pertoken_emb is not None:
            t_emb = target_pertoken_emb  # [B, 50, 1280]
            if target_pertoken_mask is not None:
                target_mask = ~target_pertoken_mask  # validity -> padding mask
        elif target_emb is not None:
            t_emb = target_emb
        else:
            raise ValueError("Must provide target_pertoken_emb or target_emb")

        # ---- Step 2: Pad/truncate to fixed lengths --------------------------
        m_emb, mirna_mask_new = self._pad_or_truncate(m_emb, self.max_mirna_len)
        t_emb, target_mask_new = self._pad_or_truncate(t_emb, self.max_target_len)

        if mirna_mask is not None:
            mirna_mask = mirna_mask[:, :self.max_mirna_len] | mirna_mask_new
        else:
            mirna_mask = mirna_mask_new

        if target_mask is not None:
            target_mask = target_mask[:, :self.max_target_len] | target_mask_new
        else:
            target_mask = target_mask_new

        # ---- Step 2b: Per-token embedding augmentation (v10) ------------------
        if self.training and self.use_augmentation and self._use_emb_augment:
            m_emb = self.evoaug.augment_pertoken_embeddings(
                m_emb, padding_mask=mirna_mask,
            )
            t_emb = self.evoaug.augment_pertoken_embeddings(
                t_emb, padding_mask=target_mask,
            )

        # ---- Step 3: Project backbone_dim -> d_model ------------------------
        m_emb = self.mirna_projection(m_emb)
        t_emb = self.target_projection(t_emb)

        # ---- Step 4: Add positional embeddings ------------------------------
        m_emb = self._add_positional_embeddings(
            m_emb, self.mirna_pos_emb, self.max_mirna_len,
        )
        t_emb = self._add_positional_embeddings(
            t_emb, self.target_pos_emb, self.max_target_len,
        )

        # ---- Step 5: Hybrid encoder ----------------------------------------
        m_emb, t_emb = self.encoder(m_emb, t_emb, mirna_mask, target_mask)

        # ---- Step 5b: Contact Map CNN (before pooling) ----------------------
        contact_feat = None
        if self.contact_map_cnn is not None:
            contact_feat = self.contact_map_cnn(m_emb, t_emb)

        # ---- Step 6: Pooling -------------------------------------------------
        if self.use_interaction_pooling:
            seq_repr = self.interaction_pooling(
                m_emb, t_emb, mirna_mask, target_mask,
            )  # [B, 2*d_model]
        else:
            m_pooled = self._mean_pool(m_emb, mirna_mask)
            t_pooled = self._mean_pool(t_emb, target_mask)
            seq_repr = torch.cat([m_pooled, t_pooled], dim=-1)  # [B, 2*d_model]

        # ---- Step 7: Collect features for classifier ------------------------
        B = seq_repr.shape[0]
        features = [seq_repr]

        # Backbone mean-pooled features
        if self.backbone_feature_mlp is not None:
            if mirna_pooled_emb is not None and target_pooled_emb is not None:
                backbone_concat = torch.cat(
                    [mirna_pooled_emb, target_pooled_emb], dim=-1,
                )
                backbone_feat = self.backbone_feature_mlp(backbone_concat)
                features.append(backbone_feat)
            else:
                features.append(
                    torch.zeros(B, self._backbone_feat_dim, device=seq_repr.device)
                )

        # BP matrix
        if self._bp_channels > 1 and mirna_seqs is not None and target_seqs is not None:
            # v10: always compute multi-channel BP on-the-fly
            bp_matrix = self._compute_bp_matrix_gpu(mirna_seqs, target_seqs)
        elif bp_matrix is None and mirna_seqs is not None and target_seqs is not None:
            bp_matrix = self._compute_bp_matrix_gpu(mirna_seqs, target_seqs)

        if bp_matrix is not None:
            # EvoAug: optionally perturb BP matrix during training
            if self.training and self.use_augmentation:
                bp_matrix = self.evoaug.augment_bp_matrix(bp_matrix)
            features.append(self.bp_cnn(bp_matrix))
        else:
            features.append(
                torch.zeros(B, self.bp_cnn.out_dim, device=seq_repr.device)
            )

        # Structural features
        if struct_features is not None:
            # EvoAug: optionally add noise during training
            if self.training and self.use_augmentation:
                struct_features = self.evoaug.augment_structural_features(
                    struct_features,
                )
            features.append(self.struct_mlp(struct_features))
        else:
            features.append(
                torch.zeros(B, self.struct_mlp.out_dim, device=seq_repr.device)
            )

        # Contact map features
        if contact_feat is not None:
            features.append(contact_feat)

        # Duplex GAT features (v22)
        if self.use_duplex_gat:
            gat_feat = self.duplex_gat(
                m_emb, t_emb,
                mirna_seqs=mirna_seqs,
                target_seqs=target_seqs,
                mirna_mask=mirna_mask,
                target_mask=target_mask,
            )
            features.append(gat_feat)

        # ---- Step 8: Concatenate & classify ---------------------------------
        combined = torch.cat(features, dim=-1)
        combined = self.pre_cls_dropout(combined)

        logits = self.classifier(combined)

        # ---- Step 9: Multi-task predictions ---------------------------------
        result = {"logits": logits}

        if self.use_multitask:
            result["aux_preds"] = self.multitask_heads(seq_repr)

        # MoE load balance loss
        if hasattr(self.classifier, "load_balance_loss"):
            result["load_balance_loss"] = self.classifier.load_balance_loss
        else:
            result["load_balance_loss"] = torch.tensor(
                0.0, device=logits.device,
            )

        return result

    # -- inference helpers -----------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return probabilities + labels."""
        self.eval()
        result = self.forward(**kwargs)
        logits = result["logits"]
        probs = F.softmax(logits, dim=-1)
        pos_probs = probs[:, 1]
        predictions = (pos_probs >= threshold).long()
        confidence = torch.where(predictions == 1, pos_probs, 1.0 - pos_probs)

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
            "confidence": confidence,
        }

    # -- utility methods -------------------------------------------------------

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def __repr__(self) -> str:
        trainable = self.trainable_parameters()
        total = self.total_parameters()
        return (
            f"DeepExoMirModelV8(\n"
            f"  trainable_params={trainable:,},\n"
            f"  total_params={total:,},\n"
            f"  backbone_dim={self.backbone_embed_dim},\n"
            f"  max_mirna_len={self.max_mirna_len},\n"
            f"  max_target_len={self.max_target_len},\n"
            f")"
        )
