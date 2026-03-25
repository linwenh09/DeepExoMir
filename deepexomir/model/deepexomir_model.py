"""Complete DeepExoMir model assembly.

Brings together all sub-modules into a single ``nn.Module`` that accepts
raw RNA sequences (or pre-computed embeddings) together with structural
features and produces binding predictions.

Model ⑦ pipeline::

    miRNA sequences ─┐
                     ├─> RNABackbone (frozen) ─> Linear projection + LayerNorm
    target sequences ┘                           + positional embeddings
                                                          │
                                                          v
                                                  CrossAttentionEncoder
                                                  (8 bidirectional layers)
                                                          │
                                                     ┌────┴────┐
                                                     v         v
                                              mean pooling   Contact Map CNN (v7)
                                              [B, 512]       [B, 128]
                                                     │         │
    base-pairing matrix ──> BasePairingCNN  ──> [B, 128] ─────┤
                                                               │
    structural features ──> StructuralMLP   ──> [B,  64] ─────┤
                                                               │
    backbone mean-pool  ──> BackboneFeatMLP ──> [B, 256] ─────┤
                                                               │
                                                      concat [B, 1088]
                                                               │
                                                               v
                                                    ClassificationHead
                                                        [B, 2] logits
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Fast character-to-RNA-ID lookup (C-level numpy, no Python loops)
_CHAR_TO_RNA_ID = np.full(256, 4, dtype=np.int8)  # default = N (4)
_CHAR_TO_RNA_ID[ord("A")] = 0
_CHAR_TO_RNA_ID[ord("U")] = 1
_CHAR_TO_RNA_ID[ord("G")] = 2
_CHAR_TO_RNA_ID[ord("C")] = 3

from deepexomir.model.backbone import RNABackbone
from deepexomir.model.classifier import ClassificationHead
from deepexomir.model.cross_attention import CrossAttentionEncoder
from deepexomir.model.structural_encoder import (
    BasePairingCNN,
    ContactMapCNN,
    StructuralMLP,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration helper
# ---------------------------------------------------------------------------

def _get(cfg: Any, key: str, default: Any = None) -> Any:
    """Retrieve *key* from a dict, namespace, or dataclass-like object."""
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _nested_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """Retrieve a nested key chain (e.g. ``backbone.embed_dim``)."""
    current = cfg
    for k in keys:
        current = _get(current, k)
        if current is None:
            return default
    return current


# ---------------------------------------------------------------------------
# DeepExoMir Model
# ---------------------------------------------------------------------------

class DeepExoMirModel(nn.Module):
    """Complete DeepExoMir model for miRNA-target interaction prediction.

    The model supports **two forward modes**:

    1. **Raw-sequence mode** -- pass ``mirna_seqs`` and ``target_seqs`` as
       lists of strings.  The frozen :class:`RNABackbone` extracts embeddings
       on the fly.
    2. **Pre-computed embedding mode** -- pass ``mirna_emb`` and ``target_emb``
       as tensors directly (useful during training when the backbone is frozen
       and embeddings can be computed once and cached).

    Parameters
    ----------
    config : dict or namespace
        Model configuration.  Expected structure matches
        ``configs/model_config.yaml``:

        .. code-block:: yaml

            backbone:
              name: multimolecule/rinalmo-giga
              embed_dim: 1280
              freeze: true
            model:
              d_model: 256
              n_heads: 8
              d_ff: 1024
              n_cross_layers: 4
              dropout: 0.2
              attention_dropout: 0.1
              max_mirna_len: 30
              max_target_len: 40
            structural:
              bp_cnn_out: 128
              struct_mlp_in: 6
              struct_mlp_out: 64
            classifier:
              hidden_dims: [256, 128]
              n_classes: 2
              platt_scaling: true

    load_backbone : bool
        If ``False``, skip loading the HuggingFace backbone model.  This is
        useful when you only intend to use pre-computed embeddings and want
        to avoid the large download (default: ``True``).
    precomputed_embeddings : bool
        If ``True`` (and ``load_backbone=False``), configure the model to
        accept pre-computed RiNALMo embeddings via ``mirna_emb`` / ``target_emb``
        kwargs.  This preserves the backbone_dim→d_model projection layers
        (e.g. 1280→256) without downloading the 650M-parameter backbone
        model.  Default: ``False``.
    cache_dir : str or Path or None
        Embedding cache directory forwarded to :class:`RNABackbone`.
    """

    def __init__(
        self,
        config: Union[Dict[str, Any], Any],
        load_backbone: bool = True,
        precomputed_embeddings: bool = False,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # ---- unpack config --------------------------------------------------
        backbone_cfg = _get(config, "backbone", {})
        model_cfg = _get(config, "model", {})
        struct_cfg = _get(config, "structural", {})
        cls_cfg = _get(config, "classifier", {})

        backbone_name: str = _get(backbone_cfg, "name", "multimolecule/rinalmo-giga")
        backbone_embed_dim: int = _get(backbone_cfg, "embed_dim", 1280)
        backbone_freeze: bool = _get(backbone_cfg, "freeze", True)

        d_model: int = _get(model_cfg, "d_model", 256)
        n_heads: int = _get(model_cfg, "n_heads", 8)
        d_ff: int = _get(model_cfg, "d_ff", 1024)
        n_cross_layers: int = _get(model_cfg, "n_cross_layers", 4)
        dropout: float = _get(model_cfg, "dropout", 0.2)
        attn_dropout: float = _get(model_cfg, "attention_dropout", 0.1)
        drop_path_rate: float = _get(model_cfg, "drop_path_rate", 0.0)
        use_swiglu: bool = _get(model_cfg, "use_swiglu", False)
        self.max_mirna_len: int = _get(model_cfg, "max_mirna_len", 30)
        self.max_target_len: int = _get(model_cfg, "max_target_len", 40)

        bp_cnn_out: int = _get(struct_cfg, "bp_cnn_out", 128)
        struct_mlp_in: int = _get(struct_cfg, "struct_mlp_in", 8)  # v7: 8 features
        struct_mlp_out: int = _get(struct_cfg, "struct_mlp_out", 64)

        # ---- Contact Map config (Model ⑦) -----------------------------------
        contact_cfg = _get(config, "contact_map", {})
        use_contact_map: bool = _get(contact_cfg, "enabled", True)
        contact_proj_dim: int = _get(contact_cfg, "proj_dim", 32)
        contact_out_dim: int = _get(contact_cfg, "out_dim", 128)
        self.use_contact_map = use_contact_map

        cls_hidden: list = _get(cls_cfg, "hidden_dims", [256, 128])
        n_classes: int = _get(cls_cfg, "n_classes", 2)
        platt: bool = _get(cls_cfg, "platt_scaling", True)

        # ---- Determine encoding mode ----------------------------------------
        # use_backbone_for_crossattn: whether to use backbone features in
        # cross-attention (per-token) for BOTH miRNA and target.
        # This is the highest-quality mode:
        #   - miRNA: pre-computed per-token backbone embeddings (from cache)
        #   - target: LIVE backbone forward pass each batch (frozen, no grad)
        self.load_backbone = load_backbone
        self.backbone: Optional[RNABackbone] = None
        self.use_precomputed = precomputed_embeddings
        # When backbone is loaded, use it for target cross-attention too
        self.use_backbone_for_crossattn = load_backbone

        # Determine projection input dimensions:
        # - When backbone is available (live or precomputed): 1280 → d_model
        # - Otherwise (lightweight only): d_model → d_model
        uses_backbone_features = load_backbone or precomputed_embeddings
        proj_in_dim = backbone_embed_dim if uses_backbone_features else d_model

        # Classifier input = cross_attn + bp_cnn + struct_mlp + backbone_feat + contact_map
        backbone_feat_dim: int = _get(model_cfg, "backbone_feat_dim", 256)
        backbone_feat_dim_val = backbone_feat_dim if uses_backbone_features else 0
        contact_dim_val = contact_out_dim if use_contact_map else 0
        cls_in_dim = (
            d_model * 2 + bp_cnn_out + struct_mlp_out
            + backbone_feat_dim_val + contact_dim_val
        )  # v7: 512 + 128 + 64 + 256 + 128 = 1088 (with backbone + contact map)

        # ---- 1. Backbone ---------------------------------------------------
        if load_backbone:
            self.backbone = RNABackbone(
                model_name=backbone_name,
                freeze=backbone_freeze,
                cache_dir=cache_dir,
            )
            actual_dim = self.backbone.embed_dim
            if actual_dim != backbone_embed_dim:
                logger.warning(
                    "Config specifies backbone embed_dim=%d but loaded model "
                    "has embed_dim=%d.  Using actual value.",
                    backbone_embed_dim, actual_dim,
                )
                backbone_embed_dim = actual_dim
                proj_in_dim = actual_dim

        self.backbone_embed_dim = backbone_embed_dim

        # Lightweight RNA Encoder – trainable 6-token embeddings.
        # Fallback when backbone is not available for a given input.
        self._rna_vocab = {"A": 0, "U": 1, "G": 2, "C": 3, "N": 4, "": 5}
        self._rna_pad_idx = 5
        self.rna_embedding = nn.Embedding(
            num_embeddings=6,  # A, U, G, C, N, PAD
            embedding_dim=d_model,
            padding_idx=self._rna_pad_idx,
        )
        logger.info(
            "Lightweight RNA encoder: d_model=%d (trainable, fallback).", d_model,
        )

        # ---- 1b. Backbone Feature MLP (mean-pooled backbone embeddings) ----
        # Processes [mirna_pooled || target_pooled] → [backbone_feat_dim]
        # and concatenates with the classification head input.
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
            logger.info(
                "Backbone feature MLP: [%d × 2] → %d.",
                backbone_embed_dim, backbone_feat_dim,
            )
        else:
            self.backbone_feature_mlp = None

        # ---- 2. Linear projections + LayerNorm -----------------------------
        # SYMMETRIC projections: both miRNA and target use backbone_dim → d_model
        # when backbone features are available (live or precomputed).
        self.mirna_projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.target_projection = nn.Sequential(
            nn.Linear(proj_in_dim, d_model),
            nn.LayerNorm(d_model),
        )
        logger.info(
            "Symmetric projections: miRNA %d→%d, target %d→%d.",
            proj_in_dim, d_model, proj_in_dim, d_model,
        )

        # ---- 3. Learnable positional embeddings -----------------------------
        self.mirna_pos_emb = nn.Embedding(self.max_mirna_len, d_model)
        self.target_pos_emb = nn.Embedding(self.max_target_len, d_model)

        # ---- 4. Cross-Attention Encoder -------------------------------------
        self.cross_encoder = CrossAttentionEncoder(
            n_layers=n_cross_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=attn_dropout,
            drop_path_rate=drop_path_rate,
            use_swiglu=use_swiglu,
        )

        # ---- 5. Structural Encoders -----------------------------------------
        self.bp_cnn = BasePairingCNN(out_dim=bp_cnn_out)
        self.struct_mlp = StructuralMLP(in_dim=struct_mlp_in, out_dim=struct_mlp_out)

        # ---- 5b. Contact Map CNN (Model ⑦) ----------------------------------
        self.contact_map_cnn: ContactMapCNN | None = None
        if use_contact_map:
            self.contact_map_cnn = ContactMapCNN(
                d_model=d_model,
                proj_dim=contact_proj_dim,
                out_dim=contact_out_dim,
            )
            logger.info(
                "Contact Map CNN: d_model=%d → proj_dim=%d → out_dim=%d",
                d_model, contact_proj_dim, contact_out_dim,
            )

        # ---- 6. Classification Head -----------------------------------------
        self.classifier = ClassificationHead(
            in_dim=cls_in_dim,
            hidden_dims=cls_hidden,
            n_classes=n_classes,
            platt_scaling=platt,
        )

        # ---- Dropout before classifier --------------------------------------
        self.pre_cls_dropout = nn.Dropout(dropout)

        # ---- GPU base-pairing lookup table (non-learnable buffer) ---------------
        # A=0, U=1, G=2, C=3, N=4, PAD=5
        bp_table = torch.full((6, 6), -1.0)  # mismatch default
        bp_table[0, 1] = 1.0; bp_table[1, 0] = 1.0  # A-U, U-A (Watson-Crick)
        bp_table[2, 3] = 1.0; bp_table[3, 2] = 1.0  # G-C, C-G
        bp_table[2, 1] = 0.5; bp_table[1, 2] = 0.5  # G-U, U-G (wobble)
        bp_table[4, :] = -0.5; bp_table[:, 4] = -0.5  # N = gap
        bp_table[5, :] = -0.5; bp_table[:, 5] = -0.5  # PAD = gap
        self.register_buffer("_bp_score_table", bp_table, persistent=False)

        logger.info(
            "DeepExoMirModel initialised: backbone_dim=%d, d_model=%d, "
            "cls_in=%d, n_cross_layers=%d, drop_path=%.2f, swiglu=%s, "
            "contact_map=%s",
            backbone_embed_dim, d_model, cls_in_dim, n_cross_layers,
            drop_path_rate, use_swiglu, use_contact_map,
        )

    # -- helpers --------------------------------------------------------------

    def _add_positional_embeddings(
        self,
        emb: torch.Tensor,
        pos_emb_layer: nn.Embedding,
        max_len: int,
    ) -> torch.Tensor:
        """Add learnable positional embeddings, truncating/padding as needed.

        Parameters
        ----------
        emb : Tensor [B, L, D]
        pos_emb_layer : nn.Embedding
        max_len : int

        Returns
        -------
        Tensor [B, L, D]
        """
        seq_len = min(emb.shape[1], max_len)
        positions = torch.arange(seq_len, device=emb.device)
        pos_emb = pos_emb_layer(positions)  # [L, D]
        emb[:, :seq_len, :] = emb[:, :seq_len, :] + pos_emb.unsqueeze(0)
        return emb

    @staticmethod
    def _mean_pool(
        emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mask-aware mean pooling over the sequence dimension.

        Parameters
        ----------
        emb : Tensor [B, L, D]
        mask : Tensor [B, L], optional
            ``True`` at **padding** positions.

        Returns
        -------
        Tensor [B, D]
        """
        if mask is not None:
            # Invert: True = valid token
            valid_mask = (~mask).unsqueeze(-1).float()  # [B, L, 1]
        else:
            valid_mask = torch.ones(
                emb.shape[0], emb.shape[1], 1,
                dtype=emb.dtype, device=emb.device,
            )

        sum_emb = (emb * valid_mask).sum(dim=1)            # [B, D]
        lengths = valid_mask.sum(dim=1).clamp(min=1.0)      # [B, 1]
        return sum_emb / lengths

    def _pad_or_truncate(
        self,
        emb: torch.Tensor,
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pad or truncate embeddings to *max_len* and return a padding mask.

        Parameters
        ----------
        emb : Tensor [B, L, D]
        max_len : int

        Returns
        -------
        emb : Tensor [B, max_len, D]
        mask : Tensor [B, max_len]
            ``True`` at padding positions.
        """
        B, L, D = emb.shape
        device = emb.device

        if L >= max_len:
            return emb[:, :max_len, :], torch.zeros(
                B, max_len, dtype=torch.bool, device=device,
            )

        # Pad
        pad = torch.zeros(B, max_len - L, D, dtype=emb.dtype, device=device)
        emb_padded = torch.cat([emb, pad], dim=1)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        mask[:, L:] = True
        return emb_padded, mask

    # -- batch tokenizer (shared by bp_matrix and lightweight encoder) ----------

    def _tokenize_batch(
        self,
        sequences: List[str],
        max_len: int,
        reverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of RNA strings into integer IDs on the model device.

        Uses numpy C-level array operations to avoid Python-level loops
        on individual characters.  Only the outer loop over sequences
        remains in Python.

        Parameters
        ----------
        sequences : list[str]
        max_len : int
        reverse : bool
            If True, reverse each sequence before tokenizing (for target
            3'->5' alignment in bp_matrix computation).

        Returns
        -------
        token_ids : Tensor [B, max_len]  (long)
        mask : Tensor [B, max_len]  (bool, True = padding)
        """
        B = len(sequences)
        # Build on CPU with numpy, then transfer once to GPU
        ids_np = np.full((B, max_len), self._rna_pad_idx, dtype=np.int64)
        mask_np = np.ones((B, max_len), dtype=np.bool_)

        for i, seq in enumerate(sequences):
            s = seq.upper().replace("T", "U")
            length = min(len(s), max_len)
            if length == 0:
                continue
            # Convert string to byte array, then lookup via numpy
            raw = np.frombuffer(s[:length].encode("ascii"), dtype=np.uint8)
            mapped = _CHAR_TO_RNA_ID[raw]  # vectorized lookup
            if reverse:
                ids_np[i, :length] = mapped[::-1]
            else:
                ids_np[i, :length] = mapped
            mask_np[i, :length] = False

        device = self._bp_score_table.device
        token_ids = torch.from_numpy(ids_np).to(device)
        mask = torch.from_numpy(mask_np).to(device)
        return token_ids, mask

    # -- GPU bp_matrix computation ---------------------------------------------

    def _compute_bp_matrix_gpu(
        self,
        mirna_seqs: List[str],
        target_seqs: List[str],
    ) -> torch.Tensor:
        """Compute base-pairing matrices entirely on GPU.

        Uses a registered buffer lookup table for vectorized scoring.
        ~100x faster than per-sample CPU computation.

        Returns
        -------
        Tensor [B, 1, max_mirna_len, max_target_len]
        """
        m_ids, _ = self._tokenize_batch(mirna_seqs, self.max_mirna_len, reverse=False)
        t_ids, _ = self._tokenize_batch(target_seqs, self.max_target_len, reverse=True)

        # Vectorized lookup: [B, 30, 40]
        bp_matrix = self._bp_score_table[m_ids[:, :, None], t_ids[:, None, :]]
        return bp_matrix.unsqueeze(1)  # [B, 1, H, W]

    # -- lightweight encoder ---------------------------------------------------

    def _encode_sequences(
        self,
        sequences: List[str],
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize and embed RNA sequences using the lightweight encoder.

        Re-uses the fast numpy-based ``_tokenize_batch`` method.

        Parameters
        ----------
        sequences : list[str]
            Raw RNA sequences (e.g. ["AUGCUA", "GCUAAU"]).
        max_len : int
            Maximum sequence length (pad/truncate).

        Returns
        -------
        embeddings : Tensor [B, max_len, d_model]
        mask : Tensor [B, max_len]
            True at padding positions.
        """
        token_ids, mask = self._tokenize_batch(sequences, max_len, reverse=False)
        embeddings = self.rna_embedding(token_ids)  # [B, max_len, d_model]
        return embeddings, mask

    # -- backbone per-token extraction ----------------------------------------

    @torch.no_grad()
    def _extract_backbone_pertoken(
        self,
        sequences: List[str],
        max_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract per-token backbone embeddings and create a padding mask.

        Runs the frozen backbone, strips special tokens (BOS/EOS), and
        pads/truncates to ``max_len``.

        Parameters
        ----------
        sequences : list[str]
            Raw RNA sequences.
        max_len : int
            Fixed output length (pad or truncate).

        Returns
        -------
        embeddings : Tensor [B, max_len, backbone_dim]
            Per-nucleotide embeddings (detached, float32).
        mask : Tensor [B, max_len]
            True at **padding** positions.
        """
        assert self.backbone is not None, "Backbone not loaded"

        device = next(self.backbone.model.parameters()).device
        encoded = self.backbone.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len + 10,  # room for special tokens
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs = self.backbone.model(
                input_ids=input_ids, attention_mask=attention_mask,
            )
        hidden = outputs.last_hidden_state  # [B, L_tok, D]

        B = hidden.shape[0]
        D = hidden.shape[2]

        # Strip BOS token (position 0)
        nuc_hidden = hidden[:, 1:, :]  # [B, L-1, D]

        # Pad or truncate to max_len
        L_avail = nuc_hidden.shape[1]
        if L_avail >= max_len:
            nuc_hidden = nuc_hidden[:, :max_len, :]
        else:
            pad = torch.zeros(B, max_len - L_avail, D, dtype=nuc_hidden.dtype, device=device)
            nuc_hidden = torch.cat([nuc_hidden, pad], dim=1)

        # Build mask: True = padding position
        pad_mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i in range(B):
            seq_len = min(len(sequences[i]), max_len)
            if seq_len < max_len:
                pad_mask[i, seq_len:] = True

        return nuc_hidden.detach().float(), pad_mask

    # -- forward --------------------------------------------------------------

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
    ) -> torch.Tensor:
        """Full forward pass.

        **Full hybrid mode** (per-token backbone for miRNA cross-attention +
        lightweight for target cross-attention + mean-pooled backbone features)::

            model(mirna_seqs=[...], target_seqs=[...],
                  mirna_pertoken_emb=..., mirna_pertoken_mask=...,
                  mirna_pooled_emb=..., target_pooled_emb=...)

        Parameters
        ----------
        mirna_seqs, target_seqs : list[str], optional
            Raw RNA sequences.
        bp_matrix : Tensor [B, 1, 30, 40]
        struct_features : Tensor [B, 6]
        mirna_emb, target_emb : Tensor, optional
            Legacy per-token embeddings.
        mirna_mask, target_mask : Tensor [B, L], optional
            Padding masks (True at padding positions).
        mirna_pooled_emb : Tensor [B, backbone_dim], optional
            Mean-pooled RiNALMo miRNA embeddings (for classifier).
        target_pooled_emb : Tensor [B, backbone_dim], optional
            Mean-pooled RiNALMo target embeddings (for classifier).
        mirna_pertoken_emb : Tensor [B, max_mirna_len, backbone_dim], optional
            Per-token RiNALMo miRNA embeddings (for cross-attention).
        mirna_pertoken_mask : Tensor [B, max_mirna_len], optional
            Validity mask for per-token miRNA embeddings (True = valid).

        Returns
        -------
        Tensor [B, n_classes]
        """
        # ---- Step 1: per-token embeddings ----------------------------------
        # miRNA: use per-token backbone embeddings if available (1280-dim),
        #        then try live backbone, then fall back to lightweight encoder.
        if mirna_pertoken_emb is not None:
            m_emb = mirna_pertoken_emb  # [B, 30, 1280]
            # Convert validity mask (True=valid) to padding mask (True=pad)
            if mirna_pertoken_mask is not None:
                mirna_mask = ~mirna_pertoken_mask
            else:
                mirna_mask = None
        elif self.backbone is not None and mirna_seqs is not None:
            # Live backbone extraction for miRNA
            m_emb, mirna_mask = self._extract_backbone_pertoken(
                mirna_seqs, self.max_mirna_len,
            )
        elif mirna_seqs is not None:
            m_emb, mirna_mask = self._encode_sequences(
                mirna_seqs, self.max_mirna_len,
            )
        elif mirna_emb is not None:
            m_emb = mirna_emb
        else:
            raise ValueError("Must provide mirna_pertoken_emb, mirna_seqs, or mirna_emb.")

        # Target: use live backbone for per-token embeddings if available,
        # otherwise fall back to lightweight encoder.
        if self.backbone is not None and target_seqs is not None:
            # Live backbone extraction for target (frozen, no gradients)
            t_emb, target_mask = self._extract_backbone_pertoken(
                target_seqs, self.max_target_len,
            )
        elif target_seqs is not None:
            t_emb, target_mask = self._encode_sequences(
                target_seqs, self.max_target_len,
            )
        elif target_emb is not None:
            t_emb = target_emb
        else:
            raise ValueError("Must provide target_seqs or target_emb.")

        # ---- Step 2: pad/truncate to fixed lengths --------------------------
        m_emb, mirna_mask_new = self._pad_or_truncate(m_emb, self.max_mirna_len)
        t_emb, target_mask_new = self._pad_or_truncate(t_emb, self.max_target_len)

        # Merge pre-existing masks with truncation masks (union of padding)
        if mirna_mask is not None:
            mirna_mask = mirna_mask[:, :self.max_mirna_len] | mirna_mask_new
        else:
            mirna_mask = mirna_mask_new

        if target_mask is not None:
            target_mask = target_mask[:, :self.max_target_len] | target_mask_new
        else:
            target_mask = target_mask_new

        # ---- Step 3: project d_model → d_model (lightweight encoder) --------
        m_emb = self.mirna_projection(m_emb)    # [B, Lm, d_model]
        t_emb = self.target_projection(t_emb)   # [B, Lt, d_model]

        # ---- Step 4: add positional embeddings ------------------------------
        m_emb = self._add_positional_embeddings(
            m_emb, self.mirna_pos_emb, self.max_mirna_len,
        )
        t_emb = self._add_positional_embeddings(
            t_emb, self.target_pos_emb, self.max_target_len,
        )

        # ---- Step 5: cross-attention encoder --------------------------------
        m_emb, t_emb = self.cross_encoder(
            m_emb, t_emb, mirna_mask, target_mask,
        )

        # ---- Step 5b: Contact Map from cross-attention output (v7) ----------
        # Compute BEFORE mean pooling to preserve per-position information
        contact_feat = None
        if self.contact_map_cnn is not None:
            contact_feat = self.contact_map_cnn(m_emb, t_emb)  # [B, contact_out]

        # ---- Step 6: mean pooling → [B, d_model] each ----------------------
        m_pooled = self._mean_pool(m_emb, mirna_mask)   # [B, d_model]
        t_pooled = self._mean_pool(t_emb, target_mask)  # [B, d_model]

        # Cross-attention representation: [B, 2*d_model]
        seq_repr = torch.cat([m_pooled, t_pooled], dim=-1)  # [B, 512]

        # ---- Step 7: collect all features -----------------------------------
        features = [seq_repr]

        B = seq_repr.shape[0]

        # Backbone mean-pooled features
        if self.backbone_feature_mlp is not None:
            if mirna_pooled_emb is not None and target_pooled_emb is not None:
                # Use pre-computed mean-pooled embeddings
                backbone_concat = torch.cat(
                    [mirna_pooled_emb, target_pooled_emb], dim=-1,
                )  # [B, 2*backbone_dim]
                backbone_feat = self.backbone_feature_mlp(backbone_concat)
                features.append(backbone_feat)
            elif self.backbone is not None and mirna_seqs is not None and target_seqs is not None:
                # Compute mean-pooled from live backbone
                with torch.no_grad():
                    m_pooled_bb = self.backbone.embed_and_pool(mirna_seqs)
                    t_pooled_bb = self.backbone.embed_and_pool(target_seqs)
                backbone_concat = torch.cat(
                    [m_pooled_bb.detach(), t_pooled_bb.detach()], dim=-1,
                )
                backbone_feat = self.backbone_feature_mlp(backbone_concat)
                features.append(backbone_feat)
            else:
                # No embeddings available → zero-pad
                features.append(
                    torch.zeros(B, self._backbone_feat_dim, device=seq_repr.device)
                )

        # Auto-compute bp_matrix on GPU when sequences are available
        if bp_matrix is None and mirna_seqs is not None and target_seqs is not None:
            bp_matrix = self._compute_bp_matrix_gpu(mirna_seqs, target_seqs)

        if bp_matrix is not None:
            bp_feat = self.bp_cnn(bp_matrix)          # [B, bp_cnn_out]
            features.append(bp_feat)
        else:
            features.append(torch.zeros(B, self.bp_cnn.out_dim, device=seq_repr.device))

        if struct_features is not None:
            sf_feat = self.struct_mlp(struct_features)  # [B, struct_mlp_out]
            features.append(sf_feat)
        else:
            features.append(torch.zeros(B, self.struct_mlp.out_dim, device=seq_repr.device))

        # Contact Map features (v7)
        if contact_feat is not None:
            features.append(contact_feat)

        # ---- Step 8: concatenate & classify ---------------------------------
        combined = torch.cat(features, dim=-1)  # [B, 704]
        combined = self.pre_cls_dropout(combined)

        logits = self.classifier(combined)  # [B, n_classes]
        return logits

    # -- inference helpers ----------------------------------------------------

    @torch.no_grad()
    def predict(
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
        threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Run inference and return probabilities + predicted labels."""
        self.eval()

        logits = self.forward(
            mirna_seqs=mirna_seqs,
            target_seqs=target_seqs,
            bp_matrix=bp_matrix,
            struct_features=struct_features,
            mirna_emb=mirna_emb,
            target_emb=target_emb,
            mirna_mask=mirna_mask,
            target_mask=target_mask,
            mirna_pooled_emb=mirna_pooled_emb,
            target_pooled_emb=target_pooled_emb,
            mirna_pertoken_emb=mirna_pertoken_emb,
            mirna_pertoken_mask=mirna_pertoken_mask,
        )

        probs = F.softmax(logits, dim=-1)                # [B, 2]
        pos_probs = probs[:, 1]                           # [B]
        predictions = (pos_probs >= threshold).long()      # [B]

        # Confidence: probability of the predicted class
        confidence = torch.where(
            predictions == 1, pos_probs, 1.0 - pos_probs,
        )

        return {
            "logits": logits,
            "probabilities": probs,
            "predictions": predictions,
            "confidence": confidence,
        }

    # -- utility methods ------------------------------------------------------

    def trainable_parameters(self) -> int:
        """Count trainable parameters (excludes frozen backbone)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_parameters(self) -> int:
        """Count all parameters including frozen backbone."""
        return sum(p.numel() for p in self.parameters())

    def freeze_backbone(self) -> None:
        """Ensure the backbone is fully frozen."""
        if self.backbone is not None:
            self.backbone._freeze()

    def __repr__(self) -> str:
        trainable = self.trainable_parameters()
        total = self.total_parameters()
        return (
            f"DeepExoMirModel(\n"
            f"  trainable_params={trainable:,},\n"
            f"  total_params={total:,},\n"
            f"  backbone_dim={self.backbone_embed_dim},\n"
            f"  max_mirna_len={self.max_mirna_len},\n"
            f"  max_target_len={self.max_target_len},\n"
            f")"
        )
