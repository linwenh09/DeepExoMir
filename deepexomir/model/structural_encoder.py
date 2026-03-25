"""Structural feature encoders for DeepExoMir miRNA-target interaction.

Three complementary sub-modules encode different aspects of the predicted
RNA duplex structure:

``BasePairingCNN``
    A 2-D convolutional encoder that processes the ``[30 x 50]`` base-pairing
    score matrix (miRNA length x target length) into a fixed-size vector.

``ContactMapCNN``
    A 2-D convolutional encoder that processes the contact map (diff+mul
    features) between miRNA and target representations.  Based on
    TEC-miTarget (Yang et al., BMC Bioinformatics, 2024).

``StructuralMLP``
    A small MLP that encodes scalar structural / thermodynamic features
    (duplex MFE, miRNA MFE, target MFE, accessibility, GC content, seed type,
    AU content, seed duplex MFE).

Model ⑦ enhancements (v7):
    - ContactMapCNN for diff+mul interaction features
    - StructuralMLP updated for 8 input features (from 6)
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Base-Pairing CNN
# ---------------------------------------------------------------------------

class BasePairingCNN(nn.Module):
    """2-D CNN encoder for the base-pairing score matrix.

    The base-pairing matrix encodes pairwise complementarity scores between
    each miRNA position (rows) and each target-site position (columns).

    Architecture::

        Conv2d(1, 32, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        Conv2d(32, 64, 3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2)
        Conv2d(64, 128, 3, padding=1) -> BatchNorm2d -> ReLU
        AdaptiveAvgPool2d(1)
        Flatten -> Linear(128, out_dim)

    Parameters
    ----------
    out_dim : int
        Output feature dimension (default: 128).
    in_channels : int
        Number of input channels (default: 1).
    """

    def __init__(self, out_dim: int = 128, in_channels: int = 1) -> None:
        super().__init__()
        self.out_dim = out_dim

        self.conv_layers = nn.Sequential(
            # Block 1: 1 -> 32, with MaxPool to increase receptive field
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)  # -> [B, 128, 1, 1]
        self.fc = nn.Linear(128, out_dim)

    def forward(self, bp_matrix: torch.Tensor) -> torch.Tensor:
        """Encode the base-pairing matrix.

        Parameters
        ----------
        bp_matrix : Tensor [B, 1, 30, 40]
            Base-pairing score matrix with shape
            ``(batch, channels, mirna_len, target_len)``.

        Returns
        -------
        Tensor [B, out_dim]
        """
        x = self.conv_layers(bp_matrix)   # [B, 128, H', W']
        x = self.pool(x)                  # [B, 128, 1, 1]
        x = x.flatten(start_dim=1)        # [B, 128]
        x = self.fc(x)                    # [B, out_dim]
        return x


# ---------------------------------------------------------------------------
# Contact Map CNN (Model ⑦ — from TEC-miTarget)
# ---------------------------------------------------------------------------

class ContactMapCNN(nn.Module):
    """2-D CNN encoder for the contact map (diff+mul features).

    Computes element-wise **difference** and **multiplication** between
    miRNA and target representations, producing a 3-D contact map that
    captures fine-grained residue-level interaction patterns.

    Based on TEC-miTarget (Yang et al., BMC Bioinformatics, 2024)::

        miRNA repr [B, Lm, D]   target repr [B, Lt, D]
                \\                    /
              Linear(D, P)     Linear(D, P)      (project to smaller dim)
                \\                    /
        diff[i,j,k] = m[i,k] - t[j,k]   [B, Lm, Lt, P]
        mul[i,j,k]  = m[i,k] * t[j,k]   [B, Lm, Lt, P]
                        |
                concat → [B, 2P, Lm, Lt]
                        |
                    4-layer CNN
                        |
                    [B, out_dim]

    Parameters
    ----------
    d_model : int
        Input representation dimensionality (default: 256).
    proj_dim : int
        Projection dimensionality for memory efficiency (default: 32).
        The contact map has ``2 * proj_dim`` channels.
    out_dim : int
        Output feature dimension (default: 128).
    """

    def __init__(
        self,
        d_model: int = 256,
        proj_dim: int = 32,
        out_dim: int = 128,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.proj_dim = proj_dim
        self.out_dim = out_dim

        # Project to smaller dim for memory efficiency
        self.mirna_proj = nn.Linear(d_model, proj_dim)
        self.target_proj = nn.Linear(d_model, proj_dim)

        in_channels = 2 * proj_dim  # diff + mul

        # 4-layer CNN (TEC-miTarget style)
        self.cnn = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # Global average pool
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Linear(32, out_dim)

    def forward(
        self,
        mirna_emb: torch.Tensor,
        target_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Compute contact map features from miRNA and target representations.

        Parameters
        ----------
        mirna_emb : Tensor [B, Lm, D]
            miRNA representations (e.g. from cross-attention encoder).
        target_emb : Tensor [B, Lt, D]
            Target representations (e.g. from cross-attention encoder).

        Returns
        -------
        Tensor [B, out_dim]
        """
        # Project to smaller dimension for memory efficiency
        m = self.mirna_proj(mirna_emb)    # [B, Lm, P]
        t = self.target_proj(target_emb)  # [B, Lt, P]

        # Compute contact map via outer operations
        m_exp = m.unsqueeze(2)  # [B, Lm,  1, P]
        t_exp = t.unsqueeze(1)  # [B,  1, Lt, P]

        diff = m_exp - t_exp    # [B, Lm, Lt, P]
        mul = m_exp * t_exp     # [B, Lm, Lt, P]

        # Concatenate and reshape for CNN: [B, 2P, Lm, Lt]
        contact = torch.cat([diff, mul], dim=-1)   # [B, Lm, Lt, 2P]
        contact = contact.permute(0, 3, 1, 2)      # [B, 2P, Lm, Lt]

        # CNN feature extraction
        x = self.cnn(contact)      # [B, 32, 1, 1]
        x = x.flatten(start_dim=1) # [B, 32]
        x = self.fc(x)             # [B, out_dim]
        return x


# ---------------------------------------------------------------------------
# Structural MLP
# ---------------------------------------------------------------------------

class StructuralMLP(nn.Module):
    """MLP encoder for scalar structural / thermodynamic features.

    v11: Configurable depth (n_layers) and hidden dimension. Supports
    feature-wise dropout that randomly zeroes entire feature columns
    during training, forcing robustness across all features.

    Architecture (default 2-layer)::

        BN(in_dim) -> Linear(in_dim, hidden) -> GELU -> DO
        Linear(hidden, out_dim) -> LN

    Architecture (3-layer, v11)::

        BN(in_dim) -> Linear(in_dim, hidden) -> GELU -> DO
        Linear(hidden, hidden) -> GELU -> DO
        Linear(hidden, out_dim) -> LN

    Parameters
    ----------
    in_dim : int
        Number of input features (default: 6).
    out_dim : int
        Output feature dimension (default: 64).
    hidden_dim : int or None
        Hidden layer dimension. If None, defaults to 64 (legacy behavior).
    n_layers : int
        Number of linear layers (default: 2). Use 3 for v11.
    dropout : float
        Dropout probability between layers (default: 0.1).
    feat_dropout : float
        Feature-wise dropout probability (default: 0.0). Zeroes entire
        feature columns during training.
    """

    def __init__(
        self,
        in_dim: int = 6,
        out_dim: int = 64,
        hidden_dim: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        feat_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        h = hidden_dim if hidden_dim is not None else 64

        layers: list[nn.Module] = [
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, h),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for _ in range(n_layers - 2):
            layers += [
                nn.Linear(h, h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]

        layers += [
            nn.Linear(h, out_dim),
            nn.LayerNorm(out_dim),
        ]

        self.mlp = nn.Sequential(*layers)
        self.feat_drop = nn.Dropout(feat_dropout) if feat_dropout > 0 else None

    def forward(self, struct_features: torch.Tensor) -> torch.Tensor:
        """Encode scalar structural features.

        Parameters
        ----------
        struct_features : Tensor [B, in_dim]

        Returns
        -------
        Tensor [B, out_dim]
        """
        # Truncate if input has more features than expected (backward compat)
        if struct_features.shape[1] > self.in_dim:
            struct_features = struct_features[:, :self.in_dim]
        if self.feat_drop is not None:
            struct_features = self.feat_drop(struct_features)
        return self.mlp(struct_features)
