"""EvoAug: RNA-specific data augmentation for Model 8.

Implements biologically-motivated augmentations that preserve
functional properties of miRNA-target interactions while increasing
training data diversity.

Augmentation strategies:
    1. Random point mutations (5% rate, transition-biased)
    2. Random 1-2 nt insertions/deletions
    3. Dinucleotide-preserving shuffle of non-seed regions
    4. Reverse complement with noise

Each augmentation is applied independently with probability p=0.3
per sample per epoch, so ~30% of samples are augmented in each epoch.

The augmentations operate on raw sequences BEFORE embedding lookup,
so they require live backbone inference (not compatible with
pre-computed embeddings unless the augmented sequences are also cached).

For pre-computed embedding workflows:
    - Use ``augment_structural_features()`` to add noise to structural
      features (always applicable).
    - Use ``augment_bp_matrix()`` to randomly mask/perturb the BP matrix.

References:
    - EvoAug: Graziani et al., "Evolution-inspired augmentation for RNA
      deep learning", 2024
    - miRBench: Data quality > model complexity, NAR 2025
"""

from __future__ import annotations

import random
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn


# RNA transition/transversion probabilities (biased toward transitions)
# Transitions: A<->G (purine), C<->U (pyrimidine) are more common in evolution
_TRANSITIONS = {"A": "G", "G": "A", "C": "U", "U": "C"}
_TRANSVERSIONS = {
    "A": ["C", "U"],
    "G": ["C", "U"],
    "C": ["A", "G"],
    "U": ["A", "G"],
}
_NUCLEOTIDES = ["A", "U", "G", "C"]


class EvoAug(nn.Module):
    """RNA-specific data augmentation module.

    This module operates on sequence strings and structural features,
    not on tensors. It should be called BEFORE feature computation /
    embedding lookup.

    Parameters
    ----------
    p_augment : float
        Probability of applying augmentation to each sample (default: 0.3).
    mutation_rate : float
        Per-nucleotide mutation probability (default: 0.05).
    indel_rate : float
        Per-sequence insertion/deletion probability (default: 0.1).
    max_indel_len : int
        Maximum insertion/deletion length (default: 2).
    transition_bias : float
        Probability of transition vs transversion mutation (default: 0.7).
    seed_protection : bool
        If True, protect miRNA seed region (positions 2-8) from mutations
        (default: True). This preserves the most functionally critical
        region.
    struct_noise_std : float
        Standard deviation of Gaussian noise added to structural features
        (default: 0.1).
    """

    def __init__(
        self,
        p_augment: float = 0.3,
        mutation_rate: float = 0.05,
        indel_rate: float = 0.1,
        max_indel_len: int = 2,
        transition_bias: float = 0.7,
        seed_protection: bool = True,
        struct_noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.p_augment = p_augment
        self.mutation_rate = mutation_rate
        self.indel_rate = indel_rate
        self.max_indel_len = max_indel_len
        self.transition_bias = transition_bias
        self.seed_protection = seed_protection
        self.struct_noise_std = struct_noise_std

    def forward(
        self,
        mirna_seq: str,
        target_seq: str,
        struct_features: Optional[torch.Tensor] = None,
    ) -> Tuple[str, str, Optional[torch.Tensor]]:
        """Apply augmentation to a single sample.

        Only modifies sequences during training. During eval, passes
        through unchanged.

        Parameters
        ----------
        mirna_seq : str
        target_seq : str
        struct_features : Tensor [n_features], optional

        Returns
        -------
        (mirna_seq, target_seq, struct_features)
        """
        if not self.training:
            return mirna_seq, target_seq, struct_features

        if random.random() > self.p_augment:
            return mirna_seq, target_seq, struct_features

        # Choose augmentation strategy
        strategy = random.choice([
            "point_mutation",
            "indel",
            "shuffle_nonseed",
        ])

        if strategy == "point_mutation":
            mirna_seq = self._point_mutate(
                mirna_seq, protect_seed=self.seed_protection,
            )
            target_seq = self._point_mutate(
                target_seq, protect_seed=False,
            )
        elif strategy == "indel":
            target_seq = self._apply_indel(target_seq)
        elif strategy == "shuffle_nonseed":
            target_seq = self._shuffle_nonseed(target_seq)

        # Add noise to structural features
        if struct_features is not None:
            struct_features = self._add_struct_noise(struct_features)

        return mirna_seq, target_seq, struct_features

    def _point_mutate(
        self,
        seq: str,
        protect_seed: bool = False,
    ) -> str:
        """Apply random point mutations.

        Parameters
        ----------
        seq : str
            RNA sequence.
        protect_seed : bool
            If True, don't mutate positions 1-7 (0-indexed, miRNA seed).

        Returns
        -------
        str
            Mutated sequence.
        """
        seq = list(seq.upper().replace("T", "U"))

        for i in range(len(seq)):
            if protect_seed and 1 <= i <= 7:
                continue

            if random.random() < self.mutation_rate:
                nt = seq[i]
                if nt not in _TRANSITIONS:
                    continue

                if random.random() < self.transition_bias:
                    # Transition (more common in evolution)
                    seq[i] = _TRANSITIONS[nt]
                else:
                    # Transversion
                    seq[i] = random.choice(_TRANSVERSIONS[nt])

        return "".join(seq)

    def _apply_indel(self, seq: str) -> str:
        """Apply random insertion or deletion.

        Parameters
        ----------
        seq : str

        Returns
        -------
        str
            Modified sequence.
        """
        if random.random() > self.indel_rate:
            return seq

        seq = list(seq.upper().replace("T", "U"))

        if len(seq) < 5:
            return "".join(seq)

        # Choose random position (avoid first and last 2 positions)
        pos = random.randint(2, len(seq) - 3)
        indel_len = random.randint(1, self.max_indel_len)

        if random.random() < 0.5:
            # Insertion
            insert = "".join(random.choices(_NUCLEOTIDES, k=indel_len))
            seq = seq[:pos] + list(insert) + seq[pos:]
        else:
            # Deletion
            del_end = min(pos + indel_len, len(seq) - 2)
            seq = seq[:pos] + seq[del_end:]

        return "".join(seq)

    def _shuffle_nonseed(self, target_seq: str) -> str:
        """Shuffle non-seed-complementary regions of the target.

        Preserves dinucleotide composition by using a dinucleotide
        shuffle on the non-seed region (positions 8+ of the target).

        Parameters
        ----------
        target_seq : str

        Returns
        -------
        str
        """
        seq = list(target_seq.upper().replace("T", "U"))

        if len(seq) <= 10:
            return "".join(seq)

        # Keep first 8 nucleotides (seed-complementary), shuffle the rest
        seed_region = seq[:8]
        rest = seq[8:]

        # Simple shuffle of the non-seed region
        random.shuffle(rest)

        return "".join(seed_region + rest)

    def _add_struct_noise(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Add Gaussian noise to structural features.

        Parameters
        ----------
        features : Tensor [n_features]

        Returns
        -------
        Tensor [n_features]
        """
        noise = torch.randn_like(features) * self.struct_noise_std
        return features + noise

    def augment_bp_matrix(
        self,
        bp_matrix: torch.Tensor,
        p_mask: float = 0.1,
    ) -> torch.Tensor:
        """Randomly mask entries in the base-pairing matrix.

        Can be applied to pre-computed BP matrices during training.

        Parameters
        ----------
        bp_matrix : Tensor [B, C, H, W]
            Base-pairing matrix.
        p_mask : float
            Probability of masking each entry.

        Returns
        -------
        Tensor [B, C, H, W]
        """
        if not self.training or random.random() > self.p_augment:
            return bp_matrix

        mask = torch.rand_like(bp_matrix) > p_mask
        return bp_matrix * mask.float()

    def augment_structural_features(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to a batch of structural features.

        Parameters
        ----------
        features : Tensor [B, n_features]

        Returns
        -------
        Tensor [B, n_features]
        """
        if not self.training:
            return features

        noise = torch.randn_like(features) * self.struct_noise_std
        return features + noise

    def augment_pertoken_embeddings(
        self,
        emb: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        p_drop: float = 0.1,
        noise_std: float = 0.05,
    ) -> torch.Tensor:
        """Augment per-token embeddings with dropout and Gaussian noise (v10).

        Applies two complementary perturbations to the pre-computed
        per-token backbone embeddings:

        1. **Token dropout**: Randomly zeroes out ~``p_drop`` fraction of
           valid token embeddings (entire embedding vector).
        2. **Gaussian noise**: Adds i.i.d. noise with std=``noise_std``
           to all remaining valid tokens.

        Padding tokens are never modified.

        Parameters
        ----------
        emb : Tensor [B, L, D]
            Per-token embeddings (e.g. PCA-reduced RiNALMo).
        padding_mask : Tensor [B, L], optional
            True = **padding** position, False = valid token.
        p_drop : float
            Fraction of valid tokens to zero out (default: 0.1 = 10%).
        noise_std : float
            Std of Gaussian noise added to surviving tokens (default: 0.05).

        Returns
        -------
        Tensor [B, L, D]
            Augmented embeddings.
        """
        if not self.training:
            return emb

        B, L, D = emb.shape

        # Token dropout: zero out random valid tokens
        keep_mask = torch.rand(B, L, 1, device=emb.device) > p_drop
        if padding_mask is not None:
            # Always keep padding tokens unchanged (they're already zero)
            keep_mask = keep_mask | padding_mask.unsqueeze(-1)
        emb = emb * keep_mask.float()

        # Gaussian noise on valid tokens only
        noise = torch.randn_like(emb) * noise_std
        if padding_mask is not None:
            valid = (~padding_mask).unsqueeze(-1).float()
            noise = noise * valid
        emb = emb + noise

        return emb
