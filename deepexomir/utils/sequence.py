"""RNA sequence manipulation utilities."""

from __future__ import annotations

import re

from deepexomir.utils.constants import (
    BP_GAP,
    BP_MISMATCH,
    BP_SCORES,
    COMPLEMENT_RNA,
    DNA_TO_RNA,
    SEED_END,
    SEED_START,
)

import numpy as np


def dna_to_rna(seq: str) -> str:
    """Convert DNA sequence to RNA (T -> U)."""
    return seq.translate(DNA_TO_RNA)


def reverse_complement_rna(seq: str) -> str:
    """Get reverse complement of an RNA sequence."""
    return seq.translate(COMPLEMENT_RNA)[::-1]


def clean_sequence(seq: str) -> str:
    """Clean and validate an RNA/DNA sequence."""
    seq = seq.strip().upper()
    seq = dna_to_rna(seq)
    seq = re.sub(r"[^AUGCN]", "", seq)
    return seq


def pad_sequence(seq: str, max_len: int, pad_char: str = "N") -> str:
    """Pad sequence to max_len with pad_char."""
    if len(seq) >= max_len:
        return seq[:max_len]
    return seq + pad_char * (max_len - len(seq))


def one_hot_encode(seq: str, max_len: int = 0) -> np.ndarray:
    """One-hot encode an RNA sequence.

    Encoding: A=[1,0,0,0], U=[0,1,0,0], G=[0,0,1,0], C=[0,0,0,1], N=[0,0,0,0]
    Returns shape [seq_len, 4] or [max_len, 4] if max_len > 0.
    """
    mapping = {"A": 0, "U": 1, "G": 2, "C": 3}
    if max_len > 0:
        seq = pad_sequence(seq, max_len)
    encoded = np.zeros((len(seq), 4), dtype=np.float32)
    for i, base in enumerate(seq):
        if base in mapping:
            encoded[i, mapping[base]] = 1.0
    return encoded


def compute_gc_content(seq: str) -> float:
    """Compute GC content of a sequence."""
    if not seq:
        return 0.0
    gc_count = sum(1 for base in seq.upper() if base in ("G", "C"))
    return gc_count / len(seq)


# Vectorized base-pairing lookup table: A=0, U=1, G=2, C=3, N=4, PAD=5
_BP_BASE_MAP = {ord("A"): 0, ord("U"): 1, ord("G"): 2, ord("C"): 3, ord("N"): 4}
_BP_SCORE_TABLE = np.full((6, 6), BP_MISMATCH, dtype=np.float32)
# Watson-Crick pairs
_BP_SCORE_TABLE[0, 1] = 1.0  # A-U
_BP_SCORE_TABLE[1, 0] = 1.0  # U-A
_BP_SCORE_TABLE[2, 3] = 1.0  # G-C
_BP_SCORE_TABLE[3, 2] = 1.0  # C-G
# Wobble pairs
_BP_SCORE_TABLE[2, 1] = 0.5  # G-U
_BP_SCORE_TABLE[1, 2] = 0.5  # U-G
# N or PAD
_BP_SCORE_TABLE[4, :] = BP_GAP
_BP_SCORE_TABLE[:, 4] = BP_GAP
_BP_SCORE_TABLE[5, :] = BP_GAP
_BP_SCORE_TABLE[:, 5] = BP_GAP


def _seq_to_ids(seq: str, max_len: int) -> np.ndarray:
    """Convert RNA sequence to integer array using _BP_BASE_MAP."""
    ids = np.full(max_len, 5, dtype=np.int8)  # 5 = PAD
    for i in range(min(len(seq), max_len)):
        ids[i] = _BP_BASE_MAP.get(ord(seq[i]), 4)  # 4 = N
    return ids


def compute_base_pairing_matrix(
    mirna_seq: str, target_seq: str, max_mirna_len: int = 30, max_target_len: int = 40
) -> np.ndarray:
    """Compute base-pairing score matrix between miRNA and target.

    The target is reversed because miRNA binds 3'->5' to the target 5'->3'.
    Uses vectorized numpy indexing for ~100x speedup over Python loops.

    Returns:
        np.ndarray of shape [max_mirna_len, max_target_len]
    """
    mirna_seq = pad_sequence(mirna_seq, max_mirna_len)
    target_seq = pad_sequence(target_seq, max_target_len)

    # Reverse target for canonical alignment (miRNA 5'->3' pairs with target 3'->5')
    target_rev = target_seq[::-1]

    m_ids = _seq_to_ids(mirna_seq, max_mirna_len)
    t_ids = _seq_to_ids(target_rev, max_target_len)

    # Vectorized lookup: [max_mirna_len, max_target_len]
    return _BP_SCORE_TABLE[m_ids[:, None], t_ids[None, :]]


def classify_seed_match(mirna_seq: str, target_seq: str) -> str:
    """Classify the seed match type between miRNA and target.

    Seed region: positions 2-8 of miRNA (index 1-7).
    Target is checked for reverse complement match.

    Returns one of: '8mer', '7mer-m8', '7mer-A1', '6mer', '6mer-GU', 'non-canonical', 'none'
    """
    if len(mirna_seq) < 8 or len(target_seq) < 8:
        return "none"

    mirna_seed = mirna_seq[SEED_START : SEED_END + 1]  # positions 2-8 (7 nt)
    mirna_pos1 = mirna_seq[0] if len(mirna_seq) > 0 else ""

    # Reverse complement of seed for matching
    seed_rc = reverse_complement_rna(mirna_seed)

    # Check if target contains the seed reverse complement
    target_upper = target_seq.upper()

    # Check for exact seed match in target
    if seed_rc in target_upper:
        idx = target_upper.index(seed_rc)
        # Check for adenine opposite position 1 (for 8mer and 7mer-A1)
        has_a_anchor = (idx + len(seed_rc) < len(target_upper) and
                        target_upper[idx + len(seed_rc)] == "A")
        # Check position 8 match
        pos8_rc = reverse_complement_rna(mirna_seq[7])
        has_pos8 = idx > 0 and target_upper[idx - 1] == pos8_rc

        if has_pos8 and has_a_anchor:
            return "8mer"
        elif has_pos8:
            return "7mer-m8"
        elif has_a_anchor:
            return "7mer-A1"
        else:
            return "6mer"

    # Check for GU wobble in seed
    gu_mismatches = 0
    seed_6mer_rc = reverse_complement_rna(mirna_seq[SEED_START : SEED_START + 6])
    for i, (s, t) in enumerate(zip(seed_6mer_rc, target_upper)):
        pair = (mirna_seq[SEED_START + i], t)
        if pair in (("G", "U"), ("U", "G")):
            gu_mismatches += 1

    if gu_mismatches == 1:
        return "6mer-GU"

    # Check for non-canonical (allow 1 mismatch or bulge in seed)
    mismatches = 0
    for i in range(min(6, len(target_upper))):
        expected = reverse_complement_rna(mirna_seq[SEED_START + i])
        if i < len(target_upper) and target_upper[i] != expected:
            mismatches += 1

    if mismatches <= 1:
        return "non-canonical"

    return "none"


def extract_seed_region(mirna_seq: str) -> str:
    """Extract the seed region (positions 2-8) from a miRNA sequence."""
    return mirna_seq[SEED_START : SEED_END + 1]
