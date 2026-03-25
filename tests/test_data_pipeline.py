"""Tests for data pipeline modules.

Covers sequence utilities, one-hot encoding, base-pairing matrix
computation, seed match classification, GC content, and padding.
"""

from __future__ import annotations

import numpy as np
import pytest

from deepexomir.utils.constants import BP_GAP, BP_MISMATCH, BP_SCORES
from deepexomir.utils.sequence import (
    classify_seed_match,
    clean_sequence,
    compute_base_pairing_matrix,
    compute_gc_content,
    one_hot_encode,
    pad_sequence,
    reverse_complement_rna,
)


# ====================================================================
# test_clean_sequence
# ====================================================================


class TestCleanSequence:
    """Verify T->U conversion, uppercasing, and invalid character removal."""

    def test_dna_to_rna_conversion(self):
        """T bases should be converted to U."""
        result = clean_sequence("ATGCTTAG")
        assert "T" not in result
        assert result == "AUGCUUAG"

    def test_uppercase_conversion(self):
        """Lowercase input should be converted to uppercase."""
        result = clean_sequence("augcuuag")
        assert result == "AUGCUUAG"

    def test_invalid_character_removal(self):
        """Non-AUGCN characters should be removed."""
        result = clean_sequence("AUG-XYZ-CUA")
        assert result == "AUGCUA"
        # Only valid RNA bases and N remain
        assert all(c in "AUGCN" for c in result)

    def test_whitespace_stripping(self):
        """Leading and trailing whitespace should be removed."""
        result = clean_sequence("  AUGCUA  ")
        assert result == "AUGCUA"

    def test_empty_sequence(self):
        """Empty input should return empty string."""
        result = clean_sequence("")
        assert result == ""

    def test_n_characters_preserved(self):
        """N (unknown base) should be preserved."""
        result = clean_sequence("AUGNNCG")
        assert result == "AUGNNCG"

    def test_mixed_dna_rna(self):
        """Mixed DNA/RNA sequences should be fully converted to RNA."""
        result = clean_sequence("AtgCuTaG")
        assert result == "AUGCUUAG"


# ====================================================================
# test_one_hot_encode
# ====================================================================


class TestOneHotEncode:
    """Verify encoding shape and values."""

    def test_basic_encoding_shape(self):
        """Encoding should have shape [seq_len, 4]."""
        seq = "AUGC"
        encoded = one_hot_encode(seq)
        assert encoded.shape == (4, 4)

    def test_with_max_len_padding(self):
        """With max_len, output should be padded to [max_len, 4]."""
        seq = "AU"
        encoded = one_hot_encode(seq, max_len=5)
        assert encoded.shape == (5, 4)
        # Padded positions (N) should be all zeros
        assert np.allclose(encoded[2:, :], 0.0)

    def test_encoding_values(self):
        """Verify correct one-hot values for each base."""
        seq = "AUGC"
        encoded = one_hot_encode(seq)
        # A = [1,0,0,0]
        np.testing.assert_array_equal(encoded[0], [1, 0, 0, 0])
        # U = [0,1,0,0]
        np.testing.assert_array_equal(encoded[1], [0, 1, 0, 0])
        # G = [0,0,1,0]
        np.testing.assert_array_equal(encoded[2], [0, 0, 1, 0])
        # C = [0,0,0,1]
        np.testing.assert_array_equal(encoded[3], [0, 0, 0, 1])

    def test_n_base_encoding(self):
        """N should map to all-zeros (no base)."""
        seq = "N"
        encoded = one_hot_encode(seq)
        np.testing.assert_array_equal(encoded[0], [0, 0, 0, 0])

    def test_each_row_sums_to_one_or_zero(self):
        """Each row should sum to exactly 1.0 (known base) or 0.0 (N)."""
        seq = "AUGCN"
        encoded = one_hot_encode(seq)
        row_sums = encoded.sum(axis=1)
        assert row_sums[0] == 1.0  # A
        assert row_sums[1] == 1.0  # U
        assert row_sums[2] == 1.0  # G
        assert row_sums[3] == 1.0  # C
        assert row_sums[4] == 0.0  # N

    def test_dtype_is_float32(self):
        """Output dtype should be float32."""
        encoded = one_hot_encode("AUGC")
        assert encoded.dtype == np.float32


# ====================================================================
# test_base_pairing_matrix
# ====================================================================


class TestBasePairingMatrix:
    """Verify matrix shape [30, 40] and known base-pairing scores."""

    def test_default_shape(self):
        """Matrix should have shape [30, 40] with default lengths."""
        mirna = "AUGCUAGCUA"
        target = "UAGCAUGCAU"
        matrix = compute_base_pairing_matrix(mirna, target)
        assert matrix.shape == (30, 40)

    def test_custom_shape(self):
        """Matrix should respect custom max_mirna_len and max_target_len."""
        mirna = "AUGC"
        target = "GCAU"
        matrix = compute_base_pairing_matrix(mirna, target, max_mirna_len=10, max_target_len=15)
        assert matrix.shape == (10, 15)

    def test_watson_crick_pairs_present(self):
        """Canonical Watson-Crick pairs should score 1.0."""
        # A-U pair should be 1.0
        assert BP_SCORES[("A", "U")] == 1.0
        assert BP_SCORES[("G", "C")] == 1.0

    def test_wobble_pairs_present(self):
        """G-U wobble pairs should score 0.5."""
        assert BP_SCORES[("G", "U")] == 0.5
        assert BP_SCORES[("U", "G")] == 0.5

    def test_mismatch_value(self):
        """Mismatched bases should use BP_MISMATCH score."""
        assert BP_MISMATCH == -1.0

    def test_gap_value(self):
        """N bases should produce BP_GAP score."""
        assert BP_GAP == -0.5

    def test_dtype(self):
        """Matrix dtype should be float32."""
        matrix = compute_base_pairing_matrix("AUGC", "GCAU")
        assert matrix.dtype == np.float32

    def test_known_pairing(self):
        """Verify a specific pairing score in the matrix.

        miRNA "A" at position 0 should pair with the reversed target.
        Target is reversed for canonical alignment.
        """
        # Simple case: miRNA = "A", target = "U" (reversed target = "U")
        # Position (0, 0) after reversal: miRNA[0]='A' vs target_rev[0]
        mirna = "A"
        target = "U"
        matrix = compute_base_pairing_matrix(mirna, target, max_mirna_len=1, max_target_len=1)
        # A-U should be 1.0 (target reversed: "U" reversed is "U")
        assert matrix[0, 0] == 1.0


# ====================================================================
# test_classify_seed_match
# ====================================================================


class TestClassifySeedMatch:
    """Test known 8mer, 7mer, 6mer classification cases."""

    def test_short_sequences_return_none(self):
        """Sequences shorter than 8 nt should return 'none'."""
        assert classify_seed_match("AUGCU", "AUGCU") == "none"

    def test_known_8mer_case(self):
        """A perfect 8mer match with pos8 and A-anchor should return '8mer'."""
        # miR-21-5p seed region (pos 2-8): AGCUUAU
        # For an 8mer, target needs: reverse complement of pos 2-8 + A at anchor + pos 8 match
        mirna = "UAGCUUAUCAGACUGAUGUUGA"
        # Build a target that has the reverse complement of the seed + flanking
        seed = mirna[1:8]  # AGCUUAU
        seed_rc = reverse_complement_rna(seed)  # AUAAGCU
        pos8_rc = reverse_complement_rna(mirna[7])  # RC of U = A
        # 8mer target: pos8_rc + seed_rc + "A"
        target = "UU" + pos8_rc + seed_rc + "A" + "UUUUUUUUU"
        result = classify_seed_match(mirna, target)
        assert result == "8mer"

    def test_6mer_case(self):
        """A seed match without pos8 match or A-anchor should return '6mer'."""
        mirna = "UAGCUUAUCAGACUGAUGUUGA"
        seed = mirna[1:8]
        seed_rc = reverse_complement_rna(seed)
        # No pos8 match (different base), no A anchor (different base)
        target = "UU" + "G" + seed_rc + "G" + "UUUUUUUUU"
        result = classify_seed_match(mirna, target)
        assert result == "6mer"

    def test_no_match(self):
        """Completely unrelated sequences should return 'none' or 'non-canonical'."""
        mirna = "UAGCUUAUCAGACUGAUGUUGA"
        target = "GGGGGGGGGGGGGGGGGGGGGG"
        result = classify_seed_match(mirna, target)
        assert result in ("none", "non-canonical")


# ====================================================================
# test_gc_content
# ====================================================================


class TestGCContent:
    """Verify GC content calculation."""

    def test_all_gc(self):
        """All G/C sequence should have GC content of 1.0."""
        assert compute_gc_content("GGCCGC") == 1.0

    def test_no_gc(self):
        """All A/U sequence should have GC content of 0.0."""
        assert compute_gc_content("AAUUAU") == 0.0

    def test_mixed(self):
        """50% GC sequence should have GC content of 0.5."""
        result = compute_gc_content("AUGC")
        assert abs(result - 0.5) < 1e-6

    def test_empty_sequence(self):
        """Empty sequence should return 0.0."""
        assert compute_gc_content("") == 0.0

    def test_single_base(self):
        """Single G should give GC content of 1.0."""
        assert compute_gc_content("G") == 1.0
        assert compute_gc_content("A") == 0.0

    def test_precision(self):
        """Verify GC content with known ratio."""
        # 2 G/C out of 6 bases = 1/3
        result = compute_gc_content("AAGCUU")
        assert abs(result - 2.0 / 6.0) < 1e-6


# ====================================================================
# test_pad_sequence
# ====================================================================


class TestPadSequence:
    """Verify padding and truncation."""

    def test_padding_short_sequence(self):
        """Short sequence should be padded to max_len."""
        result = pad_sequence("AUGC", 10)
        assert len(result) == 10
        assert result == "AUGCNNNNNN"

    def test_truncation_long_sequence(self):
        """Long sequence should be truncated to max_len."""
        result = pad_sequence("AUGCUAGCUAGCUA", 5)
        assert len(result) == 5
        assert result == "AUGCU"

    def test_exact_length(self):
        """Sequence of exact length should be unchanged."""
        result = pad_sequence("AUGC", 4)
        assert result == "AUGC"

    def test_custom_pad_char(self):
        """Custom pad character should be used."""
        result = pad_sequence("AU", 5, pad_char="X")
        assert result == "AUXXX"

    def test_empty_sequence_padding(self):
        """Empty sequence should be fully padded."""
        result = pad_sequence("", 5)
        assert result == "NNNNN"
        assert len(result) == 5
