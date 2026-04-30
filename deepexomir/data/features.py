"""Structural feature extraction for DeepExoMir miRNA-target interactions.

Computes thermodynamic and structural features used as auxiliary inputs
to the DeepExoMir model.  Optionally uses ViennaRNA (``RNA`` module) for
minimum free energy (MFE) calculations.  Falls back gracefully to
heuristic estimates if ViennaRNA is not installed.

Features computed (Model ⑦: 8 features)
-----------------------------------------
- duplex_mfe:       minimum free energy of the miRNA:target duplex
- mirna_mfe:        MFE of the miRNA secondary structure
- target_mfe:       MFE of the target-site secondary structure
- accessibility:    target_mfe - duplex_mfe (proxy for ΔG_open)
- gc_content:       GC fraction of the combined miRNA + target
- seed_match_type:  encoded integer for seed-match category
- au_content:       AU fraction of the target site (v7)
- seed_duplex_mfe:  MFE of seed region (nt 2-8) duplex only (v7)

The base-pairing matrix encodes Watson-Crick and wobble pair scores
on a [max_mirna_len x max_target_len] grid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from deepexomir.utils.constants import SEED_TYPES
from deepexomir.utils.sequence import (
    classify_seed_match,
    clean_sequence,
    compute_base_pairing_matrix as _bp_matrix,
    compute_gc_content,
    pad_sequence,
    reverse_complement_rna,
)

logger = logging.getLogger(__name__)

# ============================================================================
# ViennaRNA optional import
# ============================================================================

_VIENNA_AVAILABLE = False
_RNA = None

try:
    import RNA as _RNA  # type: ignore[import-untyped]

    _VIENNA_AVAILABLE = True
    logger.debug("ViennaRNA (RNA module) is available.")
except ImportError:
    logger.info(
        "ViennaRNA (RNA module) not found.  MFE calculations will use "
        "heuristic estimates.  Install ViennaRNA for accurate thermodynamics."
    )

# Seed-match type to integer encoding
SEED_TYPE_ENCODING: dict[str, int] = {st: i for i, st in enumerate(SEED_TYPES)}

# Default dimensions
DEFAULT_MAX_MIRNA_LEN = 30
DEFAULT_MAX_TARGET_LEN = 50

# Heuristic MFE parameters (rough approximation per base pair)
_HEURISTIC_AU_ENERGY = -0.9   # kcal/mol per A-U pair
_HEURISTIC_GC_ENERGY = -1.8   # kcal/mol per G-C pair
_HEURISTIC_GU_ENERGY = -0.5   # kcal/mol per G-U wobble


# ============================================================================
# ViennaRNA wrappers
# ============================================================================


def _vienna_fold_mfe(sequence: str) -> float:
    """Compute MFE of a single RNA sequence using ViennaRNA.

    Parameters
    ----------
    sequence : str
        RNA sequence (uppercase, AUGC only).

    Returns
    -------
    float
        Minimum free energy in kcal/mol.
    """
    if _RNA is None:
        raise RuntimeError("ViennaRNA is not available.")
    sequence = clean_sequence(sequence)
    _, mfe = _RNA.fold(sequence)
    return float(mfe)


def _vienna_duplex_mfe(mirna_seq: str, target_seq: str) -> float:
    """Compute MFE of miRNA:target duplex using ViennaRNA.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence.
    target_seq : str
        Target-site sequence.

    Returns
    -------
    float
        Duplex MFE in kcal/mol.
    """
    if _RNA is None:
        raise RuntimeError("ViennaRNA is not available.")
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    result = _RNA.duplexfold(mirna_seq, target_seq)
    return float(result.energy)


# ============================================================================
# Heuristic MFE fallbacks
# ============================================================================


def _heuristic_fold_mfe(sequence: str) -> float:
    """Estimate MFE using a simple base-pair counting heuristic.

    This is a rough approximation that counts self-complementary pairs
    in the sequence.  It should only be used when ViennaRNA is
    unavailable.

    Parameters
    ----------
    sequence : str
        RNA sequence.

    Returns
    -------
    float
        Estimated MFE in kcal/mol.
    """
    sequence = clean_sequence(sequence)
    rc = reverse_complement_rna(sequence)
    energy = 0.0
    matched = 0

    # Slide a window to find complementary stretches
    n = len(sequence)
    for i in range(n):
        for j in range(i + 4, n):  # minimum loop size of 3
            pair = (sequence[i], sequence[j])
            if pair in (("A", "U"), ("U", "A")):
                energy += _HEURISTIC_AU_ENERGY * 0.3  # discounted for folding
                matched += 1
            elif pair in (("G", "C"), ("C", "G")):
                energy += _HEURISTIC_GC_ENERGY * 0.3
                matched += 1
            elif pair in (("G", "U"), ("U", "G")):
                energy += _HEURISTIC_GU_ENERGY * 0.3
                matched += 1
            if matched >= n // 3:
                break
        if matched >= n // 3:
            break

    return round(energy, 2)


def _heuristic_duplex_mfe(mirna_seq: str, target_seq: str) -> float:
    """Estimate duplex MFE using base-pair counting.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence.
    target_seq : str
        Target-site sequence.

    Returns
    -------
    float
        Estimated duplex MFE in kcal/mol.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    target_rc = reverse_complement_rna(target_seq)

    energy = 0.0
    n = min(len(mirna_seq), len(target_rc))

    for i in range(n):
        m_base = mirna_seq[i]
        t_base = target_rc[i] if i < len(target_rc) else "N"
        pair = (m_base, t_base)

        if pair in (("A", "U"), ("U", "A")):
            energy += _HEURISTIC_AU_ENERGY
        elif pair in (("G", "C"), ("C", "G")):
            energy += _HEURISTIC_GC_ENERGY
        elif pair in (("G", "U"), ("U", "G")):
            energy += _HEURISTIC_GU_ENERGY

    return round(energy, 2)


# ============================================================================
# Public API
# ============================================================================


def compute_mfe(sequence: str) -> float:
    """Compute MFE of an RNA sequence, using ViennaRNA if available.

    Parameters
    ----------
    sequence : str
        RNA sequence.

    Returns
    -------
    float
        MFE in kcal/mol.
    """
    if _VIENNA_AVAILABLE:
        try:
            return _vienna_fold_mfe(sequence)
        except Exception as exc:
            logger.warning("ViennaRNA fold failed, using heuristic: %s", exc)
    return _heuristic_fold_mfe(sequence)


def compute_duplex_mfe(mirna_seq: str, target_seq: str) -> float:
    """Compute duplex MFE, using ViennaRNA if available.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence.
    target_seq : str
        Target-site sequence.

    Returns
    -------
    float
        Duplex MFE in kcal/mol.
    """
    if _VIENNA_AVAILABLE:
        try:
            return _vienna_duplex_mfe(mirna_seq, target_seq)
        except Exception as exc:
            logger.warning("ViennaRNA duplexfold failed, using heuristic: %s", exc)
    return _heuristic_duplex_mfe(mirna_seq, target_seq)


def compute_plfold_accessibility(
    target_seq: str,
    seed_start: int | None = None,
    seed_len: int = 7,
    window: int = 50,
    max_span: int = 30,
) -> tuple[float, float]:
    """Compute RNAplfold-based accessibility for a target sequence.

    Uses ViennaRNA ``pfl_fold_up`` to compute the probability that each
    nucleotide is unpaired in the local folding ensemble.

    Parameters
    ----------
    target_seq : str
        Target-site RNA sequence.
    seed_start : int or None
        1-indexed start position of the seed binding region on the target.
        If ``None``, defaults to ``len(target_seq) - seed_len + 1``
        (last ``seed_len`` nucleotides, standard 3'-end seed site).
    seed_len : int
        Length of the seed region (default 7).
    window : int
        RNAplfold window size (default 50).
    max_span : int
        Maximum base-pair span (default 30).

    Returns
    -------
    tuple[float, float]
        ``(seed_accessibility, site_accessibility)`` where:
        - ``seed_accessibility`` = mean single-nucleotide unpaired probability
          over the seed binding region.
        - ``site_accessibility`` = mean single-nucleotide unpaired probability
          over the entire target sequence.
    """
    if not _VIENNA_AVAILABLE or _RNA is None:
        return (0.5, 0.5)  # neutral default

    target_seq = clean_sequence(target_seq)
    n = len(target_seq)
    if n < 4:
        return (0.5, 0.5)

    try:
        # pfl_fold_up(seq, max_u, window, max_span) -> tuple of (n+1) tuples
        # up[i][u] = prob that u consecutive bases ending at pos i are unpaired
        # Position is 1-indexed; up[0] is a dummy.
        up = _RNA.pfl_fold_up(target_seq, 1, window, max_span)

        # Site accessibility: mean single-nt unpaired prob over all positions
        site_probs = [up[i][1] for i in range(1, n + 1)]
        site_acc = float(np.mean(site_probs)) if site_probs else 0.5

        # Seed accessibility: mean single-nt unpaired prob over seed region
        if seed_start is None:
            seed_start = max(1, n - seed_len + 1)
        seed_end = min(n, seed_start + seed_len - 1)
        seed_probs = [up[i][1] for i in range(seed_start, seed_end + 1)]
        seed_acc = float(np.mean(seed_probs)) if seed_probs else 0.5

        return (seed_acc, site_acc)
    except Exception as exc:
        logger.warning("pfl_fold_up failed: %s", exc)
        return (0.5, 0.5)


def compute_supp_3prime_score(mirna_seq: str, target_seq: str) -> float:
    """Compute 3' supplementary pairing score (miRNA positions 13-16).

    Quantifies the Watson-Crick and wobble base-pairing at miRNA
    positions 13--16, which provide compensatory or supplementary
    binding that strengthens target recognition.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence.
    target_seq : str
        Target-site sequence.

    Returns
    -------
    float
        Supplementary pairing score in [0, 4].  Each of the 4 positions
        contributes 1.0 for a WC pair, 0.5 for a G:U wobble, 0.0 otherwise.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)

    _WC_WOBBLE: dict[tuple[str, str], float] = {
        ("A", "U"): 1.0, ("U", "A"): 1.0,
        ("G", "C"): 1.0, ("C", "G"): 1.0,
        ("G", "U"): 0.5, ("U", "G"): 0.5,
    }

    score = 0.0
    n_target = len(target_seq)
    # miRNA nt i (1-indexed) aligns anti-parallel with target nt (n_target - i + 1)
    for mirna_pos_1based in range(13, 17):  # positions 13, 14, 15, 16
        mirna_idx = mirna_pos_1based - 1  # 0-indexed
        target_idx = n_target - mirna_pos_1based  # 0-indexed anti-parallel

        if mirna_idx >= len(mirna_seq) or target_idx < 0 or target_idx >= n_target:
            continue

        pair = (mirna_seq[mirna_idx], target_seq[target_idx])
        score += _WC_WOBBLE.get(pair, 0.0)

    return score


def compute_local_au_flanking(
    target_seq: str, seed_len: int = 8, flank_len: int = 10,
) -> float:
    """Compute AU content of the flanking region around the seed site.

    The seed binding region is assumed to be the last ``seed_len``
    nucleotides of the target.  This function computes the AU fraction
    in the ``flank_len`` nucleotides immediately upstream (5') of the
    seed site, which influences site accessibility.

    Parameters
    ----------
    target_seq : str
        Target-site sequence.
    seed_len : int
        Number of nucleotides in the seed region (default 8).
    flank_len : int
        Number of flanking nucleotides to consider (default 10).

    Returns
    -------
    float
        AU fraction in [0, 1] of the flanking region.
    """
    target_seq = clean_sequence(target_seq)
    n = len(target_seq)

    # Seed is at the 3' end: positions [n-seed_len, n)
    # Flanking region: [n-seed_len-flank_len, n-seed_len)
    flank_start = max(0, n - seed_len - flank_len)
    flank_end = max(0, n - seed_len)

    flank_region = target_seq[flank_start:flank_end]
    if not flank_region:
        return 0.5  # neutral default

    au_count = flank_region.count("A") + flank_region.count("U")
    return au_count / len(flank_region)


def compute_au_content(sequence: str) -> float:
    """Compute AU (adenine + uracil) fraction of an RNA sequence.

    Parameters
    ----------
    sequence : str
        RNA sequence.

    Returns
    -------
    float
        Fraction of A+U nucleotides in [0, 1].
    """
    if not sequence:
        return 0.0
    seq = sequence.upper()
    au_count = seq.count("A") + seq.count("U")
    return au_count / len(seq)


def compute_seed_pairing_stability(mirna_seq: str, target_seq: str) -> float:
    """Compute Seed Pairing Stability (SPS) score.

    SPS approximates the nearest-neighbor stacking energy contribution
    of each consecutive base-pair step in the seed region (miRNA pos 2-8).
    Unlike ``seed_duplex_mfe`` which captures total duplex energy, SPS
    weights individual stacking interactions using simplified Turner 2004
    parameters, matching TargetScan's context++ formulation.

    Parameters
    ----------
    mirna_seq, target_seq : str
        miRNA and target sequences.

    Returns
    -------
    float
        SPS in kcal/mol (negative = more stable). Range ~[-15, 0].
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    n_target = len(target_seq)
    if len(mirna_seq) < 8 or n_target < 8:
        return 0.0

    # Simplified nearest-neighbor stacking energies (kcal/mol, Turner 2004)
    # Key: (5'->3' pair step on one strand) -> energy
    _STACK_ENERGIES: dict[tuple[str, str, str, str], float] = {
        # (mirna_i, mirna_i+1, target_j, target_j-1) -> dG
        # AU/AU stack
        ("A", "A", "U", "U"): -0.9, ("U", "U", "A", "A"): -0.9,
        ("A", "U", "U", "A"): -1.1, ("U", "A", "A", "U"): -1.3,
        # GC/GC stack
        ("G", "G", "C", "C"): -3.3, ("C", "C", "G", "G"): -3.3,
        ("G", "C", "C", "G"): -2.4, ("C", "G", "G", "C"): -3.4,
        # AU/GC mixed
        ("A", "G", "U", "C"): -2.1, ("G", "A", "C", "U"): -2.1,
        ("A", "C", "U", "G"): -2.2, ("C", "A", "G", "U"): -2.1,
        ("U", "G", "A", "C"): -1.4, ("G", "U", "C", "A"): -2.1,
        ("U", "C", "A", "G"): -2.1, ("C", "U", "G", "A"): -1.4,
        # GU wobble stacks (subset)
        ("G", "G", "U", "C"): -1.5, ("G", "U", "U", "A"): -0.5,
        ("U", "G", "A", "U"): -0.5, ("A", "G", "U", "U"): -0.5,
    }
    _DEFAULT_STACK = -1.0  # fallback for unlisted pairs

    sps = 0.0
    for i in range(1, 7):  # seed positions 2-8 (0-indexed: 1-7), 6 stacking steps
        mi, mi1 = mirna_seq[i], mirna_seq[i + 1]
        # anti-parallel: mirna pos i+1 (1-based) -> target pos (n_target - i - 1)
        tj = n_target - i - 1
        tj1 = n_target - i - 2
        if tj < 0 or tj1 < 0 or tj >= n_target or tj1 >= n_target:
            continue
        ti, ti1 = target_seq[tj], target_seq[tj1]
        sps += _STACK_ENERGIES.get((mi, mi1, ti, ti1), _DEFAULT_STACK)

    return max(-15.0, min(0.0, sps))


def compute_comp_3prime_score(mirna_seq: str, target_seq: str) -> float:
    """Compute 3' compensatory pairing score (miRNA positions 17-21).

    Extends ``compute_supp_3prime_score`` (positions 13-16) by scoring
    the compensatory pairing region at miRNA positions 17-21.  TargetScan
    distinguishes supplementary (13-16) from compensatory (17-21) pairing.

    Returns
    -------
    float
        Score in [0, 5]. Each position: 1.0 WC, 0.5 wobble, 0.0 mismatch.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    _WC_WOBBLE = {
        ("A", "U"): 1.0, ("U", "A"): 1.0,
        ("G", "C"): 1.0, ("C", "G"): 1.0,
        ("G", "U"): 0.5, ("U", "G"): 0.5,
    }
    score = 0.0
    n_target = len(target_seq)
    for pos in range(17, 22):  # 17, 18, 19, 20, 21
        mi = pos - 1
        ti = n_target - pos
        if mi >= len(mirna_seq) or ti < 0 or ti >= n_target:
            continue
        score += _WC_WOBBLE.get((mirna_seq[mi], target_seq[ti]), 0.0)
    return score


def compute_central_pairing(mirna_seq: str, target_seq: str) -> float:
    """Compute central region pairing score (miRNA positions 9-12).

    The central region is important for TDMD (target-directed miRNA
    degradation) and cleavage-competent binding.

    Returns
    -------
    float
        Score in [0, 4]. Each position: 1.0 WC, 0.5 wobble, 0.0 mismatch.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    _WC_WOBBLE = {
        ("A", "U"): 1.0, ("U", "A"): 1.0,
        ("G", "C"): 1.0, ("C", "G"): 1.0,
        ("G", "U"): 0.5, ("U", "G"): 0.5,
    }
    score = 0.0
    n_target = len(target_seq)
    for pos in range(9, 13):  # 9, 10, 11, 12
        mi = pos - 1
        ti = n_target - pos
        if mi >= len(mirna_seq) or ti < 0 or ti >= n_target:
            continue
        score += _WC_WOBBLE.get((mirna_seq[mi], target_seq[ti]), 0.0)
    return score


def compute_duplex_pairing_stats(
    mirna_seq: str, target_seq: str,
) -> tuple[int, int, int, int]:
    """Compute full-duplex pairing statistics.

    Scans the anti-parallel alignment and counts WC pairs, wobble pairs,
    mismatches, and the longest contiguous complementary stretch.

    Returns
    -------
    tuple[int, int, int, int]
        ``(wobble_count, mismatch_count, longest_contig, seed_gc_count)``
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    n_mirna = len(mirna_seq)
    n_target = len(target_seq)

    _WC = {("A", "U"), ("U", "A"), ("G", "C"), ("C", "G")}
    _WOBBLE = {("G", "U"), ("U", "G")}
    _GC = {"G", "C"}

    wobble_count = 0
    mismatch_count = 0
    current_run = 0
    longest_contig = 0
    seed_gc_count = 0

    n_align = min(n_mirna, n_target)
    for pos_1 in range(1, n_align + 1):  # 1-indexed miRNA position
        mi = pos_1 - 1
        ti = n_target - pos_1
        if ti < 0:
            break
        pair = (mirna_seq[mi], target_seq[ti])
        if pair in _WC or pair in _WOBBLE:
            current_run += 1
            longest_contig = max(longest_contig, current_run)
            if pair in _WOBBLE:
                wobble_count += 1
        else:
            mismatch_count += 1
            current_run = 0

        # Seed GC: positions 2-8
        if 2 <= pos_1 <= 8:
            if mirna_seq[mi] in _GC:
                seed_gc_count += 1
            if target_seq[ti] in _GC:
                seed_gc_count += 1

    return wobble_count, mismatch_count, longest_contig, seed_gc_count


def compute_position_identity_features(
    mirna_seq: str, target_seq: str,
) -> dict[str, float]:
    """Compute nucleotide identity features at key positions.

    Returns one-hot-style features for miRNA position 1, position 8,
    and target position opposite miRNA pos 8 (the A1 anchor).
    Also computes a flanking dinucleotide affinity score based on
    McGeary et al. (Science 2019).

    Parameters
    ----------
    mirna_seq, target_seq : str
        miRNA and target sequences.

    Returns
    -------
    dict with keys:
        sRNA1_A, sRNA1_C, sRNA1_G  (miRNA pos-1 identity, one-hot)
        sRNA8_A, sRNA8_C, sRNA8_G  (miRNA pos-8 identity, one-hot)
        site8_A                     (1.0 if target opposite pos-8 is A)
        flanking_dinuc_score        (AU-richness of 2nt flanking seed match)
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    n_target = len(target_seq)

    # -- miRNA position 1 identity (one-hot, U is baseline) --
    pos1 = mirna_seq[0] if len(mirna_seq) >= 1 else "N"
    sRNA1_A = 1.0 if pos1 == "A" else 0.0
    sRNA1_C = 1.0 if pos1 == "C" else 0.0
    sRNA1_G = 1.0 if pos1 == "G" else 0.0

    # -- miRNA position 8 identity (one-hot, U is baseline) --
    pos8 = mirna_seq[7] if len(mirna_seq) >= 8 else "N"
    sRNA8_A = 1.0 if pos8 == "A" else 0.0
    sRNA8_C = 1.0 if pos8 == "C" else 0.0
    sRNA8_G = 1.0 if pos8 == "G" else 0.0

    # -- Target A1 anchor: nucleotide opposite miRNA pos 8 --
    # Anti-parallel: miRNA pos 8 (1-indexed) -> target pos (n_target - 8)
    t8_idx = n_target - 8
    site8_A = 0.0
    if 0 <= t8_idx < n_target:
        site8_A = 1.0 if target_seq[t8_idx] == "A" else 0.0

    # -- Flanking dinucleotide AU-richness score --
    # McGeary 2019: 2nt upstream + 2nt downstream of the 8-nt seed match site
    # Seed match occupies target positions [n_target-8, n_target)
    # Upstream flank: [n_target-10, n_target-8)
    # Downstream flank: [n_target, n_target+2) -- but these don't exist in 50nt
    # So we use the available flanking nucleotides
    au_chars = {"A", "U"}
    flank_au = 0
    flank_total = 0

    # 2nt upstream of seed match
    for offset in [n_target - 9, n_target - 10]:
        if 0 <= offset < n_target:
            flank_total += 1
            if target_seq[offset] in au_chars:
                flank_au += 1

    # 2nt downstream of seed match (positions 0,1 since target is reversed)
    # Actually in anti-parallel: downstream of seed on target = positions
    # after the seed match region. Seed match is at 3' end of target,
    # so "downstream" (toward 3') doesn't exist. Use positions at 5' end
    # of the seed region boundary: indices n_target-1 and beyond (doesn't exist)
    # In practice for 50nt target with seed at 3' end, downstream = doesn't exist
    # So just use upstream 2nt as the available flanking context
    # Also add 2nt at the very end (pos n_target-1, n_target-2 are inside seed)
    # Let's also consider 2nt at positions 0,1 of target (far 5' flank)
    # These won't be directly flanking but can serve as context

    flanking_dinuc_score = flank_au / max(flank_total, 1)

    return {
        "sRNA1_A": sRNA1_A,
        "sRNA1_C": sRNA1_C,
        "sRNA1_G": sRNA1_G,
        "sRNA8_A": sRNA8_A,
        "sRNA8_C": sRNA8_C,
        "sRNA8_G": sRNA8_G,
        "site8_A": site8_A,
        "flanking_dinuc_score": flanking_dinuc_score,
    }


def compute_vienna_advanced_features(
    mirna_seq: str, target_seq: str,
) -> dict[str, float]:
    """Compute advanced ViennaRNA-based thermodynamic features (v13).

    Features:
        dG_open:       Energy cost to unfold target locally (RNAup-style)
        dG_total:      dG_duplex + dG_open (net binding energy)
        ensemble_dG:   Ensemble free energy from partition function (RNAcofold)
        acc_5nt_up:    Accessibility in 5nt window upstream of seed
        acc_10nt_up:   Accessibility in 10nt window upstream of seed
        acc_15nt_up:   Accessibility in 15nt window upstream of seed

    Parameters
    ----------
    mirna_seq, target_seq : str

    Returns
    -------
    dict with 6 float values.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    n_target = len(target_seq)

    defaults = {
        "dG_open": 0.0,
        "dG_total": 0.0,
        "ensemble_dG": 0.0,
        "acc_5nt_up": 0.5,
        "acc_10nt_up": 0.5,
        "acc_15nt_up": 0.5,
    }

    if not _VIENNA_AVAILABLE or _RNA is None or n_target < 8:
        return defaults

    try:
        # ---- dG_open: energy to unfold target at binding site ----
        # Approximate via: MFE(target) - MFE(target_with_site_constrained_open)
        # Simpler approach: use pfl_fold_up for seed region accessibility
        # and convert probability to energy: dG_open ~ -RT * ln(p_unpaired)
        # At 37C, RT = 0.616 kcal/mol
        RT = 0.616  # kcal/mol at 37C

        up = _RNA.pfl_fold_up(target_seq, 8, 50, 30)

        # Seed region: last 8 nt of target (positions n_target-7 to n_target)
        seed_start = max(1, n_target - 7)
        seed_end = n_target

        # p(8 consecutive nt unpaired) at position seed_end
        p_seed_open = 0.01  # default very low
        if seed_end <= n_target and len(up) > seed_end:
            p_val = up[seed_end][min(8, len(up[seed_end]) - 1)]
            if p_val > 1e-10:
                p_seed_open = p_val

        dG_open = -RT * np.log(max(p_seed_open, 1e-10))
        dG_open = max(0.0, min(30.0, dG_open))  # clip to [0, 30] kcal/mol

        # ---- dG_total ----
        duplex_mfe = compute_duplex_mfe(mirna_seq, target_seq)
        dG_total = duplex_mfe + dG_open

        # ---- Ensemble free energy (partition function) ----
        # RNAcofold computes ensemble energy for the complex
        try:
            fc = _RNA.fold_compound(mirna_seq + "&" + target_seq)
            _, ensemble_dG_val = fc.pf()
            ensemble_dG = float(ensemble_dG_val)
            ensemble_dG = max(-50.0, min(0.0, ensemble_dG))
        except Exception:
            ensemble_dG = duplex_mfe  # fallback to MFE

        # ---- Multi-scale upstream accessibility ----
        # Seed match at 3' end: positions [n_target-8, n_target)
        # Upstream of seed: positions [n_target-8-window, n_target-8)
        seed_boundary = n_target - 8  # 0-indexed start of seed region

        def _mean_acc(start_1idx: int, end_1idx: int) -> float:
            """Mean single-nt unpaired prob over [start, end] (1-indexed)."""
            probs = []
            for pos in range(max(1, start_1idx), min(n_target, end_1idx) + 1):
                if pos < len(up):
                    probs.append(up[pos][1])
            return float(np.mean(probs)) if probs else 0.5

        # 5nt upstream of seed
        acc_5nt_up = _mean_acc(seed_boundary - 4, seed_boundary)
        # 10nt upstream
        acc_10nt_up = _mean_acc(seed_boundary - 9, seed_boundary)
        # 15nt upstream
        acc_15nt_up = _mean_acc(seed_boundary - 14, seed_boundary)

        return {
            "dG_open": dG_open,
            "dG_total": dG_total,
            "ensemble_dG": ensemble_dG,
            "acc_5nt_up": acc_5nt_up,
            "acc_10nt_up": acc_10nt_up,
            "acc_15nt_up": acc_15nt_up,
        }
    except Exception as exc:
        logger.warning("Advanced Vienna features failed: %s", exc)
        return defaults


def compute_seed_duplex_mfe(mirna_seq: str, target_seq: str) -> float:
    """Compute MFE of the seed region duplex (miRNA positions 2-8).

    The seed region (nucleotides 2-8, 1-indexed) is the primary
    determinant of miRNA target recognition.  This feature captures
    the thermodynamic stability of the seed:target interaction
    independently from the full duplex.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence (full length).
    target_seq : str
        Target-site sequence.

    Returns
    -------
    float
        Seed duplex MFE in kcal/mol.  Returns 0.0 if the miRNA is
        too short for seed extraction.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)

    # Seed region: positions 2-8 (0-indexed: 1:8)
    if len(mirna_seq) < 8:
        return 0.0

    seed_mirna = mirna_seq[1:8]  # 7 nt seed

    # Corresponding target region: last 7 nt of target
    # (miRNA seed binds to the 3' end of the target site)
    target_rc = reverse_complement_rna(target_seq)
    if len(target_rc) < 7:
        return 0.0
    seed_target_region = target_rc[1:8] if len(target_rc) >= 8 else target_rc[:7]
    # Re-reverse to get the original orientation for duplexfold
    seed_target = reverse_complement_rna(seed_target_region)

    mfe = compute_duplex_mfe(seed_mirna, seed_target)
    # ViennaRNA can return very large positive values for some sequences
    # Clip to physical range for a 7nt duplex (-30 to 30 kcal/mol)
    return max(-30.0, min(30.0, mfe))


def compute_structural_features(
    mirna_seq: str,
    target_seq: str,
) -> dict[str, float | int]:
    """Compute all structural features for a miRNA-target pair.

    Model ⑦: returns 8 features (expanded from 6 in v6).

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence (RNA).
    target_seq : str
        Target-site sequence (RNA).

    Returns
    -------
    dict
        Dictionary containing:
        - ``duplex_mfe``: duplex minimum free energy (kcal/mol)
        - ``mirna_mfe``: miRNA folding MFE (kcal/mol)
        - ``target_mfe``: target-site folding MFE (kcal/mol)
        - ``accessibility``: target_mfe - duplex_mfe (proxy for ΔG_open)
        - ``gc_content``: GC fraction of combined sequences
        - ``seed_match_type``: integer-encoded seed match category
        - ``au_content``: AU fraction of target site (v7)
        - ``seed_duplex_mfe``: seed region duplex MFE (v7)
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)

    duplex_mfe = compute_duplex_mfe(mirna_seq, target_seq)
    mirna_mfe = compute_mfe(mirna_seq)
    target_mfe = compute_mfe(target_seq)

    accessibility = target_mfe - duplex_mfe

    combined_seq = mirna_seq + target_seq
    gc = compute_gc_content(combined_seq)

    seed_type = classify_seed_match(mirna_seq, target_seq)
    seed_type_int = SEED_TYPE_ENCODING.get(seed_type, len(SEED_TYPES) - 1)

    au = compute_au_content(target_seq)
    seed_mfe = compute_seed_duplex_mfe(mirna_seq, target_seq)

    # ---- v10+ extended features (4 new) ----
    seed_acc, site_acc = compute_plfold_accessibility(target_seq)
    supp_3p = compute_supp_3prime_score(mirna_seq, target_seq)
    local_au = compute_local_au_flanking(target_seq)

    # ---- v11 features (8 new) ----
    sps = compute_seed_pairing_stability(mirna_seq, target_seq)
    comp_3p = compute_comp_3prime_score(mirna_seq, target_seq)
    central = compute_central_pairing(mirna_seq, target_seq)

    # MFE ratio: normalized binding efficiency (derived, no ViennaRNA call)
    sum_indiv = mirna_mfe + target_mfe
    mfe_ratio = duplex_mfe / sum_indiv if abs(sum_indiv) > 0.01 else 0.0

    # Duplex pairing statistics
    wobble_ct, mismatch_ct, longest_contig, seed_gc_ct = (
        compute_duplex_pairing_stats(mirna_seq, target_seq)
    )
    # Normalize seed GC count to fraction (14 nt in seed duplex: 7 mirna + 7 target)
    seed_gc_frac = seed_gc_ct / 14.0

    return {
        "duplex_mfe": duplex_mfe,
        "mirna_mfe": mirna_mfe,
        "target_mfe": target_mfe,
        "accessibility": accessibility,
        "gc_content": gc,
        "seed_match_type": seed_type_int,
        "au_content": au,
        "seed_duplex_mfe": seed_mfe,
        # v10+ extended features
        "plfold_seed_accessibility": seed_acc,
        "plfold_site_accessibility": site_acc,
        "supp_3prime_score": supp_3p,
        "local_au_flanking": local_au,
        # v11 features
        "seed_pairing_stability": sps,
        "comp_3prime_score": comp_3p,
        "central_pairing": central,
        "mfe_ratio": mfe_ratio,
        "wobble_count": float(wobble_ct),
        "longest_contiguous": float(longest_contig),
        "mismatch_count": float(mismatch_ct),
        "seed_gc_content": seed_gc_frac,
    }


def compute_base_pairing_matrix(
    mirna_seq: str,
    target_seq: str,
    max_mirna_len: int = DEFAULT_MAX_MIRNA_LEN,
    max_target_len: int = DEFAULT_MAX_TARGET_LEN,
) -> np.ndarray:
    """Compute the base-pairing score matrix between miRNA and target.

    Delegates to the utility function in ``deepexomir.utils.sequence``
    after cleaning the input sequences.

    Parameters
    ----------
    mirna_seq : str
        miRNA sequence.
    target_seq : str
        Target-site sequence.
    max_mirna_len : int
        Padded miRNA length (default 30).
    max_target_len : int
        Padded target length (default 40).

    Returns
    -------
    np.ndarray
        Float32 array of shape ``[max_mirna_len, max_target_len]``.
    """
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)
    return _bp_matrix(mirna_seq, target_seq, max_mirna_len, max_target_len)


def compute_features_batch(
    mirna_seqs: list[str],
    target_seqs: list[str],
) -> list[dict[str, float | int]]:
    """Compute structural features for a batch of pairs.

    Parameters
    ----------
    mirna_seqs : list[str]
        List of miRNA sequences.
    target_seqs : list[str]
        List of target-site sequences (same length as mirna_seqs).

    Returns
    -------
    list[dict]
        List of feature dictionaries, one per pair.
    """
    if len(mirna_seqs) != len(target_seqs):
        raise ValueError(
            f"Length mismatch: {len(mirna_seqs)} miRNAs vs "
            f"{len(target_seqs)} targets."
        )

    results = []
    for mirna, target in zip(mirna_seqs, target_seqs):
        try:
            feats = compute_structural_features(mirna, target)
        except Exception as exc:
            logger.warning(
                "Feature computation failed for (%s, %s): %s",
                mirna[:20],
                target[:20],
                exc,
            )
            feats = {
                "duplex_mfe": 0.0,
                "mirna_mfe": 0.0,
                "target_mfe": 0.0,
                "accessibility": 0.0,
                "gc_content": 0.0,
                "seed_match_type": len(SEED_TYPES) - 1,
            }
        results.append(feats)

    return results


def is_vienna_available() -> bool:
    """Check whether ViennaRNA is installed and importable.

    Returns
    -------
    bool
        True if the ``RNA`` module can be imported.
    """
    return _VIENNA_AVAILABLE
