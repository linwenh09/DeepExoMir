"""Negative sample generation for miRNA-target interaction data.

Generates high-quality negative (non-interacting) pairs by shuffling
the miRNA seed region while preserving global sequence properties.
Ensures that shuffled seeds do not coincidentally match any real miRNA
seed in miRBase, and that GC-content distribution is matched.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd

from deepexomir.utils.constants import SEED_END, SEED_START
from deepexomir.utils.sequence import (
    clean_sequence,
    compute_gc_content,
    extract_seed_region,
)

logger = logging.getLogger(__name__)

# Maximum number of seed-shuffle attempts before giving up on a single sample
_MAX_SHUFFLE_ATTEMPTS = 100

# GC-content tolerance for matching
_GC_TOLERANCE = 0.05


def _collect_real_seeds(mirbase_seqs: dict[str, str]) -> set[str]:
    """Extract all unique seed regions from miRBase sequences.

    Parameters
    ----------
    mirbase_seqs : dict[str, str]
        Mapping of miRNA ID to mature sequence from miRBase.

    Returns
    -------
    set[str]
        Set of all unique 7-nucleotide seed regions.
    """
    seeds: set[str] = set()
    for seq in mirbase_seqs.values():
        seq = clean_sequence(seq)
        if len(seq) >= SEED_END + 1:
            seeds.add(extract_seed_region(seq))
    logger.info("Collected %d unique real miRNA seeds from miRBase.", len(seeds))
    return seeds


def _shuffle_seed(seed: str) -> str:
    """Shuffle the nucleotides in a seed region.

    Parameters
    ----------
    seed : str
        7-nucleotide seed string.

    Returns
    -------
    str
        Shuffled seed (guaranteed different from input if len >= 2).
    """
    chars = list(seed)
    # Fisher-Yates shuffle ensuring the result differs from input
    for _ in range(_MAX_SHUFFLE_ATTEMPTS):
        random.shuffle(chars)
        shuffled = "".join(chars)
        if shuffled != seed:
            return shuffled
    # Fallback: rotate by 1
    return seed[1:] + seed[0]


def _compute_gc(seq: str) -> float:
    """Compute GC content, handling empty strings."""
    if not seq:
        return 0.0
    return compute_gc_content(seq)


def generate_negative_for_pair(
    mirna_seq: str,
    target_seq: str,
    real_seeds: set[str],
    *,
    max_attempts: int = _MAX_SHUFFLE_ATTEMPTS,
) -> Optional[tuple[str, str]]:
    """Generate one negative pair by shuffling the miRNA seed region.

    The seed region (positions 2-8) is shuffled while keeping the flanking
    nucleotides intact.  The shuffled seed is checked against all known
    real miRNA seeds in miRBase to avoid accidental positives.

    Parameters
    ----------
    mirna_seq : str
        Original miRNA sequence (RNA).
    target_seq : str
        Original target-site sequence (RNA).
    real_seeds : set[str]
        Set of known real seed sequences to avoid.
    max_attempts : int
        Maximum shuffling attempts.

    Returns
    -------
    tuple[str, str] or None
        (shuffled_mirna, target_seq) if successful, None otherwise.
    """
    mirna_seq = clean_sequence(mirna_seq)
    if len(mirna_seq) < SEED_END + 1:
        return None

    original_seed = extract_seed_region(mirna_seq)
    original_gc = _compute_gc(original_seed)

    prefix = mirna_seq[:SEED_START]
    suffix = mirna_seq[SEED_END + 1:]

    for _ in range(max_attempts):
        new_seed = _shuffle_seed(original_seed)

        # Reject if seed matches any real miRNA seed
        if new_seed in real_seeds:
            continue

        # Reject if GC content deviates too much
        new_gc = _compute_gc(new_seed)
        if abs(new_gc - original_gc) > _GC_TOLERANCE:
            continue

        shuffled_mirna = prefix + new_seed + suffix
        return (shuffled_mirna, target_seq)

    logger.debug(
        "Could not generate negative for miRNA seed '%s' after %d attempts.",
        original_seed,
        max_attempts,
    )
    return None


def generate_negatives(
    positive_df: pd.DataFrame,
    mirbase_seqs: dict[str, str],
    *,
    ratio: float = 1.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate negative samples for a DataFrame of positive interactions.

    For each positive (miRNA, target_site) pair, generates ``ratio`` negative
    pairs by shuffling the miRNA seed region.  Ensures no shuffled seed
    matches a real miRBase seed.

    Parameters
    ----------
    positive_df : pd.DataFrame
        DataFrame with at least ``mirna_seq`` and ``target_seq`` columns.
        All rows are assumed to be positive interactions.
    mirbase_seqs : dict[str, str]
        miRBase mature sequences for seed-collision checking.
    ratio : float
        Negative-to-positive ratio.  1.0 produces a balanced dataset.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame of negative samples with columns ``mirna_seq``,
        ``target_seq``, ``label`` (always 0), and ``source``.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    real_seeds = _collect_real_seeds(mirbase_seqs)

    n_target = int(len(positive_df) * ratio)
    logger.info(
        "Generating %d negative samples from %d positives (ratio=%.2f).",
        n_target,
        len(positive_df),
        ratio,
    )

    negatives: list[dict] = []
    failed = 0

    # Bin positives by GC content for distribution matching
    gc_bins: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(positive_df)):
        row = positive_df.iloc[idx]
        gc = _compute_gc(clean_sequence(row["mirna_seq"]))
        bin_key = int(gc * 20)  # 5% bins
        gc_bins[bin_key].append(idx)

    # Sample proportionally from each GC bin to match distribution
    indices = list(range(len(positive_df)))
    if n_target <= len(positive_df):
        sampled_indices = random.sample(indices, n_target)
    else:
        # Over-sample with replacement
        sampled_indices = random.choices(indices, k=n_target)

    for idx in sampled_indices:
        row = positive_df.iloc[idx]
        result = generate_negative_for_pair(
            row["mirna_seq"],
            row["target_seq"],
            real_seeds,
        )
        if result is not None:
            neg_mirna, neg_target = result
            negatives.append(
                {
                    "mirna_seq": neg_mirna,
                    "target_seq": neg_target,
                    "label": 0,
                    "source": "seed_shuffle_negative",
                }
            )
        else:
            failed += 1

    if failed > 0:
        logger.warning(
            "Failed to generate negatives for %d / %d samples.", failed, n_target
        )

    neg_df = pd.DataFrame(negatives)
    logger.info(
        "Generated %d negative samples (target was %d).", len(neg_df), n_target
    )
    return neg_df


def generate_balanced_dataset(
    positive_df: pd.DataFrame,
    mirbase_seqs: dict[str, str],
    *,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate a balanced dataset with equal positives and negatives.

    Parameters
    ----------
    positive_df : pd.DataFrame
        DataFrame of positive interactions.
    mirbase_seqs : dict[str, str]
        miRBase sequences for seed checking.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with ``label`` column (1 for positive, 0 for
        negative), shuffled.
    """
    # Ensure positives have the label column
    pos = positive_df.copy()
    pos["label"] = 1
    if "source" not in pos.columns:
        pos["source"] = "positive"

    neg = generate_negatives(
        pos, mirbase_seqs, ratio=1.0, random_seed=random_seed
    )

    combined = pd.concat([pos, neg], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=random_seed).reset_index(
        drop=True
    )

    logger.info(
        "Balanced dataset: %d total (%d pos, %d neg).",
        len(combined),
        (combined["label"] == 1).sum(),
        (combined["label"] == 0).sum(),
    )
    return combined
