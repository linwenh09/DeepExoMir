"""Pre-compute v18 features: miRNA thermodynamic asymmetry + extended context.

New features added on top of v16c (26 features):
  26: mirna_asym_5p       - 5' end instability (weaker = better RISC loading)
  27: mirna_asym_3p       - 3' end instability
  28: mirna_asym_diff     - 5'-3' asymmetry difference (positive = preferred guide)
  29: upstream_au_15nt    - AU richness 15nt upstream of seed match
  30: downstream_au_15nt  - AU richness 15nt downstream of seed match
  31: seed_dinuc_ua       - UA dinucleotide frequency in seed match region
  32: target_entropy      - Sequence entropy of target binding site

Creates v18 feature files: v16c(26) + new(7) = 33 features

Usage:
    python scripts/precompute_v18_features.py
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Nearest-neighbor stacking energies (kcal/mol) for RNA
# From Turner & Mathews (2010)
NN_ENERGIES = {
    'AA': -0.93, 'AU': -1.10, 'AC': -2.24, 'AG': -2.08,
    'UA': -1.33, 'UU': -0.93, 'UC': -2.08, 'UG': -2.35,
    'CA': -2.11, 'CU': -2.08, 'CC': -3.26, 'CG': -2.36,
    'GA': -2.35, 'GU': -2.24, 'GC': -3.42, 'GG': -3.26,
}


def compute_end_stability(seq: str, n_nt: int = 4) -> float:
    """Compute stacking energy of terminal n nucleotides.

    More negative = more stable.
    Returns sum of nearest-neighbor stacking energies.
    """
    seq = seq.upper().replace('T', 'U')
    end = seq[:n_nt]
    energy = 0.0
    for i in range(len(end) - 1):
        dinuc = end[i:i+2]
        energy += NN_ENERGIES.get(dinuc, -1.5)  # default for unknown
    return energy


def compute_sequence_entropy(seq: str) -> float:
    """Compute Shannon entropy of nucleotide composition."""
    seq = seq.upper()
    n = len(seq)
    if n == 0:
        return 0.0
    counts = Counter(seq)
    entropy = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_dinuc_frequency(seq: str, dinuc: str) -> float:
    """Compute frequency of a specific dinucleotide in a sequence."""
    seq = seq.upper().replace('T', 'U')
    dinuc = dinuc.upper()
    if len(seq) < 2:
        return 0.0
    count = sum(1 for i in range(len(seq) - 1) if seq[i:i+2] == dinuc)
    return count / (len(seq) - 1)


def compute_v18_features(mirna_seq: str, target_seq: str) -> list[float]:
    """Compute 7 new features for one miRNA-target pair."""
    mirna = mirna_seq.upper().replace('T', 'U')
    target = target_seq.upper().replace('T', 'U')

    # 1-3: miRNA thermodynamic asymmetry
    # 5' end: first 4nt of miRNA (should be unstable for good guide strand)
    # 3' end: last 4nt of miRNA
    stability_5p = compute_end_stability(mirna[:4])  # more negative = more stable
    stability_3p = compute_end_stability(mirna[-4:])
    # Asymmetry: positive means 5' is less stable (preferred as guide)
    asym_diff = stability_3p - stability_5p

    # 4-5: Upstream/downstream AU richness (relative to seed match position)
    # Target seq is 50nt, seed match is typically at 3' end of target (positions ~42-49)
    # Convention: target positions 0-14 = upstream, 35-49 = near seed match
    upstream_15 = target[:15]
    downstream_15 = target[35:]
    upstream_au = sum(1 for c in upstream_15 if c in 'AU') / max(len(upstream_15), 1)
    downstream_au = sum(1 for c in downstream_15 if c in 'AU') / max(len(downstream_15), 1)

    # 6: UA dinucleotide frequency in seed match region (target positions 42-49)
    seed_region = target[42:50]  # 8nt seed match region
    ua_freq = compute_dinuc_frequency(seed_region, 'UA')

    # 7: Target sequence entropy
    target_entropy = compute_sequence_entropy(target)

    return [
        stability_5p,      # mirna_asym_5p
        stability_3p,      # mirna_asym_3p
        asym_diff,         # mirna_asym_diff
        upstream_au,       # upstream_au_15nt
        downstream_au,     # downstream_au_15nt
        ua_freq,           # seed_dinuc_ua
        target_entropy,    # target_entropy
    ]


def process_split(data_dir: Path, split: str) -> None:
    """Process one data split."""
    parquet_path = data_dir / f"{split}.parquet"
    v16c_path = data_dir / f"{split}_structural_features_v16c.npy"
    output_path = data_dir / f"{split}_structural_features_v18.npy"

    if output_path.exists():
        existing = np.load(output_path)
        if existing.shape[1] >= 33:
            logger.info(f"  {split}: already has {existing.shape[1]} features, skipping")
            return

    if not v16c_path.exists():
        logger.error(f"  {split}: v16c features not found at {v16c_path}")
        return

    logger.info(f"  {split}: loading data...")
    df = pd.read_parquet(parquet_path)
    v16c_features = np.load(v16c_path)
    n_samples = len(df)

    logger.info(f"  {split}: computing {n_samples:,} samples...")

    new_features = np.zeros((n_samples, 7), dtype=np.float32)

    t0 = time.time()
    for i in range(n_samples):
        mirna_seq = df.iloc[i]['mirna_seq']
        target_seq = df.iloc[i]['target_seq']
        new_features[i] = compute_v18_features(mirna_seq, target_seq)

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            logger.info(f"    {i+1:,}/{n_samples:,} ({rate:.0f}/s, ETA {eta:.0f}s)")

    # Concatenate v16c + new features
    combined = np.concatenate([v16c_features, new_features], axis=1)
    np.save(output_path, combined)

    elapsed = time.time() - t0
    logger.info(f"  {split}: done in {elapsed:.1f}s, shape={combined.shape}, saved to {output_path.name}")

    # Print feature stats
    logger.info(f"  {split}: new feature stats:")
    names = ['mirna_asym_5p', 'mirna_asym_3p', 'mirna_asym_diff',
             'upstream_au_15nt', 'downstream_au_15nt', 'seed_dinuc_ua', 'target_entropy']
    for j, name in enumerate(names):
        col = new_features[:, j]
        logger.info(f"    {name}: mean={col.mean():.4f}, std={col.std():.4f}, "
                    f"min={col.min():.4f}, max={col.max():.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    logger.info("Computing v18 features (v16c + 7 new features = 33 total)")

    for split in ["train", "val", "test"]:
        process_split(args.data_dir, split)

    logger.info("All done!")


if __name__ == "__main__":
    main()
