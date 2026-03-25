"""Pre-compute v11 structural features and append to existing v10 arrays.

Adds 8 new features to the existing 12-feature v10 structural arrays:
   12: seed_pairing_stability  - nearest-neighbor stacking SPS
   13: comp_3prime_score       - 3' compensatory pairing (miRNA 17-21)
   14: central_pairing         - central region pairing (miRNA 9-12)
   15: mfe_ratio               - duplex_mfe / (mirna_mfe + target_mfe)
   16: wobble_count            - G:U wobble pairs in full duplex
   17: longest_contiguous      - longest contiguous complementary stretch
   18: mismatch_count          - total mismatches in duplex
   19: seed_gc_content         - GC fraction of seed duplex region

Usage:
    python scripts/precompute_v11_features.py
    python scripts/precompute_v11_features.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepexomir.data.features import (
    compute_seed_pairing_stability,
    compute_comp_3prime_score,
    compute_central_pairing,
    compute_duplex_pairing_stats,
)
from deepexomir.utils.sequence import clean_sequence

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

V11_FEATURE_NAMES = [
    "seed_pairing_stability",
    "comp_3prime_score",
    "central_pairing",
    "mfe_ratio",
    "wobble_count",
    "longest_contiguous",
    "mismatch_count",
    "seed_gc_content",
]


def compute_v11_features(
    mirna_seq: str,
    target_seq: str,
    duplex_mfe: float,
    mirna_mfe: float,
    target_mfe: float,
) -> list[float]:
    """Compute the 8 new v11 features for one miRNA-target pair."""
    mirna_seq = clean_sequence(mirna_seq)
    target_seq = clean_sequence(target_seq)

    sps = compute_seed_pairing_stability(mirna_seq, target_seq)
    comp_3p = compute_comp_3prime_score(mirna_seq, target_seq)
    central = compute_central_pairing(mirna_seq, target_seq)

    # MFE ratio (derived from existing features)
    sum_indiv = mirna_mfe + target_mfe
    mfe_ratio = duplex_mfe / sum_indiv if abs(sum_indiv) > 0.01 else 0.0

    wobble_ct, mismatch_ct, longest_contig, seed_gc_ct = (
        compute_duplex_pairing_stats(mirna_seq, target_seq)
    )
    seed_gc_frac = seed_gc_ct / 14.0

    return [
        sps, comp_3p, central, mfe_ratio,
        float(wobble_ct), float(longest_contig),
        float(mismatch_ct), seed_gc_frac,
    ]


def precompute_split(
    parquet_path: Path,
    v10_npy_path: Path,
    output_path: Path,
) -> None:
    """Compute v11 features and merge with existing v10 12-feature array."""
    if output_path.exists():
        existing = np.load(output_path)
        if existing.shape[1] >= 20:
            logger.info(
                "  Output already has %d features (%s). Skipping.",
                existing.shape[1], output_path.name,
            )
            return

    # Load existing v10 12-feature array
    if not v10_npy_path.exists():
        logger.error("  v10 features not found: %s", v10_npy_path)
        return

    v10_features = np.load(v10_npy_path)
    n_samples = v10_features.shape[0]
    n_v10 = v10_features.shape[1]
    logger.info(
        "  Loaded v10 features: %s (shape=%s)",
        v10_npy_path.name, v10_features.shape,
    )

    # Load parquet for sequences
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    assert len(df) == n_samples, (
        f"Sample count mismatch: {len(df)} parquet vs {n_samples} npy"
    )

    mirna_seqs = df["mirna_seq"].fillna("").astype(str).tolist()
    target_seqs = df["target_seq"].fillna("").astype(str).tolist()

    # Extract existing MFE values from v10 array (indices 0, 1, 2)
    duplex_mfes = v10_features[:, 0]
    mirna_mfes = v10_features[:, 1]
    target_mfes = v10_features[:, 2]

    logger.info("  Computing %d new features for %d samples ...", len(V11_FEATURE_NAMES), n_samples)

    start = time.time()
    new_features = np.zeros((n_samples, len(V11_FEATURE_NAMES)), dtype=np.float32)

    for i in range(n_samples):
        try:
            feats = compute_v11_features(
                mirna_seqs[i], target_seqs[i],
                float(duplex_mfes[i]), float(mirna_mfes[i]), float(target_mfes[i]),
            )
            new_features[i] = feats
        except Exception as exc:
            logger.warning("  Failed at idx %d: %s", i, exc)
            new_features[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            pct = 100.0 * (i + 1) / n_samples
            logger.info(
                "    [%5.1f%%] %d/%d (%.0f/s, ETA: %.0fs)",
                pct, i + 1, n_samples, rate, eta,
            )

    elapsed = time.time() - start
    logger.info(
        "  Computed %d new features in %.1fs (%.0f samples/s)",
        len(V11_FEATURE_NAMES), elapsed, n_samples / elapsed,
    )

    # Merge: [N, 12] + [N, 8] -> [N, 20]
    merged = np.concatenate([v10_features, new_features], axis=1)
    logger.info("  Merged shape: %s -> %s", v10_features.shape, merged.shape)

    np.save(output_path, merged)
    size_mb = output_path.stat().st_size / 1e6
    logger.info("  Saved to %s (%.1f MB)", output_path.name, size_mb)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute v11 structural features (12 -> 20).",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing parquet + v10 structural_features_v10.npy",
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    print("DeepExoMir -- v11 Feature Pre-computation")
    print("=" * 45)
    print(f"Data directory : {data_dir}")
    print(f"New features   : {len(V11_FEATURE_NAMES)} ({', '.join(V11_FEATURE_NAMES)})")
    print()

    for split in ["train", "val", "test"]:
        parquet_path = data_dir / f"{split}.parquet"
        v10_npy = data_dir / f"{split}_structural_features_v10.npy"
        output_npy = data_dir / f"{split}_structural_features_v11.npy"

        if not parquet_path.exists():
            logger.warning("  Skipping %s (parquet not found)", split)
            continue
        if not v10_npy.exists():
            logger.warning("  Skipping %s (v10 features not found)", split)
            continue

        logger.info("Processing %s ...", split)
        precompute_split(parquet_path, v10_npy, output_npy)
        print()

    print("Done! v11 features saved as *_structural_features_v11.npy")


if __name__ == "__main__":
    main()
