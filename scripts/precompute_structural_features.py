"""Pre-compute structural features for the DeepExoMir training dataset.

Computes ViennaRNA-based thermodynamic features for all samples in the
train/val/test parquet files and saves them as numpy arrays for fast
loading during training.

Model ⑦: 8 structural features per sample:
    0: duplex_mfe       — MFE of miRNA:target duplex
    1: mirna_mfe        — MFE of miRNA secondary structure
    2: target_mfe       — MFE of target secondary structure
    3: accessibility    — target_mfe - duplex_mfe (ΔG_open proxy)
    4: gc_content       — GC fraction of combined sequences
    5: seed_match_type  — integer-encoded seed match category
    6: au_content       — AU fraction of target site
    7: seed_duplex_mfe  — seed region duplex MFE

Usage:
    python scripts/precompute_structural_features.py
    python scripts/precompute_structural_features.py --data-dir data/processed
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepexomir.data.features import compute_structural_features, is_vienna_available
from deepexomir.data.dataset import STRUCTURAL_FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _compute_single(args: tuple) -> list:
    """Compute structural features for a single miRNA-target pair."""
    mirna_seq, target_seq = args
    try:
        feats = compute_structural_features(mirna_seq, target_seq)
        return [float(feats[k]) for k in STRUCTURAL_FEATURE_NAMES]
    except Exception as exc:
        logger.warning("Failed for (%s, %s): %s", mirna_seq[:20], target_seq[:20], exc)
        return [0.0] * len(STRUCTURAL_FEATURE_NAMES)


def precompute_split(parquet_path: Path, output_path: Path, n_workers: int = 1) -> None:
    """Pre-compute structural features for one data split.

    Parameters
    ----------
    parquet_path : Path
        Input parquet file containing mirna_seq and target_seq columns.
    output_path : Path
        Output .npy file to save the feature array.
    n_workers : int
        Number of parallel workers (default: 1 for ViennaRNA compatibility).
    """
    if output_path.exists():
        existing = np.load(output_path)
        logger.info(
            "  Already exists: %s (%d samples, %d features). Skipping.",
            output_path.name, existing.shape[0], existing.shape[1],
        )
        return

    df = pd.read_parquet(parquet_path, engine="pyarrow")
    n = len(df)
    logger.info("  Processing %s: %d samples ...", parquet_path.name, n)

    mirna_seqs = df["mirna_seq"].fillna("").astype(str).tolist()
    target_seqs = df["target_seq"].fillna("").astype(str).tolist()

    pairs = list(zip(mirna_seqs, target_seqs))

    start = time.time()

    if n_workers > 1:
        # Multiprocessing for speed
        with Pool(n_workers) as pool:
            results = []
            for i, result in enumerate(pool.imap(_compute_single, pairs, chunksize=500)):
                results.append(result)
                if (i + 1) % 50000 == 0:
                    elapsed = time.time() - start
                    rate = (i + 1) / elapsed
                    eta = (n - i - 1) / rate
                    logger.info(
                        "    %d/%d (%.1f/s, ETA: %.1fs)", i + 1, n, rate, eta
                    )
    else:
        # Single-threaded (safer for ViennaRNA)
        results = []
        for i, pair in enumerate(pairs):
            results.append(_compute_single(pair))
            if (i + 1) % 50000 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                logger.info(
                    "    %d/%d (%.1f/s, ETA: %.1fs)", i + 1, n, rate, eta
                )

    elapsed = time.time() - start
    feature_array = np.array(results, dtype=np.float32)

    np.save(output_path, feature_array)
    logger.info(
        "  Saved %s: shape=%s (%.1f seconds, %.0f samples/s)",
        output_path.name, feature_array.shape, elapsed, n / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-compute structural features for DeepExoMir.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing train.parquet, val.parquet, test.parquet",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel workers (1 recommended for ViennaRNA)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    n_workers = args.workers

    print("DeepExoMir — Structural Feature Pre-computation")
    print("=" * 55)
    print(f"Data directory : {data_dir}")
    print(f"Features       : {len(STRUCTURAL_FEATURE_NAMES)} ({', '.join(STRUCTURAL_FEATURE_NAMES)})")
    print(f"ViennaRNA      : {'available' if is_vienna_available() else 'NOT available (heuristic mode)'}")
    print(f"Workers        : {n_workers}")
    print()

    for split in ["train", "val", "test"]:
        parquet_path = data_dir / f"{split}.parquet"
        if not parquet_path.exists():
            logger.warning("  Skipping %s (file not found)", parquet_path)
            continue

        output_path = data_dir / f"{split}_structural_features.npy"
        precompute_split(parquet_path, output_path, n_workers=n_workers)

    print("\nDone! Features saved alongside parquet files.")


if __name__ == "__main__":
    main()
