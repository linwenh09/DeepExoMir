"""Remove contradictory training pairs from processed data.

Contradictory pairs are cases where the exact same (mirna_seq, target_seq) pair
appears with BOTH label=1 AND label=0. These create conflicting supervision
signals and place a hard ceiling on model performance.

Usage:
    python scripts/clean_contradictory_pairs.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/processed")


def remove_contradictory_pairs(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Remove rows where the same (mirna_seq, target_seq) pair has both labels."""
    n_before = len(df)

    # Find pairs that appear with both labels
    pair_labels = df.groupby(["mirna_seq", "target_seq"])["label"].nunique()
    contradictory_pairs = pair_labels[pair_labels > 1].index

    n_contradictory = len(contradictory_pairs)
    if n_contradictory == 0:
        logger.info("[%s] No contradictory pairs found.", split_name)
        return df

    # Remove ALL rows for contradictory pairs (both positive and negative)
    contradictory_set = set(map(tuple, contradictory_pairs.tolist()))
    mask = df.apply(
        lambda row: (row["mirna_seq"], row["target_seq"]) in contradictory_set,
        axis=1,
    )
    n_removed = mask.sum()
    df_clean = df[~mask].reset_index(drop=True)

    n_pos = (df_clean["label"] == 1).sum()
    n_neg = (df_clean["label"] == 0).sum()

    logger.info(
        "[%s] Removed %d contradictory pairs (%d rows). %d -> %d rows. "
        "pos=%d, neg=%d (ratio=%.3f)",
        split_name,
        n_contradictory,
        n_removed,
        n_before,
        len(df_clean),
        n_pos,
        n_neg,
        n_pos / max(n_neg, 1),
    )

    return df_clean


def main() -> None:
    for split in ["train", "val", "test"]:
        path = DATA_DIR / f"{split}.parquet"
        if not path.exists():
            logger.warning("File not found: %s", path)
            continue

        df = pd.read_parquet(path)
        logger.info("[%s] Loaded %d rows", split, len(df))

        df_clean = remove_contradictory_pairs(df, split)

        # Save cleaned data (backup original first)
        backup_path = DATA_DIR / f"{split}_original.parquet"
        if not backup_path.exists():
            df_orig = pd.read_parquet(path)
            df_orig.to_parquet(backup_path, index=False)
            logger.info("[%s] Backed up original to %s", split, backup_path)

        df_clean.to_parquet(path, index=False)
        logger.info("[%s] Saved cleaned data to %s", split, path)

    logger.info("Done.")


if __name__ == "__main__":
    main()
