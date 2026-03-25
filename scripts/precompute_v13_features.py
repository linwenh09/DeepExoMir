"""Pre-compute v13 ViennaRNA advanced features (6 new -> 34 total).

Loads existing v12 .npy (28 features), computes 6 new ViennaRNA-based
features, and concatenates to produce v13 .npy (34 features).

New features (v13):
    28. dG_open          Energy cost to unfold target at binding site
    29. dG_total         dG_duplex + dG_open (net binding energy)
    30. ensemble_dG      Ensemble free energy from partition function
    31. acc_5nt_up       Accessibility 5nt upstream of seed
    32. acc_10nt_up      Accessibility 10nt upstream of seed
    33. acc_15nt_up      Accessibility 15nt upstream of seed

WARNING: This script is SLOW (~10-20 samples/s) due to RNAcofold partition
function calls. Estimated time: ~27 hours for train set (1.9M samples).
Consider running overnight.

Usage:
    python scripts/precompute_v13_features.py --data-dir data/processed
    python scripts/precompute_v13_features.py --data-dir data/processed --split test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

from deepexomir.data.features import compute_vienna_advanced_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

V13_FEATURE_NAMES = [
    "dG_open", "dG_total", "ensemble_dG",
    "acc_5nt_up", "acc_10nt_up", "acc_15nt_up",
]

V13_DEFAULTS = [0.0, 0.0, 0.0, 0.5, 0.5, 0.5]


def precompute_split(
    parquet_path: Path,
    v12_npy: Path,
    output_npy: Path,
    resume_from: int = 0,
) -> None:
    """Compute v13 features for one split and save concatenated array."""
    # Load existing v12 features
    v12_arr = np.load(v12_npy)
    logger.info("  Loaded v12 features: %s (shape=%s)", v12_npy.name, v12_arr.shape)
    n_samples = v12_arr.shape[0]

    # Load parquet for sequences
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    assert len(df) == n_samples, (
        f"Mismatch: parquet has {len(df)} rows but v12 has {n_samples}"
    )

    # Check for partial results (resume support)
    partial_npy = output_npy.parent / (output_npy.stem + "_partial.npy")
    if partial_npy.exists() and resume_from == 0:
        partial = np.load(partial_npy)
        resume_from = int(partial.shape[0])
        logger.info("  Resuming from sample %d (partial file found)", resume_from)
        new_feats_list = partial.tolist()
    else:
        new_feats_list = []

    if resume_from > 0 and not partial_npy.exists():
        # Pre-fill with defaults for skipped samples
        new_feats_list = [V13_DEFAULTS] * resume_from

    logger.info(
        "  Computing %d new features for %d samples (starting at %d) ...",
        len(V13_FEATURE_NAMES), n_samples, resume_from,
    )
    t0 = time.time()

    mirna_seqs = df["mirna_seq"].values
    target_seqs = df["target_seq"].values

    for i in range(resume_from, n_samples):
        try:
            feats = compute_vienna_advanced_features(
                str(mirna_seqs[i]), str(target_seqs[i])
            )
            new_feats_list.append([feats[k] for k in V13_FEATURE_NAMES])
        except Exception:
            new_feats_list.append(V13_DEFAULTS)

        # Progress and checkpoint every 10K samples
        done = i - resume_from + 1
        if done % 10000 == 0:
            elapsed = time.time() - t0
            rate = done / elapsed
            remaining = n_samples - i - 1
            eta_s = remaining / rate if rate > 0 else 0
            eta_h = eta_s / 3600
            logger.info(
                "    [%5.1f%%] %d/%d (%.1f/s, ETA: %.1fh)",
                100.0 * (i + 1) / n_samples, i + 1, n_samples, rate, eta_h,
            )
            # Save partial checkpoint
            partial_arr = np.array(new_feats_list, dtype=np.float32)
            np.save(partial_npy, partial_arr)

    elapsed = time.time() - t0
    actual_computed = n_samples - resume_from
    logger.info(
        "  Computed %d features for %d samples in %.1fs (%.1f samples/s)",
        len(V13_FEATURE_NAMES), actual_computed, elapsed,
        actual_computed / elapsed if elapsed > 0 else 0,
    )

    # Concatenate v12 (28) + v13 (6) = 34 features
    new_feats_arr = np.array(new_feats_list, dtype=np.float32)
    merged = np.concatenate([v12_arr, new_feats_arr], axis=1)
    logger.info("  Merged shape: %s -> %s", v12_arr.shape, merged.shape)

    np.save(output_npy, merged)
    size_mb = output_npy.stat().st_size / (1024 * 1024)
    logger.info("  Saved to %s (%.1f MB)", output_npy.name, size_mb)

    # Clean up partial file
    if partial_npy.exists():
        partial_npy.unlink()
        logger.info("  Removed partial checkpoint")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute v13 ViennaRNA features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--split", type=str, default=None,
        help="Process only this split (train/val/test). Default: all.",
    )
    parser.add_argument(
        "--resume-from", type=int, default=0,
        help="Resume from this sample index (for crash recovery).",
    )
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    splits = [args.split] if args.split else ["train", "val", "test"]

    print()
    print("DeepExoMir -- v13 Feature Pre-computation (ViennaRNA advanced)")
    print("=" * 60)
    print(f"Data directory : {data_dir}")
    print(f"New features   : {len(V13_FEATURE_NAMES)} ({', '.join(V13_FEATURE_NAMES)})")
    print(f"Splits         : {', '.join(splits)}")
    print()
    print("WARNING: This is slow (~10-20 samples/s due to RNAcofold).")
    print("         Train set (~1.9M) may take 25+ hours.")
    print("         Progress is checkpointed every 10K samples.")
    print()

    for split in splits:
        parquet_path = data_dir / f"{split}.parquet"
        v12_npy = data_dir / f"{split}_structural_features_v12.npy"
        output_npy = data_dir / f"{split}_structural_features_v13.npy"

        if not parquet_path.exists():
            logger.warning("Skipping %s: parquet not found", split)
            continue
        if not v12_npy.exists():
            logger.warning("Skipping %s: v12 features not found", split)
            continue

        logger.info("Processing %s ...", split)
        precompute_split(parquet_path, v12_npy, output_npy, args.resume_from)

    print()
    print("Done! v13 features saved as *_structural_features_v13.npy")


if __name__ == "__main__":
    main()
