"""Pre-compute v15 features: PhastCons + GERP++ conservation scores.

Loads existing v14alt2L .npy (31 features), computes up to 6 new conservation
features, and concatenates to produce v15 .npy (up to 37 features).

New features (v15):
    31. phastcons_mean       Mean PhastCons score over target site
    32. phastcons_max        Max PhastCons score
    33. phastcons_seed_mean  Mean PhastCons score over seed region
    34. gerp_mean            Mean GERP++ RS score over target site (if available)
    35. gerp_max             Max GERP++ RS score (if available)
    36. gerp_seed_mean       Mean GERP++ RS score over seed region (if available)

Usage:
    python scripts/precompute_v15_features.py --data-dir data/processed
    python scripts/precompute_v15_features.py --data-dir data/processed --no-gerp
"""

import argparse
import logging
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_bw_features(bw, chrom, start, end, seed_len=7):
    """Extract mean, max, seed_mean from a bigWig for one region."""
    try:
        vals = bw.values(chrom, start, end)
        vals_clean = np.array([v for v in vals if v is not None and not np.isnan(v)],
                              dtype=np.float32)
        if len(vals_clean) == 0:
            return 0.0, 0.0, 0.0

        mean_val = float(np.mean(vals_clean))
        max_val = float(np.max(vals_clean))

        # Seed region
        site_len = end - start
        sl = min(seed_len, site_len)
        seed_vals = bw.values(chrom, start, start + sl)
        seed_clean = np.array([v for v in seed_vals if v is not None and not np.isnan(v)],
                              dtype=np.float32)
        seed_mean = float(np.mean(seed_clean)) if len(seed_clean) > 0 else mean_val

        return mean_val, max_val, seed_mean
    except Exception:
        return 0.0, 0.0, 0.0


def precompute_split(
    coords_parquet: Path,
    prev_npy: Path,
    output_npy: Path,
    phastcons_bw_path: Path,
    gerp_bw_path: Path | None = None,
) -> None:
    """Compute v15 conservation features for one split."""
    import pybigtools

    prev_arr = np.load(prev_npy)
    logger.info("  Loaded previous features: %s (shape=%s)", prev_npy.name, prev_arr.shape)
    n_samples = prev_arr.shape[0]

    df = pd.read_parquet(coords_parquet, engine="pyarrow")
    assert len(df) == n_samples, f"Mismatch: parquet {len(df)} vs npy {n_samples}"

    # Open bigWig files
    bw_phastcons = pybigtools.open(str(phastcons_bw_path))
    logger.info("  Opened PhastCons: %s", phastcons_bw_path.name)

    use_gerp = gerp_bw_path is not None and gerp_bw_path.exists()
    bw_gerp = None
    if use_gerp:
        bw_gerp = pybigtools.open(str(gerp_bw_path))
        logger.info("  Opened GERP++: %s", gerp_bw_path.name)

    n_new = 6 if use_gerp else 3
    new_feats = np.zeros((n_samples, n_new), dtype=np.float32)

    chroms = df["chr"].values if "chr" in df.columns else None
    starts = df["genomic_start"].values if "genomic_start" in df.columns else None
    ends = df["genomic_end"].values if "genomic_end" in df.columns else None

    if chroms is None:
        logger.warning("  No genomic coordinates! Filling with zeros.")
        merged = np.concatenate([prev_arr, new_feats], axis=1)
        np.save(output_npy, merged.astype(np.float32))
        return

    t0 = time.time()
    valid_count = 0

    for i in range(n_samples):
        chrom = str(chroms[i]) if pd.notna(chroms[i]) else None
        start = int(starts[i]) if pd.notna(starts[i]) else None
        end = int(ends[i]) if pd.notna(ends[i]) else None

        if chrom is None or start is None or end is None or start >= end:
            continue

        valid_count += 1

        # PhastCons
        pc_mean, pc_max, pc_seed = extract_bw_features(bw_phastcons, chrom, start, end)
        new_feats[i, 0] = pc_mean
        new_feats[i, 1] = pc_max
        new_feats[i, 2] = pc_seed

        # GERP++
        if use_gerp:
            # GERP uses different chrom naming (no 'chr' prefix for Ensembl)
            gerp_chrom = chrom.replace("chr", "") if chrom.startswith("chr") else chrom
            gp_mean, gp_max, gp_seed = extract_bw_features(bw_gerp, gerp_chrom, start, end)
            new_feats[i, 3] = gp_mean
            new_feats[i, 4] = gp_max
            new_feats[i, 5] = gp_seed

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            logger.info("    [%5.1f%%] %d/%d (%.0f/s, valid=%d)",
                        100.0 * (i + 1) / n_samples, i + 1, n_samples, rate, valid_count)

    elapsed = time.time() - t0
    logger.info("  Computed %d features in %.1fs (%d valid of %d, %.0f/s)",
                n_new, elapsed, valid_count, n_samples,
                n_samples / elapsed if elapsed > 0 else 0)

    merged = np.concatenate([prev_arr, new_feats], axis=1).astype(np.float32)
    logger.info("  Merged shape: %s -> %s", prev_arr.shape, merged.shape)

    np.save(output_npy, merged)
    size_mb = output_npy.stat().st_size / (1024 * 1024)
    logger.info("  Saved to %s (%.1f MB)", output_npy.name, size_mb)

    bw_phastcons.close()
    if bw_gerp:
        bw_gerp.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-compute v15 conservation features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--phastcons-bw", type=Path,
                        default=Path("data/conservation/hg38.phastCons100way.bw"))
    parser.add_argument("--gerp-bw", type=Path,
                        default=Path("data/conservation/gerp_conservation_scores.homo_sapiens.GRCh38.bw"))
    parser.add_argument("--no-gerp", action="store_true",
                        help="Skip GERP++ features (if bigWig not available)")
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    splits = [args.split] if args.split else ["train", "val", "test"]

    gerp_path = None if args.no_gerp else args.gerp_bw
    if gerp_path and not gerp_path.exists():
        logger.warning("GERP++ bigWig not found: %s. Skipping GERP features.", gerp_path)
        gerp_path = None

    feat_names = ["phastcons_mean", "phastcons_max", "phastcons_seed_mean"]
    if gerp_path:
        feat_names += ["gerp_mean", "gerp_max", "gerp_seed_mean"]

    print()
    print("DeepExoMir -- v15 Feature Pre-computation (Conservation)")
    print("=" * 60)
    print(f"Data directory : {data_dir}")
    print(f"PhastCons bw   : {args.phastcons_bw}")
    print(f"GERP++ bw      : {gerp_path or 'SKIPPED'}")
    print(f"New features   : {len(feat_names)} ({', '.join(feat_names)})")
    print(f"Splits         : {', '.join(splits)}")
    print()

    for split in splits:
        coords_parquet = data_dir / f"{split}_with_coords.parquet"
        prev_npy = data_dir / f"{split}_structural_features_v14alt2L.npy"
        output_npy = data_dir / f"{split}_structural_features_v15.npy"

        if not coords_parquet.exists():
            logger.warning("Skipping %s: coords parquet not found", split)
            continue
        if not prev_npy.exists():
            logger.warning("Skipping %s: v14alt2L features not found", split)
            continue

        logger.info("Processing %s ...", split)
        precompute_split(coords_parquet, prev_npy, output_npy,
                         args.phastcons_bw, gerp_path)

    print()
    print("Done! v15 features saved as *_structural_features_v15.npy")


if __name__ == "__main__":
    main()
