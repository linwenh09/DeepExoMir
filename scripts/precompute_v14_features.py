"""Pre-compute v14 conservation + transcript features.

Reads *_with_coords.parquet files (from map_genomic_coordinates.py) and
the PhyloP bigWig file to compute per-site conservation scores.
Also extracts transcript-level features from GENCODE GTF.

New features (v14):
    34. phylop_mean       Mean PhyloP score over the 50nt target site
    35. phylop_max        Max PhyloP score (most conserved position)
    36. phylop_seed_mean  Mean PhyloP score over seed region (last 8nt)
    37. site_in_3utr      1.0 if site is in 3'UTR, else 0.0
    38. site_in_cds       1.0 if site is in CDS/exon, else 0.0

Usage:
    python scripts/precompute_v14_features.py --data-dir data/processed
"""

import argparse
import gzip
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

V14_FEATURE_NAMES = [
    "phylop_mean", "phylop_max", "phylop_seed_mean",
    "site_in_3utr", "site_in_cds",
]

V14_DEFAULTS = [0.0, 0.0, 0.0, 0.0, 0.0]

CONSERVATION_DIR = Path("data/conservation")
PHYLOP_PATH = CONSERVATION_DIR / "hg38.phyloP100way.bw"


def load_phylop():
    """Load PhyloP bigWig file using pybigtools."""
    try:
        import pybigtools
        bw = pybigtools.open(str(PHYLOP_PATH))
        logger.info("Loaded PhyloP bigWig: %s", PHYLOP_PATH)
        return bw
    except ImportError:
        logger.error("pybigtools not installed. Run: pip install pybigtools")
        return None
    except Exception as e:
        logger.error("Failed to open PhyloP bigWig: %s", e)
        return None


def get_phylop_scores(bw, chrom: str, start: int, end: int) -> np.ndarray:
    """Get per-base PhyloP scores for a genomic region.

    Returns array of shape (end - start,) with conservation scores.
    Missing values are filled with 0.0.
    """
    try:
        # pybigtools uses "chr1" format
        if not chrom.startswith("chr"):
            chrom = f"chr{chrom}"

        vals = bw.values(chrom, int(start), int(end))
        if vals is None:
            return np.zeros(end - start, dtype=np.float32)

        arr = np.array(vals, dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr
    except Exception:
        return np.zeros(max(1, end - start), dtype=np.float32)


def precompute_split(
    coords_parquet: Path,
    prev_npy: Path,
    output_npy: Path,
    bw,
) -> None:
    """Compute v14 features for one split."""
    prev_arr = np.load(prev_npy)
    logger.info("  Loaded previous features: %s (shape=%s)", prev_npy.name, prev_arr.shape)
    n_samples = prev_arr.shape[0]

    df = pd.read_parquet(coords_parquet, engine="pyarrow")
    assert len(df) == n_samples, f"Mismatch: {len(df)} vs {n_samples}"

    new_feats = np.zeros((n_samples, len(V14_FEATURE_NAMES)), dtype=np.float32)

    logger.info("  Computing %d features for %d samples ...", len(V14_FEATURE_NAMES), n_samples)
    t0 = time.time()

    chrs = df["chr"].values
    starts = df["genomic_start"].values
    ends = df["genomic_end"].values
    features_col = df["genomic_feature"].values if "genomic_feature" in df.columns else [""]*n_samples

    for i in range(n_samples):
        chrom = str(chrs[i]) if pd.notna(chrs[i]) else ""
        start = starts[i]
        end = ends[i]
        feat = str(features_col[i]) if pd.notna(features_col[i]) else ""

        if chrom and pd.notna(start) and pd.notna(end) and bw is not None:
            start_int = int(start)
            end_int = int(end)

            scores = get_phylop_scores(bw, chrom, start_int, end_int)
            if len(scores) > 0:
                new_feats[i, 0] = float(np.mean(scores))    # phylop_mean
                new_feats[i, 1] = float(np.max(scores))     # phylop_max

                # Seed region: last 8 positions of the 50nt site
                seed_scores = scores[-8:] if len(scores) >= 8 else scores
                new_feats[i, 2] = float(np.mean(seed_scores))  # phylop_seed_mean

        # Genomic feature annotation
        feat_lower = feat.lower()
        if "3utr" in feat_lower or "three_prime" in feat_lower:
            new_feats[i, 3] = 1.0  # site_in_3utr
        if "cds" in feat_lower or "exon" in feat_lower:
            new_feats[i, 4] = 1.0  # site_in_cds

        if (i + 1) % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n_samples - i - 1) / rate
            logger.info(
                "    [%5.1f%%] %d/%d (%.0f/s, ETA: %.0fs)",
                100.0 * (i + 1) / n_samples, i + 1, n_samples, rate, eta,
            )

    elapsed = time.time() - t0
    logger.info(
        "  Computed %d features in %.1fs (%.0f samples/s)",
        len(V14_FEATURE_NAMES), elapsed, n_samples / elapsed if elapsed > 0 else 0,
    )

    # Concatenate
    merged = np.concatenate([prev_arr, new_feats], axis=1)
    logger.info("  Merged shape: %s -> %s", prev_arr.shape, merged.shape)

    np.save(output_npy, merged)
    size_mb = output_npy.stat().st_size / (1024 * 1024)
    logger.info("  Saved to %s (%.1f MB)", output_npy.name, size_mb)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute v14 conservation features")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--prev-version", type=str, default="v13",
                        help="Previous feature version to build on (default: v13)")
    parser.add_argument("--split", type=str, default=None)
    args = parser.parse_args()

    data_dir: Path = args.data_dir
    prev_ver = args.prev_version
    splits = [args.split] if args.split else ["train", "val", "test"]

    print()
    print("DeepExoMir -- v14 Feature Pre-computation (Conservation)")
    print("=" * 60)
    print(f"Data directory : {data_dir}")
    print(f"Previous ver   : {prev_ver}")
    print(f"PhyloP bigWig  : {PHYLOP_PATH}")
    print(f"New features   : {len(V14_FEATURE_NAMES)} ({', '.join(V14_FEATURE_NAMES)})")
    print()

    # Load PhyloP
    bw = load_phylop()
    if bw is None:
        logger.warning("PhyloP not available -- conservation features will be 0.0")

    for split in splits:
        coords_parquet = data_dir / f"{split}_with_coords.parquet"
        prev_npy = data_dir / f"{split}_structural_features_{prev_ver}.npy"
        output_npy = data_dir / f"{split}_structural_features_v14.npy"

        if not coords_parquet.exists():
            logger.warning("Skipping %s: coords parquet not found", split)
            continue
        if not prev_npy.exists():
            # Fall back to v12 if v13 not ready
            prev_npy_v12 = data_dir / f"{split}_structural_features_v12.npy"
            if prev_npy_v12.exists():
                logger.info("v13 not ready, using v12 for %s", split)
                prev_npy = prev_npy_v12
            else:
                logger.warning("Skipping %s: previous features not found", split)
                continue

        logger.info("Processing %s ...", split)
        precompute_split(coords_parquet, prev_npy, output_npy, bw)

    if bw is not None:
        bw.close()

    print()
    print("Done! v14 features saved as *_structural_features_v14.npy")


if __name__ == "__main__":
    main()
