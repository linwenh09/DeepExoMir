"""Download PhyloP 100-way conservation scores and GENCODE annotation.

Downloads:
  1. hg38.phyloP100way.bw  (9.2 GB) - per-base conservation scores
  2. gencode.v46.annotation.gtf.gz (~50 MB) - gene/transcript annotation

Usage:
    python scripts/download_phylop_gencode.py [--output-dir data/conservation]
"""

import argparse
import urllib.request
from pathlib import Path


DOWNLOADS = {
    "hg38.phyloP100way.bw": (
        "https://hgdownload.cse.ucsc.edu/goldenpath/hg38/phyloP100way/hg38.phyloP100way.bw",
        "9.2 GB",
    ),
    "gencode.v46.annotation.gtf.gz": (
        "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz",
        "~50 MB",
    ),
}


def download_with_progress(url: str, dest: Path, desc: str) -> None:
    """Download a file with progress reporting."""
    print(f"  Downloading {desc} ...")
    print(f"  URL: {url}")
    print(f"  -> {dest}")

    def _reporthook(count, block_size, total_size):
        downloaded = count * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 / total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  {pct:5.1f}%  ({mb:.0f}/{total_mb:.0f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\n  Done! {size_mb:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Download PhyloP + GENCODE")
    parser.add_argument("--output-dir", type=Path, default=Path("data/conservation"))
    parser.add_argument(
        "--skip-phylop", action="store_true",
        help="Skip the large PhyloP download (9.2 GB)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("DeepExoMir -- Download Conservation Data")
    print("=" * 45)
    print(f"Output directory: {out_dir}")
    print()

    for fname, (url, size_desc) in DOWNLOADS.items():
        if args.skip_phylop and "phyloP" in fname:
            print(f"  Skipping {fname} ({size_desc}) [--skip-phylop]")
            continue

        dest = out_dir / fname
        if dest.exists():
            print(f"  Already exists: {fname} ({dest.stat().st_size / (1024*1024):.1f} MB)")
            continue

        download_with_progress(url, dest, f"{fname} ({size_desc})")

    print()
    print("Done!")
    print()
    print("Next steps:")
    print("  pip install pyBigWig  # for reading bigWig files")
    print("  python scripts/precompute_v14_features.py --data-dir data/processed")


if __name__ == "__main__":
    main()
