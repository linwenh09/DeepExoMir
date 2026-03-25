"""Download miRBench raw TSV files from Zenodo (with genomic coordinates).

These files contain chr, start, end, strand columns needed for
conservation scores and transcript-level features.

Usage:
    python scripts/download_mirbench_raw.py [--output-dir data/mirbench_raw]
"""
import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

ZENODO_BASE = "https://zenodo.org/api/records/14501607/files"

FILES = [
    "AGO2_eCLIP_Manakov2022_train.tsv.gz",
    "AGO2_eCLIP_Manakov2022_test.tsv.gz",
    "AGO2_CLASH_Hejret2023_train.tsv.gz",
    "AGO2_CLASH_Hejret2023_test.tsv.gz",
    "AGO2_eCLIP_Klimentova2022_test.tsv.gz",
]


def download_file(url: str, dest: Path) -> None:
    """Download with progress reporting."""
    print(f"  Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"    -> {size_mb:.1f} MB")


def decompress_gz(gz_path: Path) -> Path:
    """Decompress .gz file and return path to decompressed file."""
    out_path = gz_path.with_suffix("")  # remove .gz
    if out_path.exists():
        print(f"  Already decompressed: {out_path.name}")
        return out_path
    print(f"  Decompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"    -> {size_mb:.1f} MB")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Download miRBench raw TSVs from Zenodo")
    parser.add_argument("--output-dir", type=Path, default=Path("data/mirbench_raw"))
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {out_dir}")
    print()

    for fname in FILES:
        gz_path = out_dir / fname
        if gz_path.exists():
            print(f"  Already downloaded: {fname}")
        else:
            url = f"{ZENODO_BASE}/{fname}/content"
            download_file(url, gz_path)

        # Decompress
        decompress_gz(gz_path)

    print()
    print("Done! All miRBench raw TSVs downloaded and decompressed.")
    print()

    # Quick peek at columns
    import pandas as pd
    sample_file = out_dir / "AGO2_CLASH_Hejret2023_train.tsv"
    if sample_file.exists():
        df = pd.read_csv(sample_file, sep="\t", nrows=5)
        print(f"Sample columns from {sample_file.name}:")
        print(f"  {list(df.columns)}")
        print()
        print(df.head(3).to_string())


if __name__ == "__main__":
    main()
