"""Compute genomic context features from GENCODE annotation.

New features:
  - log10_utr3_length: log10 of 3'UTR length containing the site
  - log10_min_dist_to_end: log10 of min distance to nearest 3'UTR end
  - relative_position: fractional position within the 3'UTR (0=start, 1=end)

Usage:
    python scripts/compute_genomic_context.py
"""
import sys
import time
import gzip
from pathlib import Path
from collections import defaultdict

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd

GTF_PATH = Path("data/conservation/gencode.v46.annotation.gtf.gz")
DATA_DIR = Path("data/processed")


def parse_gencode_utrs(gtf_path):
    """Parse GENCODE GTF to extract 3'UTR regions per transcript.

    GENCODE v46 uses 'UTR' (not 'three_prime_UTR'). We identify 3'UTR by
    comparing UTR position to stop_codon position per transcript:
    - + strand: UTR after stop_codon = 3'UTR
    - - strand: UTR before stop_codon = 3'UTR
    """
    print(f"Parsing GENCODE GTF: {gtf_path}")
    t0 = time.time()

    # First pass: collect stop_codon positions per transcript
    stop_codons = {}  # transcript_id -> (chr, stop_pos, strand)
    utrs = []  # list of (chr, start, end, strand, transcript_id)

    with gzip.open(gtf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9:
                continue

            feature_type = parts[2]
            if feature_type not in ("UTR", "stop_codon"):
                continue

            chrom = parts[0]
            start = int(parts[3]) - 1  # GTF 1-based -> 0-based
            end = int(parts[4])
            strand = parts[6]

            # Parse transcript_id
            attrs = parts[8]
            transcript_id = ""
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith("transcript_id"):
                    transcript_id = attr.split('"')[1]
                    break

            if feature_type == "stop_codon":
                stop_codons[transcript_id] = (chrom, start, end, strand)
            else:
                utrs.append((chrom, start, end, strand, transcript_id))

    # Second pass: classify UTRs as 3' or 5'
    utr3_regions = []
    for chrom, start, end, strand, tid in utrs:
        if tid not in stop_codons:
            continue
        _, sc_start, sc_end, sc_strand = stop_codons[tid]
        # 3'UTR: downstream of stop codon
        if strand == "+":
            if start >= sc_start:  # UTR starts at or after stop codon
                utr3_regions.append({
                    "chr": chrom, "start": start, "end": end,
                    "strand": strand, "transcript_id": tid,
                    "length": end - start,
                })
        else:  # - strand
            if end <= sc_end:  # UTR ends at or before stop codon
                utr3_regions.append({
                    "chr": chrom, "start": start, "end": end,
                    "strand": strand, "transcript_id": tid,
                    "length": end - start,
                })

    df = pd.DataFrame(utr3_regions)
    elapsed = time.time() - t0
    if len(df) > 0:
        print(f"  Parsed {len(df)} 3'UTR regions in {elapsed:.1f}s")
        print(f"  Unique transcripts: {df['transcript_id'].nunique()}")
    else:
        print(f"  WARNING: No 3'UTR regions found! ({elapsed:.1f}s)")
    return df


def compute_context_features(coords_df, utr3_df):
    """Compute genomic context features for each sample."""
    n = len(coords_df)
    features = np.zeros((n, 3), dtype=np.float32)

    # Build spatial index: chr -> sorted list of (start, end, length) for 3'UTRs
    utr_index = defaultdict(list)
    for _, row in utr3_df.iterrows():
        utr_index[row["chr"]].append((row["start"], row["end"], row["length"]))

    # Sort each chromosome's UTRs by start position
    for chrom in utr_index:
        utr_index[chrom].sort()

    t0 = time.time()
    found = 0

    for i in range(n):
        raw_chrom = str(coords_df.iloc[i].get("chr", ""))
        # Ensure UCSC-style "chr" prefix to match GENCODE
        chrom = raw_chrom if raw_chrom.startswith("chr") else f"chr{raw_chrom}"
        site_start = coords_df.iloc[i].get("genomic_start", np.nan)
        site_end = coords_df.iloc[i].get("genomic_end", np.nan)

        if pd.isna(site_start) or pd.isna(site_end) or not chrom:
            continue

        site_start = int(site_start)
        site_end = int(site_end)
        site_mid = (site_start + site_end) // 2

        # Find overlapping 3'UTR
        best_utr = None
        best_length = 0
        utrs = utr_index.get(chrom, [])

        for utr_start, utr_end, utr_len in utrs:
            if utr_start > site_end:
                break  # past our site
            if utr_end < site_start:
                continue  # before our site
            # Overlap found
            if utr_len > best_length:
                best_utr = (utr_start, utr_end, utr_len)
                best_length = utr_len

        if best_utr is not None:
            utr_start, utr_end, utr_len = best_utr
            found += 1

            # Feature 1: log10(3'UTR length)
            features[i, 0] = np.log10(max(utr_len, 1))

            # Feature 2: log10(min distance to nearest 3'UTR end)
            dist_to_start = abs(site_mid - utr_start)
            dist_to_end = abs(utr_end - site_mid)
            min_dist = max(min(dist_to_start, dist_to_end), 1)
            features[i, 1] = np.log10(min_dist)

            # Feature 3: relative position within 3'UTR (0-1)
            if utr_len > 0:
                features[i, 2] = (site_mid - utr_start) / utr_len
            else:
                features[i, 2] = 0.5
        else:
            # Not in a 3'UTR - use defaults
            features[i, 0] = 0.0  # no UTR
            features[i, 1] = 0.0  # no distance
            features[i, 2] = 0.5  # neutral position

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    [{100*(i+1)/n:.1f}%] {i+1}/{n} ({rate:.0f}/s, found={found})")

    elapsed = time.time() - t0
    print(f"  Computed context features in {elapsed:.1f}s ({found}/{n} in 3'UTRs)")
    return features


def main():
    print("\nDeepExoMir -- Genomic Context Feature Computation")
    print("=" * 60)

    # Parse GENCODE
    utr3_df = parse_gencode_utrs(GTF_PATH)

    for split in ["train", "val", "test"]:
        coords_path = DATA_DIR / f"{split}_with_coords.parquet"
        if not coords_path.exists():
            print(f"Skipping {split}: no coords parquet")
            continue

        print(f"\nProcessing {split}...")
        coords_df = pd.read_parquet(coords_path, engine="pyarrow")
        print(f"  Loaded {len(coords_df)} samples")

        context_feats = compute_context_features(coords_df, utr3_df)

        # Save standalone context features
        out_path = DATA_DIR / f"{split}_genomic_context.npy"
        np.save(out_path, context_feats)
        print(f"  Saved {out_path.name} (shape={context_feats.shape})")

        # Also create v16c: v16a (23 feat) + context (3 feat) = 26 features
        v16a = np.load(DATA_DIR / f"{split}_structural_features_v16a.npy")
        v16c = np.concatenate([v16a, context_feats], axis=1).astype(np.float32)
        np.save(DATA_DIR / f"{split}_structural_features_v16c.npy", v16c)
        print(f"  Saved v16c: {v16c.shape} (v16a + context)")

    print("\nDone!")


if __name__ == "__main__":
    main()
