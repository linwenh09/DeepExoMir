"""Main preprocessing pipeline for DeepExoMir.

Orchestrates the full data preparation workflow:

1. Parse miRBase mature.fa into a sequence dictionary (Homo sapiens only)
2. Parse miRTarBase Excel file, filtering for human + strong evidence
3. Load miRBench CLIP datasets
4. Generate negative samples via seed-region shuffling
5. Merge all data sources into a unified DataFrame
6. Perform gene-level train/val/test split (70/15/15)
7. Save final datasets as Parquet files

Usage::

    from deepexomir.data.preprocess import run_preprocessing_pipeline
    run_preprocessing_pipeline(data_dir=Path("data"))
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from deepexomir.data.mirbench_loader import load_all_mirbench
from deepexomir.data.negative_sampling import generate_negatives
from deepexomir.utils.sequence import clean_sequence, dna_to_rna

logger = logging.getLogger(__name__)

# Default directories relative to project root
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# miRTarBase evidence levels considered "strong"
STRONG_EVIDENCE = {
    "Luciferase reporter assay",
    "Reporter assay",
    "Western blot",
    "qRT-PCR",
    "Microarray",
    "Proteomics",
    "NGS",
    "Northern blot",
    "HITS-CLIP",
    "PAR-CLIP",
    "CLASH",
    "CLIP-seq",
    "RIP-seq",
    "Degradome-seq",
    "Degradome sequencing",
    "IMPACT-seq",
}


# ============================================================================
# Step 1: Parse miRBase mature.fa
# ============================================================================


def parse_mirbase_fasta(
    fasta_path: Path,
    species_prefix: str = "hsa",
) -> dict[str, str]:
    """Parse miRBase mature.fa and extract sequences for a species.

    Parameters
    ----------
    fasta_path : Path
        Path to the ``mature.fa`` FASTA file.
    species_prefix : str
        Species prefix to filter on (default ``"hsa"`` for Homo sapiens).

    Returns
    -------
    dict[str, str]
        Mapping of miRNA ID (e.g. ``"hsa-miR-21-5p"``) to cleaned RNA
        sequence.
    """
    if not fasta_path.exists():
        raise FileNotFoundError(f"miRBase FASTA not found: {fasta_path}")

    sequences: dict[str, str] = {}
    current_id: Optional[str] = None

    with open(fasta_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Header line: >hsa-let-7a-5p MIMAT0000062 ...
                parts = line[1:].split()
                mirna_id = parts[0]
                if mirna_id.startswith(species_prefix + "-"):
                    current_id = mirna_id
                else:
                    current_id = None
            elif current_id is not None:
                seq = clean_sequence(line)
                if seq:
                    # Accumulate in case sequence spans multiple lines
                    if current_id in sequences:
                        sequences[current_id] += seq
                    else:
                        sequences[current_id] = seq

    logger.info(
        "Parsed %d %s miRNA sequences from %s.",
        len(sequences),
        species_prefix,
        fasta_path,
    )
    return sequences


# ============================================================================
# Step 2: Parse miRTarBase
# ============================================================================


def parse_mirtarbase(
    xlsx_path: Path,
    species: str = "Homo sapiens",
    strong_only: bool = True,
) -> pd.DataFrame:
    """Parse miRTarBase Excel file and filter for human, strong evidence.

    Parameters
    ----------
    xlsx_path : Path
        Path to the miRTarBase MTI Excel file (e.g. ``hsa_MTI.xlsx``).
    species : str
        Species to filter on.
    strong_only : bool
        If True, keep only interactions supported by strong experimental
        evidence (e.g. luciferase, western blot, CLIP).

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with columns: ``mirna_id``, ``mirna_seq``,
        ``target_gene``, ``target_seq``, ``evidence``, ``source``.
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"miRTarBase file not found: {xlsx_path}")

    logger.info("Reading miRTarBase from %s ...", xlsx_path)
    df = pd.read_excel(xlsx_path, engine="openpyxl")

    # Normalise column names (miRTarBase uses varying capitalisation)
    # Known columns in miRTarBase 2025:
    #   'miRTarBase ID', 'miRNA', 'Species (miRNA)', 'Target Gene',
    #   'Target Gene (Entrez ID)', 'Species (Target Gene)',
    #   'Experiments', 'Support Type', 'References (PMID)'
    col_map = {}
    evidence_mapped = False
    for col in df.columns:
        col_lower = col.strip().lower().replace(" ", "_")
        if col_lower in ("mirna", "mirna_id") or (
            "mirna" in col_lower
            and "species" not in col_lower
            and "mirtarbase" not in col_lower
        ):
            col_map[col] = "mirna_id"
        elif "target_gene" in col_lower and "entrez" not in col_lower and "species" not in col_lower:
            col_map[col] = "target_gene"
        elif "experiment" in col_lower and not evidence_mapped:
            col_map[col] = "evidence"
            evidence_mapped = True
        elif "support" in col_lower:
            col_map[col] = "support_type"
        elif "species" in col_lower and "mirna" in col_lower:
            col_map[col] = "species_mirna"
        elif "species" in col_lower and "target" in col_lower:
            col_map[col] = "species_target"
        elif "species" in col_lower:
            col_map[col] = "species"

    df = df.rename(columns=col_map)
    logger.info("Column mapping: %s", col_map)

    # Filter by species
    species_col = None
    for candidate in ["species_mirna", "species_target", "species"]:
        if candidate in df.columns:
            species_col = candidate
            break

    if species_col:
        df = df[df[species_col].str.contains(species, case=False, na=False)]
        df = df.reset_index(drop=True)
        logger.info("After species filter (%s): %d rows.", species, len(df))

    # Filter by evidence strength
    if strong_only and "evidence" in df.columns:
        mask = df["evidence"].apply(
            lambda x: _has_strong_evidence(x) if isinstance(x, str) else False
        )
        df = df[mask]
        df = df.reset_index(drop=True)
        logger.info("After strong-evidence filter: %d rows.", len(df))

    # De-duplicate
    dedup_cols = [c for c in ["mirna_id", "target_gene"] if c in df.columns]
    if dedup_cols:
        n_before = len(df)
        df = df.drop_duplicates(subset=dedup_cols, keep="first")
        logger.info(
            "De-duplicated on %s: %d -> %d rows.", dedup_cols, n_before, len(df)
        )

    df["source"] = "miRTarBase"
    return df.reset_index(drop=True)


def _has_strong_evidence(evidence_str: str) -> bool:
    """Check if an evidence string contains at least one strong method."""
    methods = re.split(r"[;/,]", evidence_str)
    return any(m.strip() in STRONG_EVIDENCE for m in methods)


# ============================================================================
# Step 3-5: Load, merge, and generate negatives
# ============================================================================


def _prepare_mirtarbase_pairs(
    mirtarbase_df: pd.DataFrame,
    mirbase_seqs: dict[str, str],
) -> pd.DataFrame:
    """Map miRTarBase entries to miRNA sequences from miRBase.

    Parameters
    ----------
    mirtarbase_df : pd.DataFrame
        Parsed miRTarBase DataFrame with ``mirna_id`` column.
    mirbase_seqs : dict[str, str]
        miRBase ID-to-sequence mapping.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``mirna_seq``, ``target_gene``, ``label``, ``source``.
        Note: target_seq is left empty here since miRTarBase provides gene
        symbols, not binding-site sequences.
    """
    rows = []
    unmatched = 0

    for _, row in mirtarbase_df.iterrows():
        mirna_id = row.get("mirna_id", "")
        if not isinstance(mirna_id, str):
            continue

        # Try exact match first, then case-insensitive
        seq = mirbase_seqs.get(mirna_id)
        if seq is None:
            mirna_id_lower = mirna_id.lower()
            for mid, mseq in mirbase_seqs.items():
                if mid.lower() == mirna_id_lower:
                    seq = mseq
                    break

        if seq is None:
            unmatched += 1
            continue

        rows.append(
            {
                "mirna_id": mirna_id,
                "mirna_seq": seq,
                "target_gene": row.get("target_gene", ""),
                "label": 1,
                "source": "miRTarBase",
            }
        )

    if unmatched > 0:
        logger.warning(
            "%d miRTarBase entries could not be matched to miRBase sequences.",
            unmatched,
        )

    return pd.DataFrame(rows)


def merge_data_sources(
    mirbench_df: pd.DataFrame,
    mirtarbase_pairs: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Merge miRBench and miRTarBase data into a unified DataFrame.

    Parameters
    ----------
    mirbench_df : pd.DataFrame
        DataFrame from :func:`load_all_mirbench`.
    mirtarbase_pairs : pd.DataFrame, optional
        DataFrame from :func:`_prepare_mirtarbase_pairs`.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with columns: ``mirna_seq``, ``target_seq``,
        ``label``, ``source``, and optionally ``mirna_id``, ``target_gene``.
    """
    frames = [mirbench_df]

    if mirtarbase_pairs is not None and len(mirtarbase_pairs) > 0:
        # miRTarBase rows may not have target_seq
        if "target_seq" not in mirtarbase_pairs.columns:
            mirtarbase_pairs["target_seq"] = ""
        frames.append(mirtarbase_pairs)

    merged = pd.concat(frames, ignore_index=True)

    # Remove rows with missing essential columns
    merged = merged.dropna(subset=["mirna_seq"])
    merged = merged[merged["mirna_seq"].str.len() > 0]

    logger.info("Merged dataset: %d total rows.", len(merged))
    return merged


# ============================================================================
# Step 6: Gene-level train/val/test split
# ============================================================================


def gene_level_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_seed: int = 42,
    gene_col: str = "target_gene",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data by gene to prevent data leakage.

    All interactions involving the same target gene appear in the same
    split, preventing the model from memorising gene-specific patterns
    during training and then being tested on the same genes.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset.
    train_ratio : float
        Fraction for training set.
    val_ratio : float
        Fraction for validation set.
    test_ratio : float
        Fraction for test set.
    random_seed : int
        Random seed for reproducibility.
    gene_col : str
        Column name containing gene identifiers.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        "Split ratios must sum to 1.0"
    )

    rng = np.random.RandomState(random_seed)

    # If gene_col is missing or empty, fall back to row-level split
    if gene_col not in df.columns or df[gene_col].isna().all():
        logger.warning(
            "Column '%s' not found or all NaN.  Falling back to row-level split.",
            gene_col,
        )
        df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        n = len(df)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_df = df.iloc[:n_train]
        val_df = df.iloc[n_train : n_train + n_val]
        test_df = df.iloc[n_train + n_val :]
        return train_df, val_df, test_df

    # Fill NaN genes with a placeholder so they don't get dropped
    df = df.copy()
    df[gene_col] = df[gene_col].fillna("_UNKNOWN_GENE_")

    # Get unique genes and shuffle
    unique_genes = df[gene_col].unique().tolist()
    rng.shuffle(unique_genes)

    n_genes = len(unique_genes)
    n_train_genes = int(n_genes * train_ratio)
    n_val_genes = int(n_genes * val_ratio)

    train_genes = set(unique_genes[:n_train_genes])
    val_genes = set(unique_genes[n_train_genes : n_train_genes + n_val_genes])
    test_genes = set(unique_genes[n_train_genes + n_val_genes :])

    train_df = df[df[gene_col].isin(train_genes)].reset_index(drop=True)
    val_df = df[df[gene_col].isin(val_genes)].reset_index(drop=True)
    test_df = df[df[gene_col].isin(test_genes)].reset_index(drop=True)

    logger.info(
        "Gene-level split: %d genes -> train=%d (%d rows), val=%d (%d rows), "
        "test=%d (%d rows).",
        n_genes,
        len(train_genes),
        len(train_df),
        len(val_genes),
        len(val_df),
        len(test_genes),
        len(test_df),
    )

    return train_df, val_df, test_df


# ============================================================================
# Step 7: Save to Parquet
# ============================================================================


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Save train/val/test DataFrames as Parquet files.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Split DataFrames.
    output_dir : Path
        Directory to write Parquet files into.

    Returns
    -------
    dict[str, Path]
        Mapping of split name to file path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Core columns to keep (others from miRBench may have mixed types)
    core_cols = [
        "mirna_id", "mirna_seq", "target_seq", "target_gene",
        "label", "source",
    ]

    paths: dict[str, Path] = {}
    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = output_dir / f"{name}.parquet"

        # Keep only core columns that exist, cast mixed-type columns to str
        available = [c for c in core_cols if c in df.columns]
        df_out = df[available].copy()
        for col in df_out.columns:
            if df_out[col].dtype == object:
                df_out[col] = df_out[col].astype(str)

        df_out.to_parquet(path, index=False, engine="pyarrow")
        paths[name] = path
        logger.info("Saved %s split: %d rows -> %s", name, len(df_out), path)

    return paths


# ============================================================================
# Main pipeline
# ============================================================================


def run_preprocessing_pipeline(
    data_dir: Path = Path("data"),
    species_prefix: str = "hsa",
    include_mirtarbase: bool = True,
    include_mirbench: bool = True,
    negative_ratio: float = 1.0,
    random_seed: int = 42,
) -> dict[str, Path]:
    """Run the complete preprocessing pipeline.

    Parameters
    ----------
    data_dir : Path
        Root data directory containing ``raw/`` and where ``processed/``
        will be created.
    species_prefix : str
        miRBase species prefix (default ``"hsa"`` for human).
    include_mirtarbase : bool
        Whether to include miRTarBase data.
    include_mirbench : bool
        Whether to include miRBench CLIP data.
    negative_ratio : float
        Ratio of negatives to positives for seed-shuffle generation.
    random_seed : int
        Global random seed.

    Returns
    -------
    dict[str, Path]
        Paths to the saved Parquet files.
    """
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"

    # ------------------------------------------------------------------
    # Step 1: Parse miRBase
    # ------------------------------------------------------------------
    mirbase_path = raw_dir / "mature.fa"
    mirbase_seqs = parse_mirbase_fasta(mirbase_path, species_prefix)

    # ------------------------------------------------------------------
    # Step 2: Parse miRTarBase (optional)
    # ------------------------------------------------------------------
    mirtarbase_pairs: Optional[pd.DataFrame] = None
    if include_mirtarbase:
        mirtarbase_path = raw_dir / "hsa_MTI.xlsx"
        if mirtarbase_path.exists():
            mirtarbase_df = parse_mirtarbase(mirtarbase_path)
            mirtarbase_pairs = _prepare_mirtarbase_pairs(
                mirtarbase_df, mirbase_seqs
            )
            logger.info(
                "Prepared %d miRTarBase interaction pairs.",
                len(mirtarbase_pairs),
            )
        else:
            logger.warning(
                "miRTarBase file not found at %s; skipping.", mirtarbase_path
            )

    # ------------------------------------------------------------------
    # Step 3: Load miRBench (both train + test splits)
    # ------------------------------------------------------------------
    mirbench_frames = []
    if include_mirbench:
        for split_name in ["train", "test"]:
            try:
                df_split = load_all_mirbench(split=split_name)
                df_split["mirbench_split"] = split_name
                mirbench_frames.append(df_split)
                logger.info(
                    "Loaded miRBench %s: %d rows.", split_name, len(df_split)
                )
            except Exception as exc:
                logger.error(
                    "Failed to load miRBench %s data: %s", split_name, exc
                )

    if mirbench_frames:
        mirbench_df = pd.concat(mirbench_frames, ignore_index=True)
        # Drop duplicates across splits
        n_before = len(mirbench_df)
        mirbench_df = mirbench_df.drop_duplicates(
            subset=["mirna_seq", "target_seq", "label"], keep="first"
        )
        if len(mirbench_df) < n_before:
            logger.info(
                "Dropped %d cross-split duplicates in miRBench.",
                n_before - len(mirbench_df),
            )
    else:
        mirbench_df = pd.DataFrame(
            columns=["mirna_seq", "target_seq", "label", "source"]
        )

    # ------------------------------------------------------------------
    # Step 4: Merge data sources
    # ------------------------------------------------------------------
    merged = merge_data_sources(mirbench_df, mirtarbase_pairs)

    # ------------------------------------------------------------------
    # Step 5: Filter to rows with valid target_seq for model training
    # ------------------------------------------------------------------
    # miRTarBase entries have gene-level info but no binding-site sequences.
    # Keep only rows with actual target_seq for sequence-pair training.
    has_target = (
        merged["target_seq"].notna()
        & (merged["target_seq"].str.len() > 0)
        & (merged["target_seq"] != "nan")
        & (merged["target_seq"] != "")
    )
    trainable = merged[has_target].copy().reset_index(drop=True)
    annotation_only = merged[~has_target].copy().reset_index(drop=True)

    logger.info(
        "Trainable rows (with target_seq): %d.  "
        "Annotation-only rows (no target_seq): %d.",
        len(trainable),
        len(annotation_only),
    )

    # ------------------------------------------------------------------
    # Step 6: Split the trainable data
    # ------------------------------------------------------------------
    # Use miRNA-based split since miRBench data may not have target_gene
    train_df, val_df, test_df = gene_level_split(
        trainable,
        random_seed=random_seed,
        gene_col="mirna_seq",  # Split by miRNA to avoid leakage
    )

    # ------------------------------------------------------------------
    # Step 7: Save
    # ------------------------------------------------------------------
    paths = save_splits(train_df, val_df, test_df, processed_dir)

    # Save annotation-only data (miRTarBase gene-level interactions)
    if len(annotation_only) > 0:
        anno_path = processed_dir / "mirtarbase_annotations.parquet"
        anno_cols = [c for c in ["mirna_id", "mirna_seq", "target_gene", "label", "source"]
                     if c in annotation_only.columns]
        anno_out = annotation_only[anno_cols].copy()
        for col in anno_out.columns:
            if anno_out[col].dtype == object:
                anno_out[col] = anno_out[col].astype(str)
        anno_out.to_parquet(anno_path, index=False, engine="pyarrow")
        paths["annotations"] = anno_path
        logger.info(
            "Saved annotation-only data: %d rows -> %s",
            len(anno_out), anno_path,
        )

    # Also save the miRBase sequence dictionary for downstream use
    mirbase_out = processed_dir / "mirbase_hsa.parquet"
    mirbase_records = [
        {"mirna_id": mid, "mirna_seq": seq} for mid, seq in mirbase_seqs.items()
    ]
    pd.DataFrame(mirbase_records).to_parquet(
        mirbase_out, index=False, engine="pyarrow"
    )
    paths["mirbase"] = mirbase_out
    logger.info("Saved miRBase sequences: %d entries -> %s", len(mirbase_seqs), mirbase_out)

    return paths
