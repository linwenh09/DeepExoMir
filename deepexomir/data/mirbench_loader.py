"""Wrapper around the miRBench Python package for loading CLIP datasets.

miRBench provides standardised miRNA-target interaction datasets derived
from AGO2-CLIP experiments.  This module loads them, converts DNA
sequences to RNA (T -> U), and returns unified DataFrames ready for
downstream processing.

Supported datasets
------------------
- AGO2_CLASH_Hejret2023
- AGO2_eCLIP_Klimentova2022
- AGO2_eCLIP_Manakov2022

Reference:
    Hejret et al. (2023) - miRBench benchmark suite.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Datasets to load by default
DEFAULT_DATASETS = [
    "AGO2_CLASH_Hejret2023",
    "AGO2_eCLIP_Klimentova2022",
    "AGO2_eCLIP_Manakov2022",
]

# Expected column names in miRBench DataFrames
_MIRNA_COL = "noncodingRNA"
_TARGET_COL = "gene"
_LABEL_COL = "label"

# Unified column names for our pipeline
UNIFIED_COLUMNS = {
    _MIRNA_COL: "mirna_seq",
    _TARGET_COL: "target_seq",
    _LABEL_COL: "label",
}


def _dna_to_rna_series(series: pd.Series) -> pd.Series:
    """Convert all DNA sequences in a Series to RNA (T -> U)."""
    return series.str.upper().str.replace("T", "U", regex=False)


def load_mirbench_dataset(
    dataset_name: str,
    split: str = "test",
) -> pd.DataFrame:
    """Load a single miRBench dataset and convert to RNA.

    Parameters
    ----------
    dataset_name : str
        Name of the miRBench dataset, e.g. ``"AGO2_CLASH_Hejret2023"``.
    split : str
        Which split to load (``"train"`` or ``"test"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``mirna_seq``, ``target_seq``, ``label``,
        and ``source``.

    Raises
    ------
    ImportError
        If the ``miRBench`` package is not installed.
    ValueError
        If the dataset does not contain expected columns.
    """
    try:
        from miRBench.dataset import get_dataset_df
    except ImportError as exc:
        raise ImportError(
            "The 'miRBench' package is required but not installed.  "
            "Install it with: pip install miRBench"
        ) from exc

    logger.info("Loading miRBench dataset: %s (split=%s)", dataset_name, split)
    df = get_dataset_df(dataset_name, split=split)

    # Validate expected columns
    missing = {_MIRNA_COL, _TARGET_COL, _LABEL_COL} - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing expected columns: {missing}.  "
            f"Available columns: {list(df.columns)}"
        )

    # Rename to unified schema
    df = df.rename(columns=UNIFIED_COLUMNS)

    # Convert DNA -> RNA
    df["mirna_seq"] = _dna_to_rna_series(df["mirna_seq"])
    df["target_seq"] = _dna_to_rna_series(df["target_seq"])

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Tag the source
    df["source"] = dataset_name

    # Keep only the columns we need (plus any extras miRBench provides)
    required = ["mirna_seq", "target_seq", "label", "source"]
    extra_cols = [c for c in df.columns if c not in required]
    df = df[required + extra_cols]

    logger.info(
        "Loaded %d rows from %s (positives=%d, negatives=%d)",
        len(df),
        dataset_name,
        (df["label"] == 1).sum(),
        (df["label"] == 0).sum(),
    )
    return df


def load_all_mirbench(
    dataset_names: Optional[list[str]] = None,
    split: str = "test",
) -> pd.DataFrame:
    """Load and merge multiple miRBench datasets.

    Parameters
    ----------
    dataset_names : list[str], optional
        List of dataset names to load.  Defaults to :data:`DEFAULT_DATASETS`.
    split : str
        Which split to load.

    Returns
    -------
    pd.DataFrame
        Concatenated DataFrame with all datasets, including a ``source``
        column identifying each row's origin.
    """
    names = dataset_names or DEFAULT_DATASETS
    frames: list[pd.DataFrame] = []

    for name in names:
        try:
            df = load_mirbench_dataset(name, split=split)
            frames.append(df)
        except Exception as exc:
            logger.error("Failed to load miRBench dataset '%s': %s", name, exc)

    if not frames:
        logger.warning("No miRBench datasets were loaded successfully.")
        return pd.DataFrame(columns=["mirna_seq", "target_seq", "label", "source"])

    merged = pd.concat(frames, ignore_index=True)

    # Drop exact duplicates (same miRNA + target + label)
    n_before = len(merged)
    merged = merged.drop_duplicates(
        subset=["mirna_seq", "target_seq", "label"], keep="first"
    )
    n_dropped = n_before - len(merged)
    if n_dropped > 0:
        logger.info("Dropped %d cross-dataset duplicate rows.", n_dropped)

    logger.info(
        "Merged miRBench data: %d rows from %d datasets.", len(merged), len(frames)
    )
    return merged


def get_mirbench_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict for a miRBench-loaded DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame loaded by :func:`load_mirbench_dataset` or
        :func:`load_all_mirbench`.

    Returns
    -------
    dict
        Summary statistics including counts per source and label.
    """
    summary: dict = {
        "total_rows": len(df),
        "n_positive": int((df["label"] == 1).sum()),
        "n_negative": int((df["label"] == 0).sum()),
        "sources": {},
    }

    if "source" in df.columns:
        for source, grp in df.groupby("source"):
            summary["sources"][source] = {
                "total": len(grp),
                "positive": int((grp["label"] == 1).sum()),
                "negative": int((grp["label"] == 0).sum()),
            }

    return summary
