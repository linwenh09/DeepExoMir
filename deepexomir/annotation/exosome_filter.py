"""Exosome relevance filter for miRNA-target predictions.

Parses the ExoCarta miRNA cargo database to identify which miRNAs have
been experimentally detected in exosomes.  Predictions can then be
annotated with an ``is_exosomal`` flag or filtered to retain only
exosome-relevant interactions.
"""

from __future__ import annotations

import csv
import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ExosomeFilter:
    """Filter and annotate predictions by exosome relevance.

    Parameters
    ----------
    exocarta_path : str or Path, optional
        Path to the ExoCarta miRNA details file.  If provided, the file
        is loaded immediately at construction time.
    """

    def __init__(self, exocarta_path: Optional[str | Path] = None) -> None:
        self.exosome_mirnas: set[str] = set()

        if exocarta_path is not None:
            self.load_exocarta(exocarta_path)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_exocarta(self, path: str | Path) -> None:
        """Parse an ExoCarta miRNA details file.

        The ExoCarta file is typically tab-delimited with a header row.
        The loader searches for a column whose name contains ``"mirna"``
        (case-insensitive) and extracts unique miRNA identifiers, applying
        light normalisation (whitespace stripping).

        Parameters
        ----------
        path : str or Path
            Path to the ExoCarta miRNA details text file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If no miRNA column can be identified.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"ExoCarta file not found: {path}")

        logger.info("Loading ExoCarta miRNA data from %s", path)

        # Attempt pandas-based loading first (handles encoding quirks)
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, low_memory=False)
        except Exception:
            # Fallback: try latin-1 encoding
            df = pd.read_csv(
                path, sep="\t", dtype=str, encoding="latin-1", low_memory=False
            )

        mirna_col = self._find_mirna_column(df)
        raw_ids = df[mirna_col].dropna().str.strip().unique()

        for raw_id in raw_ids:
            normalised = self._normalise_mirna_id(raw_id)
            if normalised:
                self.exosome_mirnas.add(normalised)

        logger.info(
            "Loaded %d unique exosomal miRNAs from ExoCarta", len(self.exosome_mirnas)
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def is_exosomal(self, mirna_id: str) -> bool:
        """Check whether a miRNA is in the exosome set.

        Performs exact matching after whitespace stripping.  If the
        identifier is not found, a secondary lookup without the ``-3p`` /
        ``-5p`` arm suffix is attempted to accommodate minor naming
        differences.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier (e.g., ``"hsa-miR-21-5p"``).

        Returns
        -------
        bool
            ``True`` if the miRNA has been detected in exosomes.
        """
        mirna_id = mirna_id.strip()

        # Exact match
        if mirna_id in self.exosome_mirnas:
            return True

        # Fallback: strip arm designation (-3p / -5p)
        base_id = re.sub(r"-[35]p$", "", mirna_id)
        if base_id in self.exosome_mirnas:
            return True

        # Fallback: check if either arm variant is present
        if f"{base_id}-5p" in self.exosome_mirnas:
            return True
        if f"{base_id}-3p" in self.exosome_mirnas:
            return True

        return False

    # ------------------------------------------------------------------
    # Batch operations
    # ------------------------------------------------------------------

    def filter_predictions(
        self,
        predictions_df: pd.DataFrame,
        mirna_col: str = "mirna_id",
        drop_non_exosomal: bool = False,
    ) -> pd.DataFrame:
        """Annotate (and optionally filter) a predictions DataFrame.

        Adds an ``is_exosomal`` boolean column.  When *drop_non_exosomal*
        is ``True``, rows where the miRNA is not exosome-associated are
        removed.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Prediction results with a miRNA identifier column.
        mirna_col : str
            Name of the column containing miRNA identifiers.
        drop_non_exosomal : bool
            If ``True``, return only rows where ``is_exosomal`` is ``True``.

        Returns
        -------
        pd.DataFrame
            Annotated (and optionally filtered) DataFrame.
        """
        df = predictions_df.copy()

        # Resolve the miRNA column name
        if mirna_col not in df.columns:
            for candidate in ("mirna_id", "mirna", "miRNA"):
                if candidate in df.columns:
                    mirna_col = candidate
                    break
            else:
                raise KeyError(
                    f"miRNA column '{mirna_col}' not found. "
                    f"Available: {list(df.columns)}"
                )

        df["is_exosomal"] = df[mirna_col].apply(
            lambda x: self.is_exosomal(str(x))
        )

        n_exosomal = df["is_exosomal"].sum()
        logger.info(
            "Exosome annotation: %d / %d predictions are exosomal (%.1f%%)",
            n_exosomal,
            len(df),
            100.0 * n_exosomal / max(len(df), 1),
        )

        if drop_non_exosomal:
            df = df[df["is_exosomal"]].reset_index(drop=True)
            logger.info(
                "Filtered to %d exosomal predictions", len(df)
            )

        return df

    def get_exosome_mirna_list(self) -> list[str]:
        """Return a sorted list of all known exosomal miRNAs.

        Returns
        -------
        list[str]
            Alphabetically sorted miRNA identifiers.
        """
        return sorted(self.exosome_mirnas)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_mirna_column(df: pd.DataFrame) -> str:
        """Identify the miRNA column in a DataFrame by header heuristics.

        Searches for columns containing ``mirna``, ``micro``, or ``mature``
        (case-insensitive).

        Returns
        -------
        str
            Column name.

        Raises
        ------
        ValueError
            If no suitable column is found.
        """
        for col in df.columns:
            col_lower = col.lower().strip()
            if "mirna" in col_lower or "micro" in col_lower or "mature" in col_lower:
                return col

        # Last resort: first column
        if len(df.columns) > 0:
            logger.warning(
                "Could not identify miRNA column by name; defaulting to first "
                "column: '%s'",
                df.columns[0],
            )
            return df.columns[0]

        raise ValueError("DataFrame has no columns; cannot identify miRNA column.")

    @staticmethod
    def _normalise_mirna_id(raw_id: str) -> str:
        """Normalise a miRNA identifier string.

        Strips whitespace, converts to lowercase ``hsa-miR``/``hsa-let``
        prefix form where possible, and removes trailing spaces.

        Parameters
        ----------
        raw_id : str
            Raw miRNA identifier from the database file.

        Returns
        -------
        str
            Normalised identifier, or empty string if the input is invalid.
        """
        raw_id = raw_id.strip()
        if not raw_id or raw_id.lower() in ("na", "nan", "none", "-", ""):
            return ""

        # Basic sanity: should contain 'mir' or 'let' somewhere
        if "mir" not in raw_id.lower() and "let" not in raw_id.lower():
            return ""

        return raw_id

    def __repr__(self) -> str:
        return f"ExosomeFilter(n_mirnas={len(self.exosome_mirnas)})"

    def __len__(self) -> int:
        return len(self.exosome_mirnas)
