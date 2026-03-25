"""Aesthetic medicine relevance scorer for miRNA-target predictions.

Assigns a composite score (0--100) to each predicted miRNA-target
interaction based on five weighted criteria:

+-------------------------------+--------+
| Component                     | Weight |
+===============================+========+
| Target gene in key gene list  |  40 %  |
+-------------------------------+--------+
| Pathway overlap               |  25 %  |
+-------------------------------+--------+
| miRNA in key miRNA list       |  15 %  |
+-------------------------------+--------+
| miRNA is exosomal             |  10 %  |
+-------------------------------+--------+
| Prediction confidence         |  10 %  |
+-------------------------------+--------+

Uses the aesthetic-medicine pathway mappings from
:mod:`deepexomir.utils.constants` and the knowledge graph
from :mod:`deepexomir.annotation.knowledge_graph`.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph
from deepexomir.utils.constants import (
    AESTHETIC_PATHWAYS,
    ALL_AESTHETIC_GENES,
    ALL_AESTHETIC_MIRNAS,
    ALL_KEGG_PATHWAYS,
    GENE_TO_CATEGORIES,
    MIRNA_TO_CATEGORIES,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scoring weights (must sum to 1.0)
# ---------------------------------------------------------------------------
_WEIGHT_TARGET_GENE: float = 0.40
_WEIGHT_PATHWAY: float = 0.25
_WEIGHT_KEY_MIRNA: float = 0.15
_WEIGHT_EXOSOMAL: float = 0.10
_WEIGHT_CONFIDENCE: float = 0.10


class AestheticScorer:
    """Score miRNA-target pairs for aesthetic medicine relevance (0--100).

    Parameters
    ----------
    knowledge_graph : MiRNAKnowledgeGraph
        Pre-built knowledge graph with aesthetic mappings.
    exosome_mirnas : set[str]
        Set of miRNA IDs known to be packaged in exosomes (e.g., from ExoCarta).
    """

    def __init__(
        self,
        knowledge_graph: MiRNAKnowledgeGraph,
        exosome_mirnas: set[str],
    ) -> None:
        self.kg = knowledge_graph
        self.exosome_mirnas = exosome_mirnas

    # ------------------------------------------------------------------
    # Single-pair scoring
    # ------------------------------------------------------------------

    def score(
        self,
        mirna_id: str,
        target_gene: str,
        prediction_confidence: float = 0.5,
    ) -> dict[str, Any]:
        """Compute aesthetic relevance score for a single miRNA-target pair.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier (e.g., ``"hsa-miR-21-5p"``).
        target_gene : str
            Target gene symbol (e.g., ``"MITF"``).
        prediction_confidence : float
            Model prediction probability in ``[0, 1]``.

        Returns
        -------
        dict[str, Any]
            ``total_score`` (float, 0--100), ``categories`` (dict mapping
            category name to sub-score 0--100), ``is_exosomal`` (bool),
            ``pathway_hits`` (list of KEGG IDs), ``evidence_summary`` (str).
        """
        # ---- Component 1: Target gene in key list ----------------------
        gene_categories = GENE_TO_CATEGORIES.get(target_gene, [])
        gene_score = 1.0 if gene_categories else 0.0

        # ---- Component 2: Pathway overlap ------------------------------
        pathway_hits = self._find_pathway_overlaps(target_gene)
        if pathway_hits:
            pathway_score = min(len(pathway_hits) / max(len(ALL_KEGG_PATHWAYS), 1), 1.0)
        else:
            pathway_score = 0.0

        # ---- Component 3: miRNA in key miRNA list ----------------------
        mirna_categories = MIRNA_TO_CATEGORIES.get(mirna_id, [])
        mirna_score = 1.0 if mirna_categories else 0.0

        # ---- Component 4: Exosomal status ------------------------------
        is_exosomal = mirna_id in self.exosome_mirnas
        exosomal_score = 1.0 if is_exosomal else 0.0

        # ---- Component 5: Prediction confidence (clamp to [0, 1]) ------
        confidence_score = max(0.0, min(float(prediction_confidence), 1.0))

        # ---- Total composite score (0--100) ----------------------------
        total = (
            _WEIGHT_TARGET_GENE * gene_score
            + _WEIGHT_PATHWAY * pathway_score
            + _WEIGHT_KEY_MIRNA * mirna_score
            + _WEIGHT_EXOSOMAL * exosomal_score
            + _WEIGHT_CONFIDENCE * confidence_score
        ) * 100.0

        # ---- Per-category sub-scores -----------------------------------
        all_categories = set(gene_categories) | set(mirna_categories)
        category_scores = self._compute_category_scores(
            mirna_id,
            target_gene,
            prediction_confidence,
            is_exosomal,
        )

        # ---- Evidence summary ------------------------------------------
        evidence_summary = self._build_evidence_summary(
            mirna_id, target_gene, gene_categories, mirna_categories,
            is_exosomal, pathway_hits,
        )

        return {
            "total_score": round(total, 2),
            "categories": category_scores,
            "is_exosomal": is_exosomal,
            "pathway_hits": pathway_hits,
            "evidence_summary": evidence_summary,
        }

    # ------------------------------------------------------------------
    # Batch scoring
    # ------------------------------------------------------------------

    def batch_score(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Score a DataFrame of predictions and add aesthetic columns.

        The input DataFrame must contain at least the columns ``mirna_id``,
        ``target_gene``, and ``prediction_confidence`` (or ``prob``).
        Additional columns are passed through unchanged.

        Parameters
        ----------
        predictions_df : pd.DataFrame
            Prediction results.

        Returns
        -------
        pd.DataFrame
            Copy of the input with appended columns: ``aesthetic_score``,
            ``aesthetic_categories``, ``is_exosomal``, ``pathway_hits``,
            ``evidence_summary``, and one column per aesthetic category.
        """
        df = predictions_df.copy()

        # Normalise column names
        mirna_col = _resolve_column(df, ["mirna_id", "mirna", "miRNA"])
        gene_col = _resolve_column(df, ["target_gene", "gene", "Target Gene"])
        conf_col = _resolve_column(df, ["prediction_confidence", "prob", "probability", "score"])

        scores: list[float] = []
        categories_list: list[dict[str, float]] = []
        exosomal_flags: list[bool] = []
        pathway_hits_list: list[list[str]] = []
        evidence_list: list[str] = []

        for _, row in df.iterrows():
            mirna_id = str(row[mirna_col])
            target_gene = str(row[gene_col])
            confidence = float(row[conf_col]) if conf_col in df.columns else 0.5

            result = self.score(mirna_id, target_gene, confidence)
            scores.append(result["total_score"])
            categories_list.append(result["categories"])
            exosomal_flags.append(result["is_exosomal"])
            pathway_hits_list.append(result["pathway_hits"])
            evidence_list.append(result["evidence_summary"])

        df["aesthetic_score"] = scores
        df["is_exosomal"] = exosomal_flags
        df["pathway_hits"] = pathway_hits_list
        df["evidence_summary"] = evidence_list

        # Expand per-category scores into separate columns
        cat_df = pd.DataFrame(categories_list, index=df.index)
        for cat_col_name in cat_df.columns:
            df[f"cat_{cat_col_name}"] = cat_df[cat_col_name]

        # Aggregate category names where score > 0
        df["aesthetic_categories"] = [
            [k for k, v in cats.items() if v > 0]
            for cats in categories_list
        ]

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_pathway_overlaps(self, gene: str) -> list[str]:
        """Find KEGG pathways in the aesthetic set that the gene belongs to."""
        hits: list[str] = []
        gene_info = self.kg.query_gene(gene)
        for pathway in gene_info.get("pathways", []):
            if pathway in ALL_KEGG_PATHWAYS:
                hits.append(pathway)
        return hits

    def _compute_category_scores(
        self,
        mirna_id: str,
        target_gene: str,
        confidence: float,
        is_exosomal: bool,
    ) -> dict[str, float]:
        """Compute per-category relevance scores (0--100).

        For each aesthetic category, the score considers whether the gene
        and/or miRNA are among the category's key entities and whether the
        gene belongs to category-specific pathways.
        """
        category_scores: dict[str, float] = {}

        for category, info in AESTHETIC_PATHWAYS.items():
            gene_hit = 1.0 if target_gene in info["key_genes"] else 0.0
            mirna_hit = 1.0 if mirna_id in info["key_mirnas"] else 0.0

            # Check gene membership in this category's pathways
            gene_info = self.kg.query_gene(target_gene)
            gene_pathways = set(gene_info.get("pathways", []))
            cat_pathways = set(info["kegg_pathways"])
            pathway_overlap = len(gene_pathways & cat_pathways)
            pathway_hit = min(pathway_overlap / max(len(cat_pathways), 1), 1.0)

            exo_hit = 1.0 if is_exosomal else 0.0
            conf = max(0.0, min(confidence, 1.0))

            cat_total = (
                _WEIGHT_TARGET_GENE * gene_hit
                + _WEIGHT_PATHWAY * pathway_hit
                + _WEIGHT_KEY_MIRNA * mirna_hit
                + _WEIGHT_EXOSOMAL * exo_hit
                + _WEIGHT_CONFIDENCE * conf
            ) * 100.0

            category_scores[category] = round(cat_total, 2)

        return category_scores

    @staticmethod
    def _build_evidence_summary(
        mirna_id: str,
        target_gene: str,
        gene_categories: list[str],
        mirna_categories: list[str],
        is_exosomal: bool,
        pathway_hits: list[str],
    ) -> str:
        """Build a human-readable evidence summary string."""
        parts: list[str] = []

        if gene_categories:
            parts.append(
                f"{target_gene} is a key gene for: {', '.join(gene_categories)}"
            )
        if mirna_categories:
            parts.append(
                f"{mirna_id} is a key miRNA for: {', '.join(mirna_categories)}"
            )
        if is_exosomal:
            parts.append(f"{mirna_id} is exosome-associated (ExoCarta)")
        if pathway_hits:
            parts.append(f"Pathway overlaps: {', '.join(pathway_hits)}")

        if not parts:
            return "No direct aesthetic medicine evidence found."

        return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _resolve_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str:
    """Return the first column name from *candidates* that exists in *df*.

    Raises
    ------
    KeyError
        If none of the candidate names are found.
    """
    for name in candidates:
        if name in df.columns:
            return name
    raise KeyError(
        f"None of the expected columns {candidates} found in DataFrame. "
        f"Available columns: {list(df.columns)}"
    )
