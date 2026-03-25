"""Tests for annotation modules.

Covers aesthetic constants, AestheticScorer, MiRNAKnowledgeGraph,
and ExosomeFilter.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from deepexomir.annotation.aesthetic_scorer import AestheticScorer
from deepexomir.annotation.exosome_filter import ExosomeFilter
from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph
from deepexomir.utils.constants import (
    AESTHETIC_PATHWAYS,
    ALL_AESTHETIC_GENES,
    ALL_AESTHETIC_MIRNAS,
    GENE_TO_CATEGORIES,
    MIRNA_TO_CATEGORIES,
)


# ====================================================================
# test_aesthetic_constants
# ====================================================================


class TestAestheticConstants:
    """Verify all categories have required keys."""

    def test_all_categories_have_required_keys(self):
        """Each AESTHETIC_PATHWAYS category should have all required keys."""
        required_keys = {
            "display_name",
            "display_name_zh",
            "kegg_pathways",
            "key_genes",
            "key_mirnas",
            "mechanism",
        }

        for category, info in AESTHETIC_PATHWAYS.items():
            for key in required_keys:
                assert key in info, (
                    f"Category '{category}' is missing key '{key}'"
                )

    def test_expected_categories_exist(self):
        """All five aesthetic categories should be defined."""
        expected = {"whitening", "scar_removal", "hair_restoration", "anti_aging", "skin_refinement"}
        assert expected == set(AESTHETIC_PATHWAYS.keys())

    def test_key_genes_are_non_empty(self):
        """Each category should have at least one key gene."""
        for category, info in AESTHETIC_PATHWAYS.items():
            assert len(info["key_genes"]) > 0, (
                f"Category '{category}' has no key genes"
            )

    def test_key_mirnas_are_non_empty(self):
        """Each category should have at least one key miRNA."""
        for category, info in AESTHETIC_PATHWAYS.items():
            assert len(info["key_mirnas"]) > 0, (
                f"Category '{category}' has no key miRNAs"
            )

    def test_kegg_pathways_are_non_empty(self):
        """Each category should have at least one KEGG pathway."""
        for category, info in AESTHETIC_PATHWAYS.items():
            assert len(info["kegg_pathways"]) > 0, (
                f"Category '{category}' has no KEGG pathways"
            )

    def test_aggregate_sets_populated(self):
        """ALL_AESTHETIC_GENES and ALL_AESTHETIC_MIRNAS should be non-empty."""
        assert len(ALL_AESTHETIC_GENES) > 0
        assert len(ALL_AESTHETIC_MIRNAS) > 0

    def test_gene_to_categories_mapping(self):
        """GENE_TO_CATEGORIES should map each gene to at least one category."""
        for gene, categories in GENE_TO_CATEGORIES.items():
            assert isinstance(categories, list)
            assert len(categories) > 0
            for cat in categories:
                assert cat in AESTHETIC_PATHWAYS

    def test_mirna_to_categories_mapping(self):
        """MIRNA_TO_CATEGORIES should map each miRNA to at least one category."""
        for mirna, categories in MIRNA_TO_CATEGORIES.items():
            assert isinstance(categories, list)
            assert len(categories) > 0
            for cat in categories:
                assert cat in AESTHETIC_PATHWAYS

    def test_known_gene_in_whitening(self):
        """MITF should be in the whitening key genes."""
        assert "MITF" in AESTHETIC_PATHWAYS["whitening"]["key_genes"]

    def test_known_mirna_in_whitening(self):
        """hsa-miR-330-5p should be in the whitening key miRNAs."""
        assert "hsa-miR-330-5p" in AESTHETIC_PATHWAYS["whitening"]["key_mirnas"]


# ====================================================================
# test_aesthetic_scorer_known_pair
# ====================================================================


class TestAestheticScorerKnownPair:
    """miR-330-5p + MITF should yield a high whitening score."""

    @pytest.fixture
    def scorer(self):
        """Build a scorer with a minimal knowledge graph."""
        kg = MiRNAKnowledgeGraph()
        kg.add_aesthetic_mapping()
        # Add a targeting edge
        kg.add_mirna_target("hsa-miR-330-5p", "MITF", score=1.0, evidence="Reporter assay")
        # Add pathway edge
        kg.add_gene_pathway("MITF", "hsa04916", "Melanogenesis")
        # Mark as exosomal
        kg.add_exosome_info("hsa-miR-330-5p")

        exosome_mirnas = {"hsa-miR-330-5p"}
        return AestheticScorer(knowledge_graph=kg, exosome_mirnas=exosome_mirnas)

    def test_high_whitening_score(self, scorer):
        """Known whitening pair should have a high total score."""
        result = scorer.score(
            mirna_id="hsa-miR-330-5p",
            target_gene="MITF",
            prediction_confidence=0.95,
        )

        assert result["total_score"] > 50.0  # Should be high
        assert "whitening" in result["categories"]
        assert result["categories"]["whitening"] > 0

    def test_is_exosomal(self, scorer):
        """Known exosomal miRNA should be flagged."""
        result = scorer.score(
            mirna_id="hsa-miR-330-5p",
            target_gene="MITF",
            prediction_confidence=0.9,
        )
        assert result["is_exosomal"] is True

    def test_evidence_summary_not_empty(self, scorer):
        """Evidence summary should contain meaningful text."""
        result = scorer.score(
            mirna_id="hsa-miR-330-5p",
            target_gene="MITF",
            prediction_confidence=0.9,
        )
        assert len(result["evidence_summary"]) > 0
        assert result["evidence_summary"] != "No direct aesthetic medicine evidence found."


# ====================================================================
# test_aesthetic_scorer_irrelevant_pair
# ====================================================================


class TestAestheticScorerIrrelevantPair:
    """Random miRNA + random gene should get a low score."""

    @pytest.fixture
    def scorer(self):
        """Build a scorer with a minimal knowledge graph."""
        kg = MiRNAKnowledgeGraph()
        kg.add_aesthetic_mapping()
        return AestheticScorer(knowledge_graph=kg, exosome_mirnas=set())

    def test_low_total_score(self, scorer):
        """Irrelevant pair should have a low total score."""
        result = scorer.score(
            mirna_id="hsa-miR-9999-5p",
            target_gene="GENE_XYZ",
            prediction_confidence=0.5,
        )

        # Only the confidence component should contribute
        # 0.10 weight * 0.5 confidence * 100 = 5.0
        assert result["total_score"] < 10.0

    def test_not_exosomal(self, scorer):
        """Unknown miRNA should not be flagged as exosomal."""
        result = scorer.score(
            mirna_id="hsa-miR-9999-5p",
            target_gene="GENE_XYZ",
            prediction_confidence=0.5,
        )
        assert result["is_exosomal"] is False

    def test_no_pathway_hits(self, scorer):
        """Unknown gene should have no pathway hits."""
        result = scorer.score(
            mirna_id="hsa-miR-9999-5p",
            target_gene="GENE_XYZ",
            prediction_confidence=0.5,
        )
        assert len(result["pathway_hits"]) == 0

    def test_all_category_scores_are_low(self, scorer):
        """All per-category scores should be low for an irrelevant pair."""
        result = scorer.score(
            mirna_id="hsa-miR-9999-5p",
            target_gene="GENE_XYZ",
            prediction_confidence=0.5,
        )
        for cat, score in result["categories"].items():
            # Only confidence should contribute: 0.10 * 0.5 * 100 = 5.0
            assert score <= 10.0, f"Category {cat} has unexpectedly high score: {score}"


# ====================================================================
# test_knowledge_graph_construction
# ====================================================================


class TestKnowledgeGraphConstruction:
    """Build a small graph and verify node/edge counts."""

    def test_basic_construction(self):
        """Building a graph with add methods should create nodes and edges."""
        kg = MiRNAKnowledgeGraph()
        kg.add_mirna_target("hsa-miR-21-5p", "PTEN", score=0.9, evidence="CLIP-seq")
        kg.add_mirna_target("hsa-miR-21-5p", "PDCD4", score=0.85, evidence="Reporter assay")
        kg.add_gene_pathway("PTEN", "hsa04151", "PI3K-Akt signaling")

        assert kg.num_nodes >= 4  # miR-21, PTEN, PDCD4, hsa04151
        assert kg.num_edges >= 3  # 2 targeting + 1 pathway

    def test_aesthetic_mapping_adds_nodes(self):
        """add_aesthetic_mapping should create aesthetic, pathway, gene, and miRNA nodes."""
        kg = MiRNAKnowledgeGraph()
        kg.add_aesthetic_mapping()

        assert kg.num_nodes > 0
        assert kg.num_edges > 0

        # Should have aesthetic nodes
        aesthetic_nodes = [
            n for n, d in kg.graph.nodes(data=True) if d.get("node_type") == "aesthetic"
        ]
        assert len(aesthetic_nodes) == len(AESTHETIC_PATHWAYS)

    def test_exosome_info(self):
        """add_exosome_info should mark miRNAs as exosomal."""
        kg = MiRNAKnowledgeGraph()
        kg.add_exosome_info("hsa-miR-21-5p", source="ExoCarta")

        node_data = kg.graph.nodes["hsa-miR-21-5p"]
        assert node_data.get("is_exosomal") is True

    def test_disease_association(self):
        """add_disease_association should create disease nodes and edges."""
        kg = MiRNAKnowledgeGraph()
        kg.add_disease_association("hsa-miR-21-5p", "Breast Neoplasms", pmid="12345678")

        assert "disease:Breast Neoplasms" in kg.graph
        assert kg.graph.has_edge("hsa-miR-21-5p", "disease:Breast Neoplasms")

    def test_duplicate_edges_merge(self):
        """Adding the same edge twice should merge evidence, not duplicate."""
        kg = MiRNAKnowledgeGraph()
        kg.add_mirna_target("hsa-miR-21-5p", "PTEN", score=0.8, evidence="CLIP-seq")
        kg.add_mirna_target("hsa-miR-21-5p", "PTEN", score=0.9, evidence="Reporter assay")

        edge_data = kg.graph["hsa-miR-21-5p"]["PTEN"]
        assert edge_data["score"] == 0.9  # max of 0.8 and 0.9
        assert "CLIP-seq" in edge_data["evidence_types"]
        assert "Reporter assay" in edge_data["evidence_types"]

    def test_build_from_databases_with_none(self):
        """build_from_databases with all None should still add aesthetic mappings."""
        kg = MiRNAKnowledgeGraph()
        kg.build_from_databases()  # all None

        # Should have at least aesthetic nodes
        assert kg.num_nodes > 0
        assert kg.num_edges > 0

    def test_save_and_load(self):
        """Saving and loading should preserve the graph structure."""
        kg = MiRNAKnowledgeGraph()
        kg.add_mirna_target("hsa-miR-21-5p", "PTEN", score=0.9, evidence="CLIP-seq")
        kg.add_aesthetic_mapping()

        original_nodes = kg.num_nodes
        original_edges = kg.num_edges

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_kg.pkl"
            kg.save(path)
            assert path.exists()

            kg2 = MiRNAKnowledgeGraph()
            kg2.load(path)

            assert kg2.num_nodes == original_nodes
            assert kg2.num_edges == original_edges


# ====================================================================
# test_knowledge_graph_query
# ====================================================================


class TestKnowledgeGraphQuery:
    """Query miRNA, verify targets returned."""

    @pytest.fixture
    def populated_kg(self):
        """Build a knowledge graph with several relationships."""
        kg = MiRNAKnowledgeGraph()
        kg.add_aesthetic_mapping()
        kg.add_mirna_target("hsa-miR-21-5p", "PTEN", score=0.9, evidence="CLIP-seq")
        kg.add_mirna_target("hsa-miR-21-5p", "PDCD4", score=0.85, evidence="Reporter assay")
        kg.add_mirna_target("hsa-miR-29a-3p", "COL1A1", score=0.95, evidence="Reporter assay")
        kg.add_gene_pathway("PTEN", "hsa04151", "PI3K-Akt signaling")
        kg.add_exosome_info("hsa-miR-21-5p")
        kg.add_disease_association("hsa-miR-21-5p", "Breast Neoplasms", pmid="12345678")
        return kg

    def test_query_mirna_returns_targets(self, populated_kg):
        """Querying a miRNA should return its targets."""
        result = populated_kg.query_mirna("hsa-miR-21-5p")

        assert result["mirna_id"] == "hsa-miR-21-5p"
        assert len(result["targets"]) >= 2

        target_genes = [t["gene"] for t in result["targets"]]
        assert "PTEN" in target_genes
        assert "PDCD4" in target_genes

    def test_query_mirna_returns_exosomal_status(self, populated_kg):
        """Exosomal miRNAs should be flagged."""
        result = populated_kg.query_mirna("hsa-miR-21-5p")
        assert result["is_exosomal"] is True

    def test_query_mirna_returns_diseases(self, populated_kg):
        """Disease associations should be returned."""
        result = populated_kg.query_mirna("hsa-miR-21-5p")
        assert "Breast Neoplasms" in result["diseases"]

    def test_query_mirna_returns_pathways(self, populated_kg):
        """Pathways through targets should be returned."""
        result = populated_kg.query_mirna("hsa-miR-21-5p")
        assert "hsa04151" in result["pathways"]

    def test_query_unknown_mirna(self, populated_kg):
        """Unknown miRNA should return empty results."""
        result = populated_kg.query_mirna("hsa-miR-999999")
        assert len(result["targets"]) == 0
        assert result["is_exosomal"] is False

    def test_query_gene(self, populated_kg):
        """Querying a gene should return targeting miRNAs."""
        result = populated_kg.query_gene("PTEN")

        assert result["gene"] == "PTEN"
        assert len(result["targeting_mirnas"]) >= 1

        mirna_ids = [m["mirna_id"] for m in result["targeting_mirnas"]]
        assert "hsa-miR-21-5p" in mirna_ids

    def test_query_gene_pathways(self, populated_kg):
        """Gene query should return its pathways."""
        result = populated_kg.query_gene("PTEN")
        assert "hsa04151" in result["pathways"]

    def test_get_aesthetic_mirnas(self, populated_kg):
        """get_aesthetic_mirnas should return miRNAs for a category."""
        # hsa-miR-29a-3p targets COL1A1, which is a key gene for scar_removal
        result = populated_kg.get_aesthetic_mirnas("scar_removal")
        assert isinstance(result, list)
        # Should include key miRNAs for scar_removal from constants
        mirna_ids = [m["mirna_id"] for m in result]
        assert "hsa-miR-29a-3p" in mirna_ids


# ====================================================================
# test_exosome_filter
# ====================================================================


class TestExosomeFilter:
    """Verify ExosomeFilter logic."""

    def test_empty_filter(self):
        """New filter with no data should return False for everything."""
        ef = ExosomeFilter()
        assert ef.is_exosomal("hsa-miR-21-5p") is False
        assert len(ef) == 0

    def test_manual_population(self):
        """Manually adding miRNAs should work."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-21-5p")
        ef.exosome_mirnas.add("hsa-miR-155-5p")

        assert ef.is_exosomal("hsa-miR-21-5p") is True
        assert ef.is_exosomal("hsa-miR-155-5p") is True
        assert ef.is_exosomal("hsa-miR-999-5p") is False
        assert len(ef) == 2

    def test_arm_fallback(self):
        """If exact match fails, should check for arm variant matches."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-21-5p")

        # Query without arm designation: should find 5p variant
        assert ef.is_exosomal("hsa-miR-21") is True

        # Query with different arm: should find 5p variant via base match
        assert ef.is_exosomal("hsa-miR-21-3p") is True

    def test_filter_predictions_annotate(self):
        """filter_predictions should add is_exosomal column."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-21-5p")

        df = pd.DataFrame({
            "mirna_id": ["hsa-miR-21-5p", "hsa-miR-155-5p", "hsa-miR-21-5p"],
            "target_gene": ["PTEN", "TP53", "PDCD4"],
            "score": [0.9, 0.8, 0.7],
        })

        result = ef.filter_predictions(df)
        assert "is_exosomal" in result.columns
        assert result["is_exosomal"].sum() == 2  # two miR-21 entries
        assert len(result) == 3  # all rows preserved

    def test_filter_predictions_drop(self):
        """filter_predictions with drop_non_exosomal should remove rows."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-21-5p")

        df = pd.DataFrame({
            "mirna_id": ["hsa-miR-21-5p", "hsa-miR-155-5p", "hsa-miR-21-5p"],
            "target_gene": ["PTEN", "TP53", "PDCD4"],
        })

        result = ef.filter_predictions(df, drop_non_exosomal=True)
        assert len(result) == 2  # only miR-21 entries kept
        assert all(result["is_exosomal"])

    def test_get_exosome_mirna_list(self):
        """get_exosome_mirna_list should return a sorted list."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-155-5p")
        ef.exosome_mirnas.add("hsa-miR-21-5p")

        mirna_list = ef.get_exosome_mirna_list()
        assert mirna_list == ["hsa-miR-155-5p", "hsa-miR-21-5p"]

    def test_repr(self):
        """repr should show the count of miRNAs."""
        ef = ExosomeFilter()
        ef.exosome_mirnas.add("hsa-miR-21-5p")
        assert "1" in repr(ef)

    def test_load_exocarta_file_not_found(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        ef = ExosomeFilter()
        with pytest.raises(FileNotFoundError):
            ef.load_exocarta(Path("/nonexistent/path/exocarta.txt"))

    def test_load_exocarta_from_tsv(self):
        """Should parse a tab-delimited ExoCarta-like file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("miRNA\tOrganism\tStudy\n")
            f.write("hsa-miR-21-5p\tHomo sapiens\tStudy1\n")
            f.write("hsa-miR-155-5p\tHomo sapiens\tStudy2\n")
            f.write("hsa-miR-21-5p\tHomo sapiens\tStudy3\n")  # duplicate
            f.write("hsa-let-7a-5p\tHomo sapiens\tStudy4\n")
            tmppath = f.name

        ef = ExosomeFilter()
        ef.load_exocarta(Path(tmppath))

        assert len(ef) == 3  # 3 unique miRNAs
        assert ef.is_exosomal("hsa-miR-21-5p")
        assert ef.is_exosomal("hsa-miR-155-5p")
        assert ef.is_exosomal("hsa-let-7a-5p")
