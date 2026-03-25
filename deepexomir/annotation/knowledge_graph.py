"""Directed knowledge graph linking miRNAs, genes, pathways, and aesthetic functions.

Stores miRNA -> Gene -> Pathway -> AestheticFunction relationships in a
NetworkX ``DiGraph``.  Nodes carry typed attributes (``node_type`` in
``{"mirna", "gene", "pathway", "aesthetic", "exosome_source", "disease"}``),
and edges carry evidence metadata (scores, PMIDs, evidence types).

The graph can be built incrementally from individual records, or bulk-loaded
from miRTarBase, ExoCarta, and HMDD DataFrames via
:meth:`build_from_databases`.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Optional, Sequence

import networkx as nx
import pandas as pd

from deepexomir.utils.constants import (
    AESTHETIC_PATHWAYS,
    ALL_AESTHETIC_GENES,
    ALL_AESTHETIC_MIRNAS,
    ALL_KEGG_PATHWAYS,
    GENE_TO_CATEGORIES,
    MIRNA_TO_CATEGORIES,
)

logger = logging.getLogger(__name__)


class MiRNAKnowledgeGraph:
    """Directed graph encoding miRNA-target-pathway-aesthetic relationships.

    The graph supports four primary relationship types:

    1. **miRNA -> Gene**: validated or predicted targeting relationship.
    2. **Gene -> Pathway**: gene membership in KEGG pathways.
    3. **Pathway -> Aesthetic**: pathway relevance to aesthetic medicine categories.
    4. **miRNA -> Disease**: miRNA-disease associations from HMDD.

    Additionally, miRNA nodes may be annotated with exosome cargo information
    from ExoCarta.
    """

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Node / edge creation helpers
    # ------------------------------------------------------------------

    def _ensure_node(
        self,
        node_id: str,
        node_type: str,
        **attrs: Any,
    ) -> None:
        """Add a node if it does not exist, or merge attributes."""
        if node_id not in self.graph:
            self.graph.add_node(node_id, node_type=node_type, **attrs)
        else:
            # Merge new attributes into existing node
            for k, v in attrs.items():
                existing = self.graph.nodes[node_id].get(k)
                if existing is None:
                    self.graph.nodes[node_id][k] = v
                elif isinstance(existing, set) and isinstance(v, set):
                    existing.update(v)
                elif isinstance(existing, list) and isinstance(v, list):
                    for item in v:
                        if item not in existing:
                            existing.append(item)

    # ------------------------------------------------------------------
    # Public: incremental graph construction
    # ------------------------------------------------------------------

    def add_mirna_target(
        self,
        mirna_id: str,
        gene: str,
        score: float = 1.0,
        evidence: str = "unknown",
    ) -> None:
        """Add a miRNA -> gene targeting edge.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier (e.g., ``"hsa-miR-21-5p"``).
        gene : str
            Gene symbol (e.g., ``"PTEN"``).
        score : float
            Confidence / interaction score.
        evidence : str
            Evidence type (e.g., ``"CLIP-Seq"``, ``"Reporter assay"``).
        """
        self._ensure_node(mirna_id, "mirna")
        self._ensure_node(gene, "gene")

        if self.graph.has_edge(mirna_id, gene):
            edge_data = self.graph[mirna_id][gene]
            edge_data.setdefault("evidence_types", set()).add(evidence)
            edge_data["score"] = max(edge_data.get("score", 0.0), score)
        else:
            self.graph.add_edge(
                mirna_id,
                gene,
                relation="targets",
                score=score,
                evidence_types={evidence},
            )

    def add_gene_pathway(
        self,
        gene: str,
        pathway_id: str,
        pathway_name: str = "",
    ) -> None:
        """Add a gene -> pathway membership edge.

        Parameters
        ----------
        gene : str
            Gene symbol.
        pathway_id : str
            KEGG pathway identifier (e.g., ``"hsa04916"``).
        pathway_name : str
            Human-readable pathway name.
        """
        self._ensure_node(gene, "gene")
        self._ensure_node(
            pathway_id, "pathway", display_name=pathway_name
        )
        if not self.graph.has_edge(gene, pathway_id):
            self.graph.add_edge(gene, pathway_id, relation="belongs_to")

    def add_aesthetic_mapping(self) -> None:
        """Populate pathway -> aesthetic category edges from constants.

        Reads :data:`~deepexomir.utils.constants.AESTHETIC_PATHWAYS` and
        creates ``aesthetic`` nodes for each category, linking them to their
        KEGG pathways and annotating key genes / miRNAs.
        """
        for category, info in AESTHETIC_PATHWAYS.items():
            aesthetic_id = f"aesthetic:{category}"
            self._ensure_node(
                aesthetic_id,
                "aesthetic",
                display_name=info["display_name"],
                display_name_zh=info.get("display_name_zh", ""),
                mechanism=info.get("mechanism", ""),
            )

            # Pathway -> Aesthetic
            for pw_id in info["kegg_pathways"]:
                self._ensure_node(pw_id, "pathway")
                if not self.graph.has_edge(pw_id, aesthetic_id):
                    self.graph.add_edge(
                        pw_id, aesthetic_id, relation="relevant_to"
                    )

            # Annotate key genes and miRNAs (add nodes if missing)
            for gene in info["key_genes"]:
                self._ensure_node(gene, "gene")
                # Direct gene -> aesthetic shortcut edge
                if not self.graph.has_edge(gene, aesthetic_id):
                    self.graph.add_edge(
                        gene, aesthetic_id, relation="key_gene_for"
                    )

            for mirna in info["key_mirnas"]:
                self._ensure_node(mirna, "mirna")
                if not self.graph.has_edge(mirna, aesthetic_id):
                    self.graph.add_edge(
                        mirna, aesthetic_id, relation="key_mirna_for"
                    )

    def add_exosome_info(
        self,
        mirna_id: str,
        source: str = "ExoCarta",
    ) -> None:
        """Mark a miRNA as exosome-associated.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier.
        source : str
            Source database (e.g., ``"ExoCarta"``).
        """
        self._ensure_node(mirna_id, "mirna", is_exosomal=True)
        self.graph.nodes[mirna_id]["is_exosomal"] = True

        exo_node = f"exosome:{source}"
        self._ensure_node(exo_node, "exosome_source", database=source)
        if not self.graph.has_edge(mirna_id, exo_node):
            self.graph.add_edge(mirna_id, exo_node, relation="found_in_exosome")

    def add_disease_association(
        self,
        mirna_id: str,
        disease: str,
        pmid: str = "",
    ) -> None:
        """Add a miRNA -> disease association edge.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier.
        disease : str
            Disease name.
        pmid : str
            PubMed ID supporting the association.
        """
        disease_node = f"disease:{disease}"
        self._ensure_node(mirna_id, "mirna")
        self._ensure_node(disease_node, "disease", name=disease)

        if self.graph.has_edge(mirna_id, disease_node):
            edge_data = self.graph[mirna_id][disease_node]
            if pmid:
                edge_data.setdefault("pmids", set()).add(pmid)
        else:
            pmids = {pmid} if pmid else set()
            self.graph.add_edge(
                mirna_id,
                disease_node,
                relation="associated_with",
                pmids=pmids,
            )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query_mirna(self, mirna_id: str) -> dict[str, Any]:
        """Retrieve all targets, pathways, and aesthetic functions for a miRNA.

        Parameters
        ----------
        mirna_id : str
            miRNA identifier.

        Returns
        -------
        dict[str, Any]
            Keys: ``targets`` (list of dicts), ``pathways`` (list), ``aesthetic_categories``
            (list), ``diseases`` (list), ``is_exosomal`` (bool).
        """
        result: dict[str, Any] = {
            "mirna_id": mirna_id,
            "targets": [],
            "pathways": set(),
            "aesthetic_categories": [],
            "diseases": [],
            "is_exosomal": False,
        }

        if mirna_id not in self.graph:
            return result

        node_data = self.graph.nodes[mirna_id]
        result["is_exosomal"] = node_data.get("is_exosomal", False)

        for _, target, edge_data in self.graph.out_edges(mirna_id, data=True):
            target_type = self.graph.nodes[target].get("node_type", "")

            if edge_data.get("relation") == "targets":
                target_info = {
                    "gene": target,
                    "score": edge_data.get("score", 0.0),
                    "evidence_types": list(edge_data.get("evidence_types", [])),
                }
                result["targets"].append(target_info)

                # Follow gene -> pathway -> aesthetic
                for _, pathway_node, pw_edge in self.graph.out_edges(target, data=True):
                    if pw_edge.get("relation") == "belongs_to":
                        result["pathways"].add(pathway_node)

            elif edge_data.get("relation") == "key_mirna_for":
                cat_name = target.replace("aesthetic:", "")
                if cat_name not in result["aesthetic_categories"]:
                    result["aesthetic_categories"].append(cat_name)

            elif edge_data.get("relation") == "associated_with":
                disease_name = target.replace("disease:", "")
                result["diseases"].append(disease_name)

        result["pathways"] = sorted(result["pathways"])
        return result

    def query_gene(self, gene: str) -> dict[str, Any]:
        """Retrieve all miRNAs targeting a gene, its pathways, and functions.

        Parameters
        ----------
        gene : str
            Gene symbol.

        Returns
        -------
        dict[str, Any]
            Keys: ``targeting_mirnas`` (list), ``pathways`` (list),
            ``aesthetic_categories`` (list).
        """
        result: dict[str, Any] = {
            "gene": gene,
            "targeting_mirnas": [],
            "pathways": [],
            "aesthetic_categories": [],
        }

        if gene not in self.graph:
            return result

        # Incoming edges: miRNA -> gene
        for source, _, edge_data in self.graph.in_edges(gene, data=True):
            if edge_data.get("relation") == "targets":
                result["targeting_mirnas"].append({
                    "mirna_id": source,
                    "score": edge_data.get("score", 0.0),
                    "evidence_types": list(edge_data.get("evidence_types", [])),
                })

        # Outgoing edges: gene -> pathway, gene -> aesthetic
        for _, target, edge_data in self.graph.out_edges(gene, data=True):
            relation = edge_data.get("relation", "")
            if relation == "belongs_to":
                result["pathways"].append(target)
            elif relation == "key_gene_for":
                cat_name = target.replace("aesthetic:", "")
                if cat_name not in result["aesthetic_categories"]:
                    result["aesthetic_categories"].append(cat_name)

        return result

    def get_aesthetic_mirnas(self, category: str) -> list[dict[str, Any]]:
        """Get miRNAs relevant to a specific aesthetic category, ranked by connectivity.

        Parameters
        ----------
        category : str
            Aesthetic category key (e.g., ``"whitening"``, ``"anti_aging"``).

        Returns
        -------
        list[dict[str, Any]]
            Sorted list of dicts with ``mirna_id``, ``score``, ``target_count``,
            ``is_exosomal``.
        """
        aesthetic_node = f"aesthetic:{category}"
        if aesthetic_node not in self.graph:
            return []

        mirna_scores: dict[str, dict[str, Any]] = {}

        # Direct key_mirna_for edges
        for source, _, edge_data in self.graph.in_edges(aesthetic_node, data=True):
            if edge_data.get("relation") == "key_mirna_for":
                node_data = self.graph.nodes[source]
                if node_data.get("node_type") == "mirna":
                    mirna_scores[source] = {
                        "mirna_id": source,
                        "score": 1.0,
                        "target_count": 0,
                        "is_exosomal": node_data.get("is_exosomal", False),
                    }

        # Also find miRNAs that target key genes for this category
        for source, _, edge_data in self.graph.in_edges(aesthetic_node, data=True):
            if edge_data.get("relation") == "key_gene_for":
                gene = source
                gene_query = self.query_gene(gene)
                for mirna_info in gene_query["targeting_mirnas"]:
                    mid = mirna_info["mirna_id"]
                    if mid not in mirna_scores:
                        mirna_scores[mid] = {
                            "mirna_id": mid,
                            "score": 0.0,
                            "target_count": 0,
                            "is_exosomal": self.graph.nodes.get(mid, {}).get(
                                "is_exosomal", False
                            ),
                        }
                    mirna_scores[mid]["target_count"] += 1
                    mirna_scores[mid]["score"] += mirna_info.get("score", 0.0)

        ranked = sorted(
            mirna_scores.values(),
            key=lambda x: (x["score"], x["target_count"]),
            reverse=True,
        )
        return ranked

    # ------------------------------------------------------------------
    # Bulk construction from database DataFrames
    # ------------------------------------------------------------------

    def build_from_databases(
        self,
        mirtarbase_df: Optional[pd.DataFrame] = None,
        exocarta_df: Optional[pd.DataFrame] = None,
        hmdd_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Build the full knowledge graph from database DataFrames.

        Parameters
        ----------
        mirtarbase_df : pd.DataFrame, optional
            miRTarBase data with at least columns: ``miRNA``, ``Target Gene``,
            ``Experiments`` (or ``Support Type``).
        exocarta_df : pd.DataFrame, optional
            ExoCarta miRNA details with at least column: ``miRNA`` (or ``Mature miRNA``).
        hmdd_df : pd.DataFrame, optional
            HMDD data with at least columns: ``mir``, ``disease``, ``pmid``.
        """
        logger.info("Building knowledge graph from databases...")

        # 1. Aesthetic mappings (always loaded from constants)
        self.add_aesthetic_mapping()
        logger.info(
            "Added aesthetic mappings: %d categories, %d key genes, %d key miRNAs",
            len(AESTHETIC_PATHWAYS),
            len(ALL_AESTHETIC_GENES),
            len(ALL_AESTHETIC_MIRNAS),
        )

        # 2. miRTarBase: miRNA -> gene targets
        if mirtarbase_df is not None:
            self._load_mirtarbase(mirtarbase_df)

        # 3. ExoCarta: exosome annotations
        if exocarta_df is not None:
            self._load_exocarta(exocarta_df)

        # 4. HMDD: disease associations
        if hmdd_df is not None:
            self._load_hmdd(hmdd_df)

        logger.info(
            "Knowledge graph built: %d nodes, %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def _load_mirtarbase(self, df: pd.DataFrame) -> None:
        """Parse miRTarBase DataFrame and add miRNA-target edges."""
        # Normalise column names
        col_map = {}
        for col in df.columns:
            col_lower = col.lower().strip()
            if "mirna" in col_lower and "target" not in col_lower:
                col_map["mirna"] = col
            elif "target" in col_lower and "gene" in col_lower:
                col_map["gene"] = col
            elif "experiment" in col_lower or "support" in col_lower:
                col_map["evidence"] = col

        mirna_col = col_map.get("mirna", "miRNA")
        gene_col = col_map.get("gene", "Target Gene")
        evidence_col = col_map.get("evidence", "Experiments")

        count = 0
        for _, row in df.iterrows():
            mirna_id = str(row.get(mirna_col, "")).strip()
            gene = str(row.get(gene_col, "")).strip()
            evidence = str(row.get(evidence_col, "unknown")).strip()

            if mirna_id and gene:
                self.add_mirna_target(mirna_id, gene, score=1.0, evidence=evidence)
                count += 1

        logger.info("Loaded %d miRNA-target interactions from miRTarBase", count)

    def _load_exocarta(self, df: pd.DataFrame) -> None:
        """Parse ExoCarta DataFrame and mark exosomal miRNAs."""
        # Find the miRNA column
        mirna_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if "mirna" in col_lower or "mature" in col_lower:
                mirna_col = col
                break
        if mirna_col is None:
            mirna_col = df.columns[0]

        count = 0
        for _, row in df.iterrows():
            mirna_id = str(row.get(mirna_col, "")).strip()
            if mirna_id:
                self.add_exosome_info(mirna_id, source="ExoCarta")
                count += 1

        logger.info("Loaded %d exosomal miRNAs from ExoCarta", count)

    def _load_hmdd(self, df: pd.DataFrame) -> None:
        """Parse HMDD DataFrame and add disease association edges."""
        # Identify columns
        mir_col = None
        disease_col = None
        pmid_col = None
        for col in df.columns:
            col_lower = col.lower().strip()
            if "mir" in col_lower and "disease" not in col_lower:
                mir_col = col
            elif "disease" in col_lower:
                disease_col = col
            elif "pmid" in col_lower:
                pmid_col = col

        mir_col = mir_col or "mir"
        disease_col = disease_col or "disease"
        pmid_col = pmid_col or "pmid"

        count = 0
        for _, row in df.iterrows():
            mirna_id = str(row.get(mir_col, "")).strip()
            disease = str(row.get(disease_col, "")).strip()
            pmid = str(row.get(pmid_col, "")).strip()

            if mirna_id and disease:
                self.add_disease_association(mirna_id, disease, pmid)
                count += 1

        logger.info("Loaded %d miRNA-disease associations from HMDD", count)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the knowledge graph to disk.

        Parameters
        ----------
        path : str or Path
            Output file path.  Uses ``.gpickle`` extension by convention.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert sets to lists for JSON-compatible serialisation
        graph_copy = self.graph.copy()
        for _, _, data in graph_copy.edges(data=True):
            for k, v in data.items():
                if isinstance(v, set):
                    data[k] = list(v)
        for _, data in graph_copy.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, set):
                    data[k] = list(v)

        with open(path, "wb") as fh:
            pickle.dump(graph_copy, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("Knowledge graph saved to %s (%d nodes, %d edges)",
                     path, self.graph.number_of_nodes(), self.graph.number_of_edges())

    def load(self, path: str | Path) -> None:
        """Load a previously saved knowledge graph.

        Parameters
        ----------
        path : str or Path
            Path to a ``.gpickle`` file.
        """
        path = Path(path)
        with open(path, "rb") as fh:
            self.graph = pickle.load(fh)

        # Restore sets where needed
        for _, _, data in self.graph.edges(data=True):
            for k in ("evidence_types", "pmids"):
                if k in data and isinstance(data[k], list):
                    data[k] = set(data[k])
        for _, data in self.graph.nodes(data=True):
            for k, v in data.items():
                if isinstance(v, list) and k in ("evidence_types", "pmids"):
                    data[k] = set(v)

        logger.info("Knowledge graph loaded from %s (%d nodes, %d edges)",
                     path, self.graph.number_of_nodes(), self.graph.number_of_edges())

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def to_pyvis(
        self,
        subgraph_nodes: Optional[Sequence[str]] = None,
        height: str = "800px",
        width: str = "100%",
    ) -> Any:
        """Convert the graph (or a subgraph) to a ``pyvis.network.Network``.

        Parameters
        ----------
        subgraph_nodes : sequence of str, optional
            If provided, only include these nodes and edges between them.
            If ``None``, the full graph is used.
        height : str
            Height of the HTML canvas.
        width : str
            Width of the HTML canvas.

        Returns
        -------
        pyvis.network.Network
            Interactive network object.  Call ``.show("output.html")`` to render.
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError(
                "pyvis is required for visualisation. Install with: pip install pyvis"
            )

        net = Network(height=height, width=width, directed=True, notebook=False)

        # Colour mapping by node type
        colour_map = {
            "mirna": "#e74c3c",
            "gene": "#3498db",
            "pathway": "#2ecc71",
            "aesthetic": "#f39c12",
            "exosome_source": "#9b59b6",
            "disease": "#e67e22",
        }

        if subgraph_nodes is not None:
            sub = self.graph.subgraph(subgraph_nodes).copy()
        else:
            sub = self.graph

        for node, data in sub.nodes(data=True):
            node_type = data.get("node_type", "unknown")
            colour = colour_map.get(node_type, "#95a5a6")
            label = data.get("display_name", node)
            net.add_node(str(node), label=str(label), color=colour, title=node_type)

        for u, v, data in sub.edges(data=True):
            relation = data.get("relation", "")
            net.add_edge(str(u), str(v), title=relation, label=relation)

        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -3000,
              "springLength": 150
            }
          }
        }
        """)

        return net

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return self.graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.graph.number_of_edges()

    def __repr__(self) -> str:
        return (
            f"MiRNAKnowledgeGraph(nodes={self.num_nodes}, edges={self.num_edges})"
        )
