"""Knowledge Graph Explorer page.

Provides an interactive interface to query and visualise the miRNA-target-
pathway-aesthetic knowledge graph.  Users can search by miRNA ID, gene symbol,
or pathway and view the resulting sub-network rendered via pyvis.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, List, Optional

import streamlit as st
import streamlit.components.v1 as components

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Knowledge graph loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading knowledge graph / \u8f09\u5165\u77e5\u8b58\u5716\u8b5c...")
def _load_knowledge_graph() -> Any:
    """Load and return the MiRNAKnowledgeGraph instance."""
    try:
        from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph

        kg = MiRNAKnowledgeGraph()

        kg_path = Path("data") / "knowledge_graph.gpickle"
        if kg_path.exists():
            kg.load(kg_path)
            logger.info("Knowledge graph loaded from %s", kg_path)
        else:
            # Build minimal graph from aesthetic mappings
            kg.add_aesthetic_mapping()
            logger.info(
                "No persisted graph found. Built from aesthetic mappings only."
            )

        return kg
    except Exception as exc:
        logger.error("Failed to load knowledge graph: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Graph rendering helpers
# ---------------------------------------------------------------------------

def _render_pyvis_graph(
    kg: Any,
    node_ids: List[str],
    height: str = "650px",
) -> Optional[str]:
    """Render a sub-graph of the knowledge graph as an HTML string via pyvis.

    Parameters
    ----------
    kg : MiRNAKnowledgeGraph
        The knowledge graph instance.
    node_ids : list[str]
        Nodes to include in the sub-graph.
    height : str
        CSS height of the rendered graph canvas.

    Returns
    -------
    str or None
        HTML string of the rendered graph, or ``None`` on failure.
    """
    if not node_ids:
        return None

    try:
        net = kg.to_pyvis(subgraph_nodes=node_ids, height=height)

        # Write to a temporary file and read back
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".html", delete=False, encoding="utf-8",
        ) as tmp:
            net.save_graph(tmp.name)
            tmp_path = tmp.name

        html_content = Path(tmp_path).read_text(encoding="utf-8")
        return html_content

    except ImportError:
        st.error(
            "pyvis is required for graph visualisation. "
            "Install with: `pip install pyvis`  \n"
            "\u8996\u89ba\u5316\u9700\u8981 pyvis \u5957\u4ef6\uff0c\u8acb\u57f7\u884c "
            "`pip install pyvis` \u5b89\u88dd\u3002"
        )
        return None
    except Exception as exc:
        logger.error("Graph rendering failed: %s", exc)
        st.error(f"Graph rendering failed: {exc}")
        return None


def _collect_neighbourhood(kg: Any, query: str, depth: int = 1) -> List[str]:
    """Collect all nodes within *depth* hops of *query* in the knowledge graph.

    Traverses both incoming and outgoing edges.
    """
    if query not in kg.graph:
        return []

    visited = {query}
    frontier = {query}

    for _ in range(depth):
        next_frontier = set()
        for node in frontier:
            # Outgoing
            for _, target in kg.graph.out_edges(node):
                if target not in visited:
                    next_frontier.add(target)
                    visited.add(target)
            # Incoming
            for source, _ in kg.graph.in_edges(node):
                if source not in visited:
                    next_frontier.add(source)
                    visited.add(source)
        frontier = next_frontier

    return list(visited)


# ---------------------------------------------------------------------------
# Node detail panel
# ---------------------------------------------------------------------------

def _render_node_details(kg: Any, node_id: str) -> None:
    """Display details for a selected node."""
    if node_id not in kg.graph:
        st.info(f"Node `{node_id}` not found in the knowledge graph.")
        return

    node_data = kg.graph.nodes[node_id]
    node_type = node_data.get("node_type", "unknown")

    st.markdown(f"**Node:** `{node_id}`")
    st.markdown(f"**Type:** `{node_type}`")

    if node_data.get("display_name"):
        st.markdown(f"**Display Name:** {node_data['display_name']}")
    if node_data.get("display_name_zh"):
        st.markdown(f"**\u4e2d\u6587\u540d\u7a31:** {node_data['display_name_zh']}")
    if node_data.get("mechanism"):
        st.markdown(f"**Mechanism:** {node_data['mechanism']}")
    if node_data.get("is_exosomal"):
        st.markdown("**Exosomal:** Yes")

    # Connected edges
    out_edges = list(kg.graph.out_edges(node_id, data=True))
    in_edges = list(kg.graph.in_edges(node_id, data=True))

    if out_edges:
        st.markdown("**Outgoing edges:**")
        edge_rows = []
        for _, target, edata in out_edges:
            relation = edata.get("relation", "unknown")
            score = edata.get("score", "")
            edge_rows.append({
                "Target": target,
                "Relation": relation,
                "Score": score if score != "" else "-",
            })
        st.dataframe(
            edge_rows,
            use_container_width=True,
            hide_index=True,
        )

    if in_edges:
        st.markdown("**Incoming edges:**")
        edge_rows = []
        for source, _, edata in in_edges:
            relation = edata.get("relation", "unknown")
            score = edata.get("score", "")
            edge_rows.append({
                "Source": source,
                "Relation": relation,
                "Score": score if score != "" else "-",
            })
        st.dataframe(
            edge_rows,
            use_container_width=True,
            hide_index=True,
        )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_explore_page() -> None:
    """Render the knowledge graph explorer page."""
    st.header(
        "\U0001f578\ufe0f \u77e5\u8b58\u5716\u8b5c\u700f\u89bd / "
        "Knowledge Graph Explorer"
    )
    st.markdown(
        "Search by miRNA ID, gene symbol, or pathway to explore the "
        "interaction network.  \n"
        "\u4f9d miRNA \u7de8\u865f\u3001\u57fa\u56e0\u540d\u7a31\u6216\u8def\u5f91"
        "\u641c\u5c0b\u4ee5\u63a2\u7d22\u4e92\u4f5c\u7528\u7db2\u7d61\u3002"
    )

    kg = _load_knowledge_graph()

    if kg is None:
        st.error(
            "Knowledge graph could not be loaded. Please ensure the data "
            "files are available. / \u77e5\u8b58\u5716\u8b5c\u7121\u6cd5\u8f09\u5165"
            "\uff0c\u8acb\u78ba\u8a8d\u8cc7\u6599\u6a94\u6848\u5df2\u5c31\u7dd2\u3002"
        )
        return

    st.caption(
        f"Graph contains **{kg.num_nodes:,}** nodes and "
        f"**{kg.num_edges:,}** edges."
    )

    st.divider()

    # --- Search / Query UI ---------------------------------------------------
    col_query, col_filters = st.columns([2, 1])

    with col_query:
        query = st.text_input(
            "\u641c\u5c0b / Search",
            placeholder="e.g., hsa-miR-21-5p, MITF, hsa04916",
            help=(
                "Enter a miRNA ID (hsa-miR-xxx), gene symbol (MITF), "
                "KEGG pathway (hsa04916), or aesthetic category "
                "(aesthetic:whitening). / \u8f38\u5165 miRNA \u7de8\u865f\u3001"
                "\u57fa\u56e0\u7b26\u865f\u3001KEGG \u8def\u5f91\u6216"
                "\u91ab\u7f8e\u985e\u5225\u3002"
            ),
        )

    with col_filters:
        st.markdown("**Filters / \u7be9\u9078**")

        evidence_filter = st.multiselect(
            "Evidence Type / \u8b49\u64da\u985e\u578b",
            options=[
                "CLIP-Seq",
                "Reporter assay",
                "Western blot",
                "qRT-PCR",
                "Microarray",
                "Other",
            ],
            default=[],
            help=(
                "Filter edges by experimental evidence type. "
                "Leave empty to show all. / "
                "\u4f9d\u5be6\u9a57\u8b49\u64da\u985e\u578b\u7be9\u9078\u908a\u3002"
            ),
        )

        confidence_threshold = st.slider(
            "Confidence Threshold / \u4fe1\u5fc3\u5ea6\u9580\u6abb",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
        )

        aesthetic_filter = st.multiselect(
            "Aesthetic Category / \u91ab\u7f8e\u985e\u5225",
            options=[
                "whitening",
                "scar_removal",
                "hair_restoration",
                "anti_aging",
                "skin_refinement",
            ],
            default=[],
            help=(
                "Filter by aesthetic medicine category. / "
                "\u4f9d\u91ab\u7f8e\u985e\u5225\u7be9\u9078\u3002"
            ),
        )

        depth = st.slider(
            "Neighbourhood Depth / \u9130\u57df\u6df1\u5ea6",
            min_value=1,
            max_value=3,
            value=1,
        )

    # --- Execute query -------------------------------------------------------
    if query.strip():
        query_clean = query.strip()

        # Try exact match first; if not found, try as aesthetic prefix
        if query_clean not in kg.graph:
            # Try with aesthetic: prefix
            if f"aesthetic:{query_clean}" in kg.graph:
                query_clean = f"aesthetic:{query_clean}"
            else:
                # Partial match search
                matches = [
                    n for n in kg.graph.nodes
                    if query_clean.lower() in str(n).lower()
                ]
                if matches:
                    st.info(
                        f"Exact match not found. Showing results for "
                        f"`{matches[0]}` ({len(matches)} partial matches). / "
                        f"\u672a\u627e\u5230\u5b8c\u5168\u7b26\u5408\u7684\u7d50\u679c"
                        f"\uff0c\u986f\u793a `{matches[0]}` \u7684\u7d50\u679c\u3002"
                    )
                    query_clean = matches[0]
                else:
                    st.warning(
                        f"No matching nodes found for `{query_clean}`. / "
                        f"\u627e\u4e0d\u5230\u7b26\u5408 `{query_clean}` "
                        f"\u7684\u7bc0\u9ede\u3002"
                    )
                    return

        # Collect neighbourhood
        node_ids = _collect_neighbourhood(kg, query_clean, depth=depth)

        # Apply aesthetic category filter
        if aesthetic_filter:
            aesthetic_nodes = {
                f"aesthetic:{cat}" for cat in aesthetic_filter
            }
            filtered = []
            for nid in node_ids:
                ntype = kg.graph.nodes[nid].get("node_type", "")
                if ntype == "aesthetic":
                    if nid in aesthetic_nodes:
                        filtered.append(nid)
                else:
                    filtered.append(nid)
            node_ids = filtered if filtered else node_ids

        st.divider()

        # --- Graph visualisation -----------------------------------------------
        col_graph, col_detail = st.columns([3, 1])

        with col_graph:
            st.subheader(
                f"\u7db2\u7d61\u5716 / Network Graph ({len(node_ids)} nodes)"
            )
            html_content = _render_pyvis_graph(kg, node_ids, height="650px")

            if html_content:
                components.html(html_content, height=680, scrolling=True)
            else:
                st.info(
                    "No graph to display. / \u7121\u5716\u8868\u53ef\u986f\u793a\u3002"
                )

        with col_detail:
            st.subheader("\u7bc0\u9ede\u8a73\u60c5 / Node Details")

            # Let user pick a node to inspect
            selected_node = st.selectbox(
                "Select node / \u9078\u64c7\u7bc0\u9ede",
                options=sorted(node_ids),
                index=0 if node_ids else None,
            )

            if selected_node:
                _render_node_details(kg, selected_node)

    else:
        # Show help when no query
        st.info(
            "Enter a query above to explore the knowledge graph. "
            "Try `hsa-miR-21-5p`, `MITF`, or `whitening`. / "
            "\u5728\u4e0a\u65b9\u8f38\u5165\u67e5\u8a62\u4ee5\u63a2\u7d22"
            "\u77e5\u8b58\u5716\u8b5c\u3002\u8a66\u8a66 `hsa-miR-21-5p`\u3001"
            "`MITF` \u6216 `whitening`\u3002"
        )

        # Show graph statistics
        if kg is not None:
            st.markdown("### \u5716\u8b5c\u7d71\u8a08 / Graph Statistics")

            node_types: dict[str, int] = {}
            for _, data in kg.graph.nodes(data=True):
                nt = data.get("node_type", "unknown")
                node_types[nt] = node_types.get(nt, 0) + 1

            col_stats = st.columns(min(len(node_types), 4))
            for i, (nt, count) in enumerate(
                sorted(node_types.items(), key=lambda x: -x[1])
            ):
                col_stats[i % len(col_stats)].metric(
                    nt.replace("_", " ").title(), f"{count:,}"
                )
