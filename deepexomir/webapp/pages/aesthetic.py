"""Aesthetic Medicine Dashboard page.

Displays five aesthetic medicine categories relevant to XunLian Group's
exosome products.  Each category tab shows key miRNAs, target genes,
KEGG pathway information, mechanism descriptions, and summary charts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading aesthetic data / \u8f09\u5165\u91ab\u7f8e\u8cc7\u6599...")
def _load_aesthetic_data() -> Dict[str, Any]:
    """Load the aesthetic pathway data from constants and knowledge graph."""
    try:
        from deepexomir.utils.constants import AESTHETIC_PATHWAYS
        from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph

        kg = MiRNAKnowledgeGraph()

        kg_path = Path("data") / "knowledge_graph.gpickle"
        if kg_path.exists():
            kg.load(kg_path)
        else:
            kg.add_aesthetic_mapping()

        return {
            "pathways": AESTHETIC_PATHWAYS,
            "kg": kg,
        }
    except Exception as exc:
        logger.error("Failed to load aesthetic data: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# KEGG pathway descriptions (brief)
# ---------------------------------------------------------------------------

_KEGG_DESCRIPTIONS = {
    "hsa04916": "Melanogenesis -- melanin biosynthesis and regulation",
    "hsa04350": "TGF-beta signaling -- growth, differentiation, fibrosis",
    "hsa04310": "Wnt signaling -- cell fate, polarity, hair follicle cycling",
    "hsa04151": "PI3K-Akt signaling -- cell survival, growth, metabolism",
    "hsa04060": "Cytokine-cytokine receptor interaction",
    "hsa04115": "p53 signaling -- cell cycle arrest, apoptosis, senescence",
    "hsa04210": "Apoptosis -- programmed cell death",
    "hsa04211": "Longevity regulating pathway",
    "hsa04150": "mTOR signaling -- cell growth, autophagy",
    "hsa04510": "Focal adhesion -- cell-ECM interactions",
    "hsa04512": "ECM-receptor interaction -- extracellular matrix signalling",
}

# Chinese mechanism descriptions for each category
_MECHANISM_ZH = {
    "whitening": (
        "\u900f\u904e\u6291\u5236 MITF/TYR \u8ef8\u4ee5\u964d\u4f4e"
        "\u9ed1\u8272\u7d20\u751f\u6210\uff0c\u9054\u5230\u7f8e\u767d\u6548\u679c\u3002"
        "\u95dc\u9375 miRNA \u53ef\u76f4\u63a5\u6291\u5236 MITF \u8f49\u9304"
        "\u56e0\u5b50\u6216\u5176\u4e0b\u6e38\u9176\u7d20 TYR\u3001TYRP1\u3001DCT\u3002"
    ),
    "scar_removal": (
        "\u900f\u904e\u6291\u5236 TGF-beta/Smad \u4fe1\u865f\u901a\u8def"
        "\u964d\u4f4e\u81a0\u539f\u86cb\u767d\u904e\u5ea6\u751f\u6210\uff0c\u6e1b\u5c11"
        "\u7e96\u7dad\u5316\u53ca\u75a4\u75d5\u5f62\u6210\u3002miR-29 \u5bb6\u65cf"
        "\u662f\u6700\u91cd\u8981\u7684\u6297\u7e96\u7dad\u5316 miRNA\u3002"
    ),
    "hair_restoration": (
        "\u6d3b\u5316 Wnt/beta-catenin \u4fe1\u865f\u901a\u8def\u4fc3\u9032"
        "\u6bdb\u56ca\u518d\u751f\u8207\u9aea\u8272\u7d20\u7d30\u80de\u529f\u80fd"
        "\u6062\u5fa9\uff0c\u5be6\u73fe\u767d\u9aee\u9006\u8f49\u8207\u751f\u9aee"
        "\u6548\u679c\u3002"
    ),
    "anti_aging": (
        "\u8abf\u7bc0 p53/SIRT1 \u8870\u8001\u901a\u8def\u4ee5\u53ca\u7aef\u7c92"
        "\u7dad\u8b77\uff0c\u5ef6\u7de9\u7d30\u80de\u8870\u8001\uff0c\u4fc3\u9032"
        "\u7d30\u80de\u56de\u6625\u8207\u81ea\u566a\u6a5f\u5236\u3002"
    ),
    "skin_refinement": (
        "\u5e73\u8861\u81a0\u539f\u86cb\u767d\u5408\u6210\u8207 MMP \u5a92\u4ecb\u7684"
        "\u964d\u89e3\uff0c\u7dad\u6301\u76ae\u819a\u7d30\u80de\u5916\u57fa\u8cea"
        "\u5065\u5eb7\uff0c\u63d0\u5347\u808c\u819a\u7d30\u7dfb\u5ea6\u8207\u5f48\u6027\u3002"
    ),
}


# ---------------------------------------------------------------------------
# Category tab rendering
# ---------------------------------------------------------------------------

def _render_category_tab(
    category: str,
    info: Dict[str, Any],
    kg: Any,
) -> None:
    """Render the content for a single aesthetic category tab."""

    display_name = info["display_name"]
    display_name_zh = info.get("display_name_zh", "")

    st.markdown(
        f"### {display_name_zh}  \n"
        f"*{display_name}*"
    )

    # --- Mechanism description ------------------------------------------------
    st.markdown("#### \u4f5c\u7528\u6a5f\u5236 / Mechanism")
    col_en, col_zh = st.columns(2)
    with col_en:
        st.markdown(f"**English:** {info.get('mechanism', 'N/A')}")
    with col_zh:
        mechanism_zh = _MECHANISM_ZH.get(category, "\u66ab\u7121\u8aaa\u660e")
        st.markdown(f"**\u4e2d\u6587:** {mechanism_zh}")

    st.divider()

    # --- Key miRNAs -----------------------------------------------------------
    col_mirna, col_gene = st.columns(2)

    with col_mirna:
        st.markdown("#### \u95dc\u9375 miRNA / Key miRNAs")

        # Get ranked miRNAs from KG if available
        mirna_list = info.get("key_mirnas", [])
        if kg is not None:
            try:
                ranked = kg.get_aesthetic_mirnas(category)
                if ranked:
                    mirna_df = pd.DataFrame(ranked)
                    mirna_df = mirna_df.rename(columns={
                        "mirna_id": "miRNA ID",
                        "score": "Relevance Score",
                        "target_count": "Target Count",
                        "is_exosomal": "Exosomal",
                    })
                    st.dataframe(
                        mirna_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Relevance Score": st.column_config.NumberColumn(
                                format="%.2f",
                            ),
                            "Exosomal": st.column_config.CheckboxColumn(),
                        },
                    )
                else:
                    # Fallback to simple list
                    mirna_df = pd.DataFrame(
                        {"miRNA ID": mirna_list, "Rank": range(1, len(mirna_list) + 1)}
                    )
                    st.dataframe(
                        mirna_df, use_container_width=True, hide_index=True,
                    )
            except Exception:
                mirna_df = pd.DataFrame(
                    {"miRNA ID": mirna_list, "Rank": range(1, len(mirna_list) + 1)}
                )
                st.dataframe(
                    mirna_df, use_container_width=True, hide_index=True,
                )
        else:
            mirna_df = pd.DataFrame(
                {"miRNA ID": mirna_list, "Rank": range(1, len(mirna_list) + 1)}
            )
            st.dataframe(
                mirna_df, use_container_width=True, hide_index=True,
            )

    # --- Key genes ------------------------------------------------------------
    with col_gene:
        st.markdown("#### \u95dc\u9375\u57fa\u56e0 / Key Target Genes")
        gene_list = info.get("key_genes", [])
        gene_df = pd.DataFrame(
            {"Gene Symbol": gene_list, "Rank": range(1, len(gene_list) + 1)}
        )
        st.dataframe(gene_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- KEGG Pathways --------------------------------------------------------
    st.markdown("#### KEGG \u4fe1\u865f\u901a\u8def / Signalling Pathways")
    kegg_ids = info.get("kegg_pathways", [])
    kegg_rows = []
    for kid in kegg_ids:
        desc = _KEGG_DESCRIPTIONS.get(kid, "")
        kegg_rows.append({
            "Pathway ID": kid,
            "Description": desc,
            "KEGG Link": f"https://www.kegg.jp/pathway/{kid}",
        })
    kegg_df = pd.DataFrame(kegg_rows)
    st.dataframe(
        kegg_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "KEGG Link": st.column_config.LinkColumn("KEGG Link"),
        },
    )

    # --- Charts (plotly) ------------------------------------------------------
    st.markdown("#### \u7d71\u8a08\u5716\u8868 / Statistics")
    _render_category_charts(category, info, kg)


def _render_category_charts(
    category: str,
    info: Dict[str, Any],
    kg: Any,
) -> None:
    """Render plotly charts for a category."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError:
        st.info(
            "Install plotly for interactive charts: `pip install plotly` / "
            "\u5b89\u88dd plotly \u4ee5\u986f\u793a\u4e92\u52d5\u5716\u8868\u3002"
        )
        return

    col_chart1, col_chart2 = st.columns(2)

    # Chart 1: miRNA count by exosomal status
    mirna_list = info.get("key_mirnas", [])
    exosomal_count = 0
    non_exosomal_count = 0

    if kg is not None:
        for mid in mirna_list:
            node_data = kg.graph.nodes.get(mid, {})
            if node_data.get("is_exosomal", False):
                exosomal_count += 1
            else:
                non_exosomal_count += 1
    else:
        non_exosomal_count = len(mirna_list)

    with col_chart1:
        pie_data = pd.DataFrame({
            "Status": [
                "Exosomal / \u5916\u6ccc\u9ad4",
                "Non-exosomal / \u975e\u5916\u6ccc\u9ad4",
            ],
            "Count": [exosomal_count, non_exosomal_count],
        })
        fig_pie = px.pie(
            pie_data,
            values="Count",
            names="Status",
            title="miRNA Exosome Status / miRNA \u5916\u6ccc\u9ad4\u72c0\u614b",
            color_discrete_sequence=["#3498db", "#bdc3c7"],
            hole=0.4,
        )
        fig_pie.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            font=dict(size=12),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Chart 2: Genes per pathway
    with col_chart2:
        kegg_ids = info.get("kegg_pathways", [])
        gene_list = info.get("key_genes", [])

        pathway_gene_counts = []
        for kid in kegg_ids:
            count = 0
            if kg is not None:
                for gene in gene_list:
                    gene_data = kg.query_gene(gene)
                    if kid in gene_data.get("pathways", []):
                        count += 1
            else:
                # Approximate: distribute genes across pathways
                count = len(gene_list) // max(len(kegg_ids), 1)

            pathway_gene_counts.append({
                "Pathway": kid,
                "Gene Count": count,
            })

        if pathway_gene_counts:
            bar_df = pd.DataFrame(pathway_gene_counts)
            fig_bar = px.bar(
                bar_df,
                x="Pathway",
                y="Gene Count",
                title="Key Genes per Pathway / \u5404\u901a\u8def\u95dc\u9375\u57fa\u56e0\u6578",
                color_discrete_sequence=["#2ecc71"],
            )
            fig_bar.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
                font=dict(size=12),
            )
            st.plotly_chart(fig_bar, use_container_width=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def _render_summary(pathways: Dict[str, Any]) -> None:
    """Render cross-category summary statistics and charts."""
    st.subheader("\u7e3d\u89bd / Summary")

    # Collect stats
    stats = []
    for cat, info in pathways.items():
        stats.append({
            "Category": info.get("display_name_zh", cat),
            "Category (EN)": info.get("display_name", cat),
            "Key miRNAs": len(info.get("key_mirnas", [])),
            "Key Genes": len(info.get("key_genes", [])),
            "KEGG Pathways": len(info.get("kegg_pathways", [])),
        })

    stats_df = pd.DataFrame(stats)

    col_table, col_chart = st.columns([1, 1])

    with col_table:
        st.dataframe(stats_df, use_container_width=True, hide_index=True)

    with col_chart:
        try:
            import plotly.express as px

            melted = stats_df.melt(
                id_vars=["Category"],
                value_vars=["Key miRNAs", "Key Genes", "KEGG Pathways"],
                var_name="Metric",
                value_name="Count",
            )

            fig = px.bar(
                melted,
                x="Category",
                y="Count",
                color="Metric",
                barmode="group",
                title=(
                    "\u5404\u985e\u5225\u95dc\u9375\u5143\u7d20\u6578\u91cf / "
                    "Key Elements per Category"
                ),
                color_discrete_sequence=["#e74c3c", "#3498db", "#2ecc71"],
            )
            fig.update_layout(
                height=400,
                margin=dict(l=20, r=20, t=40, b=80),
                font=dict(size=11),
                xaxis_tickangle=-30,
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.info(
                "Install plotly for charts: `pip install plotly` / "
                "\u5b89\u88dd plotly \u4ee5\u986f\u793a\u5716\u8868\u3002"
            )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

_CATEGORY_TAB_NAMES = {
    "whitening": "\u7f8e\u767d Whitening",
    "scar_removal": "\u6deb\u75a4 Scar Removal",
    "hair_restoration": "\u767d\u9aee\u9006\u8f49 Hair",
    "anti_aging": "\u7d30\u80de\u56de\u6625 Anti-Aging",
    "skin_refinement": "\u808c\u819a\u7d30\u7dfb Skin",
}


def render_aesthetic_page() -> None:
    """Render the aesthetic medicine dashboard page."""
    st.header(
        "\u2728 \u91ab\u7f8e\u61c9\u7528\u5100\u8868\u677f / "
        "Aesthetic Medicine Dashboard"
    )
    st.markdown(
        "Explore miRNA-target interactions relevant to aesthetic medicine "
        "applications for exosome-based products.  \n"
        "\u63a2\u7d22\u8207\u91ab\u7f8e\u5916\u6ccc\u9ad4\u7522\u54c1\u76f8\u95dc\u7684 "
        "miRNA-\u9776\u6a19\u4e92\u4f5c\u7528\u3002"
    )

    data = _load_aesthetic_data()
    if not data:
        st.error(
            "Aesthetic data could not be loaded. Please check your "
            "installation. / \u91ab\u7f8e\u8cc7\u6599\u7121\u6cd5\u8f09\u5165"
            "\uff0c\u8acb\u6aa2\u67e5\u5b89\u88dd\u3002"
        )
        return

    pathways = data["pathways"]
    kg = data.get("kg")

    # --- Summary statistics at top -------------------------------------------
    _render_summary(pathways)

    st.divider()

    # --- Category tabs -------------------------------------------------------
    category_keys = list(pathways.keys())
    tab_labels = [
        _CATEGORY_TAB_NAMES.get(k, k) for k in category_keys
    ]
    tabs = st.tabs(tab_labels)

    for tab, cat_key in zip(tabs, category_keys):
        with tab:
            _render_category_tab(cat_key, pathways[cat_key], kg)
