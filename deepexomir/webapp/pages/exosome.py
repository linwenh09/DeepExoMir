"""Exosome miRNA Catalog page.

Provides a searchable and filterable table of exosome-enriched miRNAs
sourced from ExoCarta and annotated with aesthetic medicine categories.
Supports expanding rows for detail views and CSV export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(
    show_spinner="Loading exosome miRNA catalog / \u8f09\u5165\u5916\u6ccc\u9ad4 miRNA \u76ee\u9304..."
)
def _load_exosome_catalog() -> pd.DataFrame:
    """Build the exosome miRNA catalog DataFrame.

    Combines ExoCarta data (if available) with aesthetic pathway annotations
    from constants.  If the ExoCarta file is not present, a demonstration
    catalog is built from the known aesthetic miRNAs.

    Returns
    -------
    pd.DataFrame
        Columns: miRNA ID, Sequence, Aesthetic Categories, Top Targets,
        Score, Source.
    """
    from deepexomir.utils.constants import (
        AESTHETIC_PATHWAYS,
        ALL_AESTHETIC_MIRNAS,
        MIRNA_TO_CATEGORIES,
    )

    # Attempt to load ExoCarta data
    exo_mirnas: Set[str] = set()
    exocarta_path = Path("data") / "ExoCarta_miRNA_details.txt"

    if exocarta_path.exists():
        try:
            from deepexomir.annotation.exosome_filter import ExosomeFilter

            exo_filter = ExosomeFilter(exocarta_path)
            exo_mirnas = exo_filter.exosome_mirnas
            logger.info("Loaded %d miRNAs from ExoCarta", len(exo_mirnas))
        except Exception as exc:
            logger.warning("Could not load ExoCarta: %s", exc)

    # Attempt to load knowledge graph for target info
    kg = None
    try:
        from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph

        kg = MiRNAKnowledgeGraph()
        kg_path = Path("data") / "knowledge_graph.gpickle"
        if kg_path.exists():
            kg.load(kg_path)
        else:
            kg.add_aesthetic_mapping()
    except Exception:
        pass

    # Combine exosome miRNAs with aesthetic miRNAs
    all_mirnas = exo_mirnas | ALL_AESTHETIC_MIRNAS

    # Known miRNA sequences (subset for demonstration)
    _KNOWN_SEQUENCES = {
        "hsa-miR-21-5p": "UAGCUUAUCAGACUGAUGUUGA",
        "hsa-miR-29a-3p": "UAGCACCAUCUGAAAUCGGUUA",
        "hsa-miR-29b-3p": "UAGCACCAUUUGAAAUCAGUGUU",
        "hsa-miR-29c-3p": "UAGCACCAUUUGAAAUCGGUUA",
        "hsa-miR-34a-5p": "UGGCAGUGUCUUAGCUGGUUGU",
        "hsa-miR-146a-5p": "UGAGAACUGAAUUCCAUGGGUU",
        "hsa-miR-155-5p": "UUAAUGCUAAUUGUGAUAGGGGU",
        "hsa-miR-200a-3p": "UAACACUGUCUGGUAACGAUGU",
        "hsa-miR-200b-3p": "UAAUACUGCCUGGUAAUGAUGA",
        "hsa-miR-145-5p": "GUCCAGUUUUCCCAGGAAUCCCU",
        "hsa-miR-137": "UUAUUGCUUAAGAAUACGCGUAG",
        "hsa-miR-141-3p": "UAACACUGUCUGGUAAAGAUGG",
        "hsa-miR-330-5p": "UCUCUGGGCCUGUGUCUUAGGC",
        "hsa-miR-181a-5p": "AACAUUCAACGCUGUCGGUGAGU",
        "hsa-miR-125b-5p": "UCCCUGAGACCCUAACUUGUGA",
        "hsa-miR-148a-3p": "UCAGUGCACUACAGAACUUUGU",
        "hsa-miR-192-5p": "CUGACCUAUGAAUUGACAGCC",
        "hsa-miR-205-5p": "UCCUUCAUUCCACCGGAGUCUG",
        "hsa-miR-218-5p": "UUGUGCUUGAUCUAACCAUGU",
        "hsa-miR-31-5p": "AGGCAAGAUGCUGGCAUAGCUG",
        "hsa-miR-16-5p": "UAGCAGCACGUAAAUAUUGGCG",
        "hsa-miR-22-5p": "AGUUCUUCAGUGGCAAGCUUUA",
        "hsa-miR-133a-3p": "UUUGGUCCCCUUCAACCAGCUG",
        "hsa-miR-152-3p": "UCAGUGCAUGACAGAACUUGG",
        "hsa-miR-196a-5p": "UAGGUAGUUUCAUGUUGUUGGG",
        "hsa-miR-340-5p": "UUAUAAAGCAAUGAGACUGAUU",
        "hsa-miR-25-3p": "CAUUGCACUUGUCUCGGUCUGA",
        "hsa-miR-182-5p": "UUUGGCAAUGGUAGAACUCACACU",
        "hsa-miR-508-3p": "UGAUUGUAGCCUUUUGGAGUAGA",
        "hsa-miR-211-5p": "UUCCCUUUGUCAUCCUUCGCCU",
        "hsa-miR-149-5p": "UCUGGCUCCGUGUCUUCACUCCC",
        "hsa-miR-92a-3p": "UAUUGCACUUGUCCCGGCCUGU",
        "hsa-miR-140-5p": "CAGUGGUUUUACCCUAUGGUAG",
        "hsa-miR-214-5p": "UGCCUGUCUACACUUGCUGUGC",
        "hsa-miR-24-3p": "UGGCUCAGUUCAGCAGGAACAG",
        "hsa-miR-199a-5p": "CCCAGUGUUCAGACUACCUGUUC",
    }

    rows = []
    for mirna_id in sorted(all_mirnas):
        # Aesthetic categories
        categories = MIRNA_TO_CATEGORIES.get(mirna_id, [])
        cat_display = ", ".join(
            AESTHETIC_PATHWAYS[c]["display_name_zh"]
            for c in categories
            if c in AESTHETIC_PATHWAYS
        ) if categories else "-"

        # Sequence
        seq = _KNOWN_SEQUENCES.get(mirna_id, "-")

        # Top targets from KG
        top_targets = "-"
        target_score = 0.0
        if kg is not None:
            try:
                mirna_info = kg.query_mirna(mirna_id)
                targets = mirna_info.get("targets", [])
                if targets:
                    sorted_targets = sorted(
                        targets, key=lambda x: x.get("score", 0), reverse=True
                    )
                    top_names = [t["gene"] for t in sorted_targets[:5]]
                    top_targets = ", ".join(top_names)
                    target_score = sum(
                        t.get("score", 0) for t in sorted_targets[:5]
                    )
            except Exception:
                pass

        # Source
        source = "ExoCarta" if mirna_id in exo_mirnas else "Aesthetic DB"

        rows.append({
            "miRNA ID": mirna_id,
            "Sequence": seq,
            "Aesthetic Categories": cat_display,
            "Top Targets": top_targets,
            "Score": round(target_score, 2),
            "Source": source,
        })

    if not rows:
        # Fallback demo data
        for mirna_id in sorted(ALL_AESTHETIC_MIRNAS):
            categories = MIRNA_TO_CATEGORIES.get(mirna_id, [])
            cat_display = ", ".join(
                AESTHETIC_PATHWAYS[c]["display_name_zh"]
                for c in categories
                if c in AESTHETIC_PATHWAYS
            ) if categories else "-"
            seq = _KNOWN_SEQUENCES.get(mirna_id, "-")
            rows.append({
                "miRNA ID": mirna_id,
                "Sequence": seq,
                "Aesthetic Categories": cat_display,
                "Top Targets": "-",
                "Score": 0.0,
                "Source": "Aesthetic DB",
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Detail expander
# ---------------------------------------------------------------------------

def _render_mirna_detail(mirna_id: str, catalog_df: pd.DataFrame) -> None:
    """Show expanded details for a selected miRNA."""
    row = catalog_df[catalog_df["miRNA ID"] == mirna_id]
    if row.empty:
        st.warning(f"No data for {mirna_id}")
        return

    row = row.iloc[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**miRNA ID:** `{row['miRNA ID']}`")
        st.markdown(f"**Sequence:** `{row['Sequence']}`")
        st.markdown(f"**Source:** {row['Source']}")
    with col2:
        st.markdown(f"**Aesthetic Categories:** {row['Aesthetic Categories']}")
        st.markdown(f"**Top Targets:** {row['Top Targets']}")
        st.markdown(f"**Aggregate Score:** {row['Score']}")

    # Additional KG info
    try:
        from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph

        kg = MiRNAKnowledgeGraph()
        kg_path = Path("data") / "knowledge_graph.gpickle"
        if kg_path.exists():
            kg.load(kg_path)
        else:
            kg.add_aesthetic_mapping()

        info = kg.query_mirna(mirna_id)
        if info.get("targets"):
            st.markdown("**All known targets:**")
            target_df = pd.DataFrame(info["targets"])
            target_df = target_df.rename(columns={
                "gene": "Gene",
                "score": "Score",
                "evidence_types": "Evidence",
            })
            st.dataframe(target_df, use_container_width=True, hide_index=True)

        if info.get("pathways"):
            st.markdown(
                f"**Associated pathways:** {', '.join(info['pathways'])}"
            )

        if info.get("diseases"):
            st.markdown(
                f"**Disease associations:** {', '.join(info['diseases'][:10])}"
            )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_exosome_page() -> None:
    """Render the exosome miRNA catalog page."""
    st.header(
        "\U0001f9eb \u5916\u6ccc\u9ad4 miRNA \u76ee\u9304 / "
        "Exosome miRNA Catalog"
    )
    st.markdown(
        "Browse and filter exosome-enriched miRNAs from ExoCarta with "
        "aesthetic medicine annotations.  \n"
        "\u700f\u89bd\u4e26\u7be9\u9078\u4f86\u81ea ExoCarta \u7684"
        "\u5916\u6ccc\u9ad4 miRNA\uff0c\u542b\u91ab\u7f8e\u8a3b\u91cb\u3002"
    )

    catalog_df = _load_exosome_catalog()

    if catalog_df.empty:
        st.warning(
            "No exosome miRNA data available. Please configure the ExoCarta "
            "database. / \u7121\u5916\u6ccc\u9ad4 miRNA \u8cc7\u6599\uff0c"
            "\u8acb\u914d\u7f6e ExoCarta \u8cc7\u6599\u5eab\u3002"
        )
        _show_setup_instructions()
        return

    st.caption(f"Total miRNAs: **{len(catalog_df)}**")

    st.divider()

    # --- Filters --------------------------------------------------------------
    col_search, col_cat, col_source = st.columns([2, 1, 1])

    with col_search:
        search_query = st.text_input(
            "\u641c\u5c0b miRNA / Search miRNA",
            placeholder="e.g., miR-21, miR-29a",
            help="Filter by miRNA ID (partial match supported)",
        )

    with col_cat:
        # Extract unique categories
        all_categories_raw = set()
        for cats in catalog_df["Aesthetic Categories"]:
            if cats != "-":
                for c in cats.split(", "):
                    all_categories_raw.add(c.strip())
        all_categories_list = sorted(all_categories_raw)

        category_filter = st.multiselect(
            "\u91ab\u7f8e\u985e\u5225 / Aesthetic Category",
            options=all_categories_list,
            default=[],
        )

    with col_source:
        source_options = sorted(catalog_df["Source"].unique().tolist())
        source_filter = st.multiselect(
            "\u4f86\u6e90 / Source",
            options=source_options,
            default=[],
        )

    # --- Apply filters --------------------------------------------------------
    filtered_df = catalog_df.copy()

    if search_query.strip():
        mask = filtered_df["miRNA ID"].str.contains(
            search_query.strip(), case=False, na=False
        )
        filtered_df = filtered_df[mask]

    if category_filter:
        mask = filtered_df["Aesthetic Categories"].apply(
            lambda x: any(cat in x for cat in category_filter)
        )
        filtered_df = filtered_df[mask]

    if source_filter:
        filtered_df = filtered_df[filtered_df["Source"].isin(source_filter)]

    st.markdown(
        f"**Showing {len(filtered_df)} of {len(catalog_df)} miRNAs / "
        f"\u986f\u793a {len(filtered_df)} / {len(catalog_df)} \u7b46 miRNA**"
    )

    # --- Data table -----------------------------------------------------------
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "miRNA ID": st.column_config.TextColumn("miRNA ID", width="medium"),
            "Sequence": st.column_config.TextColumn("Sequence", width="large"),
            "Aesthetic Categories": st.column_config.TextColumn(
                "Aesthetic Categories", width="large"
            ),
            "Top Targets": st.column_config.TextColumn(
                "Top Targets", width="large"
            ),
            "Score": st.column_config.NumberColumn("Score", format="%.2f"),
            "Source": st.column_config.TextColumn("Source", width="small"),
        },
        height=500,
    )

    # --- Detail expander ------------------------------------------------------
    st.divider()
    st.subheader("\u8a73\u7d30\u8cc7\u8a0a / Detailed View")

    if not filtered_df.empty:
        selected_mirna = st.selectbox(
            "Select miRNA for details / \u9078\u64c7 miRNA \u67e5\u770b\u8a73\u60c5",
            options=filtered_df["miRNA ID"].tolist(),
        )

        if selected_mirna:
            with st.expander(
                f"\u8a73\u7d30: {selected_mirna}", expanded=True
            ):
                _render_mirna_detail(selected_mirna, catalog_df)

    # --- Export ---------------------------------------------------------------
    st.divider()
    csv_data = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Catalog as CSV / \u4e0b\u8f09\u76ee\u9304",
        data=csv_data,
        file_name="deepexomir_exosome_catalog.csv",
        mime="text/csv",
    )


def _show_setup_instructions() -> None:
    """Display setup instructions when data is not available."""
    st.markdown(
        """
        ### Setup Instructions / \u8a2d\u5b9a\u6307\u5357

        1. Download the ExoCarta miRNA data file from
           [ExoCarta](http://www.exocarta.org) and place it at:
           ```
           data/ExoCarta_miRNA_details.txt
           ```

        2. Build the knowledge graph by running:
           ```bash
           python -m deepexomir.annotation.knowledge_graph
           ```

        3. Restart this application.

        ---

        1. \u5f9e [ExoCarta](http://www.exocarta.org) \u4e0b\u8f09 miRNA
           \u8cc7\u6599\u6a94\u4e26\u653e\u7f6e\u65bc `data/ExoCarta_miRNA_details.txt`
        2. \u57f7\u884c `python -m deepexomir.annotation.knowledge_graph`
           \u5efa\u7acb\u77e5\u8b58\u5716\u8b5c
        3. \u91cd\u65b0\u555f\u52d5\u672c\u61c9\u7528\u7a0b\u5f0f
        """
    )
