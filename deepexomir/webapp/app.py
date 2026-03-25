"""DeepExoMir -- Streamlit Web Application.

Main entry point for the DeepExoMir miRNA target prediction platform.
Provides a bilingual (English / Traditional Chinese) interface with four
main pages: Prediction, Knowledge Graph Explorer, Aesthetic Medicine
Dashboard, and Exosome miRNA Catalog.

Usage::

    streamlit run deepexomir/webapp/app.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="DeepExoMir - \u5916\u6ccc\u9ad4 miRNA \u9776\u6a19\u9810\u6e2c\u5e73\u53f0",
    page_icon="\U0001f9ec",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "DeepExoMir v0.1.0  \n"
            "Exosome miRNA Target Prediction Platform  \n"
            "\u8a0a\u806f\u96c6\u5718 (XunLian Group)"
        ),
    },
)

# ---------------------------------------------------------------------------
# Inject custom CSS
# ---------------------------------------------------------------------------

_CSS_PATH = Path(__file__).parent / "assets" / "style.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Logo / branding area
    st.markdown(
        """
        <div class="sidebar-logo">
            <h2 style="margin-bottom:0;">DeepExoMir</h2>
            <p style="color:#6c8ebf; font-size:0.85rem; margin-top:0;">
                \u8a0a\u806f\u96c6\u5718 &middot; XunLian Group
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Navigation
    page = st.radio(
        "\u5c0e\u89bd / Navigation",
        options=[
            "\U0001f50d \u9810\u6e2c / Predict",
            "\U0001f578\ufe0f \u77e5\u8b58\u5716\u8b5c / Explore",
            "\u2728 \u91ab\u7f8e / Aesthetic",
            "\U0001f9eb \u5916\u6ccc\u9ad4 / Exosome",
        ],
        index=0,
        label_visibility="visible",
    )

    st.divider()

    # Model info
    st.markdown("#### \u6a21\u578b\u8cc7\u8a0a / Model Info")
    st.caption(
        "**Backbone:** RiNALMo-Giga (1.28 B)  \n"
        "**Architecture:** Cross-Attention + Structural  \n"
        "**Training:** miRTarBase + ExoCarta  \n"
        "**Version:** 0.1.0-dev"
    )

    st.divider()
    st.caption("\u00a9 2025 \u8a0a\u806f\u96c6\u5718 | DeepExoMir Project")

# ---------------------------------------------------------------------------
# Title / Header
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="main-header">
        <h1>DeepExoMir</h1>
        <p class="subtitle">
            \u5916\u6ccc\u9ad4 miRNA \u9776\u6a19\u9810\u6e2c\u5e73\u53f0 &nbsp;|&nbsp;
            Exosome miRNA Target Prediction Platform
        </p>
        <p class="intro">
            \u7d50\u5408\u6df1\u5ea6\u5b78\u7fd2\u8207\u77e5\u8b58\u5716\u8b5c\uff0c\u5c08\u70ba\u91ab\u7f8e\u5916\u6ccc\u9ad4\u61c9\u7528\u8a2d\u8a08\u7684 miRNA \u9776\u6a19\u4e92\u4f5c\u7528\u9810\u6e2c\u5de5\u5177\u3002<br/>
            A deep-learning and knowledge-graph powered tool for predicting miRNA-target
            interactions, designed for aesthetic medicine exosome applications.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Page routing
# ---------------------------------------------------------------------------

if "\u9810\u6e2c / Predict" in page:
    from deepexomir.webapp.pages.predict import render_predict_page
    render_predict_page()

elif "\u77e5\u8b58\u5716\u8b5c / Explore" in page:
    from deepexomir.webapp.pages.explore import render_explore_page
    render_explore_page()

elif "\u91ab\u7f8e / Aesthetic" in page:
    from deepexomir.webapp.pages.aesthetic import render_aesthetic_page
    render_aesthetic_page()

elif "\u5916\u6ccc\u9ad4 / Exosome" in page:
    from deepexomir.webapp.pages.exosome import render_exosome_page
    render_exosome_page()
