"""DeepExoMir -- Streamlit Web Application for miRNA Target Prediction.

A modern, interactive web interface for predicting miRNA-target interactions
using the DeepExoMir deep learning model.

Usage:
    streamlit run app/app.py

Requirements:
    pip install streamlit plotly
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
import yaml

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.WARNING)

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "checkpoints" / "v8_fast"
    / "checkpoint_epoch024_val_auc_0.8292.pt"
)
DEFAULT_MODEL_CONFIG = PROJECT_ROOT / "configs" / "model_config_v8_pca256.yaml"
# Prefer multi-DB cache (superset), fall back to original PCA-256 cache
_MULTIDB_EMB = PROJECT_ROOT / "data" / "embeddings_cache_multidb_pca256"
_ORIG_EMB = PROJECT_ROOT / "data" / "embeddings_cache_pca256"
DEFAULT_EMBEDDINGS_DIR = _MULTIDB_EMB if _MULTIDB_EMB.exists() else _ORIG_EMB

# Pre-loaded miRNA database for quick lookup
MIRBASE_PATH = PROJECT_ROOT / "data" / "raw" / "mature.fa"


# ============================================================================
# Caching helpers
# ============================================================================

@st.cache_resource
def load_model(checkpoint_path: str, model_config_path: str):
    """Load the DeepExoMir model (cached across sessions)."""
    from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

    with open(model_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = DeepExoMirModelV8(config, precomputed_embeddings=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return model, device


@st.cache_resource
def load_embedding_stores(embeddings_dir: str):
    """Load pre-computed embedding stores (cached)."""
    from deepexomir.data.dataset import PerTokenEmbeddingStore, PooledEmbeddingStore

    emb_dir = Path(embeddings_dir)
    stores = {}

    if (emb_dir / "mirna_pertoken_metadata.pt").exists():
        stores["mirna_pertoken"] = PerTokenEmbeddingStore(emb_dir, "mirna")
    if (emb_dir / "target_pertoken_metadata.pt").exists():
        stores["target_pertoken"] = PerTokenEmbeddingStore(emb_dir, "target")
    if (emb_dir / "mirna_metadata.pt").exists():
        stores["mirna_pooled"] = PooledEmbeddingStore(emb_dir, "mirna")
    if (emb_dir / "target_metadata.pt").exists():
        stores["target_pooled"] = PooledEmbeddingStore(emb_dir, "target")

    return stores


@st.cache_data
def load_mirbase() -> dict[str, str]:
    """Load miRBase mature miRNA sequences."""
    if not MIRBASE_PATH.exists():
        return {}

    mirnas = {}
    current_name = None
    current_seq = []

    with open(MIRBASE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_name and current_seq:
                    mirnas[current_name] = "".join(current_seq)
                parts = line[1:].split()
                current_name = parts[0] if parts else None
                current_seq = []
            else:
                current_seq.append(line)

    if current_name and current_seq:
        mirnas[current_name] = "".join(current_seq)

    return {k: v for k, v in mirnas.items() if k.startswith("hsa-")}


def predict_single(
    model, device, stores, mirna_seq: str, target_seq: str,
) -> dict:
    """Run prediction for a single miRNA-target pair."""
    # Standardize sequences
    mirna_seq = mirna_seq.upper().replace("T", "U").strip()
    target_seq = target_seq.upper().replace("T", "U").strip()

    # Pad/truncate target to 50nt
    if len(target_seq) > 50:
        start = (len(target_seq) - 50) // 2
        target_seq = target_seq[start:start + 50]
    elif len(target_seq) < 50:
        pad_total = 50 - len(target_seq)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        target_seq = "N" * pad_left + target_seq + "N" * pad_right

    # Truncate miRNA to 30nt
    if len(mirna_seq) > 30:
        mirna_seq = mirna_seq[:30]

    # Look up embeddings
    kwargs = {"mirna_seqs": [mirna_seq], "target_seqs": [target_seq]}

    # Per-token embeddings
    if "mirna_pertoken" in stores:
        try:
            emb, mask = stores["mirna_pertoken"].lookup(mirna_seq)
            kwargs["mirna_pertoken_emb"] = emb.unsqueeze(0).to(device)
            kwargs["mirna_pertoken_mask"] = mask.unsqueeze(0).to(device)
        except KeyError:
            pass

    if "target_pertoken" in stores:
        try:
            emb, mask = stores["target_pertoken"].lookup(target_seq)
            kwargs["target_pertoken_emb"] = emb.unsqueeze(0).to(device)
            kwargs["target_pertoken_mask"] = mask.unsqueeze(0).to(device)
        except KeyError:
            pass

    # Pooled embeddings
    if "mirna_pooled" in stores:
        try:
            kwargs["mirna_pooled_emb"] = stores["mirna_pooled"].lookup(mirna_seq).unsqueeze(0).to(device)
        except KeyError:
            pass
    if "target_pooled" in stores:
        try:
            kwargs["target_pooled_emb"] = stores["target_pooled"].lookup(target_seq).unsqueeze(0).to(device)
        except KeyError:
            pass

    # Check if we have the required embeddings
    has_mirna_emb = "mirna_pertoken_emb" in kwargs
    has_target_emb = "target_pertoken_emb" in kwargs

    if not has_mirna_emb or not has_target_emb:
        return {
            "error": True,
            "message": "Sequence not found in pre-computed embedding cache. "
                       "Please use sequences from the training dataset, or run "
                       "the embedding precomputation script for new sequences.",
            "mirna_seq": mirna_seq,
            "target_seq": target_seq,
        }

    # Run model
    start = time.time()
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", enabled=(device == "cuda")):
            result = model(**kwargs)

    logits = result["logits"]
    probs = F.softmax(logits, dim=-1)
    pos_prob = probs[0, 1].item()
    elapsed = time.time() - start

    return {
        "error": False,
        "probability": pos_prob,
        "prediction": "INTERACTION" if pos_prob >= 0.41 else "NO INTERACTION",
        "confidence": max(pos_prob, 1 - pos_prob),
        "mirna_seq": mirna_seq,
        "target_seq": target_seq,
        "inference_time_ms": elapsed * 1000,
    }


def predict_batch(
    model, device, stores, pairs: list[tuple[str, str]],
) -> list[dict]:
    """Run prediction for multiple pairs."""
    results = []
    for mirna_seq, target_seq in pairs:
        result = predict_single(model, device, stores, mirna_seq, target_seq)
        results.append(result)
    return results


# ============================================================================
# Streamlit UI
# ============================================================================

def main():
    st.set_page_config(
        page_title="DeepExoMir",
        page_icon="🧬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .result-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .result-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-header">DeepExoMir</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Deep Learning miRNA Target Prediction Platform</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("Model Configuration")

        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value=str(DEFAULT_CHECKPOINT),
            help="Path to the trained model checkpoint (.pt file)",
        )
        model_config_path = st.text_input(
            "Model Config",
            value=str(DEFAULT_MODEL_CONFIG),
            help="Path to model architecture YAML config",
        )
        embeddings_dir = st.text_input(
            "Embeddings Directory",
            value=str(DEFAULT_EMBEDDINGS_DIR),
            help="Directory with pre-computed RiNALMo embeddings",
        )

        st.divider()
        st.header("Model Info")
        st.markdown("""
        - **Architecture**: HybridEncoder + MoE
        - **Backbone**: RiNALMo-giga (PCA-256)
        - **Best val AUC**: 0.8292
        - **Test AUC**: 0.8127
        - **Test AUPR**: 0.8359
        - **Parameters**: 21.2M trainable
        """)

        st.divider()
        st.markdown("*Built for XunLian Group*")

    # Load model and resources
    try:
        model, device = load_model(checkpoint_path, model_config_path)
        stores = load_embedding_stores(embeddings_dir)
        mirbase = load_mirbase()
        model_loaded = True
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model_loaded = False
        return

    # Show device info
    device_info = f"GPU: {torch.cuda.get_device_name()}" if device == "cuda" else "CPU"
    st.caption(f"Running on: {device_info} | miRBase: {len(mirbase):,} human miRNAs loaded")

    # Tabs
    tab1, tab2, tab3 = st.tabs([
        "Single Prediction",
        "Batch Prediction",
        "miRNA Explorer",
    ])

    # ---- Tab 1: Single Prediction ----
    with tab1:
        st.subheader("Predict miRNA-Target Interaction")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**miRNA Sequence**")
            # miRNA selector
            mirna_input_method = st.radio(
                "Input method", ["Select from miRBase", "Enter sequence"],
                horizontal=True, key="mirna_method",
            )

            if mirna_input_method == "Select from miRBase":
                mirna_names = sorted(mirbase.keys())
                # Default to hsa-miR-21-5p if available
                default_idx = 0
                if "hsa-miR-21-5p" in mirna_names:
                    default_idx = mirna_names.index("hsa-miR-21-5p")
                selected_mirna = st.selectbox(
                    "Select miRNA", mirna_names,
                    index=default_idx, key="mirna_select",
                )
                mirna_seq = mirbase.get(selected_mirna, "")
                st.code(mirna_seq, language=None)
            else:
                mirna_seq = st.text_input(
                    "Enter miRNA sequence (RNA: AUGC)",
                    value="UAGCUUAUCAGACUGAUGUUGA",
                    key="mirna_input",
                )

        with col2:
            st.markdown("**Target Site Sequence (50nt)**")
            target_seq = st.text_area(
                "Enter target 3'UTR site sequence (up to 50nt)",
                value="AUAAGCUAGAUAACCGAAAGUGCAAUCGAUUUGUACACUUCAAGCUGCUU",
                height=100,
                key="target_input",
            )

        if st.button("Predict", type="primary", use_container_width=True):
            if not mirna_seq or not target_seq:
                st.warning("Please enter both miRNA and target sequences.")
            else:
                with st.spinner("Running prediction..."):
                    result = predict_single(model, device, stores, mirna_seq, target_seq)

                if result.get("error"):
                    st.error(result["message"])
                else:
                    prob = result["probability"]
                    pred = result["prediction"]

                    # Display result
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Interaction Probability", f"{prob:.4f}")
                    with col_b:
                        st.metric("Prediction", pred)
                    with col_c:
                        st.metric(
                            "Confidence",
                            f"{result['confidence']:.1%}",
                        )

                    # Visual indicator
                    if pred == "INTERACTION":
                        st.success(
                            f"This miRNA-target pair is predicted to **interact** "
                            f"(probability: {prob:.4f}, threshold: 0.41)"
                        )
                    else:
                        st.info(
                            f"This miRNA-target pair is predicted to **not interact** "
                            f"(probability: {prob:.4f}, threshold: 0.41)"
                        )

                    # Probability bar
                    st.progress(prob, text=f"Interaction probability: {prob:.4f}")

                    st.caption(
                        f"Inference time: {result['inference_time_ms']:.1f}ms | "
                        f"Optimal threshold: 0.41"
                    )

    # ---- Tab 2: Batch Prediction ----
    with tab2:
        st.subheader("Batch Prediction")
        st.markdown(
            "Upload a CSV/TSV file with `mirna_seq` and `target_seq` columns, "
            "or paste sequences below."
        )

        batch_method = st.radio(
            "Input method",
            ["Upload file", "Paste sequences"],
            horizontal=True, key="batch_method",
        )

        if batch_method == "Upload file":
            uploaded = st.file_uploader(
                "Upload CSV/TSV", type=["csv", "tsv", "txt"],
                key="batch_upload",
            )
            if uploaded is not None:
                try:
                    sep = "\t" if uploaded.name.endswith((".tsv", ".txt")) else ","
                    batch_df = pd.read_csv(uploaded, sep=sep)
                    st.dataframe(batch_df.head(10))
                except Exception as e:
                    st.error(f"Failed to read file: {e}")
                    batch_df = None
            else:
                batch_df = None
        else:
            text_input = st.text_area(
                "Paste pairs (one per line: mirna_seq,target_seq)",
                value="UAGCUUAUCAGACUGAUGUUGA,AUAAGCUAGAUAACCGAAAGUGCAAUCGAUUUGUACACUUCAAGCUGCUU",
                height=150,
                key="batch_paste",
            )
            if text_input.strip():
                rows = []
                for line in text_input.strip().split("\n"):
                    parts = line.strip().split(",")
                    if len(parts) >= 2:
                        rows.append({"mirna_seq": parts[0].strip(), "target_seq": parts[1].strip()})
                batch_df = pd.DataFrame(rows)
            else:
                batch_df = None

        if batch_df is not None and st.button("Run Batch Prediction", type="primary"):
            if "mirna_seq" not in batch_df.columns or "target_seq" not in batch_df.columns:
                st.error("File must contain 'mirna_seq' and 'target_seq' columns.")
            else:
                pairs = list(zip(batch_df["mirna_seq"], batch_df["target_seq"]))

                progress = st.progress(0, text="Processing...")
                results = []
                for i, (m, t) in enumerate(pairs):
                    r = predict_single(model, device, stores, m, t)
                    results.append(r)
                    progress.progress((i + 1) / len(pairs), text=f"Processing {i+1}/{len(pairs)}...")

                progress.empty()

                # Build results DataFrame
                results_df = pd.DataFrame([
                    {
                        "mirna_seq": r["mirna_seq"],
                        "target_seq": r["target_seq"],
                        "probability": r.get("probability", None),
                        "prediction": r.get("prediction", r.get("message", "ERROR")),
                        "confidence": r.get("confidence", None),
                    }
                    for r in results
                ])

                st.dataframe(results_df, use_container_width=True)

                # Summary stats
                valid = results_df.dropna(subset=["probability"])
                if len(valid) > 0:
                    n_interact = (valid["prediction"] == "INTERACTION").sum()
                    st.info(
                        f"Results: {n_interact}/{len(valid)} pairs predicted as interacting "
                        f"({100*n_interact/len(valid):.1f}%)"
                    )

                # Download button
                csv_data = results_df.to_csv(index=False)
                st.download_button(
                    "Download Results (CSV)",
                    csv_data,
                    file_name="deepexomir_predictions.csv",
                    mime="text/csv",
                )

    # ---- Tab 3: miRNA Explorer ----
    with tab3:
        st.subheader("miRNA Database Explorer")
        st.markdown("Browse human miRNAs from miRBase v22.1")

        search = st.text_input("Search miRNA name", value="hsa-miR-21", key="mirna_search")

        if search:
            matches = {k: v for k, v in mirbase.items() if search.lower() in k.lower()}

            if matches:
                st.info(f"Found {len(matches)} matching miRNAs")

                # Display as table
                explorer_df = pd.DataFrame([
                    {"miRNA": name, "Sequence": seq, "Length": len(seq)}
                    for name, seq in sorted(matches.items())
                ])
                st.dataframe(explorer_df, use_container_width=True, height=400)
            else:
                st.warning("No miRNAs found matching your search.")


if __name__ == "__main__":
    main()
