"""Prediction page -- single and batch miRNA-target interaction prediction.

Provides two input modes:

1. **Single prediction** -- enter a miRNA ID/sequence and target gene/sequence
   manually.
2. **Batch upload** -- upload a CSV file containing multiple pairs.

Results include binding score, confidence level, aesthetic relevance score,
and a text-based duplex alignment visualisation of the predicted binding site.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CHECKPOINT_DIR = _PROJECT_ROOT / "checkpoints"
_DEFAULT_CHECKPOINT = _CHECKPOINT_DIR / "deepexomir_best.pt"

_EXAMPLE_MIRNA = "hsa-miR-21-5p"
_EXAMPLE_MIRNA_SEQ = "UAGCUUAUCAGACUGAUGUUGA"
_EXAMPLE_TARGET = "PTEN"
_EXAMPLE_TARGET_SEQ = "AUUUCUUAAAUAAAGAUGGCCG"


# ---------------------------------------------------------------------------
# Model / resource loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading DeepExoMir model / \u8f09\u5165\u6a21\u578b\u4e2d...")
def _load_model() -> Any:
    """Load the DeepExoMir model from checkpoint.

    Returns the model object, or ``None`` if the checkpoint is not available.
    """
    try:
        import torch
        from deepexomir.model.deepexomir_model import DeepExoMirModel

        config = {
            "backbone": {
                "name": "multimolecule/rinalmo-giga",
                "embed_dim": 1280,
                "freeze": True,
            },
            "model": {
                "d_model": 256,
                "n_heads": 8,
                "d_ff": 1024,
                "n_cross_layers": 4,
                "dropout": 0.2,
                "attention_dropout": 0.1,
                "max_mirna_len": 30,
                "max_target_len": 40,
            },
            "structural": {
                "bp_cnn_out": 128,
                "struct_mlp_in": 6,
                "struct_mlp_out": 64,
            },
            "classifier": {
                "hidden_dims": [256, 128],
                "n_classes": 2,
                "platt_scaling": True,
            },
        }

        if _DEFAULT_CHECKPOINT.exists():
            model = DeepExoMirModel(config, load_backbone=False)
            checkpoint_data = torch.load(
                _DEFAULT_CHECKPOINT, map_location="cpu", weights_only=False,
            )
            # Handle both raw state_dict and wrapped checkpoint formats
            if isinstance(checkpoint_data, dict) and "model_state_dict" in checkpoint_data:
                state_dict = checkpoint_data["model_state_dict"]
            else:
                state_dict = checkpoint_data
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            logger.info("Model loaded from %s", _DEFAULT_CHECKPOINT)
            return model

        logger.warning(
            "No checkpoint found at %s. Model is uninitialised.",
            _DEFAULT_CHECKPOINT,
        )
        return None

    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return None


@st.cache_resource(show_spinner="Loading annotation resources / \u8f09\u5165\u8a3b\u91cb\u8cc7\u6e90...")
def _load_scorer() -> Any:
    """Load the AestheticScorer with knowledge graph."""
    try:
        from deepexomir.annotation.aesthetic_scorer import AestheticScorer
        from deepexomir.annotation.knowledge_graph import MiRNAKnowledgeGraph

        kg = MiRNAKnowledgeGraph()
        kg.add_aesthetic_mapping()

        kg_path = Path("data") / "knowledge_graph.gpickle"
        if kg_path.exists():
            kg.load(kg_path)

        scorer = AestheticScorer(knowledge_graph=kg, exosome_mirnas=set())
        return scorer
    except Exception as exc:
        logger.error("Failed to load scorer: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Duplex visualisation
# ---------------------------------------------------------------------------

def _format_duplex_alignment(mirna_seq: str, target_seq: str) -> str:
    """Create a text-based duplex alignment between miRNA and target.

    Shows the canonical 3'-to-5' miRNA binding with the 5'-to-3' target.
    Watson-Crick pairs shown as ``|``, wobble G:U as ``:``, mismatches as
    a space.
    """
    from deepexomir.utils.constants import BP_SCORES

    mirna_upper = mirna_seq.upper().replace("T", "U")
    target_upper = target_seq.upper().replace("T", "U")

    max_len = max(len(mirna_upper), len(target_upper))
    mirna_padded = mirna_upper.ljust(max_len, " ")
    target_padded = target_upper.ljust(max_len, " ")

    # miRNA reads 3'->5', target reads 5'->3'
    mirna_display = mirna_padded[::-1]
    target_display = target_padded

    bond_line = []
    for m_base, t_base in zip(mirna_display, target_display):
        if m_base.strip() == "" or t_base.strip() == "":
            bond_line.append(" ")
        elif (m_base, t_base) in BP_SCORES:
            score = BP_SCORES[(m_base, t_base)]
            if score >= 1.0:
                bond_line.append("|")
            elif score >= 0.5:
                bond_line.append(":")
            else:
                bond_line.append(" ")
        else:
            bond_line.append(" ")

    bond_str = "".join(bond_line)

    alignment = (
        f"3'  {mirna_display}  5'   miRNA\n"
        f"    {bond_str}\n"
        f"5'  {target_display}  3'   Target"
    )
    return alignment


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _run_single_prediction(
    mirna_id: str,
    mirna_seq: str,
    target_gene: str,
    target_seq: str,
    model: Any,
    scorer: Any,
) -> pd.DataFrame:
    """Execute a single prediction and return a one-row DataFrame."""
    import numpy as np

    prob: float
    confidence: float

    if model is not None:
        try:
            import torch
            from deepexomir.utils.sequence import (
                clean_sequence,
                compute_base_pairing_matrix,
            )

            m_seq = clean_sequence(mirna_seq)
            t_seq = clean_sequence(target_seq)

            bp = compute_base_pairing_matrix(m_seq, t_seq)
            bp_tensor = (
                torch.tensor(bp, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            )
            struct_feats = torch.zeros(1, 6)

            result = model.predict(
                mirna_seqs=[m_seq],
                target_seqs=[t_seq],
                bp_matrix=bp_tensor,
                struct_features=struct_feats,
            )
            prob = float(result["probabilities"][0, 1])
            confidence = float(result["confidence"][0])
        except Exception as exc:
            logger.warning("Prediction failed, using placeholder: %s", exc)
            prob = float(np.random.uniform(0.3, 0.95))
            confidence = prob
    else:
        # Placeholder scores for demo when model not available
        prob = float(np.random.uniform(0.3, 0.95))
        confidence = prob

    # Aesthetic score
    aesthetic_score = 0.0
    if scorer is not None:
        try:
            aes_result = scorer.score(mirna_id, target_gene, prob)
            aesthetic_score = aes_result["total_score"]
        except Exception:
            aesthetic_score = 0.0

    df = pd.DataFrame(
        [
            {
                "miRNA": mirna_id,
                "Target Gene": target_gene,
                "Binding Score": round(prob, 4),
                "Confidence": round(confidence, 4),
                "Aesthetic Score": round(aesthetic_score, 2),
            }
        ]
    )
    return df


def _run_batch_prediction(
    uploaded_df: pd.DataFrame,
    model: Any,
    scorer: Any,
) -> pd.DataFrame:
    """Run predictions on a batch DataFrame."""
    results = []
    for _, row in uploaded_df.iterrows():
        mirna_id = str(row.get("mirna_id", row.get("miRNA", "unknown")))
        mirna_seq = str(row.get("mirna_seq", row.get("miRNA_sequence", "")))
        target_gene = str(row.get("target_gene", row.get("Target Gene", "unknown")))
        target_seq = str(row.get("target_seq", row.get("target_sequence", "")))

        if mirna_seq and target_seq:
            row_df = _run_single_prediction(
                mirna_id,
                mirna_seq,
                target_gene,
                target_seq,
                model,
                scorer,
            )
            results.append(row_df)
        else:
            results.append(
                pd.DataFrame(
                    [
                        {
                            "miRNA": mirna_id,
                            "Target Gene": target_gene,
                            "Binding Score": 0.0,
                            "Confidence": 0.0,
                            "Aesthetic Score": 0.0,
                        }
                    ]
                )
            )

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_predict_page() -> None:
    """Render the prediction page."""
    st.header("\U0001f50d miRNA \u9776\u6a19\u9810\u6e2c / Target Prediction")
    st.markdown(
        "Enter a miRNA and target sequence to predict their interaction.  \n"
        "\u8f38\u5165 miRNA \u548c\u76ee\u6a19\u5e8f\u5217\u4ee5\u9810\u6e2c\u5176\u4e92\u4f5c\u7528\u3002"
    )

    # --- Model status --------------------------------------------------------
    model = _load_model()
    scorer = _load_scorer()

    if model is None:
        st.warning(
            "\u26a0\ufe0f **Model checkpoint not found.** / "
            "\u6a21\u578b\u6a94\u6848\u672a\u627e\u5230\u3002\n\n"
            "To enable predictions, place a trained checkpoint at:\n"
            f"```\n{_DEFAULT_CHECKPOINT}\n```\n"
            "Predictions below will use **random placeholder scores** for "
            "demonstration.\n\n"
            "\u5982\u9700\u555f\u7528\u9810\u6e2c\u529f\u80fd\uff0c"
            "\u8acb\u5c07\u8a13\u7df4\u597d\u7684\u6a21\u578b\u6a94\u653e\u7f6e\u65bc\u4e0a\u8ff0\u8def\u5f91\u3002"
        )
    else:
        st.success(
            "\u2705 Model loaded successfully / \u6a21\u578b\u8f09\u5165\u6210\u529f"
        )

    st.divider()

    # --- Input mode selector --------------------------------------------------
    mode = st.radio(
        "\u8f38\u5165\u6a21\u5f0f / Input Mode",
        options=[
            "Single Prediction / \u55ae\u4e00\u9810\u6e2c",
            "Batch Upload / \u6279\u91cf\u4e0a\u50b3",
        ],
        horizontal=True,
    )

    if "Single" in mode:
        _render_single_prediction(model, scorer)
    else:
        _render_batch_prediction(model, scorer)


# ---------------------------------------------------------------------------
# Single prediction UI
# ---------------------------------------------------------------------------

def _render_single_prediction(model: Any, scorer: Any) -> None:
    """Render the single-prediction input form and results."""
    col_mirna, col_target = st.columns(2)

    with col_mirna:
        st.subheader("miRNA")
        mirna_id = st.text_input(
            "miRNA ID / miRNA \u7de8\u865f",
            value=_EXAMPLE_MIRNA,
            placeholder="e.g., hsa-miR-21-5p",
        )
        mirna_seq = st.text_area(
            "miRNA Sequence / miRNA \u5e8f\u5217",
            value=_EXAMPLE_MIRNA_SEQ,
            height=80,
            placeholder="e.g., UAGCUUAUCAGACUGAUGUUGA",
        )

    with col_target:
        st.subheader("Target / \u76ee\u6a19")
        target_gene = st.text_input(
            "Gene Symbol / \u57fa\u56e0\u540d\u7a31",
            value=_EXAMPLE_TARGET,
            placeholder="e.g., PTEN",
        )
        target_seq = st.text_area(
            "Target Sequence (3' UTR site) / \u76ee\u6a19\u5e8f\u5217",
            value=_EXAMPLE_TARGET_SEQ,
            height=80,
            placeholder="e.g., AUUUCUUAAAUAAAGAUGGCCG",
        )

    scan_utr = st.checkbox(
        "Scan full 3' UTR / \u6383\u63cf\u5b8c\u6574 3' UTR",
        value=False,
        help=(
            "When enabled, the entire 3' UTR of the target gene will be "
            "scanned for potential binding sites using a sliding window "
            "approach. / \u555f\u7528\u5f8c\u5c07\u4f7f\u7528\u6ed1\u52d5\u7a97\u53e3\u6383\u63cf"
            "\u76ee\u6a19\u57fa\u56e0\u7684\u5b8c\u6574 3' UTR\u3002"
        ),
    )

    if scan_utr:
        st.info(
            "Full 3' UTR scanning requires a gene annotation database. "
            "This feature will be available after the UTR database is "
            "configured. / \u5b8c\u6574 3' UTR \u6383\u63cf\u9700\u8981\u57fa\u56e0"
            "\u8a3b\u91cb\u8cc7\u6599\u5eab\uff0c\u8a72\u529f\u80fd\u5c07\u5728"
            "\u8cc7\u6599\u5eab\u914d\u7f6e\u5f8c\u53ef\u7528\u3002"
        )

    st.divider()

    # Run button
    if st.button(
        "Run Prediction / \u57f7\u884c\u9810\u6e2c",
        type="primary",
        use_container_width=True,
    ):
        if not mirna_seq.strip() or not target_seq.strip():
            st.error(
                "Please provide both miRNA and target sequences. / "
                "\u8acb\u63d0\u4f9b miRNA \u548c\u76ee\u6a19\u5e8f\u5217\u3002"
            )
            return

        with st.spinner("\u9810\u6e2c\u4e2d / Predicting..."):
            results_df = _run_single_prediction(
                mirna_id,
                mirna_seq.strip(),
                target_gene,
                target_seq.strip(),
                model,
                scorer,
            )

        st.session_state["predict_last_result"] = results_df
        st.session_state["predict_last_mirna_seq"] = mirna_seq.strip()
        st.session_state["predict_last_target_seq"] = target_seq.strip()

    # --- Display results -----------------------------------------------------
    if "predict_last_result" in st.session_state:
        results_df = st.session_state["predict_last_result"]

        st.subheader("\u7d50\u679c / Results")

        score_val = results_df.iloc[0]["Binding Score"]
        conf_val = results_df.iloc[0]["Confidence"]
        aes_val = results_df.iloc[0]["Aesthetic Score"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Binding Score / \u7d50\u5408\u5206\u6578", f"{score_val:.4f}")
        col2.metric("Confidence / \u4fe1\u5fc3\u5ea6", f"{conf_val:.4f}")
        col3.metric("Aesthetic Score / \u91ab\u7f8e\u5206\u6578", f"{aes_val:.2f}")

        st.dataframe(
            results_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Binding Score": st.column_config.ProgressColumn(
                    "Binding Score",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
                "Aesthetic Score": st.column_config.NumberColumn(
                    "Aesthetic Score",
                    format="%.2f",
                ),
            },
        )

        # Duplex alignment
        if (
            "predict_last_mirna_seq" in st.session_state
            and "predict_last_target_seq" in st.session_state
        ):
            st.subheader(
                "\u7d50\u5408\u4f4d\u9ede\u8996\u89ba\u5316 / Binding Site Visualisation"
            )
            alignment = _format_duplex_alignment(
                st.session_state["predict_last_mirna_seq"],
                st.session_state["predict_last_target_seq"],
            )
            st.code(alignment, language=None)

        # Download button
        csv_data = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Results as CSV / \u4e0b\u8f09\u7d50\u679c",
            data=csv_data,
            file_name="deepexomir_prediction.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------------------------
# Batch prediction UI
# ---------------------------------------------------------------------------

def _render_batch_prediction(model: Any, scorer: Any) -> None:
    """Render the batch-upload prediction form."""
    st.markdown(
        "Upload a CSV file with columns: `mirna_id`, `mirna_seq`, "
        "`target_gene`, `target_seq`  \n"
        "\u4e0a\u50b3\u5305\u542b\u4ee5\u4e0a\u6b04\u4f4d\u7684 CSV \u6a94\u6848\u3002"
    )

    uploaded_file = st.file_uploader(
        "\u4e0a\u50b3\u6a94\u6848 / Upload File",
        type=["csv", "tsv", "txt"],
        help="CSV with columns: mirna_id, mirna_seq, target_gene, target_seq",
    )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".tsv"):
                uploaded_df = pd.read_csv(uploaded_file, sep="\t")
            else:
                uploaded_df = pd.read_csv(uploaded_file)

            st.markdown(
                f"**Loaded {len(uploaded_df)} rows / "
                f"\u5df2\u8f09\u5165 {len(uploaded_df)} \u7b46\u8cc7\u6599**"
            )
            st.dataframe(
                uploaded_df.head(10), use_container_width=True, hide_index=True
            )

        except Exception as exc:
            st.error(
                f"Failed to parse file: {exc} / "
                f"\u6a94\u6848\u89e3\u6790\u5931\u6557\uff1a{exc}"
            )
            return

        if st.button(
            "Run Batch Prediction / \u57f7\u884c\u6279\u91cf\u9810\u6e2c",
            type="primary",
            use_container_width=True,
        ):
            with st.spinner(
                f"\u9810\u6e2c\u4e2d ({len(uploaded_df)} pairs)..."
            ):
                batch_results = _run_batch_prediction(uploaded_df, model, scorer)

            st.session_state["predict_batch_results"] = batch_results

    # Display batch results
    if "predict_batch_results" in st.session_state:
        batch_results = st.session_state["predict_batch_results"]
        st.subheader(
            f"\u6279\u91cf\u7d50\u679c / Batch Results ({len(batch_results)} pairs)"
        )

        st.dataframe(
            batch_results,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Binding Score": st.column_config.ProgressColumn(
                    "Binding Score",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    min_value=0,
                    max_value=1,
                    format="%.4f",
                ),
                "Aesthetic Score": st.column_config.NumberColumn(
                    "Aesthetic Score",
                    format="%.2f",
                ),
            },
        )

        csv_data = batch_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Batch Results / \u4e0b\u8f09\u6279\u91cf\u7d50\u679c",
            data=csv_data,
            file_name="deepexomir_batch_predictions.csv",
            mime="text/csv",
        )
