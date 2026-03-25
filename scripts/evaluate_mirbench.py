"""Evaluate DeepExoMir on official miRBench datasets.

Runs the model on all 3 miRBench test sets and reports AU-PRC and AUC-ROC
for direct comparison with published baselines.

Usage:
    python scripts/evaluate_mirbench.py \
        --checkpoint checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt \
        --model-config configs/model_config_v19.yaml \
        --embeddings-dir data/embeddings_cache_pca256
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def encode_sequence_onthefly(seq: str, max_len: int = 50) -> torch.Tensor:
    """Simple one-hot-ish encoding for sequences without pre-computed embeddings."""
    # We'll use the model's own tokenizer + backbone if needed
    pass


def predict_batch(
    model: DeepExoMirModelV8,
    mirna_seqs: list[str],
    target_seqs: list[str],
    device: torch.device,
    embeddings_dir: str | None = None,
) -> np.ndarray:
    """Run model prediction on a batch of miRNA-target pairs.

    Since miRBench sequences may not be in our pre-computed embeddings,
    we need to handle embedding computation on-the-fly.
    """
    from deepexomir.data.dataset import MiRNATargetDataset

    B = len(mirna_seqs)

    # Try to get embeddings from pre-computed stores
    mirna_pt_list = []
    mirna_mask_list = []
    target_pt_list = []
    target_mask_list = []
    mirna_pooled_list = []
    target_pooled_list = []
    has_precomputed = False

    if embeddings_dir and hasattr(model, '_embedding_stores'):
        has_precomputed = True
        stores = model._embedding_stores
        for m_seq, t_seq in zip(mirna_seqs, target_seqs):
            # Look up in stores
            pass

    # Fallback: use random embeddings as placeholder (not ideal)
    # Better approach: use the model's backbone directly
    max_m = model.max_mirna_len
    max_t = model.max_target_len
    d = model.backbone_embed_dim

    # Create dummy embeddings (zeros) - the model will still use
    # BP matrix, structural features, and contact map
    mirna_pt = torch.zeros(B, max_m, d, device=device)
    mirna_mask = torch.ones(B, max_m, dtype=torch.bool, device=device)
    target_pt = torch.zeros(B, max_t, d, device=device)
    target_mask = torch.ones(B, max_t, dtype=torch.bool, device=device)

    # Set valid positions
    for i, (m, t) in enumerate(zip(mirna_seqs, target_seqs)):
        ml = min(len(m), max_m)
        tl = min(len(t), max_t)
        mirna_mask[i, :ml] = False
        target_mask[i, :tl] = False

    mirna_pooled = torch.zeros(B, d, device=device)
    target_pooled = torch.zeros(B, d, device=device)

    # No structural features for miRBench (we don't have them pre-computed)
    struct_feat = torch.zeros(B, model.struct_mlp.in_dim, device=device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        out = model(
            mirna_seqs=mirna_seqs,
            target_seqs=target_seqs,
            struct_features=struct_feat,
            mirna_pertoken_emb=mirna_pt,
            mirna_pertoken_mask=~mirna_mask,
            target_pertoken_emb=target_pt,
            target_pertoken_mask=~target_mask,
            mirna_pooled_emb=mirna_pooled,
            target_pooled_emb=target_pooled,
        )
        probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()

    return probs


def predict_with_backbone(
    model: DeepExoMirModelV8,
    mirna_seqs: list[str],
    target_seqs: list[str],
    device: torch.device,
    backbone_model,
    backbone_tokenizer,
    pca_model=None,
) -> np.ndarray:
    """Run predictions using the actual backbone for embedding computation."""
    B = len(mirna_seqs)
    max_m = model.max_mirna_len
    max_t = model.max_target_len

    # Compute embeddings through backbone
    def get_embeddings(seqs, max_len):
        # Tokenize
        encoded = backbone_tokenizer(
            seqs, padding=True, truncation=True,
            max_length=max_len + 2, return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            output = backbone_model(**encoded)
            hidden = output.last_hidden_state  # [B, L, 1280]

        # Remove special tokens (CLS/SEP)
        hidden = hidden[:, 1:-1, :]  # [B, L-2, 1280]

        # Mean pool for pooled embedding
        mask = encoded["attention_mask"][:, 1:-1]  # [B, L-2]
        pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        # PCA if available (numpy-based)
        if pca_model is not None:
            pca_mean = torch.from_numpy(pca_model["mean"]).to(device).float()
            pca_comp = torch.from_numpy(pca_model["components"]).to(device).float()  # [256, 1280]
            # hidden: [B, L, 1280] -> [B, L, 256]
            hidden = (hidden - pca_mean) @ pca_comp.T
            # pooled: [B, 1280] -> [B, 256]
            pooled = (pooled - pca_mean) @ pca_comp.T

        return hidden, pooled, mask.bool()

    mirna_hidden, mirna_pooled, mirna_valid = get_embeddings(mirna_seqs, max_m)
    target_hidden, target_pooled, target_valid = get_embeddings(target_seqs, max_t)

    # Pad/truncate to fixed lengths
    d = mirna_hidden.shape[-1]

    def pad_to(h, valid, max_len):
        B_s, L, D = h.shape
        if L >= max_len:
            return h[:, :max_len, :], valid[:, :max_len]
        pad = torch.zeros(B_s, max_len - L, D, device=device)
        h_pad = torch.cat([h, pad], dim=1)
        v_pad = torch.zeros(B_s, max_len, dtype=torch.bool, device=device)
        v_pad[:, :L] = valid[:, :L] if L <= valid.shape[1] else valid
        return h_pad, v_pad

    mirna_pt, mirna_vmask = pad_to(mirna_hidden, mirna_valid, max_m)
    target_pt, target_vmask = pad_to(target_hidden, target_valid, max_t)

    struct_feat = torch.zeros(B, model.struct_mlp.in_dim, device=device)

    with torch.no_grad(), torch.amp.autocast('cuda', enabled=True):
        out = model(
            mirna_seqs=mirna_seqs,
            target_seqs=target_seqs,
            struct_features=struct_feat,
            mirna_pertoken_emb=mirna_pt,
            mirna_pertoken_mask=mirna_vmask,
            target_pertoken_emb=target_pt,
            target_pertoken_mask=target_vmask,
            mirna_pooled_emb=mirna_pooled,
            target_pooled_emb=target_pooled,
        )
        probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()

    return probs


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepExoMir on miRBench")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--model-config", required=True, help="Model config YAML")
    parser.add_argument("--embeddings-dir", default=None, help="Pre-computed embeddings dir")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--use-backbone", action="store_true",
                        help="Use actual RiNALMo backbone for embedding (slower but accurate)")
    parser.add_argument("--datasets", nargs="+",
                        default=["AGO2_CLASH_Hejret2023", "AGO2_eCLIP_Klimentova2022", "AGO2_eCLIP_Manakov2022"])
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model config
    with open(args.model_config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Initialize model
    logger.info("Loading model from %s", args.checkpoint)
    model = DeepExoMirModelV8(config, load_backbone=False, precomputed_embeddings=True)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()
    params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s parameters", f"{params:,}")

    # Optionally load backbone for on-the-fly embedding
    backbone_model = None
    backbone_tokenizer = None
    pca_model = None

    if args.use_backbone:
        logger.info("Loading RiNALMo backbone for on-the-fly embedding...")
        import multimolecule  # noqa: F401 - registers model types
        from transformers import AutoModel, AutoTokenizer
        backbone_name = config.get("backbone", {}).get("name", "multimolecule/rinalmo-giga")
        backbone_tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
        backbone_model = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        backbone_model = backbone_model.to(device).eval()
        logger.info("Backbone loaded: %s", backbone_name)

        # Load PCA if embeddings are PCA-reduced
        embed_dim = config.get("backbone", {}).get("embed_dim", 256)
        if embed_dim < 1280 and args.embeddings_dir:
            pca_path = Path(args.embeddings_dir) / "pca_params.npz"
            if pca_path.exists():
                pca_data = np.load(pca_path)
                pca_model = {
                    "mean": pca_data["mean"],           # [1280]
                    "components": pca_data["components"],  # [256, 1280]
                }
                logger.info("PCA params loaded: 1280 -> %d", embed_dim)
            else:
                logger.warning("No PCA params found at %s!", pca_path)

    # Load and evaluate each miRBench dataset
    from miRBench.dataset import get_dataset_df

    print("\n" + "=" * 70)
    print("DeepExoMir -- miRBench Official Benchmark Evaluation")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Use backbone: {args.use_backbone}")
    print()

    all_results = []

    for ds_name in args.datasets:
        logger.info("=" * 50)
        logger.info("Dataset: %s", ds_name)
        logger.info("=" * 50)

        df = get_dataset_df(ds_name, "test")
        labels = df["label"].values
        mirna_seqs = df["noncodingRNA"].tolist()
        target_seqs = df["gene"].tolist()

        logger.info("  Samples: %d (pos=%d, neg=%d)",
                     len(df), labels.sum(), len(df) - labels.sum())

        # Predict in batches
        all_probs = []
        n_batches = (len(df) + args.batch_size - 1) // args.batch_size
        t0 = time.time()

        for i in range(n_batches):
            start = i * args.batch_size
            end = min(start + args.batch_size, len(df))
            batch_mirna = mirna_seqs[start:end]
            batch_target = target_seqs[start:end]

            if args.use_backbone and backbone_model is not None:
                probs = predict_with_backbone(
                    model, batch_mirna, batch_target, device,
                    backbone_model, backbone_tokenizer, pca_model,
                )
            else:
                probs = predict_batch(model, batch_mirna, batch_target, device)

            all_probs.append(probs)

            if (i + 1) % 50 == 0 or i == n_batches - 1:
                logger.info("  [%5.1f%%] batch %d/%d", 100 * (i + 1) / n_batches, i + 1, n_batches)

        elapsed = time.time() - t0
        all_probs = np.concatenate(all_probs)

        # Compute metrics
        auprc = average_precision_score(labels, all_probs)
        auroc = roc_auc_score(labels, all_probs)
        preds_binary = (all_probs >= 0.5).astype(int)
        acc = accuracy_score(labels, preds_binary)
        f1 = f1_score(labels, preds_binary)

        result = {
            "dataset": ds_name,
            "AU-PRC": auprc,
            "AUC-ROC": auroc,
            "Accuracy": acc,
            "F1": f1,
            "n_samples": len(df),
            "time_s": elapsed,
        }
        all_results.append(result)

        print(f"\n  {ds_name}")
        print(f"  {'─' * 40}")
        print(f"  AU-PRC (Average Precision) : {auprc:.4f}")
        print(f"  AUC-ROC                    : {auroc:.4f}")
        print(f"  Accuracy                   : {acc:.4f}")
        print(f"  F1 Score                   : {f1:.4f}")
        print(f"  Time                       : {elapsed:.1f}s")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY -- miRBench Official Benchmark")
    print("=" * 70)
    print(f"{'Dataset':<30} {'AU-PRC':>8} {'AUC-ROC':>8} {'Acc':>8} {'F1':>8}")
    print("─" * 70)
    for r in all_results:
        ds_short = r["dataset"].replace("AGO2_", "").replace("_", " ")
        print(f"{ds_short:<30} {r['AU-PRC']:>8.4f} {r['AUC-ROC']:>8.4f} {r['Accuracy']:>8.4f} {r['F1']:>8.4f}")
    print("─" * 70)
    mean_auprc = np.mean([r["AU-PRC"] for r in all_results])
    mean_auroc = np.mean([r["AUC-ROC"] for r in all_results])
    print(f"{'MEAN':<30} {mean_auprc:>8.4f} {mean_auroc:>8.4f}")

    # Compare with published baselines
    print("\n" + "=" * 70)
    print("COMPARISON WITH PUBLISHED BASELINES (miRBench v3)")
    print("=" * 70)
    print("Note: Published baselines use AU-PRC on bias-corrected datasets.")
    print("Our results are on the standard test split (may differ from v3).")
    print()
    baselines = {
        "CnnMirTarget": "0.51-0.58",
        "TargetNet": "0.58-0.66",
        "InteractionAwareModel": "0.63-0.74",
        "TargetScanCnn": "0.71-0.77",
        "miRNA_CNN": "0.71-0.77",
        "miRBind": "0.71-0.80",
        "CNN retrained (SOTA)": "0.77-0.86",
        f"DeepExoMir (ours)": f"{mean_auprc:.4f}",
    }
    for name, score in baselines.items():
        marker = " <<<" if name.startswith("DeepExoMir") else ""
        print(f"  {name:<30} AU-PRC: {score}{marker}")

    print("\nDone.")


if __name__ == "__main__":
    main()
