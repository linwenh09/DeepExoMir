"""Evaluate heterogeneous ensemble of DeepExoMir models with different configs.

Loads multiple checkpoints with their own model configs and feature versions,
averages predictions.

Usage:
    python scripts/evaluate_hetero_ensemble.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    accuracy_score, f1_score, matthews_corrcoef,
)
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_predictions(checkpoint_path, model_config_path, data_path, embeddings_dir,
                    device, feature_version=None, batch_size=256):
    """Get predictions from a single model with correct feature version."""
    from deepexomir.data.dataset import create_dataloader
    from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

    model_config = load_yaml(model_config_path)

    test_loader = create_dataloader(
        parquet_path=data_path,
        batch_size=batch_size, shuffle=False, num_workers=0,
        skip_structural=False, skip_bp_matrix=False,
        embeddings_dir=embeddings_dir, persistent_workers=False,
        feature_version=feature_version,
    )

    model = DeepExoMirModelV8(model_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    probs_list = []
    labels_list = []

    with torch.no_grad():
        for batch_data, labels in test_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch_data.items()}

            with autocast(device_type="cuda", enabled=(device == "cuda")):
                kwargs = dict(
                    mirna_seqs=batch.get("mirna_seq"),
                    target_seqs=batch.get("target_seq"),
                    bp_matrix=batch.get("base_pairing_matrix"),
                    struct_features=batch.get("structural_features"),
                    mirna_pooled_emb=batch.get("mirna_pooled_emb"),
                    target_pooled_emb=batch.get("target_pooled_emb"),
                    mirna_pertoken_emb=batch.get("mirna_pertoken_emb"),
                    mirna_pertoken_mask=batch.get("mirna_pertoken_mask"),
                    target_pertoken_emb=batch.get("target_pertoken_emb"),
                    target_pertoken_mask=batch.get("target_pertoken_mask"),
                )
                outputs = model(**kwargs)

            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            prob = F.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            probs_list.append(prob)
            labels_list.append(labels.cpu().numpy())

    del model
    torch.cuda.empty_cache()

    return np.concatenate(probs_list), np.concatenate(labels_list)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_path = Path("data/processed/test.parquet")
    embeddings_dir = Path("data/embeddings_cache_pca256")

    # Define ensemble members with CORRECT feature versions
    models = [
        {
            "name": "v19 (33 feat, InteractionPool)",
            "checkpoint": "checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt",
            "model_config": "configs/model_config_v19.yaml",
            "feature_version": "v18",
            "val_auc": 0.8521,
        },
        {
            "name": "v18 (33 feat, miRNA struct)",
            "checkpoint": "checkpoints/v18/checkpoint_epoch024_val_auc_0.8500.pt",
            "model_config": "configs/model_config_v18.yaml",
            "feature_version": "v18",
            "val_auc": 0.8500,
        },
        {
            "name": "v16c (26 feat, +context)",
            "checkpoint": "checkpoints/v16c/checkpoint_epoch022_val_auc_0.8496.pt",
            "model_config": "configs/model_config_v16c.yaml",
            "feature_version": "v16c",
            "val_auc": 0.8496,
        },
        {
            "name": "v14-alt-2L (31 feat)",
            "checkpoint": "checkpoints/v14alt2L/checkpoint_epoch027_val_auc_0.8503.pt",
            "model_config": "configs/model_config_v14alt2L.yaml",
            "feature_version": "v14alt2L",
            "val_auc": 0.8503,
        },
        {
            "name": "v16a (23 feat, pruned)",
            "checkpoint": "checkpoints/v16a/checkpoint_epoch027_val_auc_0.8502.pt",
            "model_config": "configs/model_config_v16a.yaml",
            "feature_version": "v16a",
            "val_auc": 0.8502,
        },
        {
            "name": "v22 (33 feat, DuplexGAT)",
            "checkpoint": "checkpoints/v22/checkpoint_epoch027_val_auc_0.8507.pt",
            "model_config": "configs/model_config_v22.yaml",
            "feature_version": "v18",
            "val_auc": 0.8507,
        },
    ]

    all_probs = []
    labels = None

    for m in models:
        logger.info(f"Loading {m['name']} (features: {m['feature_version']}) ...")
        probs, lbls = get_predictions(
            Path(m["checkpoint"]), Path(m["model_config"]),
            data_path, embeddings_dir, device,
            feature_version=m["feature_version"],
        )
        all_probs.append(probs)
        if labels is None:
            labels = lbls
        auc_i = roc_auc_score(labels, probs)
        aupr_i = average_precision_score(labels, probs)
        logger.info(f"  {m['name']}: test_AUC={auc_i:.4f}, test_AUPR={aupr_i:.4f}")

    # Average predictions
    probs_array = np.stack(all_probs, axis=0)
    avg_probs = probs_array.mean(axis=0)

    # Also try weighted average (by val_AUC)
    val_aucs = np.array([m["val_auc"] for m in models])
    weights = val_aucs / val_aucs.sum()
    weighted_probs = (probs_array * weights[:, None]).sum(axis=0)

    # Evaluate different ensemble strategies
    for name, probs in [("Simple Average", avg_probs), ("Weighted Average", weighted_probs)]:
        auc = roc_auc_score(labels, probs)
        aupr = average_precision_score(labels, probs)

        best_f1, best_t = 0, 0.5
        for t in np.arange(0.3, 0.7, 0.01):
            preds = (probs >= t).astype(int)
            f1 = f1_score(labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        preds_default = (probs >= 0.5).astype(int)
        preds_optimal = (probs >= best_t).astype(int)

        print(f"\n{'='*60}")
        print(f"Ensemble: {name}")
        print(f"{'='*60}")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"  AUC-PR:   {aupr:.4f}")
        print(f"  Accuracy: {accuracy_score(labels, preds_default):.4f}")
        print(f"  F1 (t=0.5): {f1_score(labels, preds_default):.4f}")
        print(f"  MCC (t=0.5): {matthews_corrcoef(labels, preds_default):.4f}")
        print(f"  Optimal threshold: {best_t:.2f}")
        print(f"  F1 (optimal): {best_f1:.4f}")
        print(f"  MCC (optimal): {matthews_corrcoef(labels, preds_optimal):.4f}")

    # Also try pairwise ensembles
    print(f"\n{'='*60}")
    print("Pairwise Ensembles")
    print(f"{'='*60}")
    names = [m["name"] for m in models]
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            pair_probs = (all_probs[i] + all_probs[j]) / 2
            auc = roc_auc_score(labels, pair_probs)
            aupr = average_precision_score(labels, pair_probs)
            print(f"  {names[i]} + {names[j]}: AUC={auc:.4f}, AUPR={aupr:.4f}")


if __name__ == "__main__":
    main()
