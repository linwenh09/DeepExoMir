"""Evaluate an ensemble of DeepExoMir models.

Loads multiple checkpoints, averages their prediction probabilities,
and computes metrics on the test set.

Usage:
    python scripts/evaluate_ensemble.py \
        --checkpoint-dirs checkpoints/ensemble/seed42 checkpoints/ensemble/seed123 checkpoints/ensemble/seed456 \
        --model-config configs/model_config_v8_pca256.yaml \
        --embeddings-dir data/embeddings_cache_pca256
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.amp import autocast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def find_best_checkpoint(checkpoint_dir: Path) -> Path:
    """Find the best checkpoint in a directory by val_auc in filename."""
    pts = list(checkpoint_dir.glob("checkpoint_epoch*_val_auc_*.pt"))
    if not pts:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    best = max(pts, key=lambda p: float(p.stem.split("val_auc_")[1]))
    return best


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepExoMir ensemble")
    parser.add_argument("--checkpoint-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=Path("data/processed/test.parquet"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/model_config_v8_pca256.yaml"))
    parser.add_argument("--embeddings-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_config = load_yaml(args.model_config)

    from deepexomir.data.dataset import create_dataloader
    from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

    # Create test dataloader (v8 needs structural features and bp_matrix)
    test_loader = create_dataloader(
        parquet_path=args.data,
        batch_size=args.batch_size, shuffle=False, num_workers=0,
        skip_structural=False, skip_bp_matrix=False,
        embeddings_dir=args.embeddings_dir, persistent_workers=False,
    )
    n_samples = len(test_loader.dataset)
    logger.info(f"Test samples: {n_samples:,}")

    # Collect predictions from each model
    all_probs = []
    all_labels = None

    for ckpt_dir in args.checkpoint_dirs:
        ckpt_path = find_best_checkpoint(ckpt_dir)
        logger.info(f"Loading: {ckpt_path.name}")

        model = DeepExoMirModelV8(model_config)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
                if all_labels is None:
                    labels_list.append(labels.cpu().numpy())

            all_probs.append(np.concatenate(probs_list))
            if all_labels is None:
                all_labels = np.concatenate(labels_list)

        del model
        torch.cuda.empty_cache()

    # Average predictions
    probs_array = np.stack(all_probs, axis=0)  # [n_models, n_samples]
    avg_probs = probs_array.mean(axis=0)

    # Compute metrics
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        accuracy_score, f1_score, matthews_corrcoef,
        precision_score, recall_score,
    )

    auc = roc_auc_score(all_labels, avg_probs)
    aupr = average_precision_score(all_labels, avg_probs)

    # Find optimal threshold
    best_f1 = 0
    best_t = 0.5
    for t in np.arange(0.3, 0.7, 0.01):
        preds = (avg_probs >= t).astype(int)
        f1 = f1_score(all_labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    preds_default = (avg_probs >= 0.5).astype(int)
    preds_optimal = (avg_probs >= best_t).astype(int)

    print("\n" + "=" * 60)
    print("Ensemble Evaluation Results")
    print("=" * 60)
    print(f"Models: {len(args.checkpoint_dirs)}")
    print(f"Test samples: {n_samples:,}")
    print()
    print("Threshold = 0.5:")
    print(f"  AUC-ROC:  {auc:.4f}")
    print(f"  AUC-PR:   {aupr:.4f}")
    print(f"  Accuracy: {accuracy_score(all_labels, preds_default):.4f}")
    print(f"  F1:       {f1_score(all_labels, preds_default):.4f}")
    print(f"  MCC:      {matthews_corrcoef(all_labels, preds_default):.4f}")
    print()
    print(f"Optimal threshold = {best_t:.2f}:")
    print(f"  F1:       {best_f1:.4f}")
    print(f"  MCC:      {matthews_corrcoef(all_labels, preds_optimal):.4f}")
    print()

    # Individual model AUCs
    print("Individual model AUCs:")
    for i, (ckpt_dir, probs) in enumerate(zip(args.checkpoint_dirs, all_probs)):
        individual_auc = roc_auc_score(all_labels, probs)
        print(f"  {ckpt_dir.name}: {individual_auc:.4f}")
    print(f"  Ensemble:  {auc:.4f}")


if __name__ == "__main__":
    main()
