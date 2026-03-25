"""Evaluate a trained DeepExoMir model on the test set.

Usage:
    # Model v7:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --embeddings-dir data/embeddings_cache

    # Model v8 (PCA-256):
    python scripts/evaluate.py --checkpoint checkpoints/v8_fast/checkpoint_epoch024_val_auc_0.8292.pt --model-config configs/model_config_v8_pca256.yaml --embeddings-dir data/embeddings_cache_pca256
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DeepExoMir model.")
    parser.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to the model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/processed/test.parquet"),
        help="Path to the test Parquet file.",
    )
    parser.add_argument(
        "--model-config", type=Path, default=Path("configs/model_config.yaml"),
        help="Path to the model architecture config.",
    )
    parser.add_argument(
        "--embeddings-dir", type=Path, default=None,
        help="Directory containing pre-computed RiNALMo embeddings.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect).",
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Classification threshold. If None, finds optimal on the data.",
    )
    return parser.parse_args()


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the threshold that maximizes F1 score."""
    from sklearn.metrics import f1_score
    best_f1, best_thresh = 0.0, 0.5
    for thresh in np.arange(0.1, 0.9, 0.01):
        preds = (y_prob >= thresh).astype(np.int64)
        f1 = f1_score(y_true, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return float(best_thresh)


def main() -> None:
    import yaml
    from deepexomir.data.dataset import create_dataloader
    from deepexomir.training.evaluator import Evaluator

    args = parse_args()
    use_precomputed = args.embeddings_dir is not None

    print("DeepExoMir Model Evaluation")
    print("=" * 50)

    # Load model config
    if not args.model_config.exists():
        raise FileNotFoundError(f"Model config not found: {args.model_config}")
    with open(args.model_config, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    # Detect model version from config
    model_version = "v7"
    cls_cfg = model_config.get("classifier", {})
    if cls_cfg.get("type") == "moe":
        model_version = "v8"

    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print(f"Model version   : {model_version}")
    print(f"Checkpoint      : {args.checkpoint}")
    print(f"Test data       : {args.data}")
    if use_precomputed:
        print(f"Embeddings dir  : {args.embeddings_dir}")
    print()

    # Create dataloader (v8 needs structural features and bp_matrix)
    skip_struct = model_version == "v7"
    skip_bp = model_version == "v7"

    print("Creating test dataloader ...")
    test_loader = create_dataloader(
        parquet_path=args.data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        skip_structural=skip_struct,
        skip_bp_matrix=skip_bp,
        embeddings_dir=args.embeddings_dir,
    )
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Test batches: {len(test_loader)}")
    print()

    # Initialize model
    print("Initializing model ...")
    use_backbone = model_config.get("backbone", {}).get("load_backbone", False)

    if model_version == "v8":
        from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
        model = DeepExoMirModelV8(
            config=model_config,
            load_backbone=use_backbone,
            precomputed_embeddings=use_precomputed,
        )
    else:
        from deepexomir.model.deepexomir_model import DeepExoMirModel
        model = DeepExoMirModel(
            config=model_config,
            load_backbone=use_backbone,
            precomputed_embeddings=use_precomputed,
        )

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint_data = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
    ckpt_epoch = checkpoint_data.get("epoch", "?")
    ckpt_metric = checkpoint_data.get("val_auc", "?")
    print(f"  Checkpoint epoch : {ckpt_epoch}")
    print(f"  Checkpoint metric: {ckpt_metric}")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters       : {n_params:,}")
    print()

    model = model.to(device)
    model.eval()

    # Run evaluation
    print("Running evaluation ...")
    start_time = time.time()

    use_amp = device == "cuda"
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, (batch, labels) in enumerate(test_loader):
            # Move data to device
            if isinstance(batch, dict):
                batch = {
                    k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type=device, enabled=use_amp):
                kwargs = dict(
                    mirna_seqs=batch.get("mirna_seq"),
                    target_seqs=batch.get("target_seq"),
                    bp_matrix=batch.get("base_pairing_matrix"),
                    struct_features=batch.get("structural_features"),
                    mirna_pooled_emb=batch.get("mirna_pooled_emb"),
                    target_pooled_emb=batch.get("target_pooled_emb"),
                    mirna_pertoken_emb=batch.get("mirna_pertoken_emb"),
                    mirna_pertoken_mask=batch.get("mirna_pertoken_mask"),
                )
                if model_version == "v8":
                    kwargs["target_pertoken_emb"] = batch.get("target_pertoken_emb")
                    kwargs["target_pertoken_mask"] = batch.get("target_pertoken_mask")

                output = model(**kwargs)

            if isinstance(output, dict):
                logits = output["logits"]
            else:
                logits = output

            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_labels.append(labels.cpu().numpy())
            all_probs.append(probs)

            if (batch_idx + 1) % 50 == 0:
                pct = 100.0 * (batch_idx + 1) / len(test_loader)
                print(f"  [{pct:5.1f}%] batch {batch_idx + 1}/{len(test_loader)}")

    elapsed = time.time() - start_time
    print(f"\nEvaluation complete in {elapsed:.1f}s")
    print()

    # Compute metrics
    y_true = np.concatenate(all_labels)
    y_prob = np.concatenate(all_probs)

    # Default threshold metrics
    y_pred_default = (y_prob >= 0.5).astype(np.int64)
    metrics_default = Evaluator.compute_metrics(y_true, y_pred_default, y_prob)

    print("Test Set Metrics (threshold=0.5)")
    print("-" * 40)
    print(Evaluator.format_metrics(metrics_default))
    print()

    # Find optimal threshold
    optimal_thresh = find_optimal_threshold(y_true, y_prob)
    y_pred_optimal = (y_prob >= optimal_thresh).astype(np.int64)
    metrics_optimal = Evaluator.compute_metrics(y_true, y_pred_optimal, y_prob)

    print(f"Test Set Metrics (optimal threshold={optimal_thresh:.2f})")
    print("-" * 40)
    print(Evaluator.format_metrics(metrics_optimal))
    print()

    # Label distribution
    n_pos = (y_true == 1).sum()
    n_neg = (y_true == 0).sum()
    print(f"Label Distribution: pos={n_pos}, neg={n_neg}, ratio={n_pos/max(n_neg,1):.3f}")

    # Prediction distribution
    pred_pos = (y_pred_optimal == 1).sum()
    pred_neg = (y_pred_optimal == 0).sum()
    print(f"Prediction Dist  : pos={pred_pos}, neg={pred_neg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
