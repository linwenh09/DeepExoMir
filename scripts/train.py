"""Train the DeepExoMir model.

Usage:
    # Lightweight encoder (fast baseline):
    python scripts/train.py

    # With pre-computed RiNALMo embeddings (recommended):
    python scripts/train.py --embeddings-dir data/embeddings_cache

    # Full options:
    python scripts/train.py \\
        --config configs/train_config.yaml \\
        --model-config configs/model_config.yaml \\
        --embeddings-dir data/embeddings_cache

Loads configuration files, creates datasets and dataloaders, initializes
the model, creates a Trainer, and runs the full training loop with early
stopping and checkpointing.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

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
    parser = argparse.ArgumentParser(
        description="Train the DeepExoMir miRNA-target interaction model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config.yaml"),
        help="Path to the training configuration YAML file.",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=Path("configs/model_config.yaml"),
        help="Path to the model architecture configuration YAML file.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override data directory from config.",
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing pre-computed RiNALMo embeddings "
            "(from scripts/precompute_embeddings.py).  When set, training "
            "uses backbone-quality embeddings without loading the 650M model."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Resume training from a checkpoint file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: auto-detect cuda/cpu).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def load_yaml(path: Path) -> dict:
    """Load a YAML configuration file."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def main() -> None:
    from deepexomir.data.dataset import create_dataloader
    from deepexomir.model.deepexomir_model import DeepExoMirModel
    from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8
    from deepexomir.training.trainer import Trainer

    args = parse_args()

    # ---- Set seed ----
    if args.seed is not None:
        set_seed(args.seed)

    # ---- Load configs ----
    print("DeepExoMir Model Training")
    print("=" * 50)

    train_config = load_yaml(args.config)
    model_config = load_yaml(args.model_config)

    print(f"Training config : {args.config}")
    print(f"Model config    : {args.model_config}")

    # ---- Resolve paths ----
    data_cfg = train_config.get("data", {})
    data_dir = args.data_dir or Path(data_cfg.get("data_dir", "data/processed"))
    cache_dir = data_cfg.get("embeddings_cache", None)
    batch_size = data_cfg.get("batch_size", 64)
    num_workers = data_cfg.get("num_workers", 0)
    embeddings_dir = args.embeddings_dir

    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"

    use_precomputed = embeddings_dir is not None

    print(f"Training data   : {train_path}")
    print(f"Validation data : {val_path}")
    print(f"Batch size      : {batch_size}")
    if use_precomputed:
        print(f"Embeddings dir  : {embeddings_dir}")
        print(f"Encoder mode    : pre-computed RiNALMo embeddings")
    else:
        print(f"Encoder mode    : lightweight trainable encoder")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device          : {device}")
    print()

    # ---- Create datasets and dataloaders ----
    print("Creating dataloaders ...")
    # Model v7: structural features pre-computed as .npy files.
    # The dataset auto-loads them if available. BP matrix still on GPU.
    struct_npy = data_dir / "train_structural_features.npy"
    has_precomputed_struct = struct_npy.exists()
    if has_precomputed_struct:
        print(f"  Pre-computed structural features: FOUND")
    else:
        print(f"  Pre-computed structural features: NOT FOUND")
        print(f"    Run: python scripts/precompute_structural_features.py")
        print(f"    Structural features will be zeros (reduces accuracy).")

    train_loader = create_dataloader(
        parquet_path=train_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        cache_dir=cache_dir,
        skip_structural=not has_precomputed_struct,  # v7: use if available
        skip_bp_matrix=True,  # model computes on GPU
        embeddings_dir=embeddings_dir,
        persistent_workers=num_workers > 0,
    )
    val_loader = create_dataloader(
        parquet_path=val_path,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        cache_dir=cache_dir,
        skip_structural=not has_precomputed_struct,  # v7: use if available
        skip_bp_matrix=True,  # model computes on GPU
        embeddings_dir=embeddings_dir,
        persistent_workers=num_workers > 0,
    )
    print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print()

    # ---- Initialize model ----
    print("Initializing model ...")
    # Determine model version (v7 = CrossAttention, v8 = HybridEncoder + MoE)
    model_version = train_config.get("training", {}).get("model_version", "v7")

    # Determine encoder mode:
    #  - backbone.load_backbone: true in config = live backbone (downloads 650M model)
    #  - --embeddings-dir (without live backbone) = pre-computed embeddings mode
    #  - neither = lightweight trainable encoder only (fast baseline)
    use_backbone = model_config.get("backbone", {}).get("load_backbone", False)

    if model_version == "v8":
        # v8: Check per-token target embeddings are available
        if embeddings_dir is not None:
            target_pertoken_meta = Path(embeddings_dir) / "target_pertoken_metadata.pt"
            if not target_pertoken_meta.exists():
                print()
                print("ERROR: Model v8 requires per-token target embeddings!")
                print(f"  Expected: {target_pertoken_meta}")
                print(f"  Run: python scripts/precompute_target_pertoken.py")
                print(f"  Then retry training.")
                sys.exit(1)
        model = DeepExoMirModelV8(
            config=model_config,
            load_backbone=use_backbone,
            precomputed_embeddings=use_precomputed,
            cache_dir=cache_dir,
        )
    else:
        model = DeepExoMirModel(
            config=model_config,
            load_backbone=use_backbone,
            precomputed_embeddings=use_precomputed,
            cache_dir=cache_dir,
        )

    mode_desc = "live backbone" if use_backbone else (
        "pre-computed embeddings" if use_precomputed else "lightweight encoder"
    )
    print(f"  Model version       : {model_version}")
    print(f"  Encoder mode        : {mode_desc}")
    print(f"  Backbone dim        : {model.backbone_embed_dim}")
    if model_version == "v8":
        print(f"  Hybrid encoder      : BiConvGate + CrossAttention (8 layers)")
        print(f"  Classifier          : MoE (4 experts, top-2)")
        print(f"  Multi-task heads    : {'ENABLED' if getattr(model, 'use_multitask', False) else 'disabled'}")
        print(f"  EvoAug augmentation : {'ENABLED' if getattr(model, 'use_augmentation', False) else 'disabled'}")
    else:
        print(f"  Backbone in cross   : {'BOTH sides' if use_backbone else 'miRNA only (pre-computed)' if use_precomputed else 'neither'}")
    print(f"  Contact Map CNN     : {'ENABLED' if model.use_contact_map else 'disabled'}")
    _struct_arr = getattr(train_loader.dataset, '_precomputed_struct', None)
    _n_sf = f"{_struct_arr.shape[1]} features" if _struct_arr is not None else "? features"
    print(f"  Structural features : {'pre-computed (' + _n_sf + ')' if has_precomputed_struct else 'zeros (disabled)'}")
    print(f"  Trainable parameters: {model.trainable_parameters():,}")
    print(f"  Total parameters    : {model.total_parameters():,}")
    print()

    # ---- Resume from checkpoint if specified ----
    if args.checkpoint is not None:
        if args.checkpoint.exists():
            print(f"Loading checkpoint: {args.checkpoint}")
            checkpoint_data = torch.load(
                args.checkpoint, map_location="cpu", weights_only=False
            )
            model.load_state_dict(checkpoint_data["model_state_dict"], strict=False)
            print("  Checkpoint loaded successfully.")
        else:
            logger.warning("Checkpoint file not found: %s", args.checkpoint)
    print()

    # ---- Create Trainer and run training ----
    print("Starting training ...")
    print("-" * 50)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=device,
    )

    results = trainer.fit()

    # ---- Print final results ----
    print()
    print("Training Complete")
    print("=" * 50)
    print(f"Total epochs trained : {results['total_epochs']}")
    print(f"Best epoch           : {results['best_epoch'] + 1}")
    print(f"Early stopping       : {results['stopped_early']}")
    print(f"Best checkpoint      : {results['best_checkpoint']}")
    print()

    best_metrics = results.get("best_val_metrics", {})
    if best_metrics:
        print("Best Validation Metrics:")
        print("-" * 40)
        for key, value in sorted(best_metrics.items()):
            if isinstance(value, float):
                print(f"  {key:25s}: {value:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
