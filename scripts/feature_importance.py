"""Analyze feature importance for DeepExoMir structural features.

Uses permutation importance on the best v14alt2L model checkpoint.
For each feature, shuffles its values and measures AUC drop.
Larger drop = more important feature.

Usage:
    python scripts/feature_importance.py
"""
import sys
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

# Feature names for v14alt2L (31 features)
FEATURE_NAMES = [
    # v7-v10 (12)
    "duplex_mfe", "mirna_mfe", "target_mfe", "accessibility",
    "gc_content", "seed_match_type", "au_content", "seed_duplex_mfe",
    "plfold_seed_acc", "plfold_site_acc", "supp_3prime", "local_au_flank",
    # v11 (8)
    "seed_pair_stab", "comp_3prime", "central_pair", "mfe_ratio",
    "wobble_count", "longest_contig", "mismatch_count", "seed_gc",
    # v13 (6)
    "dG_open", "dG_total", "ensemble_dG", "acc_5nt_up", "acc_10nt_up", "acc_15nt_up",
    # v14 (5)
    "phylop_mean", "phylop_max", "phylop_seed_mean", "site_in_3utr", "site_in_cds",
]


def batch_to_model_kwargs(batch, device):
    """Convert dataset batch dict to model forward() keyword arguments."""
    kwargs = {}

    # Map dataset keys to model forward parameter names
    key_map = {
        "structural_features": "struct_features",
        "bp_matrix": "bp_matrix",
        "mirna_emb": "mirna_emb",
        "target_emb": "target_emb",
        "mirna_pooled_emb": "mirna_pooled_emb",
        "target_pooled_emb": "target_pooled_emb",
        "mirna_pertoken_emb": "mirna_pertoken_emb",
        "mirna_pertoken_mask": "mirna_pertoken_mask",
        "target_pertoken_emb": "target_pertoken_emb",
        "target_pertoken_mask": "target_pertoken_mask",
        "mirna_mask": "mirna_mask",
        "target_mask": "target_mask",
    }

    for ds_key, model_key in key_map.items():
        if ds_key in batch and batch[ds_key] is not None:
            v = batch[ds_key]
            if torch.is_tensor(v):
                kwargs[model_key] = v.to(device)
            else:
                kwargs[model_key] = v

    return kwargs


def main():
    import yaml
    from deepexomir.data.dataset import MiRNATargetDataset
    from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model config
    config_path = Path("configs/model_config_v14alt2L.yaml")
    with open(config_path, encoding="utf-8") as f:
        model_cfg = yaml.safe_load(f)

    # Load validation dataset (smaller than test, faster)
    dataset = MiRNATargetDataset(
        parquet_path=Path("data/processed/val.parquet"),
        embeddings_dir=Path("data/embeddings_cache_pca256"),
    )

    # Use a subset for speed (50K samples)
    n_sub = min(50000, len(dataset))
    indices = np.random.RandomState(42).choice(len(dataset), n_sub, replace=False)

    print(f"Using {n_sub} validation samples for permutation importance")

    # Build model
    model = DeepExoMirModelV8(model_cfg)
    ckpt_path = Path("checkpoints/v14alt2L/checkpoint_epoch027_val_auc_0.8503.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.to(device)
    model.eval()

    # Collect batch data
    from torch.utils.data import DataLoader, Subset

    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=512, shuffle=False, num_workers=0)

    all_labels = []
    all_struct_feats = []
    all_batches = []  # store raw batch dicts (on CPU)

    print("Collecting data...")
    for batch in loader:
        all_labels.append(batch["label"].numpy())
        if "structural_features" in batch:
            all_struct_feats.append(batch["structural_features"].numpy())
        # Keep batch on CPU, move to GPU in forward calls
        all_batches.append(batch)

    labels = np.concatenate(all_labels)
    struct_arr = np.concatenate(all_struct_feats) if all_struct_feats else None

    if struct_arr is None:
        print("ERROR: No structural features found!")
        return

    n_features = min(struct_arr.shape[1], len(FEATURE_NAMES))
    print(f"Structural features shape: {struct_arr.shape}")
    print(f"Analyzing {n_features} features")

    # Helper: run model on all batches and return probs
    def get_probs(batches):
        all_probs = []
        with torch.no_grad():
            for batch in batches:
                kwargs = batch_to_model_kwargs(batch, device)
                out = model(**kwargs)
                probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs)

    # Baseline AUC
    print("\nComputing baseline AUC...")
    baseline_probs = get_probs(all_batches)
    baseline_auc = roc_auc_score(labels, baseline_probs)
    print(f"Baseline AUC: {baseline_auc:.6f}")

    # Permutation importance for each feature
    print(f"\nPermutation importance (3 repeats each, {n_features} features):")
    print("-" * 60)

    results = []
    t0 = time.time()
    for feat_idx in range(n_features):
        drops = []
        for rep in range(3):
            # Shuffle this feature across all batches
            perm = np.random.RandomState(42 + rep).permutation(len(struct_arr))
            shuffled_vals = struct_arr[perm, feat_idx]

            # Apply shuffled feature to batches
            offset = 0
            all_probs_perm = []
            with torch.no_grad():
                for batch in all_batches:
                    bs = batch["label"].shape[0]
                    orig_struct = batch["structural_features"].clone()

                    # Replace feature
                    batch["structural_features"][:, feat_idx] = torch.tensor(
                        shuffled_vals[offset:offset+bs], dtype=torch.float32,
                    )

                    kwargs = batch_to_model_kwargs(batch, device)
                    out = model(**kwargs)
                    probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
                    all_probs_perm.append(probs)

                    # Restore
                    batch["structural_features"] = orig_struct
                    offset += bs

            perm_probs = np.concatenate(all_probs_perm)
            perm_auc = roc_auc_score(labels, perm_probs)
            drops.append(baseline_auc - perm_auc)

        mean_drop = np.mean(drops)
        std_drop = np.std(drops)
        results.append((feat_idx, FEATURE_NAMES[feat_idx], mean_drop, std_drop))

        elapsed = time.time() - t0
        eta = elapsed / (feat_idx + 1) * (n_features - feat_idx - 1)
        print(f"  [{feat_idx+1:2d}/{n_features}] {FEATURE_NAMES[feat_idx]:<20} AUC drop={mean_drop:+.6f} (ETA: {eta:.0f}s)")

    # Sort by importance (largest drop first)
    results.sort(key=lambda x: -x[2])

    print(f"\n{'Rank':>4}  {'Feature':<20}  {'AUC Drop':>10}  {'Std':>8}  {'Importance':>12}")
    print("=" * 65)
    for rank, (idx, name, drop, std) in enumerate(results, 1):
        bar = "#" * max(0, int(drop * 5000))
        print(f"{rank:4d}  {name:<20}  {drop:10.6f}  {std:8.6f}  {bar}")

    # Group analysis
    results_dict = {idx: drop for idx, _, drop, _ in results}

    print("\n\n=== Feature Group Analysis ===")
    groups = {
        "v7-v10 (basic)": list(range(0, 12)),
        "v11 (pairing)": list(range(12, 20)),
        "v13 (ViennaRNA)": list(range(20, 26)),
        "v14 (conservation)": list(range(26, 31)),
    }
    for gname, gidx in groups.items():
        gdrop = sum(results_dict.get(i, 0) for i in gidx)
        print(f"  {gname:<20}: total AUC drop = {gdrop:.6f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
