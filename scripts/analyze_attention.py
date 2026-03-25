#!/usr/bin/env python
"""Attention pattern biological analysis for DeepExoMir.

Extracts and visualizes attention weights from cross-attention layers
and interaction pooling to reveal biological insights about miRNA
targeting mechanisms.

Analyses performed:
    1. Seed region attention (miRNA positions 2-8)
    2. 3' compensatory region (miRNA positions 13-16)
    3. Position-specific attention heatmaps
    4. Case study attention maps for specific miRNA-target pairs
    5. Per-layer attention progression

Usage:
    python scripts/analyze_attention.py [--n_samples 5000] [--batch_size 128]

Outputs saved to results/attention_analysis/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deepexomir.data.dataset import (
    MiRNATargetDataset,
    mirna_target_collate_fn,
)
from deepexomir.model.deepexomir_v8 import DeepExoMirModelV8

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Matplotlib setup for publication-quality figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
})

OUTPUT_DIR = PROJECT_ROOT / "results" / "attention_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Biological region definitions (0-indexed positions in the miRNA)
SEED_REGION = (1, 8)          # positions 2-8 (canonical seed)
SUPPLEMENTARY_3P = (12, 16)   # positions 13-16 (3' compensatory)
CENTRAL_REGION = (8, 12)      # positions 9-12 (central region)
THREE_PRIME = (16, 22)        # positions 17-22 (3' end)

REGION_COLORS = {
    "Seed (2-8)": "#E74C3C",
    "Central (9-12)": "#F39C12",
    "3' Comp. (13-16)": "#3498DB",
    "3' End (17-22)": "#2ECC71",
    "Flanking": "#95A5A6",
}


# ---------------------------------------------------------------------------
# Attention Hook System
# ---------------------------------------------------------------------------

class AttentionCaptureHook:
    """Register forward hooks on attention modules to capture attention
    weights without modifying model code.

    Captures:
        - Cross-attention weights from each CrossAttentionBlock
          (both mirna_cross_attn and target_cross_attn)
        - Interaction pooling attention weights (self_pool, cross_pool)
    """

    def __init__(self, model: DeepExoMirModelV8) -> None:
        self.model = model
        self.hooks: list = []
        self.captured: Dict[str, torch.Tensor] = {}
        self._register_hooks()

    def _register_hooks(self) -> None:
        """Walk the model tree and register hooks on _MultiHeadAttention
        and _PoolingAttention modules."""

        # 1. Cross-attention layers inside HybridEncoder blocks
        for block_idx, block in enumerate(self.model.encoder.blocks):
            if not block.has_cross_attn:
                continue
            ca_block = block.cross_attn_block

            # mirna_cross_attn: miRNA query attends to target key/value
            # -> attention shape [B, n_heads, Lm, Lt]
            self._hook_mha(
                ca_block.mirna_cross_attn,
                f"cross_attn_block{block_idx}_mirna2target",
            )

            # target_cross_attn: target query attends to miRNA key/value
            # -> attention shape [B, n_heads, Lt, Lm]
            self._hook_mha(
                ca_block.target_cross_attn,
                f"cross_attn_block{block_idx}_target2mirna",
            )

        # 2. Interaction pooling attention
        # Note: self_pool is called twice (miRNA->miRNA, target->target)
        # and cross_pool is called twice (miRNA->target, target->miRNA).
        # We use a call counter to differentiate them.
        if self.model.use_interaction_pooling:
            ip = self.model.interaction_pooling
            self._pool_call_counts = {}
            self._hook_pooling_attn_multi(
                ip.self_pool, "interaction_self_pool",
                call_names=["self_mirna2mirna", "self_target2target"],
            )
            self._hook_pooling_attn_multi(
                ip.cross_pool, "interaction_cross_pool",
                call_names=["cross_mirna2target", "cross_target2mirna"],
            )

    def _hook_mha(self, module: nn.Module, name: str) -> None:
        """Hook a _MultiHeadAttention module to capture softmax weights."""

        def hook_fn(mod, inp, out, name=name):
            # Recompute attention weights from Q, K (no dropout in eval)
            query, key, value = inp[0], inp[1], inp[2]
            key_padding_mask = inp[3] if len(inp) > 3 else None

            B, Lq, _ = query.shape
            Lk = key.shape[1]
            n_heads = mod.n_heads
            d_k = mod.d_k

            Q = mod.w_q(query).view(B, Lq, n_heads, d_k).transpose(1, 2)
            K = mod.w_k(key).view(B, Lk, n_heads, d_k).transpose(1, 2)

            scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
            if key_padding_mask is not None:
                mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
                scores = scores.masked_fill(mask, float("-inf"))

            attn_weights = F.softmax(scores, dim=-1)  # [B, H, Lq, Lk]
            self.captured[name] = attn_weights.detach().cpu()

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def _hook_pooling_attn_multi(
        self, module: nn.Module, base_name: str, call_names: List[str],
    ) -> None:
        """Hook a _PoolingAttention module that is called multiple times.
        Uses a counter to assign different names to each call."""
        counter_key = base_name
        self._pool_call_counts[counter_key] = 0

        def hook_fn(mod, inp, out, base=base_name, names=call_names):
            query, key, value = inp[0], inp[1], inp[2]
            key_mask = inp[3] if len(inp) > 3 else None

            B, Lq, D = query.shape
            Lk = key.shape[1]
            n_heads = mod.n_heads
            head_dim = mod.head_dim
            scale = mod.scale

            q = mod.q_proj(query).view(B, Lq, n_heads, head_dim).transpose(1, 2)
            k = mod.k_proj(key).view(B, Lk, n_heads, head_dim).transpose(1, 2)

            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            if key_mask is not None:
                attn = attn.masked_fill(
                    key_mask.unsqueeze(1).unsqueeze(2), float("-inf"),
                )
            attn = F.softmax(attn, dim=-1)  # [B, H, 1, Lk]

            call_idx = self._pool_call_counts[base]
            if call_idx < len(names):
                self.captured[f"interaction_{names[call_idx]}"] = attn.detach().cpu()
            self._pool_call_counts[base] = call_idx + 1

        handle = module.register_forward_hook(hook_fn)
        self.hooks.append(handle)

    def clear(self) -> None:
        self.captured.clear()
        if hasattr(self, "_pool_call_counts"):
            for k in self._pool_call_counts:
                self._pool_call_counts[k] = 0

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_model(
    model_config_path: Path,
    checkpoint_path: Path,
    device: torch.device,
) -> DeepExoMirModelV8:
    """Load model from config + checkpoint."""
    with open(model_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model = DeepExoMirModelV8(
        config=config,
        load_backbone=False,
        precomputed_embeddings=True,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

    # Handle EMA state dict if present
    if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
        logger.info("Using EMA state dict from checkpoint")
        state_dict = ckpt["ema_state_dict"]

    # Remove unexpected keys
    model_keys = set(model.state_dict().keys())
    filtered = {k: v for k, v in state_dict.items() if k in model_keys}
    missing = model_keys - set(filtered.keys())
    if missing:
        logger.warning("Missing keys: %d (e.g. %s)", len(missing), list(missing)[:3])

    model.load_state_dict(filtered, strict=False)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded model: %d params, device=%s",
        model.trainable_parameters(), device,
    )
    return model


# ---------------------------------------------------------------------------
# Data Preparation
# ---------------------------------------------------------------------------

def prepare_test_data(
    n_samples: int = 5000,
    positive_only: bool = True,
    feature_version: str = "v18",
) -> Tuple[MiRNATargetDataset, pd.DataFrame]:
    """Load test dataset and select a subset."""
    data_dir = PROJECT_ROOT / "data" / "processed"
    emb_dir = PROJECT_ROOT / "data" / "embeddings_cache_pca256"

    dataset = MiRNATargetDataset(
        parquet_path=data_dir / "test.parquet",
        skip_bp_matrix=True,  # model computes on GPU
        embeddings_dir=emb_dir,
        feature_version=feature_version,
    )

    df = dataset.df.copy()
    if positive_only:
        df = df[df["label"] == 1]

    if n_samples < len(df):
        df = df.sample(n=n_samples, random_state=42)

    logger.info("Selected %d samples for analysis", len(df))
    return dataset, df


def build_subset_loader(
    dataset: MiRNATargetDataset,
    indices: List[int],
    batch_size: int = 128,
) -> DataLoader:
    """Create a DataLoader for a subset of indices."""
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=mirna_target_collate_fn,
        pin_memory=True,
    )


# ---------------------------------------------------------------------------
# Attention Extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_attention_weights(
    model: DeepExoMirModelV8,
    loader: DataLoader,
    hook: AttentionCaptureHook,
    device: torch.device,
    mirna_seqs_all: List[str],
) -> Dict[str, Any]:
    """Run inference and accumulate attention statistics.

    Returns
    -------
    dict with:
        - cross_attn_mirna2target: dict[layer_name] -> accumulated [Lm, Lt]
        - cross_attn_target2mirna: dict[layer_name] -> accumulated [Lt, Lm]
        - interaction_pool_weights: dict[pool_name] -> accumulated weights
        - mirna_position_attention: np.ndarray [Lm] averaged cross-attn to target
        - n_samples: int
        - per_sample_attns: list of dicts for case studies
    """
    max_mirna_len = model.max_mirna_len   # 30
    max_target_len = model.max_target_len  # 50

    # Accumulators
    cross_attn_m2t_accum = {}  # layer -> [Lm, Lt]
    cross_attn_t2m_accum = {}  # layer -> [Lt, Lm]
    pool_accum = {}
    # For miRNA position analysis, accumulate the max attention weight
    # per miRNA position (across target positions), which shows how
    # "focused" each miRNA position is on the target.
    mirna_pos_max_attn_accum = np.zeros(max_mirna_len, dtype=np.float64)
    # Also accumulate target->miRNA attention (which miRNA positions
    # are most attended TO by target positions)
    target_to_mirna_importance = np.zeros(max_mirna_len, dtype=np.float64)
    n_total = 0

    # Per-sample storage for case studies (indexed by mirna_seq)
    per_sample_attns = []

    sample_idx = 0
    for batch_features, batch_labels in loader:
        # Move tensors to device
        model_kwargs = {
            "mirna_seqs": batch_features["mirna_seq"],
            "target_seqs": batch_features["target_seq"],
        }
        # Map dataset key -> model kwarg name
        key_map = {
            "mirna_pooled_emb": "mirna_pooled_emb",
            "target_pooled_emb": "target_pooled_emb",
            "mirna_pertoken_emb": "mirna_pertoken_emb",
            "mirna_pertoken_mask": "mirna_pertoken_mask",
            "target_pertoken_emb": "target_pertoken_emb",
            "target_pertoken_mask": "target_pertoken_mask",
            "structural_features": "struct_features",
        }
        for data_key, model_key in key_map.items():
            if data_key in batch_features:
                val = batch_features[data_key]
                if isinstance(val, torch.Tensor):
                    model_kwargs[model_key] = val.to(device)

        hook.clear()
        _ = model(**model_kwargs)

        B = len(batch_features["mirna_seq"])
        n_total += B

        # Process captured attention weights
        for name, attn in hook.captured.items():
            # attn: [B, H, Lq, Lk]
            # Average over heads -> [B, Lq, Lk]
            attn_avg = attn.float().mean(dim=1)  # [B, Lq, Lk]

            if name.startswith("interaction_"):
                # Interaction pooling attention -- handle before cross-attn
                # since some names also contain "mirna2target"
                # Pooling attn: [B, H, 1, Lk] -> [B, Lk]
                pool_attn = attn.float().mean(dim=1).squeeze(1)  # [B, Lk]
                if name not in pool_accum:
                    pool_accum[name] = np.zeros(
                        pool_attn.shape[-1], dtype=np.float64,
                    )
                pool_accum[name] += pool_attn.sum(dim=0).numpy()

            elif "mirna2target" in name:
                if name not in cross_attn_m2t_accum:
                    cross_attn_m2t_accum[name] = np.zeros(
                        (max_mirna_len, max_target_len), dtype=np.float64,
                    )
                cross_attn_m2t_accum[name] += attn_avg.sum(dim=0).numpy()
                # Max attention per miRNA position (how focused each pos is)
                mirna_pos_max_attn_accum += attn_avg.max(dim=2).values.sum(dim=0).numpy()

            elif "target2mirna" in name:
                if name not in cross_attn_t2m_accum:
                    cross_attn_t2m_accum[name] = np.zeros(
                        (max_target_len, max_mirna_len), dtype=np.float64,
                    )
                cross_attn_t2m_accum[name] += attn_avg.sum(dim=0).numpy()
                # How much each miRNA position is attended to by all target pos
                # attn_avg: [B, Lt, Lm] -> sum over target query dim -> [B, Lm]
                target_to_mirna_importance += attn_avg.sum(dim=1).sum(dim=0).numpy()

        # Store per-sample attention for case studies (first batch only
        # to save memory; we pick case studies from these)
        if len(per_sample_attns) < 500:
            for i in range(min(B, 500 - len(per_sample_attns))):
                sample_data = {
                    "mirna_seq": batch_features["mirna_seq"][i],
                    "target_seq": batch_features["target_seq"][i],
                    "label": batch_labels[i].item(),
                    "attns": {},
                }
                for name, attn in hook.captured.items():
                    if "mirna2target" in name and name.startswith("cross_attn"):
                        # [H, Lm, Lt] for this sample
                        sample_data["attns"][name] = attn[i].float().numpy()
                per_sample_attns.append(sample_data)

        if n_total % 2000 < B:
            logger.info("  Processed %d / ? samples", n_total)

    # Normalize accumulators
    for k in cross_attn_m2t_accum:
        cross_attn_m2t_accum[k] /= n_total
    for k in cross_attn_t2m_accum:
        cross_attn_t2m_accum[k] /= n_total
    for k in pool_accum:
        pool_accum[k] /= n_total
    mirna_pos_max_attn_accum /= n_total
    target_to_mirna_importance /= n_total

    return {
        "cross_attn_m2t": cross_attn_m2t_accum,
        "cross_attn_t2m": cross_attn_t2m_accum,
        "pool_weights": pool_accum,
        "mirna_pos_max_attn": mirna_pos_max_attn_accum,
        "target_to_mirna_importance": target_to_mirna_importance,
        "n_samples": n_total,
        "per_sample_attns": per_sample_attns,
    }


# ---------------------------------------------------------------------------
# Visualization Functions
# ---------------------------------------------------------------------------

def _get_mirna_region_label(pos: int) -> str:
    """Return the biological region label for a 0-indexed miRNA position."""
    pos1 = pos + 1  # 1-indexed
    if 2 <= pos1 <= 8:
        return "Seed (2-8)"
    elif 9 <= pos1 <= 12:
        return "Central (9-12)"
    elif 13 <= pos1 <= 16:
        return "3' Comp. (13-16)"
    elif 17 <= pos1 <= 22:
        return "3' End (17-22)"
    else:
        return "Flanking"


def plot_mirna_position_attention_bar(
    mirna_pos_attn: np.ndarray,
    n_layers: int,
    save_path: Path,
    max_pos: int = 25,
) -> None:
    """Bar chart of average attention by miRNA position, colored by region."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    positions = np.arange(max_pos)
    values = mirna_pos_attn[:max_pos] / n_layers  # normalize per layer

    # Color by region
    colors = [REGION_COLORS[_get_mirna_region_label(p)] for p in positions]

    bars = ax.bar(positions, values, color=colors, edgecolor="white", linewidth=0.3)

    # Add region annotations
    ax.axvspan(0.5, 7.5, alpha=0.08, color="#E74C3C", zorder=0)
    ax.axvspan(11.5, 15.5, alpha=0.08, color="#3498DB", zorder=0)

    ax.set_xlabel("miRNA Position (1-indexed)")
    ax.set_ylabel("Mean Cross-Attention Weight")
    ax.set_title("Target Attention to miRNA Positions (Averaged)")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p + 1) for p in positions], fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=c, label=l) for l, c in REGION_COLORS.items()
    ]
    ax.legend(
        handles=legend_elements, loc="upper right",
        fontsize=7, framealpha=0.9, ncol=2,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_cross_attention_heatmap(
    attn_matrix: np.ndarray,
    title: str,
    save_path: Path,
    mirna_len: int = 25,
    target_len: int = 40,
    mirna_seq: Optional[str] = None,
    target_seq: Optional[str] = None,
) -> None:
    """Heatmap of cross-attention (miRNA position x target position)."""
    data = attn_matrix[:mirna_len, :target_len]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#FFFFFF", "#FFF3E0", "#FF9800", "#E65100", "#B71C1C"],
    )

    im = ax.imshow(
        data, aspect="auto", cmap=cmap,
        interpolation="nearest",
    )

    # Axis labels
    if mirna_seq:
        ytick_labels = [f"{i+1}:{c}" for i, c in enumerate(mirna_seq[:mirna_len])]
        ax.set_yticks(range(mirna_len))
        ax.set_yticklabels(ytick_labels, fontsize=6, fontfamily="monospace")
    else:
        ax.set_ylabel("miRNA Position")
        ax.set_yticks(range(0, mirna_len, 5))
        ax.set_yticklabels([str(i + 1) for i in range(0, mirna_len, 5)])

    if target_seq:
        xtick_labels = [f"{c}" for c in target_seq[:target_len]]
        ax.set_xticks(range(target_len))
        ax.set_xticklabels(xtick_labels, fontsize=5, fontfamily="monospace", rotation=0)
    else:
        ax.set_xlabel("Target Position")
        ax.set_xticks(range(0, target_len, 5))
        ax.set_xticklabels([str(i + 1) for i in range(0, target_len, 5)])

    # Highlight seed region
    ax.axhline(y=0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=7.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.text(-0.5, 4, "seed", fontsize=6, color="red", ha="right", va="center",
            fontweight="bold")

    # 3' compensatory
    ax.axhline(y=11.5, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(y=15.5, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_title(title, fontsize=11, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Attention Weight", fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_per_layer_attention_heatmaps(
    cross_attn_m2t: Dict[str, np.ndarray],
    save_path: Path,
    mirna_len: int = 25,
    target_len: int = 40,
) -> None:
    """Grid of cross-attention heatmaps, one per layer."""
    n_layers = len(cross_attn_m2t)
    if n_layers == 0:
        return

    fig, axes = plt.subplots(
        1, n_layers, figsize=(4 * n_layers, 4), sharey=True,
    )
    if n_layers == 1:
        axes = [axes]

    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#FFFFFF", "#FFF3E0", "#FF9800", "#E65100", "#B71C1C"],
    )

    sorted_keys = sorted(cross_attn_m2t.keys())

    for i, (key, ax) in enumerate(zip(sorted_keys, axes)):
        data = cross_attn_m2t[key][:mirna_len, :target_len]
        im = ax.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")

        # Extract layer index from name
        layer_label = key.split("block")[1].split("_")[0]
        ax.set_title(f"Layer {layer_label}", fontsize=10)

        ax.set_xlabel("Target Position")
        ax.set_xticks(range(0, target_len, 10))
        ax.set_xticklabels([str(j + 1) for j in range(0, target_len, 10)])

        if i == 0:
            ax.set_ylabel("miRNA Position")
            ax.set_yticks(range(0, mirna_len, 5))
            ax.set_yticklabels([str(j + 1) for j in range(0, mirna_len, 5)])

        # Highlight seed region
        ax.axhline(y=0.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(y=7.5, color="red", linewidth=0.5, linestyle="--", alpha=0.5)

    fig.suptitle(
        "Cross-Attention (miRNA -> Target) Across Layers",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_region_attention_comparison(
    cross_attn_t2m: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Grouped bar chart comparing attention to biological regions per layer.

    Uses target->miRNA attention to show which miRNA positions are
    most attended to by target positions.
    """
    regions = {
        "Seed (2-8)": (1, 8),
        "Central (9-12)": (8, 12),
        "3' Comp. (13-16)": (12, 16),
        "3' End (17-22)": (16, 22),
    }

    sorted_keys = sorted(cross_attn_t2m.keys())
    n_layers = len(sorted_keys)
    n_regions = len(regions)

    # For each layer, sum attention over target query positions
    # to get how much each miRNA position is attended to
    data = np.zeros((n_layers, n_regions))
    for li, key in enumerate(sorted_keys):
        t2m = cross_attn_t2m[key]  # [Lt, Lm]
        mirna_attn = t2m.sum(axis=0)  # [Lm]
        for ri, (rname, (start, end)) in enumerate(regions.items()):
            data[li, ri] = mirna_attn[start:end].mean()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    x = np.arange(n_layers)
    width = 0.18
    offsets = np.arange(n_regions) - (n_regions - 1) / 2

    for ri, (rname, _) in enumerate(regions.items()):
        color = REGION_COLORS[rname]
        ax.bar(
            x + offsets[ri] * width, data[:, ri], width,
            label=rname, color=color, edgecolor="white", linewidth=0.3,
        )

    layer_labels = []
    for key in sorted_keys:
        idx = key.split("block")[1].split("_")[0]
        layer_labels.append(f"Layer {idx}")

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Attention to miRNA Regions Across Cross-Attention Layers")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_interaction_pooling_weights(
    pool_weights: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Visualize interaction pooling attention weights."""
    n_plots = len(pool_weights)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 3))
    if n_plots == 1:
        axes = [axes]

    for ax, (name, weights) in zip(axes, pool_weights.items()):
        L = len(weights)
        positions = np.arange(L)

        # Determine if this is miRNA-length or target-length
        if L == 30:
            colors = [REGION_COLORS[_get_mirna_region_label(p)] for p in positions]
            xlabel = "miRNA Position"
            max_show = 25
        else:
            colors = ["#7F8C8D"] * L
            xlabel = "Target Position"
            max_show = min(L, 45)

        ax.bar(
            positions[:max_show], weights[:max_show],
            color=colors[:max_show], edgecolor="white", linewidth=0.3,
        )

        label = name.replace("interaction_", "").replace("_", " ").title()
        ax.set_title(f"Interaction Pooling: {label}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Attention Weight")
        ax.set_xticks(range(0, max_show, 5))
        ax.set_xticklabels([str(i + 1) for i in range(0, max_show, 5)])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_case_study(
    sample: Dict[str, Any],
    mirna_name: str,
    save_path: Path,
) -> None:
    """Detailed attention map for a single miRNA-target pair."""
    mirna_seq = sample["mirna_seq"]
    target_seq = sample["target_seq"]

    # Average across all cross-attention layers
    attn_layers = [v for k, v in sample["attns"].items() if "mirna2target" in k]
    if not attn_layers:
        return

    # Average over heads and layers
    # Each layer: [H, Lm, Lt]
    avg_attn = np.mean(
        [a.mean(axis=0) for a in attn_layers], axis=0,
    )  # [Lm, Lt]

    mirna_len = min(len(mirna_seq), 25)
    target_len = min(len(target_seq), 40)

    fig, (ax_heat, ax_bar) = plt.subplots(
        2, 1, figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # Heatmap
    data = avg_attn[:mirna_len, :target_len]
    cmap = LinearSegmentedColormap.from_list(
        "attn", ["#FFFFFF", "#FFF3E0", "#FF9800", "#E65100", "#B71C1C"],
    )
    im = ax_heat.imshow(data, aspect="auto", cmap=cmap, interpolation="nearest")

    # Y-axis: miRNA sequence
    ytick_labels = [f"{i+1}:{c}" for i, c in enumerate(mirna_seq[:mirna_len])]
    ax_heat.set_yticks(range(mirna_len))
    ax_heat.set_yticklabels(ytick_labels, fontsize=6, fontfamily="monospace")

    # X-axis: target sequence
    xtick_labels = list(target_seq[:target_len])
    ax_heat.set_xticks(range(target_len))
    ax_heat.set_xticklabels(xtick_labels, fontsize=5, fontfamily="monospace")

    ax_heat.axhline(y=0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_heat.axhline(y=7.5, color="red", linewidth=0.8, linestyle="--", alpha=0.7)
    ax_heat.axhline(y=11.5, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)
    ax_heat.axhline(y=15.5, color="blue", linewidth=0.5, linestyle="--", alpha=0.5)

    ax_heat.set_title(f"{mirna_name} Attention Map", fontsize=11, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02)
    cbar.set_label("Attn", fontsize=7)

    # Bar chart: miRNA position-wise attention
    mirna_attn = data.sum(axis=1)  # sum over target positions
    positions = np.arange(mirna_len)
    colors = [REGION_COLORS[_get_mirna_region_label(p)] for p in positions]
    ax_bar.bar(positions, mirna_attn, color=colors, edgecolor="white", linewidth=0.3)
    ax_bar.set_xticks(positions)
    ax_bar.set_xticklabels([str(p + 1) for p in positions], fontsize=7)
    ax_bar.set_xlabel("miRNA Position")
    ax_bar.set_ylabel("Total Attn")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_seed_vs_nonseed_statistics(
    cross_attn_t2m: Dict[str, np.ndarray],
    save_path: Path,
) -> None:
    """Bar comparison of seed vs non-seed attention across layers.

    Uses target->miRNA attention to show which miRNA positions
    are most attended to by target positions.
    """
    sorted_keys = sorted(cross_attn_t2m.keys())

    seed_vals = []
    nonseed_vals = []
    layer_labels = []

    for key in sorted_keys:
        t2m = cross_attn_t2m[key]  # [Lt, Lm]
        mirna_attn = t2m.sum(axis=0)  # [Lm]

        seed_mean = mirna_attn[1:8].mean()  # positions 2-8
        nonseed_mean = np.concatenate([mirna_attn[:1], mirna_attn[8:22]]).mean()

        seed_vals.append(seed_mean)
        nonseed_vals.append(nonseed_mean)

        idx = key.split("block")[1].split("_")[0]
        layer_labels.append(f"Layer {idx}")

    fig, ax = plt.subplots(figsize=(5, 3.5))

    x = np.arange(len(sorted_keys))
    width = 0.3

    ax.bar(x - width / 2, seed_vals, width, label="Seed (pos 2-8)",
           color="#E74C3C", edgecolor="white", linewidth=0.3)
    ax.bar(x + width / 2, nonseed_vals, width, label="Non-seed",
           color="#95A5A6", edgecolor="white", linewidth=0.3)

    # Compute and annotate fold enrichment
    for i in range(len(sorted_keys)):
        if nonseed_vals[i] > 0:
            fold = seed_vals[i] / nonseed_vals[i]
            ax.text(
                x[i], max(seed_vals[i], nonseed_vals[i]) * 1.05,
                f"{fold:.1f}x", ha="center", fontsize=8, fontweight="bold",
                color="#E74C3C",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(layer_labels)
    ax.set_ylabel("Mean Attention Weight")
    ax.set_title("Seed Region Enrichment in Cross-Attention")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def plot_target2mirna_heatmap(
    cross_attn_t2m: Dict[str, np.ndarray],
    save_path: Path,
    mirna_len: int = 25,
    target_len: int = 40,
) -> None:
    """Average target->miRNA attention to show which miRNA positions
    are most attended to by the target sequence."""

    # Average across all layers
    all_mats = list(cross_attn_t2m.values())
    avg = np.mean(all_mats, axis=0)  # [Lt, Lm]

    # Sum over target query positions to get miRNA "importance"
    mirna_importance = avg[:target_len, :mirna_len].sum(axis=0)  # [Lm]

    fig, ax = plt.subplots(figsize=(7, 3))

    positions = np.arange(mirna_len)
    colors = [REGION_COLORS[_get_mirna_region_label(p)] for p in positions]
    ax.bar(positions, mirna_importance, color=colors, edgecolor="white", linewidth=0.3)

    ax.axvspan(0.5, 7.5, alpha=0.08, color="#E74C3C", zorder=0)
    ax.axvspan(11.5, 15.5, alpha=0.08, color="#3498DB", zorder=0)

    ax.set_xlabel("miRNA Position (1-indexed)")
    ax.set_ylabel("Total Attention from Target")
    ax.set_title("Target -> miRNA Attention: Which miRNA Positions are Attended?")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(p + 1) for p in positions], fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("Saved: %s", save_path)


def generate_statistics_report(
    results: Dict[str, Any],
    save_path: Path,
) -> None:
    """Generate a text report with key statistics."""
    lines = []
    lines.append("=" * 70)
    lines.append("DeepExoMir Attention Analysis Report")
    lines.append("=" * 70)
    lines.append(f"Total samples analyzed: {results['n_samples']}")
    lines.append("")

    # Seed region statistics -- use target->miRNA importance
    # (how much target positions attend to each miRNA position)
    t2m_importance = results["target_to_mirna_importance"]
    n_cross_layers = len(results["cross_attn_t2m"])
    attn_per_layer = t2m_importance / max(n_cross_layers, 1)

    seed_attn = attn_per_layer[1:8].mean()
    nonseed_attn = np.concatenate([attn_per_layer[:1], attn_per_layer[8:22]]).mean()
    comp_attn = attn_per_layer[12:16].mean()
    central_attn = attn_per_layer[8:12].mean()

    lines.append("--- miRNA Position Attention (averaged over layers) ---")
    lines.append(f"  Seed region (pos 2-8) mean attention:    {seed_attn:.6f}")
    lines.append(f"  Non-seed (pos 1 + 9-22) mean attention:  {nonseed_attn:.6f}")
    lines.append(f"  Seed enrichment fold:                    {seed_attn / max(nonseed_attn, 1e-10):.2f}x")
    lines.append(f"  Central region (pos 9-12) mean:          {central_attn:.6f}")
    lines.append(f"  3' compensatory (pos 13-16) mean:        {comp_attn:.6f}")
    lines.append(f"  3' comp enrichment vs non-seed:          {comp_attn / max(nonseed_attn, 1e-10):.2f}x")
    lines.append("")

    # Per-layer seed enrichment (using target->miRNA attention)
    lines.append("--- Per-Layer Seed Enrichment (target -> miRNA attention) ---")
    for key in sorted(results["cross_attn_t2m"].keys()):
        t2m = results["cross_attn_t2m"][key]  # [Lt, Lm]
        # Sum over target query positions: how much each miRNA pos is attended to
        mirna_attn_layer = t2m.sum(axis=0)  # [Lm]
        s = mirna_attn_layer[1:8].mean()
        ns = np.concatenate([mirna_attn_layer[:1], mirna_attn_layer[8:22]]).mean()
        idx = key.split("block")[1].split("_")[0]
        lines.append(f"  Layer {idx}: seed={s:.6f}, non-seed={ns:.6f}, fold={s/max(ns,1e-10):.2f}x")
    lines.append("")

    # Top-5 miRNA positions
    lines.append("--- Top-5 miRNA Positions by Attention ---")
    top5 = np.argsort(attn_per_layer)[::-1][:5]
    for rank, pos in enumerate(top5, 1):
        region = _get_mirna_region_label(pos)
        lines.append(f"  #{rank}: Position {pos+1} ({region}): {attn_per_layer[pos]:.6f}")
    lines.append("")

    # Interaction pooling
    if results["pool_weights"]:
        lines.append("--- Interaction Pooling Weights ---")
        for name, weights in results["pool_weights"].items():
            if len(weights) == 30:
                seed_pool = weights[1:8].mean()
                total_pool = weights[:22].mean()
                lines.append(
                    f"  {name}: seed_mean={seed_pool:.6f}, "
                    f"overall_mean={total_pool:.6f}, "
                    f"ratio={seed_pool/max(total_pool,1e-10):.2f}x"
                )
            else:
                lines.append(f"  {name}: mean={weights.mean():.6f}, max={weights.max():.6f}")
        lines.append("")

    report = "\n".join(lines)
    save_path.write_text(report, encoding="utf-8")
    print(report)
    logger.info("Saved report: %s", save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DeepExoMir attention analysis")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Number of positive samples to analyze")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for inference")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint path (default: v19 best)")
    parser.add_argument("--model_config", type=str, default=None,
                        help="Model config path (default: v19)")
    parser.add_argument("--feature_version", type=str, default="v18",
                        help="Structural feature version")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Paths
    config_path = Path(args.model_config) if args.model_config else (
        PROJECT_ROOT / "configs" / "model_config_v19.yaml"
    )
    ckpt_path = Path(args.checkpoint) if args.checkpoint else (
        PROJECT_ROOT / "checkpoints" / "v19" / "checkpoint_epoch034_val_auc_0.8521.pt"
    )

    # 1. Load model
    logger.info("Loading model from %s", ckpt_path.name)
    model = load_model(config_path, ckpt_path, device)

    # 2. Register attention hooks
    hook = AttentionCaptureHook(model)
    logger.info("Registered %d attention hooks", len(hook.hooks))

    # 3. Prepare data
    logger.info("Loading test data (positive samples only)...")
    dataset, subset_df = prepare_test_data(
        n_samples=args.n_samples,
        positive_only=True,
        feature_version=args.feature_version,
    )

    # Get indices in the original dataset
    indices = subset_df.index.tolist()
    loader = build_subset_loader(dataset, indices, batch_size=args.batch_size)

    # 4. Extract attention weights
    logger.info("Extracting attention weights from %d samples...", len(indices))
    results = extract_attention_weights(
        model, loader, hook, device,
        mirna_seqs_all=subset_df["mirna_seq"].tolist(),
    )
    logger.info("Extraction complete. %d samples processed.", results["n_samples"])

    # 5. Generate figures

    # 5a. miRNA position attention bar chart (target -> miRNA importance)
    n_cross_layers = len(results["cross_attn_m2t"])
    plot_mirna_position_attention_bar(
        results["target_to_mirna_importance"],
        n_layers=n_cross_layers,
        save_path=OUTPUT_DIR / "fig1_mirna_position_attention.png",
    )

    # 5b. Average cross-attention heatmap (all layers combined)
    if results["cross_attn_m2t"]:
        avg_m2t = np.mean(
            list(results["cross_attn_m2t"].values()), axis=0,
        )
        plot_cross_attention_heatmap(
            avg_m2t,
            title="Average Cross-Attention: miRNA -> Target",
            save_path=OUTPUT_DIR / "fig2_avg_cross_attention_heatmap.png",
        )

    # 5c. Per-layer cross-attention heatmaps
    plot_per_layer_attention_heatmaps(
        results["cross_attn_m2t"],
        save_path=OUTPUT_DIR / "fig3_per_layer_cross_attention.png",
    )

    # 5d. Region comparison across layers (using target->miRNA attention)
    plot_region_attention_comparison(
        results["cross_attn_t2m"],
        save_path=OUTPUT_DIR / "fig4_region_attention_comparison.png",
    )

    # 5e. Seed vs non-seed enrichment (using target->miRNA attention)
    plot_seed_vs_nonseed_statistics(
        results["cross_attn_t2m"],
        save_path=OUTPUT_DIR / "fig5_seed_enrichment.png",
    )

    # 5f. Target -> miRNA attention
    if results["cross_attn_t2m"]:
        plot_target2mirna_heatmap(
            results["cross_attn_t2m"],
            save_path=OUTPUT_DIR / "fig6_target2mirna_attention.png",
        )

    # 5g. Interaction pooling weights
    plot_interaction_pooling_weights(
        results["pool_weights"],
        save_path=OUTPUT_DIR / "fig7_interaction_pooling.png",
    )

    # 5h. Case studies
    logger.info("Generating case study attention maps...")

    # Build seq-to-name map from mirbase
    mirbase_path = PROJECT_ROOT / "data" / "processed" / "mirbase_hsa.parquet"
    seq2name = {}
    if mirbase_path.exists():
        mb = pd.read_parquet(mirbase_path)
        for _, row in mb.iterrows():
            seq2name[row["mirna_seq"]] = row["mirna_id"]

    # Find representative samples for case studies
    # Pick different miRNAs for diversity
    seen_mirnas = set()
    case_studies = []
    for sample in results["per_sample_attns"]:
        if sample["label"] != 1:
            continue
        name = seq2name.get(sample["mirna_seq"], "unknown")
        if name in seen_mirnas or name == "unknown":
            continue
        seen_mirnas.add(name)
        case_studies.append((name, sample))
        if len(case_studies) >= 6:
            break

    for name, sample in case_studies:
        safe_name = name.replace("/", "_").replace("\\", "_")
        plot_case_study(
            sample, name,
            save_path=OUTPUT_DIR / f"fig8_case_study_{safe_name}.png",
        )

    # 6. Generate statistics report
    generate_statistics_report(
        results,
        save_path=OUTPUT_DIR / "attention_analysis_report.txt",
    )

    # 7. Save raw attention data for further analysis
    np.savez_compressed(
        OUTPUT_DIR / "attention_data.npz",
        mirna_pos_max_attn=results["mirna_pos_max_attn"],
        target_to_mirna_importance=results["target_to_mirna_importance"],
        **{f"m2t_{k}": v for k, v in results["cross_attn_m2t"].items()},
        **{f"t2m_{k}": v for k, v in results["cross_attn_t2m"].items()},
        **{f"pool_{k}": v for k, v in results["pool_weights"].items()},
        n_samples=results["n_samples"],
    )
    logger.info("Saved raw attention data: %s", OUTPUT_DIR / "attention_data.npz")

    # Cleanup
    hook.remove_hooks()
    logger.info("Done. All figures saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
