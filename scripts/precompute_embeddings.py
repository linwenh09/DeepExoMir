"""Pre-compute mean-pooled RiNALMo embeddings for all unique sequences.

Produces memory-mapped embedding stores optimised for fast training lookup:

    data/embeddings_cache/
        mirna_embeddings.npy    # memmap [N_mirna,  embed_dim] float16
        mirna_metadata.pt       # {"sequences", "seq_to_idx", "embed_dim", ...}
        target_embeddings.npy   # memmap [N_target, embed_dim] float16
        target_metadata.pt      # {"sequences", "seq_to_idx", "embed_dim", ...}

Embeddings are **mean-pooled** over valid tokens (attention-mask aware).
This keeps storage manageable (~3 GB for 1.27M targets) while still
capturing the rich contextual representation from RiNALMo-giga (650M).

During training the Dataset opens these memmaps in read-only mode and
performs O(1) lookups per sample, avoiding any backbone forward pass.

Usage:
    python scripts/precompute_embeddings.py \\
        --data-dir data/processed \\
        --cache-dir data/embeddings_cache \\
        --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Patch transformers compatibility BEFORE any multimolecule import
from deepexomir.utils.compat import patch_multimolecule_compat  # noqa: E402
patch_multimolecule_compat()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-compute mean-pooled RNA foundation model embeddings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data/processed"),
        help="Directory containing train.parquet, val.parquet, test.parquet.",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("data/embeddings_cache"),
        help="Directory to store pre-computed embedding files.",
    )
    parser.add_argument(
        "--model-name", type=str, default="multimolecule/rinalmo-giga",
        help="HuggingFace model identifier for the RNA backbone.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batch size for embedding extraction (reduce if OOM).",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device to use (default: auto-detect cuda/cpu).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def collect_unique_sequences(data_dir: Path) -> tuple[list[str], list[str]]:
    """Collect unique miRNA and target sequences from all parquet splits."""
    import pandas as pd

    mirna_seqs: set[str] = set()
    target_seqs: set[str] = set()

    for split in ["train", "val", "test"]:
        path = data_dir / f"{split}.parquet"
        if not path.exists():
            logger.warning("File not found, skipping: %s", path)
            continue

        df = pd.read_parquet(path, engine="pyarrow")
        mirna_col = df["mirna_seq"].dropna().astype(str)
        target_col = df["target_seq"].dropna().astype(str)
        mirna_seqs.update(mirna_col[mirna_col.str.len() > 0].unique())
        target_seqs.update(target_col[target_col.str.len() > 0].unique())
        print(f"  Loaded {split}: {len(df):,} samples")

    return sorted(mirna_seqs), sorted(target_seqs)


def mean_pool(
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Attention-mask–aware mean pooling.

    Parameters
    ----------
    hidden_states : Tensor [B, L, D]
    attention_mask : Tensor [B, L]   (1 = valid, 0 = pad)

    Returns
    -------
    Tensor [B, D]
    """
    mask = attention_mask.unsqueeze(-1).float()         # [B, L, 1]
    summed = (hidden_states * mask).sum(dim=1)           # [B, D]
    lengths = mask.sum(dim=1).clamp(min=1.0)             # [B, 1]
    return summed / lengths


def compute_pertoken_embedding_store(
    backbone_model: torch.nn.Module,
    tokenizer,
    sequences: list[str],
    embed_dim: int,
    max_seq_len: int,
    batch_size: int,
    device: str,
    cache_dir: Path,
    prefix: str,
    label: str,
) -> None:
    """Process sequences through backbone, save per-token embeddings + masks.

    Output files:
        {prefix}_pertoken_embeddings.npy  — memmap [N, max_seq_len, embed_dim] float16
        {prefix}_pertoken_masks.npy       — memmap [N, max_seq_len] bool (True=valid)
        {prefix}_pertoken_metadata.pt     — dict with seq_to_idx, etc.
    """
    N = len(sequences)
    if N == 0:
        return

    emb_path = cache_dir / f"{prefix}_pertoken_embeddings.npy"
    mask_path = cache_dir / f"{prefix}_pertoken_masks.npy"
    meta_path = cache_dir / f"{prefix}_pertoken_metadata.pt"

    # Create memory-mapped arrays
    emb_mmap = np.memmap(
        emb_path, dtype=np.float16, mode="w+",
        shape=(N, max_seq_len, embed_dim),
    )
    mask_mmap = np.memmap(
        mask_path, dtype=np.bool_, mode="w+",
        shape=(N, max_seq_len),
    )

    start_time = time.time()
    processed = 0

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_seqs = sequences[batch_start:batch_end]

        # Tokenize — max_length accounts for special tokens (BOS/EOS)
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len + 10,  # extra room for special tokens
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=(device == "cuda"),
            ):
                outputs = backbone_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
            hidden = outputs.last_hidden_state  # [B, L_tok, D]

        # Strip special tokens (typically BOS at pos 0 and EOS at end).
        # The tokenizer adds special tokens, so nucleotide embeddings
        # start at position 1 and end before the EOS token.
        # We extract positions [1 : 1+max_seq_len].
        B_cur = hidden.shape[0]
        nuc_hidden = hidden[:, 1:, :]           # drop BOS token
        nuc_mask = attention_mask[:, 1:]         # drop BOS mask

        # Trim to max_seq_len
        L_avail = nuc_hidden.shape[1]
        if L_avail > max_seq_len:
            nuc_hidden = nuc_hidden[:, :max_seq_len, :]
            nuc_mask = nuc_mask[:, :max_seq_len]

        # Pad if shorter (shouldn't happen for miRNAs but safety)
        if L_avail < max_seq_len:
            pad_len = max_seq_len - L_avail
            nuc_hidden = torch.nn.functional.pad(nuc_hidden, (0, 0, 0, pad_len))
            nuc_mask = torch.nn.functional.pad(nuc_mask, (0, pad_len))

        # Also mask out the EOS token if it falls within the window.
        # EOS tokens have mask=1 from the tokenizer, but they are not
        # nucleotide positions. We detect them by checking if the
        # original sequence length is shorter than the available tokens.
        for i in range(B_cur):
            seq_len = len(batch_seqs[i])
            if seq_len < max_seq_len:
                nuc_mask[i, seq_len:] = 0  # mask out EOS and beyond

        # Write to memmaps
        emb_mmap[batch_start:batch_end] = (
            nuc_hidden.cpu().to(torch.float16).numpy()
        )
        mask_mmap[batch_start:batch_end] = nuc_mask.cpu().bool().numpy()

        processed += len(batch_seqs)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0

        if processed % (batch_size * 10) == 0 or processed == N:
            pct = 100 * processed / N
            print(
                f"  [{label} per-token] {processed:>7,}/{N:,} "
                f"({pct:5.1f}%) | {rate:.1f} seq/s"
            )

    emb_mmap.flush()
    mask_mmap.flush()
    del emb_mmap, mask_mmap

    seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}
    metadata = {
        "sequences": sequences,
        "seq_to_idx": seq_to_idx,
        "embed_dim": embed_dim,
        "max_seq_len": max_seq_len,
        "n_sequences": N,
        "dtype": "float16",
        "mode": "per_token",
    }
    torch.save(metadata, meta_path)

    elapsed = time.time() - start_time
    emb_size_mb = emb_path.stat().st_size / (1024 * 1024)
    print(f"  [{label} per-token] Complete in {elapsed:.1f}s")
    print(f"  [{label} per-token] Saved: {emb_path.name} ({emb_size_mb:,.1f} MB)")


def compute_embedding_store(
    backbone_model: torch.nn.Module,
    tokenizer,
    sequences: list[str],
    embed_dim: int,
    batch_size: int,
    device: str,
    cache_dir: Path,
    prefix: str,   # "mirna" or "target"
    label: str,    # display label
) -> None:
    """Process sequences through the backbone, mean-pool, write memmap."""
    N = len(sequences)
    if N == 0:
        return

    emb_path = cache_dir / f"{prefix}_embeddings.npy"
    meta_path = cache_dir / f"{prefix}_metadata.pt"

    # Create memory-mapped array  [N, embed_dim]  float16
    emb_mmap = np.memmap(
        emb_path, dtype=np.float16, mode="w+", shape=(N, embed_dim),
    )

    start_time = time.time()
    processed = 0

    for batch_start in range(0, N, batch_size):
        batch_end = min(batch_start + batch_size, N)
        batch_seqs = sequences[batch_start:batch_end]

        # Tokenize
        encoded = tokenizer(
            batch_seqs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward through frozen backbone
        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda", enabled=(device == "cuda"),
            ):
                outputs = backbone_model(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
            hidden = outputs.last_hidden_state  # [B, L, D]

        # Mean-pool over valid tokens → [B, D]
        pooled = mean_pool(hidden, attention_mask)

        # Write to memmap
        emb_mmap[batch_start:batch_end] = pooled.cpu().to(torch.float16).numpy()

        processed += len(batch_seqs)
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0

        if processed % (batch_size * 20) == 0 or processed == N:
            pct = 100 * processed / N
            print(
                f"  [{label}] {processed:>7,}/{N:,} "
                f"({pct:5.1f}%) "
                f"| {rate:.1f} seq/s"
            )

    # Flush memmap to disk
    emb_mmap.flush()
    del emb_mmap

    # Build index: sequence string -> row index
    seq_to_idx = {seq: idx for idx, seq in enumerate(sequences)}

    # Save metadata
    metadata = {
        "sequences": sequences,
        "seq_to_idx": seq_to_idx,
        "embed_dim": embed_dim,
        "n_sequences": N,
        "dtype": "float16",
        "pool": "mean",
    }
    torch.save(metadata, meta_path)

    elapsed = time.time() - start_time
    emb_size_mb = emb_path.stat().st_size / (1024 * 1024)
    print(f"  [{label}] Complete in {elapsed:.1f}s")
    print(f"  [{label}] Saved: {emb_path.name} ({emb_size_mb:,.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import multimolecule  # noqa: F401  — registers model types with Auto classes
    from transformers import AutoModel, AutoTokenizer

    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    print("DeepExoMir Embedding Pre-computation (mean-pooled)")
    print("=" * 60)
    print(f"Data directory  : {args.data_dir.resolve()}")
    print(f"Cache directory : {args.cache_dir.resolve()}")
    print(f"Backbone model  : {args.model_name}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Pooling         : mean (attention-mask aware)")
    print(f"Storage dtype   : float16")
    print(f"Device          : {device}")
    print()

    # ---- Collect unique sequences ----------------------------------------
    print("Collecting unique sequences ...")
    mirna_seqs, target_seqs = collect_unique_sequences(args.data_dir)
    n_mirna = len(mirna_seqs)
    n_target = len(target_seqs)
    print(f"\nUnique miRNA sequences : {n_mirna:,}")
    print(f"Unique target sequences: {n_target:,}")
    print()

    if n_mirna == 0 and n_target == 0:
        print("No sequences found.  Exiting.")
        return

    # ---- Load backbone model (directly via HuggingFace) ------------------
    print(f"Loading backbone model: {args.model_name} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    model = AutoModel.from_pretrained(
        args.model_name, trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    embed_dim = model.config.hidden_size
    print(f"Backbone loaded in {time.time() - t0:.1f}s: embed_dim={embed_dim}")
    print()

    # ---- Create cache directory ------------------------------------------
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---- Process miRNA sequences -----------------------------------------
    if mirna_seqs:
        # Mean-pooled embeddings (for classifier-level features)
        print(f"Processing {n_mirna:,} unique miRNA sequences (mean-pooled) ...")
        compute_embedding_store(
            backbone_model=model,
            tokenizer=tokenizer,
            sequences=mirna_seqs,
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            device=device,
            cache_dir=args.cache_dir,
            prefix="mirna",
            label="miRNA",
        )
        print()

        # Per-token embeddings (for cross-attention — only feasible for miRNAs)
        max_mirna_len = 30
        pertoken_mb = n_mirna * max_mirna_len * embed_dim * 2 / (1024 * 1024)
        print(
            f"Processing {n_mirna:,} unique miRNA sequences (per-token, "
            f"max_len={max_mirna_len}, ~{pertoken_mb:.0f} MB) ..."
        )
        compute_pertoken_embedding_store(
            backbone_model=model,
            tokenizer=tokenizer,
            sequences=mirna_seqs,
            embed_dim=embed_dim,
            max_seq_len=max_mirna_len,
            batch_size=args.batch_size,
            device=device,
            cache_dir=args.cache_dir,
            prefix="mirna",
            label="miRNA",
        )
        print()

    # ---- Process target sequences ----------------------------------------
    if target_seqs:
        est_mb = n_target * embed_dim * 2 / (1024 * 1024)
        n_batches = (n_target + args.batch_size - 1) // args.batch_size
        print(
            f"Processing {n_target:,} unique target sequences "
            f"(~{est_mb:,.0f} MB, {n_batches:,} batches) ..."
        )
        compute_embedding_store(
            backbone_model=model,
            tokenizer=tokenizer,
            sequences=target_seqs,
            embed_dim=embed_dim,
            batch_size=args.batch_size,
            device=device,
            cache_dir=args.cache_dir,
            prefix="target",
            label="target",
        )
        print()

    # ---- Summary ---------------------------------------------------------
    print("Pre-computation complete!")
    print("-" * 60)
    cache_files = list(args.cache_dir.glob("*"))
    total_size_mb = sum(
        f.stat().st_size for f in cache_files if f.is_file()
    ) / (1024 * 1024)
    print(f"Total unique sequences: {n_mirna + n_target:,}")
    print(f"Total cache size      : {total_size_mb:,.1f} MB")
    print(f"Cache directory       : {args.cache_dir.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
