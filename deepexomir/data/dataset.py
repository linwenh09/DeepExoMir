"""PyTorch Dataset and DataLoader utilities for DeepExoMir.

Provides a map-style Dataset that loads miRNA-target interaction data
from Parquet files and returns tensors suitable for the DeepExoMir model.

Each sample yields:
    - mirna_seq:             str  (raw RNA sequence)
    - target_seq:            str  (raw RNA sequence)
    - base_pairing_matrix:   Tensor [1, 30, 50]
    - structural_features:   Tensor [8]   (v7: expanded from 6)
    - label:                 int  (1 = interacting, 0 = non-interacting)

Optionally (when ``embeddings_dir`` is provided):
    - mirna_pooled_emb:      Tensor [embed_dim]   (mean-pooled backbone embedding)
    - target_pooled_emb:     Tensor [embed_dim]   (mean-pooled backbone embedding)

Model ⑦ enhancements:
    - Supports pre-computed structural features (.npy files)
    - 8 structural features (added au_content, seed_duplex_mfe)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from deepexomir.data.features import (
    compute_base_pairing_matrix,
    compute_structural_features,
)

logger = logging.getLogger(__name__)

# Feature names in the order they appear in the structural_features tensor
# Model ⑦: expanded from 6 to 8 features (added au_content, seed_duplex_mfe)
STRUCTURAL_FEATURE_NAMES = [
    "duplex_mfe",
    "mirna_mfe",
    "target_mfe",
    "accessibility",
    "gc_content",
    "seed_match_type",
    "au_content",                # v7: AU fraction of target site
    "seed_duplex_mfe",           # v7: seed region duplex MFE
    "plfold_seed_accessibility", # v10+: RNAplfold unpaired prob at seed site
    "plfold_site_accessibility", # v10+: RNAplfold mean unpaired prob (whole site)
    "supp_3prime_score",         # v10+: 3' supplementary pairing (miRNA 13-16)
    "local_au_flanking",         # v10+: AU content of upstream flanking region
    "seed_pairing_stability",    # v11: nearest-neighbor stacking SPS
    "comp_3prime_score",         # v11: 3' compensatory pairing (miRNA 17-21)
    "central_pairing",           # v11: central region pairing (miRNA 9-12)
    "mfe_ratio",                 # v11: duplex_mfe / (mirna_mfe + target_mfe)
    "wobble_count",              # v11: G:U wobble pairs in full duplex
    "longest_contiguous",        # v11: longest contiguous complementary stretch
    "mismatch_count",            # v11: total mismatches in duplex
    "seed_gc_content",           # v11: GC fraction of seed duplex region
    "sRNA1_A",                   # v12: miRNA pos-1 is A (one-hot)
    "sRNA1_C",                   # v12: miRNA pos-1 is C (one-hot)
    "sRNA1_G",                   # v12: miRNA pos-1 is G (one-hot)
    "sRNA8_A",                   # v12: miRNA pos-8 is A (one-hot)
    "sRNA8_C",                   # v12: miRNA pos-8 is C (one-hot)
    "sRNA8_G",                   # v12: miRNA pos-8 is G (one-hot)
    "site8_A",                   # v12: target A1 anchor opposite pos-8
    "flanking_dinuc_score",      # v12: AU-richness of seed flanking dinucs
    "dG_open",                   # v13: energy cost to unfold target at binding site
    "dG_total",                  # v13: dG_duplex + dG_open (net binding energy)
    "ensemble_dG",               # v13: ensemble free energy from partition function
    "acc_5nt_up",                # v13: accessibility 5nt upstream of seed
    "acc_10nt_up",               # v13: accessibility 10nt upstream of seed
    "acc_15nt_up",               # v13: accessibility 15nt upstream of seed
    "phylop_mean",               # v14: mean PhyloP conservation over target site
    "phylop_max",                # v14: max PhyloP conservation score
    "phylop_seed_mean",          # v14: mean PhyloP conservation over seed region
    "site_in_3utr",              # v14: 1.0 if site is in 3'UTR
    "site_in_cds",               # v14: 1.0 if site is in CDS/exon
    "phastcons_mean",            # v15: mean PhastCons conservation over target site
    "phastcons_max",             # v15: max PhastCons conservation score
    "phastcons_seed_mean",       # v15: mean PhastCons over seed region
    "gerp_mean",                 # v15: mean GERP++ RS score over target site
    "gerp_max",                  # v15: max GERP++ RS score
    "gerp_seed_mean",            # v15: mean GERP++ RS over seed region
]

# Default sequence length limits (must match model expectations)
MAX_MIRNA_LEN = 30
MAX_TARGET_LEN = 50


# ============================================================================
# Embedding Store Loader (mean-pooled)
# ============================================================================

class PooledEmbeddingStore:
    """Memory-mapped mean-pooled embedding store.

    Wraps a numpy memmap of shape ``[N, embed_dim]`` and a sequence-to-index
    dictionary produced by ``scripts/precompute_embeddings.py``.

    Parameters
    ----------
    cache_dir : Path
        Directory containing ``{prefix}_embeddings.npy`` and
        ``{prefix}_metadata.pt``.
    prefix : str
        File prefix – ``"mirna"`` or ``"target"``.
    """

    def __init__(self, cache_dir: Path, prefix: str) -> None:
        meta_path = cache_dir / f"{prefix}_metadata.pt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Embedding metadata not found: {meta_path}.  "
                f"Run scripts/precompute_embeddings.py first."
            )

        metadata = torch.load(meta_path, map_location="cpu", weights_only=False)
        self.seq_to_idx: dict[str, int] = metadata["seq_to_idx"]
        self.embed_dim: int = metadata["embed_dim"]
        n = metadata["n_sequences"]

        emb_path = cache_dir / f"{prefix}_embeddings.npy"

        # Open as read-only memory-mapped array
        self.embeddings = np.memmap(
            emb_path, dtype=np.float16, mode="r", shape=(n, self.embed_dim),
        )
        logger.info(
            "Loaded %s embedding store: %d sequences, dim=%d (mean-pooled)",
            prefix, n, self.embed_dim,
        )

    def lookup(self, sequence: str) -> torch.Tensor:
        """Return the mean-pooled embedding for a single sequence.

        Returns
        -------
        Tensor [embed_dim]  (float32)
        """
        idx = self.seq_to_idx.get(sequence)
        if idx is None:
            raise KeyError(
                f"Sequence not found in embedding store: {sequence!r:.60s}"
            )
        # .copy() is required for thread-safety with memmap + DataLoader workers
        return torch.from_numpy(self.embeddings[idx].copy()).float()


# ============================================================================
# Embedding Store Loader (per-token)
# ============================================================================

class PerTokenEmbeddingStore:
    """Memory-mapped per-token embedding store.

    Wraps a numpy memmap of shape ``[N, max_seq_len, embed_dim]`` and a
    corresponding mask ``[N, max_seq_len]`` (True = valid token).

    Parameters
    ----------
    cache_dir : Path
        Directory containing ``{prefix}_pertoken_embeddings.npy``,
        ``{prefix}_pertoken_masks.npy``, and
        ``{prefix}_pertoken_metadata.pt``.
    prefix : str
        File prefix – ``"mirna"`` or ``"target"``.
    """

    def __init__(self, cache_dir: Path, prefix: str) -> None:
        meta_path = cache_dir / f"{prefix}_pertoken_metadata.pt"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Per-token embedding metadata not found: {meta_path}.  "
                f"Run scripts/precompute_embeddings.py first."
            )

        metadata = torch.load(meta_path, map_location="cpu", weights_only=False)
        self.seq_to_idx: dict[str, int] = metadata["seq_to_idx"]
        self.embed_dim: int = metadata["embed_dim"]
        self.max_seq_len: int = metadata["max_seq_len"]
        n = metadata["n_sequences"]

        emb_path = cache_dir / f"{prefix}_pertoken_embeddings.npy"
        mask_path = cache_dir / f"{prefix}_pertoken_masks.npy"

        self.embeddings = np.memmap(
            emb_path, dtype=np.float16, mode="r",
            shape=(n, self.max_seq_len, self.embed_dim),
        )
        self.masks = np.memmap(
            mask_path, dtype=np.bool_, mode="r",
            shape=(n, self.max_seq_len),
        )
        logger.info(
            "Loaded %s per-token embedding store: %d sequences, "
            "max_len=%d, dim=%d",
            prefix, n, self.max_seq_len, self.embed_dim,
        )

    def lookup(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-token embeddings and mask for a single sequence.

        Returns
        -------
        emb : Tensor [max_seq_len, embed_dim]  (float32)
        mask : Tensor [max_seq_len]  (bool, True = valid)
        """
        idx = self.seq_to_idx.get(sequence)
        if idx is None:
            raise KeyError(
                f"Sequence not found in per-token store: {sequence!r:.60s}"
            )
        emb = torch.from_numpy(self.embeddings[idx].copy()).float()
        mask = torch.from_numpy(self.masks[idx].copy())
        return emb, mask


# ============================================================================
# Dataset
# ============================================================================

class MiRNATargetDataset(Dataset):
    """PyTorch Dataset for miRNA-target interaction prediction.

    Parameters
    ----------
    parquet_path : Path or str
        Path to a Parquet file containing at minimum ``mirna_seq``,
        ``target_seq``, and ``label`` columns.
    max_mirna_len : int
        Maximum miRNA sequence length (sequences are padded/truncated).
    max_target_len : int
        Maximum target-site sequence length.
    cache_dir : Path or str, optional
        If provided, pre-computed features are cached to this directory.
        Cache keys are derived from sequence content hashes.
    precompute : bool
        If True (and ``cache_dir`` is set), compute and cache all features
        on initialisation.  Otherwise features are computed on-the-fly.
    embeddings_dir : Path or str, optional
        Directory containing pre-computed mean-pooled RiNALMo embeddings.
        When set, each sample additionally returns ``mirna_pooled_emb``
        and ``target_pooled_emb`` tensors.
    """

    def __init__(
        self,
        parquet_path: Path | str,
        max_mirna_len: int = MAX_MIRNA_LEN,
        max_target_len: int = MAX_TARGET_LEN,
        cache_dir: Optional[Path | str] = None,
        precompute: bool = False,
        skip_structural: bool = False,
        skip_bp_matrix: bool = False,
        embeddings_dir: Optional[Path | str] = None,
        feature_version: Optional[str] = None,
        signal_ablation: Optional[dict] = None,
    ) -> None:
        """
        signal_ablation: optional dict for retrain-from-scratch ablation studies.
            Supported keys:
              'rnalm'        : if True, zeros miRNA and target pooled / per-token
                               embeddings BEFORE returning the sample.
              'conservation' : if True, zeros the 5 PhyloP-related indices in the
                               structural_features vector (positions identified by
                               STRUCTURAL_FEATURE_NAMES index lookup).
              'structure'    : if True, zeros the base_pairing_matrix AND zeros the
                               28 non-PhyloP structural-feature positions.
            Each flag independently simulates a "retrain without signal X" condition
            at data-loading time. Use for Phase-2.3 retrain-from-scratch ablations.
        """
        super().__init__()
        self.parquet_path = Path(parquet_path)
        self.max_mirna_len = max_mirna_len
        self.max_target_len = max_target_len

        # Signal ablation (for retrain-from-scratch ablation studies)
        self.signal_ablation = signal_ablation or {}
        self._ablate_rnalm = bool(self.signal_ablation.get("rnalm", False))
        self._ablate_conservation = bool(self.signal_ablation.get("conservation", False))
        self._ablate_structure = bool(self.signal_ablation.get("structure", False))

        # Pre-compute PhyloP index positions (for conservation ablation)
        # These features are zeroed when ablate_conservation = True
        _PHYLOP_NAMES = {"phylop_mean", "phylop_max", "phylop_seed_mean",
                          "site_in_3utr", "site_in_cds"}
        self._phylop_indices = [i for i, n in enumerate(STRUCTURAL_FEATURE_NAMES)
                                 if n in _PHYLOP_NAMES]
        self._non_phylop_indices = [i for i, n in enumerate(STRUCTURAL_FEATURE_NAMES)
                                     if n not in _PHYLOP_NAMES]
        if any([self._ablate_rnalm, self._ablate_conservation, self._ablate_structure]):
            logger.info(
                "SIGNAL ABLATION ACTIVE: rnalm=%s, conservation=%s, structure=%s "
                "(phylop indices to zero: %s)",
                self._ablate_rnalm, self._ablate_conservation, self._ablate_structure,
                self._phylop_indices,
            )

        # Load data
        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {self.parquet_path}"
            )

        self.df = pd.read_parquet(self.parquet_path, engine="pyarrow")
        self._validate_columns()

        self.skip_structural = skip_structural
        self.skip_bp_matrix = skip_bp_matrix

        # ---- Pre-computed structural features (.npy from precompute script) --
        self._precomputed_struct: np.ndarray | None = None
        # Prefer newest version: v14 > v13 > v12 > v11 > v10 > original
        _stem = self.parquet_path.stem
        _parent = self.parquet_path.parent
        struct_npy_path = _parent / f"{_stem}_structural_features.npy"
        if feature_version and not skip_structural:
            # Explicit version requested (e.g. "v14alt2L", "v16a", "v16c")
            candidate = _parent / f"{_stem}_structural_features_{feature_version}.npy"
            if candidate.exists():
                struct_npy_path = candidate
            else:
                logger.warning("Requested feature_version=%s not found: %s", feature_version, candidate)
        else:
            for ver in ["v18", "v16c", "v16a", "v16b", "v15", "v14alt2L", "v13alt2L", "v14", "v13", "v12", "v11", "v10"]:
                candidate = _parent / f"{_stem}_structural_features_{ver}.npy"
                if candidate.exists() and not skip_structural:
                    struct_npy_path = candidate
                    break
        if struct_npy_path.exists() and not skip_structural:
            self._precomputed_struct = np.load(struct_npy_path).astype(np.float32)
            logger.info(
                "Loaded pre-computed structural features: %s (shape=%s)",
                struct_npy_path.name, self._precomputed_struct.shape,
            )
            # Verify feature count matches
            n_expected = len(STRUCTURAL_FEATURE_NAMES)
            n_loaded = self._precomputed_struct.shape[1]
            if n_loaded < n_expected:
                # Pad with zeros for backward compatibility (v6 → v7 transition)
                logger.warning(
                    "Pre-computed features have %d columns, expected %d. "
                    "Padding with zeros for missing features.",
                    n_loaded, n_expected,
                )
                pad = np.zeros(
                    (self._precomputed_struct.shape[0], n_expected - n_loaded),
                    dtype=np.float32,
                )
                self._precomputed_struct = np.concatenate(
                    [self._precomputed_struct, pad], axis=1,
                )

        # Fill any NaN sequences with empty strings
        self.df["mirna_seq"] = self.df["mirna_seq"].fillna("").astype(str)
        self.df["target_seq"] = self.df["target_seq"].fillna("").astype(str)
        self.df["label"] = self.df["label"].astype(int)

        logger.info(
            "Loaded dataset from %s: %d samples (pos=%d, neg=%d).",
            self.parquet_path.name,
            len(self.df),
            (self.df["label"] == 1).sum(),
            (self.df["label"] == 0).sum(),
        )

        # ---- Pre-computed embedding stores (optional) --------------------
        self._mirna_store: Optional[PooledEmbeddingStore] = None
        self._target_store: Optional[PooledEmbeddingStore] = None
        self._mirna_pertoken_store: Optional[PerTokenEmbeddingStore] = None
        self._target_pertoken_store: Optional[PerTokenEmbeddingStore] = None
        self.use_precomputed_embeddings = False
        self.use_pertoken_mirna = False
        self.use_pertoken_target = False

        if embeddings_dir is not None:
            emb_dir = Path(embeddings_dir)
            # Mean-pooled stores (always loaded when embeddings_dir given)
            self._mirna_store = PooledEmbeddingStore(emb_dir, "mirna")
            self._target_store = PooledEmbeddingStore(emb_dir, "target")
            self.use_precomputed_embeddings = True

            # Per-token miRNA store (loaded if available)
            pertoken_meta = emb_dir / "mirna_pertoken_metadata.pt"
            if pertoken_meta.exists():
                self._mirna_pertoken_store = PerTokenEmbeddingStore(
                    emb_dir, "mirna",
                )
                self.use_pertoken_mirna = True

            # Per-token target store (v8: loaded if available)
            target_pertoken_meta = emb_dir / "target_pertoken_metadata.pt"
            if target_pertoken_meta.exists():
                self._target_pertoken_store = PerTokenEmbeddingStore(
                    emb_dir, "target",
                )
                self.use_pertoken_target = True

            logger.info(
                "Pre-computed embeddings enabled from %s "
                "(mirna_dim=%d, target_dim=%d, mirna_pertoken=%s, target_pertoken=%s).",
                emb_dir,
                self._mirna_store.embed_dim,
                self._target_store.embed_dim,
                self.use_pertoken_mirna,
                self.use_pertoken_target,
            )

        # ---- Structural feature cache ------------------------------------
        self.cache_dir: Optional[Path] = None
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory cache for structural features (populated lazily or eagerly)
        self._feature_cache: dict[int, dict[str, float | int]] = {}

        if precompute and self.cache_dir is not None:
            self._precompute_all()

    def _validate_columns(self) -> None:
        """Ensure required columns are present."""
        required = {"mirna_seq", "target_seq", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(
                f"Parquet file is missing required columns: {missing}. "
                f"Available: {list(self.df.columns)}"
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a single sample as a dictionary.

        Returns
        -------
        dict
            Always present:
            - ``mirna_seq``: str
            - ``target_seq``: str
            - ``label``: int

            When pre-computed embeddings are available:
            - ``mirna_pooled_emb``: Tensor [embed_dim]
            - ``target_pooled_emb``: Tensor [embed_dim]

            Otherwise/additionally:
            - ``base_pairing_matrix``: Tensor [1, max_mirna_len, max_target_len]
            - ``structural_features``: Tensor [6]
        """
        row = self.df.iloc[idx]
        mirna_seq: str = row["mirna_seq"]
        target_seq: str = row["target_seq"]
        label: int = int(row["label"])

        result: dict[str, Any] = {
            "mirna_seq": mirna_seq,
            "target_seq": target_seq,
            "label": label,
        }

        # ---- Pre-computed mean-pooled backbone embeddings ----------------
        if self.use_precomputed_embeddings:
            result["mirna_pooled_emb"] = self._mirna_store.lookup(mirna_seq)
            result["target_pooled_emb"] = self._target_store.lookup(target_seq)

        # ---- Per-token miRNA backbone embeddings --------------------------
        if self.use_pertoken_mirna:
            emb, mask = self._mirna_pertoken_store.lookup(mirna_seq)
            result["mirna_pertoken_emb"] = emb       # [max_mirna_len, D]
            result["mirna_pertoken_mask"] = mask      # [max_mirna_len] bool

        # ---- Per-token target backbone embeddings (v8) -------------------
        if self.use_pertoken_target:
            emb, mask = self._target_pertoken_store.lookup(target_seq)
            result["target_pertoken_emb"] = emb      # [max_target_len, D]
            result["target_pertoken_mask"] = mask     # [max_target_len] bool

        # ---- Base-pairing matrix (skip if model computes on GPU) ---------
        if not self.skip_bp_matrix:
            bp_matrix = self._get_bp_matrix(idx, mirna_seq, target_seq)
            result["base_pairing_matrix"] = torch.from_numpy(bp_matrix).unsqueeze(0)

        # ---- Structural features -------------------------------------------
        if self._precomputed_struct is not None:
            # Use pre-computed features (fast path, v7)
            result["structural_features"] = torch.from_numpy(
                self._precomputed_struct[idx].copy()
            )
        elif self.skip_structural:
            result["structural_features"] = torch.zeros(
                len(STRUCTURAL_FEATURE_NAMES), dtype=torch.float32
            )
        else:
            feats = self._get_structural_features(idx, mirna_seq, target_seq)
            feat_values = [float(feats[k]) for k in STRUCTURAL_FEATURE_NAMES]
            result["structural_features"] = torch.tensor(feat_values, dtype=torch.float32)

        # ---- Signal ablation (for Phase 2.3 retrain-from-scratch studies) --
        if self._ablate_rnalm:
            # Zero RiNALMo-derived embeddings (pooled + per-token if present)
            if "mirna_pooled_emb" in result:
                result["mirna_pooled_emb"] = torch.zeros_like(result["mirna_pooled_emb"])
            if "target_pooled_emb" in result:
                result["target_pooled_emb"] = torch.zeros_like(result["target_pooled_emb"])
            if "mirna_pertoken_emb" in result:
                result["mirna_pertoken_emb"] = torch.zeros_like(result["mirna_pertoken_emb"])
            if "target_pertoken_emb" in result:
                result["target_pertoken_emb"] = torch.zeros_like(result["target_pertoken_emb"])
        if self._ablate_conservation:
            # Zero the 5 PhyloP-related positions in structural_features
            sf = result["structural_features"].clone()
            for i in self._phylop_indices:
                if i < sf.shape[0]:
                    sf[i] = 0.0
            result["structural_features"] = sf
        if self._ablate_structure:
            # Zero base-pairing matrix AND zero non-PhyloP structural features
            if "base_pairing_matrix" in result:
                result["base_pairing_matrix"] = torch.zeros_like(result["base_pairing_matrix"])
            sf = result["structural_features"].clone()
            for i in self._non_phylop_indices:
                if i < sf.shape[0]:
                    sf[i] = 0.0
            result["structural_features"] = sf

        return result

    # ------------------------------------------------------------------
    # Feature access with caching
    # ------------------------------------------------------------------

    def _cache_key(self, mirna_seq: str, target_seq: str) -> str:
        """Generate a deterministic cache key from sequences."""
        raw = f"{mirna_seq}|{target_seq}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _get_bp_matrix(
        self, idx: int, mirna_seq: str, target_seq: str
    ) -> np.ndarray:
        """Compute base-pairing matrix (vectorized, fast enough to skip caching)."""
        return compute_base_pairing_matrix(
            mirna_seq, target_seq, self.max_mirna_len, self.max_target_len
        )

    def _get_structural_features(
        self, idx: int, mirna_seq: str, target_seq: str
    ) -> dict[str, float | int]:
        """Get structural features, from cache or freshly computed."""
        if idx in self._feature_cache:
            return self._feature_cache[idx]

        # Try disk cache
        if self.cache_dir is not None:
            key = self._cache_key(mirna_seq, target_seq)
            cache_path = self.cache_dir / f"feat_{key}.npz"
            if cache_path.exists():
                data = np.load(cache_path, allow_pickle=True)
                feats = dict(data["features"].item())
                self._feature_cache[idx] = feats
                return feats

        feats = compute_structural_features(mirna_seq, target_seq)
        self._feature_cache[idx] = feats

        # Save to disk cache
        if self.cache_dir is not None:
            key = self._cache_key(mirna_seq, target_seq)
            np.savez(
                self.cache_dir / f"feat_{key}.npz",
                features=feats,
            )

        return feats

    def _precompute_all(self) -> None:
        """Pre-compute and cache structural features (can be slow for large datasets).

        Note: BP matrices are not cached since vectorized computation is fast.
        """
        logger.info("Pre-computing structural features for %d samples ...", len(self.df))
        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            self._get_structural_features(idx, row["mirna_seq"], row["target_seq"])
            if (idx + 1) % 5000 == 0:
                logger.info("  Pre-computed %d / %d", idx + 1, len(self.df))
        logger.info("Pre-computation complete.")


# ============================================================================
# Collate function
# ============================================================================


def mirna_target_collate_fn(
    batch: list[dict[str, Any]],
) -> tuple[dict[str, Any], torch.Tensor]:
    """Custom collate function for :class:`MiRNATargetDataset`.

    Stacks tensors and collects string sequences into lists.

    Parameters
    ----------
    batch : list[dict]
        List of sample dicts from :meth:`MiRNATargetDataset.__getitem__`.

    Returns
    -------
    tuple[dict, Tensor]
        - Batched feature dictionary
        - Labels tensor ``[B]`` (long)
    """
    mirna_seqs = [sample["mirna_seq"] for sample in batch]
    target_seqs = [sample["target_seq"] for sample in batch]
    labels = torch.tensor(
        [sample["label"] for sample in batch], dtype=torch.long
    )

    features: dict[str, Any] = {
        "mirna_seq": mirna_seqs,
        "target_seq": target_seqs,
    }

    # Pre-computed mean-pooled backbone embeddings
    if "mirna_pooled_emb" in batch[0]:
        features["mirna_pooled_emb"] = torch.stack(
            [sample["mirna_pooled_emb"] for sample in batch], dim=0
        )
        features["target_pooled_emb"] = torch.stack(
            [sample["target_pooled_emb"] for sample in batch], dim=0
        )

    # Per-token miRNA backbone embeddings (for cross-attention)
    if "mirna_pertoken_emb" in batch[0]:
        features["mirna_pertoken_emb"] = torch.stack(
            [sample["mirna_pertoken_emb"] for sample in batch], dim=0
        )  # [B, max_mirna_len, backbone_dim]
        features["mirna_pertoken_mask"] = torch.stack(
            [sample["mirna_pertoken_mask"] for sample in batch], dim=0
        )  # [B, max_mirna_len] bool

    # Per-token target backbone embeddings (v8: for cross-attention + GAT)
    if "target_pertoken_emb" in batch[0]:
        features["target_pertoken_emb"] = torch.stack(
            [sample["target_pertoken_emb"] for sample in batch], dim=0
        )  # [B, max_target_len, backbone_dim]
        features["target_pertoken_mask"] = torch.stack(
            [sample["target_pertoken_mask"] for sample in batch], dim=0
        )  # [B, max_target_len] bool

    # Optional tensors (may be skipped when model computes them on GPU)
    if "base_pairing_matrix" in batch[0]:
        features["base_pairing_matrix"] = torch.stack(
            [sample["base_pairing_matrix"] for sample in batch], dim=0
        )
    if "structural_features" in batch[0]:
        features["structural_features"] = torch.stack(
            [sample["structural_features"] for sample in batch], dim=0
        )

    return features, labels


# ============================================================================
# DataLoader factory
# ============================================================================


def create_dataloader(
    parquet_path: Path | str,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0,
    cache_dir: Optional[Path | str] = None,
    precompute: bool = False,
    skip_structural: bool = False,
    skip_bp_matrix: bool = False,
    embeddings_dir: Optional[Path | str] = None,
    feature_version: Optional[str] = None,
    signal_ablation: Optional[dict] = None,
    **kwargs: Any,
) -> DataLoader:
    """Create a DataLoader from a Parquet file.

    Convenience function that wraps :class:`MiRNATargetDataset` and
    :class:`torch.utils.data.DataLoader`.

    Parameters
    ----------
    parquet_path : Path or str
        Path to the Parquet dataset file.
    batch_size : int
        Batch size.
    shuffle : bool
        Whether to shuffle data each epoch.
    num_workers : int
        Number of data-loading worker processes.
    cache_dir : Path or str, optional
        Feature cache directory.
    precompute : bool
        Whether to pre-compute all features at init.
    embeddings_dir : Path or str, optional
        Directory containing pre-computed mean-pooled RiNALMo embeddings.
    **kwargs
        Additional keyword arguments passed to ``DataLoader``.

    Returns
    -------
    DataLoader
        Configured DataLoader instance.
    """
    dataset = MiRNATargetDataset(
        parquet_path=parquet_path,
        cache_dir=cache_dir,
        precompute=precompute,
        skip_structural=skip_structural,
        skip_bp_matrix=skip_bp_matrix,
        embeddings_dir=embeddings_dir,
        feature_version=feature_version,
        signal_ablation=signal_ablation,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mirna_target_collate_fn,
        pin_memory=torch.cuda.is_available(),
        **kwargs,
    )
