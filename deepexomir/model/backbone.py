"""RNA foundation model embedding extractor.

Loads a pre-trained RNA language model from HuggingFace and extracts
per-token embeddings for downstream cross-attention processing.  The backbone
is kept frozen throughout training; only the downstream projection layers and
cross-attention blocks are optimised.

Supported backbones (in priority order):
    1. RiNALMo-giga  -- multimolecule/rinalmo-giga  (650M params, 1280-dim)
    2. RNA-FM         -- multimolecule/rnafm          (99.5M params,  640-dim)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
import torch.nn as nn

# Patch transformers compat before any multimolecule import
from deepexomir.utils.compat import patch_multimolecule_compat
patch_multimolecule_compat()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone registry
# ---------------------------------------------------------------------------

_BACKBONE_REGISTRY: Dict[str, Dict[str, Union[str, int]]] = {
    "multimolecule/rinalmo-giga": {
        "hf_name": "multimolecule/rinalmo-giga",
        "embed_dim": 1280,
    },
    "multimolecule/rnafm": {
        "hf_name": "multimolecule/rnafm",
        "embed_dim": 640,
    },
}

_FALLBACK_ORDER = ["multimolecule/rinalmo-giga", "multimolecule/rnafm"]


def _load_backbone(
    model_name: str,
) -> Tuple[nn.Module, "AutoTokenizer", int]:  # type: ignore[name-defined]
    """Attempt to load *model_name* from HuggingFace; fall back if unavailable.

    Returns:
        (model, tokenizer, embed_dim)
    """
    import multimolecule  # noqa: F401  — registers model types with Auto classes
    from transformers import AutoModel, AutoTokenizer

    errors: List[str] = []
    names_to_try = (
        [model_name] if model_name in _BACKBONE_REGISTRY else list(_FALLBACK_ORDER)
    )

    for name in names_to_try:
        info = _BACKBONE_REGISTRY.get(name)
        if info is None:
            continue
        try:
            logger.info("Loading RNA backbone: %s ...", name)
            tokenizer = AutoTokenizer.from_pretrained(
                info["hf_name"], trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                info["hf_name"], trust_remote_code=True
            )
            logger.info(
                "Loaded %s  (embed_dim=%d)", name, info["embed_dim"]
            )
            return model, tokenizer, int(info["embed_dim"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc}")
            logger.warning("Failed to load %s: %s", name, exc)

    raise RuntimeError(
        "Could not load any RNA backbone model.  Tried:\n"
        + "\n".join(errors)
    )


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Simple disk-based tensor cache keyed by sequence content hash."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(sequences: List[str]) -> str:
        """Deterministic hash for a list of sequences."""
        content = "\n".join(sequences)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def get(self, sequences: List[str]) -> Optional[torch.Tensor]:
        path = self.cache_dir / f"{self._key(sequences)}.pt"
        if path.exists():
            return torch.load(path, map_location="cpu", weights_only=True)
        return None

    def put(self, sequences: List[str], tensor: torch.Tensor) -> None:
        path = self.cache_dir / f"{self._key(sequences)}.pt"
        torch.save(tensor.cpu(), path)


# ---------------------------------------------------------------------------
# RNABackbone
# ---------------------------------------------------------------------------

class RNABackbone(nn.Module):
    """Frozen RNA foundation model for sequence embedding extraction.

    Uses RiNALMo-giga (650M params, 1280-dim) from the *multimolecule*
    HuggingFace hub.  Falls back to RNA-FM (99.5M params, 640-dim) when
    RiNALMo is not available.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default: ``"multimolecule/rinalmo-giga"``).
    freeze : bool
        If ``True`` (default), all backbone parameters are frozen and the
        module is set to ``eval()`` permanently.
    cache_dir : str or Path or None
        Directory for caching embeddings to disk.  ``None`` disables caching.
    max_length : int
        Maximum token length accepted by the tokenizer (default: 1024).
    pool : {"none", "mean"}
        ``"none"`` returns per-token embeddings ``[B, L, D]``;
        ``"mean"`` returns mean-pooled embeddings ``[B, D]``.
    """

    def __init__(
        self,
        model_name: str = "multimolecule/rinalmo-giga",
        freeze: bool = True,
        cache_dir: Optional[Union[str, Path]] = None,
        max_length: int = 1024,
        pool: Literal["none", "mean"] = "none",
    ) -> None:
        super().__init__()

        self.model, self.tokenizer, self._embed_dim = _load_backbone(model_name)
        self.max_length = max_length
        self.pool = pool

        if freeze:
            self._freeze()

        self._cache: Optional[_EmbeddingCache] = None
        if cache_dir is not None:
            self._cache = _EmbeddingCache(Path(cache_dir))

    # -- properties ----------------------------------------------------------

    @property
    def embed_dim(self) -> int:
        """Dimensionality of the backbone output embeddings."""
        return self._embed_dim

    # -- freezing ------------------------------------------------------------

    def _freeze(self) -> None:
        """Freeze all backbone parameters and set to eval mode."""
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        logger.info("RNABackbone: all parameters frozen.")

    def train(self, mode: bool = True) -> "RNABackbone":
        """Override to keep the backbone always in eval mode."""
        super().train(mode)
        # The backbone must remain in eval mode even when the outer model
        # is set to train mode.
        self.model.eval()
        return self

    # -- forward -------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        sequences: List[str],
        attention_mask_out: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Extract embeddings for a batch of RNA sequences.

        Parameters
        ----------
        sequences : list[str]
            Raw RNA sequences (e.g. ``["AUGCUAG...", "GCUAAU..."]``).
        attention_mask_out : bool
            If ``True``, also return the attention mask ``[B, L]``.

        Returns
        -------
        embeddings : torch.Tensor
            ``[B, L, D]`` if ``self.pool == "none"`` or ``[B, D]`` if
            ``self.pool == "mean"``.
        attention_mask : torch.Tensor, optional
            ``[B, L]`` returned only when *attention_mask_out* is ``True``.
        """
        # Check disk cache first
        if self._cache is not None:
            cached = self._cache.get(sequences)
            if cached is not None:
                device = next(self.model.parameters()).device
                cached = cached.to(device)
                if attention_mask_out:
                    # Reconstruct a trivial mask (all ones up to cached length)
                    mask = torch.ones(
                        cached.shape[0], cached.shape[1],
                        dtype=torch.long, device=device,
                    )
                    return cached, mask
                return cached

        # Tokenize
        device = next(self.model.parameters()).device
        encoded = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # Forward through backbone
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: [B, L, D]
        embeddings = outputs.last_hidden_state

        # Optionally cache the raw per-token embeddings before pooling
        if self._cache is not None:
            self._cache.put(sequences, embeddings)

        # Pooling
        if self.pool == "mean":
            # Mask-aware mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).float()  # [B, L, 1]
            sum_emb = (embeddings * mask_expanded).sum(dim=1)     # [B, D]
            lengths = mask_expanded.sum(dim=1).clamp(min=1.0)     # [B, 1]
            embeddings = sum_emb / lengths

        if attention_mask_out:
            return embeddings, attention_mask
        return embeddings

    # -- convenience ---------------------------------------------------------

    def embed_and_pool(self, sequences: List[str]) -> torch.Tensor:
        """Return mean-pooled embeddings ``[B, D]`` regardless of ``self.pool``."""
        old_pool = self.pool
        self.pool = "mean"
        try:
            return self.forward(sequences)
        finally:
            self.pool = old_pool

    def __repr__(self) -> str:
        return (
            f"RNABackbone(embed_dim={self._embed_dim}, pool={self.pool!r}, "
            f"cached={self._cache is not None})"
        )
