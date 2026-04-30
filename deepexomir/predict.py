"""Single-pair prediction API for DeepExoMir.

Convenience wrapper around the project's internal evaluation pipeline so
external users can score one or a small list of (miRNA, target window)
pairs without invoking the full miRBench command-line script.

Example
-------
>>> from deepexomir.config import load_config
>>> from deepexomir.predict import load_model, score_pair
>>> cfg = load_config("configs/model_config_v19_noStructure.yaml")
>>> model, backbone = load_model(
...     checkpoint="checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt",
...     config=cfg,
...     load_backbone=True,
... )
>>> p = score_pair(model, backbone,
...                mirna_seq="GUGAAAUGUUUAGGACCACUAG",
...                target_seq="AGGCUUAUGCAUUUCAGAUUU")
>>> print(f"score = {p:.4f}")

The current implementation requires the RiNALMo backbone (used to embed
sequences on the fly).  CPU inference is supported but slower; we
recommend a GPU with at least 8 GB of memory for batched scoring.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import torch

from deepexomir.model import DeepExoMirModelV8

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]


def load_model(
    checkpoint: PathLike,
    config: Mapping[str, Any],
    load_backbone: bool = True,
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[DeepExoMirModelV8, Optional[Dict[str, Any]]]:
    """Load a DeepExoMir checkpoint plus (optionally) the RiNALMo backbone.

    Parameters
    ----------
    checkpoint : str or Path
        Path to a ``.pt`` checkpoint produced by ``scripts/train.py``.  The
        loader accepts both the raw state dict and a wrapping dict with
        ``model_state_dict`` / ``state_dict`` keys.
    config : Mapping
        Model config dict (load via :func:`deepexomir.config.load_config`).
    load_backbone : bool
        If True, also load the frozen RiNALMo backbone + tokenizer + (when
        applicable) the precomputed PCA parameters used during training.
        Required for inference on raw sequences.  Default: True.
    device : str or torch.device or None
        Inference device.  If None, defaults to CUDA when available else CPU.

    Returns
    -------
    (model, backbone_dict)
        ``model`` is a :class:`DeepExoMirModelV8` in eval mode on ``device``.
        ``backbone_dict`` is None if ``load_backbone=False``; otherwise a
        dict with keys ``model``, ``tokenizer`` and (optionally) ``pca``.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)

    model = DeepExoMirModelV8(config, load_backbone=False, precomputed_embeddings=True)
    ckpt = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, Mapping) else ckpt
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device).eval()

    backbone_dict: Optional[Dict[str, Any]] = None
    if load_backbone:
        backbone_dict = _load_rinalmo_backbone(config, device)

    return model, backbone_dict


def _load_rinalmo_backbone(
    config: Mapping[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    import multimolecule  # noqa: F401 -- side effect: registers model types
    from transformers import AutoModel, AutoTokenizer

    backbone_name = config.get("backbone", {}).get("name", "multimolecule/rinalmo-giga")
    tokenizer = AutoTokenizer.from_pretrained(backbone_name, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
    backbone = backbone.to(device).eval()
    return {"model": backbone, "tokenizer": tokenizer, "name": backbone_name}


@torch.no_grad()
def score_pair(
    model: DeepExoMirModelV8,
    backbone: Dict[str, Any],
    mirna_seq: str,
    target_seq: str,
    pca: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Score a single (miRNA, target window) pair.

    Returns
    -------
    float in [0, 1]
        Predicted probability that the pair is a CLIP-seq-supported
        binding site under the current model.  Interpret as a *ranking*
        signal; for cross-miRNA prioritization use percentile thresholds
        rather than absolute cutoffs (see Section 2.6 of the paper for the
        calibration analysis).
    """
    return score_batch(model, backbone, [mirna_seq], [target_seq], pca=pca)[0]


@torch.no_grad()
def score_batch(
    model: DeepExoMirModelV8,
    backbone: Dict[str, Any],
    mirna_seqs: List[str],
    target_seqs: List[str],
    pca: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Score a list of (miRNA, target window) pairs.

    Returns
    -------
    np.ndarray of shape (len(mirna_seqs),)
        Predicted probabilities in [0, 1].
    """
    if len(mirna_seqs) != len(target_seqs):
        raise ValueError("mirna_seqs and target_seqs must have the same length")
    device = next(model.parameters()).device

    bb_model = backbone["model"]
    bb_tokenizer = backbone["tokenizer"]

    max_m = model.max_mirna_len
    max_t = model.max_target_len

    def _embed(seqs: List[str], max_len: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc = bb_tokenizer(seqs, padding=True, truncation=True,
                           max_length=max_len + 2, return_tensors="pt").to(device)
        out = bb_model(**enc)
        hidden = out.last_hidden_state[:, 1:-1, :]            # drop CLS/SEP
        amask = enc["attention_mask"][:, 1:-1]
        denom = amask.sum(1, keepdim=True).clamp(min=1)
        pooled = (hidden * amask.unsqueeze(-1)).sum(1) / denom

        if pca is not None:
            mean = torch.from_numpy(pca["mean"]).to(device).float()
            comp = torch.from_numpy(pca["components"]).to(device).float()
            hidden = (hidden - mean) @ comp.T
            pooled = (pooled - mean) @ comp.T

        L = hidden.shape[1]
        D = hidden.shape[-1]
        if L < max_len:
            pad_h = torch.zeros(hidden.shape[0], max_len - L, D, device=device)
            hidden = torch.cat([hidden, pad_h], dim=1)
            pad_v = torch.zeros(amask.shape[0], max_len - L, dtype=torch.bool, device=device)
            valid = torch.cat([amask.bool(), pad_v], dim=1)
        else:
            hidden = hidden[:, :max_len, :]
            valid = amask.bool()[:, :max_len]
        return hidden, pooled, valid

    m_hidden, m_pooled, m_valid = _embed(mirna_seqs, max_m)
    t_hidden, t_pooled, t_valid = _embed(target_seqs, max_t)

    B = len(mirna_seqs)
    struct_feat = torch.zeros(B, model.struct_mlp.in_dim, device=device)

    out = model(
        mirna_seqs=mirna_seqs,
        target_seqs=target_seqs,
        struct_features=struct_feat,
        mirna_pertoken_emb=m_hidden,
        mirna_pertoken_mask=m_valid,
        target_pertoken_emb=t_hidden,
        target_pertoken_mask=t_valid,
        mirna_pooled_emb=m_pooled,
        target_pooled_emb=t_pooled,
    )
    probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().numpy()
    return probs


def load_pca(embeddings_dir: PathLike) -> Optional[Dict[str, np.ndarray]]:
    """Load PCA parameters from a precomputed-embeddings directory, if any.

    Returns None if no ``pca_params.npz`` is found.
    """
    p = Path(embeddings_dir) / "pca_params.npz"
    if not p.exists():
        return None
    data = np.load(p)
    return {"mean": data["mean"], "components": data["components"]}


__all__ = ["load_model", "score_pair", "score_batch", "load_pca"]
