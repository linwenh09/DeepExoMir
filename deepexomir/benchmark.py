"""miRBench benchmark evaluation API for DeepExoMir.

Thin wrapper around the project's ``scripts/evaluate_mirbench.py`` flow,
exposed as a callable Python function so users can reproduce Table 1 (and
the calibration figure) without invoking the CLI.

Example
-------
>>> from deepexomir.config import load_config
>>> from deepexomir.predict import load_model
>>> from deepexomir.benchmark import evaluate_mirbench_test_sets
>>> cfg = load_config("configs/model_config_v19_noStructure.yaml")
>>> model, backbone = load_model(
...     "checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt",
...     cfg, load_backbone=True)
>>> results = evaluate_mirbench_test_sets(model, backbone)
>>> for ds, m in results.items():
...     print(f"{ds:<28} AU-PRC = {m['au_prc']:.4f}")
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from deepexomir.predict import score_batch

logger = logging.getLogger(__name__)

DEFAULT_DATASETS: List[str] = [
    "AGO2_CLASH_Hejret2023",
    "AGO2_eCLIP_Klimentova2022",
    "AGO2_eCLIP_Manakov2022",
]


def evaluate_mirbench_test_sets(
    model,
    backbone: Mapping[str, Any],
    pca: Optional[Mapping[str, np.ndarray]] = None,
    datasets: Optional[Iterable[str]] = None,
    batch_size: int = 256,
) -> Dict[str, Dict[str, float]]:
    """Evaluate a loaded model on the three miRBench held-out test sets.

    Parameters
    ----------
    model
        DeepExoMir model returned by :func:`deepexomir.predict.load_model`.
    backbone
        Backbone dict (model + tokenizer) returned by ``load_model``.
    pca : optional
        PCA parameters (load via :func:`deepexomir.predict.load_pca`).  Only
        needed when the model was trained with PCA-reduced embeddings.
    datasets : iterable of str, optional
        miRBench dataset IDs.  Defaults to the three CLIP-seq-validated
        held-out test sets used in Table 1.
    batch_size : int
        Inference batch size.  256 fits comfortably on an 8 GB GPU.

    Returns
    -------
    dict
        Mapping ``dataset_id -> {au_prc, roc_auc, accuracy, f1, n,
        n_positives, time_s}``.
    """
    from miRBench.dataset import get_dataset_df

    if datasets is None:
        datasets = DEFAULT_DATASETS

    results: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        df = get_dataset_df(ds, "test")
        labels = df["label"].values.astype(np.int64)
        mirna_seqs = df["noncodingRNA"].tolist()
        target_seqs = df["gene"].tolist()

        n = len(df)
        probs = np.empty(n, dtype=np.float32)
        t0 = time.time()
        n_batches = (n + batch_size - 1) // batch_size
        for i in range(n_batches):
            s = i * batch_size
            e = min(s + batch_size, n)
            probs[s:e] = score_batch(
                model, backbone, mirna_seqs[s:e], target_seqs[s:e], pca=pca,
            ).astype(np.float32)
            if (i + 1) % 50 == 0:
                logger.info("  %s: %d / %d batches done", ds, i + 1, n_batches)
        elapsed = time.time() - t0

        preds = (probs >= 0.5).astype(np.int64)
        results[ds] = {
            "au_prc": float(average_precision_score(labels, probs)),
            "roc_auc": float(roc_auc_score(labels, probs)),
            "accuracy": float(accuracy_score(labels, preds)),
            "f1": float(f1_score(labels, preds)),
            "n": int(n),
            "n_positives": int(labels.sum()),
            "time_s": float(elapsed),
            "scores": probs,    # kept for downstream paired bootstrap / calibration
            "labels": labels,
        }
        logger.info(
            "%s  AU-PRC=%.4f  ROC-AUC=%.4f  acc=%.4f  F1=%.4f  n=%d  (%.1fs)",
            ds, results[ds]["au_prc"], results[ds]["roc_auc"],
            results[ds]["accuracy"], results[ds]["f1"], n, elapsed,
        )
    return results


def summarize_results(results: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Compute mean AU-PRC, ROC-AUC across datasets (Table 1 row)."""
    return {
        "mean_au_prc": float(np.mean([r["au_prc"] for r in results.values()])),
        "mean_roc_auc": float(np.mean([r["roc_auc"] for r in results.values()])),
        "mean_accuracy": float(np.mean([r["accuracy"] for r in results.values()])),
        "mean_f1": float(np.mean([r["f1"] for r in results.values()])),
        "n_datasets": len(results),
    }


__all__ = ["evaluate_mirbench_test_sets", "summarize_results", "DEFAULT_DATASETS"]
