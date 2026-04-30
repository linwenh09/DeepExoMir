"""Reproduce Table 1 of the DeepExoMir paper.

Runs the v19 reference checkpoint (and DeepExoMir-Lite, if requested) on
the three miRBench held-out test sets and writes a TSV with mean AU-PRC,
ROC-AUC, accuracy, and F1 for each model.

Usage
-----
Default (v19 main on GPU)::

    python scripts/reproduce_table1.py \\
        --checkpoint checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt \\
        --config configs/model_config_v19.yaml \\
        --output results/table1_v19.tsv

DeepExoMir-Lite (v19_noStructure)::

    python scripts/reproduce_table1.py \\
        --checkpoint checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt \\
        --config configs/model_config_v19_noStructure.yaml \\
        --output results/table1_lite.tsv \\
        --score-label DeepExoMir_Lite

CPU fallback::

    python scripts/reproduce_table1.py ... --device cpu --batch-size 32

Notes
-----
* The expected reference output is pinned at
  ``tests/expected_outputs/table1_v19.tsv`` and
  ``tests/expected_outputs/table1_lite.tsv``.  The smoke test compares
  against these.
* Evaluation requires the miRBench Python package (will download test
  sets on first run; ~50 MB).
* On an RTX 5090 the full evaluation runs in ~90 seconds per checkpoint;
  on a single CPU core, ~10 minutes.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from deepexomir.benchmark import (
    DEFAULT_DATASETS,
    evaluate_mirbench_test_sets,
    summarize_results,
)
from deepexomir.config import load_config
from deepexomir.predict import load_model, load_pca

logger = logging.getLogger("reproduce_table1")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table 1 of the DeepExoMir paper")
    p.add_argument("--checkpoint", required=True,
                   help="Path to the .pt checkpoint to evaluate")
    p.add_argument("--config", required=True,
                   help="YAML model config (e.g. configs/model_config_v19.yaml)")
    p.add_argument("--output", required=True,
                   help="TSV output path for the per-dataset metrics")
    p.add_argument("--device", default=None,
                   help="cuda or cpu (default: cuda when available)")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Inference batch size (256 fits on 8 GB GPU)")
    p.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                   help="miRBench test set IDs (default: 3 standard test sets)")
    p.add_argument("--score-label", default="DeepExoMir_v19",
                   help="Suffix for per-sample score files when --save-scores is set")
    p.add_argument("--save-scores", default=None,
                   help="If set, save per-sample probabilities under this directory")
    p.add_argument("--embeddings-dir", default=None,
                   help="Optional precomputed-embeddings directory containing pca_params.npz")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)

    logger.info("Loading model from %s", args.checkpoint)
    model, backbone = load_model(
        checkpoint=args.checkpoint,
        config=cfg,
        load_backbone=True,
        device=args.device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %.1fM parameters on %s",
                n_params / 1e6, next(model.parameters()).device)

    pca = load_pca(args.embeddings_dir) if args.embeddings_dir else None
    if pca is not None:
        logger.info("Loaded PCA: components shape=%s", pca["components"].shape)

    logger.info("Evaluating on %d miRBench test set(s): %s",
                len(args.datasets), args.datasets)
    results = evaluate_mirbench_test_sets(
        model, backbone, pca=pca,
        datasets=args.datasets,
        batch_size=args.batch_size,
    )
    summary = summarize_results(results)

    # ------------------------------------------------------------------
    # Write TSV (per-dataset rows + a 'mean' row at the bottom)
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset", "n_samples", "n_positives", "AU-PRC", "ROC-AUC",
                  "accuracy", "F1", "time_s"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(fieldnames)
        for ds, m in results.items():
            w.writerow([
                ds, m["n"], m["n_positives"],
                f"{m['au_prc']:.4f}", f"{m['roc_auc']:.4f}",
                f"{m['accuracy']:.4f}", f"{m['f1']:.4f}",
                f"{m['time_s']:.1f}",
            ])
        w.writerow([
            "MEAN", "", "",
            f"{summary['mean_au_prc']:.4f}", f"{summary['mean_roc_auc']:.4f}",
            f"{summary['mean_accuracy']:.4f}", f"{summary['mean_f1']:.4f}",
            "",
        ])
    logger.info("Wrote per-dataset metrics: %s", out_path)

    # Per-sample scores (for downstream paired bootstrap / calibration)
    if args.save_scores:
        save_dir = Path(args.save_scores)
        save_dir.mkdir(parents=True, exist_ok=True)
        for ds, m in results.items():
            np.save(save_dir / f"{ds}_test_{args.score_label}.npy", m["scores"])
            np.save(save_dir / f"{ds}_test_labels.npy", m["labels"])
        logger.info("Saved per-sample scores under %s", save_dir)

    # ------------------------------------------------------------------
    # Print human-readable summary to stdout
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"DeepExoMir Table 1 reproduction -- {args.checkpoint}")
    print("=" * 70)
    print(f"{'Dataset':<32} {'AU-PRC':>8} {'ROC-AUC':>8} {'Acc':>8} {'F1':>8}")
    print("-" * 70)
    for ds, m in results.items():
        print(f"{ds:<32} {m['au_prc']:>8.4f} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} {m['f1']:>8.4f}")
    print("-" * 70)
    print(f"{'MEAN':<32} {summary['mean_au_prc']:>8.4f} "
          f"{summary['mean_roc_auc']:>8.4f} {summary['mean_accuracy']:>8.4f} "
          f"{summary['mean_f1']:>8.4f}")
    print("\nExpected (paper Table 1, v19 main):  mean AU-PRC = 0.855")
    print("Expected (paper Table 1, Lite):      mean AU-PRC = 0.863")
    return 0


if __name__ == "__main__":
    sys.exit(main())
