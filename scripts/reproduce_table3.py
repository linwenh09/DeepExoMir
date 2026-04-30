"""Reproduce Table 3 (retrain-from-scratch ablation) of the DeepExoMir paper.

Compares per-test-set AU-PRC of v19 main against three retrain-from-scratch
ablation variants (v19_noRNALM, v19_noConservation, v19_noStructure /
DeepExoMir-Lite) using paired sample-level bootstrap.

There are two ways to run this:

(A) From pre-computed per-sample probability files (fast, deterministic)
    -- this is the path the paper used.  Required input is a directory
    containing the four ``*.npy`` per-sample score files plus the matching
    ``*_labels.npy`` per dataset.  This is what the manuscript repository
    ships under ``manuscript/BiB_submission/baseline_scores/``.

(B) From scratch, by re-evaluating each checkpoint against miRBench
    (slow; requires the four checkpoints + RiNALMo backbone).  Pass
    ``--mode rerun`` together with ``--checkpoints-dir``.

Usage (mode A, recommended)::

    python scripts/reproduce_table3.py \\
        --scores-dir manuscript/BiB_submission/baseline_scores \\
        --output results/table3.tsv

Usage (mode B, full re-run)::

    python scripts/reproduce_table3.py --mode rerun \\
        --checkpoints-dir checkpoints \\
        --output results/table3.tsv

Notes
-----
* The bootstrap uses 10,000 resamples on the smaller test sets and 2,000
  on the 327,129-sample Manakov set, paired sample-level (same indices
  for both methods within each resample, two-sided p-value).
* ``--seed 42`` is fixed by default; deltas should match the paper's
  Table 3 to four decimal places when the input score files are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger("reproduce_table3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

DATASETS: List[str] = [
    "AGO2_CLASH_Hejret2023",
    "AGO2_eCLIP_Klimentova2022",
    "AGO2_eCLIP_Manakov2022",
]

ABLATIONS: Dict[str, str] = {
    "v19_noRNALM":         "v19_noRNALM",
    "v19_noConservation":  "v19_noConservation",
    "v19_noStructure":     "v19_noStructure",   # this is DeepExoMir-Lite
}
MAIN_LABEL = "DeepExoMir_v19"


def _load(scores_dir: Path, ds: str, label: str) -> np.ndarray:
    p = scores_dir / f"{ds}_test_{label}.npy"
    if not p.is_file():
        raise FileNotFoundError(f"Missing score file: {p}")
    return np.load(p)


def paired_bootstrap(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float]:
    """Return (mean delta = AUPRC_a - AUPRC_b, ci_low, ci_high, p_two_sided)."""
    n = len(labels)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        # Skip degenerate resamples (all-positive or all-negative) which break AP
        y = labels[idx]
        if y.min() == y.max():
            deltas[i] = np.nan
            continue
        deltas[i] = (
            average_precision_score(y, scores_a[idx])
            - average_precision_score(y, scores_b[idx])
        )
    deltas = deltas[~np.isnan(deltas)]
    mean_delta = float(deltas.mean())
    ci_low, ci_high = np.quantile(deltas, [0.025, 0.975])
    # Two-sided p-value: fraction of resamples where the sign flipped
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    p = max(p, 1.0 / len(deltas))     # avoid p == 0 due to finite resamples
    return mean_delta, float(ci_low), float(ci_high), float(p)


def stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def run_from_scores(scores_dir: Path, n_boot_small: int, n_boot_large: int,
                    seed: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    for ds in DATASETS:
        labels = _load(scores_dir, ds, "labels")
        main = _load(scores_dir, ds, MAIN_LABEL)
        n_boot = n_boot_large if ds.endswith("Manakov2022") else n_boot_small
        main_auprc = float(average_precision_score(labels, main))
        logger.info("%s: n=%d, main AU-PRC=%.4f, n_boot=%d",
                    ds, len(labels), main_auprc, n_boot)
        for human, label in ABLATIONS.items():
            ablated = _load(scores_dir, ds, label)
            abl_auprc = float(average_precision_score(labels, ablated))
            mean_delta, lo, hi, pval = paired_bootstrap(
                labels, main, ablated, n_boot, rng,
            )
            rows.append({
                "ablation":        human,
                "dataset":         ds,
                "main_AUPRC":      main_auprc,
                "ablated_AUPRC":   abl_auprc,
                "delta_main_minus_ablated": mean_delta,
                "ci_low":          lo,
                "ci_high":         hi,
                "p_two_sided":     pval,
                "sig":             stars(pval),
            })
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Reproduce Table 3 retrain-from-scratch ablation")
    p.add_argument("--mode", choices=["scores", "rerun"], default="scores",
                   help="'scores' uses pre-computed per-sample .npy files (fast). "
                        "'rerun' re-evaluates each checkpoint from scratch (slow; not implemented in v0.1).")
    p.add_argument("--scores-dir", default="manuscript/BiB_submission/baseline_scores",
                   help="Directory containing per-sample score files (mode=scores)")
    p.add_argument("--output", required=True, help="TSV output path")
    p.add_argument("--n-boot-small", type=int, default=10000,
                   help="Bootstrap resamples for Hejret/Klimentova (default: 10,000)")
    p.add_argument("--n-boot-large", type=int, default=2000,
                   help="Bootstrap resamples for the 327k Manakov set (default: 2,000)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "rerun":
        logger.error("--mode rerun is not yet implemented in v0.1; use --mode scores.")
        return 2
    rows = run_from_scores(
        Path(args.scores_dir),
        n_boot_small=args.n_boot_small,
        n_boot_large=args.n_boot_large,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["ablation", "dataset", "main_AUPRC", "ablated_AUPRC",
                  "delta_main_minus_ablated", "ci_low", "ci_high",
                  "p_two_sided", "sig"]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        for r in rows:
            for k in ("main_AUPRC", "ablated_AUPRC", "delta_main_minus_ablated",
                      "ci_low", "ci_high", "p_two_sided"):
                r[k] = f"{r[k]:.4f}" if abs(r[k]) >= 1e-4 else f"{r[k]:.2e}"
            w.writerow(r)
    logger.info("Wrote %s", out)

    # Console summary
    print("\n" + "=" * 88)
    print("Table 3 reproduction -- retrain-from-scratch ablations vs v19 main")
    print("=" * 88)
    print(f"{'Ablation':<22} {'Dataset':<28} {'AU-PRC':>8} {'Δ main-abl':>12} {'p':>10}  {'sig'}")
    print("-" * 88)
    for r in rows:
        ds_short = r["dataset"].replace("AGO2_", "").replace("_", " ")
        print(f"{r['ablation']:<22} {ds_short:<28} "
              f"{r['ablated_AUPRC']:>8} {r['delta_main_minus_ablated']:>12} "
              f"{r['p_two_sided']:>10}  {r['sig']}")
    print("\nExpected (paper Table 3): noRNALM Δ ~ +0.029 mean (significant on 2/3); "
          "noConservation Δ ~ +0.005 (n.s. small-n); noStructure Δ ~ -0.008 "
          "(Lite slightly outperforms v19 on Hejret + Klimentova).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
