"""Calibration analysis for DeepExoMir v19 + Lite (v19_noStructure).

Computes:
  - Reliability diagram (10 bins) for each test set
  - Expected Calibration Error (ECE)
  - Maximum Calibration Error (MCE)
  - Brier score

Outputs:
  - figures/fig_calibration.png  (3 test sets x 2 models)
  - tables/calibration_metrics.tsv

Run:
  cd manuscript/BiB_submission
  python scripts/calibration_analysis.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------
# Paths.  Score files ship under manuscript/BiB_submission/baseline_scores/
# (also archived in the Zenodo deposit).  We fall back to a sibling
# baseline_scores/ directory so users with a custom layout can override it.
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_SCORE_DIR = ROOT / "manuscript" / "BiB_submission" / "baseline_scores"
SCORE_DIR = _DEFAULT_SCORE_DIR if _DEFAULT_SCORE_DIR.is_dir() else (ROOT / "baseline_scores")
OUT_DIR = ROOT / "results"
OUT_DIR.mkdir(exist_ok=True)
OUT_FIG = OUT_DIR / "fig10_calibration.png"
OUT_FIG_PDF = OUT_DIR / "fig10_calibration.pdf"
OUT_FIG_TIFF = OUT_DIR / "fig10_calibration.tiff"
OUT_TSV = OUT_DIR / "calibration_metrics.tsv"

DATASETS = [
    ("AGO2_CLASH_Hejret2023", "Hejret CLASH"),
    ("AGO2_eCLIP_Klimentova2022", "Klimentová eCLIP"),
    ("AGO2_eCLIP_Manakov2022", "Manakov eCLIP"),
]

MODELS = [
    ("DeepExoMir_v19", "v19 (reference)", "#1f77b4"),
    ("v19_noStructure", "DeepExoMir-Lite", "#d62728"),
]

N_BINS = 10


def load_scores_labels(dataset_id, model_id):
    score_path = SCORE_DIR / f"{dataset_id}_test_{model_id}.npy"
    label_path = SCORE_DIR / f"{dataset_id}_test_labels.npy"
    if not score_path.exists() or not label_path.exists():
        return None, None
    return np.load(score_path), np.load(label_path)


def calibration_curve(scores, labels, n_bins=N_BINS):
    """Return bin centers, mean predicted prob per bin, mean empirical prob per bin."""
    # Clip to [0, 1] (some baselines output unbounded logits — for v19 we use sigmoid)
    s = np.clip(scores, 0.0, 1.0)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_indices = np.digitize(s, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    pred_probs = np.zeros(n_bins)
    emp_probs = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        mask = bin_indices == b
        bin_counts[b] = mask.sum()
        if mask.any():
            pred_probs[b] = s[mask].mean()
            emp_probs[b] = labels[mask].mean()
        else:
            pred_probs[b] = bin_centers[b]
            emp_probs[b] = np.nan
    return bin_centers, pred_probs, emp_probs, bin_counts


def ece_mce(pred_probs, emp_probs, bin_counts):
    """Expected Calibration Error and Maximum Calibration Error."""
    n = bin_counts.sum()
    if n == 0:
        return np.nan, np.nan
    weights = bin_counts / n
    abs_diff = np.abs(pred_probs - emp_probs)
    abs_diff = np.where(np.isnan(abs_diff), 0.0, abs_diff)
    ece = float((weights * abs_diff).sum())
    mce = float(abs_diff.max())
    return ece, mce


def brier_score(scores, labels):
    s = np.clip(scores, 0.0, 1.0)
    return float(np.mean((s - labels) ** 2))


# ---------------------------------------------------------------------
# Compute metrics
# ---------------------------------------------------------------------
records = []
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)

for ax, (ds_id, ds_label) in zip(axes, DATASETS):
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.0,
            label="Perfect calibration")

    for m_id, m_label, color in MODELS:
        scores, labels = load_scores_labels(ds_id, m_id)
        if scores is None:
            print(f"[skip] {ds_id} / {m_id} missing")
            continue
        bc, pp, ep, cnt = calibration_curve(scores, labels)
        ece, mce = ece_mce(pp, ep, cnt)
        brier = brier_score(scores, labels)
        records.append({
            "dataset": ds_id,
            "model": m_id,
            "n_samples": int(scores.size),
            "n_positives": int(labels.sum()),
            "ECE": ece,
            "MCE": mce,
            "Brier": brier,
        })
        # Plot only bins that have data
        valid = ~np.isnan(ep)
        ax.plot(pp[valid], ep[valid], "o-", color=color, label=m_label,
                linewidth=1.8, markersize=6)
        print(f"  {ds_id:<28} {m_id:<18} ECE={ece:.4f} MCE={mce:.4f} Brier={brier:.4f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_title(f"{ds_label}\n(n={scores.size:,})", fontsize=10)
    ax.grid(alpha=0.3)

axes[0].set_ylabel("Empirical probability")
axes[0].legend(loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.savefig(OUT_FIG_PDF, dpi=300, bbox_inches="tight")
print(f"Saved {OUT_FIG}")
print(f"Saved {OUT_FIG_PDF}")

# Also save a TIFF for GPB submission
OUT_FIG_TIFF.parent.mkdir(parents=True, exist_ok=True)
from PIL import Image
img = Image.open(OUT_FIG)
if img.mode == "RGBA":
    img = img.convert("RGB")
img.save(OUT_FIG_TIFF, dpi=(300, 300), compression="tiff_lzw")
print(f"Saved {OUT_FIG_TIFF}")

# Save metrics table
import csv
with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(records)
print(f"Saved {OUT_TSV}")

# Print summary table
print("\n=== Calibration metrics summary ===")
print(f"{'dataset':<28} {'model':<18} {'ECE':>8} {'MCE':>8} {'Brier':>8}")
for r in records:
    print(f"{r['dataset']:<28} {r['model']:<18} {r['ECE']:>8.4f} {r['MCE']:>8.4f} {r['Brier']:>8.4f}")
