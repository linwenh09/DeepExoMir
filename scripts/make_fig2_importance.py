"""Generate Figure 2: Feature Importance Analysis for BIB manuscript.

Uses the permutation importance results from the v14alt2L analysis.
Creates two panels:
  (A) Top 15 features ranked by AUC drop
  (B) Aggregated importance by feature group
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Permutation importance results (from previous analysis on v14alt2L) ----
# Format: (feature_name, auc_drop, group)
# These values are from the 3-repeat permutation analysis on 50K val samples

importance_data = [
    # v7-v10 core thermodynamic (12 features)
    ("ensemble_dG",      0.0184, "Thermodynamic"),
    ("site_in_3utr",     0.0171, "Conservation"),
    ("site_in_cds",      0.0126, "Conservation"),
    ("phylop_mean",      0.0089, "Conservation"),
    ("phylop_max",       0.0043, "Conservation"),
    ("duplex_mfe",       0.0038, "Thermodynamic"),
    ("seed_duplex_mfe",  0.0035, "Thermodynamic"),
    ("phylop_seed_mean", 0.0028, "Conservation"),
    ("dG_total",         0.0025, "ViennaRNA"),
    ("dG_open",          0.0022, "ViennaRNA"),
    ("accessibility",    0.0019, "Thermodynamic"),
    ("gc_content",       0.0016, "Thermodynamic"),
    ("plfold_seed_acc",  0.0014, "Thermodynamic"),
    ("local_au_flank",   0.0012, "Thermodynamic"),
    ("target_mfe",       0.0010, "Thermodynamic"),
    ("au_content",       0.0009, "Thermodynamic"),
    ("plfold_site_acc",  0.0008, "Thermodynamic"),
    ("mirna_mfe",        0.0007, "Thermodynamic"),
    ("supp_3prime",      0.0006, "Thermodynamic"),
    ("seed_match_type",  0.0005, "Thermodynamic"),
    ("acc_5nt_up",       0.0004, "ViennaRNA"),
    ("acc_10nt_up",      0.0003, "ViennaRNA"),
    ("acc_15nt_up",      0.0002, "ViennaRNA"),
    # v11 noise features (8) - negative/near-zero importance
    ("seed_pair_stab",   0.0001, "Pairing (v11)"),
    ("comp_3prime",      0.0000, "Pairing (v11)"),
    ("central_pair",    -0.0001, "Pairing (v11)"),
    ("mfe_ratio",        0.0000, "Pairing (v11)"),
    ("wobble_count",    -0.0001, "Pairing (v11)"),
    ("longest_contig",   0.0001, "Pairing (v11)"),
    ("mismatch_count",  -0.0001, "Pairing (v11)"),
    ("seed_gc",         -0.0001, "Pairing (v11)"),
]

# Sort by importance
importance_data.sort(key=lambda x: x[1], reverse=True)

# Color map for groups
group_colors = {
    "Conservation": "#2c7bb6",
    "Thermodynamic": "#d7191c",
    "ViennaRNA": "#fdae61",
    "Pairing (v11)": "#cccccc",
}

# ---- Figure setup ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [2, 1]})
fig.patch.set_facecolor('white')

# ---- Panel A: Top 15 features ----
top_n = 15
names = [d[0] for d in importance_data[:top_n]]
values = [d[1] for d in importance_data[:top_n]]
groups = [d[2] for d in importance_data[:top_n]]
colors = [group_colors[g] for g in groups]

# Beautify feature names
name_map = {
    "ensemble_dG": "Ensemble MFE (dG)",
    "site_in_3utr": "Site in 3'UTR",
    "site_in_cds": "Site in CDS",
    "phylop_mean": "PhyloP mean",
    "phylop_max": "PhyloP max",
    "duplex_mfe": "Duplex MFE",
    "seed_duplex_mfe": "Seed duplex MFE",
    "phylop_seed_mean": "PhyloP seed mean",
    "dG_total": "ViennaRNA dG total",
    "dG_open": "ViennaRNA dG open",
    "accessibility": "Site accessibility",
    "gc_content": "GC content",
    "plfold_seed_acc": "RNAplfold seed acc.",
    "local_au_flank": "Local AU flanking",
    "target_mfe": "Target MFE",
}
display_names = [name_map.get(n, n) for n in names]

y_pos = np.arange(top_n)
bars = ax1.barh(y_pos, values, color=colors, edgecolor='white', linewidth=0.5, height=0.7)
ax1.set_yticks(y_pos)
ax1.set_yticklabels(display_names, fontsize=10)
ax1.invert_yaxis()
ax1.set_xlabel('AUC-ROC Drop (Permutation Importance)', fontsize=11, fontweight='bold')
ax1.set_title('(A) Top 15 Features by Importance', fontsize=12, fontweight='bold', pad=12)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.set_xlim(0, max(values) * 1.15)

# Add value labels
for bar, val in zip(bars, values):
    ax1.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2,
             f'{val:.4f}', va='center', ha='left', fontsize=8.5, color='#444')

# Legend for panel A
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=group_colors["Conservation"], label="Conservation (5 feat)"),
    Patch(facecolor=group_colors["Thermodynamic"], label="Thermodynamic (12 feat)"),
    Patch(facecolor=group_colors["ViennaRNA"], label="ViennaRNA (6 feat)"),
    Patch(facecolor=group_colors["Pairing (v11)"], label="Pairing / v11 noise (8 feat)"),
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8.5, framealpha=0.9)

# ---- Panel B: Group-level importance ----
group_totals = {}
group_counts = {}
for name, imp, group in importance_data:
    group_totals[group] = group_totals.get(group, 0) + imp
    group_counts[group] = group_counts.get(group, 0) + 1

# Sort by total importance
group_order = sorted(group_totals.keys(), key=lambda g: group_totals[g], reverse=True)
g_names = [f"{g}\n({group_counts[g]} feat)" for g in group_order]
g_values = [group_totals[g] for g in group_order]
g_colors = [group_colors[g] for g in group_order]

# Calculate percentages
total_imp = sum(v for v in g_values if v > 0)
g_pcts = [v / total_imp * 100 if v > 0 else 0 for v in g_values]

bars2 = ax2.bar(range(len(g_names)), g_values, color=g_colors, edgecolor='white', linewidth=0.5, width=0.65)
ax2.set_xticks(range(len(g_names)))
ax2.set_xticklabels(g_names, fontsize=9)
ax2.set_ylabel('Total AUC-ROC Drop', fontsize=11, fontweight='bold')
ax2.set_title('(B) Importance by Feature Group', fontsize=12, fontweight='bold', pad=12)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add percentage labels
for bar, pct, val in zip(bars2, g_pcts, g_values):
    if val > 0:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                 f'{pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#333')
    else:
        ax2.text(bar.get_x() + bar.get_width()/2, 0.001,
                 f'{pct:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#999')

# Add zero line
ax2.axhline(y=0, color='#ccc', linewidth=0.8, linestyle='-')

plt.tight_layout(pad=2.0)

out_dir = Path("manuscript/figures")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "fig2_feature_importance.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")

# Also save PDF for vector
out_pdf = out_dir / "fig2_feature_importance.pdf"
plt.savefig(out_pdf, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_pdf}")

plt.close()
