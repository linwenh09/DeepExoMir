# DeepExoMir reproducibility guide

This document tells reviewers exactly how to (a) install DeepExoMir,
(b) reproduce the Table 1 / Table 3 benchmark results, and (c) predict
target sites for a custom miRNA-UTR pair.

## TL;DR — fastest path for reviewers (~3 min, CPU-only)

The Table 3 reproduction does not require the RiNALMo backbone (it
operates on pre-computed per-sample probability scores shipped in the
repository).  This is the cheapest end-to-end check.

```bash
git clone https://github.com/linwenh09/DeepExoMir.git
cd DeepExoMir
pip install numpy pandas scikit-learn matplotlib pyyaml pytest
pytest tests/               # ~3 min, runs Table 3 + Calibration smoke tests
```

Expected output: `2 passed` (the rest are skipped automatically by
`tests/conftest.py` when torch / CUDA are unavailable).  The smoke tests
do not import the `deepexomir` package or torch -- they read the
per-sample probability score files shipped under
`manuscript/BiB_submission/baseline_scores/` and call the standalone
analysis scripts under `scripts/`, then diff each script's output
against the pinned reference in `tests/expected_outputs/`.

## Three-tier reproducibility

### Tier 1 — Smoke tests (3 min, CPU)

Confirms that `scripts/reproduce_table3.py` and
`scripts/calibration_analysis.py` produce the exact numbers reported in
the paper, given the per-sample score files shipped under
`manuscript/BiB_submission/baseline_scores/`.

```bash
pytest tests/ -v
```

### Tier 2 — Table 1 reproduction (~90 s GPU / ~10 min CPU)

Re-evaluates the v19 main and DeepExoMir-Lite checkpoints against the
three miRBench held-out test sets from scratch.  Requires the trained
checkpoints from Zenodo (DOI 10.5281/zenodo.19216306) and the RiNALMo
RNA language model (auto-downloaded by the `multimolecule` package on
first run).

```bash
# v19 main (paper Table 1 reference column)
python scripts/reproduce_table1.py \
    --checkpoint checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt \
    --config     configs/model_config_v19.yaml \
    --output     results/table1_v19.tsv

# DeepExoMir-Lite (recommended production variant)
python scripts/reproduce_table1.py \
    --checkpoint checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt \
    --config     configs/model_config_v19_noStructure.yaml \
    --output     results/table1_lite.tsv \
    --score-label DeepExoMir_Lite
```

Expected mean AU-PRC: **0.855** (v19), **0.863** (Lite).

CPU fallback: add `--device cpu --batch-size 32`.

### Tier 3 — Multi-seed retraining (~14 hours per seed, RTX 5090)

Re-trains DeepExoMir-Lite from scratch under additional random seeds
and verifies that test AU-PRC SD < 0.002.  Used to produce the
multi-seed stability statement in the paper (test ROC-AUC SD = 0.001
across four runs).

```bash
for seed in 42 123 456; do
    python scripts/train.py \
        --config        configs/train_config_v19_noStructure.yaml \
        --model-config  configs/model_config_v19_noStructure.yaml \
        --seed          $seed \
        --output_dir    checkpoints/lite_seed${seed}
done
python scripts/_run_multi_seed.py --checkpoints "checkpoints/lite_seed*"
```

This tier is optional and not part of the smoke tests.

## Container

A `Dockerfile` is included at the repository root for users who prefer
a sealed runtime.  Build it locally (no `docker pull` needed):

```bash
docker build -t deepexomir:latest .
docker run --rm -it --gpus all -v $(pwd):/workspace deepexomir:latest \
    pytest tests/
```

The image is intentionally not published to a registry; the Dockerfile
is the canonical artifact and is short enough to audit.

## What's in the repository

| Path | Purpose |
|------|---------|
| `deepexomir/` | Python package: model, data, training |
| `deepexomir/predict.py` | `load_model`, `score_pair`, `score_batch` |
| `deepexomir/benchmark.py` | `evaluate_mirbench_test_sets` |
| `deepexomir/config.py` | YAML config loader |
| `scripts/reproduce_table1.py` | re-runs Table 1 benchmark |
| `scripts/reproduce_table3.py` | re-runs retrain-from-scratch ablation comparison |
| `scripts/calibration_analysis.py` | regenerates Figure 10 + ECE/MCE/Brier metrics |
| `scripts/evaluate_mirbench.py` | underlying CLI for `reproduce_table1.py` |
| `notebooks/01_quickstart.ipynb` | 10-minute walk-through |
| `tests/expected_outputs/` | pinned reference TSVs for the smoke tests |
| `tests/test_*.py` | Pytest reproducibility checks |
| `manuscript/BiB_submission/baseline_scores/` | per-sample probability score files for v19, Lite, three retrain ablations, and 8 miRBench baselines on each of the three test sets |

## What is *not* shipped in the GitHub mirror

These are large or environment-specific and live on Zenodo (DOI
10.5281/zenodo.19216306):

* Trained checkpoints (`checkpoints/v19/...`,
  `checkpoints/v19_noStructure/...`, `checkpoints/v19_no15b/...`)
* Pre-computed PCA-reduced RiNALMo embeddings cache
  (`embeddings_cache_pca256/`)
* Raw miRBench training/test parquet files (also obtainable through
  the `miRBench` Python package)

After downloading the Zenodo archive, unzip into the repository root
and the directory layout above will be reproduced exactly.

## Versioning & archival

| Resource | URL |
|----------|-----|
| Active development | https://github.com/linwenh09/DeepExoMir (MIT) |
| Persistent snapshot | https://doi.org/10.5281/zenodo.19216306 |
| Issue tracker | https://github.com/linwenh09/DeepExoMir/issues |

## Citing the resource

```bibtex
@article{lin2026deepexomir,
  title   = {{DeepExoMir}: A Reproducible RNA Language Model Framework
             for CLIP-seq-Supported MicroRNA Target-Site Prioritization},
  author  = {Lin, Wen-Hsien and Hsiung, Chia-Ni and Lien, Wen-Yu and Sieber, Martin},
  journal = {(under review at International Journal of Molecular Sciences)},
  year    = {2026},
  doi     = {10.5281/zenodo.19216306}
}
```

## Known limitations

1. The retrospective scoring pipeline omits 5 PhyloP conservation
   features (per-site BigWig lookups not included in the public
   release). This contributes a measurable but small (<1% ROC-AUC)
   gap on the Hejret cross-validation sanity check (Section 2.7 of
   the paper).
2. AI-assistance: generative AI tools (ChatGPT and Claude) were used
   for language polishing and pre-submission review only. They were
   not used to generate data, perform analyses, produce figures, or
   draw scientific conclusions.
3. All four authors are employed by GGA Corp., BIONET Therapeutics
   Corp., or BIONET Corp. The companies may have a commercial interest
   in exosome-related applications. To enable independent
   verification, all source code, trained model weights, benchmark
   outputs, and the dual-probe ablation pipeline are publicly released
   under the MIT license.

## Contact

Open an issue: https://github.com/linwenh09/DeepExoMir/issues  
Correspondence: BryceLin@bionetTX.com
