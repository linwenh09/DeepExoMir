# DeepExoMir

A reproducible RNA language model framework for CLIP-seq-supported microRNA target-site prioritization.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19216306.svg)](https://doi.org/10.5281/zenodo.19216306)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## What this is

DeepExoMir is a deep learning framework that ranks candidate miRNA target
sites against AGO2 CLIP-sequencing-validated negatives.  We released two
checkpoints:

* **DeepExoMir v19** -- pre-specified reference model used for the
  paper's Table 1 baseline comparisons.
* **DeepExoMir-Lite** (recommended for production) -- ablation-derived
  simplified variant; same RNA language model backbone, no base-pairing
  CNN and no ViennaRNA dependency, ~35% fewer trainable parameters,
  equivalent or marginally better mean test AU-PRC.

A dual-probe ablation protocol (inference-time feature masking +
retrain-from-scratch with each signal zeroed at every training sample)
shows that the frozen RiNALMo RNA language model is the dominant
operationally necessary signal; PhyloP conservation and explicit duplex
structure are largely redundant once RiNALMo is present.

## Performance (paper Table 1)

Evaluated on three [miRBench](https://github.com/katarinagresova/miRBench)
held-out test sets with bias-corrected CLIP-seq-validated negatives:

| Variant | Hejret CLASH (n=965) | Klimentová eCLIP (n=954) | Manakov eCLIP (n=327k) | Mean AU-PRC |
|---|:---:|:---:|:---:|:---:|
| **DeepExoMir v19** (reference) | 0.848 | 0.868 | 0.848 | **0.855** |
| **DeepExoMir-Lite** (production) | 0.866 | 0.871 | 0.853 | **0.863** |
| Best retrained CNN baseline | 0.84 | 0.82 | 0.81 | 0.82 |

All eight retrained miRBench baselines are beaten at paired sample-level
bootstrap p < 0.001 (Table 1 of the paper).

## Installation

```bash
git clone https://github.com/linwenh09/DeepExoMir.git
cd DeepExoMir

# Either: conda environment (recommended for GPU work)
conda create -n deepexomir python=3.11
conda activate deepexomir
pip install -e .
pip install miRBench

# Or: Docker container (CPU and GPU)
docker build -t deepexomir:latest .
docker run --rm -it --gpus all -v $(pwd):/workspace deepexomir:latest bash
```

Trained checkpoints are released at
[Zenodo (DOI 10.5281/zenodo.19216306)](https://doi.org/10.5281/zenodo.19216306)
and live under `checkpoints/` after download.

## Quickstart

```python
from deepexomir.config import load_config
from deepexomir.predict import load_model, score_pair

cfg = load_config("configs/model_config_v19_noStructure.yaml")
model, backbone = load_model(
    "checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt",
    cfg, load_backbone=True,
)

p = score_pair(model, backbone,
               mirna_seq="GUGAAAUGUUUAGGACCACUAG",   # hsa-miR-203a-3p
               target_seq="AGGCUUAUGCAUUUCAGAUUU")   # KITLG 3'UTR window
print(f"score = {p:.4f}")
```

A walk-through covering single-pair scoring and full Table 1 reproduction
lives in `notebooks/01_quickstart.ipynb`.

## Reproducing the paper

| What | Command | Runtime |
|---|---|:---:|
| Table 1 (v19) | `python scripts/reproduce_table1.py --checkpoint checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt --config configs/model_config_v19.yaml --output results/table1_v19.tsv` | ~90 s GPU / 10 min CPU |
| Table 1 (Lite) | `python scripts/reproduce_table1.py --checkpoint checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt --config configs/model_config_v19_noStructure.yaml --output results/table1_lite.tsv --score-label DeepExoMir_Lite` | ~75 s GPU / 8 min CPU |
| Table 3 (retrain ablation) | `python scripts/reproduce_table3.py --output results/table3.tsv` | ~3 min CPU |
| Figure 10 (calibration) | `python scripts/calibration_analysis.py` | ~5 s CPU |
| Smoke tests | `pytest tests/` | ~3 min CPU |

`tests/expected_outputs/` contains pinned reference outputs; the smoke
tests in `tests/` compare each script's output against the pinned
reference and fail on numerical drift.

## Repository layout

```
DeepExoMir/
├── deepexomir/                  # Python package
│   ├── config.py                # YAML config loader
│   ├── predict.py               # load_model / score_pair / score_batch
│   ├── benchmark.py             # evaluate_mirbench_test_sets
│   ├── data/                    # Dataset loaders + feature extraction
│   ├── model/                   # Architecture (DeepExoMirModelV8)
│   └── training/                # Trainer, callbacks
├── configs/                     # Model + training YAML configs
├── scripts/
│   ├── reproduce_table1.py      # Re-run benchmark eval
│   ├── reproduce_table3.py      # Re-run retrain-from-scratch ablations
│   ├── calibration_analysis.py  # Reliability diagrams + ECE/MCE/Brier
│   └── evaluate_mirbench.py     # Lower-level evaluation (used by reproduce_table1)
├── notebooks/01_quickstart.ipynb
├── tests/
│   ├── expected_outputs/        # Pinned reference TSVs
│   └── test_*.py                # Reproducibility smoke tests
├── manuscript/                  # Paper sources (gitignored from main branch)
├── Dockerfile
└── requirements.txt
```

## Citation

```bibtex
@article{lin2026deepexomir,
  title   = {{DeepExoMir}: A Reproducible RNA Language Model Framework
             for CLIP-seq-Supported MicroRNA Target-Site Prioritization},
  author  = {Lin, Wen-Hsien and Hsiung, Chia-Ni and Lien, Wen-Yu and Sieber, Martin},
  journal = {(under review)},
  year    = {2026},
  doi     = {10.5281/zenodo.19216306}
}
```

## License

MIT (see [LICENSE](LICENSE)).

All four authors are employed by GGA Corp., BIONET Therapeutics Corp., or
BIONET Corp.  These companies develop exosome-based products for cosmetic
and aesthetic-medicine applications and may have a commercial interest in
applications of this framework.  Source code, trained model weights,
benchmark outputs, and the dual-probe ablation pipeline are released
publicly under the MIT license to enable independent verification.

## Contact

Wen-Hsien Lin -- BryceLin@bionetTX.com
AI and Data Applications Division, GGA Corp.
