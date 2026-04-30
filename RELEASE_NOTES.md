# DeepExoMir release notes

## v1.1.0 (2026-04-30) -- IJMS submission snapshot

This is the persistent archive that pairs with the IJMS submission of
"DeepExoMir: A Reproducible RNA Language Model Framework for
CLIP-seq-Supported MicroRNA Target-Site Prioritization".  It supersedes
v1.0.0 (2026-03-25, Zenodo deposit on which the original BIB submission
was based) and contains everything needed to reproduce Tables 1, 3 and
the Figure 10 calibration analysis end-to-end.

### What is new since v1.0.0

* **Public reproducibility surface** -- new `deepexomir.config`,
  `deepexomir.predict`, and `deepexomir.benchmark` modules, plus the
  command-line scripts `scripts/reproduce_table1.py`,
  `scripts/reproduce_table3.py`, and `scripts/calibration_analysis.py`.
* **Pytest smoke tests** (`tests/test_reproduce_table3.py`,
  `tests/test_calibration.py`) with pinned reference outputs under
  `tests/expected_outputs/`.  Reviewers can run
  `pytest tests/test_reproduce_table3.py tests/test_calibration.py -v`
  end-to-end on a CPU-only laptop in roughly three minutes -- no
  checkpoint or RiNALMo download required for the smoke tier.  A
  `tests/conftest.py` skips GPU-only tests automatically when CUDA is
  unavailable, so `pytest tests/` (without arguments) is also safe.
* **Per-sample probability score files** for v19 main, DeepExoMir-Lite,
  the three retrain ablations, and eight miRBench baselines on each of
  the three CLIP-seq test sets, shipped as
  `manuscript/BiB_submission/baseline_scores/`.  These are what make
  Table 3 bit-deterministic without re-running RiNALMo.
* **Docker container** -- a `Dockerfile` at the repository root that
  builds a CPU-only smoke-test image (~1 GB).  The image is intentionally
  not published to a registry; the Dockerfile is the canonical artefact.
* **Calibration analysis (Section 2.6 of the manuscript)** -- ECE, MCE,
  and Brier metrics for v19 main and DeepExoMir-Lite on each test set,
  plus Figure 10 reliability diagrams.
* **Quickstart notebook** (`notebooks/01_quickstart.ipynb`) walking
  through `load_model`, `score_pair`, and `score_batch`.
* **DeepExoMir-Lite** (`v19_noStructure`) -- the production-recommended
  variant that drops the base-pairing CNN and ViennaRNA dependency.
  Mean miRBench AU-PRC 0.863 against v19's 0.855.

### Contents of this archive

| Path | Purpose | Approx. size |
|------|---------|--------------|
| `deepexomir/` | Python package: model, data, training, predict, benchmark, config | ~1 MB |
| `scripts/` | Reproduction CLIs (`reproduce_table1.py`, `reproduce_table3.py`, `calibration_analysis.py`, ...) | ~1 MB |
| `tests/` | Pytest smoke tests + pinned reference outputs | ~1 MB |
| `configs/` | YAML model and training configs (v19, v19_noStructure, ...) | <1 MB |
| `manuscript/BiB_submission/baseline_scores/` | Per-sample probability files (v19, Lite, three ablations, eight miRBench baselines) | ~252 MB |
| `checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt` | v19 main reference checkpoint (paper Table 1 column) | ~104 MB |
| `checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt` | DeepExoMir-Lite checkpoint (production variant) | ~104 MB |
| `checkpoints/v19_noConservation/checkpoint_epoch034_val_auc_0.8535.pt` | Retrain-from-scratch ablation, no PhyloP | ~104 MB |
| `checkpoints/v19_noRNALM/checkpoint_epoch042_val_auc_0.8410.pt` | Retrain-from-scratch ablation, no RiNALMo | ~104 MB |
| `Dockerfile` | CPU-only smoke-test container recipe | <1 KB |
| `README.md`, `REPRODUCIBILITY.md`, `LICENSE`, `pyproject.toml`, `requirements.txt` | Project metadata | <100 KB |

### Not included in this archive

* **PCA-reduced RiNALMo embeddings cache** (~32 GB).  Embeddings are
  deterministic given the published RiNALMo-giga checkpoint and the PCA
  fit shipped under `deepexomir/data/`.  The full cache will be
  regenerated automatically by `scripts/cache_embeddings_pca.py` on
  first run.
* **Raw miRBench training/test parquet files**.  Available through the
  `miRBench` Python package or directly from the miRBench Zenodo
  deposit; we mirror only the per-sample probability scores needed for
  Table 3 reproducibility.

### How to use this archive

1. Download the zip and extract into an empty directory.  The directory
   layout above will be reproduced exactly.
2. Follow `REPRODUCIBILITY.md` (three-tier: smoke tests -> Table 1 ->
   multi-seed retrain).
3. The active development mirror lives at
   <https://github.com/linwenh09/DeepExoMir>; this Zenodo deposit is the
   immutable snapshot that pairs with the manuscript.

### Citation

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

### License

MIT -- see `LICENSE`.
