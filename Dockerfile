# DeepExoMir reproducibility container -- CPU-only smoke-test image
#
# Build:
#   docker build -t deepexomir:latest .
#
# Run smoke tests (default; ~3 min, ~1 GB image):
#   docker run --rm -v $(pwd):/workspace -w /workspace deepexomir:latest
#
# Drop into a shell for ad-hoc inspection:
#   docker run -it --rm -v $(pwd):/workspace -w /workspace deepexomir:latest bash
#
# This image is intentionally minimal.  It contains only what is needed
# to reproduce the Table 3 retrain-from-scratch ablation comparison and
# the Figure 10 calibration analysis, both of which operate on the
# pre-computed per-sample probability score files shipped under
# manuscript/BiB_submission/baseline_scores/.  It does NOT contain the
# RiNALMo backbone, CUDA wheels, or ViennaRNA.  For full Table 1
# re-evaluation against the miRBench checkpoints, follow the README
# pip-install path on a host with a GPU.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="DeepExoMir"
LABEL org.opencontainers.image.description="Reproducible RNA language model framework for CLIP-seq-supported miRNA target-site prediction (CPU smoke-test image)"
LABEL org.opencontainers.image.source="https://github.com/linwenh09/DeepExoMir"
LABEL org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /workspace

# Minimal deps for the smoke tests.  Versions deliberately loose so the
# image still rebuilds cleanly when wheels age out of PyPI; the smoke
# tests are bit-deterministic with seed=42 and tolerate minor numerical
# drift across BLAS / scikit-learn versions (see tests/test_*.py).
RUN pip install --upgrade pip && \
    pip install \
        "numpy>=1.24,<3" \
        "pandas>=2.0" \
        "scikit-learn>=1.3" \
        "matplotlib>=3.7" \
        "pyyaml>=6.0" \
        "pytest>=7.4"

# Copy only what the smoke tests need (no checkpoints, no embeddings
# cache).  The user can also bind-mount the repo over /workspace at run
# time to test against an in-progress edit -- this is the recommended
# reviewer flow.
COPY pyproject.toml /workspace/pyproject.toml
COPY README.md      /workspace/README.md
COPY deepexomir     /workspace/deepexomir
COPY scripts        /workspace/scripts
COPY tests          /workspace/tests
COPY configs        /workspace/configs
COPY manuscript/BiB_submission/baseline_scores \
                    /workspace/manuscript/BiB_submission/baseline_scores

# Default: run the smoke tests.  Override with `bash` for an
# interactive shell.
CMD ["pytest", "tests/", "-v"]
