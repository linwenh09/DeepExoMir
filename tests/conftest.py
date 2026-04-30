"""pytest fixtures and collection rules for the DeepExoMir test suite.

The test directory contains a mixture of:

  * **Smoke tests** -- ``test_reproduce_table3.py`` and
    ``test_calibration.py``.  These read pre-computed per-sample
    probability score files and produce the paper's Table 3 + Figure 10
    bit-deterministically.  They do **not** require torch, CUDA, or any
    of the model training data; they run in ~3 minutes on a CPU-only
    laptop and are the entry point used by the README, the Dockerfile,
    and ``scripts/verify_docker.sh``.

  * **Internal tests** -- ``test_model.py``, ``test_training.py``,
    ``test_v8_training.py``, ``test_data_pipeline.py``,
    ``test_annotation.py``.  These exercise the live training stack and
    require torch, an installed CUDA toolchain (some explicitly call
    ``.cuda()``), the full miRBench parquet files, and the embeddings
    cache.  They are useful during local development on a GPU host and
    are skipped automatically when those prerequisites are missing.

This file makes ``pytest tests/`` succeed cleanly in the CPU-only
Docker image and on a barebones host install (``pip install -e .``).
On a fully provisioned GPU host all tests run.

The mechanism is intentionally minimal: we collect each module and skip
those that fail to import their heavy prerequisites or whose first lines
require CUDA.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SMOKE_TESTS = {
    "test_reproduce_table3",
    "test_calibration",
}


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def _cuda_available() -> bool:
    if not _torch_available():
        return False
    try:
        import torch  # noqa: WPS433  -- intentional lazy import
    except Exception:
        return False
    return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Skip non-smoke tests when their heavy prerequisites are missing.

    Smoke tests run unconditionally; everything else is skipped at
    collection time on environments without torch + CUDA, so the suite
    stays green on the published Docker image and on plain ``pip
    install -e .`` checkouts.
    """
    if _cuda_available():
        return  # full GPU host -- run everything

    has_torch = _torch_available()
    skip_no_torch = pytest.mark.skip(reason="torch not installed; CPU-only smoke build")
    skip_no_cuda = pytest.mark.skip(
        reason="CUDA not available; this test is gated to GPU hosts via tests/conftest.py",
    )

    for item in items:
        module_name = Path(str(item.fspath)).stem
        if module_name in _SMOKE_TESTS:
            continue
        if not has_torch:
            item.add_marker(skip_no_torch)
        else:
            item.add_marker(skip_no_cuda)
