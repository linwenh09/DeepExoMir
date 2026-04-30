#!/usr/bin/env bash
#
# Verify the DeepExoMir reproducibility container end-to-end.
#
# Usage (run from the repository root, with Docker Desktop running):
#
#   bash scripts/verify_docker.sh
#
# What it does:
#   1. Builds the smoke-test image with `docker build -t deepexomir:latest .`
#   2. Runs the two smoke tests inside the container:
#        pytest tests/test_reproduce_table3.py tests/test_calibration.py -v
#   3. Prints a final PASS/FAIL line.
#
# Expected runtime: ~5 min on first build (pip install of numpy, pandas,
# scikit-learn, matplotlib, pytest).  Re-run is ~3 min (Docker layer cache
# hits, smoke tests still run).

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO}"

echo "=== Step 1/3: docker build ==="
docker build -t deepexomir:latest . | tail -20
echo

echo "=== Step 2/3: docker run smoke tests ==="
docker run --rm -v "${REPO}:/workspace" -w /workspace deepexomir:latest \
    pytest tests/test_calibration.py tests/test_reproduce_table3.py -v
echo

echo "=== Step 3/3: image size ==="
docker images deepexomir:latest --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}'
echo

echo "PASS: Docker container reproduces the smoke-test results."
