"""Smoke test: scripts/calibration_analysis.py output matches the pinned reference.

Run::

    pytest tests/test_calibration.py -v

Verifies that the calibration metrics for v19 main and DeepExoMir-Lite on
all three miRBench test sets are bit-stable across runs.  Tolerance is
1e-6 because the script does no random sampling -- ECE/MCE/Brier are
deterministic given the same per-sample score files.
"""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "calibration_analysis.py"
EXPECTED = REPO / "tests" / "expected_outputs" / "calibration_metrics.tsv"


def _read_tsv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_calibration_metrics_match_pinned() -> None:
    res = subprocess.run([sys.executable, str(SCRIPT)],
                          capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, f"calibration_analysis failed:\n{res.stderr}"
    out = REPO / "results" / "calibration_metrics.tsv"
    assert out.exists(), "expected calibration_metrics.tsv was not created"

    got = _read_tsv(out)
    want = _read_tsv(EXPECTED)
    assert len(got) == len(want), f"row count mismatch: got {len(got)}, want {len(want)}"

    for g, w in zip(got, want):
        assert g["dataset"] == w["dataset"]
        assert g["model"] == w["model"]
        assert int(g["n_samples"]) == int(w["n_samples"])
        for col in ("ECE", "MCE", "Brier"):
            assert abs(float(g[col]) - float(w[col])) < 1e-6, (
                f"{g['dataset']}/{g['model']} {col}: got {g[col]} vs want {w[col]}"
            )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
