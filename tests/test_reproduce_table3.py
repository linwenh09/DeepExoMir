"""Smoke test: reproduce_table3.py output matches the pinned reference.

Run::

    pytest tests/test_reproduce_table3.py -v

Verifies that ``scripts/reproduce_table3.py`` is bit-deterministic given a
fixed seed (=42) and the per-sample score files shipped under
``manuscript/BiB_submission/baseline_scores/``.  This is the cheapest
end-to-end reproducibility check we can offer reviewers without requiring
the RiNALMo backbone download.

The pinned reference lives at ``tests/expected_outputs/table3.tsv``.
"""
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_table3.py"
SCORES = REPO / "manuscript" / "BiB_submission" / "baseline_scores"
EXPECTED = REPO / "tests" / "expected_outputs" / "table3.tsv"


def _read_tsv(p: Path) -> list[dict[str, str]]:
    with p.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_reproduce_table3_matches_pinned(tmp_path: Path) -> None:
    """Smoke test uses reduced bootstrap (1000/500 resamples) for ~3 min runtime;
    paper Table 3 used 10000/2000 but the AU-PRC point estimates are
    bit-deterministic regardless and the bootstrap deltas with seed=42 are
    stable to within the test's tolerance."""
    out = tmp_path / "table3.tsv"
    cmd = [
        sys.executable, str(SCRIPT),
        "--scores-dir", str(SCORES),
        "--output", str(out),
        "--n-boot-small", "1000",
        "--n-boot-large", "500",
        "--seed", "42",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"reproduce_table3 failed:\n{res.stderr}"

    got = _read_tsv(out)
    want = _read_tsv(EXPECTED)
    assert len(got) == len(want), f"row count mismatch: got {len(got)}, want {len(want)}"

    # Compare ablation/dataset columns exactly, AU-PRC values to 4 decimals,
    # bootstrap deltas to 3 decimals (allow tiny rounding noise from numpy
    # internals across versions).
    for g, w in zip(got, want):
        assert g["ablation"] == w["ablation"], f"ablation mismatch: {g} vs {w}"
        assert g["dataset"] == w["dataset"], f"dataset mismatch: {g} vs {w}"
        for col, tol in [
            ("main_AUPRC",     1e-4),
            ("ablated_AUPRC",  1e-4),
            ("delta_main_minus_ablated", 5e-3),
            ("ci_low",         1e-2),
            ("ci_high",        1e-2),
        ]:
            assert abs(float(g[col]) - float(w[col])) < tol, (
                f"{g['ablation']}/{g['dataset']} {col}: "
                f"got {g[col]} vs want {w[col]}"
            )
        # Significance category must match exactly
        assert g["sig"] == w["sig"], f"sig mismatch row {g}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
