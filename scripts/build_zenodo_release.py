"""Build the Zenodo release zip for DeepExoMir v0.2.0.

Produces ``deepexomir_zenodo_v0.2.0.zip`` at the repository root.  The zip
mirrors the directory layout the README describes, so reviewers can extract
into an empty directory and the cross-references to ``checkpoints/...`` and
``manuscript/BiB_submission/baseline_scores/`` resolve out of the box.

Usage::

    python scripts/build_zenodo_release.py [--output PATH]

This script is intentionally kept small and readable: the canonical
release manifest is the ``MANIFEST`` list below, not an external config
file.
"""
from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Files / directories to include.  Tuples are (source_path_on_disk,
# arcname_in_zip).  Globs are expanded under each entry below.
MANIFEST: list[tuple[Path, str]] = [
    # --- top-level project files ----------------------------------------
    (REPO / "README.md",          "README.md"),
    (REPO / "RELEASE_NOTES.md",   "RELEASE_NOTES.md"),
    (REPO / "LICENSE",            "LICENSE"),
    (REPO / "pyproject.toml",     "pyproject.toml"),
    (REPO / "requirements.txt",   "requirements.txt"),
    (REPO / "Dockerfile",         "Dockerfile"),
    (REPO / ".dockerignore",      ".dockerignore"),

    # --- the four checkpoints referenced in the paper -------------------
    (REPO / "checkpoints" / "v19" / "checkpoint_epoch034_val_auc_0.8521.pt",
     "checkpoints/v19/checkpoint_epoch034_val_auc_0.8521.pt"),
    (REPO / "checkpoints" / "v19_noStructure" / "checkpoint_epoch010_val_auc_0.8238.pt",
     "checkpoints/v19_noStructure/checkpoint_epoch010_val_auc_0.8238.pt"),
    (REPO / "checkpoints" / "v19_noConservation" / "checkpoint_epoch034_val_auc_0.8535.pt",
     "checkpoints/v19_noConservation/checkpoint_epoch034_val_auc_0.8535.pt"),
    (REPO / "checkpoints" / "v19_noRNALM" / "checkpoint_epoch042_val_auc_0.8410.pt",
     "checkpoints/v19_noRNALM/checkpoint_epoch042_val_auc_0.8410.pt"),

    # --- reproducibility documentation ----------------------------------
    (REPO / "REPRODUCIBILITY.md",   "REPRODUCIBILITY.md"),
    (REPO / "ZENODO_UPLOAD_GUIDE.md", "ZENODO_UPLOAD_GUIDE.md"),
]

# Whole directory trees to include (excluding pycache / pyc / large
# accidental artefacts).  Each entry is (source_dir, arcname_dir).
DIR_MANIFEST: list[tuple[Path, str]] = [
    (REPO / "deepexomir",                                                "deepexomir"),
    (REPO / "scripts",                                                   "scripts"),
    (REPO / "tests",                                                     "tests"),
    (REPO / "configs",                                                   "configs"),
    (REPO / "manuscript" / "BiB_submission" / "baseline_scores",
     "manuscript/BiB_submission/baseline_scores"),
    (REPO / "notebooks",                                                 "notebooks"),
]

# Substrings / suffixes to skip when walking DIR_MANIFEST.
SKIP_NAMES = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ipynb_checkpoints"}
SKIP_SUFFIX = (".pyc", ".pyo", ".pyd", ".log", ".tmp")


def _walk(src: Path, arc: str):
    """Yield (file_path, arcname) for each file under ``src``."""
    for root, dirs, files in os.walk(src):
        dirs[:] = [d for d in dirs if d not in SKIP_NAMES]
        for fname in files:
            if fname.endswith(SKIP_SUFFIX):
                continue
            fp = Path(root) / fname
            rel = fp.relative_to(src).as_posix()
            yield fp, f"{arc}/{rel}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=REPO / "deepexomir_zenodo_v0.2.0.zip",
        help="output zip path (default: deepexomir_zenodo_v0.2.0.zip)",
    )
    args = parser.parse_args()

    out = args.output
    if out.exists():
        out.unlink()

    print(f"Writing {out} ...")
    n_files = 0
    total_bytes = 0
    with zipfile.ZipFile(
        out, "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as zf:
        for src, arcname in MANIFEST:
            if not src.exists():
                print(f"  WARN: missing {src}", file=sys.stderr)
                continue
            zf.write(src, arcname)
            n_files += 1
            total_bytes += src.stat().st_size

        for src_dir, arc in DIR_MANIFEST:
            if not src_dir.is_dir():
                print(f"  WARN: missing dir {src_dir}", file=sys.stderr)
                continue
            for fp, arcname in _walk(src_dir, arc):
                zf.write(fp, arcname)
                n_files += 1
                total_bytes += fp.stat().st_size

    print(f"  OK  -- {n_files} files, "
          f"{total_bytes / 1024 / 1024:.1f} MB raw, "
          f"{out.stat().st_size / 1024 / 1024:.1f} MB compressed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
