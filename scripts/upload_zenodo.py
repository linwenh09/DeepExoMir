"""Automate the New-Version upload of deepexomir_zenodo_v1.1.0.zip to Zenodo.

This script does **everything except press Publish**.  It:

  1. Calls the Zenodo REST API to start a New Version draft from the
     existing record (DOI 10.5281/zenodo.19216306).
  2. Deletes the old v1.0.0 zip from that draft.
  3. Uploads ``deepexomir_zenodo_v1.1.0.zip`` from the repository root.
  4. Sets the ``version`` field to ``1.1.0`` and replaces the
     description with the v1.1.0 release text.
  5. Prints the draft's URL so you can open it in the browser, review,
     and click **Publish** by hand.

Why the script stops short of Publish: publishing on Zenodo is
irreversible (the file list and DOI are frozen forever after publish).
The script intentionally leaves the final review-and-publish step to
the human owner of the deposit.

Usage
-----

1. Generate a personal access token at:

       https://zenodo.org/account/settings/applications/tokens/new/

   Tick the scopes:  ``deposit:write``  and  ``deposit:actions``.
   Copy the token (Zenodo only shows it once).

2. Set it as an environment variable, then run the script::

       # bash / git-bash on Windows
       export ZENODO_TOKEN='paste-the-token-here'
       python scripts/upload_zenodo.py

       # PowerShell
       $env:ZENODO_TOKEN = 'paste-the-token-here'
       python scripts/upload_zenodo.py

3. The script will print a URL of the form
   ``https://zenodo.org/uploads/<NEW_ID>``.  Open it in the browser,
   review the metadata + file list, and click **Publish**.

The token is read from the environment, never from the command line, so
it does not appear in your shell history.  The script does not log,
echo, or persist the token.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ZENODO_BASE = "https://zenodo.org/api"
PARENT_RECORD_ID = "19216306"
NEW_VERSION = "1.1.0"

REPO = Path(__file__).resolve().parents[1]
ZIP_PATH = REPO / "deepexomir_zenodo_v1.1.0.zip"

DESCRIPTION = """\
DeepExoMir v1.1.0 -- the IJMS submission snapshot for "DeepExoMir: A
Reproducible RNA Language Model Framework for CLIP-seq-Supported
MicroRNA Target-Site Prioritization".  Supersedes v1.0.0 (the original
BIB submission deposit) by adding:

  * a public reproducibility surface (deepexomir.config /
    deepexomir.predict / deepexomir.benchmark, with command-line
    scripts/reproduce_table1.py, scripts/reproduce_table3.py,
    scripts/calibration_analysis.py)
  * pytest smoke tests with pinned references (tests/test_*.py +
    tests/expected_outputs/), runnable in ~3 min CPU on a clean venv
    with only numpy + pandas + scikit-learn + matplotlib + pyyaml +
    pytest installed
  * per-sample probability score files for v19, DeepExoMir-Lite, three
    retrain-from-scratch ablations, and eight miRBench baselines on the
    three CLIP-seq test sets
  * a CPU-only smoke-test Dockerfile at the repository root
  * the four trained checkpoints used in Tables 1 and 3 (v19,
    v19_noStructure / Lite, v19_noConservation, v19_noRNALM)
  * the calibration analysis from manuscript Section 2.6

The PCA-reduced RiNALMo embeddings cache (~32 GB) is not included
because it is regenerable from the published RiNALMo-giga checkpoint.
The miRBench raw parquets are also not mirrored because they are
available through the miRBench Python package.

See RELEASE_NOTES.md inside the zip for the full diff against v1.0.0.

License: MIT (https://github.com/linwenh09/DeepExoMir/blob/master/LICENSE)
Active development: https://github.com/linwenh09/DeepExoMir
"""


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _hdrs(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _check(r: requests.Response, where: str) -> dict:
    if r.status_code >= 300:
        sys.exit(f"FAILED at {where}: {r.status_code}\n{r.text[:1000]}")
    return r.json() if r.content else {}


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> int:
    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        sys.exit(
            "ZENODO_TOKEN environment variable is not set.\n"
            "Generate a token at https://zenodo.org/account/settings/applications/tokens/new/\n"
            "with scopes  deposit:write  and  deposit:actions.\n"
            "Then:\n"
            "  bash:       export ZENODO_TOKEN='...'\n"
            "  PowerShell: $env:ZENODO_TOKEN = '...'"
        )

    if not ZIP_PATH.exists():
        sys.exit(f"missing {ZIP_PATH} -- run scripts/build_zenodo_release.py first")

    size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"Source archive: {ZIP_PATH.name} ({size_mb:.1f} MB)")

    # -----------------------------------------------------------------
    # Step 1: New Version
    # -----------------------------------------------------------------
    print(f"Step 1: starting New Version on Zenodo record {PARENT_RECORD_ID} ...")
    r = requests.post(
        f"{ZENODO_BASE}/deposit/depositions/{PARENT_RECORD_ID}/actions/newversion",
        headers=_hdrs(token),
        timeout=60,
    )
    info = _check(r, "newversion")
    new_draft_url = info["links"]["latest_draft"]
    new_id = new_draft_url.rstrip("/").rsplit("/", 1)[-1]
    print(f"  new draft id = {new_id}")

    # -----------------------------------------------------------------
    # Step 2: Inspect the draft so we know what to delete
    # -----------------------------------------------------------------
    print("Step 2: inspecting the new draft ...")
    r = requests.get(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        headers=_hdrs(token),
        timeout=60,
    )
    draft = _check(r, "get draft")
    bucket_url = draft["links"]["bucket"]
    old_files = draft.get("files", [])
    print(f"  draft has {len(old_files)} carry-over file(s) from v1.0.0")

    # -----------------------------------------------------------------
    # Step 3: Delete carry-over files
    # -----------------------------------------------------------------
    for f in old_files:
        fid = f["id"]
        fname = f.get("filename", "<unknown>")
        print(f"Step 3: deleting carry-over file '{fname}' ...")
        r = requests.delete(
            f"{ZENODO_BASE}/deposit/depositions/{new_id}/files/{fid}",
            headers=_hdrs(token),
            timeout=60,
        )
        if r.status_code >= 300:
            sys.exit(f"failed to delete {fname}: {r.status_code}\n{r.text[:500]}")

    # -----------------------------------------------------------------
    # Step 4: Upload the v1.1.0 zip via the bucket API
    # -----------------------------------------------------------------
    print(f"Step 4: uploading {ZIP_PATH.name} ({size_mb:.1f} MB) ...")
    with ZIP_PATH.open("rb") as fh:
        r = requests.put(
            f"{bucket_url}/{ZIP_PATH.name}",
            headers=_hdrs(token),
            data=fh,
            timeout=600,
        )
    _check(r, "upload zip")
    print("  upload OK")

    # -----------------------------------------------------------------
    # Step 5: Patch metadata (version + description only)
    # -----------------------------------------------------------------
    print("Step 5: setting version and description ...")
    existing = draft["metadata"]
    existing["version"] = NEW_VERSION
    existing["description"] = DESCRIPTION
    # Belt-and-suspenders: enforce MIT and Software in case v1.0.0 is
    # mis-categorised.  Comment these two lines out if you want to keep
    # whatever was on the v1.0.0 record verbatim.
    existing["license"] = "MIT"
    existing["upload_type"] = "software"

    r = requests.put(
        f"{ZENODO_BASE}/deposit/depositions/{new_id}",
        headers={**_hdrs(token), "Content-Type": "application/json"},
        data=json.dumps({"metadata": existing}),
        timeout=60,
    )
    _check(r, "patch metadata")
    print("  metadata OK")

    # -----------------------------------------------------------------
    # Step 6: Print the draft URL for human review + publish
    # -----------------------------------------------------------------
    review_url = f"https://zenodo.org/uploads/{new_id}"
    print()
    print("=" * 72)
    print("DRAFT READY for review.")
    print(f"  Open: {review_url}")
    print()
    print("In the browser:")
    print("  1. Verify the file list shows only deepexomir_zenodo_v1.1.0.zip")
    print("  2. Verify Version = 1.1.0, License = MIT, Type = Software")
    print("  3. Click Publish.")
    print()
    print("This script intentionally does NOT press Publish.  Publishing on")
    print("Zenodo is irreversible (the DOI and file list are frozen forever),")
    print("so a human owner must do that step.")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
