# Zenodo upload guide -- DeepExoMir v0.2.0 (GPB submission)

Step-by-step instructions for replacing the existing Zenodo deposit
(DOI 10.5281/zenodo.19216306) with the new v0.2.0 release.  This is a
**user action** -- it requires your Zenodo login.

## TL;DR

1. Build the new zip (already done by `scripts/build_zenodo_release.py`):
   `deepexomir_zenodo_v0.2.0.zip` at the repo root.
2. Log in to <https://zenodo.org>.
3. Open the existing record: <https://zenodo.org/records/19216306>.
4. Click **New version** (top right, under "Versions" panel).
5. Delete the old zip from the new draft and upload `deepexomir_zenodo_v0.2.0.zip`.
6. Update version field to `0.2.0`, paste the new description (below).
7. Click **Save**, then **Publish**.

Publishing creates a **new DOI** for v0.2.0 (e.g. `10.5281/zenodo.NNNNNN`).
The original DOI (`10.5281/zenodo.19216306`) keeps resolving to v0.1.0 --
**this is fine**.  The "concept DOI" (which the manuscript should cite if
you want "always the latest version") is the same across all versions
and is shown in the Zenodo record's right-hand sidebar.

The manuscript currently cites the **concept DOI** equivalent
`10.5281/zenodo.19216306`, which Zenodo will automatically redirect to
the latest version after publishing v0.2.0.  No manuscript edit needed.

## Detailed walk-through

### 1. Confirm the build

```bash
ls -lh deepexomir_zenodo_v0.2.0.zip
unzip -l deepexomir_zenodo_v0.2.0.zip | tail -5
```

Expected: ~700-800 MB compressed, ~1300 files.  The last line of `unzip
-l` should show four ``checkpoints/v19*/checkpoint_*.pt`` entries.

### 2. Open the existing record

Go to <https://zenodo.org/records/19216306>.  You should see the v0.1.0
deposit titled "DeepExoMir: ..." with the existing zip listed.

### 3. Start a new version

Look for the **"Versions"** panel on the right-hand side of the record.
Click the **"New version"** button.  Zenodo opens a draft pre-populated
with all v0.1.0 metadata.

### 4. Replace the file

In the draft's "Files" section:

* **Delete** the old `deepexomir_zenodo.zip` (click the trash icon).
* **Upload** the new `deepexomir_zenodo_v0.2.0.zip` (drag & drop, or
  click the upload area).
* Wait for the upload to finish -- the progress bar must reach 100% and
  the file must show "Pending" or "Available".  Do not navigate away.

### 5. Update the metadata

Most fields stay the same.  The two that change are **Version** and
**Description**.

#### Version

Change from `1.0` (or whatever v0.1.0 used) to `0.2.0`.

#### Description

Replace with the following text (copy verbatim):

```
DeepExoMir v0.2.0 -- the GPB submission snapshot for "DeepExoMir: A
Reproducible RNA Language Model Framework for CLIP-seq-Supported
MicroRNA Target-Site Prioritization".  Supersedes v0.1.0 (the original
BIB submission deposit) by adding:

  * a public reproducibility surface (deepexomir.config /
    deepexomir.predict / deepexomir.benchmark, with command-line
    scripts/reproduce_table1.py, scripts/reproduce_table3.py,
    scripts/calibration_analysis.py)
  * pytest smoke tests with pinned references (tests/test_*.py +
    tests/expected_outputs/), runnable in ~3 min CPU
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

See RELEASE_NOTES.md inside the zip for the full diff against v0.1.0.

License: MIT (https://github.com/linwenh09/DeepExoMir/blob/master/LICENSE)
Active development: https://github.com/linwenh09/DeepExoMir
```

#### Other fields to **leave alone**

* Title -- keep "DeepExoMir: A Reproducible RNA Language Model Framework
  for CLIP-seq-Supported MicroRNA Target-Site Prioritization" (update if
  the v0.1.0 title was different).
* Authors -- Wen-Hsien Lin, Chia-Ni Hsiung, Wen-Yu Lien, Martin Sieber.
* License -- MIT.
* Resource type -- Software.
* Keywords -- microRNA, target prediction, RNA language model, CLIP-seq,
  reproducibility, miRBench, RiNALMo.
* Related identifiers -- if v0.1.0 already has the GitHub URL listed as
  "is supplemented by", keep it; otherwise add it.

### 6. Save and publish

* Click **Save** (this saves the draft without publishing).
* Review the preview pane on the right.
* Click **Publish** when satisfied.

Publishing is **irreversible**: you cannot delete a published Zenodo
record (only restrict access).  Double-check the file list and
description before clicking Publish.

### 7. Capture the new DOI

After publishing, the new record's URL will be of the form
`https://zenodo.org/records/<NNNNNNNN>` and its DOI of the form
`10.5281/zenodo.<NNNNNNNN>`.

The **concept DOI** (always-latest) is shown in the right sidebar as
"Cite all versions" and remains `10.5281/zenodo.19216306`.

The manuscript's BibTeX entry already uses the concept DOI, so no edit
is needed there.

If the GPB submission system requires a "Data availability" URL, use
either:

* the concept-record URL: `https://zenodo.org/records/19216306`
* or the v0.2.0 record URL: `https://zenodo.org/records/<NNNNNNNN>`

Either works; the concept URL is preferable for long-term citation.

## Troubleshooting

### "File too large"

Zenodo's per-file cap is 50 GB and per-record cap is 50 GB.  This zip is
~750 MB so neither limit applies.  If upload fails, retry on a wired
connection -- Zenodo's flaky CDN occasionally truncates large uploads
mid-stream.

### "Cannot edit metadata of published record"

Once published, only the description and a few other fields can be
edited (use the "Edit" button on the published record).  The file list
is frozen.  If you uploaded the wrong zip, create *another* new version
rather than trying to edit in place.

### "DOI doesn't appear in CrossRef"

Zenodo registers DOIs with DataCite, not CrossRef.  DataCite indexing
typically completes within 60 minutes of publishing.  The DOI will
resolve immediately via doi.org regardless.
