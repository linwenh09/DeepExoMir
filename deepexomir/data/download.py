"""Database download module for DeepExoMir.

Downloads miRNA-target interaction databases from public repositories:
- miRBase v22.1 (mature miRNA sequences)
- miRTarBase 2025 (experimentally validated MTIs)
- ExoCarta (exosome-associated miRNA data)
- HMDD v4.0 (human miRNA disease database)

Uses requests + tqdm for progress-bar-enabled downloads, with checksum
verification and automatic retry logic.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default output directory (relative to project root)
DEFAULT_RAW_DIR = Path("data/raw")

# ============================================================================
# Known data sources
# ============================================================================

SOURCES = {
    "mirbase_mature": {
        "url": "https://www.mirbase.org/download/mature.fa",
        "filename": "mature.fa",
        "description": "miRBase v22.1 mature miRNA sequences (FASTA)",
        "sha256": None,  # Will be populated after first verified download
    },
    "exocarta_mirna": {
        "url": "http://exocarta.org/Archive/EXOCARTA_MIRNA_DETAILS_5.txt",
        "filename": "exocarta_mirna_details.txt",
        "description": "ExoCarta miRNA cargo data (tab-delimited)",
        "sha256": None,
    },
    "hmdd_v4": {
        "url": "https://www.cuilab.cn/static/hmdd3/data/alldata.txt",
        "filename": "hmdd_v4_alldata.txt",
        "description": "HMDD v4.0 miRNA-disease associations",
        "sha256": None,
    },
}

# miRTarBase requires a web-form submission and cannot be directly downloaded.
MIRTARBASE_MANUAL_URL = (
    "https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2025/cache/download/"
    "9.0/hsa_MTI.xlsx"
)


# ============================================================================
# Core download helpers
# ============================================================================


def _compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hex digest for a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 60,
    chunk_size: int = 8192,
) -> Path:
    """Download a single file with progress bar, retry, and optional checksum.

    Parameters
    ----------
    url : str
        Source URL.
    dest : Path
        Destination file path (will be created/overwritten).
    expected_sha256 : str, optional
        If provided, verify downloaded file against this digest.
    max_retries : int
        Maximum number of download attempts.
    timeout : int
        Request timeout in seconds.
    chunk_size : int
        Streaming chunk size in bytes.

    Returns
    -------
    Path
        The path to the downloaded file.

    Raises
    ------
    RuntimeError
        If all retry attempts fail or checksum does not match.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    last_error: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                "Downloading %s (attempt %d/%d)", url, attempt, max_retries
            )
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            desc = dest.name

            with (
                open(dest, "wb") as fh,
                tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=desc,
                    disable=total_size == 0,
                ) as pbar,
            ):
                for chunk in response.iter_content(chunk_size=chunk_size):
                    fh.write(chunk)
                    pbar.update(len(chunk))

            # Verify checksum if provided
            if expected_sha256 is not None:
                actual = _compute_sha256(dest)
                if actual != expected_sha256:
                    raise RuntimeError(
                        f"Checksum mismatch for {dest.name}: "
                        f"expected {expected_sha256}, got {actual}"
                    )
                logger.info("Checksum verified for %s", dest.name)

            logger.info("Downloaded %s -> %s", url, dest)
            return dest

        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            logger.warning(
                "Attempt %d failed for %s: %s", attempt, url, exc
            )
            if attempt < max_retries:
                backoff = 2 ** attempt
                logger.info("Retrying in %d seconds ...", backoff)
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to download {url} after {max_retries} attempts: {last_error}"
    )


# ============================================================================
# Public API
# ============================================================================


def download_mirbase(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download miRBase v22.1 mature.fa.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save into.  Defaults to ``data/raw/``.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded FASTA file.
    """
    info = SOURCES["mirbase_mature"]
    out_dir = output_dir or DEFAULT_RAW_DIR
    dest = out_dir / info["filename"]

    if dest.exists() and not force:
        logger.info("miRBase mature.fa already exists at %s, skipping.", dest)
        return dest

    return _download_file(
        info["url"], dest, expected_sha256=info["sha256"]
    )


def download_mirtarbase(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Attempt to download miRTarBase 2025.

    miRTarBase typically requires navigating a web form.  This function will
    first attempt a direct download of the known URL.  If that fails, it
    prints instructions for manual download.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save into.  Defaults to ``data/raw/``.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the expected output file (may not exist if manual download
        is required).
    """
    out_dir = output_dir or DEFAULT_RAW_DIR
    dest = out_dir / "hsa_MTI.xlsx"

    if dest.exists() and not force:
        logger.info("miRTarBase file already exists at %s, skipping.", dest)
        return dest

    # Attempt direct download (may fail due to web-form requirement)
    try:
        _download_file(
            MIRTARBASE_MANUAL_URL,
            dest,
            max_retries=2,
            timeout=30,
        )
        return dest
    except RuntimeError:
        logger.warning(
            "Automatic download of miRTarBase failed.  "
            "Manual download is required."
        )
        _print_mirtarbase_instructions(dest)
        return dest


def _print_mirtarbase_instructions(dest: Path) -> None:
    """Print manual-download instructions for miRTarBase."""
    msg = (
        "\n"
        "=" * 70 + "\n"
        "  miRTarBase Manual Download Instructions\n"
        "=" * 70 + "\n"
        "1. Visit: https://mirtarbase.cuhk.edu.cn/\n"
        "2. Navigate to 'Download' section.\n"
        "3. Download the Homo sapiens MTI file (hsa_MTI.xlsx).\n"
        f"4. Save it to: {dest}\n"
        "=" * 70 + "\n"
    )
    logger.info(msg)
    print(msg)


def download_exocarta(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download ExoCarta miRNA cargo data.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save into.  Defaults to ``data/raw/``.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded text file.
    """
    info = SOURCES["exocarta_mirna"]
    out_dir = output_dir or DEFAULT_RAW_DIR
    dest = out_dir / info["filename"]

    if dest.exists() and not force:
        logger.info("ExoCarta file already exists at %s, skipping.", dest)
        return dest

    return _download_file(
        info["url"], dest, expected_sha256=info["sha256"]
    )


def download_hmdd(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> Path:
    """Download HMDD v4.0 miRNA-disease association data.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save into.  Defaults to ``data/raw/``.
    force : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded text file.
    """
    info = SOURCES["hmdd_v4"]
    out_dir = output_dir or DEFAULT_RAW_DIR
    dest = out_dir / info["filename"]

    if dest.exists() and not force:
        logger.info("HMDD file already exists at %s, skipping.", dest)
        return dest

    return _download_file(
        info["url"], dest, expected_sha256=info["sha256"]
    )


def download_all(
    output_dir: Optional[Path] = None,
    force: bool = False,
) -> dict[str, Path]:
    """Download all available databases.

    Parameters
    ----------
    output_dir : Path, optional
        Directory to save into.  Defaults to ``data/raw/``.
    force : bool
        Re-download even if files already exist.

    Returns
    -------
    dict[str, Path]
        Mapping of database name to downloaded file path.
    """
    results: dict[str, Path] = {}

    for name, func in [
        ("mirbase_mature", download_mirbase),
        ("mirtarbase", download_mirtarbase),
        ("exocarta", download_exocarta),
        ("hmdd_v4", download_hmdd),
    ]:
        try:
            results[name] = func(output_dir=output_dir, force=force)
        except Exception as exc:
            logger.error("Failed to download %s: %s", name, exc)
            results[name] = Path("")  # sentinel for failure

    return results


def verify_checksum(filepath: Path, expected_sha256: str) -> bool:
    """Verify SHA-256 checksum of an existing file.

    Parameters
    ----------
    filepath : Path
        Path to the file.
    expected_sha256 : str
        Expected hex digest.

    Returns
    -------
    bool
        True if checksum matches.
    """
    if not filepath.exists():
        logger.warning("File does not exist: %s", filepath)
        return False

    actual = _compute_sha256(filepath)
    match = actual == expected_sha256
    if not match:
        logger.warning(
            "Checksum mismatch for %s: expected %s, got %s",
            filepath,
            expected_sha256,
            actual,
        )
    return match
