# src/dataset.py
"""Utility to download and parse the SIFT1M (or siftsmall) dataset.

The dataset consists of floating‑point vectors stored in the Texmex
`.fvecs` format and ground‑truth nearest‑neighbor indices stored in the
`.ivecs` format.

Each record in these binary files starts with a 4‑byte ``int32`` giving the
dimension *D* (always 128 for SIFT).  It is followed by *D* ``float32`` values
for ``.fvecs`` or *D* ``int32`` values for ``.ivecs``.

The module provides:
* ``download_dataset`` – fetches the required files from the public
  repository.
* ``read_fvecs`` and ``read_ivecs`` – parsers for the binary formats.
* ``validate_ground_truth`` – computes exact Euclidean (L2) distances for the
  first 1 000 query vectors against the base set and checks that the nearest
  neighbor matches the provided ground‑truth.

Running the module directly executes the download, parsing, and validation
steps.  The script exits with ``0`` on success and prints a short summary.
"""

import os
import urllib.request
import struct
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration – change these URLs if you wish to use a different source.
# ---------------------------------------------------------------------------
# Updated download logic: use the official Texmex FTP archive.
TAR_URL = "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

# After extraction, these filenames will be present.
FILES = {
    "base": "sift_base.fvecs",
    "learn": "sift_learn.fvecs",
    "query": "sift_query.fvecs",
    "groundtruth": "sift_groundtruth.ivecs",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ---------------------------------------------------------------------------
# Helper functions for downloading and parsing.
# ---------------------------------------------------------------------------

def _download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` if it does not already exist."""
    if dest.exists():
        print(f"[skip] {dest.name} already present")
        return
    print(f"[download] {url} → {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, open(dest, "wb") as out_file:
        out_file.write(response.read())
    print(f"[done] {dest.name}")


def download_dataset() -> dict:
    """Download the SIFT1M dataset archive, extract required files, and return their paths.

    Returns
    -------
    dict
        Mapping from logical name (``base``, ``learn`` …) to absolute ``Path`` objects.
    """
    archive_path = DATA_DIR / "sift.tar.gz"
    # Download tarball if not present.
    if not archive_path.exists():
        print(f"[download] {TAR_URL} → {archive_path}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TAR_URL) as response, open(archive_path, "wb") as out_file:
            out_file.write(response.read())
        print("[done] archive downloaded")
    else:
        print("[skip] archive already present")
    # Extract archive.
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    # Build paths dict.
    paths = {key: DATA_DIR / fname for key, fname in FILES.items()}
    return paths


def read_fvecs(path: Path) -> np.ndarray:
    """Read a ``.fvecs`` file.

    Parameters
    ----------
    path : Path
        Path to the ``.fvecs`` file.

    Returns
    -------
    np.ndarray
        2‑D ``float32`` array of shape ``(num_vectors, dim)``.
    """
    with open(path, "rb") as f:
        # Read the entire file into memory – files are < 1 GB for SIFT1M.
        data = f.read()
    offset = 0
    vectors = []
    while offset < len(data):
        # Dimension (int32 little‑endian)
        dim = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        vec = struct.unpack_from(f"<{dim}f", data, offset)
        offset += 4 * dim
        vectors.append(vec)
    return np.array(vectors, dtype=np.float32)


def read_ivecs(path: Path) -> np.ndarray:
    """Read a ``.ivecs`` file (used for ground‑truth indices)."""
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    rows = []
    while offset < len(data):
        dim = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        idx = struct.unpack_from(f"<{dim}i", data, offset)
        offset += 4 * dim
        rows.append(idx)
    return np.array(rows, dtype=np.int32)


def _l2_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute squared L2 distance between two sets of vectors.

    ``a`` shape: (N, D) ; ``b`` shape: (M, D)
    Returns shape (N, M).
    """
    # Using (a-b)^2 = a^2 + b^2 - 2ab trick for efficiency.
    a_sq = np.sum(a ** 2, axis=1)[:, None]
    b_sq = np.sum(b ** 2, axis=1)[None, :]
    ab = a @ b.T
    return a_sq + b_sq - 2 * ab


def validate_ground_truth(base: np.ndarray, query: np.ndarray, gt: np.ndarray, num_queries: int = 1000) -> bool:
    """Validate that the provided ground‑truth matches exact brute‑force search.

    Parameters
    ----------
    base : np.ndarray
        Base dataset vectors.
    query : np.ndarray
        Query vectors.
    gt : np.ndarray
        Ground‑truth indices (shape ``(num_queries, k)``).
    num_queries : int, optional
        Number of queries to test (default 1000).
    """
    if query.shape[0] < num_queries:
        raise ValueError("Not enough query vectors for validation")
    # Compute exact distances for the first ``num_queries``.
    distances = _l2_distance(query[:num_queries], base)
    # Argmin gives the nearest neighbour index.
    nn_idx = np.argmin(distances, axis=1)
    # Ground‑truth first neighbour.
    gt_first = gt[:num_queries, 0]
    matches = nn_idx == gt_first
    accuracy = matches.mean()
    print(f"Validation results – first‑neighbor accuracy on {num_queries} queries: {accuracy * 100:.2f}%")
    return accuracy == 1.0


def main() -> None:
    # Step 1 – download files.
    paths = download_dataset()

    # Step 2 – parse binary files.
    print("[parse] Loading base vectors…")
    base = read_fvecs(paths["base"].resolve())
    print("[parse] Loading query vectors…")
    query = read_fvecs(paths["query"].resolve())
    print("[parse] Loading ground‑truth indices…")
    gt = read_ivecs(paths["groundtruth"].resolve())

    # Basic sanity checks.
    assert base.shape[1] == 128, "Base vectors must have dimension 128"
    assert query.shape[1] == 128, "Query vectors must have dimension 128"
    assert gt.shape[0] == query.shape[0], "Ground‑truth rows must match number of queries"

    # Step 3 – validation.
    print("[validate] Computing brute‑force distances for the first 1,000 queries…")
    ok = validate_ground_truth(base, query, gt, num_queries=1000)
    if ok:
        print("[success] Ground‑truth validation passed.")
    else:
        print("[warning] Validation failed – some nearest neighbours differ.")


if __name__ == "__main__":
    main()
