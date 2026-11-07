"""SIFT1M workload loader used by :mod:`bench.runner`.

This module is intentionally separate from ``src/dataset.py``: that file is a
TEXMEX ``.fvecs``/``.ivecs`` downloader/validator maintained by the dataset
side of the project.  The benchmark only needs a uniform in-memory workload
(``base``, ``queries``, ``ground_truth``), so it uses the ANN-Benchmarks SIFT1M
HDF5 bundle and can recompute exact ground truth for subsets.
"""

from __future__ import annotations

import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

SIFT1M_URL = "https://ann-benchmarks.com/sift-128-euclidean.hdf5"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Dataset:
    """A vector-search workload plus exact top-k ground truth."""

    name: str
    base: np.ndarray
    queries: np.ndarray
    ground_truth: np.ndarray
    path: Optional[str] = None

    @property
    def dim(self) -> int:
        return int(self.base.shape[1])

    @property
    def n_base(self) -> int:
        return int(self.base.shape[0])

    @property
    def n_queries(self) -> int:
        return int(self.queries.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, n_base={self.n_base:,}, "
            f"n_queries={self.n_queries:,}, dim={self.dim}, "
            f"gt_k={self.ground_truth.shape[1]})"
        )


def read_fvecs(path: str | os.PathLike) -> np.ndarray:
    """Read a TEXMEX ``.fvecs`` file into an ``(n, d)`` float32 array."""

    raw = np.fromfile(path, dtype=np.float32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    dim = int(raw[0].view(np.int32))
    if dim <= 0:
        raise ValueError(f"Invalid fvecs dimension {dim} in {path}")
    rows = raw.size // (dim + 1)
    return raw.reshape(rows, dim + 1)[:, 1:].astype(np.float32, copy=False)


def read_ivecs(path: str | os.PathLike) -> np.ndarray:
    """Read a TEXMEX ``.ivecs`` file into an ``(n, d)`` int32 array."""

    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        return np.empty((0, 0), dtype=np.int32)
    dim = int(raw[0])
    rows = raw.size // (dim + 1)
    return raw.reshape(rows, dim + 1)[:, 1:].astype(np.int32, copy=False)


def _download(url: str, destination: Path, chunk_size: int = 1 << 20) -> None:
    """Download *url* atomically to *destination* with a small progress meter."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=destination.name, dir=destination.parent)
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        print(f"[dataset] downloading {url}\n[dataset] -> {destination}", flush=True)
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            next_report = 0
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total and downloaded >= next_report:
                    pct = 100.0 * downloaded / total
                    print(
                        f"[dataset]   {downloaded / 1e6:7.1f} / "
                        f"{total / 1e6:7.1f} MB ({pct:5.1f}%)",
                        flush=True,
                    )
                    next_report = downloaded + 100 * 1_000_000
        os.replace(tmp_path, destination)
        print(
            f"[dataset] saved {destination} "
            f"({destination.stat().st_size / 1e6:.1f} MB)"
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _exact_ground_truth(
    base: np.ndarray,
    queries: np.ndarray,
    k: int,
    chunk: int = 128,
) -> np.ndarray:
    """Exact L2 top-k using FAISS, chunked to keep peak memory bounded."""

    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - dependency error
        raise RuntimeError("FAISS is required to recompute exact ground truth") from exc

    index = faiss.IndexFlatL2(base.shape[1])
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    out = np.empty((queries.shape[0], k), dtype=np.int64)
    for start in range(0, queries.shape[0], chunk):
        stop = min(start + chunk, queries.shape[0])
        _, ids = index.search(
            np.ascontiguousarray(queries[start:stop], dtype=np.float32), k
        )
        out[start:stop] = ids.astype(np.int64)
    return out


def load_sift1m(
    path: str | os.PathLike | None = None,
    *,
    data_dir: str | os.PathLike | None = None,
    subset: int | None = None,
    n_queries: int | None = None,
    gt_k: int = 100,
    download: bool = True,
    recompute_gt: bool = False,
) -> Dataset:
    """Load SIFT1M from an HDF5 file, downloading it if necessary."""

    import h5py

    if path is None:
        root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
        path = root / "sift-128-euclidean.hdf5"
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Dataset not found: {path}")
        _download(SIFT1M_URL, path)

    with h5py.File(path, "r") as handle:
        if "train" not in handle or "test" not in handle:
            raise ValueError(f"{path} does not contain 'train'/'test' datasets")
        base = np.asarray(handle["train"], dtype=np.float32)
        queries = np.asarray(handle["test"], dtype=np.float32)
        provided_gt = (
            np.asarray(handle["neighbors"], dtype=np.int64)
            if "neighbors" in handle
            else None
        )

    n_total = base.shape[0]
    if subset is not None and 0 < subset < n_total:
        base = base[:subset]
    if n_queries is not None and 0 < n_queries < queries.shape[0]:
        queries = queries[:n_queries]

    use_provided = (
        not recompute_gt
        and provided_gt is not None
        and base.shape[0] == n_total
        and provided_gt.shape[0] >= queries.shape[0]
        and provided_gt.shape[1] >= gt_k
    )
    if use_provided:
        ground_truth = np.ascontiguousarray(provided_gt[: queries.shape[0], :gt_k])
    else:
        if gt_k > base.shape[0]:
            raise ValueError("gt_k cannot exceed the number of base vectors")
        print(
            f"[dataset] computing exact top-{gt_k} ground truth on "
            f"{base.shape[0]:,} base vectors x {queries.shape[0]:,} queries",
            flush=True,
        )
        ground_truth = _exact_ground_truth(base, queries, gt_k)

    name = "SIFT1M" if base.shape[0] == n_total else f"SIFT1M[{base.shape[0]}]"
    return Dataset(
        name=name,
        base=np.ascontiguousarray(base, dtype=np.float32),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        ground_truth=ground_truth,
        path=str(path),
    )


def load_fvecs(
    base_path: str | os.PathLike,
    query_path: str | os.PathLike,
    *,
    gt_path: str | os.PathLike | None = None,
    gt_k: int = 100,
    subset: int | None = None,
    n_queries: int | None = None,
    recompute_gt: bool = False,
) -> Dataset:
    """Load TEXMEX ``.fvecs``/``.ivecs`` files."""

    base = read_fvecs(base_path)
    queries = read_fvecs(query_path)
    if subset is not None and 0 < subset < base.shape[0]:
        base = base[:subset]
    if n_queries is not None and 0 < n_queries < queries.shape[0]:
        queries = queries[:n_queries]

    if gt_path is not None and not recompute_gt:
        ground_truth = read_ivecs(gt_path)[: queries.shape[0], :gt_k].astype(np.int64)
    else:
        ground_truth = _exact_ground_truth(base, queries, gt_k)

    return Dataset(
        name=Path(base_path).stem,
        base=np.ascontiguousarray(base, dtype=np.float32),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        ground_truth=ground_truth,
        path=str(base_path),
    )


def make_synthetic(
    n_base: int = 20_000,
    n_queries: int = 500,
    dim: int = 128,
    n_clusters: int = 64,
    gt_k: int = 100,
    seed: int = 42,
) -> Dataset:
    """Create a small clustered workload for smoke tests."""

    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim)).astype(np.float32)
    labels = rng.integers(0, n_clusters, size=n_base)
    base = centres[labels] + rng.normal(
        scale=0.35, size=(n_base, dim)
    ).astype(np.float32)
    q_labels = rng.integers(0, n_clusters, size=n_queries)
    queries = centres[q_labels] + rng.normal(
        scale=0.35, size=(n_queries, dim)
    ).astype(np.float32)
    ground_truth = _exact_ground_truth(base, queries, gt_k)
    return Dataset(
        name=f"synthetic-{n_base}",
        base=np.ascontiguousarray(base, dtype=np.float32),
        queries=np.ascontiguousarray(queries, dtype=np.float32),
        ground_truth=ground_truth,
    )


__all__ = [
    "Dataset",
    "load_sift1m",
    "load_fvecs",
    "make_synthetic",
    "read_fvecs",
    "read_ivecs",
]
