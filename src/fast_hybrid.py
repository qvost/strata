"""ctypes wrapper around the compiled two-stage traversal kernel.

The kernel is optional: :class:`bench.runner` falls back to the pure-Python
reference implementation in :mod:`src.hybrid` when a C++ compiler is not
available.  The two implementations follow the same algorithm and produce the
same candidate ordering up to floating-point tie-breaking.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .hnsw import FAISSGraph
from .quantization import BQCodec, SQ8Codec

_LIB_SOURCE = Path(__file__).resolve().parents[1] / "bench" / "fast_hybrid.cpp"
# Keep this in sync with bench/build_fast_hybrid.sh.
_LIB_PATH = _LIB_SOURCE.with_name("libfast_hybrid.so")


class _GraphView(ctypes.Structure):
    _fields_ = [
        ("levels", ctypes.POINTER(ctypes.c_int32)),
        ("offsets", ctypes.POINTER(ctypes.c_int64)),
        ("neighbors", ctypes.POINTER(ctypes.c_int32)),
        ("ntotal", ctypes.c_int64),
        ("M0", ctypes.c_int32),
        ("M", ctypes.c_int32),
        ("entry", ctypes.c_int32),
        ("max_level", ctypes.c_int32),
    ]


def build_library(force: bool = False) -> Path:
    """Compile ``fast_hybrid.cpp`` if the shared object is missing/stale."""

    if not force and _LIB_PATH.exists():
        if _LIB_PATH.stat().st_mtime >= _LIB_SOURCE.stat().st_mtime:
            return _LIB_PATH
    _LIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "g++",
        "-O3",
        "-march=native",
        "-std=c++20",
        "-shared",
        "-fPIC",
        str(_LIB_SOURCE),
        "-o",
        str(_LIB_PATH),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return _LIB_PATH


def library_available() -> bool:
    try:
        build_library()
        return True
    except (OSError, subprocess.SubprocessError):
        return False


class FastHybridSearcher:
    """Batch two-stage search backed by the compiled C++ kernel."""

    def __init__(
        self,
        graph: FAISSGraph,
        bq: BQCodec,
        sq8: SQ8Codec,
        base: np.ndarray,
        bq_codes: np.ndarray,
        sq8_codes: np.ndarray,
        *,
        rerank: str = "sq8",
    ) -> None:
        if rerank not in {"sq8", "fp32"}:
            raise ValueError("rerank must be 'sq8' or 'fp32'")
        self.graph = graph
        self.bq = bq
        self.sq8 = sq8
        self.base = np.ascontiguousarray(base, dtype=np.float32)
        self.bq_codes = np.ascontiguousarray(bq_codes, dtype=np.uint64)
        self.sq8_codes = np.ascontiguousarray(sq8_codes, dtype=np.uint8)
        self.rerank = rerank

        self._levels = np.ascontiguousarray(graph.levels, dtype=np.int32)
        self._offsets = np.ascontiguousarray(graph.offsets, dtype=np.int64)
        self._neighbors = np.ascontiguousarray(graph.neighbors, dtype=np.int32)
        self._scales2 = np.ascontiguousarray(sq8.scales2, dtype=np.float32)

        self._view = _GraphView(
            self._levels.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            self._offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            self._neighbors.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(graph.ntotal),
            int(graph.M0),
            int(graph.M),
            int(graph.entry_point),
            int(graph.max_level),
        )

        self._lib = ctypes.CDLL(str(build_library()))
        self._fn = self._lib.fast_hybrid_search_batch
        self._fn.argtypes = [
            ctypes.POINTER(_GraphView),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int64),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_int64),
        ]
        self._fn.restype = ctypes.c_int

    def search_batch(
        self,
        queries: np.ndarray,
        ef_search: int,
        *,
        k: int = 10,
        ef_bq: int = 64,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(ids, distances, latency_ns)`` for a query batch."""

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        n_queries, dim = queries.shape
        if n_queries == 0:
            return (
                np.empty((0, k), dtype=np.int64),
                np.empty((0, k), dtype=np.float32),
                np.empty(0, dtype=np.int64),
            )
        query_bq = np.ascontiguousarray(self.bq.encode(queries), dtype=np.uint64)
        query_code = np.ascontiguousarray(
            self.sq8.prepare_query(queries), dtype=np.float32
        )
        ids = np.full((n_queries, k), -1, dtype=np.int64)
        dists = np.zeros((n_queries, k), dtype=np.float32)
        latency_ns = np.zeros(n_queries, dtype=np.int64)

        base_ptr = (
            self.base.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            if self.rerank == "fp32"
            else ctypes.POINTER(ctypes.c_float)()
        )
        rc = self._fn(
            ctypes.byref(self._view),
            self.bq_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            int(self.bq.n_words),
            self.sq8_codes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            self._scales2.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            base_ptr,
            queries.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            query_bq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            query_code.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(n_queries),
            int(dim),
            int(ef_search),
            int(ef_bq),
            int(k),
            0 if self.rerank == "sq8" else 1,
            ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
            dists.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            latency_ns.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        )
        if rc != 0:
            raise RuntimeError(f"fast_hybrid_search_batch failed with code {rc}")
        return ids, dists, latency_ns

    def search(
        self,
        query: np.ndarray,
        ef_search: int,
        *,
        k: int = 10,
        ef_bq: int = 64,
    ) -> np.ndarray:
        ids, _, _ = self.search_batch(
            query.reshape(1, -1), ef_search, k=k, ef_bq=ef_bq
        )
        return ids[0]


__all__ = [
    "FastHybridSearcher",
    "build_library",
    "library_available",
]
