"""FAISS HNSW helpers with direct access to the graph topology.

FAISS stores, for every node ``i``:

* ``levels[i]`` -- the number of layers the node participates in (so the
  highest layer index is ``levels[i] - 1``);
* ``offsets[i]`` -- the start of the node's neighbour block;
* the first ``M0 = 2 * M`` slots are layer-0 neighbours;
* every subsequent layer consumes ``M`` slots.

Keeping the graph arrays in NumPy lets the hybrid searcher walk the same
topology as the FP32 baseline while scoring edges with packed Hamming
distance.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - dependency error
    raise RuntimeError("faiss-cpu is required for the HNSW benchmark") from exc


class FAISSGraph:
    """Read-only NumPy view of a FAISS HNSW graph."""

    def __init__(self, index: "faiss.IndexHNSWFlat") -> None:
        hnsw = index.hnsw
        self.levels = faiss.vector_to_array(hnsw.levels).astype(np.int32, copy=False)
        self.offsets = faiss.vector_to_array(hnsw.offsets).astype(np.int64, copy=False)
        self.neighbors = faiss.vector_to_array(hnsw.neighbors).astype(np.int32, copy=False)
        self.M0 = int(hnsw.nb_neighbors(0))
        self.M = int(hnsw.nb_neighbors(1))
        self.entry_point = int(hnsw.entry_point)
        self.ntotal = int(index.ntotal)
        self.max_level = int(self.levels[self.entry_point]) - 1
        self._empty = np.empty(0, dtype=np.int32)

    def layers(self, node: int) -> int:
        return int(self.levels[node])

    def neighbors_at(self, node: int, layer: int) -> np.ndarray:
        n_layers = int(self.levels[node])
        if layer < 0 or layer >= n_layers:
            return self._empty
        width = self.M0 if layer == 0 else self.M
        start = int(self.offsets[node])
        if layer > 0:
            start += self.M0 + (layer - 1) * self.M
        block = self.neighbors[start : start + width]
        return block[block >= 0]

    def graph_bytes(self) -> int:
        if self.offsets.size == 0:
            return 0
        return int(self.neighbors.nbytes + self.levels.nbytes + self.offsets.nbytes)


class HNSWIndex:
    """Thin wrapper around ``faiss.IndexHNSWFlat`` with single-query search."""

    def __init__(
        self,
        vectors: np.ndarray,
        *,
        M: int = 32,
        ef_construction: int = 200,
        verbose: bool = False,
    ) -> None:
        self.dim = int(vectors.shape[1])
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.index = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
        self.index.hnsw.efConstruction = self.ef_construction
        if verbose:
            print(
                f"[hnsw] building M={M} efConstruction={ef_construction} "
                f"on {vectors.shape[0]:,} vectors",
                flush=True,
            )
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))
        self._graph: Optional[FAISSGraph] = None

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def set_ef_search(self, ef_search: int) -> None:
        self.index.hnsw.efSearch = int(ef_search)

    def search_one(
        self, query: np.ndarray, k: int, ef_search: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.set_ef_search(ef_search)
        distances, ids = self.index.search(
            np.ascontiguousarray(query, dtype=np.float32).reshape(1, -1), int(k)
        )
        return ids[0].astype(np.int64, copy=False), distances[0].astype(
            np.float32, copy=False
        )

    def search_batch(
        self, queries: np.ndarray, k: int, ef_search: int
    ) -> tuple[np.ndarray, np.ndarray]:
        self.set_ef_search(ef_search)
        distances, ids = self.index.search(
            np.ascontiguousarray(queries, dtype=np.float32), int(k)
        )
        return ids.astype(np.int64, copy=False), distances.astype(
            np.float32, copy=False
        )

    def graph(self) -> FAISSGraph:
        if self._graph is None:
            self._graph = FAISSGraph(self.index)
        return self._graph

    def storage_bytes(self) -> int:
        """Approximate bytes held by the FAISS index (vectors + graph)."""

        vectors = int(self.ntotal) * int(self.dim) * 4
        graph = self.graph().graph_bytes()
        return vectors + graph


__all__ = ["FAISSGraph", "HNSWIndex"]
