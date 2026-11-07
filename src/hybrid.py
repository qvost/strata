"""Two-stage HNSW traversal: BQ coarse frontier, SQ8/FP32 fine reranking.

The search follows the HNSW topology produced by :class:`src.hnsw.HNSWIndex`:

1. Greedy descent through layers ``L > 0`` using packed 1-bit Hamming
   distance.
2. A layer-0 beam search with a *fixed* binary candidate buffer
   ``ef_bq`` (default 64).  This stage never touches the full-precision
   payload.
3. A fine beam search seeded with the binary candidate pool.  Edges are
   scored with SQ8 ADC (or exact FP32 L2) and the final top-k is returned.

The binary stage is cache-friendly: each node is a ``ceil(d/64)`` word
signature, so a 128-d SIFT descriptor costs 16 bytes instead of 512 bytes.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .hnsw import FAISSGraph
from .quantization import BQCodec, SQ8Codec


@dataclass
class HybridStats:
    upper_hops: int = 0
    bq_visited: int = 0
    fine_visited: int = 0
    pool_size: int = 0


class HybridSearcher:
    """BQ-guided HNSW traversal with SQ8/FP32 reranking."""

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
        self.base = base
        self.bq_codes = bq_codes
        self.sq8_codes = sq8_codes
        self.rerank = rerank
        self.last_stats = HybridStats()

    # ------------------------------------------------------------------
    # Distance helpers
    # ------------------------------------------------------------------
    def _hamming_many(self, query_words: np.ndarray, ids: np.ndarray) -> np.ndarray:
        return self.bq.hamming(query_words, self.bq_codes[ids])

    def _fine_many(
        self, query: np.ndarray, query_code: np.ndarray, ids: np.ndarray
    ) -> np.ndarray:
        if ids.size == 0:
            return np.empty(0, dtype=np.float32)
        if self.rerank == "sq8":
            codes = self.sq8_codes[ids].astype(np.float32, copy=False)
            diff = codes - query_code[None, :]
            return (diff * diff) @ self.sq8.scales2
        diff = self.base[ids] - query[None, :]
        return np.einsum("ij,ij->i", diff, diff, dtype=np.float32)

    # ------------------------------------------------------------------
    # Stage 1: binary navigation
    # ------------------------------------------------------------------
    def greedy_upper(self, query_words: np.ndarray) -> int:
        """Greedy descent from the entry point through layers > 0."""

        graph = self.graph
        current = graph.entry_point
        current_dist = self.bq.hamming_one(query_words, self.bq_codes[current])
        hops = 0
        for layer in range(graph.max_level, 0, -1):
            changed = True
            while changed:
                changed = False
                neighbors = graph.neighbors_at(current, layer)
                if neighbors.size == 0:
                    break
                dists = self._hamming_many(query_words, neighbors)
                best = int(np.argmin(dists))
                if int(dists[best]) < current_dist:
                    current = int(neighbors[best])
                    current_dist = int(dists[best])
                    hops += 1
                    changed = True
        self.last_stats.upper_hops = hops
        return current

    def beam_layer0_bq(
        self, query_words: np.ndarray, entry: int, ef_bq: int
    ) -> list[tuple[int, int]]:
        """Layer-0 beam search scored exclusively with Hamming distance."""

        ef_bq = max(1, int(ef_bq))
        graph = self.graph
        entry_dist = self.bq.hamming_one(query_words, self.bq_codes[entry])
        candidates: list[tuple[int, int]] = [(entry_dist, entry)]
        results: list[tuple[int, int]] = [(-entry_dist, entry)]
        visited = {entry}
        while candidates:
            distance, node = heapq.heappop(candidates)
            worst = -results[0][0]
            if len(results) >= ef_bq and distance > worst:
                break
            neighbors = graph.neighbors_at(node, 0)
            if neighbors.size == 0:
                continue
            fresh = [
                int(n) for n in neighbors.tolist() if int(n) not in visited
            ]
            if not fresh:
                continue
            visited.update(fresh)
            dists = self._hamming_many(query_words, np.asarray(fresh, dtype=np.int64))
            for neighbor, dist in zip(fresh, dists.tolist()):
                if len(results) < ef_bq or dist < -results[0][0]:
                    heapq.heappush(candidates, (int(dist), neighbor))
                    heapq.heappush(results, (-int(dist), neighbor))
                    if len(results) > ef_bq:
                        heapq.heappop(results)
        self.last_stats.bq_visited = len(visited)
        self.last_stats.pool_size = len(results)
        return sorted((-neg_dist, node) for neg_dist, node in results)

    # ------------------------------------------------------------------
    # Stage 2: fine reranking / expansion
    # ------------------------------------------------------------------
    def fine_search(
        self,
        query: np.ndarray,
        query_code: np.ndarray,
        pool: list[tuple[int, int]],
        ef_search: int,
        k: int,
    ) -> np.ndarray:
        """Warm-started SQ8/FP32 beam search over layer 0."""

        ef = max(int(ef_search), int(k), 1)
        pool_ids = [node for _, node in pool]
        pool_np = np.asarray(pool_ids, dtype=np.int64)
        dists = self._fine_many(query, query_code, pool_np)

        candidates: list[tuple[float, int]] = [
            (float(dist), int(node)) for dist, node in zip(dists.tolist(), pool_ids)
        ]
        results: list[tuple[float, int]] = [
            (-float(dist), int(node)) for dist, node in zip(dists.tolist(), pool_ids)
        ]
        heapq.heapify(candidates)
        heapq.heapify(results)
        while len(results) > ef:
            heapq.heappop(results)

        visited = set(pool_ids)
        while candidates:
            distance, node = heapq.heappop(candidates)
            if len(results) >= ef and distance > -results[0][0]:
                break
            neighbors = self.graph.neighbors_at(node, 0)
            if neighbors.size == 0:
                continue
            fresh = [
                int(n) for n in neighbors.tolist() if int(n) not in visited
            ]
            if not fresh:
                continue
            visited.update(fresh)
            fresh_np = np.asarray(fresh, dtype=np.int64)
            fine_dists = self._fine_many(query, query_code, fresh_np)
            for neighbor, dist in zip(fresh, fine_dists.tolist()):
                if len(results) < ef or dist < -results[0][0]:
                    heapq.heappush(candidates, (float(dist), neighbor))
                    heapq.heappush(results, (-float(dist), neighbor))
                    if len(results) > ef:
                        heapq.heappop(results)

        self.last_stats.fine_visited = len(visited)
        best = sorted((-neg_dist, node) for neg_dist, node in results)[:k]
        return np.fromiter((node for _, node in best), dtype=np.int64, count=len(best))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_pool(
        self, query: np.ndarray, ef_bq: int = 64
    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Run the binary upper-layer + layer-0 coarse stages."""

        query_words = self.bq.encode(query)
        entry = self.greedy_upper(query_words)
        pool = self.beam_layer0_bq(query_words, entry, ef_bq)
        return query_words, pool

    def search(
        self,
        query: np.ndarray,
        ef_search: int,
        *,
        k: int = 10,
        ef_bq: int = 64,
    ) -> np.ndarray:
        query_words, pool = self.build_pool(query, ef_bq=ef_bq)
        query_code = self.sq8.prepare_query(query)
        return self.fine_search(query, query_code, pool, ef_search, k)

    def search_from_pool(
        self,
        query: np.ndarray,
        query_words: np.ndarray,
        pool: list[tuple[int, int]],
        ef_search: int,
        *,
        k: int = 10,
    ) -> np.ndarray:
        query_code = self.sq8.prepare_query(query)
        return self.fine_search(query, query_code, pool, ef_search, k)


__all__ = ["HybridSearcher", "HybridStats"]
