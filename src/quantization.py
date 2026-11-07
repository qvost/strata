"""Compressed codes and distance kernels used by the benchmark.

Two codecs are provided:

* :class:`SQ8Codec` -- per-dimension 8-bit scalar quantization with asymmetric
  distance computation (ADC): the query stays FP32 while base vectors are
  stored as ``uint8``.  The hot loop is a vectorised NumPy expression that
  computes ``sum(scale**2 * (q_code - code)**2)`` without materialising a
  dequantised copy of the base matrix.
* :class:`BQCodec` -- 1-bit binary quantization.  Each vector is packed into
  ``ceil(d / 64)`` ``uint64`` words; Hamming distance is an XOR followed by a
  hardware/vectorised popcount (``numpy.bitwise_count``).

The default BQ threshold is the per-dimension median.  This matters for SIFT:
raw SIFT coordinates are non-negative, so a zero threshold would set every bit
to one and destroy the distance signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _popcount(values: np.ndarray) -> np.ndarray:
    """Element-wise popcount for unsigned integer arrays.

    ``numpy.bitwise_count`` (NumPy >= 2.0) compiles down to a vectorised
    POPCNT loop.  The byte lookup fallback keeps the module usable on older
    NumPy releases.
    """

    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(values)
    # Portable fallback: view as uint8 and use a precomputed byte table.
    table = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    as_bytes = np.ascontiguousarray(values).view(np.uint8)
    return table[as_bytes].reshape(values.shape)


@dataclass
class SQ8Codec:
    """Per-dimension uint8 scalar quantizer with ADC distance."""

    mins: np.ndarray
    scales: np.ndarray
    name: str = "sq8"

    def __post_init__(self) -> None:
        self.mins = np.ascontiguousarray(self.mins, dtype=np.float32)
        self.scales = np.ascontiguousarray(self.scales, dtype=np.float32)
        self.scales2 = np.ascontiguousarray(self.scales * self.scales, dtype=np.float32)

    @classmethod
    def fit(cls, base: np.ndarray) -> "SQ8Codec":
        mins = base.min(axis=0).astype(np.float32)
        maxs = base.max(axis=0).astype(np.float32)
        scales = (maxs - mins) / 255.0
        # Constant dimensions get a unit scale so they quantize to code 0.
        scales = np.where(scales > 0, scales, np.float32(1.0)).astype(np.float32)
        return cls(mins=mins, scales=scales)

    @property
    def bits(self) -> int:
        return 8

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        codes = np.rint((vectors - self.mins) / self.scales)
        return np.clip(codes, 0, 255).astype(np.uint8)

    def decode(self, codes: np.ndarray) -> np.ndarray:
        return codes.astype(np.float32) * self.scales + self.mins

    def prepare_query(self, query: np.ndarray) -> np.ndarray:
        """Transform a query into code space for ADC."""

        return (query.astype(np.float32) - self.mins) / self.scales

    def adc_l2(self, query_code: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Asymmetric L2 distances from one prepared query to many codes.

        Parameters
        ----------
        query_code:
            ``(d,)`` output of :meth:`prepare_query`.
        codes:
            ``(m, d)`` or ``(d,)`` uint8 codes.

        Returns
        -------
        ``(m,)`` float32 squared L2 distances.
        """

        one = codes.ndim == 1
        matrix = codes.reshape(1, -1) if one else codes
        diff = matrix.astype(np.float32) - query_code[None, :]
        dists = (diff * diff) @ self.scales2
        return dists.astype(np.float32, copy=False) if not one else dists.reshape(())

    def adc_l2_one(self, query_code: np.ndarray, code: np.ndarray) -> float:
        diff = code.astype(np.float32) - query_code
        return float(np.dot(diff * diff, self.scales2))

    def memory_bytes(self, n_vectors: int, dim: int) -> int:
        return int(n_vectors) * int(dim) * 1


@dataclass
class BQCodec:
    """Per-dimension 1-bit quantizer with packed Hamming distance."""

    thresholds: np.ndarray
    dim: int
    name: str = "bq"
    n_words: int = field(init=False)
    _bits: int = field(init=False, default=1)

    def __post_init__(self) -> None:
        self.dim = int(self.dim)
        self.n_words = (self.dim + 63) // 64
        self.thresholds = np.ascontiguousarray(self.thresholds, dtype=np.float32)

    @classmethod
    def fit(cls, base: np.ndarray, mode: str = "median") -> "BQCodec":
        if mode == "zero":
            thresholds = np.zeros(base.shape[1], dtype=np.float32)
        elif mode == "mean":
            thresholds = base.mean(axis=0).astype(np.float32)
        elif mode == "median":
            thresholds = np.median(base, axis=0).astype(np.float32)
        else:
            raise ValueError(f"Unknown BQ threshold mode: {mode!r}")
        return cls(thresholds=thresholds, dim=base.shape[1])

    @property
    def bits(self) -> int:
        return 1

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        one = vectors.ndim == 1
        matrix = vectors.reshape(1, -1) if one else vectors
        bits = matrix > self.thresholds[None, :]
        pad = self.n_words * 64 - self.dim
        if pad:
            bits = np.pad(bits, ((0, 0), (0, pad)), mode="constant", constant_values=False)
        packed = np.packbits(bits, axis=1, bitorder="little")
        packed = np.ascontiguousarray(packed)
        words = packed.view(np.uint64).reshape(matrix.shape[0], self.n_words)
        return words[0] if one else words

    def hamming(self, query_words: np.ndarray, codes: np.ndarray) -> np.ndarray:
        """Hamming distances from one packed query to many packed codes."""

        one = codes.ndim == 1
        matrix = codes.reshape(1, -1) if one else codes
        xor = np.bitwise_xor(matrix, query_words[None, :])
        dists = _popcount(xor).sum(axis=1, dtype=np.int32)
        return dists if not one else dists.reshape(())

    def hamming_one(self, query_words: np.ndarray, code: np.ndarray) -> int:
        return int(_popcount(np.bitwise_xor(code, query_words)).sum(dtype=np.int64))

    def decode_sign(self, codes: np.ndarray) -> np.ndarray:
        """Unpack to an FP32 matrix of +/-1 values (for a binary HNSW index)."""

        one = codes.ndim == 1
        matrix = codes.reshape(1, -1) if one else codes
        raw = np.ascontiguousarray(matrix).view(np.uint8).reshape(matrix.shape[0], -1)
        bits = np.unpackbits(raw, axis=1, bitorder="little")[:, : self.dim]
        signs = np.where(bits, np.float32(1.0), np.float32(-1.0)).astype(np.float32)
        return signs[0] if one else signs

    def memory_bytes(self, n_vectors: int, dim: int) -> int:
        n_words = (int(dim) + 63) // 64
        return int(n_vectors) * n_words * 8


def hamming_distance_matrix(
    query_codes: np.ndarray, base_codes: np.ndarray, *, block: int = 4096
) -> np.ndarray:
    """Full Hamming distance matrix, blocked to bound temporary memory.

    Mostly useful for tests and for exact binary recall diagnostics.
    """

    nq = query_codes.shape[0]
    out = np.empty((nq, base_codes.shape[0]), dtype=np.int32)
    for start in range(0, nq, block):
        stop = min(start + block, nq)
        xor = np.bitwise_xor(
            query_codes[start:stop, None, :], base_codes[None, :, :]
        )
        out[start:stop] = _popcount(xor).sum(axis=2, dtype=np.int32)
    return out


__all__ = [
    "SQ8Codec",
    "BQCodec",
    "hamming_distance_matrix",
]
