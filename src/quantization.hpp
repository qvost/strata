/**
 * quantization.hpp
 *
 * Zero-allocation quantization and distance kernels for 128-dimensional
 * float32 vectors. Targets C++20 compiled with -O3 -march=native -mavx2.
 *
 * Supported schemes:
 *   - Binary Quantization (BQ):  sign-bit thresholding → two uint64_t words.
 *   - Scalar Quantization 8-bit (SQ8): symmetric min-max → uint8_t[128].
 *
 * Memory layout:
 *   All packed vector buffers MUST be aligned to 64-byte cache lines
 *   (alignas(64)) to allow safe 256-bit (32-byte) AVX2 loads.
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <span>
#include <immintrin.h>  // AVX2 / POPCNT intrinsics

// ─── constants ───────────────────────────────────────────────────────────────

static constexpr int DIMS = 128;

// ─── Binary Quantization (BQ) ────────────────────────────────────────────────

/**
 * Packed binary representation of a 128-D vector.
 * Bit i in bq[i/64] corresponds to dimension i.
 * bit = 1 when v[i] >= 0, else 0.
 *
 * Stored as two 64-bit words (128 bits total).
 */
struct alignas(16) BQVector {
    uint64_t bq[2];
};

/**
 * Quantize a 128-D float32 vector into its BQ representation.
 *
 * @param src  Input float32 vector of exactly 128 elements.
 * @param out  Output BQ vector (bit i = (src[i] >= 0)).
 */
void bq_quantize(std::span<const float, DIMS> src, BQVector& out) noexcept;

/**
 * Hamming distance between two BQ vectors.
 * Uses hardware POPCNT on 64-bit words.
 *
 * @return Number of differing bits (0 … 128).
 */
[[nodiscard]]
int bq_hamming(const BQVector& a, const BQVector& b) noexcept;

/**
 * AVX2 batch Hamming distance: one query vs. N packed BQ vectors.
 *
 * Processes 4 database vectors per AVX2 iteration.
 * Falls back to scalar path for the tail.
 *
 * @param query    Query BQ vector.
 * @param db       Pointer to an array of N BQVectors, 64-byte aligned.
 * @param n        Number of database vectors.
 * @param out_dist Output array of N int distances (caller-allocated).
 */
void bq_batch_hamming(
    const BQVector&  query,
    const BQVector*  db,
    std::size_t      n,
    int*             out_dist) noexcept;

// ─── Scalar Quantization 8-bit (SQ8) ─────────────────────────────────────────

/**
 * Packed SQ8 representation of a 128-D vector.
 * Each element is mapped from float32 → uint8_t using symmetric min-max scaling:
 *
 *   scale  = max(|v_min|, |v_max|)
 *   q[i]   = round( v[i] / scale * 127.5 ) + 128   ∈ [0, 255]
 *
 * Inverse:  v_approx[i] = (q[i] - 128) / 127.5 * scale
 *
 * The scale is stored alongside the bytes so the caller can reconstruct or
 * compute asymmetric distances without extra metadata lookups.
 */
struct alignas(64) SQ8Vector {
    uint8_t  q[DIMS]; ///< Quantized coordinates.
    float    scale;   ///< Symmetric abs-max scale factor.
    uint8_t  _pad[64 - DIMS % 64 - sizeof(float)]; ///< Pad to 64-byte boundary.
};
static_assert(sizeof(SQ8Vector) % 64 == 0,
    "SQ8Vector must be a multiple of 64 bytes for aligned array access");

/**
 * Quantize a 128-D float32 vector into SQ8 form.
 *
 * @param src  Input float32 vector of exactly 128 elements.
 * @param out  Output SQ8 vector.
 */
void sq8_quantize(std::span<const float, DIMS> src, SQ8Vector& out) noexcept;

/**
 * Asymmetric Distance Computation (ADC):
 * Squared L2 distance between an uncompressed FP32 query and a SQ8 base vector.
 *
 * The base vector is dequantized on-the-fly inside the AVX2 kernel:
 *   dq[i] = (base.q[i] - 128) * (base.scale / 127.5f)
 *   dist  = sum_i ( query[i] - dq[i] )^2
 *
 * @param query  Uncompressed float32 query, 128 elements.
 * @param base   SQ8-compressed base vector.
 * @return       Approximate squared L2 distance.
 */
[[nodiscard]]
float sq8_adc_l2(std::span<const float, DIMS> query,
                 const SQ8Vector&             base) noexcept;

/**
 * AVX2 batch ADC squared-L2: one FP32 query vs. N SQ8 base vectors.
 *
 * @param query    Uncompressed float32 query, 128 elements, 32-byte aligned.
 * @param db       Array of N SQ8Vectors, 64-byte aligned.
 * @param n        Number of database vectors.
 * @param out_dist Output array of N floats (caller-allocated).
 */
void sq8_batch_adc_l2(
    std::span<const float, DIMS> query,
    const SQ8Vector*             db,
    std::size_t                  n,
    float*                       out_dist) noexcept;
