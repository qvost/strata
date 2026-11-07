/**
 * quantization.cpp
 *
 * Implements zero-allocation BQ and SQ8 quantization / distance kernels for
 * 128-dimensional float32 vectors.
 *
 * Compile with: -O3 -march=native -mavx2
 */

#include "quantization.hpp"

#include <algorithm>  // std::max, std::clamp
#include <cmath>      // std::abs, std::round, std::fma
#include <bit>        // std::popcount (C++20)
#include <nmmintrin.h> // _mm_popcnt_u64 (SSE4.2 / POPCNT)
#include <immintrin.h> // AVX2

// ─── BQ helpers ──────────────────────────────────────────────────────────────

void bq_quantize(std::span<const float, DIMS> src, BQVector& out) noexcept
{
    out.bq[0] = 0;
    out.bq[1] = 0;

    for (int i = 0; i < 64; ++i) {
        if (src[i] >= 0.0f) out.bq[0] |= (uint64_t{1} << i);
    }
    for (int i = 64; i < 128; ++i) {
        if (src[i] >= 0.0f) out.bq[1] |= (uint64_t{1} << (i - 64));
    }
}

int bq_hamming(const BQVector& a, const BQVector& b) noexcept
{
    return static_cast<int>(
        _mm_popcnt_u64(a.bq[0] ^ b.bq[0]) +
        _mm_popcnt_u64(a.bq[1] ^ b.bq[1]));
}

// ─── BQ AVX2 batch Hamming ───────────────────────────────────────────────────
//
// Strategy: unroll 4 database BQVectors per iteration.
// Each BQVector is 2×64-bit words (16 bytes).  We load them as 128-bit lanes
// and XOR with the broadcast query words, then POPCNT per 64-bit half.
//
// AVX2 does not have a 64-bit POPCNT lane instruction, so we use the
// scalar _mm_popcnt_u64 on extracted 64-bit halves.  For 4 vectors
// per iteration we still benefit from the reduced loop overhead and
// any CPU-level instruction-level parallelism.

void bq_batch_hamming(
    const BQVector&  query,
    const BQVector*  db,
    std::size_t      n,
    int*             out_dist) noexcept
{
    const uint64_t q0 = query.bq[0];
    const uint64_t q1 = query.bq[1];

    std::size_t i = 0;

    // ── 4-way unrolled scalar POPCNT loop ────────────────────────────────────
    // Note: AVX2 lacks vectorised 64-bit popcount lanes, so we keep the
    // POPCNT calls scalar but unroll to let the CPU pipeline them.
    for (; i + 4 <= n; i += 4) {
        const BQVector& d0 = db[i + 0];
        const BQVector& d1 = db[i + 1];
        const BQVector& d2 = db[i + 2];
        const BQVector& d3 = db[i + 3];

        uint64_t x00 = d0.bq[0] ^ q0,  x01 = d0.bq[1] ^ q1;
        uint64_t x10 = d1.bq[0] ^ q0,  x11 = d1.bq[1] ^ q1;
        uint64_t x20 = d2.bq[0] ^ q0,  x21 = d2.bq[1] ^ q1;
        uint64_t x30 = d3.bq[0] ^ q0,  x31 = d3.bq[1] ^ q1;

        out_dist[i + 0] = static_cast<int>(_mm_popcnt_u64(x00) + _mm_popcnt_u64(x01));
        out_dist[i + 1] = static_cast<int>(_mm_popcnt_u64(x10) + _mm_popcnt_u64(x11));
        out_dist[i + 2] = static_cast<int>(_mm_popcnt_u64(x20) + _mm_popcnt_u64(x21));
        out_dist[i + 3] = static_cast<int>(_mm_popcnt_u64(x30) + _mm_popcnt_u64(x31));
    }

    // ── scalar tail ──────────────────────────────────────────────────────────
    for (; i < n; ++i) {
        out_dist[i] = bq_hamming(query, db[i]);
    }
}

// ─── SQ8 helpers ─────────────────────────────────────────────────────────────

void sq8_quantize(std::span<const float, DIMS> src, SQ8Vector& out) noexcept
{
    // Find symmetric abs-max scale.
    float absmax = 0.0f;
    for (int i = 0; i < DIMS; ++i) {
        float a = std::abs(src[i]);
        if (a > absmax) absmax = a;
    }

    out.scale = absmax;

    if (absmax == 0.0f) {
        // Zero vector — fill mid-point (128).
        for (int i = 0; i < DIMS; ++i) out.q[i] = 128;
        return;
    }

    // Map: v -> round(v / absmax * 127.5) + 128  ∈ [0, 255]
    const float inv_scale = 127.5f / absmax;
    for (int i = 0; i < DIMS; ++i) {
        float mapped = std::fma(src[i], inv_scale, 128.0f);
        // Clamp to [0, 255] then round.
        int qi = static_cast<int>(std::round(mapped));
        out.q[i] = static_cast<uint8_t>(std::clamp(qi, 0, 255));
    }
}


// ─── SQ8 AVX2 ADC single (internal helper used by batch) ─────────────────────
//
// Processes 32 dimensions per AVX2 pass (8 floats × 4 passes = 128 dims).
// uint8 → int32 → float conversion, dequantize, subtract, square, accumulate.

static float sq8_adc_l2_avx2(const float*       query,   // 128 floats, 32-byte aligned
                               const uint8_t*     qbytes,  // 128 uint8, any alignment
                               float              dq_scale) noexcept
{
    // Broadcast the integer bias (128) and the dq scale factor.
    const __m256  vscale  = _mm256_set1_ps(dq_scale);
    const __m256  vbias   = _mm256_set1_ps(128.0f);
    __m256        vacc0   = _mm256_setzero_ps();
    __m256        vacc1   = _mm256_setzero_ps();
    __m256        vacc2   = _mm256_setzero_ps();
    __m256        vacc3   = _mm256_setzero_ps();

    // 4 passes × 32 bytes each → 128 dimensions total.
    // Each pass: load 32 uint8 bytes → 4×8 int32 → 4×8 float.
    for (int pass = 0; pass < 4; ++pass) {
        const uint8_t* b   = qbytes + pass * 32;
        const float*   q   = query  + pass * 32;

        // Load 32 uint8 bytes as a 256-bit register.
        __m256i raw = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b));

        // Unpack to 32-bit integers in four 8-lane float vectors.
        // _mm256_cvtepu8_epi32 takes the lower 8 bytes of a 128-bit lane.
        __m128i lo128 = _mm256_castsi256_si128(raw);    // bytes 0..15
        __m128i hi128 = _mm256_extracti128_si256(raw, 1); // bytes 16..31

        // Convert each 8-byte chunk to 8 int32s, then to 8 floats.
        __m256i i32_0 = _mm256_cvtepu8_epi32(lo128);            // bytes 0..7
        __m256i i32_1 = _mm256_cvtepu8_epi32(_mm_srli_si128(lo128, 8)); // bytes 8..15
        __m256i i32_2 = _mm256_cvtepu8_epi32(hi128);            // bytes 16..23
        __m256i i32_3 = _mm256_cvtepu8_epi32(_mm_srli_si128(hi128, 8)); // bytes 24..31

        __m256 f0 = _mm256_cvtepi32_ps(i32_0);
        __m256 f1 = _mm256_cvtepi32_ps(i32_1);
        __m256 f2 = _mm256_cvtepi32_ps(i32_2);
        __m256 f3 = _mm256_cvtepi32_ps(i32_3);

        // Dequantize: (byte - 128) * scale
        // = (f - bias) * scale
        f0 = _mm256_mul_ps(_mm256_sub_ps(f0, vbias), vscale);
        f1 = _mm256_mul_ps(_mm256_sub_ps(f1, vbias), vscale);
        f2 = _mm256_mul_ps(_mm256_sub_ps(f2, vbias), vscale);
        f3 = _mm256_mul_ps(_mm256_sub_ps(f3, vbias), vscale);

        // Load query floats (32-byte aligned for first pass; use loadu for safety).
        __m256 q0 = _mm256_loadu_ps(q + 0);
        __m256 q1 = _mm256_loadu_ps(q + 8);
        __m256 q2 = _mm256_loadu_ps(q + 16);
        __m256 q3 = _mm256_loadu_ps(q + 24);

        // diff = query - dq
        __m256 d0 = _mm256_sub_ps(q0, f0);
        __m256 d1 = _mm256_sub_ps(q1, f1);
        __m256 d2 = _mm256_sub_ps(q2, f2);
        __m256 d3 = _mm256_sub_ps(q3, f3);

        // Accumulate squared diff using FMA: acc += d * d + 0
        vacc0 = _mm256_fmadd_ps(d0, d0, vacc0);
        vacc1 = _mm256_fmadd_ps(d1, d1, vacc1);
        vacc2 = _mm256_fmadd_ps(d2, d2, vacc2);
        vacc3 = _mm256_fmadd_ps(d3, d3, vacc3);
    }

    // Horizontal reduction across all 4 accumulators.
    __m256 sum01 = _mm256_add_ps(vacc0, vacc1);
    __m256 sum23 = _mm256_add_ps(vacc2, vacc3);
    __m256 sum   = _mm256_add_ps(sum01, sum23);

    // Fold upper 128-bit lane into lower.
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 s  = _mm_add_ps(lo, hi);

    // Horizontal sum within 128-bit register.
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// ─── Public single ADC (delegates to AVX2 helper) ────────────────────────────

float sq8_adc_l2(std::span<const float, DIMS> query,
                  const SQ8Vector&             base) noexcept
{
    const float dq_scale = base.scale / 127.5f;
    return sq8_adc_l2_avx2(query.data(), base.q, dq_scale);
}

// ─── SQ8 AVX2 batch ADC ──────────────────────────────────────────────────────

void sq8_batch_adc_l2(
    std::span<const float, DIMS> query,
    const SQ8Vector*             db,
    std::size_t                  n,
    float*                       out_dist) noexcept
{
    for (std::size_t i = 0; i < n; ++i) {
        out_dist[i] = sq8_adc_l2(query, db[i]);
    }
}
