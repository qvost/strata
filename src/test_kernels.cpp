/**
 * test_kernels.cpp
 *
 * Verification harness for BQ and SQ8 quantization / distance kernels.
 *
 * Tests:
 *   1. BQ quantize + single Hamming: compare with naive bit-counting reference.
 *   2. BQ batch Hamming (AVX2 path): all N results match the scalar reference.
 *   3. SQ8 quantize + single ADC: result within tolerance of brute-force float L2.
 *   4. SQ8 batch ADC: all N results match the scalar single-vector path.
 *
 * Compile:
 *   g++ -O3 -march=native -mavx2 src/test_kernels.cpp src/quantization.cpp \
 *       -o test_kernels -std=c++20
 *
 * Run:
 *   ./test_kernels
 *
 * Exit code 0 = all tests passed.
 */

#include "quantization.hpp"

#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <span>

// ─── utilities ───────────────────────────────────────────────────────────────

static constexpr int N_VECTORS = 1000;

// Aligned allocator helper (C++17 / C++20 new with alignment).
template<typename T>
static T* alloc_aligned(std::size_t n, std::size_t alignment = 64)
{
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, n * sizeof(T)) != 0) {
        std::perror("posix_memalign");
        std::exit(1);
    }
    return reinterpret_cast<T*>(ptr);
}

// Generate a random float in [-range, +range].
static float randf(std::mt19937& rng, float range = 1.0f)
{
    std::uniform_real_distribution<float> dist(-range, range);
    return dist(rng);
}

// Fill a 128-D float vector with uniform random values.
static void fill_random(float* v, std::mt19937& rng, float range = 1.0f)
{
    for (int i = 0; i < DIMS; ++i) v[i] = randf(rng, range);
}

// ─── Reference implementations (simple scalar, no intrinsics) ────────────────

static int ref_hamming(const BQVector& a, const BQVector& b)
{
    uint64_t x0 = a.bq[0] ^ b.bq[0];
    uint64_t x1 = a.bq[1] ^ b.bq[1];
    return static_cast<int>(__builtin_popcountll(x0) + __builtin_popcountll(x1));
}

static float ref_l2_sq(const float* a, const float* b, int d = DIMS)
{
    float acc = 0.0f;
    for (int i = 0; i < d; ++i) {
        float diff = a[i] - b[i];
        acc += diff * diff;
    }
    return acc;
}

// ─── Test helpers ─────────────────────────────────────────────────────────────

static int g_tests_run    = 0;
static int g_tests_failed = 0;

#define CHECK(cond, msg)                                      \
    do {                                                      \
        ++g_tests_run;                                        \
        if (!(cond)) {                                        \
            ++g_tests_failed;                                 \
            std::fprintf(stderr,                              \
                "FAIL [%s:%d] %s\n", __FILE__, __LINE__, msg);\
        }                                                     \
    } while (0)

// ─── Test 1: BQ single Hamming ───────────────────────────────────────────────

static void test_bq_single_hamming(std::mt19937& rng)
{
    std::puts("[BQ] Single Hamming distance vs. reference ...");

    alignas(64) float va[DIMS], vb[DIMS];
    BQVector qa{}, qb{};

    int failures = 0;
    for (int t = 0; t < N_VECTORS; ++t) {
        fill_random(va, rng);
        fill_random(vb, rng);

        bq_quantize(std::span<const float, DIMS>(va, DIMS), qa);
        bq_quantize(std::span<const float, DIMS>(vb, DIMS), qb);

        int got = bq_hamming(qa, qb);
        int exp = ref_hamming(qa, qb);

        if (got != exp) {
            if (++failures <= 5)
                std::fprintf(stderr, "  mismatch t=%d: got=%d exp=%d\n", t, got, exp);
        }
    }
    CHECK(failures == 0, "BQ single Hamming: all 1000 pairs match reference");
    std::printf("  %d/%d pairs OK\n", N_VECTORS - failures, N_VECTORS);
}

// ─── Test 2: BQ batch Hamming ────────────────────────────────────────────────

static void test_bq_batch_hamming(std::mt19937& rng)
{
    std::puts("[BQ] Batch Hamming (AVX2 path) vs. scalar reference ...");

    // Allocate 64-byte aligned database.
    BQVector* db  = alloc_aligned<BQVector>(N_VECTORS, 64);
    int*      got = alloc_aligned<int>(N_VECTORS, 64);
    int*      exp = alloc_aligned<int>(N_VECTORS, 64);

    alignas(64) float vtmp[DIMS];
    BQVector query{};

    // Build random database.
    for (int i = 0; i < N_VECTORS; ++i) {
        fill_random(vtmp, rng);
        bq_quantize(std::span<const float, DIMS>(vtmp, DIMS), db[i]);
    }
    fill_random(vtmp, rng);
    bq_quantize(std::span<const float, DIMS>(vtmp, DIMS), query);

    // Scalar reference.
    for (int i = 0; i < N_VECTORS; ++i)
        exp[i] = ref_hamming(query, db[i]);

    // Batch kernel under test.
    bq_batch_hamming(query, db, N_VECTORS, got);

    int failures = 0;
    for (int i = 0; i < N_VECTORS; ++i) {
        if (got[i] != exp[i]) {
            if (++failures <= 5)
                std::fprintf(stderr, "  mismatch i=%d: got=%d exp=%d\n", i, got[i], exp[i]);
        }
    }
    CHECK(failures == 0, "BQ batch Hamming: all 1000 distances match reference");
    std::printf("  %d/%d vectors OK\n", N_VECTORS - failures, N_VECTORS);

    std::free(db);
    std::free(got);
    std::free(exp);
}

// ─── Test 3: SQ8 quantize + single ADC ──────────────────────────────────────

static void test_sq8_single_adc(std::mt19937& rng)
{
    std::puts("[SQ8] Single ADC vs. brute-force float L2 ...");

    alignas(64) float vq[DIMS], vb[DIMS];
    SQ8Vector base{};

    // Acceptable quantization error:
    // max per-element quantization error ≈ scale/127.5 * 0.5 for each dim.
    // Squared total worst-case ≈ DIMS * (scale/127.5 * 0.5)^2.
    // We test with a loose relative tolerance.
    const float rel_tol = 0.05f; // 5% relative tolerance

    int failures = 0;
    for (int t = 0; t < N_VECTORS; ++t) {
        fill_random(vq, rng, 2.0f);
        fill_random(vb, rng, 2.0f);

        sq8_quantize(std::span<const float, DIMS>(vb, DIMS), base);

        // Dequantize base manually for a reference float L2.
        alignas(64) float dq[DIMS];
        const float dq_scale = base.scale / 127.5f;
        for (int i = 0; i < DIMS; ++i)
            dq[i] = (static_cast<float>(base.q[i]) - 128.0f) * dq_scale;

        float ref  = ref_l2_sq(vq, dq);
        float got  = sq8_adc_l2(std::span<const float, DIMS>(vq, DIMS), base);
        float diff = std::abs(got - ref);
        float tol  = rel_tol * (ref + 1e-6f);

        if (diff > tol) {
            if (++failures <= 5)
                std::fprintf(stderr,
                    "  mismatch t=%d: got=%.6f ref=%.6f diff=%.6f tol=%.6f\n",
                    t, got, ref, diff, tol);
        }
    }
    CHECK(failures == 0, "SQ8 single ADC: all 1000 results within tolerance");
    std::printf("  %d/%d pairs OK\n", N_VECTORS - failures, N_VECTORS);
}

// ─── Test 4: SQ8 batch ADC ───────────────────────────────────────────────────

static void test_sq8_batch_adc(std::mt19937& rng)
{
    std::puts("[SQ8] Batch ADC vs. single ADC reference ...");

    SQ8Vector* db      = alloc_aligned<SQ8Vector>(N_VECTORS, 64);
    float*     got     = alloc_aligned<float>(N_VECTORS, 64);
    float*     exp_arr = alloc_aligned<float>(N_VECTORS, 64);

    alignas(64) float vq[DIMS], vtmp[DIMS];
    fill_random(vq, rng, 2.0f);

    for (int i = 0; i < N_VECTORS; ++i) {
        fill_random(vtmp, rng, 2.0f);
        sq8_quantize(std::span<const float, DIMS>(vtmp, DIMS), db[i]);
    }

    // Scalar reference: single ADC on each.
    for (int i = 0; i < N_VECTORS; ++i)
        exp_arr[i] = sq8_adc_l2(std::span<const float, DIMS>(vq, DIMS), db[i]);

    // Batch under test.
    sq8_batch_adc_l2(std::span<const float, DIMS>(vq, DIMS), db, N_VECTORS, got);

    int failures = 0;
    for (int i = 0; i < N_VECTORS; ++i) {
        // Both call the same AVX2 kernel; results must be bit-exact.
        if (got[i] != exp_arr[i]) {
            if (++failures <= 5)
                std::fprintf(stderr,
                    "  mismatch i=%d: got=%.8f exp=%.8f\n", i, got[i], exp_arr[i]);
        }
    }
    CHECK(failures == 0, "SQ8 batch ADC: all 1000 distances bit-exact vs. single ADC");
    std::printf("  %d/%d vectors OK\n", N_VECTORS - failures, N_VECTORS);

    std::free(db);
    std::free(got);
    std::free(exp_arr);
}

// ─── Additional edge case: BQ quantize correctness ───────────────────────────

static void test_bq_quantize_correctness()
{
    std::puts("[BQ] Quantize bit-pattern correctness ...");

    alignas(64) float v[DIMS];
    // Alternate sign: even dims positive, odd dims negative.
    for (int i = 0; i < DIMS; ++i)
        v[i] = (i % 2 == 0) ? 1.0f : -1.0f;

    BQVector bq{};
    bq_quantize(std::span<const float, DIMS>(v, DIMS), bq);

    // Expected: bits at even positions set, odd positions clear.
    uint64_t expected_word = 0x5555555555555555ULL; // alternating bits 0,2,4,...
    CHECK(bq.bq[0] == expected_word, "BQ bq[0] alternating pattern");
    CHECK(bq.bq[1] == expected_word, "BQ bq[1] alternating pattern");

    // All-positive vector → all bits set.
    for (int i = 0; i < DIMS; ++i) v[i] = 1.0f;
    bq_quantize(std::span<const float, DIMS>(v, DIMS), bq);
    CHECK(bq.bq[0] == ~uint64_t{0}, "BQ all-positive: bq[0] = 0xFFFF...");
    CHECK(bq.bq[1] == ~uint64_t{0}, "BQ all-positive: bq[1] = 0xFFFF...");

    // All-negative vector → all bits clear.
    for (int i = 0; i < DIMS; ++i) v[i] = -1.0f;
    bq_quantize(std::span<const float, DIMS>(v, DIMS), bq);
    CHECK(bq.bq[0] == 0ULL, "BQ all-negative: bq[0] = 0");
    CHECK(bq.bq[1] == 0ULL, "BQ all-negative: bq[1] = 0");
}

// ─── Additional edge case: SQ8 zero-vector ───────────────────────────────────

static void test_sq8_zero_vector()
{
    std::puts("[SQ8] Zero-vector quantization ...");

    alignas(64) float v[DIMS] = {};  // zero-initialise
    SQ8Vector sq{};
    sq8_quantize(std::span<const float, DIMS>(v, DIMS), sq);

    CHECK(sq.scale == 0.0f, "SQ8 zero vector: scale == 0");
    bool all128 = true;
    for (int i = 0; i < DIMS; ++i)
        if (sq.q[i] != 128) { all128 = false; break; }
    CHECK(all128, "SQ8 zero vector: all bytes == 128");
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main()
{
    std::mt19937 rng(42); // deterministic seed for reproducibility

    std::puts("=== Quantization Kernel Tests ===");
    std::puts("");

    test_bq_quantize_correctness();
    test_sq8_zero_vector();
    test_bq_single_hamming(rng);
    test_bq_batch_hamming(rng);
    test_sq8_single_adc(rng);
    test_sq8_batch_adc(rng);

    std::puts("");
    std::printf("=== Results: %d/%d tests passed ===\n",
        g_tests_run - g_tests_failed, g_tests_run);

    if (g_tests_failed > 0) {
        std::fprintf(stderr, "FAILED: %d test(s)\n", g_tests_failed);
        return 1;
    }
    std::puts("ALL TESTS PASSED");
    return 0;
}
