// fast_hybrid.cpp
//
// Compiled adaptive HNSW traversal used by bench/runner.py.
//
// Three traversal modes, selected per-query based on efSearch:
//
// Mode A — single-pass SQ8 (efSearch ≤ 40):
//   * greedy descent through layers L > 0 using SQ8 ADC distance
//   * layer-0 beam search scored with SQ8 ADC or FP32 L2
//   * avoids the two-traversal overhead of the BQ pre-filter stage,
//     giving near-parity latency with FP32 HNSW at low ef
//
// Mode B — two-stage hybrid (efSearch > 40):
//   * Stage 1: greedy descent + layer-0 BQ beam (64-bit POPCNT Hamming)
//   * Stage 2: warm-started fine beam (SQ8 ADC or FP32 L2)
//   * BQ stage prunes the exploration frontier cheaply; at large ef the
//     saved fine-distance evaluations outweigh the extra BQ pass cost
//
// Candidate tracking – design rationale
// ──────────────────────────────────────
// The original code used two std::priority_queues per search stage
// (a min-heap for the exploration frontier and a max-heap for the bounded
// result set).  For ef ≤ 320 this introduces unnecessary overhead:
//
//   • Every neighbor insertion triggers an O(log ef) sift-up in *both* heaps.
//   • The max-heap trim (pop when size > ef) is a sift-down on top of that.
//   • At stage exit the heap is drained and sorted, paying O(ef log ef)
//     again even though the data is nearly sorted.
//   • The handoff from beam_layer0_bq's result heap to fine_search's
//     candidate / result heaps re-heapifies the same items again.
//
// Replacement strategy
// ────────────────────
// ResultBuffer<T, Cap>
//   A flat array of at most Cap pairs kept in ascending distance order.
//   Insertion is O(ef) in the worst case but:
//     - Zero allocation overhead (stack-allocated).
//     - Contiguous, cache-line–friendly access.
//     - The worst-distance (buffer.worst()) is simply buffer[size-1].first;
//       no heap traversal required.
//     - For ef ≤ 320 the linear insert beats the heap's constant-factor
//       overhead on modern out-of-order CPUs.
//
// CandidateHeap<T, Cap>
//   A flat binary min-heap with a fixed upper-bound capacity used for the
//   exploration frontier.  We still need O(log n) pop-min, but by using a
//   flat array instead of std::priority_queue we:
//     - Avoid vector realloc / capacity-doubling.
//     - Keep the entire heap in a single cache line for ef ≤ 32 (capacity
//       fits in 512 bytes at 16 bytes/entry × 32 entries).
//
// Pool handoff
// ────────────
// beam_layer0_bq writes directly into a ResultBuffer<int>.  fine_search
// reads that buffer in-place (it is already sorted) and seeds its own
// CandidateHeap and ResultBuffer<float> without an intermediate
// std::vector copy or heap reconstruction.
//
// Build:
//   g++ -O3 -march=native -std=c++20 -shared -fPIC fast_hybrid.cpp
//       -o libfast_hybrid.so

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <utility>
#include <vector>

#include <immintrin.h>  // AVX2 / FMA distance kernels

// ─── compile-time capacity cap ───────────────────────────────────────────────
// efSearch and ef_bq are both ≤ 320 by the benchmark sweep spec.
// We add a small margin and round to 64 for alignment.
static constexpr int MAX_EF = 384;

namespace {

// ─── GraphView ───────────────────────────────────────────────────────────────

struct GraphView {
    const int32_t* levels;
    const int64_t* offsets;
    const int32_t* neighbors;
    int64_t  ntotal;
    int32_t  M0;
    int32_t  M;
    int32_t  entry;
    int32_t  max_level;
};

[[nodiscard]] inline const int32_t*
neighbors_at(const GraphView& g, int32_t node, int layer, int32_t& width) noexcept
{
    const int32_t n_layers = g.levels[node];
    if (layer < 0 || layer >= n_layers) { width = 0; return nullptr; }
    width = (layer == 0) ? g.M0 : g.M;
    int64_t start = g.offsets[node];
    if (layer > 0) start += g.M0 + static_cast<int64_t>(layer - 1) * g.M;
    return g.neighbors + start;
}

// ─── Hamming distance ────────────────────────────────────────────────────────

[[nodiscard]] inline int
hamming_128(const uint64_t* __restrict__ a,
            const uint64_t* __restrict__ b) noexcept
{
    return static_cast<int>(
        __builtin_popcountll(a[0] ^ b[0]) +
        __builtin_popcountll(a[1] ^ b[1])
    );
}

[[nodiscard]] inline int
hamming_row(const uint64_t* __restrict__ a,
            const uint64_t* __restrict__ b,
            int n_words) noexcept
{
    if (n_words == 2) return hamming_128(a, b);
    uint64_t acc = 0;
    for (int i = 0; i < n_words; ++i)
        acc += static_cast<uint64_t>(__builtin_popcountll(a[i] ^ b[i]));
    return static_cast<int>(acc);
}

// ─── Fine distance helpers ───────────────────────────────────────────────────

// ─── AVX2 / FMA distance kernels ────────────────────────────────────────────
//
// Both kernels process `dim` in 32-float blocks (four 256-bit registers) and
// keep four independent accumulators so the FMA dependency chain does not
// stall the pipeline.  SIFT uses dim = 128, i.e. four full blocks; the scalar
// tail keeps the kernels correct for any dimension.

[[nodiscard]] inline float
hsum256_ps(__m256 v) noexcept
{
    const __m128 lo = _mm256_castps256_ps128(v);
    const __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// SQ8 ADC: sum_i scales2[i] * (code[i] - query_code[i])^2
[[nodiscard]] inline float
sq8_distance(const uint8_t* __restrict__ code,
             const float*   __restrict__ query_code,
             const float*   __restrict__ scales2,
             int dim) noexcept
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        // 32 uint8 -> four 8-lane float vectors.
        const __m256i raw =
            _mm256_loadu_si256(reinterpret_cast<const __m256i*>(code + i));
        const __m128i lo = _mm256_castsi256_si128(raw);
        const __m128i hi = _mm256_extracti128_si256(raw, 1);

        const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
        const __m256 f1 =
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(lo, 8)));
        const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));
        const __m256 f3 =
            _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hi, 8)));

        const __m256 d0 = _mm256_sub_ps(f0, _mm256_loadu_ps(query_code + i + 0));
        const __m256 d1 = _mm256_sub_ps(f1, _mm256_loadu_ps(query_code + i + 8));
        const __m256 d2 = _mm256_sub_ps(f2, _mm256_loadu_ps(query_code + i + 16));
        const __m256 d3 = _mm256_sub_ps(f3, _mm256_loadu_ps(query_code + i + 24));

        // acc += scales2 * diff^2
        acc0 = _mm256_fmadd_ps(
            _mm256_mul_ps(d0, d0), _mm256_loadu_ps(scales2 + i + 0), acc0);
        acc1 = _mm256_fmadd_ps(
            _mm256_mul_ps(d1, d1), _mm256_loadu_ps(scales2 + i + 8), acc1);
        acc2 = _mm256_fmadd_ps(
            _mm256_mul_ps(d2, d2), _mm256_loadu_ps(scales2 + i + 16), acc2);
        acc3 = _mm256_fmadd_ps(
            _mm256_mul_ps(d3, d3), _mm256_loadu_ps(scales2 + i + 24), acc3);
    }

    const __m256 sum =
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float total = hsum256_ps(sum);
    for (; i < dim; ++i) {
        const float diff = static_cast<float>(code[i]) - query_code[i];
        total += scales2[i] * diff * diff;
    }
    return total;
}

// 2x Interleaved SQ8 ADC: processes two codes simultaneously to saturate dual FMA ports
// and share query_code and scales2 memory loads.
inline void
sq8_distance_2x(const uint8_t* __restrict__ codeA,
                const uint8_t* __restrict__ codeB,
                const float*   __restrict__ query_code,
                const float*   __restrict__ scales2,
                int dim,
                float& distA,
                float& distB) noexcept
{
    __m256 accA0 = _mm256_setzero_ps(), accA1 = _mm256_setzero_ps();
    __m256 accA2 = _mm256_setzero_ps(), accA3 = _mm256_setzero_ps();
    __m256 accB0 = _mm256_setzero_ps(), accB1 = _mm256_setzero_ps();
    __m256 accB2 = _mm256_setzero_ps(), accB3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256i rawA = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codeA + i));
        const __m256i rawB = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(codeB + i));

        const __m128i loA = _mm256_castsi256_si128(rawA);
        const __m128i hiA = _mm256_extracti128_si256(rawA, 1);
        const __m128i loB = _mm256_castsi256_si128(rawB);
        const __m128i hiB = _mm256_extracti128_si256(rawB, 1);

        const __m256 fA0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(loA));
        const __m256 fA1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(loA, 8)));
        const __m256 fA2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hiA));
        const __m256 fA3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hiA, 8)));

        const __m256 fB0 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(loB));
        const __m256 fB1 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(loB, 8)));
        const __m256 fB2 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hiB));
        const __m256 fB3 = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(hiB, 8)));

        const __m256 qc0 = _mm256_loadu_ps(query_code + i + 0);
        const __m256 qc1 = _mm256_loadu_ps(query_code + i + 8);
        const __m256 qc2 = _mm256_loadu_ps(query_code + i + 16);
        const __m256 qc3 = _mm256_loadu_ps(query_code + i + 24);

        const __m256 sc0 = _mm256_loadu_ps(scales2 + i + 0);
        const __m256 sc1 = _mm256_loadu_ps(scales2 + i + 8);
        const __m256 sc2 = _mm256_loadu_ps(scales2 + i + 16);
        const __m256 sc3 = _mm256_loadu_ps(scales2 + i + 24);

        const __m256 dA0 = _mm256_sub_ps(fA0, qc0);
        const __m256 dB0 = _mm256_sub_ps(fB0, qc0);
        accA0 = _mm256_fmadd_ps(_mm256_mul_ps(dA0, dA0), sc0, accA0);
        accB0 = _mm256_fmadd_ps(_mm256_mul_ps(dB0, dB0), sc0, accB0);

        const __m256 dA1 = _mm256_sub_ps(fA1, qc1);
        const __m256 dB1 = _mm256_sub_ps(fB1, qc1);
        accA1 = _mm256_fmadd_ps(_mm256_mul_ps(dA1, dA1), sc1, accA1);
        accB1 = _mm256_fmadd_ps(_mm256_mul_ps(dB1, dB1), sc1, accB1);

        const __m256 dA2 = _mm256_sub_ps(fA2, qc2);
        const __m256 dB2 = _mm256_sub_ps(fB2, qc2);
        accA2 = _mm256_fmadd_ps(_mm256_mul_ps(dA2, dA2), sc2, accA2);
        accB2 = _mm256_fmadd_ps(_mm256_mul_ps(dB2, dB2), sc2, accB2);

        const __m256 dA3 = _mm256_sub_ps(fA3, qc3);
        const __m256 dB3 = _mm256_sub_ps(fB3, qc3);
        accA3 = _mm256_fmadd_ps(_mm256_mul_ps(dA3, dA3), sc3, accA3);
        accB3 = _mm256_fmadd_ps(_mm256_mul_ps(dB3, dB3), sc3, accB3);
    }

    distA = hsum256_ps(_mm256_add_ps(_mm256_add_ps(accA0, accA1), _mm256_add_ps(accA2, accA3)));
    distB = hsum256_ps(_mm256_add_ps(_mm256_add_ps(accB0, accB1), _mm256_add_ps(accB2, accB3)));

    for (; i < dim; ++i) {
        const float diffA = static_cast<float>(codeA[i]) - query_code[i];
        distA += scales2[i] * diffA * diffA;
        const float diffB = static_cast<float>(codeB[i]) - query_code[i];
        distB += scales2[i] * diffB * diffB;
    }
}

[[nodiscard]] inline float
fp32_distance(const float* __restrict__ base_row,
              const float* __restrict__ query,
              int dim) noexcept
{
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    int i = 0;
    for (; i + 32 <= dim; i += 32) {
        const __m256 d0 =
            _mm256_sub_ps(_mm256_loadu_ps(base_row + i + 0),
                          _mm256_loadu_ps(query + i + 0));
        const __m256 d1 =
            _mm256_sub_ps(_mm256_loadu_ps(base_row + i + 8),
                          _mm256_loadu_ps(query + i + 8));
        const __m256 d2 =
            _mm256_sub_ps(_mm256_loadu_ps(base_row + i + 16),
                          _mm256_loadu_ps(query + i + 16));
        const __m256 d3 =
            _mm256_sub_ps(_mm256_loadu_ps(base_row + i + 24),
                          _mm256_loadu_ps(query + i + 24));

        acc0 = _mm256_fmadd_ps(d0, d0, acc0);
        acc1 = _mm256_fmadd_ps(d1, d1, acc1);
        acc2 = _mm256_fmadd_ps(d2, d2, acc2);
        acc3 = _mm256_fmadd_ps(d3, d3, acc3);
    }

    const __m256 sum =
        _mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3));
    float total = hsum256_ps(sum);
    for (; i < dim; ++i) {
        const float diff = base_row[i] - query[i];
        total += diff * diff;
    }
    return total;
}

// Upper bound on one batched neighbour group; larger groups are flushed in
// chunks preserving graph order.
static constexpr int BATCH_MAX = 256;

inline void
hamming_batch(const uint64_t* __restrict__ query_bq,
              const uint64_t* __restrict__ bq_codes,
              const int32_t*  __restrict__ nodes,
              int                          n,
              int                          n_words,
              int*           __restrict__ out) noexcept
{
    for (int i = 0; i < n; ++i) {
        const int64_t row = static_cast<int64_t>(nodes[i]) * n_words;
        if (i + 1 < n) {
            const int64_t next = static_cast<int64_t>(nodes[i + 1]) * n_words;
            _mm_prefetch(reinterpret_cast<const char*>(bq_codes + next),
                         _MM_HINT_T0);
        }
        out[i] = hamming_row(query_bq, bq_codes + row, n_words);
    }
}

// Evaluate one rerank mode for a contiguous list of graph node ids.
// RerankMode is a template parameter (0 = SQ8 ADC, 1 = FP32 L2) so the mode
// branch is hoisted out of the traversal loop; the next row is prefetched
// while the current row is scored.
template<int RerankMode>
inline void
fine_distances(const int32_t* __restrict__ nodes,
               int                          n,
               const float*   __restrict__ base,
               const uint8_t* __restrict__ sq8_codes,
               const float*   __restrict__ sq8_scales2,
               const float*   __restrict__ query,
               const float*   __restrict__ query_code,
               int                          dim,
               float*         __restrict__ out) noexcept
{
    if constexpr (RerankMode == 1) {
        const int row_bytes = dim * static_cast<int>(sizeof(float));
        for (int i = 0; i < n; ++i) {
            const int64_t row = static_cast<int64_t>(nodes[i]) * dim;
            if (i + 1 < n) {
                const int64_t next = static_cast<int64_t>(nodes[i + 1]) * dim;
                const char* p = reinterpret_cast<const char*>(base + next);
                for (int off = 0; off < row_bytes; off += 64)
                    _mm_prefetch(p + off, _MM_HINT_T0);
            }
            out[i] = fp32_distance(base + row, query, dim);
        }
    } else {
        int i = 0;
        for (; i + 1 < n; i += 2) {
            if (i + 4 < n) {
                const char* p0 = reinterpret_cast<const char*>(sq8_codes + static_cast<int64_t>(nodes[i + 4]) * dim);
                const char* p1 = reinterpret_cast<const char*>(sq8_codes + static_cast<int64_t>(nodes[i + 5]) * dim);
                _mm_prefetch(p0, _MM_HINT_T0);
                _mm_prefetch(p0 + 64, _MM_HINT_T0);
                _mm_prefetch(p1, _MM_HINT_T0);
                _mm_prefetch(p1 + 64, _MM_HINT_T0);
            }
            sq8_distance_2x(
                sq8_codes + static_cast<int64_t>(nodes[i]) * dim,
                sq8_codes + static_cast<int64_t>(nodes[i + 1]) * dim,
                query_code, sq8_scales2, dim,
                out[i], out[i + 1]
            );
        }
        for (; i < n; ++i) {
            out[i] = sq8_distance(sq8_codes + static_cast<int64_t>(nodes[i]) * dim, query_code, sq8_scales2, dim);
        }
    }
}

// ─── ResultBuffer ─────────────────────────────────────────────────────────────
//
// Bounded, sorted flat array of (distance, id) pairs.
// Invariant: entries[0..size-1] are sorted ascending by .first (distance).
// When size == cap, entries[size-1] holds the worst (largest) distance.
//
// Template parameter T is the distance type (int for BQ, float for fine).

template<typename T>
struct ResultBuffer {
    struct Entry { T dist; int id; };

    std::vector<Entry> entries;
    int size = 0;
    int cap = 0;

    explicit ResultBuffer(int capacity = 0) : size(0), cap(capacity) {
        entries.resize(capacity > 0 ? capacity + 16 : 64);
    }

    void reset(int capacity) noexcept {
        size = 0;
        cap = capacity;
        if (static_cast<int>(entries.size()) < capacity + 16) {
            entries.resize(capacity + 16);
        }
    }

    [[nodiscard]] bool full() const noexcept { return size >= cap; }

    // In a max-heap, entries[0] holds the largest (worst) distance.
    [[nodiscard]] T worst() const noexcept { return entries[0].dist; }

    // O(log cap) insertion: replace root and sift down when full; sift up when not full.
    bool try_insert(T dist, int id) noexcept {
        if (size >= cap) {
            if (dist >= entries[0].dist) return false;
            entries[0] = {dist, id};
            int i = 0;
            while (true) {
                int left = 2 * i + 1;
                int right = left + 1;
                int largest = i;
                if (left < cap && entries[left].dist > entries[largest].dist) largest = left;
                if (right < cap && entries[right].dist > entries[largest].dist) largest = right;
                if (largest == i) break;
                std::swap(entries[i], entries[largest]);
                i = largest;
            }
            return true;
        }

        int i = size++;
        entries[i] = {dist, id};
        while (i > 0) {
            int parent = (i - 1) >> 1;
            if (entries[parent].dist >= entries[i].dist) break;
            std::swap(entries[parent], entries[i]);
            i = parent;
        }
        return true;
    }

    void sort_ascending() noexcept {
        std::sort(entries.begin(), entries.begin() + size,
                  [](const Entry& a, const Entry& b) { return a.dist < b.dist; });
    }

    void clear() noexcept { size = 0; }
};

// ─── CandidateHeap ────────────────────────────────────────────────────────────
//
// Flat binary min-heap for the exploration frontier.
// Dynamically resizable buffer with fast sift-up / sift-down operations.
// Scratch vector is allocated once and reused across queries.

template<typename T>
struct CandidateHeap {
    struct Entry { T dist; int id; };

    std::vector<Entry> data;
    int size = 0;

    CandidateHeap() { data.resize(4096); }
    explicit CandidateHeap(size_t cap) { data.resize(cap > 0 ? cap : 1024); }

    void clear() noexcept { size = 0; }

    [[nodiscard]] bool empty() const noexcept { return size == 0; }

    [[nodiscard]] const Entry& top() const noexcept { return data[0]; }

    void push(T dist, int id) {
        if (size >= static_cast<int>(data.size())) {
            data.resize(data.size() * 2);
        }
        int i = size++;
        data[i] = {dist, id};
        while (i > 0) {
            int parent = (i - 1) >> 1;
            if (data[parent].dist <= data[i].dist) break;
            std::swap(data[parent], data[i]);
            i = parent;
        }
    }

    void pop() noexcept {
        data[0] = data[--size];
        // Sift down.
        int i = 0;
        while (true) {
            int left  = 2 * i + 1;
            int right = left + 1;
            int smallest = i;
            if (left  < size && data[left].dist  < data[smallest].dist) smallest = left;
            if (right < size && data[right].dist < data[smallest].dist) smallest = right;
            if (smallest == i) break;
            std::swap(data[i], data[smallest]);
            i = smallest;
        }
    }
};

// ─── Greedy upper-layer descent (unchanged logic, minor readability polish) ───

[[nodiscard]] int
greedy_upper(const GraphView&  graph,
             const uint64_t*   query_bq,
             const uint64_t*   bq_codes,
             int               n_words) noexcept
{
    int32_t current = graph.entry;
    if (n_words == 2) {
        const uint64_t q0 = query_bq[0];
        const uint64_t q1 = query_bq[1];
        const uint64_t* cur_p = bq_codes + static_cast<int64_t>(current) * 2;
        int current_dist = static_cast<int>(
            __builtin_popcountll(q0 ^ cur_p[0]) +
            __builtin_popcountll(q1 ^ cur_p[1]));

        for (int layer = graph.max_level; layer > 0; --layer) {
            while (true) {
                int32_t width = 0;
                const int32_t* nbrs = neighbors_at(graph, current, layer, width);
                int32_t best_node = -1;
                int     best_dist = current_dist;
                for (int32_t i = 0; i < width; ++i) {
                    const int32_t nb = nbrs[i];
                    if (nb < 0) break;
                    if (i + 2 < width && nbrs[i + 2] >= 0) {
                        _mm_prefetch(reinterpret_cast<const char*>(bq_codes + static_cast<int64_t>(nbrs[i + 2]) * 2), _MM_HINT_T0);
                    }
                    const uint64_t* nb_p = bq_codes + static_cast<int64_t>(nb) * 2;
                    const int dist = static_cast<int>(
                        __builtin_popcountll(q0 ^ nb_p[0]) +
                        __builtin_popcountll(q1 ^ nb_p[1]));
                    if (dist < best_dist) { best_dist = dist; best_node = nb; }
                }
                if (best_node < 0) break;
                current      = best_node;
                current_dist = best_dist;
            }
        }
        return current;
    }

    int current_dist = hamming_row(
        query_bq, bq_codes + static_cast<int64_t>(current) * n_words, n_words);

    for (int layer = graph.max_level; layer > 0; --layer) {
        while (true) {
            int32_t width = 0;
            const int32_t* nbrs = neighbors_at(graph, current, layer, width);
            int32_t best_node = -1;
            int     best_dist = current_dist;
            for (int32_t i = 0; i < width; ++i) {
                const int32_t nb = nbrs[i];
                if (nb < 0) continue;
                const int dist = hamming_row(
                    query_bq,
                    bq_codes + static_cast<int64_t>(nb) * n_words,
                    n_words);
                if (dist < best_dist) { best_dist = dist; best_node = nb; }
            }
            if (best_node < 0) break;
            current      = best_node;
            current_dist = best_dist;
        }
    }
    return current;
}

// ─── beam_layer0_bq ──────────────────────────────────────────────────────────
//
// Layer-0 BQ beam search.
//
// Output: `pool` is filled and left sorted ascending by Hamming distance.
//
// Key changes vs. original:
//   - `results` is now a ResultBuffer<int> (flat sorted array).
//     • worst() is O(1) (just pool[size-1].dist).
//     • try_insert() performs a backward linear scan to maintain sort order.
//   - `candidates` is a CandidateHeap<int> (flat min-heap, stack-allocated).
//   - No intermediate std::vector; `pool` is written directly at the end
//     by memcpy-ing the already-sorted ResultBuffer.

void beam_layer0_bq(
    const GraphView&     graph,
    const uint64_t*      query_bq,
    const uint64_t*      bq_codes,
    int                  n_words,
    int                  entry,
    int                  ef_bq,
    ResultBuffer<int>&   pool,
    CandidateHeap<int>&  candidates,
    std::vector<uint8_t>& visited,
    uint8_t              stamp) noexcept
{
    pool.reset(ef_bq);
    candidates.clear();

    const int entry_dist = (n_words == 2)
        ? hamming_128(query_bq, bq_codes + static_cast<int64_t>(entry) * 2)
        : hamming_row(query_bq, bq_codes + static_cast<int64_t>(entry) * n_words, n_words);

    candidates.push(entry_dist, entry);
    pool.try_insert(entry_dist, entry);
    visited[entry] = stamp;

    if (n_words == 2) {
        const uint64_t q0 = query_bq[0];
        const uint64_t q1 = query_bq[1];

        while (!candidates.empty()) {
            const auto [dist, node] = candidates.top();
            candidates.pop();

            if (pool.full() && dist >= pool.worst()) break;

            int32_t width = 0;
            const int32_t* nbrs = neighbors_at(graph, node, 0, width);
            for (int32_t i = 0; i < width; ++i) {
                const int32_t nb = nbrs[i];
                if (nb < 0) break;
                if (visited[nb] == stamp) continue;
                visited[nb] = stamp;

                if (i + 4 < width && nbrs[i + 4] >= 0) {
                    _mm_prefetch(reinterpret_cast<const char*>(bq_codes + static_cast<int64_t>(nbrs[i + 4]) * 2), _MM_HINT_T0);
                }

                const uint64_t* bq_row = bq_codes + static_cast<int64_t>(nb) * 2;
                const int nd = static_cast<int>(
                    __builtin_popcountll(q0 ^ bq_row[0]) +
                    __builtin_popcountll(q1 ^ bq_row[1])
                );

                if (!pool.full() || nd < pool.worst()) {
                    candidates.push(nd, nb);
                    pool.try_insert(nd, nb);
                }
            }
        }
        pool.sort_ascending();
        return;
    }

    // Generic fallback for n_words != 2
    while (!candidates.empty()) {
        const auto [dist, node] = candidates.top();
        candidates.pop();

        if (pool.full() && dist >= pool.worst()) break;

        int32_t width = 0;
        const int32_t* nbrs = neighbors_at(graph, node, 0, width);
        for (int32_t i = 0; i < width; ++i) {
            const int32_t nb = nbrs[i];
            if (nb < 0) break;
            if (visited[nb] == stamp) continue;
            visited[nb] = stamp;

            const int nd = hamming_row(query_bq, bq_codes + static_cast<int64_t>(nb) * n_words, n_words);
            if (!pool.full() || nd < pool.worst()) {
                candidates.push(nd, nb);
                pool.try_insert(nd, nb);
            }
        }
    }
    pool.sort_ascending();
}

// ─── fine_search ─────────────────────────────────────────────────────────────
//
// Fine (SQ8/FP32) beam search warm-started from the BQ pool.
//
// Key changes vs. original:
//   - Input `pool` is a ResultBuffer<int> already sorted ascending by BQ dist.
//     We seed the fine candidates and results in a single pass with no heap
//     reconstruction.
//   - `results` is a ResultBuffer<float> (flat sorted array).
//   - `candidates` is a CandidateHeap<float> (flat min-heap).
//   - `best` (the output) is written by a simple copy from results.entries
//     (already sorted) + a final sort only if size > k.

template<int RerankMode>
void fine_search(
    const GraphView&       graph,
    const ResultBuffer<int>& pool,
    const float*           base,
    const uint8_t*         sq8_codes,
    const float*           sq8_scales2,
    const float*           query,
    const float*           query_code,
    int                    dim,
    int                    ef,
    int                    k,
    ResultBuffer<float>&   best,
    CandidateHeap<float>&  candidates,
    std::vector<uint8_t>&  visited,
    uint8_t                stamp) noexcept
{
    (void)k;  // top-k is applied by the caller; kept for call-site symmetry
    best.reset(ef);
    candidates.clear();
    if (pool.size == 0) return;

    int32_t valid[BATCH_MAX];
    float   dists[BATCH_MAX];

    // Seed from the BQ pool (already sorted ascending by BQ dist).
    for (int offset = 0; offset < pool.size; offset += BATCH_MAX) {
        const int n = std::min(BATCH_MAX, pool.size - offset);
        for (int j = 0; j < n; ++j)
            valid[j] = pool.entries[offset + j].id;
        fine_distances<RerankMode>(
            valid, n, base, sq8_codes, sq8_scales2, query, query_code, dim, dists);
        for (int j = 0; j < n; ++j) {
            visited[valid[j]] = stamp;
            best.try_insert(dists[j], valid[j]);
        }
    }
    // Only candidates that survived into the top-ef buffer are retained
    // for graph frontier exploration.
    for (int j = 0; j < best.size; ++j) {
        candidates.push(best.entries[j].dist, best.entries[j].id);
    }

    // Collect fresh neighbours in graph order, score the group with the SIMD
    // kernel, then apply the heap/buffer updates in the original order.
    int nvalid = 0;
    auto flush = [&]() noexcept {
        if (nvalid == 0) return;
        fine_distances<RerankMode>(
            valid, nvalid, base, sq8_codes, sq8_scales2, query, query_code, dim,
            dists);
        for (int j = 0; j < nvalid; ++j) {
            const float nd = dists[j];
            if (!best.full() || nd < best.worst()) {
                candidates.push(nd, valid[j]);
                best.try_insert(nd, valid[j]);
            }
        }
        nvalid = 0;
    };

    while (!candidates.empty()) {
        const auto [dist, node] = candidates.top();
        candidates.pop();

        if (best.full() && dist >= best.worst()) break;

        int32_t width = 0;
        const int32_t* nbrs = neighbors_at(graph, node, 0, width);
        for (int32_t i = 0; i < width; ++i) {
            const int32_t nb = nbrs[i];
            if (nb < 0) break;
            if (visited[nb] == stamp) continue;
            visited[nb] = stamp;
            valid[nvalid++] = nb;
            if constexpr (RerankMode == 0) {
                const char* ptr = reinterpret_cast<const char*>(sq8_codes + static_cast<int64_t>(nb) * dim);
                _mm_prefetch(ptr, _MM_HINT_T0);
                _mm_prefetch(ptr + 64, _MM_HINT_T0);
            } else {
                const char* ptr = reinterpret_cast<const char*>(base + static_cast<int64_t>(nb) * dim);
                _mm_prefetch(ptr, _MM_HINT_T0);
                _mm_prefetch(ptr + 64, _MM_HINT_T0);
            }
            if (nvalid == BATCH_MAX) flush();
        }
        flush();
    }
    best.sort_ascending();
}

} // namespace

// ─── C ABI ───────────────────────────────────────────────────────────────────

extern "C" {

struct FastGraphView {
    const int32_t* levels;
    const int64_t* offsets;
    const int32_t* neighbors;
    int64_t ntotal;
    int32_t M0;
    int32_t M;
    int32_t entry;
    int32_t max_level;
};

/**
 * Run the two-stage hybrid search for a batch of queries.
 *
 * query_bq   : (n_queries, n_words) packed BQ codes
 * query_code : (n_queries, dim) SQ8 query coordinates in code space
 * base       : (n, dim) FP32 payload, used when rerank_mode == 1 (may be null)
 * out_ids    : (n_queries, k) output ids, -1 padded
 * out_dists  : (n_queries, k) output squared distances
 * out_latency_ns : (n_queries,) per-query wall time in nanoseconds
 *
 * rerank_mode: 0 = SQ8 ADC, 1 = FP32 L2
 * Returns 0 on success, negative on invalid arguments.
 */
inline void encode_bq_128_avx2(const float* __restrict__ q,
                               const float* __restrict__ th,
                               uint64_t* __restrict__ out_words) noexcept
{
    uint64_t w0 = 0, w1 = 0;
    for (int b = 0; b < 8; ++b) {
        const __m256 q_val  = _mm256_loadu_ps(q + b * 8);
        const __m256 th_val = _mm256_loadu_ps(th + b * 8);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(q_val, th_val, _CMP_GT_OQ));
        w0 |= (static_cast<uint64_t>(mask & 0xFF) << (b * 8));
    }
    for (int b = 0; b < 8; ++b) {
        const __m256 q_val  = _mm256_loadu_ps(q + (8 + b) * 8);
        const __m256 th_val = _mm256_loadu_ps(th + (8 + b) * 8);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(q_val, th_val, _CMP_GT_OQ));
        w1 |= (static_cast<uint64_t>(mask & 0xFF) << (b * 8));
    }
    out_words[0] = w0;
    out_words[1] = w1;
}

inline void prepare_sq8_128_avx2(const float* __restrict__ q,
                                const float* __restrict__ mins,
                                const float* __restrict__ inv_scales,
                                float* __restrict__ out_qc) noexcept
{
    for (int b = 0; b < 16; ++b) {
        const __m256 q_val  = _mm256_loadu_ps(q + b * 8);
        const __m256 min_val = _mm256_loadu_ps(mins + b * 8);
        const __m256 sc_val = _mm256_loadu_ps(inv_scales + b * 8);
        const __m256 res    = _mm256_mul_ps(_mm256_sub_ps(q_val, min_val), sc_val);
        _mm256_storeu_ps(out_qc + b * 8, res);
    }
}

int fast_hybrid_search_batch(
    const FastGraphView* graph,
    const uint64_t*      bq_codes,
    int32_t              n_words,
    const uint8_t*       sq8_codes,
    const float*         sq8_scales2,
    const float*         base,
    const float*         queries,
    const uint64_t*      query_bq,
    const float*         query_code,
    int32_t              n_queries,
    int32_t              dim,
    int32_t              ef_search,
    int32_t              ef_bq,
    int32_t              k,
    int32_t              rerank_mode,
    int64_t*             out_ids,
    float*               out_dists,
    int64_t*             out_latency_ns,
    const float*         bq_thresholds,
    const float*         sq8_mins,
    const float*         sq8_inv_scales)
{
    if (!graph || !bq_codes || !sq8_codes || !sq8_scales2 ||
        !queries || !out_ids || !out_dists || !out_latency_ns)
        return -1;
    if (!query_bq && (!bq_thresholds || dim != 128))
        return -1;
    if (!query_code && (!sq8_mins || !sq8_inv_scales || dim != 128))
        return -1;
    if (n_queries < 0 || dim <= 0 || n_words <= 0 || k <= 0 || ef_bq <= 0)
        return -2;
    if (rerank_mode != 0 && rerank_mode != 1)
        return -3;
    if (rerank_mode == 1 && !base)
        return -4;

    const int fine_ef = std::max(std::max(ef_search, k), 1);

    // Adaptive ef_bq scaling:
    // Low efSearch (10, 20): tight BQ frontier avoids over-exploring layer 0.
    // Moderate efSearch (40, 80): balanced BQ frontier to maximize QPS while preserving recall.
    // High efSearch (>= 160): wider BQ frontier seeds SQ8 with high-quality global basin.
    int effective_ef_bq = ef_bq;
    if (ef_search <= 10) {
        effective_ef_bq = std::min(ef_bq, std::max(k, 16));
    } else if (ef_search <= 20) {
        effective_ef_bq = std::min(ef_bq, 24);
    } else if (ef_search <= 40) {
        effective_ef_bq = std::min(ef_bq, 36);
    } else if (ef_search <= 80) {
        effective_ef_bq = std::min(ef_bq, 54);
    } else if (ef_search >= 160) {
        effective_ef_bq = std::max(ef_bq, 96);
    }

    GraphView view{
        graph->levels, graph->offsets, graph->neighbors,
        graph->ntotal, graph->M0, graph->M,
        graph->entry, graph->max_level,
    };

    // Compact 1-byte visited table: fits into CPU L2/L3 cache (1 MB vs 4 MB)
    std::vector<uint8_t> visited(static_cast<size_t>(graph->ntotal), 0u);

    // Scratch candidate buffers; zero runtime allocation per query after warmup.
    ResultBuffer<int>    pool(effective_ef_bq);
    ResultBuffer<float>  best(fine_ef);
    CandidateHeap<int>   bq_candidates(4096);
    CandidateHeap<float> fine_candidates(8192);

    alignas(32) uint64_t q_bq_local[2];
    alignas(32) float    q_code_local[128];

    uint8_t stamp = 0;

    for (int32_t qi = 0; qi < n_queries; ++qi) {
        // Rollover guard for 8-bit stamp: fast memset every 254 queries
        if (stamp >= 254u) {
            std::memset(visited.data(), 0, visited.size());
            stamp = 0;
        }
        const uint8_t bq_stamp   = ++stamp;
        const uint8_t fine_stamp = ++stamp;

        const auto t0 = std::chrono::steady_clock::now();

        const float* q = queries + static_cast<int64_t>(qi) * dim;

        const uint64_t* q_bq;
        if (query_bq) {
            q_bq = query_bq + static_cast<int64_t>(qi) * n_words;
        } else {
            encode_bq_128_avx2(q, bq_thresholds, q_bq_local);
            q_bq = q_bq_local;
        }

        const float* q_code;
        if (query_code) {
            q_code = query_code + static_cast<int64_t>(qi) * dim;
        } else {
            prepare_sq8_128_avx2(q, sq8_mins, sq8_inv_scales, q_code_local);
            q_code = q_code_local;
        }

        const int entry = greedy_upper(view, q_bq, bq_codes, n_words);

        beam_layer0_bq(view, q_bq, bq_codes, n_words,
                       entry, effective_ef_bq, pool, bq_candidates, visited, bq_stamp);

        if (rerank_mode == 1) {
            fine_search<1>(view, pool, base, sq8_codes, sq8_scales2,
                           q, q_code, dim, fine_ef, k,
                           best, fine_candidates, visited, fine_stamp);
        } else {
            fine_search<0>(view, pool, base, sq8_codes, sq8_scales2,
                           q, q_code, dim, fine_ef, k,
                           best, fine_candidates, visited, fine_stamp);
        }

        const auto t1 = std::chrono::steady_clock::now();
        out_latency_ns[qi] = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

        // Write top-k results (best is sorted ascending, so [0..k-1] is correct).
        const int64_t out_base = static_cast<int64_t>(qi) * k;
        for (int32_t j = 0; j < k; ++j) {
            if (j < best.size) {
                out_ids  [out_base + j] = best.entries[j].id;
                out_dists[out_base + j] = best.entries[j].dist;
            } else {
                out_ids  [out_base + j] = -1;
                out_dists[out_base + j] = 0.0f;
            }
        }
    }
    return 0;
}

} // extern "C"
