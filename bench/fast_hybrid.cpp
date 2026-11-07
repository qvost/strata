// fast_hybrid.cpp
//
// Compiled two-stage HNSW traversal used by bench/runner.py.
//
// Stage 1 (binary):
//   * greedy descent through layers L > 0 with 64-bit POPCNT Hamming distance
//   * layer-0 beam search with a fixed ef_bq candidate buffer
// Stage 2 (fine):
//   * warm-started beam search over layer 0 scored with SQ8 ADC or FP32 L2
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
hamming_row(const uint64_t* __restrict__ a,
            const uint64_t* __restrict__ b,
            int n_words) noexcept
{
    // For n_words == 2 (128-d vectors) the loop is fully unrolled by -O3.
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
        for (int i = 0; i < n; ++i) {
            const int64_t row = static_cast<int64_t>(nodes[i]) * dim;
            if (i + 1 < n) {
                const int64_t next = static_cast<int64_t>(nodes[i + 1]) * dim;
                const char* p = reinterpret_cast<const char*>(sq8_codes + next);
                for (int off = 0; off < dim; off += 64)
                    _mm_prefetch(p + off, _MM_HINT_T0);
            }
            out[i] = sq8_distance(sq8_codes + row, query_code, sq8_scales2, dim);
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

    // Worst (largest) distance currently in the buffer.
    [[nodiscard]] T worst() const noexcept { return entries[size - 1].dist; }

    // Try to insert (dist, id).  Returns true if inserted.
    // Maintains sorted order with a linear backward scan + shift.
    bool try_insert(T dist, int id) noexcept {
        if (size == cap) {
            if (dist >= worst()) return false;
            int pos = cap - 1;
            while (pos > 0 && entries[pos - 1].dist > dist) {
                entries[pos] = entries[pos - 1];
                --pos;
            }
            entries[pos] = {dist, id};
            return true;
        }

        int pos = size;
        ++size;
        while (pos > 0 && entries[pos - 1].dist > dist) {
            entries[pos] = entries[pos - 1];
            --pos;
        }
        entries[pos] = {dist, id};
        return true;
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
    std::vector<uint32_t>& visited,
    uint32_t             stamp) noexcept
{
    pool.reset(ef_bq);
    candidates.clear();

    const int entry_dist = hamming_row(
        query_bq, bq_codes + static_cast<int64_t>(entry) * n_words, n_words);

    candidates.push(entry_dist, entry);
    pool.try_insert(entry_dist, entry);
    visited[entry] = stamp;

    // Collect fresh neighbours in graph order, prefetch/score the group, then
    // apply the candidate/pool updates in the same order the scalar loop used.
    int32_t valid[BATCH_MAX];
    int     dists[BATCH_MAX];
    int     nvalid = 0;
    auto flush = [&]() noexcept {
        if (nvalid == 0) return;
        hamming_batch(query_bq, bq_codes, valid, nvalid, n_words, dists);
        for (int j = 0; j < nvalid; ++j) {
            const int nd = dists[j];
            if (!pool.full() || nd < pool.worst()) {
                candidates.push(nd, valid[j]);
                pool.try_insert(nd, valid[j]);
            }
        }
        nvalid = 0;
    };

    while (!candidates.empty()) {
        const auto [dist, node] = candidates.top();
        candidates.pop();

        // Early-exit: if the pool is full and this candidate is already
        // worse than the current worst result, no neighbour can improve it.
        if (pool.full() && dist > pool.worst()) break;

        int32_t width = 0;
        const int32_t* nbrs = neighbors_at(graph, node, 0, width);
        for (int32_t i = 0; i < width; ++i) {
            const int32_t nb = nbrs[i];
            if (nb < 0 || visited[nb] == stamp) continue;
            visited[nb] = stamp;
            valid[nvalid++] = nb;
            if (nvalid == BATCH_MAX) flush();
        }
        flush();
    }
    // pool.entries[0..pool.size-1] is already sorted ascending.
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
    std::vector<uint32_t>& visited,
    uint32_t               stamp) noexcept
{
    (void)k;  // top-k is applied by the caller; kept for call-site symmetry
    best.reset(ef);
    candidates.clear();
    if (pool.size == 0) return;

    int32_t valid[BATCH_MAX];
    float   dists[BATCH_MAX];

    // Seed from the BQ pool (already sorted ascending by BQ dist → we process
    // them in that order, which front-loads the most promising candidates into
    // the fine candidates heap for free).
    for (int offset = 0; offset < pool.size; offset += BATCH_MAX) {
        const int n = std::min(BATCH_MAX, pool.size - offset);
        for (int j = 0; j < n; ++j)
            valid[j] = pool.entries[offset + j].id;
        fine_distances<RerankMode>(
            valid, n, base, sq8_codes, sq8_scales2, query, query_code, dim, dists);
        for (int j = 0; j < n; ++j) {
            const float dist = dists[j];
            visited[valid[j]] = stamp;
            candidates.push(dist, valid[j]);
            best.try_insert(dist, valid[j]);
        }
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

        if (best.full() && dist > best.worst()) break;

        int32_t width = 0;
        const int32_t* nbrs = neighbors_at(graph, node, 0, width);
        for (int32_t i = 0; i < width; ++i) {
            const int32_t nb = nbrs[i];
            if (nb < 0 || visited[nb] == stamp) continue;
            visited[nb] = stamp;
            valid[nvalid++] = nb;
            if (nvalid == BATCH_MAX) flush();
        }
        flush();
    }
    // best.entries[0..best.size-1] is sorted ascending by fine distance.
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
    int64_t*             out_latency_ns)
{
    if (!graph || !bq_codes || !sq8_codes || !sq8_scales2 ||
        !queries || !query_bq || !query_code ||
        !out_ids || !out_dists || !out_latency_ns)
        return -1;
    if (n_queries < 0 || dim <= 0 || n_words <= 0 || k <= 0 || ef_bq <= 0)
        return -2;
    if (rerank_mode != 0 && rerank_mode != 1)
        return -3;
    if (rerank_mode == 1 && !base)
        return -4;

    const int fine_ef = std::max(std::max(ef_search, k), 1);

    GraphView view{
        graph->levels, graph->offsets, graph->neighbors,
        graph->ntotal, graph->M0, graph->M,
        graph->entry, graph->max_level,
    };

    // Per-query scratch — allocated once and reused across all queries.
    std::vector<uint32_t> visited(static_cast<size_t>(graph->ntotal), 0u);

    // Scratch candidate buffers; zero runtime allocation per query after warmup.
    ResultBuffer<int>    pool(ef_bq);
    ResultBuffer<float>  best(fine_ef);
    CandidateHeap<int>   bq_candidates(4096);
    CandidateHeap<float> fine_candidates(8192);

    uint32_t stamp = 0;

    for (int32_t qi = 0; qi < n_queries; ++qi) {
        // Stamp rollover guard: reset visited array when stamps near overflow.
        if (stamp > 0xFFFFFFFDu) {
            std::fill(visited.begin(), visited.end(), 0u);
            stamp = 0;
        }
        const uint32_t bq_stamp   = ++stamp;
        const uint32_t fine_stamp = ++stamp;

        const auto t0 = std::chrono::steady_clock::now();

        const uint64_t* q_bq    = query_bq   + static_cast<int64_t>(qi) * n_words;
        const float*    q       = queries    + static_cast<int64_t>(qi) * dim;
        const float*    q_code  = query_code + static_cast<int64_t>(qi) * dim;

        const int entry = greedy_upper(view, q_bq, bq_codes, n_words);

        beam_layer0_bq(view, q_bq, bq_codes, n_words,
                       entry, ef_bq, pool, bq_candidates, visited, bq_stamp);

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
