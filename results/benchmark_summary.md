## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M &nbsp;|&nbsp; **base:** 1,000,000 &nbsp;|&nbsp; **queries:** 2,000 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.7584 | 0.032 | 0.044 | 30060.4 | 3028 | 0.8709 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.7382 | 0.025 | 0.035 | 38810.3 | 160 | 0.2056 |
| 20 | A — Baseline FP32 HNSW | 0.8754 | 0.049 | 0.066 | 19950.2 | 4706 | 0.9021 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.8690 | 0.039 | 0.050 | 25581.6 | 228 | 0.1982 |
| 40 | A — Baseline FP32 HNSW | 0.9517 | 0.083 | 0.105 | 11976.1 | 7886 | 0.9058 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9439 | 0.066 | 0.081 | 15251.9 | 363 | 0.2073 |
| 80 | A — Baseline FP32 HNSW | 0.9851 | 0.146 | 0.179 | 6968.0 | 14448 | 0.9008 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9754 | 0.113 | 0.137 | 8875.6 | 621 | 0.2239 |
| 160 | A — Baseline FP32 HNSW | 0.9961 | 0.273 | 0.333 | 3732.8 | 24563 | 0.9161 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9850 | 0.208 | 0.246 | 4923.4 | 1130 | 0.2336 |
| 320 | A — Baseline FP32 HNSW | 0.9990 | 0.512 | 0.625 | 1996.2 | 44010 | 0.9265 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9872 | 0.344 | 0.419 | 2924.6 | 1677 | 0.2182 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | A (0.7584) | D (38810.3) | Two-Stage Hybrid (BQ + rerank) |
| 20 | A (0.8754) | D (25581.6) | Two-Stage Hybrid (BQ + rerank) |
| 40 | A (0.9517) | D (15251.9) | Two-Stage Hybrid (BQ + rerank) |
| 80 | A (0.9851) | D (8875.6) | Two-Stage Hybrid (BQ + rerank) |
| 160 | A (0.9961) | D (4923.4) | Two-Stage Hybrid (BQ + rerank) |
| 320 | A (0.9990) | D (2924.6) | Two-Stage Hybrid (BQ + rerank) |

_Generated 2026-11-07T11:53:39+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
