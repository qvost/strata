## Two-Stage Traversal Benchmark

**Dataset:** synthetic-5000 &nbsp;|&nbsp; **base:** 5,000 &nbsp;|&nbsp; **queries:** 100 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=8, efConstruction=60 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

> Note: hardware counters could not be opened; LLC columns are `n/a`.

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.9170 | 0.008 | 0.012 | 110078.3 | 9 | 0.0207 |
| 10 | B — Uniform SQ8 HNSW | 0.8970 | 0.008 | 0.012 | 112753.0 | 7 | 0.0178 |
| 10 | C — Uniform BQ HNSW | 0.2700 | 0.009 | 0.011 | 106057.2 | 13 | 0.0258 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.9630 | 0.480 | 0.721 | 1951.4 | 132 | 0.1892 |
| 40 | A — Baseline FP32 HNSW | 0.9970 | 0.013 | 0.018 | 72013.7 | 7 | 0.0103 |
| 40 | B — Uniform SQ8 HNSW | 0.9730 | 0.012 | 0.019 | 75250.4 | 2 | 0.0040 |
| 40 | C — Uniform BQ HNSW | 0.2730 | 0.015 | 0.019 | 63139.6 | 9 | 0.0093 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9750 | 0.593 | 0.847 | 1587.3 | 139 | 0.1653 |
| 80 | A — Baseline FP32 HNSW | 0.9990 | 0.018 | 0.022 | 53400.2 | 4 | 0.0046 |
| 80 | B — Uniform SQ8 HNSW | 0.9750 | 0.018 | 0.022 | 54468.2 | 2 | 0.0021 |
| 80 | C — Uniform BQ HNSW | 0.2730 | 0.024 | 0.031 | 39483.1 | 69 | 0.0462 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9750 | 0.820 | 1.080 | 1161.1 | 472 | 0.3682 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.9630) | B (112753.0) | Uniform SQ8 HNSW |
| 40 | A (0.9970) | B (75250.4) | Uniform SQ8 HNSW |
| 80 | A (0.9990) | B (54468.2) | Uniform SQ8 HNSW |

_Generated 2026-11-07T09:39:55+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
