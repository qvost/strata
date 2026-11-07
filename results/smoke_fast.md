## Two-Stage Traversal Benchmark

**Dataset:** synthetic-5000 &nbsp;|&nbsp; **base:** 5,000 &nbsp;|&nbsp; **queries:** 100 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=16, efConstruction=100 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.9560 | 0.009 | 0.013 | 101809.1 | 4 | 0.0094 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.9720 | 0.046 | 0.056 | 20877.2 | 8 | 0.0374 |
| 40 | A — Baseline FP32 HNSW | 1.0000 | 0.014 | 0.018 | 66356.8 | 2 | 0.0025 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9760 | 0.051 | 0.060 | 18521.1 | 1 | 0.0037 |
| 80 | A — Baseline FP32 HNSW | 1.0000 | 0.021 | 0.026 | 45092.6 | 1 | 0.0007 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9760 | 0.068 | 0.111 | 13034.3 | 28 | 0.0536 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.9720) | A (101809.1) | Baseline FP32 HNSW |
| 40 | A (1.0000) | A (66356.8) | Baseline FP32 HNSW |
| 80 | A (1.0000) | A (45092.6) | Baseline FP32 HNSW |

_Generated 2026-11-07T09:43:19+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
