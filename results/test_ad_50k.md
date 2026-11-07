## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.8586 | 0.027 | 0.038 | 33865.9 | 1056 | 0.5629 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8768 | 0.161 | 0.240 | 5667.7 | 586 | 0.3445 |
| 20 | A — Baseline FP32 HNSW | 0.9504 | 0.038 | 0.049 | 25792.5 | 1243 | 0.4522 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9536 | 0.186 | 0.223 | 5274.6 | 584 | 0.3245 |
| 40 | A — Baseline FP32 HNSW | 0.9898 | 0.062 | 0.076 | 16052.6 | 2140 | 0.4608 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9846 | 0.239 | 0.315 | 4103.5 | 870 | 0.3621 |
| 80 | A — Baseline FP32 HNSW | 0.9988 | 0.139 | 0.232 | 6633.1 | 5493 | 0.6379 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 0.330 | 0.417 | 3006.4 | 1261 | 0.3760 |
| 160 | A — Baseline FP32 HNSW | 1.0000 | 0.232 | 0.325 | 4171.4 | 9138 | 0.6144 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.542 | 0.961 | 1709.7 | 2348 | 0.4765 |
| 320 | A — Baseline FP32 HNSW | 1.0000 | 0.472 | 0.670 | 2073.9 | 17000 | 0.6948 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.821 | 1.198 | 1163.1 | 3033 | 0.4383 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8768) | A (33865.9) | Baseline FP32 HNSW |
| 20 | D (0.9536) | A (25792.5) | Baseline FP32 HNSW |
| 40 | A (0.9898) | A (16052.6) | Baseline FP32 HNSW |
| 80 | A (0.9988) | A (6633.1) | Baseline FP32 HNSW |
| 160 | A (1.0000) | A (4171.4) | Baseline FP32 HNSW |
| 320 | A (1.0000) | A (2073.9) | Baseline FP32 HNSW |

_Generated 2026-11-07T10:00:45+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
