## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.8586 | 0.021 | 0.033 | 44005.8 | 729 | 0.4340 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8746 | 0.092 | 0.124 | 9913.2 | 278 | 0.3735 |
| 20 | A — Baseline FP32 HNSW | 0.9504 | 0.030 | 0.044 | 31235.6 | 1023 | 0.3998 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9530 | 0.100 | 0.133 | 9412.5 | 276 | 0.3622 |
| 40 | A — Baseline FP32 HNSW | 0.9898 | 0.053 | 0.106 | 16604.4 | 2236 | 0.4734 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9848 | 0.122 | 0.165 | 7862.7 | 341 | 0.3216 |
| 80 | A — Baseline FP32 HNSW | 0.9988 | 0.090 | 0.151 | 10273.3 | 3422 | 0.4363 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 0.152 | 0.210 | 6195.7 | 409 | 0.2454 |
| 160 | A — Baseline FP32 HNSW | 1.0000 | 0.176 | 0.289 | 5322.3 | 6557 | 0.4825 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.209 | 0.376 | 4467.1 | 548 | 0.2149 |
| 320 | A — Baseline FP32 HNSW | 1.0000 | 0.358 | 0.540 | 2674.8 | 12837 | 0.5639 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.309 | 0.394 | 3186.6 | 782 | 0.2031 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8746) | A (44005.8) | Baseline FP32 HNSW |
| 20 | D (0.9530) | A (31235.6) | Baseline FP32 HNSW |
| 40 | A (0.9898) | A (16604.4) | Baseline FP32 HNSW |
| 80 | A (0.9988) | A (10273.3) | Baseline FP32 HNSW |
| 160 | A (1.0000) | A (5322.3) | Baseline FP32 HNSW |
| 320 | A (1.0000) | D (3186.6) | Two-Stage Hybrid (BQ + rerank) |

_Generated 2026-11-07T10:17:17+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
