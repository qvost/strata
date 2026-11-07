## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M &nbsp;|&nbsp; **base:** 1,000,000 &nbsp;|&nbsp; **queries:** 2,000 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.7584 | 0.041 | 0.058 | 23489.5 | 2966 | 0.9170 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.7382 | 0.103 | 0.124 | 9205.9 | 199 | 0.2094 |
| 20 | A — Baseline FP32 HNSW | 0.8754 | 0.066 | 0.088 | 14794.8 | 4619 | 0.9300 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.8689 | 0.141 | 0.212 | 6563.8 | 318 | 0.2560 |
| 40 | A — Baseline FP32 HNSW | 0.9517 | 0.115 | 0.144 | 8737.5 | 7732 | 0.9203 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9446 | 0.219 | 0.356 | 4191.2 | 534 | 0.2673 |
| 80 | A — Baseline FP32 HNSW | 0.9851 | 0.214 | 0.265 | 4726.3 | 13614 | 0.9257 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9756 | 0.320 | 0.439 | 3005.0 | 769 | 0.2717 |
| 160 | A — Baseline FP32 HNSW | 0.9961 | 0.371 | 0.468 | 2712.9 | 23892 | 0.9362 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9850 | 0.497 | 0.619 | 1999.9 | 1236 | 0.2535 |
| 320 | A — Baseline FP32 HNSW | 0.9990 | 0.715 | 0.903 | 1418.3 | 42792 | 0.9534 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9872 | 0.572 | 0.728 | 1746.8 | 1824 | 0.2362 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | A (0.7584) | A (23489.5) | Baseline FP32 HNSW |
| 20 | A (0.8754) | A (14794.8) | Baseline FP32 HNSW |
| 40 | A (0.9517) | A (8737.5) | Baseline FP32 HNSW |
| 80 | A (0.9851) | A (4726.3) | Baseline FP32 HNSW |
| 160 | A (0.9961) | A (2712.9) | Baseline FP32 HNSW |
| 320 | A (0.9990) | D (1746.8) | Two-Stage Hybrid (BQ + rerank) |

_Generated 2026-11-07T11:41:26+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
