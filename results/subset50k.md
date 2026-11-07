## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.8586 | 0.019 | 0.048 | 43167.9 | 677 | 0.4030 |
| 10 | B — Uniform SQ8 HNSW | 0.8510 | 0.018 | 0.026 | 52440.0 | 552 | 0.2927 |
| 10 | C — Uniform BQ HNSW | 0.1320 | 0.019 | 0.025 | 49603.4 | 532 | 0.2014 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8762 | 1.138 | 1.370 | 874.9 | 1045 | 0.3293 |
| 20 | A — Baseline FP32 HNSW | 0.9504 | 0.026 | 0.032 | 37977.7 | 512 | 0.2027 |
| 20 | B — Uniform SQ8 HNSW | 0.9440 | 0.026 | 0.034 | 36942.1 | 694 | 0.2447 |
| 20 | C — Uniform BQ HNSW | 0.1330 | 0.030 | 0.038 | 32559.0 | 791 | 0.2078 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9534 | 1.262 | 1.511 | 782.1 | 1307 | 0.3418 |
| 40 | A — Baseline FP32 HNSW | 0.9898 | 0.044 | 0.054 | 22423.4 | 1183 | 0.2789 |
| 40 | B — Uniform SQ8 HNSW | 0.9818 | 0.040 | 0.047 | 24700.9 | 812 | 0.1694 |
| 40 | C — Uniform BQ HNSW | 0.1354 | 0.054 | 0.083 | 17227.1 | 1798 | 0.3005 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9846 | 1.565 | 1.790 | 644.6 | 1796 | 0.3553 |
| 80 | A — Baseline FP32 HNSW | 0.9988 | 0.072 | 0.084 | 14015.8 | 1412 | 0.1872 |
| 80 | B — Uniform SQ8 HNSW | 0.9910 | 0.071 | 0.086 | 14086.0 | 1693 | 0.2024 |
| 80 | C — Uniform BQ HNSW | 0.1360 | 0.087 | 0.102 | 11619.2 | 1698 | 0.1606 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 2.046 | 2.317 | 496.6 | 2262 | 0.3297 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8762) | B (52440.0) | Uniform SQ8 HNSW |
| 20 | D (0.9534) | A (37977.7) | Baseline FP32 HNSW |
| 40 | A (0.9898) | B (24700.9) | Uniform SQ8 HNSW |
| 80 | A (0.9988) | B (14086.0) | Uniform SQ8 HNSW |

_Generated 2026-11-07T09:40:11+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
