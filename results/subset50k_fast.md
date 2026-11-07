## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.8586 | 0.018 | 0.023 | 52306.2 | 571 | 0.2795 |
| 10 | B — Uniform SQ8 HNSW | 0.8510 | 0.018 | 0.024 | 53260.8 | 429 | 0.2447 |
| 10 | C — Uniform BQ HNSW | 0.1320 | 0.025 | 0.033 | 38325.5 | 1118 | 0.3874 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8768 | 0.093 | 0.110 | 10140.6 | 160 | 0.1003 |
| 20 | A — Baseline FP32 HNSW | 0.9504 | 0.025 | 0.031 | 38702.9 | 542 | 0.1817 |
| 20 | B — Uniform SQ8 HNSW | 0.9440 | 0.026 | 0.033 | 37207.4 | 478 | 0.1889 |
| 20 | C — Uniform BQ HNSW | 0.1330 | 0.039 | 0.080 | 22353.6 | 1405 | 0.3669 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9536 | 0.120 | 0.163 | 8062.9 | 272 | 0.1469 |
| 40 | A — Baseline FP32 HNSW | 0.9898 | 0.042 | 0.049 | 24021.3 | 798 | 0.1814 |
| 40 | B — Uniform SQ8 HNSW | 0.9818 | 0.041 | 0.052 | 23733.7 | 778 | 0.1822 |
| 40 | C — Uniform BQ HNSW | 0.1354 | 0.062 | 0.088 | 15426.5 | 1919 | 0.2905 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9846 | 0.132 | 0.150 | 7569.6 | 83 | 0.0354 |
| 80 | A — Baseline FP32 HNSW | 0.9988 | 0.074 | 0.098 | 13075.5 | 1809 | 0.2298 |
| 80 | B — Uniform SQ8 HNSW | 0.9910 | 0.077 | 0.135 | 11499.0 | 2348 | 0.3032 |
| 80 | C — Uniform BQ HNSW | 0.1360 | 0.112 | 0.148 | 8810.1 | 3347 | 0.2938 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 0.192 | 0.297 | 4933.8 | 310 | 0.0926 |
| 160 | A — Baseline FP32 HNSW | 1.0000 | 0.133 | 0.177 | 7337.5 | 2998 | 0.2242 |
| 160 | B — Uniform SQ8 HNSW | 0.9920 | 0.142 | 0.203 | 6780.3 | 3479 | 0.2669 |
| 160 | C — Uniform BQ HNSW | 0.1354 | 0.203 | 0.292 | 4709.3 | 7220 | 0.3733 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.274 | 0.350 | 3637.1 | 240 | 0.0509 |
| 320 | A — Baseline FP32 HNSW | 1.0000 | 0.259 | 0.330 | 3813.0 | 6170 | 0.2786 |
| 320 | B — Uniform SQ8 HNSW | 0.9920 | 0.260 | 0.301 | 3876.3 | 4732 | 0.2247 |
| 320 | C — Uniform BQ HNSW | 0.1362 | 0.396 | 0.505 | 2483.3 | 13657 | 0.4331 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.441 | 0.568 | 2257.8 | 684 | 0.0988 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8768) | B (53260.8) | Uniform SQ8 HNSW |
| 20 | D (0.9536) | A (38702.9) | Baseline FP32 HNSW |
| 40 | A (0.9898) | A (24021.3) | Baseline FP32 HNSW |
| 80 | A (0.9988) | A (13075.5) | Baseline FP32 HNSW |
| 160 | A (1.0000) | A (7337.5) | Baseline FP32 HNSW |
| 320 | A (1.0000) | B (3876.3) | Uniform SQ8 HNSW |

_Generated 2026-11-07T09:43:23+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
