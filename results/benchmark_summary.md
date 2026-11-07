## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M &nbsp;|&nbsp; **base:** 1,000,000 &nbsp;|&nbsp; **queries:** 2,000 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | A — Baseline FP32 HNSW | 0.7584 | 0.034 | 0.048 | 28197.1 | 3068 | 0.8803 |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.7713 | 0.205 | 0.235 | 4743.6 | 436 | 0.1168 |
| 20 | A — Baseline FP32 HNSW | 0.8754 | 0.053 | 0.068 | 18876.1 | 4727 | 0.8925 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.8842 | 0.223 | 0.251 | 4420.8 | 476 | 0.1204 |
| 40 | A — Baseline FP32 HNSW | 0.9517 | 0.088 | 0.110 | 11499.4 | 7954 | 0.9028 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9476 | 0.258 | 0.295 | 3816.1 | 608 | 0.1371 |
| 80 | A — Baseline FP32 HNSW | 0.9851 | 0.161 | 0.196 | 6347.6 | 13991 | 0.8979 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9760 | 0.328 | 0.373 | 3040.2 | 835 | 0.1534 |
| 160 | A — Baseline FP32 HNSW | 0.9961 | 0.293 | 0.358 | 3489.8 | 24620 | 0.9142 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9851 | 0.451 | 0.523 | 2240.8 | 1265 | 0.1692 |
| 320 | A — Baseline FP32 HNSW | 0.9990 | 0.539 | 0.659 | 1897.5 | 43318 | 0.9238 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9872 | 0.693 | 0.832 | 1459.7 | 2177 | 0.1846 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.7713) | A (28197.1) | Baseline FP32 HNSW |
| 20 | D (0.8842) | A (18876.1) | Baseline FP32 HNSW |
| 40 | A (0.9517) | A (11499.4) | Baseline FP32 HNSW |
| 80 | A (0.9851) | A (6347.6) | Baseline FP32 HNSW |
| 160 | A (0.9961) | A (3489.8) | Baseline FP32 HNSW |
| 320 | A (0.9990) | A (1897.5) | Baseline FP32 HNSW |

_Generated 2026-11-07T10:42:25+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
