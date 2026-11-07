## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8746 | 0.058 | 0.090 | 15127.3 | 126 | 0.1997 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9530 | 0.060 | 0.077 | 15737.3 | 62 | 0.0988 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9848 | 0.068 | 0.087 | 14227.2 | 44 | 0.0490 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 0.087 | 0.108 | 11264.1 | 53 | 0.0379 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.124 | 0.157 | 7977.5 | 87 | 0.0397 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.199 | 0.364 | 4533.0 | 249 | 0.0727 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8746) | D (15127.3) | Two-Stage Hybrid (BQ + rerank) |
| 20 | D (0.9530) | D (15737.3) | Two-Stage Hybrid (BQ + rerank) |
| 40 | D (0.9848) | D (14227.2) | Two-Stage Hybrid (BQ + rerank) |
| 80 | D (0.9916) | D (11264.1) | Two-Stage Hybrid (BQ + rerank) |
| 160 | D (0.9920) | D (7977.5) | Two-Stage Hybrid (BQ + rerank) |
| 320 | D (0.9920) | D (4533.0) | Two-Stage Hybrid (BQ + rerank) |

_Generated 2026-11-07T10:15:10+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
