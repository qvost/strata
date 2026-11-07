## Two-Stage Traversal Benchmark

**Dataset:** SIFT1M[50000] &nbsp;|&nbsp; **base:** 50,000 &nbsp;|&nbsp; **queries:** 500 &nbsp;|&nbsp; **dim:** 128 &nbsp;|&nbsp; **k:** 10

**HNSW:** M=32, efConstruction=200 &nbsp;|&nbsp; **BQ buffer:** ef_bq=64 &nbsp;|&nbsp; **hybrid rerank:** sq8 &nbsp;|&nbsp; **threads:** 1

| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |
| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | D — Two-Stage Hybrid (BQ + rerank) | 0.8768 | 0.067 | 0.081 | 13831.6 | 130 | 0.1816 |
| 20 | D — Two-Stage Hybrid (BQ + rerank) | 0.9536 | 0.072 | 0.083 | 13622.9 | 30 | 0.0450 |
| 40 | D — Two-Stage Hybrid (BQ + rerank) | 0.9846 | 0.084 | 0.133 | 10894.6 | 80 | 0.0858 |
| 80 | D — Two-Stage Hybrid (BQ + rerank) | 0.9916 | 0.103 | 0.117 | 9575.5 | 32 | 0.0223 |
| 160 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.145 | 0.165 | 6913.3 | 41 | 0.0189 |
| 320 | D — Two-Stage Hybrid (BQ + rerank) | 0.9920 | 0.224 | 0.359 | 4231.1 | 186 | 0.0556 |

### Per-efSearch leaders

| efSearch | Best Recall@10 | Best QPS | Fastest config |
| ---: | :--- | :--- | :--- |
| 10 | D (0.8768) | D (13831.6) | Two-Stage Hybrid (BQ + rerank) |
| 20 | D (0.9536) | D (13622.9) | Two-Stage Hybrid (BQ + rerank) |
| 40 | D (0.9846) | D (10894.6) | Two-Stage Hybrid (BQ + rerank) |
| 80 | D (0.9916) | D (9575.5) | Two-Stage Hybrid (BQ + rerank) |
| 160 | D (0.9920) | D (6913.3) | Two-Stage Hybrid (BQ + rerank) |
| 320 | D (0.9920) | D (4231.1) | Two-Stage Hybrid (BQ + rerank) |

_Generated 2026-11-07T10:13:30+00:00 on Linux-7.2.11-arch1-1-x86_64-with-glibc2.44 (13th Gen Intel(R) Core(TM) i7-13620H)._
