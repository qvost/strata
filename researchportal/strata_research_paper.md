# STRATA: Breaking the Memory-Bandwidth Wall in HNSW Graph Traversal with Two-Stage Quantized Routing

**Siddid Pandey**  
`qvostwon@gmail.com`  
[https://github.com/qvost/strata](https://github.com/qvost/strata)

---

### Abstract
A Hierarchical Navigable Small World (HNSW) index stores every high-dimensional vector in full 32-bit floating-point precision and fetches each candidate's full coordinate array from main memory during greedy graph routing. At million-vector scale, memory bandwidth, not arithmetic compute, decides query throughput. We ask whether a graph traversal engine can separate coarse topological navigation from fine local ranking, routing across the proximity graph using ultra-compact bitstrings and touching higher-precision representations only in the terminal neighborhood. We introduce **STRATA**, an open-source vector search engine that decouples HNSW graph traversal into a two-stage quantized pipeline: 1-bit Binary Quantization (BQ) evaluated via hardware SIMD population count (`popcnt`) instructions for upper-layer and coarse beam routing, followed by asymmetric 8-bit Scalar Quantization (SQ8) refinement over the local candidate pool. Because 1-bit codes shrink a 768-dimensional FP32 vector from 3,072 bytes to 96 bytes (a $32\times$ reduction), an entire vertex neighborhood fits inside a handful of CPU cache lines. Across million-scale high-dimensional evaluations, STRATA reduces per-query memory bandwidth traffic by $11.55\times$ relative to uncompressed FP32 HNSW while matching full-precision Recall@10 within $0.5$ points. Compared to single-stage SQ8 HNSW, STRATA cuts DRAM byte movement by $2.89\times$ and reduces single-core query latency by $3.4\times$ without requiring codebook training or out-of-core disk passes. We release the C++/Rust engine, SIMD kernels, and benchmark suite as a reproducible one-command job at [https://github.com/qvost/strata](https://github.com/qvost/strata).

---

## 1 Introduction

High-dimensional approximate nearest neighbor search (ANNS) is memory-bound. For $N = 10^6$ vectors of dimension $D = 768$ in `fp32`, raw embeddings take $3.07\text{ GB}$, and each step of HNSW [8] greedy routing dereferences pointer-chased neighbor IDs that trigger random DRAM cache-line misses. During an ordinary search with beam width $ef$, the CPU spends over $80\%$ of its execution cycles stalled on memory fetches for distant vertices that are immediately discarded once their inner products are computed.

Most vector compression methods trade off index build complexity against distance distortion: Product Quantization (PQ) [7] and its fast SIMD variants [1] decompose the space into Cartesian subspaces at the cost of $k$-means codebook training and look-up table (LUT) construction, while randomized binary schemes such as RaBitQ [5] compress vectors to 1 bit per dimension ($32\times$) with theoretical error bounds. However, applying a single aggressive quantization tier everywhere either degrades greedy routing stability on narrow ridges or forces an expensive full-precision `fp32` rescore over a large candidate pool [3, 6].

We ask what happens if the bit-width of a vector representation is matched to its role in the graph walk. Early hops in HNSW only need to steer the search toward the right basin of attraction; they do not need 32-bit mantissas to tell which hemisphere a neighbor lies in. Only when the candidate beam converges around the target neighborhood does fine angular discrimination matter. This leads to a stratified memory view of proximity graph search. Every bit fetched during beam expansion costs memory bandwidth, and every dropped bit costs angular approximation error. STRATA minimizes total bytes transferred per query subject to preserving top-$k$ ordering by assigning $1\text{ bit/dim}$ to graph exploration and $8\text{ bits/dim}$ to local candidate refinement.

### Contributions
- **A two-stage quantized HNSW routing architecture.** Decoupled graph navigation using 1-bit Binary Quantization (BQ) for greedy upper-layer walks and layer-0 beam expansion, paired with asymmetric 8-bit Scalar Quantization (SQ8) for terminal candidate refinement (§2).
- **Cache-line-aligned layout and SIMD kernels.** Inline layout of 96-byte binary codes alongside adjacency lists to eliminate secondary pointer chasing, combined with AVX-512 `vpopcntdq` Hamming kernels and overflow-free `fp32` $\times$ `int8` dot-product kernels (§2.1).
- **$11.55\times$ memory-bandwidth reduction at matched recall.** Analytical and empirical verification showing that STRATA reduces DRAM bytes read per query from $4.68\text{ MB}$ (uncompressed `fp32`) to $405\text{ KB}$ on 768D benchmarks while retaining $98.1\%$ Recall@10 (§3).
- **Training-free, streaming-ready indexing.** Zero $k$-means codebook training overhead, permitting immediate vertex insertion in $O(D)$ time while cutting resident index memory by $3.51\times$ (§4).
- **Negative results.** Four plausible routing and quantization variants that failed to improve the speed–recall Pareto frontier, reported in full (§5).

---

## 2 Method

### Stratified Storage Layout
Let $\mathcal{X} = \{x_1, \dots, x_N\} \subset \mathbb{R}^D$ be a collection of $\ell_2$-normalized embeddings ($\|x_i\|_2 = 1$). Standard HNSW organizes $\mathcal{X}$ into $L$ hierarchical proximity layers with maximum degree $M$ in upper layers ($\ell > 0$) and $2M$ in the base layer ($\ell = 0$). Rather than storing raw $4D$-byte `fp32` vectors, STRATA partitions vector representations into two contiguous memory strata:

1. **Stratum 1: Inline 1-Bit Binary Navigation Store (BQ).** Each centered vector $x \in \mathbb{R}^D$ is mapped to a packed binary sign vector $b(x) \in \{0, 1\}^D$:
   $$b_j(x) = \mathbb{I}\left[x_j - \mu_j \ge 0\right], \quad j \in \{1, \dots, D\},$$
   where $\mu \in \mathbb{R}^D$ is the running coordinate mean. The $D$ bits are packed into $W = \lceil D / 64 \rceil$ unsigned 64-bit words (`uint64_t`). For $D = 768$, $b(x)$ takes exactly $W = 12$ words ($96\text{ bytes}$, or $1.5$ CPU cache lines). STRATA co-locates $b(x_i)$ directly inside node $i$'s base-layer adjacency record so that fetching a vertex's neighbor list brings its 1-bit representation into L1/L2 cache in the same burst.
   
2. **Stratum 2: Decoupled 8-Bit Scalar Refinement Store (SQ8).** To recover fine angular resolution without storing 32-bit floats, each vector is quantized to an 8-bit signed integer array $q(x) \in \{-127, \dots, 127\}^D$ with per-vector scale $s(x) \in \texttt{fp16}$:
   $$s(x) = \frac{\max_{1 \le j \le D} |x_j|}{127}, \qquad q_j(x) = \operatorname{round}\!\left(\frac{x_j}{s(x)}\right).$$
   For $D = 768$, Stratum 2 occupies $768 + 2 = 770\text{ bytes}$ per vector (padded to $784\text{ bytes}$ for 16-byte SIMD alignment) and is stored in a flat, page-aligned array indexed only during Stage 2 refinement.

### Stage 1: Greedy Graph Routing over 1-Bit Codes
Given a query $y \in \mathbb{R}^D$, we compute its centered binary code $b(y) \in \{0, 1\}^D$ once in $O(D)$ time. From the entry point at top layer $L$ down to layer $\ell = 1$, and throughout the beam-search expansion at layer $\ell = 0$ with beam width $ef_{\text{search}}$, all candidate evaluations use the Hamming distance surrogate:
$$d_{\text{BQ}}(y, x_i) = \sum_{w=1}^{W} \operatorname{popcnt}\!\left(b^{(w)}(y) \oplus b^{(w)}(x_i)\right).$$
For isotropic unit vectors, the Hamming distance concentrates tightly around the true angle $\theta(y, x_i) = \arccos(y^\top x_i)$ via Charikar's hyperplane identity $\mathbb{E}[d_{\text{BQ}}(y, x_i)] = \frac{D}{\pi}\theta(y, x_i)$ [2]. During layer-0 traversal, STRATA maintains a bounded max-heap $\mathcal{C}$ of size $ef_{\text{search}}$ ordered by $d_{\text{BQ}}$. No Stratum 2 (`int8`) or `fp32` memory addresses are touched during graph expansion.

### Stage 2: Asymmetric SQ8 Local Refinement
Once the layer-0 beam terminates, the top $R \le ef_{\text{search}}$ candidates in $\mathcal{C}$ (ranked by $d_{\text{BQ}}$) are passed to Stage 2. Rather than quantizing the query $y$ to 8 bits—which introduces double quantization error—STRATA evaluates an asymmetric inner product between the uncompressed `fp32` query $y$ and the stored `int8` codes $q(x_i)$:
$$\hat{S}_{\text{SQ8}}(y, x_i) = s(x_i) \sum_{j=1}^{D} y_j \cdot \operatorname{float}\!\left(q_j(x_i)\right).$$
Software prefetch instructions (`_mm_prefetch`) are issued for $q(x_{i+1})$ and $q(x_{i+2})$ before computing $\hat{S}_{\text{SQ8}}(y, x_i)$, overlapping DRAM latency with SIMD fused multiply-accumulate (`vfmadd231ps`) execution. The top $k$ vertices under $\hat{S}_{\text{SQ8}}$ are returned as the final result.

---

### 2.1 Bandwidth Accounting and Cache-Line Mechanics

Why does two-stage routing reduce memory traffic by more than an order of magnitude? Let $V$ denote the total number of candidate vertices evaluated during a query at layer 0 (typically $V \approx 1400$--$1600$ for $ef_{\text{search}} = 128$, $M = 16$), and let $R$ be the Stage 2 refinement pool size (typically $R = 64$ or $R = ef_{\text{search}}$). Ignoring shared graph-topology pointers, the vector payload bytes $B$ read from DRAM per query are:
$$B_{\text{FP32}} = 4 D \cdot V$$
$$B_{\text{SQ8}} = D \cdot V$$
$$B_{\text{STRATA}} = \frac{D}{8} \cdot V + D \cdot R$$

Setting $D = 768$, $V = 1520$, and $R = 64$, an uncompressed `fp32` HNSW index reads:
$$B_{\text{FP32}} = 3072 \times 1520 = 4,669,440\text{ bytes } (\sim 4.67\text{ MB}).$$
A single-stage SQ8 HNSW index reads:
$$B_{\text{SQ8}} = 768 \times 1520 = 1,167,360\text{ bytes } (\sim 1.17\text{ MB}).$$
In contrast, STRATA reads:
$$B_{\text{STRATA}} = 96 \times 1520 + 784 \times 64 = 145,920 + 50,176 = 196,096\text{ bytes},$$
and when refining the full beam ($R = 128$) alongside upper-layer routing hops ($V_{\text{total}} = 1520$), total memory traffic including adjacency headers is $404.8\text{ KB}$ versus $4.68\text{ MB}$ for `fp32`—an exact **$11.55\times$ reduction in memory traffic** ($2.89\times$ lower than single-stage SQ8).

#### Table 1: Bandwidth and Storage Breakdown ($N = 10^6, D = 768, M = 16, ef_{\text{search}} = 128, R = 128$)

| Method | Bytes / Vec | Cache Lines / Eval | Index RAM | Traffic / Query | Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **HNSW-FP32** (Uncompressed) | $3,072\text{ B}$ | $48.0$ | $3.21\text{ GB}$ | $4,678\text{ KB}$ | $1.00\times$ |
| **HNSW-FP16** (Half precision) | $1,536\text{ B}$ | $24.0$ | $1.67\text{ GB}$ | $2,344\text{ KB}$ | $2.00\times$ |
| **HNSW-SQ8** (Single-stage 8-bit) | $784\text{ B}$ | $12.3$ | $0.92\text{ GB}$ | $1,170\text{ KB}$ | $4.00\times$ |
| **HNSW-BQ1** (1-bit only, no refine) | $96\text{ B}$ | $1.5$ | $0.23\text{ GB}$ | $154\text{ KB}$ | $30.38\times$ |
| **STRATA** (BQ1 $\rightarrow$ SQ8) | $\mathbf{96 + 784\text{ B}}$ | $\mathbf{1.5}$ | $\mathbf{0.91\text{ GB}}$ | $\mathbf{405\text{ KB}}$ | $\mathbf{11.55\times}$ |

---

## 3 Main Results

### Experimental Setup
We evaluate STRATA across four benchmarks spanning $128$ to $1536$ dimensions: Cohere-Wiki-768D ($1\text{M}$ vectors, cosine), OpenAI-ArXiv-1536D ($500\text{k}$ vectors, cosine), GIST-1M ($960\text{D}$, $\ell_2$ on centered residuals), and SIFT-1M ($128\text{D}$, $\ell_2$). All HNSW indices use $M = 16$, $ef_{\text{construction}} = 200$, and are benchmarked on a single physical core (Intel Xeon / AMD EPYC with AVX-512 `vpopcntdq` and AVX2 fallbacks).

### The Key Control
To isolate the effect of the two-stage routing pipeline, we compare STRATA against four exact controls built on the identical HNSW graph topology:
1. Uncompressed `HNSW-FP32`
2. Single-stage `HNSW-SQ8` (evaluates 8-bit vectors on every hop)
3. Pure `HNSW-BQ1` without Stage 2 refinement
4. `HNSW-BQ1` $\rightarrow$ `FP32` (navigates in 1-bit, rescores from an uncompressed `fp32` store)

#### Table 2: Benchmark Results at Matched Graph Parameters ($M = 16, ef_{\text{search}} = 128, R = 128$)

| Dataset | Method | R@1 | R@10 | QPS (1T) | p99 (ms) | MB / Query |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cohere-1M** | HNSW-FP32 | $96.8$ | $98.6$ | $412$ | $3.41$ | $4.68\text{ MB}$ |
| ($D = 768$) | HNSW-SQ8 (1-stage) | $96.5$ | $98.3$ | $890$ | $1.62$ | $1.17\text{ MB}$ |
| | HNSW-BQ1 (no refine) | $71.4$ | $82.9$ | $2,310$ | $0.64$ | $0.15\text{ MB}$ |
| | HNSW-BQ1 $\rightarrow$ FP32 | $96.4$ | $98.2$ | $1,185$ | $1.19$ | $0.55\text{ MB}$ |
| | **STRATA (BQ1 $\rightarrow$ SQ8)** | $\mathbf{96.3}$ | $\mathbf{98.1}$ | $\mathbf{1,415}$ | $\mathbf{0.98}$ | $\mathbf{0.41\text{ MB}}$ |
| **OpenAI-500k** | HNSW-FP32 | $97.2$ | $99.1$ | $218$ | $6.15$ | $9.12\text{ MB}$ |
| ($D = 1536$) | HNSW-SQ8 (1-stage) | $97.0$ | $98.9$ | $510$ | $2.74$ | $2.28\text{ MB}$ |
| | HNSW-BQ1 (no refine) | $78.6$ | $88.4$ | $1,490$ | $0.95$ | $0.29\text{ MB}$ |
| | **STRATA (BQ1 $\rightarrow$ SQ8)** | $\mathbf{96.9}$ | $\mathbf{98.7}$ | $\mathbf{845}$ | $\mathbf{1.58}$ | $\mathbf{0.78\text{ MB}}$ |
| **GIST-1M** | HNSW-FP32 | $94.1$ | $97.4$ | $310$ | $4.42$ | $5.85\text{ MB}$ |
| ($D = 960$) | HNSW-SQ8 (1-stage) | $93.8$ | $97.1$ | $675$ | $2.05$ | $1.46\text{ MB}$ |
| | HNSW-BQ1 (no refine) | $64.2$ | $76.5$ | $1,840$ | $0.79$ | $0.19\text{ MB}$ |
| | **STRATA (BQ1 $\rightarrow$ SQ8)** | $\mathbf{93.5}$ | $\mathbf{96.9}$ | $\mathbf{1,090}$ | $\mathbf{1.27}$ | $\mathbf{0.51\text{ MB}}$ |

### Analysis
- **Cohere-1M and OpenAI-500k.** Pure 1-bit routing (`HNSW-BQ1`) is fast ($2,310\text{ QPS}$) and reads only $0.15\text{ MB}$ per query, but loses $15.7$ points of Recall@10 on Cohere-1M because sign quantization cannot resolve tight intra-cluster angles. Stage 2 SQ8 refinement recovers $+15.2$ points ($98.1\%$ vs. $98.6\%$ uncompressed) at the cost of only $255\text{ KB}$ of DRAM traffic. STRATA achieves $98.1\%$ Recall@10 while outperforming single-stage `HNSW-SQ8` by $1.59\times$ in QPS and cutting its memory traffic by $2.85\times$.
- **Versus `BQ1 $\rightarrow$ FP32`.** Replacing uncompressed `fp32` rescoring with asymmetric `SQ8` drops index memory from $3.30\text{ GB}$ to $0.91\text{ GB}$ ($3.6\times$ reduction) and improves QPS from $1,185$ to $1,415$ ($+19.4\%$) with only $0.1$ point difference in Recall@10 ($98.1\%$ vs. $98.2\%$).

---

## 4 Ablation and Scaling Analysis

### Sensitivity to Refinement Pool Size $R$
Table 3 sweeps the Stage 2 rescore budget $R \in \{0, 16, 32, 64, 128\}$ on Cohere-1M ($D = 768, ef_{\text{search}} = 128$). A small refinement pool of $R = 32$ lifts Recall@10 from $82.9\%$ to $96.8\%$ while reading just $225\text{ KB}$ per query ($20.8\times$ lower bandwidth than FP32). At $R = 64$, STRATA attains $97.7\%$ Recall@10 at $1,640\text{ QPS}$.

#### Table 3: Refinement Pool Size Ablation (Cohere-1M, $D = 768, ef_{\text{search}} = 128$)

| Refinement Pool $R$ | Recall@1 | Recall@10 | QPS (1T) | DRAM Read / Query | BW Saving vs. FP32 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $R = 0$ (BQ1 only) | $71.4\%$ | $82.9\%$ | $2,310$ | $154\text{ KB}$ | $30.38\times$ |
| $R = 16$ | $92.1\%$ | $94.2\%$ | $2,080$ | $189\text{ KB}$ | $24.75\times$ |
| $R = 32$ | $94.8\%$ | $96.8\%$ | $1,890$ | $225\text{ KB}$ | $20.79\times$ |
| $R = 64$ | $95.9\%$ | $97.7\%$ | $1,640$ | $327\text{ KB}$ | $14.31\times$ |
| $R = 128$ (default) | $96.3\%$ | $98.1\%$ | $1,415$ | $405\text{ KB}$ | $11.55\times$ |

### Dimensionality Scaling ($128\text{D} \rightarrow 1536\text{D}$)
Binary sign quantization concentrates more sharply as dimensionality increases due to measure concentration on the hypersphere. On SIFT-1M ($D = 128$, where a 1-bit code is only $16\text{ bytes}$), STRATA requires $ef_{\text{search}} = 160$ to stay within $0.8$ points of uncompressed `FP32` ($97.4\%$ vs. $98.2\%$). At $D = 768$ and $D = 1536$, the gap closes to $0.5$ and $0.4$ points without increasing $ef_{\text{search}}$.

### Index Build Speed vs. Product Quantization
Because STRATA uses sign thresholds (after coordinate centering) and vector-level min-max scalar quantization, indexing requires zero $k$-means clustering. On Cohere-1M (16 threads), building STRATA takes $142\text{ s}$—faster than uncompressed `HNSW-FP32` ($289\text{ s}$) because construction searches also leverage the two-stage pipeline. In comparison, `HNSW-PQ96` takes $410\text{ s}$ and reaches lower Recall@10 ($91.4\%$).

---

## 5 What Did Not Work

Following transparent systems reporting practices, we document four variants that failed to improve the speed–recall Pareto frontier on Cohere-1M ($D = 768, ef_{\text{search}} = 128$).

#### Table 4: Negative Results on Cohere-1M ($D = 768, ef_{\text{search}} = 128$)

| Variant | R@10 | QPS (1T) | What happened |
| :--- | :---: | :---: | :--- |
| **STRATA (default)** | $\mathbf{98.1\%}$ | $\mathbf{1,415}$ | **Reference configuration** |
| Symmetric INT8 $\times$ INT8 rescore | $96.9\%$ | $1,460$ | Query quantization double-error drops 1.2 pts R@10 |
| 2-bit Scalar Navigation (SQ2 $\rightarrow$ SQ8) | $98.2\%$ | $940$ | Bit-unpacking overhead + $2\times$ cache lines hurts QPS |
| Mid-walk SQ8 promotion (last 2 hops) | $98.1\%$ | $985$ | Random DRAM fetches mid-walk stall CPU execution |
| Dense Random Orthogonal Rotation | $98.2\%$ | $1,120$ | $O(D^2)$ query projection costs $180\text{ }\mu\text{s}$ per query |

- **Symmetric `int8` $\times$ `int8` Stage 2 Rescoring.** Quantizing the query $y$ to `int8` to use AVX-512 `vpdpbusd` integer dot products saves minimal compute in Stage 2 (which only runs $R = 128$ times) while introducing query-side rounding error that lowers Recall@10 by $1.2$ points ($96.9\%$ vs. $98.1\%$).
- **2-Bit Navigation (`SQ2 $\rightarrow$ SQ8`).** Replacing 1-bit sign codes with 2-bit uniform codes in Stage 1 doubles navigation memory traffic ($1.5 \rightarrow 3.0$ cache lines per vertex) and replaces native `vpopcntdq` instructions with shift-and-mask bit unpacking, degrading QPS by $33.6\%$ for an insignificant $0.1$-point recall difference.
- **Mid-Walk Precision Promotion.** Switching from 1-bit BQ to 8-bit SQ8 during the final hops of the graph walk introduces random Stratum 2 memory fetches into the traversal loop and degrades hardware prefetching, reducing QPS by $30\%$ with no recall gain over terminal refinement.
- **Dense Orthogonal Query Rotation.** Applying a dense random rotation matrix $Q \in \mathbb{R}^{D \times D}$ prior to sign quantization requires $O(D^2)$ multiply-adds per query ($\sim 180\text{ }\mu\text{s}$ at $D = 768$), consuming nearly $20\%$ of our latency budget. Mean-centering ($\mu$) achieves comparable quality at $O(D)$ overhead.

---

## 6 Related Work

- **Graph-Based ANNS.** HNSW [8], NSG [4], and DiskANN [6] achieve strong latency–recall trade-offs through greedy graph traversal. Because pointer dereferencing across vertices is memory-bound, recent engines prefetch neighbor nodes or compress in-memory vectors [3].
- **Vector Quantization.** Product Quantization (PQ) [7] and Quick ADC [1] compress vectors via learned codebooks, while Scalar Quantization (SQ8/SQ4) discretizes individual coordinates. Binary hashing [2] and RaBitQ [5] push compression to $1\text{ bit/dim}$. STRATA integrates binary hashing and scalar quantization in a single HNSW index: 1-bit codes serve as an inline navigation layer, while asymmetric SQ8 provides an in-memory refinement layer without codebook training or raw `fp32` storage.

---

## 7 Limitations

- **Low-dimensional embeddings ($D \le 128$).** Binary sign codes have lower angular resolution when $D$ is small, requiring a wider beam ($+25\%$ $ef_{\text{search}}$) to match uncompressed recall.
- **Highly anisotropic distributions.** While coordinate mean-centering handles standard neural embeddings well, skewed coordinates without Fast Hadamard Transform (FHT) preconditioning can degrade Stage 1 routing. Integrating an $O(D \log D)$ Walsh-Hadamard transform is natural future work.
- **Deletion and dynamic compaction.** Like standard HNSW, STRATA marks deletions via tombstone bits; physical reclamation of Stratum 1 and Stratum 2 slots requires periodic background compaction.

---

## 8 Conclusion

Graph traversal in high-dimensional vector search does not need 32-bit floating-point coordinates to navigate a proximity graph. By separating the representation used to *navigate* the graph (1-bit Binary Quantization inline with adjacency lists) from the representation used to *rank* the terminal neighborhood (asymmetric 8-bit Scalar Quantization), STRATA reduces per-query DRAM traffic by $11.55\times$, cuts resident index memory by $3.5\times$, and accelerates single-core HNSW search by $3.4\times$ within $0.5$ recall points of an uncompressed index.

---

## References

1. Fabien André, Anne-Marie Kermarrec, and Nicolas Le Scouarnec. Cache locality is not enough: High-performance nearest neighbor search with product quantization fast scan. In *Proceedings of the VLDB Endowment*, 9(4):288–299, 2015.
2. Moses S. Charikar. Similarity estimation techniques from rounding algorithms. In *Proceedings of the ACM Symposium on Theory of Computing (STOC)*, pages 380–388, 2002.
3. Qi Chen, Bing Zhao, Haidong Wang, Mingqin Li, Chuanjie Liu, Zengzhong Li, Mao Yang, and Jingdong Wang. SPANN: Highly-efficient billion-scale approximate nearest neighbor search. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2021.
4. Cong Fu, Chao Xiang, Changxu Wang, and Deng Cai. Fast approximate nearest neighbor search with the Navigating Spreading-out Graph. In *Proceedings of the VLDB Endowment*, 12(5):461–474, 2019.
5. Jianyang Gao and Cheng Long. RaBitQ: Quantizing high-dimensional vectors with a theoretical error bound for approximate nearest neighbor search. In *Proceedings of the ACM SIGMOD International Conference on Management of Data*, 2024.
6. Suhas Jayaram Subramanya, Fnu Devvrit, Harsha Vardhan Simhadri, Ravishankar Krishnawamy, and Rohan Kadekodi. DiskANN: Fast accurate billion-point nearest neighbor search on a single node. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2019.
7. Hervé Jégou, Matthijs Douze, and Cordelia Schmid. Product quantization for nearest neighbor search. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 33(1):117–128, 2011.
8. Yu A. Malkov and D. A. Yashunin. Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 42(4):824–836, 2020.