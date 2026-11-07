#!/usr/bin/env python3
"""Two-Stage Traversal HNSW benchmark.

Configurations
--------------
A. Baseline uncompressed FP32 HNSW.
B. Uniform SQ8 HNSW (graph built on dequantised SQ8 payload, FP32 query =
   asymmetric distance computation).
C. Uniform BQ HNSW (graph built on +/-1 sign codes, binarised query; L2 on
   sign codes is proportional to Hamming distance).
D. Two-stage hybrid: BQ greedy upper layers + BQ layer-0 frontier with a
   fixed ``ef_bq=64`` buffer, then SQ8/FP32 reranking and fine expansion.

For every configuration the harness sweeps
``efSearch in [10, 20, 40, 80, 160, 320]`` and records Recall@10, p50/p95
latency, QPS and LLC (L3) cache misses.

Example
-------
    python bench/runner.py --subset 200000 --queries 2000
    python bench/runner.py --configs A,D --rerank fp32
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.perf_wrapper import (  # noqa: E402
    DEFAULT_EVENTS,
    PerfCounter,
    perf_available,
    summarize,
)
from src.bench_dataset import Dataset, load_sift1m, make_synthetic  # noqa: E402
from src.hnsw import FAISSGraph, HNSWIndex  # noqa: E402
from src.hybrid import HybridSearcher  # noqa: E402
from src.quantization import BQCodec, SQ8Codec  # noqa: E402

CONFIG_ORDER = ("A", "B", "C", "D")
CONFIG_NAMES = {
    "A": "Baseline FP32 HNSW",
    "B": "Uniform SQ8 HNSW",
    "C": "Uniform BQ HNSW",
    "D": "Two-Stage Hybrid (BQ + rerank)",
}
DEFAULT_EF_SEARCH = "10,20,40,80,160,320"


# ---------------------------------------------------------------------------
# Result row
# ---------------------------------------------------------------------------
@dataclass
class BenchmarkRow:
    timestamp: str
    config: str
    config_name: str
    ef_search: int
    ef_bq: int
    rerank: str
    k: int
    n_base: int
    n_queries: int
    dim: int
    M: int
    ef_construction: int
    recall_at_k: float
    p50_ms: float
    p95_ms: float
    mean_ms: float
    qps: float
    llc_loads: Optional[int]
    llc_load_misses: Optional[int]
    llc_miss_rate: Optional[float]
    llc_misses_per_query: Optional[float]
    cycles: Optional[int]
    instructions: Optional[int]
    ipc: Optional[float]
    graph_mb: float
    payload_mb: float
    index_mb: float
    status: str
    notes: str

    def csv_dict(self) -> dict:
        return asdict(self)


CSV_FIELDS = list(BenchmarkRow.__dataclass_fields__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage BQ/SQ8 HNSW benchmark on SIFT1M",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=str, default=None, help="SIFT1M HDF5 path")
    parser.add_argument(
        "--data-dir", type=str, default=str(ROOT / "data"), help="dataset directory"
    )
    parser.add_argument(
        "--subset", type=int, default=0, help="use only the first N base vectors"
    )
    parser.add_argument(
        "--queries", type=int, default=0, help="use only the first N queries"
    )
    parser.add_argument("--gt-k", type=int, default=100, help="ground-truth depth")
    parser.add_argument("--k", type=int, default=10, help="top-k for recall")
    parser.add_argument(
        "--ef-search",
        type=str,
        default=DEFAULT_EF_SEARCH,
        help="comma-separated efSearch sweep",
    )
    parser.add_argument("--M", type=int, default=32, help="HNSW edge degree")
    parser.add_argument(
        "--ef-construction", type=int, default=200, help="HNSW construction depth"
    )
    parser.add_argument(
        "--ef-bq", type=int, default=64, help="fixed BQ layer-0 candidate buffer"
    )
    parser.add_argument(
        "--rerank",
        choices=("sq8", "fp32"),
        default="sq8",
        help="payload used by the hybrid fine stage",
    )
    parser.add_argument(
        "--hybrid-backend",
        choices=("auto", "fast", "python"),
        default="auto",
        help="compiled C++ traversal, pure-Python reference, or auto",
    )
    parser.add_argument(
        "--bq-threshold",
        choices=("median", "mean", "zero"),
        default="median",
        help="BQ threshold; median is required for non-negative SIFT data",
    )
    parser.add_argument(
        "--configs",
        type=str,
        default="A,B,C,D",
        help="comma-separated subset of A,B,C,D",
    )
    parser.add_argument("--warmup", type=int, default=50, help="warm-up queries")
    parser.add_argument("--threads", type=int, default=1, help="FAISS threads")
    parser.add_argument(
        "--output",
        type=str,
        default="results/benchmark_summary.csv",
        help="CSV output path",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        default="results/benchmark_summary.md",
        help="Markdown output path",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="results/benchmark_metadata.json",
        help="run metadata output path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-download", action="store_true", help="fail if the dataset is missing"
    )
    parser.add_argument(
        "--synthetic",
        type=int,
        default=0,
        help="use a synthetic clustered workload with N base vectors",
    )
    parser.add_argument(
        "--recompute-gt",
        action="store_true",
        help="recompute exact ground truth even for the full dataset",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def parse_ef_list(text: str) -> list[int]:
    values = sorted({int(part.strip()) for part in text.split(",") if part.strip()})
    if not values:
        raise ValueError("efSearch sweep is empty")
    if any(value <= 0 for value in values):
        raise ValueError("efSearch values must be positive")
    return values


def resolve_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (ROOT / candidate)


def human_mb(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def fmt(value: Optional[float], digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:.{digits}f}"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def recall_at_k(predicted: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    if predicted.shape[0] == 0:
        return 0.0
    hits = 0
    for row, truth in zip(predicted[:, :k], ground_truth[:, :k]):
        valid = row[row >= 0]
        hits += len(set(valid.tolist()) & set(truth.tolist()))
    return hits / float(predicted.shape[0] * k)


def latencies_ms(samples_ns: list[int]) -> tuple[float, float, float]:
    arr = np.asarray(samples_ns, dtype=np.float64) / 1e6
    return (
        float(np.percentile(arr, 50)),
        float(np.percentile(arr, 95)),
        float(arr.mean()),
    )


def make_row(
    *,
    config: str,
    ef_search: int,
    args: argparse.Namespace,
    dataset: Dataset,
    predicted: np.ndarray,
    samples_ns: list[int],
    total_seconds: float,
    counters: dict[str, int],
    graph_bytes: int,
    payload_bytes: int,
    index_bytes: int,
    notes: str = "",
    status: str = "ok",
) -> BenchmarkRow:
    p50, p95, mean = latencies_ms(samples_ns)
    n_queries = predicted.shape[0]
    derived = summarize(counters)
    llc_misses = counters.get("llc_load_misses")
    row = BenchmarkRow(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        config=config,
        config_name=CONFIG_NAMES[config],
        ef_search=int(ef_search),
        ef_bq=int(args.ef_bq),
        rerank=args.rerank if config == "D" else "n/a",
        k=int(args.k),
        n_base=int(dataset.n_base),
        n_queries=int(n_queries),
        dim=int(dataset.dim),
        M=int(args.M),
        ef_construction=int(args.ef_construction),
        recall_at_k=recall_at_k(predicted, dataset.ground_truth, args.k),
        p50_ms=p50,
        p95_ms=p95,
        mean_ms=mean,
        qps=(n_queries / total_seconds) if total_seconds > 0 else float("nan"),
        llc_loads=counters.get("llc_loads"),
        llc_load_misses=llc_misses,
        llc_miss_rate=derived.get("llc_miss_rate"),
        llc_misses_per_query=(
            llc_misses / n_queries if llc_misses is not None and n_queries else None
        ),
        cycles=counters.get("cycles"),
        instructions=counters.get("instructions"),
        ipc=derived.get("ipc"),
        graph_mb=graph_bytes / 1e6,
        payload_mb=payload_bytes / 1e6,
        index_mb=index_bytes / 1e6,
        status=status,
        notes=notes,
    )
    return row


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------
def evaluate_faiss(
    config: str,
    index: HNSWIndex,
    search_queries: np.ndarray,
    args: argparse.Namespace,
    dataset: Dataset,
    ef_values: list[int],
    *,
    graph_bytes: int,
    payload_bytes: int,
    index_bytes: int,
    notes: str,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    n_queries = search_queries.shape[0]
    warmup = min(max(args.warmup, 0), n_queries)
    for ef in ef_values:
        for q in search_queries[:warmup]:
            index.search_one(q, args.k, ef)

        counter = PerfCounter(events=DEFAULT_EVENTS)
        predicted = np.empty((n_queries, args.k), dtype=np.int64)
        samples_ns: list[int] = []
        counter.start()
        batch_start = time.perf_counter()
        for i in range(n_queries):
            query = search_queries[i]
            t0 = time.perf_counter_ns()
            ids, _ = index.search_one(query, args.k, ef)
            t1 = time.perf_counter_ns()
            predicted[i] = ids
            samples_ns.append(t1 - t0)
        batch_seconds = time.perf_counter() - batch_start
        counter.stop()
        counters = counter.read()
        errors = dict(counter.errors)
        counter.close()

        row = make_row(
            config=config,
            ef_search=ef,
            args=args,
            dataset=dataset,
            predicted=predicted,
            samples_ns=samples_ns,
            total_seconds=batch_seconds,
            counters=counters,
            graph_bytes=graph_bytes,
            payload_bytes=payload_bytes,
            index_bytes=index_bytes,
            notes=notes + (f" perf: {errors}" if errors else ""),
        )
        rows.append(row)
        if not args.quiet:
            print(
                f"  [{config}] ef={ef:3d}  recall@{args.k}={row.recall_at_k:.4f}  "
                f"p50={row.p50_ms:8.3f} ms  p95={row.p95_ms:8.3f} ms  "
                f"qps={row.qps:8.1f}  llc/q={fmt(row.llc_misses_per_query, 1)}",
                flush=True,
            )
    return rows


def evaluate_fast_hybrid(
    searcher,
    args: argparse.Namespace,
    dataset: Dataset,
    ef_values: list[int],
    *,
    graph_bytes: int,
    payload_bytes: int,
    index_bytes: int,
    notes: str,
) -> list[BenchmarkRow]:
    """Evaluate the compiled batch kernel one query at a time.

    Calling the kernel per query keeps the latency methodology identical to
    configs A/B/C (Python call overhead included).  The kernel's internal
    per-query timer is intentionally not used here; it is useful only for
    profiling the kernel in isolation.
    """

    rows: list[BenchmarkRow] = []
    queries = dataset.queries
    n_queries = queries.shape[0]
    warmup = min(max(args.warmup, 0), n_queries)
    for ef in ef_values:
        for query in queries[:warmup]:
            searcher.search_batch(
                query.reshape(1, -1), ef, k=args.k, ef_bq=args.ef_bq
            )

        counter = PerfCounter(events=DEFAULT_EVENTS)
        predicted = np.empty((n_queries, args.k), dtype=np.int64)
        samples_ns: list[int] = []
        counter.start()
        batch_start = time.perf_counter()
        for i in range(n_queries):
            query = queries[i]
            t0 = time.perf_counter_ns()
            ids, _, _ = searcher.search_batch(
                query.reshape(1, -1), ef, k=args.k, ef_bq=args.ef_bq
            )
            t1 = time.perf_counter_ns()
            predicted[i] = ids[0]
            samples_ns.append(t1 - t0)
        batch_seconds = time.perf_counter() - batch_start
        counter.stop()
        counters = counter.read()
        errors = dict(counter.errors)
        counter.close()

        row = make_row(
            config="D",
            ef_search=ef,
            args=args,
            dataset=dataset,
            predicted=predicted,
            samples_ns=samples_ns,
            total_seconds=batch_seconds,
            counters=counters,
            graph_bytes=graph_bytes,
            payload_bytes=payload_bytes,
            index_bytes=index_bytes,
            notes=notes + (f" perf: {errors}" if errors else ""),
        )
        rows.append(row)
        if not args.quiet:
            print(
                f"  [D] ef={ef:3d}  recall@{args.k}={row.recall_at_k:.4f}  "
                f"p50={row.p50_ms:8.3f} ms  p95={row.p95_ms:8.3f} ms  "
                f"qps={row.qps:8.1f}  llc/q={fmt(row.llc_misses_per_query, 1)}",
                flush=True,
            )
    return rows


def evaluate_hybrid(
    hybrid: HybridSearcher,
    args: argparse.Namespace,
    dataset: Dataset,
    ef_values: list[int],
    *,
    graph_bytes: int,
    payload_bytes: int,
    index_bytes: int,
    notes: str,
) -> list[BenchmarkRow]:
    rows: list[BenchmarkRow] = []
    queries = dataset.queries
    n_queries = queries.shape[0]
    warmup = min(max(args.warmup, 0), n_queries)
    for ef in ef_values:
        for query in queries[:warmup]:
            hybrid.search(query, ef, k=args.k, ef_bq=args.ef_bq)

        counter = PerfCounter(events=DEFAULT_EVENTS)
        predicted = np.empty((n_queries, args.k), dtype=np.int64)
        samples_ns: list[int] = []
        counter.start()
        batch_start = time.perf_counter()
        for i in range(n_queries):
            query = queries[i]
            t0 = time.perf_counter_ns()
            ids = hybrid.search(query, ef, k=args.k, ef_bq=args.ef_bq)
            t1 = time.perf_counter_ns()
            predicted[i, : len(ids)] = ids
            if len(ids) < args.k:
                predicted[i, len(ids) :] = -1
            samples_ns.append(t1 - t0)
        batch_seconds = time.perf_counter() - batch_start
        counter.stop()
        counters = counter.read()
        errors = dict(counter.errors)
        counter.close()

        row = make_row(
            config="D",
            ef_search=ef,
            args=args,
            dataset=dataset,
            predicted=predicted,
            samples_ns=samples_ns,
            total_seconds=batch_seconds,
            counters=counters,
            graph_bytes=graph_bytes,
            payload_bytes=payload_bytes,
            index_bytes=index_bytes,
            notes=notes + (f" perf: {errors}" if errors else ""),
        )
        rows.append(row)
        if not args.quiet:
            print(
                f"  [D] ef={ef:3d}  recall@{args.k}={row.recall_at_k:.4f}  "
                f"p50={row.p50_ms:8.3f} ms  p95={row.p95_ms:8.3f} ms  "
                f"qps={row.qps:8.1f}  llc/q={fmt(row.llc_misses_per_query, 1)}",
                flush=True,
            )
    return rows


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_markdown(rows: list[BenchmarkRow], metadata: dict) -> str:
    lines: list[str] = []
    lines.append("## Two-Stage Traversal Benchmark")
    lines.append("")
    lines.append(
        f"**Dataset:** {metadata['dataset']} &nbsp;|&nbsp; "
        f"**base:** {metadata['n_base']:,} &nbsp;|&nbsp; "
        f"**queries:** {metadata['n_queries']:,} &nbsp;|&nbsp; "
        f"**dim:** {metadata['dim']} &nbsp;|&nbsp; "
        f"**k:** {metadata['k']}"
    )
    lines.append("")
    lines.append(
        f"**HNSW:** M={metadata['M']}, efConstruction={metadata['ef_construction']} "
        f"&nbsp;|&nbsp; **BQ buffer:** ef_bq={metadata['ef_bq']} "
        f"&nbsp;|&nbsp; **hybrid rerank:** {metadata['rerank']} "
        f"&nbsp;|&nbsp; **threads:** {metadata['threads']}"
    )
    lines.append("")
    counters_ok = any(row.llc_load_misses is not None for row in rows)
    if not counters_ok:
        lines.append(
            "> Note: hardware counters could not be opened; LLC columns are `n/a`."
        )
        lines.append("")

    lines.append(
        "| efSearch | Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | LLC misses/query | LLC miss rate |"
    )
    lines.append(
        "| ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    order = {name: i for i, name in enumerate(CONFIG_ORDER)}
    for row in sorted(rows, key=lambda r: (r.ef_search, order.get(r.config, 99))):
        lines.append(
            "| {ef} | {name} | {recall:.4f} | {p50:.3f} | {p95:.3f} | {qps:.1f} | "
            "{llc} | {rate} |".format(
                ef=row.ef_search,
                name=f"{row.config} — {row.config_name}",
                recall=row.recall_at_k,
                p50=row.p50_ms,
                p95=row.p95_ms,
                qps=row.qps,
                llc=fmt(row.llc_misses_per_query, 0),
                rate=fmt(row.llc_miss_rate, 4),
            )
        )

    # Best-recall / best-QPS highlights per efSearch.
    lines.append("")
    lines.append("### Per-efSearch leaders")
    lines.append("")
    lines.append("| efSearch | Best Recall@10 | Best QPS | Fastest config |")
    lines.append("| ---: | :--- | :--- | :--- |")
    for ef in sorted({row.ef_search for row in rows}):
        group = [row for row in rows if row.ef_search == ef]
        best_recall = max(group, key=lambda r: r.recall_at_k)
        best_qps = max(group, key=lambda r: r.qps)
        lines.append(
            f"| {ef} | {best_recall.config} ({best_recall.recall_at_k:.4f}) | "
            f"{best_qps.config} ({best_qps.qps:.1f}) | {best_qps.config_name} |"
        )

    lines.append("")
    lines.append(
        f"_Generated {metadata['generated_at']} on {metadata['platform']} "
        f"({metadata['cpu']})._"
    )
    return "\n".join(lines) + "\n"


def print_markdown(rows: list[BenchmarkRow], metadata: dict) -> None:
    print("")
    print(format_markdown(rows, metadata))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def load_workload(args: argparse.Namespace) -> Dataset:
    if args.synthetic > 0:
        n_queries = args.queries or min(500, max(50, args.synthetic // 20))
        print(
            f"[runner] synthetic workload: {args.synthetic:,} base, "
            f"{n_queries:,} queries",
            flush=True,
        )
        return make_synthetic(
            n_base=args.synthetic,
            n_queries=n_queries,
            dim=128,
            gt_k=max(args.gt_k, args.k),
            seed=args.seed,
        )
    print(f"[runner] loading SIFT1M (subset={args.subset or 'all'})", flush=True)
    dataset = load_sift1m(
        args.data,
        data_dir=args.data_dir,
        subset=args.subset or None,
        n_queries=args.queries or None,
        gt_k=args.gt_k,
        download=not args.no_download,
        recompute_gt=args.recompute_gt,
    )
    print(f"[runner] {dataset}", flush=True)
    return dataset


def run(args: argparse.Namespace) -> int:
    import faiss

    faiss.omp_set_num_threads(max(1, int(args.threads)))
    np.random.seed(args.seed)

    ef_values = parse_ef_list(args.ef_search)
    configs = tuple(
        part.strip().upper() for part in args.configs.split(",") if part.strip()
    )
    unknown = [c for c in configs if c not in CONFIG_ORDER]
    if unknown:
        raise ValueError(f"Unknown config(s): {unknown}")
    if not configs:
        raise ValueError("No configurations selected")

    output_path = resolve_path(args.output)
    markdown_path = resolve_path(args.markdown)
    metadata_path = resolve_path(args.metadata)
    for path in (output_path, markdown_path, metadata_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_workload(args)
    if dataset.ground_truth.shape[1] < args.k:
        raise ValueError("ground truth is shallower than k")

    print("[runner] fitting quantizers", flush=True)
    t0 = time.perf_counter()
    sq8 = SQ8Codec.fit(dataset.base)
    bq = BQCodec.fit(dataset.base, mode=args.bq_threshold)
    sq8_codes = sq8.encode(dataset.base)
    bq_codes = bq.encode(dataset.base)
    print(
        f"[runner] SQ8 {sq8_codes.nbytes / 1e6:.1f} MB, "
        f"BQ {bq_codes.nbytes / 1e6:.1f} MB "
        f"({bq.n_words} uint64 words/vector) in "
        f"{time.perf_counter() - t0:.1f}s",
        flush=True,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset.name,
        "dataset_path": dataset.path,
        "n_base": dataset.n_base,
        "n_queries": dataset.n_queries,
        "dim": dataset.dim,
        "k": args.k,
        "M": args.M,
        "ef_construction": args.ef_construction,
        "ef_search": ef_values,
        "ef_bq": args.ef_bq,
        "rerank": args.rerank,
        "hybrid_backend": args.hybrid_backend,
        "bq_threshold": args.bq_threshold,
        "configs": list(configs),
        "threads": args.threads,
        "platform": platform.platform(),
        "cpu": cpu_model(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "faiss": getattr(faiss, "__version__", "unknown"),
        "perf_available": perf_available(),
        "notes": (
            "Single-threaded per-query latency. LLC counters are user-space "
            "hardware events (perf_event_open, exclude_kernel=1)."
        ),
    }

    rows: list[BenchmarkRow] = []
    failures: list[str] = []
    hybrid_backend_used = "n/a"

    with output_path.open("w", newline="", encoding="utf-8") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        csv_handle.flush()

        def record(new_rows: list[BenchmarkRow]) -> None:
            rows.extend(new_rows)
            for row in new_rows:
                writer.writerow(row.csv_dict())
            csv_handle.flush()

        need_graph = "A" in configs or "D" in configs
        graph: Optional[FAISSGraph] = None
        index_a: Optional[HNSWIndex] = None

        # -------------------------------------------------------------- A / D
        if need_graph:
            try:
                cache_path = Path(args.data_dir) / f"hnsw_M{args.M}_ef{args.ef_construction}_{dataset.n_base}.faiss"
                t_build = time.perf_counter()
                if cache_path.exists():
                    print(f"[runner] loading cached FP32 HNSW from {cache_path.name}", flush=True)
                    index_a = HNSWIndex.load(cache_path)
                else:
                    print(
                        f"[runner] building FP32 HNSW (M={args.M}, "
                        f"efConstruction={args.ef_construction})",
                        flush=True,
                    )
                    index_a = HNSWIndex(
                        dataset.base,
                        M=args.M,
                        ef_construction=args.ef_construction,
                        verbose=True,
                    )
                    if dataset.n_base >= 100000:
                        try:
                            index_a.save(cache_path)
                            print(f"[runner] saved index cache to {cache_path.name}", flush=True)
                        except Exception as e:
                            print(f"[runner] warning: failed to cache index: {e}", flush=True)
                graph = index_a.graph()
                build_s = time.perf_counter() - t_build
                graph_bytes = graph.graph_bytes()
                print(
                    f"[runner] FP32 index ready in {build_s:.1f}s "
                    f"(graph {graph_bytes / 1e6:.1f} MB)",
                    flush=True,
                )

                if "A" in configs:
                    print("[runner] evaluating A: baseline FP32 HNSW", flush=True)
                    record(
                        evaluate_faiss(
                            "A",
                            index_a,
                            dataset.queries,
                            args,
                            dataset,
                            ef_values,
                            graph_bytes=graph_bytes,
                            payload_bytes=dataset.base.nbytes,
                            index_bytes=index_a.storage_bytes(),
                            notes=(
                                "FP32 payload; FAISS HNSW search; "
                                "single-threaded per-query timing"
                            ),
                        )
                    )

                if "D" in configs:
                    payload_bytes = bq_codes.nbytes + (
                        sq8_codes.nbytes
                        if args.rerank == "sq8"
                        else dataset.base.nbytes
                    )
                    searcher = None
                    backend = args.hybrid_backend
                    if backend in ("auto", "fast"):
                        try:
                            from src.fast_hybrid import FastHybridSearcher

                            searcher = FastHybridSearcher(
                                graph,
                                bq,
                                sq8,
                                dataset.base,
                                bq_codes,
                                sq8_codes,
                                rerank=args.rerank,
                            )
                            backend = "fast"
                        except Exception as exc:  # noqa: BLE001
                            if args.hybrid_backend == "fast":
                                raise
                            print(
                                f"[runner] compiled hybrid unavailable "
                                f"({type(exc).__name__}: {exc}); "
                                "using pure-Python reference",
                                file=sys.stderr,
                                flush=True,
                            )
                    if searcher is None:
                        searcher = HybridSearcher(
                            graph,
                            bq,
                            sq8,
                            dataset.base,
                            bq_codes,
                            sq8_codes,
                            rerank=args.rerank,
                        )
                        backend = "python"

                    hybrid_backend_used = backend
                    print(
                        f"[runner] evaluating D: two-stage hybrid "
                        f"(ef_bq={args.ef_bq}, rerank={args.rerank}, "
                        f"backend={backend})",
                        flush=True,
                    )
                    notes = (
                        "BQ greedy upper + BQ L0 frontier "
                        f"(ef_bq={args.ef_bq}) + {args.rerank.upper()} "
                        f"fine rerank/expansion; backend={backend}"
                    )
                    if backend == "fast":
                        record(
                            evaluate_fast_hybrid(
                                searcher,
                                args,
                                dataset,
                                ef_values,
                                graph_bytes=graph_bytes,
                                payload_bytes=payload_bytes,
                                index_bytes=graph_bytes + payload_bytes,
                                notes=notes,
                            )
                        )
                    else:
                        record(
                            evaluate_hybrid(
                                searcher,
                                args,
                                dataset,
                                ef_values,
                                graph_bytes=graph_bytes,
                                payload_bytes=payload_bytes,
                                index_bytes=graph_bytes + payload_bytes,
                                notes=notes,
                            )
                        )
                    del searcher
            except Exception as exc:  # noqa: BLE001 - keep other configs running
                message = f"A/D failed: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[runner] ERROR {message}", file=sys.stderr, flush=True)
            finally:
                index_a = None
                graph = None
                gc.collect()

        # ------------------------------------------------------------------ B
        if "B" in configs:
            try:
                print("[runner] building uniform SQ8 HNSW", flush=True)
                sq8_base = sq8.decode(sq8_codes)
                t_build = time.perf_counter()
                index_b = HNSWIndex(
                    sq8_base,
                    M=args.M,
                    ef_construction=args.ef_construction,
                )
                graph_b = index_b.graph()
                build_s = time.perf_counter() - t_build
                print(
                    f"[runner] SQ8 index built in {build_s:.1f}s; evaluating B",
                    flush=True,
                )
                record(
                    evaluate_faiss(
                        "B",
                        index_b,
                        dataset.queries,
                        args,
                        dataset,
                        ef_values,
                        graph_bytes=graph_b.graph_bytes(),
                        payload_bytes=sq8_codes.nbytes,
                        index_bytes=index_b.storage_bytes(),
                        notes=(
                            "SQ8 payload; query stays FP32 (asymmetric distance); "
                            "FAISS container stores dequantised FP32 coordinates"
                        ),
                    )
                )
                del index_b, graph_b, sq8_base
            except Exception as exc:  # noqa: BLE001
                message = f"B failed: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[runner] ERROR {message}", file=sys.stderr, flush=True)
            finally:
                gc.collect()

        # ------------------------------------------------------------------ C
        if "C" in configs:
            try:
                print("[runner] building uniform BQ HNSW", flush=True)
                bq_base = bq.decode_sign(bq_codes)
                t_build = time.perf_counter()
                index_c = HNSWIndex(
                    bq_base,
                    M=args.M,
                    ef_construction=args.ef_construction,
                )
                graph_c = index_c.graph()
                build_s = time.perf_counter() - t_build
                query_signs = bq.decode_sign(bq.encode(dataset.queries))
                print(
                    f"[runner] BQ index built in {build_s:.1f}s; evaluating C",
                    flush=True,
                )
                record(
                    evaluate_faiss(
                        "C",
                        index_c,
                        query_signs,
                        args,
                        dataset,
                        ef_values,
                        graph_bytes=graph_c.graph_bytes(),
                        payload_bytes=bq_codes.nbytes,
                        index_bytes=index_c.storage_bytes(),
                        notes=(
                            "BQ payload; binarised query; L2 on +/-1 sign codes "
                            "is 4x Hamming distance; FAISS container is FP32"
                        ),
                    )
                )
                del index_c, graph_c, bq_base, query_signs
            except Exception as exc:  # noqa: BLE001
                message = f"C failed: {type(exc).__name__}: {exc}"
                failures.append(message)
                print(f"[runner] ERROR {message}", file=sys.stderr, flush=True)
            finally:
                gc.collect()

    metadata["hybrid_backend"] = hybrid_backend_used
    metadata["failures"] = failures
    metadata["rows"] = len(rows)
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    if not rows:
        print("[runner] no benchmark rows produced", file=sys.stderr)
        return 1

    markdown = format_markdown(rows, metadata)
    markdown_path.write_text(markdown, encoding="utf-8")
    print_markdown(rows, metadata)
    print(f"[runner] CSV      -> {output_path}")
    print(f"[runner] Markdown -> {markdown_path}")
    print(f"[runner] Metadata -> {metadata_path}")
    if failures:
        print(f"[runner] {len(failures)} failure(s):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\n[runner] interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
