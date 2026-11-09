"""Asynchronous Stress Testing & Concurrency Benchmark for Adaptive-RAG.

Evaluates high-concurrency performance across multiple query archetypes:
1. Fast-Path Greeting (Target: < 20ms)
2. Policy FAQ Hybrid Retrieval (Target: < 300ms)
3. Multi-hop Graph-RAG Conflict (Target: < 400ms)
4. Domain Secondary Development Tool (Tax / Loyalty)

Supports in-memory ASGI direct execution or live HTTP endpoints.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Literal

# Ensure UTF-8 console output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import httpx

from app.main import create_app

BENCHMARK_QUERIES = [
    {"archetype": "Greeting (Fast Path)", "query": "你好，在吗？"},
    {"archetype": "Policy FAQ (Hybrid RAG)", "query": "7天无理由退货的运费谁承担？"},
    {"archetype": "Graph-RAG Multi-Hop Conflict", "query": "定制马克杯过了七天，杯口开裂能不能退货？"},
    {"archetype": "Cross-Border Duty Tool", "query": "跨境海淘买一瓶1000元的高档香水要交多少税？"},
    {"archetype": "Loyalty Points Tool", "query": "我当前账户有多少积分？能兑换什么优惠券？"},
]


@dataclass
class RequestRecord:
    archetype: str
    latency_ms: float
    status_code: int
    success: bool
    error: str | None = None


@dataclass
class BenchmarkSummary:
    concurrency: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    total_duration_sec: float
    qps: float
    latency_p50_ms: float
    latency_p90_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    archetype_stats: dict[str, dict] = field(default_factory=dict)


async def send_worker_task(
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[str, str]],
    results: list[RequestRecord],
) -> None:
    while True:
        try:
            item = queue.get_nowait()
        except asyncio.QueueEmpty:
            break

        archetype, query = item
        t0 = time.perf_counter()
        try:
            resp = await client.post(
                "/api/v1/chat",
                json={
                    "message": query,
                    "session_id": "bench_session",
                    "user_id": "u_demo",
                },
                timeout=30.0,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            is_success = resp.status_code == 200
            err_msg = None if is_success else f"HTTP {resp.status_code}: {resp.text[:100]}"
            results.append(
                RequestRecord(
                    archetype=archetype,
                    latency_ms=elapsed_ms,
                    status_code=resp.status_code,
                    success=is_success,
                    error=err_msg,
                )
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results.append(
                RequestRecord(
                    archetype=archetype,
                    latency_ms=elapsed_ms,
                    status_code=0,
                    success=False,
                    error=str(e),
                )
            )
        finally:
            queue.task_done()


def calculate_quantiles(data: list[float], q: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * q)
    return sorted_data[min(idx, len(sorted_data) - 1)]


async def run_benchmark(
    concurrency: int = 5,
    requests_per_archetype: int = 4,
) -> BenchmarkSummary:
    app = create_app()
    results: list[RequestRecord] = []
    queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    # Populate work queue
    for item in BENCHMARK_QUERIES:
        for _ in range(requests_per_archetype):
            queue.put_nowait((item["archetype"], item["query"]))

    total_tasks = queue.qsize()
    print(f"\n🚀 开始执行并发压力基准测试: 并发 Worker 数={concurrency}, 总请求量={total_tasks}")

    # Use in-memory ASGI transport for direct high-speed evaluation without port binding
    transport = httpx.ASGITransport(app=app)
    start_time = time.perf_counter()

    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        workers = [
            asyncio.create_task(send_worker_task(client, queue, results))
            for _ in range(concurrency)
        ]
        await asyncio.gather(*workers)

    duration = time.perf_counter() - start_time
    success_records = [r for r in results if r.success]
    all_latencies = [r.latency_ms for r in success_records]

    # Archetype breakdown
    archetype_stats: dict[str, dict] = {}
    for item in BENCHMARK_QUERIES:
        arch = item["archetype"]
        arch_records = [r for r in success_records if r.archetype == arch]
        if arch_records:
            latencies = [r.latency_ms for r in arch_records]
            archetype_stats[arch] = {
                "count": len(arch_records),
                "mean_ms": statistics.mean(latencies),
                "p50_ms": calculate_quantiles(latencies, 0.50),
                "p95_ms": calculate_quantiles(latencies, 0.95),
            }

    summary = BenchmarkSummary(
        concurrency=concurrency,
        total_requests=total_tasks,
        successful_requests=len(success_records),
        failed_requests=total_tasks - len(success_records),
        total_duration_sec=duration,
        qps=total_tasks / duration if duration > 0 else 0.0,
        latency_p50_ms=calculate_quantiles(all_latencies, 0.50),
        latency_p90_ms=calculate_quantiles(all_latencies, 0.90),
        latency_p95_ms=calculate_quantiles(all_latencies, 0.95),
        latency_p99_ms=calculate_quantiles(all_latencies, 0.99),
        archetype_stats=archetype_stats,
    )
    return summary


def print_report(s: BenchmarkSummary) -> None:
    print("\n" + "=" * 80)
    print("📊 Adaptive-RAG 高并发压力测试结果报告 (Benchmark Report)")
    print("=" * 80)
    print(f"并发 Worker 数 (Workers):    {s.concurrency}")
    print(f"总请求量 (Total Requests):   {s.total_requests}")
    print(f"成功请求 (Successful):       {s.successful_requests} ({s.successful_requests / s.total_requests * 100:.1f}%)")
    print(f"失败请求 (Failed):           {s.failed_requests}")
    print(f"总耗时 (Total Duration):     {s.total_duration_sec:.2f} s")
    print(f"吞吐量 (Throughput / QPS):   {s.qps:.2f} req/s")
    print("-" * 80)
    print("分位数耗时统计 (Latency Quantiles):")
    print(f"  • P50 耗时: {s.latency_p50_ms:.1f} ms")
    print(f"  • P90 耗时: {s.latency_p90_ms:.1f} ms")
    print(f"  • P95 耗时: {s.latency_p95_ms:.1f} ms")
    print(f"  • P99 耗时: {s.latency_p99_ms:.1f} ms")
    print("-" * 80)
    print(f"{'业务场景 (Archetype)':<35} | {'样本':<4} | {'均值 (ms)':<9} | {'P50 (ms)':<9} | {'P95 (ms)':<9}")
    print("-" * 80)
    for arch, stats in s.archetype_stats.items():
        print(
            f"{arch:<35} | {stats['count']:<4} | {stats['mean_ms']:<9.1f} | "
            f"{stats['p50_ms']:<9.1f} | {stats['p95_ms']:<9.1f}"
        )
    print("=" * 80)


if __name__ == "__main__":
    summary = asyncio.run(run_benchmark(concurrency=4, requests_per_archetype=2))
    print_report(summary)
