"""High-concurrency load testing and latency percentile benchmarking."""

from __future__ import annotations

import asyncio
import statistics
import time
from typing import Any

import httpx

from app.config.settings import clear_settings_cache
from app.main import create_app

BENCHMARK_SCENARIOS = [
    {
        "name": "Direct (Greeting)",
        "payload": {"message": "你好", "user_id": "u_bench"},
    },
    {
        "name": "Single-Hop (FAQ Policy)",
        "payload": {"message": "七天无理由退换货有哪些要求？", "user_id": "u_bench"},
    },
    {
        "name": "Agent (Order Status)",
        "payload": {"message": "查一下订单 ORD10001 的物流轨迹", "user_id": "u_bench"},
    },
]


async def run_scenario_concurrency(
    client: httpx.AsyncClient,
    scenario: dict[str, Any],
    concurrency: int,
    total_requests: int,
) -> dict[str, Any]:
    payload = scenario["payload"]
    latencies: list[float] = []
    successes = 0
    failures = 0

    semaphore = asyncio.Semaphore(concurrency)

    async def _send_one() -> None:
        nonlocal successes, failures
        async with semaphore:
            start = time.perf_counter()
            try:
                response = await client.post("/api/v1/chat", json=payload, timeout=60.0)
                dur = (time.perf_counter() - start) * 1000.0
                if response.status_code == 200 and response.json().get("answer"):
                    latencies.append(dur)
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1

    t0 = time.perf_counter()
    tasks = [asyncio.create_task(_send_one()) for _ in range(total_requests)]
    await asyncio.gather(*tasks)
    total_time = time.perf_counter() - t0

    latencies.sort()
    count = len(latencies)
    p50 = latencies[int(count * 0.50)] if count else 0.0
    p90 = latencies[int(count * 0.90)] if count else 0.0
    p95 = latencies[int(count * 0.95)] if count else 0.0
    p99 = latencies[int(count * 0.99)] if count else 0.0
    avg = statistics.mean(latencies) if count else 0.0
    qps = successes / total_time if total_time > 0 else 0.0

    return {
        "scenario": scenario["name"],
        "concurrency": concurrency,
        "total": total_requests,
        "success": successes,
        "failure": failures,
        "total_time_s": round(total_time, 2),
        "qps": round(qps, 2),
        "avg_ms": round(avg, 1),
        "p50_ms": round(p50, 1),
        "p90_ms": round(p90, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
    }


async def main() -> None:
    clear_settings_cache()
    app = create_app()

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
            print("=" * 70)
            print("  Adaptive-RAG Concurrency & Latency Percentile Benchmark")
            print("=" * 70)

            # Warm up
            await client.post("/api/v1/chat", json={"message": "你好", "user_id": "u_warmup"})

            results: list[dict[str, Any]] = []
            for scenario in BENCHMARK_SCENARIOS:
                for conc in [1, 5, 10]:
                    total = conc * 2
                    res = await run_scenario_concurrency(
                        client, scenario, concurrency=conc, total_requests=total
                    )
                    results.append(res)
                    print(
                        f"[{res['scenario']}] Concurrency={res['concurrency']:2d} | "
                        f"QPS={res['qps']:5.1f} | Avg={res['avg_ms']:6.1f}ms | "
                        f"P50={res['p50_ms']:6.1f}ms | P95={res['p95_ms']:6.1f}ms | "
                        f"Success={res['success']}/{res['total']}"
                    )

            print("=" * 70)
            print("Benchmark completed successfully.")


if __name__ == "__main__":
    asyncio.run(main())
