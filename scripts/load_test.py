"""
scripts/load_test.py — Reproducible load test for the AgentIQ FastAPI app.

Drives concurrent requests against the app *in-process* via
httpx.AsyncClient(transport=ASGITransport(app=app)) -- no real server needs
to be running, and no real network call is made to OpenAI or Tavily. Both
are mocked with a small artificial delay to stand in for a plausible
upstream response time.

This measures APPLICATION PERFORMANCE ONLY: FastAPI request handling,
Pydantic validation, the rate limiter, LangGraph orchestration overhead
(security → router → node → generator), and response serialization. It
deliberately does NOT measure THIRD-PARTY PROVIDER PERFORMANCE (real OpenAI/
Tavily latency varies with their load, your plan tier, and network
conditions, and is a completely separate number from anything below).

Usage:
    python -m scripts.load_test                  # default: 10, 25, 50 concurrent
    python -m scripts.load_test --levels 5 20     # custom concurrency levels
    python -m scripts.load_test --requests-per-level 100

For a REAL end-to-end load test against actual OpenAI/Tavily latency and
your actual deployed rate limits/quotas, see the "Real-provider load
testing" section in README.md's Production Readiness section -- that is
NOT run automatically here because it costs money and can trip real
provider rate limits.
"""

import argparse
import asyncio
import time
from unittest.mock import patch

import httpx
from httpx import ASGITransport
from langchain_core.messages import AIMessage


def _mock_llm_invoke(*args, **kwargs):
    """Stand-in for a real OpenAI call: fixed content, small artificial
    delay representative of a fast gpt-4o-mini round trip. This is a mock,
    not a benchmark of OpenAI -- see the module docstring."""
    time.sleep(0.05)
    return AIMessage(content="direct")


def _mock_web_search(query, max_results=None):
    time.sleep(0.05)
    return [{"title": "Mock Result", "url": "https://example.com", "content": "mock content", "score": 0.9}]


async def _one_request(client: httpx.AsyncClient, query: str) -> tuple[float, int]:
    start = time.perf_counter()
    try:
        response = await client.post("/chat", json={"query": query}, timeout=30.0)
        status = response.status_code
    except Exception:
        status = -1
    elapsed_ms = (time.perf_counter() - start) * 1000
    return elapsed_ms, status


async def _run_level(app, concurrency: int, total_requests: int) -> dict:
    transport = ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def _bounded(i: int):
            async with semaphore:
                return await _one_request(client, f"What is the Transformer architecture? (req {i})")

        wall_start = time.perf_counter()
        results = await asyncio.gather(*[_bounded(i) for i in range(total_requests)])
        wall_elapsed = time.perf_counter() - wall_start

    latencies = sorted(r[0] for r in results)
    statuses = [r[1] for r in results]
    error_count = sum(1 for s in statuses if s != 200)

    def _pct(p: float) -> float:
        idx = min(len(latencies) - 1, int(len(latencies) * p))
        return latencies[idx]

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_seconds": round(wall_elapsed, 3),
        "throughput_rps": round(total_requests / wall_elapsed, 2),
        "p50_ms": round(_pct(0.50), 1),
        "p95_ms": round(_pct(0.95), 1),
        "p99_ms": round(_pct(0.99), 1),
        "min_ms": round(latencies[0], 1),
        "max_ms": round(latencies[-1], 1),
        "error_count": error_count,
        "error_rate": round(error_count / total_requests, 4),
    }


async def main(levels: list[int], requests_per_level: int) -> None:
    # Import here so mocks below apply before the module-level LLM/client
    # singletons are constructed on first use.
    from api.main import _rate_counters, app

    print("=" * 70)
    print("  AgentIQ Load Test — APPLICATION PERFORMANCE ONLY")
    print("  (LLM and web search are mocked; this does NOT measure")
    print("   OpenAI/Tavily provider latency — see module docstring)")
    print("=" * 70)

    # ASGITransport requests all share one synthetic client IP. The 30
    # req/min/IP rate limiter (api/main.py) would then reject most of a
    # 100-request burst as abuse from a single client -- which is CORRECT
    # limiter behavior, but it's a different thing than pipeline throughput
    # and would make every level below look artificially error-prone.
    # Bypassed here (patched to a no-op) so these numbers isolate
    # application/orchestration performance; the limiter's own behavior
    # under concurrent load is demonstrated separately right after.
    results = []
    for concurrency in levels:
        _rate_counters.clear()

        mock_llm = type("MockLLM", (), {"invoke": staticmethod(_mock_llm_invoke)})()
        with patch("agent.nodes._get_llm", return_value=mock_llm), \
             patch("tools.web_search.web_search", side_effect=_mock_web_search), \
             patch("api.main._check_rate_limit"):
            level_result = await _run_level(app, concurrency, requests_per_level)
        results.append(level_result)

        print(f"\nConcurrency: {concurrency} | Requests: {requests_per_level}")
        print(f"  Throughput : {level_result['throughput_rps']} req/s")
        print(f"  p50        : {level_result['p50_ms']} ms")
        print(f"  p95        : {level_result['p95_ms']} ms")
        print(f"  p99        : {level_result['p99_ms']} ms")
        print(f"  Error rate : {level_result['error_rate']:.2%} ({level_result['error_count']} errors)")

    # ── Separately: confirm the rate limiter itself fires under real
    # concurrent load, not just in the sequential unit tests in test_api.py ──
    print("\n" + "-" * 70)
    print("  Rate limiter under concurrent load (limiter NOT bypassed)")
    print("  30 req/min/IP; firing 60 concurrent requests from one IP")
    print("-" * 70)
    _rate_counters.clear()
    mock_llm = type("MockLLM", (), {"invoke": staticmethod(_mock_llm_invoke)})()
    with patch("agent.nodes._get_llm", return_value=mock_llm), \
         patch("tools.web_search.web_search", side_effect=_mock_web_search):
        limiter_result = await _run_level(app, concurrency=60, total_requests=60)
    allowed = limiter_result["total_requests"] - limiter_result["error_count"]
    print(f"  Allowed: {allowed} | Rejected (429): {limiter_result['error_count']}")
    _rate_counters.clear()

    print("\n" + "=" * 70)
    print("  Done. These numbers reflect this machine, right now, with")
    print("  mocked upstreams -- re-run locally to reproduce, don't treat")
    print("  them as a portable absolute benchmark.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--levels", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--requests-per-level", type=int, default=100)
    args = parser.parse_args()

    asyncio.run(main(args.levels, args.requests_per_level))
