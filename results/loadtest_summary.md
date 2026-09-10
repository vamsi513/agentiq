# AgentIQ load test — Locust, local instance

Run with `tests/load/locustfile.py` against a local `uvicorn api.main:app`
started with:

```
AGENTIQ_MOCK_LLM=1 AGENTIQ_MOCK_TAVILY=1 AGENTIQ_MOCK_FAIL_PCT=0.15 \
RATE_LIMIT_PER_MIN=100000 REDIS_URL=redis://localhost:6379/9 \
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The LLM and Tavily are deterministic in-process fakes (`agent/loadtest_mocks.py`),
so the run costs nothing and hits no rate limits. `AGENTIQ_MOCK_FAIL_PCT=0.15`
makes **15% of router calls emit a garbage route** (to exercise
`sanitize_route`) and **15% of searches raise a transient error** (to
exercise the retry + fallback path). Each Locust request uses a unique
query suffix so the Redis response cache never serves a hit — these
numbers are graph-execution under load, not cache-hit latency.

Single uvicorn worker. Each level ran headless for 45s.

| Concurrent users | Requests | HTTP failures | Throughput | p50 | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|---|---|
| 10 | 313 | 0 (0.00%) | 7.0 req/s | 97 ms | 130 ms | 890 ms | 1600 ms | 2622 ms |
| 25 | 808 | 0 (0.00%) | 18.0 req/s | 96 ms | 120 ms | 130 ms | 1400 ms | 3226 ms |
| 50 | 1617 | 0 (0.00%) | 36.0 req/s | 98 ms | 130 ms | 210 ms | 1400 ms | 2968 ms |

**Derived agent-behaviour rates (all three levels):**

| Metric | Value |
|---|---|
| task_completion_rate | **1.000** — every request returned a usable answer, not the graceful error string |
| invalid_tool_call_rate | **0.000** — with 15% of router outputs deliberately garbage, `sanitize_route` corrected 100% of them; none reached a tool |
| tool_failure_recovery_rate | **1.000** (763/763 web_search turns across the three runs) — with 15% of searches failing transiently, every web_search turn still produced an answer via retry + fallback |

**Reading the latency:** p50 ~97 ms is dominated by the mocked LLM's
simulated inference sleep (~40–120 ms for the two calls a turn makes) plus
FastAPI/graph overhead. The p99/max tail (~1.4–3.2 s) is the
`wait_exponential_jitter(initial=0.5, max=4.0)` retry backoff on the
injected 15% search failures. **This is not a real end-to-end latency
number** — a real LLM adds 1–2 s per call. The value of this run is the
throughput scaling (roughly linear 7 → 18 → 36 req/s, no errors) and the
100% completion / 0% invalid-tool / 100% recovery rates under sustained
concurrency.

Not tested against the deployed EC2 / OpenShift / Streamlit Cloud
instances.
