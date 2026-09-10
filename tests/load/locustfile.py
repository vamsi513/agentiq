"""
tests/load/locustfile.py — Locust load test for AgentIQ's /chat endpoint.

Run against a LOCAL instance only, with the LLM and Tavily mocked so a run
costs nothing:

    AGENTIQ_MOCK_LLM=1 AGENTIQ_MOCK_TAVILY=1 AGENTIQ_MOCK_FAIL_PCT=0.15 \
    REDIS_URL=redis://localhost:6379/9 \
    uvicorn api.main:app --host 127.0.0.1 --port 8000

    locust -f tests/load/locustfile.py --host http://127.0.0.1:8000 \
      --headless -u 50 -r 10 -t 60s --csv results/locust

AGENTIQ_MOCK_FAIL_PCT=0.15 makes ~15% of router calls emit a garbage
route and ~15% of searches raise a transient error, so the run exercises
sanitize_route() and the retry+fallback path under load.

Custom stats reported at the end:
  task_completion_rate  - responses that returned a usable answer
  invalid_tool_call_rate - responses whose final route wasn't one of the
                           three allowed routes (should be 0)
  tool_failure_recovery_rate - web_search turns that still produced an
                               answer despite the injected search failures
"""

import random

from locust import HttpUser, between, events, task

_QUERIES = [
    "what is retrieval augmented generation",          # -> retrieval
    "explain how transformer attention works",          # -> retrieval
    "what are vector embeddings",                       # -> retrieval
    "latest AI news today",                             # -> web_search
    "current weather in London",                        # -> web_search
    "what is the price of gold right now",              # -> web_search
    "hello there",                                      # -> direct
    "thanks for the help",                              # -> direct
    "who are you",                                      # -> direct
]
_ALLOWED = {"retrieval", "web_search", "direct"}

_stats = {
    "total": 0,
    "completed": 0,
    "invalid_route": 0,
    "web_turns": 0,
    "web_recovered": 0,
}


class ChatUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task
    def chat(self):
        query = f"{random.choice(_QUERIES)} #{random.randint(1, 10**9)}"
        with self.client.post("/chat", json={"query": query}, catch_response=True) as resp:
            _stats["total"] += 1
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                body = resp.json()
            except Exception:
                resp.failure("non-JSON response")
                return

            route = body.get("route_decision", "")
            answer = body.get("answer", "") or ""
            # A "completed" turn: real answer text, not the graceful error string.
            completed = bool(answer) and "encountered an error" not in answer.lower()

            if route not in _ALLOWED:
                _stats["invalid_route"] += 1
                resp.failure(f"invalid route: {route!r}")
                return

            if completed:
                _stats["completed"] += 1
            else:
                resp.failure("no usable answer")

            # web_search turns: did we still get an answer despite injected
            # search failures? (the query keywords force the web_search route)
            if route == "web_search":
                _stats["web_turns"] += 1
                if completed:
                    _stats["web_recovered"] += 1

            resp.success()


@events.quitting.add_listener
def _report(environment, **kw):
    t = _stats["total"] or 1
    w = _stats["web_turns"] or 1
    print("\n" + "=" * 60)
    print("AGENTIQ LOAD TEST — derived rates")
    print("=" * 60)
    print(f"total requests:              {_stats['total']}")
    print(f"task_completion_rate:        {_stats['completed'] / t:.3f}")
    print(f"invalid_tool_call_rate:      {_stats['invalid_route'] / t:.3f}")
    print(f"tool_failure_recovery_rate:  {_stats['web_recovered'] / w:.3f} "
          f"({_stats['web_recovered']}/{_stats['web_turns']} web_search turns)")
