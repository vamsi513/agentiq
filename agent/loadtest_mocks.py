"""
agent/loadtest_mocks.py — deterministic in-process stand-ins for the LLM
and Tavily, used only for load testing so a run costs nothing and hits no
rate limits.

Activated by config flags (all default off):
  AGENTIQ_MOCK_LLM=1        -> _get_llm() returns FakeChatModel
  AGENTIQ_MOCK_TAVILY=1     -> web_search uses FakeTavilyClient
  AGENTIQ_MOCK_FAIL_PCT=0.1 -> this fraction of router calls emit a
                               garbage route (to exercise sanitize_route)
                               and this fraction of searches raise a
                               transient error (to exercise retry+fallback)

Nothing here is imported unless a flag is set.
"""

import random
import time

from langchain_core.messages import AIMessage

from config import settings

_ROUTE_KEYWORDS = {
    "web_search": ("news", "today", "current", "latest", "price", "weather", "score", "right now"),
    "retrieval": ("rag", "retrieval", "augmented", "embedding", "transformer", "attention",
                  "rlhf", "vector", "langgraph", " llm"),
}


def _fail_pct() -> float:
    try:
        return float(settings.mock_fail_pct)
    except (TypeError, ValueError):
        return 0.0


class FakeChatModel:
    """Minimal ChatOpenAI stand-in: .invoke(messages) -> AIMessage.

    Routing prompts get a keyword-based decision (or, at AGENTIQ_MOCK_FAIL_
    PCT, a deliberately invalid one). Everything else gets a canned answer.
    A small sleep simulates real inference latency without a network call.
    """

    def invoke(self, messages, *args, **kwargs):
        time.sleep(random.uniform(0.02, 0.06))
        system_text, user_text = _split_messages(messages)

        if "routing classifier" in system_text.lower():
            if random.random() < _fail_pct():
                return AIMessage(content=random.choice(
                    ["do_a_barrel_roll", "retrieval; then web_search", "42", "```json {}```"]
                ))
            # Decide from the user's query only -- the routing system prompt
            # itself lists example keywords for every route.
            lower = user_text.lower()
            for route, kws in _ROUTE_KEYWORDS.items():
                if any(kw in lower for kw in kws):
                    return AIMessage(content=route)
            return AIMessage(content="direct")

        return AIMessage(content="This is a mocked answer produced during load testing.")

    # LangChain sometimes calls these; keep them harmless.
    async def ainvoke(self, messages, *args, **kwargs):
        return self.invoke(messages, *args, **kwargs)

    def bind(self, **kwargs):
        return self


def _split_messages(messages) -> tuple[str, str]:
    """Return (system text, last user/human text)."""
    system_parts, last_user = [], ""
    for m in messages:
        if isinstance(m, dict):
            role, content = m.get("role", ""), str(m.get("content", ""))
        else:
            role = getattr(m, "type", "") or m.__class__.__name__.lower()
            content = str(getattr(m, "content", m))
        if role in ("system",):
            system_parts.append(content)
        elif role in ("user", "human", "humanmessage"):
            last_user = content
    return "\n".join(system_parts), last_user


class FakeTavilyClient:
    """AsyncTavilyClient stand-in. At AGENTIQ_MOCK_FAIL_PCT it raises a
    transient error so the real retry + fallback path in tools/web_search.py
    runs for real; otherwise it returns canned results."""

    async def search(self, query: str, **kwargs):
        import httpx

        if random.random() < _fail_pct():
            raise httpx.ConnectError("mock: simulated transient search failure")
        return {
            "results": [
                {
                    "title": f"Mock result for {query[:40]}",
                    "url": "https://example.com/mock",
                    "content": "Mocked search content for load testing.",
                    "score": 0.9,
                }
            ]
        }
