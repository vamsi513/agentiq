"""
tests/test_failure_modes.py — Deterministic tests for external-dependency and
internal failure modes.

All external services (Tavily, OpenAI) are mocked -- nothing here makes a
real network call or depends on a paid API key, so it runs safely in CI.

Covers: Tavily 429 / timeout / 5xx / connection error, LLM timeout / error,
empty retrieval, malformed request, invalid router output, and the bounded
(non-storm) nature of the retry logic in tools/web_search.py.
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import tools.web_search  # noqa: F401 — ensures tools.web_search is registered in sys.modules
from api.main import app
from tavily.errors import InvalidAPIKeyError, UsageLimitExceededError

client = TestClient(app)


def _reset_tavily_client():
    sys.modules["tools.web_search"]._client = None


def _http_status_error(status_code: int) -> httpx.HTTPStatusError:
    response = MagicMock()
    response.status_code = status_code
    return httpx.HTTPStatusError("error", request=MagicMock(), response=response)


# ── Tavily: rate limit (429) ─────────────────────────────────────────────────

class TestTavilyRateLimit:
    @pytest.mark.asyncio
    async def test_429_returns_fallback_without_retrying(self):
        """429 must NOT be retried -- retrying a rate limit just hammers the
        same window harder. Assert both the fallback behavior and that the
        client was called exactly once (no retry storm)."""
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=UsageLimitExceededError("Too many requests."))
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert "rate limited" in results[0]["content"]
        assert mock_client.search.call_count == 1  # not retried
        _reset_tavily_client()


# ── Tavily: timeout ───────────────────────────────────────────────────────────

class TestTavilyTimeout:
    @pytest.mark.asyncio
    async def test_client_side_timeout_returns_fallback(self):
        """A call that hangs past _SEARCH_TIMEOUT_SECONDS must produce a
        fallback, not hang the caller indefinitely. Also confirms the
        asyncio.wait_for-based timeout genuinely cancels the awaited
        coroutine rather than leaving it running -- the hang task raises
        asyncio.CancelledError internally when wait_for gives up on it."""
        ws = sys.modules["tools.web_search"]
        cancelled = False

        async def _hang(*args, **kwargs):
            nonlocal cancelled
            try:
                await asyncio.sleep(ws._SEARCH_TIMEOUT_SECONDS + 5)
            except asyncio.CancelledError:
                cancelled = True
                raise

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_hang)
        ws._client = mock_client

        # _SEARCH_TIMEOUT_SECONDS is read at call time so patching it here is
        # effective; _MAX_ATTEMPTS is baked into the @retry decorator at
        # import time and can't be overridden per-test, so this still runs
        # the real 3 attempts (bounded either way, just not shortened).
        with patch.object(ws, "_SEARCH_TIMEOUT_SECONDS", 0.2), \
             patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert "timed out" in results[0]["content"]
        assert cancelled is True  # proves real cancellation, not an abandoned task
        _reset_tavily_client()

    @pytest.mark.asyncio
    async def test_httpx_timeout_exception_is_retried_then_falls_back(self):
        """httpx.TimeoutException IS transient -- confirm it gets retried up
        to _MAX_ATTEMPTS (bounded, not unbounded) before falling back."""
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        # Retried exactly _MAX_ATTEMPTS times -- bounded, proves no retry storm.
        assert mock_client.search.call_count == 3
        _reset_tavily_client()


# ── Tavily: 5xx ───────────────────────────────────────────────────────────────

class TestTavilyServerError:
    @pytest.mark.asyncio
    async def test_5xx_is_retried_then_falls_back(self):
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_http_status_error(503))
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert mock_client.search.call_count == 3  # bounded retry, not infinite
        _reset_tavily_client()

    @pytest.mark.asyncio
    async def test_4xx_other_than_429_is_not_retried(self):
        """A 4xx that isn't 429 (e.g. 400) can't be fixed by retrying --
        confirm it's treated as non-transient."""
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=_http_status_error(400))
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert mock_client.search.call_count == 1  # not retried
        _reset_tavily_client()


# ── Tavily: connection failure ────────────────────────────────────────────────

class TestTavilyConnectionFailure:
    @pytest.mark.asyncio
    async def test_connection_error_is_retried_then_falls_back(self):
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=httpx.ConnectError("refused"))
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert mock_client.search.call_count == 3
        _reset_tavily_client()


# ── Tavily: invalid API key ───────────────────────────────────────────────────

class TestTavilyInvalidApiKey:
    @pytest.mark.asyncio
    async def test_invalid_api_key_returns_fallback_without_retry(self):
        ws = sys.modules["tools.web_search"]

        mock_client = MagicMock()
        mock_client.search = AsyncMock(side_effect=InvalidAPIKeyError())
        ws._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5
            results = await ws.web_search("test query")

        assert results[0]["is_fallback"] is True
        assert mock_client.search.call_count == 1  # never fixed by retrying
        _reset_tavily_client()


# ── LLM failures ──────────────────────────────────────────────────────────────

class TestLLMFailures:
    def test_router_falls_back_to_direct_on_llm_exception(self):
        from agent.nodes import router_node
        from langchain_core.messages import HumanMessage

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = TimeoutError("LLM request timed out")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = router_node({"messages": [HumanMessage(content="test")]})

        assert result["route_decision"] == "direct"

    def test_generator_returns_error_message_on_llm_exception(self):
        from agent.nodes import generator_node
        from langchain_core.messages import HumanMessage

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = ConnectionError("connection reset")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = generator_node({
                "query": "test", "context": "some context",
                "messages": [HumanMessage(content="test")], "turn_count": 0,
            })

        assert "error" in result
        assert len(result["messages"]) == 1


# ── Empty retrieval ───────────────────────────────────────────────────────────

class TestEmptyRetrieval:
    def test_retriever_node_handles_empty_results_gracefully(self):
        from agent.nodes import retriever_node

        with patch("retrieval.vectorstore.query_vectorstore", return_value=[]):
            result = retriever_node({"query": "an extremely obscure query", "session_id": "s1"})

        assert result["sources"] == []
        assert result["context"]  # some placeholder context, not a crash


# ── Malformed API requests ────────────────────────────────────────────────────

class TestMalformedRequests:
    def test_empty_query_is_rejected_by_schema(self):
        response = client.post("/chat", json={"query": ""})
        assert response.status_code == 422

    def test_missing_query_field_is_rejected(self):
        response = client.post("/chat", json={})
        assert response.status_code == 422

    def test_oversized_query_is_rejected_by_schema(self):
        response = client.post("/chat", json={"query": "x" * 5000})
        assert response.status_code == 422

    def test_non_string_query_is_rejected(self):
        response = client.post("/chat", json={"query": 12345})
        assert response.status_code == 422

    def test_malformed_json_body_is_rejected(self):
        response = client.post(
            "/chat", content=b"{not valid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


# ── Router produces an unexpected value ───────────────────────────────────────

class TestRouterInvalidOutput:
    def test_unrecognized_llm_route_output_defaults_to_retrieval(self):
        """router_node sanitizes unexpected LLM output rather than passing
        it through unchecked -- confirms the existing safety net."""
        from agent.nodes import router_node
        from langchain_core.messages import AIMessage, HumanMessage

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="something_unexpected")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = router_node({"messages": [HumanMessage(content="test")]})

        assert result["route_decision"] == "retrieval"

    def test_graph_conditional_edge_never_crashes_on_unknown_route(self):
        from agent.graph import _route_decision

        assert _route_decision({"route_decision": "totally_unknown_value"}) == "direct"
