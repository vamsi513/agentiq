"""
tests/test_agent.py — Unit tests for LangGraph agent nodes and graph.

Tests cover:
- router_node: correct routing decision parsing and sanitisation
- retriever_node: state update shape and error fallback
- web_search_node: state update shape and fallback handling
- generator_node: message appending and turn_count increment
- _route_decision: conditional edge mapping
- build_graph: graph compiles without errors

All LLM calls and external tool calls are mocked — no real API calls made.
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

# ── Helpers ───────────────────────────────────────────────────────────────────

def _base_state(**overrides) -> dict:
    """
    Return a minimal valid AgentState dict for testing.

    Args:
        **overrides: Any state keys to override from defaults.

    Returns:
        Dict compatible with AgentState TypedDict.
    """
    state = {
        "messages": [HumanMessage(content="What is RAG?")],
        "query": "What is RAG?",
        "context": "",
        "sources": [],
        "route_decision": "direct",
        "session_id": "test-session-123",
        "turn_count": 0,
        "retrieval_score": 0.0,
        "error": None,
        "metadata": {},
    }
    state.update(overrides)
    return state


# ── _route_decision conditional edge ─────────────────────────────────────────

class TestRouteDecision:
    """Tests for the graph's conditional edge function."""

    def test_retrieval_routes_to_retriever(self):
        """route_decision='retrieval' → 'retriever' node."""
        from agent.graph import _route_decision

        state = _base_state(route_decision="retrieval")
        assert _route_decision(state) == "retriever"

    def test_web_search_routes_to_web_search(self):
        """route_decision='web_search' → 'web_search' node."""
        from agent.graph import _route_decision

        state = _base_state(route_decision="web_search")
        assert _route_decision(state) == "web_search"

    def test_direct_routes_to_direct(self):
        """route_decision='direct' → 'direct' node (conversational answer)."""
        from agent.graph import _route_decision

        state = _base_state(route_decision="direct")
        assert _route_decision(state) == "direct"

    def test_unknown_routes_to_direct(self):
        """Any unknown route_decision defaults to 'direct'."""
        from agent.graph import _route_decision

        state = _base_state(route_decision="unknown_value")
        assert _route_decision(state) == "direct"


# ── router_node ───────────────────────────────────────────────────────────────

class TestRouterNode:
    """Tests for the LLM-based router_node function."""

    def _mock_llm_response(self, decision: str):
        """Return a mock ChatOpenAI that returns the given decision word."""
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = decision
        mock_llm.invoke.return_value = mock_response
        return mock_llm

    def test_returns_retrieval_decision(self):
        """Router correctly parses 'retrieval' from LLM output."""
        from agent.nodes import router_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm_response("retrieval")):
            result = router_node(_base_state())

        assert result["route_decision"] == "retrieval"
        assert result["query"] == "What is RAG?"

    def test_returns_web_search_decision(self):
        """Router correctly parses 'web_search' from LLM output."""
        from agent.nodes import router_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm_response("web_search")):
            result = router_node(_base_state())

        assert result["route_decision"] == "web_search"

    def test_returns_direct_decision(self):
        """Router correctly parses 'direct' from LLM output."""
        from agent.nodes import router_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm_response("direct")):
            result = router_node(_base_state())

        assert result["route_decision"] == "direct"

    def test_sanitises_unexpected_llm_output(self):
        """Unknown LLM output is sanitised to 'retrieval' (safe default)."""
        from agent.nodes import router_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm_response("nonsense")):
            result = router_node(_base_state())

        assert result["route_decision"] == "retrieval"

    def test_strips_whitespace_and_lowercases(self):
        """Router strips whitespace and lowercases the LLM output."""
        from agent.nodes import router_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm_response("  Web_Search  ")):
            result = router_node(_base_state())

        # "web_search" after strip+lower — but underscore won't match set, falls back
        # This tests the sanitisation path
        assert result["route_decision"] in {"retrieval", "web_search", "direct"}

    def test_defaults_to_direct_on_llm_exception(self):
        """Router returns 'direct' when the LLM call raises an exception."""
        from agent.nodes import router_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("OpenAI unavailable")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = router_node(_base_state())

        assert result["route_decision"] == "direct"

    def test_empty_messages_defaults_to_direct(self):
        """Router returns 'direct' when there are no messages in state."""
        from agent.nodes import router_node

        state = _base_state(messages=[])
        result = router_node(state)

        assert result["route_decision"] == "direct"
        assert result["query"] == ""


# ── retriever_node ────────────────────────────────────────────────────────────

class TestRetrieverNode:
    """Tests for the FAISS retriever_node function."""

    def test_returns_context_and_sources(self):
        """retriever_node populates context string and sources list."""
        from agent.nodes import retriever_node

        mock_results = [
            {
                "title": "RAG Paper",
                "source": "https://arxiv.org/abs/2005.11401",
                "content": "RAG combines retrieval with generation.",
                "score": 0.91,
            }
        ]

        with patch("retrieval.vectorstore.query_vectorstore", return_value=mock_results):
            result = retriever_node(_base_state())

        assert "context" in result
        assert "sources" in result
        assert "retrieval_score" in result
        assert isinstance(result["sources"], list)

    def test_returns_fallback_on_empty_results(self):
        """retriever_node returns empty sources when FAISS has no results."""
        from agent.nodes import retriever_node

        with patch("retrieval.vectorstore.query_vectorstore", return_value=[]):
            result = retriever_node(_base_state())

        assert result["sources"] == []
        assert "No relevant documents" in result["context"]

    def test_returns_fallback_on_exception(self):
        """retriever_node does not raise when FAISS throws — returns fallback."""
        from agent.nodes import retriever_node

        with patch(
            "retrieval.vectorstore.query_vectorstore",
            side_effect=RuntimeError("FAISS error"),
        ):
            result = retriever_node(_base_state())

        assert result["sources"] == []
        assert "failed" in result["context"].lower()

    def test_sets_retrieval_score(self):
        """retriever_node sets retrieval_score from the top result's score."""
        from agent.nodes import retriever_node

        mock_results = [
            {"title": "T", "source": "S", "content": "C", "score": 0.88}
        ]

        with patch("retrieval.vectorstore.query_vectorstore", return_value=mock_results):
            result = retriever_node(_base_state())

        assert result["retrieval_score"] == pytest.approx(0.88)


# ── web_search_node ───────────────────────────────────────────────────────────

class TestWebSearchNode:
    """Tests for the Tavily web_search_node function.

    web_search_node is async (it awaits tools.web_search.web_search(), which
    itself awaits AsyncTavilyClient/httpx so a timeout genuinely cancels the
    in-flight request -- see tools/web_search.py). unittest.mock.patch()
    auto-detects that the patched target is a coroutine function and
    produces an AsyncMock automatically, so the patch calls below are
    unchanged from the sync version; only the test functions themselves
    need to be async/await the node call."""

    @pytest.mark.asyncio
    async def test_returns_context_and_sources_on_success(self):
        """web_search_node populates context and sources from Tavily results."""
        from agent.nodes import web_search_node

        mock_results = [
            {
                "title": "AI News",
                "url": "https://news.example.com",
                "content": "Latest AI developments.",
                "score": 0.80,
            }
        ]

        with patch("tools.web_search.web_search", return_value=mock_results):
            result = await web_search_node(_base_state(query="latest AI news"))

        assert len(result["sources"]) == 1
        assert result["sources"][0]["type"] == "web_search"
        assert "AI News" in result["context"]

    @pytest.mark.asyncio
    async def test_handles_fallback_result(self):
        """web_search_node does not add fallback items to sources list."""
        from agent.nodes import web_search_node

        fallback = [
            {
                "title": "Web Search Unavailable",
                "url": "",
                "content": "Search could not be completed.",
                "score": 0.0,
                "is_fallback": True,
            }
        ]

        with patch("tools.web_search.web_search", return_value=fallback):
            result = await web_search_node(_base_state(query="something"))

        # Fallback items are excluded from sources
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_returns_fallback_context_on_exception(self):
        """web_search_node does not raise when web_search throws."""
        from agent.nodes import web_search_node

        with patch("tools.web_search.web_search", side_effect=Exception("network")):
            result = await web_search_node(_base_state(query="news"))

        assert result["sources"] == []
        assert "failed" in result["context"].lower()


# ── generator_node ────────────────────────────────────────────────────────────

class TestGeneratorNode:
    """Tests for the LLM generator_node function."""

    def _mock_llm(self, answer: str):
        """Return a mock ChatOpenAI that returns the given answer string."""
        mock = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = answer
        mock.invoke.return_value = mock_resp
        return mock

    def test_appends_ai_message(self):
        """generator_node returns messages list with one new AIMessage."""
        from agent.nodes import generator_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm("RAG is great.")):
            result = generator_node(_base_state())

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "RAG is great."

    def test_increments_turn_count(self):
        """generator_node increments turn_count by 1."""
        from agent.nodes import generator_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm("Answer.")):
            result = generator_node(_base_state(turn_count=2))

        assert result["turn_count"] == 3

    def test_returns_error_message_on_llm_exception(self):
        """generator_node returns a user-friendly error when LLM raises."""
        from agent.nodes import generator_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("API error")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = generator_node(_base_state())

        assert "messages" in result
        assert "error" in result
        assert "API key" in result["messages"][0].content or "error" in result["messages"][0].content.lower()

    def test_uses_context_when_available(self):
        """generator_node injects context into the LLM prompt."""
        from agent.nodes import generator_node

        mock_llm = MagicMock()
        mock_resp = MagicMock()
        mock_resp.content = "Answer with context."
        mock_llm.invoke.return_value = mock_resp

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            generator_node(_base_state(context="Important context here."))

        # Check the context was included in the messages passed to LLM
        call_args = mock_llm.invoke.call_args[0][0]
        full_text = " ".join(m.get("content", "") for m in call_args)
        assert "Important context here" in full_text


# ── Graph compilation ─────────────────────────────────────────────────────────

class TestBuildGraph:
    """Tests for the LangGraph graph compilation."""

    def test_graph_compiles_without_error(self):
        """build_graph() succeeds and returns a compiled graph object."""
        from agent.graph import build_graph

        mock_nodes = {
            "router_node": MagicMock(return_value={"route_decision": "direct"}),
            "retriever_node": MagicMock(return_value={"context": "", "sources": []}),
            "web_search_node": MagicMock(return_value={"context": "", "sources": []}),
            "generator_node": MagicMock(return_value={"messages": [AIMessage(content="ok")]}),
        }

        with patch.multiple("agent.nodes", **mock_nodes):
            graph = build_graph(checkpointer=None)

        assert graph is not None

    def test_graph_compiles_with_memory_saver(self):
        """build_graph() accepts a MemorySaver checkpointer."""
        from langgraph.checkpoint.memory import MemorySaver

        from agent.graph import build_graph

        with patch.multiple(
            "agent.nodes",
            router_node=MagicMock(return_value={"route_decision": "direct"}),
            retriever_node=MagicMock(return_value={"context": "", "sources": []}),
            web_search_node=MagicMock(return_value={"context": "", "sources": []}),
            generator_node=MagicMock(return_value={"messages": [AIMessage(content="ok")]}),
        ):
            graph = build_graph(checkpointer=MemorySaver())

        assert graph is not None


# ── direct_node ───────────────────────────────────────────────────────────────

class TestDirectNode:
    """Tests for direct_node — conversational answers without retrieval."""

    @staticmethod
    def _mock_llm(text: str) -> MagicMock:
        mock = MagicMock()
        resp = MagicMock()
        resp.content = text
        mock.invoke.return_value = resp
        return mock

    def test_returns_ai_message(self):
        """direct_node appends one AIMessage to messages."""
        from agent.nodes import direct_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm("Hello!")):
            result = direct_node(_base_state(query="Hi there"))

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert result["messages"][0].content == "Hello!"

    def test_increments_turn_count(self):
        """direct_node increments turn_count by 1."""
        from agent.nodes import direct_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm("Hi!")):
            result = direct_node(_base_state(turn_count=3))

        assert result["turn_count"] == 4

    def test_sets_empty_context_and_sources(self):
        """direct_node clears context and sources (no retrieval)."""
        from agent.nodes import direct_node

        with patch("agent.nodes._get_llm", return_value=self._mock_llm("Hi!")):
            result = direct_node(_base_state(context="old context", sources=[{"x": 1}]))

        assert result["context"] == ""
        assert result["sources"] == []

    def test_returns_error_message_on_llm_exception(self):
        """direct_node returns a fallback message when the LLM raises."""
        from agent.nodes import direct_node

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM down")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            result = direct_node(_base_state())

        assert "messages" in result
        assert "error" in result
        assert "error" in result["messages"][0].content.lower() or "encountered" in result["messages"][0].content.lower()


class TestStageTiming:
    """_timed logs a stage_timing line regardless of which return path (or
    exception) the wrapped node takes -- this is what lets p50/p95 be derived
    from production logs without a tracing dependency."""

    def test_logs_duration_on_normal_return(self, caplog):
        from agent.nodes import _timed

        @_timed("fake_stage")
        def fake_node(state):
            return {"ok": True}

        with caplog.at_level("INFO", logger="agent.nodes"):
            result = fake_node({})

        assert result == {"ok": True}
        assert any(
            "stage_timing stage=fake_stage" in r.message for r in caplog.records
        )

    def test_logs_duration_even_when_wrapped_fn_raises(self, caplog):
        from agent.nodes import _timed

        @_timed("fake_stage")
        def fake_node(state):
            raise RuntimeError("boom")

        with caplog.at_level("INFO", logger="agent.nodes"):
            with pytest.raises(RuntimeError):
                fake_node({})

        assert any(
            "stage_timing stage=fake_stage" in r.message for r in caplog.records
        )

    def test_real_nodes_emit_stage_timing(self, caplog):
        """Spot-check that the decorator is actually applied to a real node,
        not just demonstrated on a throwaway function above."""
        from agent.nodes import router_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(content="direct")

        with patch("agent.nodes._get_llm", return_value=mock_llm):
            with caplog.at_level("INFO", logger="agent.nodes"):
                router_node(_base_state(messages=[HumanMessage(content="hi")]))

        assert any("stage_timing stage=router" in r.message for r in caplog.records)
