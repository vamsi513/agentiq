"""
tests/test_streaming.py — Tests for the SSE token-filtering logic in api/streaming.py.

Regression coverage for a bug where the direct-answer route's LLM tokens
were silently dropped: the event consumer only forwarded tokens whose
``metadata.langgraph_node`` was "generator", but the direct route's LLM
call runs inside a node registered as "direct" in the graph. Nothing
caught this because run_agent_sync (used by the eval harness) reads the
final state directly and never exercises the streaming token filter at all.
"""

import json
from unittest.mock import MagicMock, patch

import pytest

from api.streaming import stream_agent_response


def _chat_model_stream_event(node_name: str, text: str) -> dict:
    chunk = MagicMock()
    chunk.content = text
    return {
        "event": "on_chat_model_stream",
        "metadata": {"langgraph_node": node_name},
        "data": {"chunk": chunk},
    }


def _make_graph(events: list[dict]) -> MagicMock:
    graph = MagicMock()

    async def _astream_events(*args, **kwargs):
        for event in events:
            yield event

    graph.astream_events = _astream_events
    return graph


class TestStreamAgentResponse:
    @pytest.mark.asyncio
    async def test_forwards_tokens_from_generator_node(self):
        """Sanity check: retrieval/web_search route tokens (generator node) stream through."""
        graph = _make_graph([_chat_model_stream_event("generator", "Hello")])

        with patch("agent.graph.get_graph", return_value=graph):
            chunks = [c async for c in stream_agent_response("query", "session-1")]

        tokens = [json.loads(c.removeprefix("data: ").strip())["data"] for c in chunks if "\"type\": \"token\"" in c]
        assert tokens == ["Hello"]

    @pytest.mark.asyncio
    async def test_forwards_tokens_from_direct_node(self):
        """Regression test: direct-route tokens (direct node) must also stream through.

        Before the fix, only node_name == "generator" was forwarded, so every
        "Hello, how are you?"-style direct answer produced zero token events —
        the UI showed a blinking cursor and never displayed a response.
        """
        graph = _make_graph([_chat_model_stream_event("direct", "Hi there!")])

        with patch("agent.graph.get_graph", return_value=graph):
            chunks = [c async for c in stream_agent_response("Hello", "session-2")]

        tokens = [json.loads(c.removeprefix("data: ").strip())["data"] for c in chunks if "\"type\": \"token\"" in c]
        assert tokens == ["Hi there!"]

    @pytest.mark.asyncio
    async def test_ignores_tokens_from_unrelated_nodes(self):
        """Tokens tagged with a node name that isn't generator/direct are dropped."""
        graph = _make_graph([_chat_model_stream_event("router", "retrieval")])

        with patch("agent.graph.get_graph", return_value=graph):
            chunks = [c async for c in stream_agent_response("query", "session-3")]

        tokens = [c for c in chunks if "\"type\": \"token\"" in c]
        assert tokens == []
