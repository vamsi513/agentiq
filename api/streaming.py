"""
api/streaming.py — Server-Sent Events (SSE) streaming helper for AgentIQ.

Provides an async generator that runs the LangGraph agent and yields
SSE-formatted chunks as the answer is produced.  The Streamlit frontend
and any SSE-capable client can consume this stream directly.

SSE event format (each chunk is a JSON-encoded StreamChunk):
    data: {"type": "token",   "data": "Hello"}\\n\\n
    data: {"type": "sources", "data": [...]}\\n\\n
    data: {"type": "done",    "data": null}\\n\\n
    data: {"type": "error",   "data": "message"}\\n\\n
"""

import json
import logging
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage

from agent.memory import get_thread_config
from config import settings

logger = logging.getLogger(__name__)


def _sse(type_: str, data) -> str:
    """
    Format a payload as a single SSE data line.

    Args:
        type_: Event type string (token / sources / done / error).
        data: JSON-serialisable payload.

    Returns:
        SSE-formatted string with trailing double newline.
    """
    payload = json.dumps({"type": type_, "data": data}, ensure_ascii=False)
    return f"data: {payload}\n\n"


async def stream_agent_response(
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the AgentIQ graph and stream SSE chunks to the caller.

    The generator yields in three phases:
    1. ``token`` events — one per word/chunk of the generated answer.
    2. ``sources`` event — full list of cited sources.
    3. ``done`` event — signals end of stream.

    On any error a single ``error`` event is yielded before the generator
    returns, so the client always receives a terminal event.

    Args:
        query: User's question string.
        session_id: Conversation session ID for MemorySaver threading.

    Yields:
        SSE-formatted strings ready to be written to the HTTP response.
    """
    from agent.graph import get_graph

    graph = get_graph()
    config = get_thread_config(session_id)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": "",
        "sources": [],
        "route_decision": "direct",
        "session_id": session_id,
        "turn_count": 0,
        "retrieval_score": 0.0,
        "error": None,
        "metadata": {},
    }

    logger.info("Streaming response for session=%s query='%.60s'", session_id, query)

    try:
        # astream_events gives us fine-grained token-level events
        answer_chunks: list[str] = []
        final_state: dict = {}

        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")
            name = event.get("name", "")

            # Stream tokens from the generator node's LLM call
            if kind == "on_chat_model_stream" and "generator" in name:
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    token = chunk.content
                    answer_chunks.append(token)
                    yield _sse("token", token)

            # Capture final state from generator node output
            elif kind == "on_chain_end" and name == "generator":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict):
                    final_state = output

        # If astream_events didn't yield tokens (direct path), run invoke
        if not answer_chunks:
            result = graph.invoke(initial_state, config=config)
            messages = result.get("messages", [])
            if messages:
                answer = messages[-1].content
                # Simulate streaming word by word
                for word in answer.split(" "):
                    yield _sse("token", word + " ")
            final_state = result

        # Emit sources
        sources = final_state.get("sources", [])
        if sources:
            yield _sse("sources", sources)

        yield _sse("done", None)
        logger.info("Stream complete for session=%s", session_id)

    except Exception as exc:
        logger.exception("Streaming error for session=%s: %s", session_id, exc)
        yield _sse("error", f"Agent error: {type(exc).__name__}. Please try again.")
        yield _sse("done", None)


async def run_agent_sync(
    query: str,
    session_id: str,
) -> dict:
    """
    Run the agent graph synchronously and return the full result dict.

    Used by the non-streaming /chat endpoint and Streamlit's fallback path.

    Args:
        query: User's question string.
        session_id: Conversation session ID.

    Returns:
        Dict with keys: ``answer``, ``sources``, ``route_decision``,
        ``session_id``, ``turn_count``, ``retrieval_score``.
    """
    from agent.graph import get_graph

    graph = get_graph()
    config = get_thread_config(session_id)

    initial_state = {
        "messages": [HumanMessage(content=query)],
        "query": query,
        "context": "",
        "sources": [],
        "route_decision": "direct",
        "session_id": session_id,
        "turn_count": 0,
        "retrieval_score": 0.0,
        "error": None,
        "metadata": {},
    }

    try:
        result = graph.invoke(initial_state, config=config)
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "No response generated."

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "route_decision": result.get("route_decision", "direct"),
            "session_id": session_id,
            "turn_count": result.get("turn_count", 0),
            "retrieval_score": result.get("retrieval_score", 0.0),
        }

    except Exception as exc:
        logger.exception("Agent sync run failed: %s", exc)
        return {
            "answer": f"Error: {type(exc).__name__} — please check your API keys and try again.",
            "sources": [],
            "route_decision": "direct",
            "session_id": session_id,
            "turn_count": 0,
            "retrieval_score": 0.0,
        }
