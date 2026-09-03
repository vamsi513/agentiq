"""
api/streaming.py — Server-Sent Events (SSE) streaming helper for AgentIQ.

Provides an async generator that runs the LangGraph agent and yields
SSE-formatted chunks as the answer is produced.  The Streamlit frontend
and any SSE-capable client can consume this stream directly.

SSE event format (each chunk is a JSON-encoded payload):
    data: {"type": "route",   "data": "retrieval"}\\n\\n
    data: {"type": "token",   "data": "Hello"}\\n\\n
    data: {"type": "sources", "data": [...]}\\n\\n
    data: {"type": "done",    "data": null}\\n\\n
    data: {"type": "error",   "data": "message"}\\n\\n

Key design decisions:
- Token filtering uses ``metadata.langgraph_node`` (not event ``name``) to
  identify which graph node produced each LLM token.
- Sources are captured from the retriever/web_search node ``on_chain_end``
  events, not from the generator (which only returns messages).
- ``run_agent_sync`` uses ``graph.ainvoke()`` to avoid blocking the event loop.
"""

import json
import logging
from collections.abc import AsyncGenerator

from langchain_core.messages import HumanMessage

from agent.memory import get_thread_config

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


def _build_initial_state(query: str, session_id: str) -> dict:
    """
    Construct the initial AgentState dict for a new graph invocation.

    Args:
        query: User's question string.
        session_id: Conversation session identifier.

    Returns:
        Complete initial state dict compatible with AgentState TypedDict.
    """
    return {
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


async def stream_agent_response(
    query: str,
    session_id: str,
) -> AsyncGenerator[str, None]:
    """
    Run the AgentIQ graph and stream SSE chunks to the caller.

    Event sequence:
    1. ``route``   — the router's decision (retrieval/web_search/direct),
                     sent once the router node completes, before any tokens.
    2. ``token``   — one per LLM output token from the generator node.
    3. ``sources`` — list of cited sources (after all tokens).
    4. ``done``    — stream termination signal (always sent last).
    5. ``error``   — on failure, followed immediately by ``done``.

    The route is sent as its own event rather than left for the client to
    infer from whether sources arrived — a failed retrieval/web_search call
    (network error, empty index) legitimately produces zero sources, and a
    client inferring route from source presence alone would mislabel that
    as the direct route instead of showing the real path that failed.

    Token detection uses ``metadata.langgraph_node`` (the correct LangGraph
    0.2.x field — NOT the top-level event ``name``), checked against both
    "generator" (retrieval/web_search routes) and "direct" (the direct-
    answer route, which streams its own LLM call rather than going through
    the generator node).

    Sources are captured from retriever/web_search ``on_chain_end`` output
    dicts.  Generator output is intentionally excluded because it only
    returns ``messages``, not sources.

    Args:
        query: User's question string.
        session_id: Conversation session ID for MemorySaver threading.

    Yields:
        SSE-formatted strings ready to be written to the HTTP response.
    """
    from agent.graph import get_graph

    graph = get_graph()
    config = get_thread_config(session_id)
    initial_state = _build_initial_state(query, session_id)

    logger.info("Streaming response: session=%s query='%.60s'", session_id, query)

    try:
        captured_sources: list = []

        async for event in graph.astream_events(
            initial_state,
            config=config,
            version="v2",
        ):
            kind = event.get("event", "")

            # ── Stream tokens from the generator node's LLM call ──────────────
            # Use metadata.langgraph_node (not event name) to identify the node.
            if kind == "on_chat_model_stream":
                node_name = event.get("metadata", {}).get("langgraph_node", "")
                if node_name in ("generator", "direct"):
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield _sse("token", chunk.content)

            # ── Capture the router's decision and sources from earlier nodes ──
            # Generator only returns messages — sources come from earlier nodes.
            elif kind == "on_chain_end":
                node_name = event.get("name", "")
                output = event.get("data", {}).get("output", {})
                if node_name == "router" and isinstance(output, dict):
                    route = output.get("route_decision", "")
                    if route:
                        yield _sse("route", route)
                elif node_name in ("retriever", "web_search") and isinstance(output, dict):
                    if output.get("sources"):
                        captured_sources = output["sources"]
                elif node_name == "security" and isinstance(output, dict):
                    # A blocked query never reaches the router or any LLM
                    # call, so there are no on_chat_model_stream token events
                    # to forward -- emit the fixed refusal message here as a
                    # single token so the client still renders something
                    # instead of a blank message.
                    if output.get("route_decision") == "blocked":
                        blocked_messages = output.get("messages", [])
                        yield _sse("route", "blocked")
                        if blocked_messages:
                            yield _sse("token", blocked_messages[-1].content)

        # Emit collected sources then terminate
        if captured_sources:
            yield _sse("sources", captured_sources)

        yield _sse("done", None)
        logger.info("Stream complete: session=%s", session_id)

    except Exception as exc:
        logger.exception("Streaming error: session=%s", session_id)
        yield _sse("error", f"Agent error: {type(exc).__name__}. Please try again.")
        yield _sse("done", None)


async def run_agent_sync(
    query: str,
    session_id: str,
) -> dict:
    """
    Run the agent graph and return the full result dict (non-streaming).

    Uses ``graph.ainvoke()`` (async) to avoid blocking FastAPI's event loop.
    Calling the synchronous ``graph.invoke()`` inside an async function would
    stall all other requests — ainvoke is the correct approach.

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
    initial_state = _build_initial_state(query, session_id)

    try:
        # ainvoke is the async-safe version of invoke — never blocks event loop
        result = await graph.ainvoke(initial_state, config=config)
        messages = result.get("messages", [])
        answer = messages[-1].content if messages else "No response generated."

        return {
            "answer": answer,
            "sources": result.get("sources", []),
            "context": result.get("context", ""),
            "route_decision": result.get("route_decision", "direct"),
            "session_id": session_id,
            "turn_count": result.get("turn_count", 0),
            "retrieval_score": result.get("retrieval_score", 0.0),
        }

    except Exception:
        logger.exception("Agent sync run failed")
        raise
