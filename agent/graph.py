"""
agent/graph.py — Complete LangGraph StateGraph for AgentIQ.

Graph topology:
    START → router → [retriever | web_search | direct] → [generator | END]

Conditional routing is driven by the ``route_decision`` field written by
the router node.  MemorySaver checkpointing gives every session in-process
multi-turn memory keyed by ``thread_id``.
"""

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.state import AgentState

logger = logging.getLogger(__name__)


# ── Routing helper ────────────────────────────────────────────────────────────

def _route_decision(
    state: AgentState,
) -> Literal["retriever", "web_search", "direct"]:
    decision = state.get("route_decision", "direct")
    logger.debug("Routing to: %s", decision)
    if decision == "retrieval":
        return "retriever"
    if decision == "web_search":
        return "web_search"
    return "direct"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer: MemorySaver | None = None):
    """
    Construct and compile the AgentIQ LangGraph StateGraph.

    Args:
        checkpointer: Optional MemorySaver for persistent conversation
                      memory.  When provided, every run is persisted under
                      its ``thread_id`` so multi-turn history is retained.

    Returns:
        Compiled LangGraph ``CompiledGraph`` ready to invoke or stream.
    """
    from agent.nodes import (
        direct_node,
        generator_node,
        retriever_node,
        router_node,
        web_search_node,
    )

    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("direct", direct_node)
    graph.add_node("generator", generator_node)

    # ── Entry ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # ── Conditional routing ───────────────────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "retriever": "retriever",
            "web_search": "web_search",
            "direct": "direct",
        },
    )

    # ── Convergence: retriever/web_search feed generator; direct goes to END ──
    graph.add_edge("retriever", "generator")
    graph.add_edge("web_search", "generator")
    graph.add_edge("direct", END)
    graph.add_edge("generator", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "AgentIQ graph compiled (checkpointer=%s)",
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return compiled


# ── Application-level singleton ───────────────────────────────────────────────

_graph = None


def get_graph():
    """
    Return the process-wide compiled graph singleton with MemorySaver.

    Lazy-initialised on first call so imports don't trigger compilation.

    Returns:
        Compiled LangGraph application with persistent memory.
    """
    global _graph
    if _graph is None:
        from agent.memory import get_memory
        _graph = build_graph(checkpointer=get_memory())
    return _graph
