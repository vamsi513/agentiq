"""
agent/graph.py — LangGraph StateGraph definition for AgentIQ.

This module wires together all agent nodes into a directed state graph
with conditional routing.  The graph topology is:

    START → router → {retriever | web_search | generator} → generator → END

Conditional edge logic lives in ``_route_decision`` which reads the
``route_decision`` field written by the router node.

This file currently contains a **skeleton** — node implementations are
stubs imported from ``agent.nodes`` (to be completed in Commit 5).  The
full graph with real node logic and MemorySaver checkpointing is wired
in the final version of this file (Commit 6).
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from agent.state import AgentState

logger = logging.getLogger(__name__)


# ── Routing helper ────────────────────────────────────────────────────────────

def _route_decision(
    state: AgentState,
) -> Literal["retriever", "web_search", "generator"]:
    """
    Conditional edge function: maps the router's decision to the next node.

    Args:
        state: Current agent state containing ``route_decision``.

    Returns:
        Name of the next node to execute.
    """
    decision = state.get("route_decision", "direct")
    logger.debug("Routing decision: %s", decision)

    if decision == "retrieval":
        return "retriever"
    if decision == "web_search":
        return "web_search"
    # "direct" or any unexpected value goes straight to generator
    return "generator"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Construct and compile the AgentIQ LangGraph StateGraph.

    Node stubs are imported lazily so that missing dependencies during
    early development do not prevent the graph skeleton from loading.

    Args:
        checkpointer: Optional LangGraph checkpointer (e.g., MemorySaver)
                      for persistent conversation memory.  When provided,
                      every graph run is persisted under its ``thread_id``.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to invoke or stream.
    """
    # Import nodes here to avoid circular imports at module load time
    from agent.nodes import generator_node, retriever_node, router_node, web_search_node

    graph = StateGraph(AgentState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("generator", generator_node)

    # ── Entry point ───────────────────────────────────────────────────────────
    graph.add_edge(START, "router")

    # ── Conditional routing from router ───────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        _route_decision,
        {
            "retriever": "retriever",
            "web_search": "web_search",
            "generator": "generator",
        },
    )

    # ── All retrieval/search paths converge at generator ─────────────────────
    graph.add_edge("retriever", "generator")
    graph.add_edge("web_search", "generator")

    # ── Generator always terminates the run ───────────────────────────────────
    graph.add_edge("generator", END)

    # ── Compile (optionally with memory checkpointer) ─────────────────────────
    compile_kwargs = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    compiled = graph.compile(**compile_kwargs)
    logger.info("AgentIQ graph compiled (checkpointer=%s)", type(checkpointer).__name__)
    return compiled
