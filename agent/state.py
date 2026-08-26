"""
agent/state.py — AgentState definition for the LangGraph research agent.

The state is the single source of truth that flows between every node in
the graph.  LangGraph passes a copy of the state into each node; nodes
return a partial dict with only the keys they want to update, and
LangGraph merges those updates back.

The ``messages`` field uses ``operator.add`` as its reducer so that each
node can *append* new messages rather than replace the entire list —
this is the standard LangChain/LangGraph message-accumulation pattern.
"""

import operator
from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Shared state passed between every node in the AgentIQ graph.

    Attributes:
        messages: Full conversation history (human + AI messages).
                  Uses operator.add so nodes append rather than replace.
        query: The current user query string extracted from the last
               human message.
        context: Retrieved context string assembled by the retriever or
                 web-search node; consumed by the generator node.
        sources: List of source dicts, each containing at minimum
                 ``{"title": str, "url": str, "content": str}``.
                 Populated by the retriever or web-search node.
        route_decision: One of ``"retrieval"``, ``"web_search"``, or
                        ``"direct"`` — set by the router node and used
                        by the conditional edge to pick the next node.
        session_id: Stable identifier for the conversation session.
                    Used by MemorySaver to thread checkpoints.
        turn_count: Number of completed conversation turns in this
                    session.  Incremented by the generator node after
                    each response.
        retrieval_score: Cosine similarity score of the top retrieved
                         chunk (0.0–1.0).  Set by the retriever node;
                         0.0 when web search or direct path is used.
        error: Human-readable error message if any node failed during
               the current turn.  None when everything succeeded.
        metadata: Optional free-form dict for any extra per-run data
                  (e.g., latency measurements, token counts).
    """

    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    sources: list[dict[str, Any]]
    route_decision: str
    session_id: str
    turn_count: int
    retrieval_score: float
    error: str | None
    metadata: dict[str, Any] | None
