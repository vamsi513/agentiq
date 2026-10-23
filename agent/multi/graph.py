"""The multi agent LangGraph.

Topology:

    START -> security -> [blocked -> END]
                      -> planner -> supervisor
    supervisor -> doc_researcher and/or web_researcher -> supervisor
               -> writer                                -> supervisor
               -> verifier                              -> supervisor
               -> publish -> END

Every agent returns to the supervisor, and the supervisor is the only place a
next step is chosen. That shape is what makes the control flow testable: the
edges are fixed, and which one is taken is decided by ``decide_next``, a pure
function of state with no model call in it.

The two researchers are dispatched together from one conditional edge. A
conditional edge that returns a list of node names fans out to all of them in
the same superstep, and LangGraph joins them back at the supervisor once both
have finished. They never write the same state key, so the join needs no
reducer.

The security node is reused unchanged from the single agent path. It is the
first node in both graphs, so a query that is blocked there reaches no
planner, no researcher and no tool in either mode.
"""

from __future__ import annotations

import logging
from typing import Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from agent.multi.state import (
    PHASE_DONE,
    PHASE_RESEARCH,
    PHASE_VERIFY,
    PHASE_WRITE,
    SOURCE_DOCUMENTS,
    SOURCE_WEB,
    MultiAgentState,
    planned_sources,
)

logger = logging.getLogger(__name__)

NODE_DOC = "doc_researcher"
NODE_WEB = "web_researcher"
NODE_WRITER = "writer"
NODE_VERIFIER = "verifier"
NODE_PUBLISH = "publish"


def _security_gate(state: MultiAgentState) -> Literal["blocked", "continue"]:
    """Send a query the security node blocked straight to END."""
    return "blocked" if state.get("route_decision") == "blocked" else "continue"


def research_targets(state: MultiAgentState) -> list[str]:
    """Which researchers to run this round.

    A retry request names exactly one source, and on a retry round that is the
    only researcher worth paying for. Otherwise every source the plan asks for
    runs, in one parallel superstep.

    Returns at least one node name. Returning an empty list from a conditional
    edge would strand the graph with no next node, so a round that somehow has
    no target falls through to the document researcher, which is a no-op when
    the plan gives it nothing to do.
    """
    retry = state.get("retry_request")
    if retry and state.get("research_rounds", 0) > 1:
        source = retry.get("source")
        if source == SOURCE_WEB:
            return [NODE_WEB]
        if source == SOURCE_DOCUMENTS:
            return [NODE_DOC]

    wanted = planned_sources(state)
    targets = []
    if SOURCE_DOCUMENTS in wanted:
        targets.append(NODE_DOC)
    if SOURCE_WEB in wanted:
        targets.append(NODE_WEB)
    return targets or [NODE_DOC]


def _dispatch(state: MultiAgentState) -> list[str]:
    """Map the supervisor's chosen phase onto the node or nodes to run."""
    phase = state.get("phase", PHASE_DONE)
    if phase == PHASE_RESEARCH:
        return research_targets(state)
    if phase == PHASE_WRITE:
        return [NODE_WRITER]
    if phase == PHASE_VERIFY:
        return [NODE_VERIFIER]
    return [NODE_PUBLISH]


def build_multi_graph(checkpointer: MemorySaver | None = None):
    """Construct and compile the multi agent graph.

    Args:
        checkpointer: Optional checkpointer giving each session multi turn
                      memory keyed by ``thread_id``, as in the single agent
                      graph.

    Returns:
        A compiled LangGraph application.
    """
    from agent.multi.planner import planner_node
    from agent.multi.researchers import doc_researcher_node, web_researcher_node
    from agent.multi.supervisor import supervisor_node
    from agent.multi.verifier import verifier_node
    from agent.multi.writer import publish_node, writer_node
    from agent.nodes import security_node

    graph = StateGraph(MultiAgentState)

    graph.add_node("security", security_node)
    graph.add_node("planner", planner_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node(NODE_DOC, doc_researcher_node)
    graph.add_node(NODE_WEB, web_researcher_node)
    graph.add_node(NODE_WRITER, writer_node)
    graph.add_node(NODE_VERIFIER, verifier_node)
    graph.add_node(NODE_PUBLISH, publish_node)

    graph.add_edge(START, "security")
    graph.add_conditional_edges(
        "security",
        _security_gate,
        {"blocked": END, "continue": "planner"},
    )
    graph.add_edge("planner", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _dispatch,
        {
            NODE_DOC: NODE_DOC,
            NODE_WEB: NODE_WEB,
            NODE_WRITER: NODE_WRITER,
            NODE_VERIFIER: NODE_VERIFIER,
            NODE_PUBLISH: NODE_PUBLISH,
        },
    )

    # Every agent reports back to the supervisor. The two researchers share
    # this edge, which is where LangGraph joins a parallel round.
    for node in (NODE_DOC, NODE_WEB, NODE_WRITER, NODE_VERIFIER):
        graph.add_edge(node, "supervisor")

    graph.add_edge(NODE_PUBLISH, END)

    compiled = graph.compile(checkpointer=checkpointer)
    logger.info(
        "AgentIQ multi agent graph compiled (checkpointer=%s)",
        type(checkpointer).__name__ if checkpointer else "None",
    )
    return compiled
