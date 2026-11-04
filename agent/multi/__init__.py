"""Multi agent graph for AgentIQ.

A planner, two researchers that run in parallel, a verifier and a writer,
coordinated by a deterministic supervisor. Selected with AGENT_MODE=multi;
AGENT_MODE=single keeps the original graph and is the default.

See agent/multi/graph.py for the topology and agent/multi/supervisor.py for
the control flow and its caps.
"""

from agent.multi.graph import build_multi_graph
from agent.multi.state import MultiAgentState

__all__ = ["MultiAgentState", "build_multi_graph"]
