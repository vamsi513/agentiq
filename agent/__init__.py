"""
agent — LangGraph-based multi-step research agent package.

Exports the compiled LangGraph application and the AgentState type so
callers only need to import from this top-level package.
"""

from agent.graph import build_graph
from agent.state import AgentState

__all__ = ["AgentState", "build_graph"]
