"""
tools — External tool integrations for the AgentIQ research agent.

Currently exposes:
- web_search: Tavily-powered live web search with graceful fallback.
"""

from tools.web_search import WebSearchResult, web_search

__all__ = ["WebSearchResult", "web_search"]
