"""
tools/web_search.py — Tavily Search API wrapper with robust error handling.

Provides a single ``web_search`` function that queries Tavily for live web
results and returns a normalised list of result dicts.  All error cases are
caught and logged; the function never raises to its callers — instead it
returns a fallback payload so the generator node can still produce a response
(via direct LLM knowledge) when the search API is unavailable.

Tavily free tier: 1,000 searches / month.
Get a key at: https://tavily.com
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class WebSearchResult:
    """
    Normalised container for a single web search result.

    Attributes:
        title: Page title from the search result.
        url: Canonical URL of the result page.
        content: Snippet or extracted content from the page.
        score: Tavily relevance score (0–1); 0.0 if unavailable.
    """

    title: str
    url: str
    content: str
    score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise to a plain dict for JSON-safe transport.

        Returns:
            Dict with keys ``title``, ``url``, ``content``, ``score``.
        """
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
        }


def _build_client():
    """
    Lazily construct a TavilyClient, returning None if the key is missing.

    Returns:
        TavilyClient instance, or None if TAVILY_API_KEY is not set.
    """
    if not settings.is_tavily_configured():
        logger.warning("TAVILY_API_KEY not set — web search is disabled.")
        return None
    try:
        from tavily import TavilyClient  # type: ignore[import]
        return TavilyClient(api_key=settings.tavily_api_key)
    except ImportError:
        logger.error("tavily-python is not installed. Run: pip install tavily-python")
        return None
    except Exception as exc:
        logger.exception("Unexpected error constructing TavilyClient: %s", exc)
        return None


# Module-level cached client (None until first call)
_client = None


def _get_client():
    """
    Return the cached Tavily client, building it on first call.

    Returns:
        TavilyClient instance or None.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


def web_search(
    query: str,
    max_results: int | None = None,
) -> list[dict[str, Any]]:
    """
    Search the live web using the Tavily Search API.

    Performs a Tavily ``search`` call and normalises the results into a
    consistent list of dicts.  If Tavily is misconfigured, rate-limited,
    or unreachable the function logs the error and returns a single
    fallback result so the generator node always has *something* to work
    with.

    Args:
        query: Natural language search query.
        max_results: Maximum number of results to return.  Defaults to
                     ``settings.max_web_results``.

    Returns:
        List of dicts, each with keys:
        ``title``, ``url``, ``content``, ``score``.
        Always returns at least one item (the fallback on failure).
    """
    k = max_results or settings.max_web_results
    client = _get_client()

    if client is None:
        return [_fallback_result(query, reason="Tavily API key not configured")]

    logger.info("Tavily web search: query='%.80s', max_results=%d", query, k)

    try:
        response = client.search(
            query=query,
            max_results=k,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        )

        raw_results: list[dict[str, Any]] = response.get("results", [])

        if not raw_results:
            logger.warning("Tavily returned 0 results for query: %.80s", query)
            return [_fallback_result(query, reason="No results found")]

        normalised = [
            WebSearchResult(
                title=r.get("title", "Untitled"),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=float(r.get("score", 0.0)),
            ).to_dict()
            for r in raw_results
        ]

        logger.info("Tavily returned %d results.", len(normalised))
        return normalised

    except Exception as exc:
        # Catch-all: network errors, rate limits, auth failures, etc.
        logger.exception("Tavily search failed for query '%.80s': %s", query, exc)
        return [_fallback_result(query, reason=f"Search error: {type(exc).__name__}")]


def _fallback_result(query: str, reason: str = "Search unavailable") -> dict[str, Any]:
    """
    Construct a sentinel fallback result when Tavily cannot be reached.

    The generator node treats this as context and falls back to direct LLM
    knowledge, making the failure transparent to the user.

    Args:
        query: Original search query (included for traceability).
        reason: Human-readable explanation of why search failed.

    Returns:
        Single result dict with ``is_fallback=True`` flag.
    """
    logger.warning("Returning fallback search result. Reason: %s", reason)
    return {
        "title": "Web Search Unavailable",
        "url": "",
        "content": (
            f"Live web search could not be completed ({reason}). "
            "The answer below is based on the model's training knowledge only."
        ),
        "score": 0.0,
        "is_fallback": True,
    }
