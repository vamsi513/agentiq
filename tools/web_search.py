"""
tools/web_search.py — Tavily Search API wrapper with robust error handling.

Provides a single ``web_search`` coroutine that queries Tavily for live web
results and returns a normalised list of result dicts.  All error cases are
caught and logged; the function never raises to its callers — instead it
returns a fallback payload so the generator node can still produce a response
(via direct LLM knowledge) when the search API is unavailable.

Resilience notes:
- The sync ``TavilyClient`` hardcodes a 100s per-request timeout internally
  (``requests.post(..., timeout=100)``) with no configurable override, and
  wrapping a blocking ``requests`` call in a worker thread + Future.result()
  cannot actually cancel it: a timed-out call keeps running in the
  background, consuming a thread-pool slot, until Tavily's own 100s timeout
  eventually fires on its own. That's a real resource-leak risk under
  repeated timeouts (a burst of hangs can exhaust the pool).
- ``AsyncTavilyClient`` (same package, `tavily.async_tavily`) uses
  ``httpx.AsyncClient`` instead, which respects asyncio cancellation --
  wrapping it in ``asyncio.wait_for()`` genuinely cancels the underlying
  connection when our timeout fires, rather than abandoning a thread to run
  to completion unsupervised. Verified by inspecting the installed
  tavily-python source (.venv/lib/*/site-packages/tavily/async_tavily.py)
  rather than assumed.
- Retries are bounded (_MAX_ATTEMPTS total) and apply only to genuinely
  transient failures: connection errors, timeouts, and 5xx responses.
  429 (rate limit) and auth/bad-request errors are NOT retried -- retrying
  a 429 just hammers the same limit harder, and retrying a bad API key can
  never succeed. See _is_transient_tavily_error.

Tavily free tier: 1,000 searches / month.
Get a key at: https://tavily.com
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any

import httpx
from tavily.errors import BadRequestError, InvalidAPIKeyError, UsageLimitExceededError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from config import settings

logger = logging.getLogger(__name__)

_SEARCH_TIMEOUT_SECONDS = 15.0
_MAX_ATTEMPTS = 3  # 1 initial attempt + 2 bounded retries


def _is_transient_tavily_error(exc: BaseException) -> bool:
    """
    True for errors worth retrying, false otherwise.

    Retried: our own client-side timeout (asyncio.TimeoutError, raised by
    asyncio.wait_for), httpx transport-level failures (connection resets,
    DNS failures, httpx's own internal timeouts), and 5xx responses -- all
    of these can plausibly succeed on a second attempt.

    Not retried: 429 (retrying just re-hits the same rate-limit window
    harder instead of backing off from it), invalid/missing API key, and
    malformed-request errors -- none of these are fixed by trying again.
    """
    if isinstance(exc, (TimeoutError, httpx.TransportError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response is not None and exc.response.status_code >= 500
    return False


@retry(
    retry=retry_if_exception(_is_transient_tavily_error),
    stop=stop_after_attempt(_MAX_ATTEMPTS),
    wait=wait_exponential_jitter(initial=0.5, max=4.0),
    reraise=True,
)
async def _search_with_bounded_timeout(client: Any, query: str, k: int) -> dict:
    """
    Run client.search() with a hard wall-clock timeout that actually cancels
    the in-flight request when it fires (asyncio.wait_for cancels the
    underlying asyncio task; httpx's AsyncClient closes the connection in
    response, unlike the old thread+Future.result approach which left the
    sync client's blocking call running to completion unsupervised).

    Retried per _is_transient_tavily_error / _MAX_ATTEMPTS above.
    """
    return await asyncio.wait_for(
        client.search(
            query=query,
            max_results=k,
            search_depth="advanced",
            include_answer=False,
            include_raw_content=False,
        ),
        timeout=_SEARCH_TIMEOUT_SECONDS,
    )


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
    Lazily construct an AsyncTavilyClient, returning None if the key is missing.

    Returns:
        AsyncTavilyClient instance, or None if TAVILY_API_KEY is not set.
    """
    if not settings.is_tavily_configured():
        logger.warning("TAVILY_API_KEY not set — web search is disabled.")
        return None
    try:
        from tavily import AsyncTavilyClient  # type: ignore[import]
        return AsyncTavilyClient(api_key=settings.tavily_api_key)
    except ImportError:
        logger.error("tavily-python is not installed. Run: pip install tavily-python")
        return None
    except Exception:
        logger.exception("Unexpected error constructing AsyncTavilyClient")
        return None


# Module-level cached client (None until first call)
_client = None


def _get_client():
    """
    Return the cached Tavily client, building it on first call.

    Returns:
        AsyncTavilyClient instance or None.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client


async def web_search(
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
        response = await _search_with_bounded_timeout(client, query, k)

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

    # Specific, actionable categories first -- these produce a clearer
    # fallback reason and a distinct log line for observability. The final
    # bare `except Exception` is the safety net for anything unanticipated
    # (never let a search-tool failure take down the whole agent turn).
    except UsageLimitExceededError:
        logger.warning("Tavily rate limit hit for query '%.80s' — not retrying.", query)
        return [_fallback_result(query, reason="rate limited")]
    except InvalidAPIKeyError:
        logger.error("Tavily API key is invalid — check TAVILY_API_KEY.")
        return [_fallback_result(query, reason="search provider misconfigured")]
    except BadRequestError as exc:
        logger.warning("Tavily rejected the request for query '%.80s': %s", query, exc)
        return [_fallback_result(query, reason="invalid search request")]
    except TimeoutError:
        logger.warning(
            "Tavily search timed out after %d attempt(s) for query '%.80s'.",
            _MAX_ATTEMPTS, query,
        )
        return [_fallback_result(query, reason="search timed out")]
    except httpx.TransportError:
        logger.warning("Tavily connection failed for query '%.80s'.", query)
        return [_fallback_result(query, reason="search provider unreachable")]
    except httpx.HTTPStatusError as exc:
        logger.exception("Tavily returned an HTTP error for query '%.80s'", query)
        return [_fallback_result(query, reason=f"search provider error ({exc})")]
    except Exception as exc:
        logger.exception("Tavily search failed unexpectedly for query '%.80s'", query)
        return [_fallback_result(query, reason=f"unexpected error: {type(exc).__name__}")]


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
