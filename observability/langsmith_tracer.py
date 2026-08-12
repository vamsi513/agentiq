"""
observability/langsmith_tracer.py — LangSmith tracing integration for AgentIQ.

Enables end-to-end observability of every LangGraph run: inputs, outputs,
latencies, token usage, and intermediate node transitions are streamed to
the LangSmith platform in real time.

Configuration (add to .env):
    LANGCHAIN_TRACING_V2=true
    LANGCHAIN_API_KEY=<your-langsmith-api-key>
    LANGCHAIN_PROJECT=agentiq            # LangSmith project name
    LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

All configuration is read from environment variables by the langchain SDK
automatically when LANGCHAIN_TRACING_V2=true is set.  This module provides
helpers to programmatically enable/disable tracing and create traced runs.
"""

import logging
import os
from contextlib import contextmanager
from typing import Any, Generator

logger = logging.getLogger(__name__)

try:
    from langsmith import Client
    from langsmith.run_helpers import traceable
    _LANGSMITH_AVAILABLE = True
except ImportError:
    _LANGSMITH_AVAILABLE = False
    logger.warning("langsmith not installed — tracing unavailable")

    def traceable(*args, **kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn
        return decorator if args and callable(args[0]) else decorator


_client: Any = None


def _is_tracing_enabled() -> bool:
    return (
        _LANGSMITH_AVAILABLE
        and os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
        and bool(os.getenv("LANGCHAIN_API_KEY", ""))
    )


def get_langsmith_client() -> Any:
    """
    Return a singleton LangSmith client, or None if tracing is disabled.

    Returns:
        langsmith.Client instance, or None.
    """
    global _client
    if not _is_tracing_enabled():
        return None
    if _client is None:
        _client = Client()
        logger.info(
            "LangSmith client initialized — project: %s",
            os.getenv("LANGCHAIN_PROJECT", "agentiq"),
        )
    return _client


def configure_tracing() -> bool:
    """
    Enable LangSmith tracing by setting environment variables if not already set.

    Call once during application startup (e.g. in app.py or main.py).

    Returns:
        True if tracing is active, False otherwise.
    """
    if not _LANGSMITH_AVAILABLE:
        logger.warning("langsmith package not installed; skipping tracing setup")
        return False

    api_key = os.getenv("LANGCHAIN_API_KEY", "")
    if not api_key:
        logger.info("LANGCHAIN_API_KEY not set — LangSmith tracing disabled")
        return False

    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "agentiq")
    os.environ.setdefault(
        "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
    )

    logger.info(
        "LangSmith tracing enabled — project='%s' endpoint='%s'",
        os.environ["LANGCHAIN_PROJECT"],
        os.environ["LANGCHAIN_ENDPOINT"],
    )
    return True


@contextmanager
def trace_run(
    run_name: str,
    inputs: dict[str, Any],
    run_type: str = "chain",
    tags: list[str] | None = None,
) -> Generator[Any, None, None]:
    """
    Context manager that wraps a block in a named LangSmith traced run.

    Usage::

        with trace_run("retrieval", {"query": query}) as run:
            results = query_vectorstore(query)
            # outputs are logged on context exit

    Args:
        run_name:  Human-readable name shown in the LangSmith UI.
        inputs:    Dict of input values for the run.
        run_type:  LangSmith run type (chain, retriever, tool, llm, etc.)
        tags:      Optional list of tag strings for filtering.

    Yields:
        The LangSmith run object if tracing is active, else None.
    """
    client = get_langsmith_client()
    if client is None:
        yield None
        return

    run = client.create_run(
        name=run_name,
        run_type=run_type,
        inputs=inputs,
        tags=tags or ["agentiq"],
    )
    try:
        yield run
    except Exception as exc:
        if run is not None:
            client.update_run(run.id, error=str(exc))
        raise
    finally:
        if run is not None:
            client.update_run(run.id, end_time=None)


def log_feedback(
    run_id: str,
    score: float,
    comment: str = "",
    key: str = "user_rating",
) -> None:
    """
    Log human feedback for a specific traced run.

    Args:
        run_id:  LangSmith run ID (from the trace URL or run object).
        score:   Numeric score, e.g. 0.0 (bad) to 1.0 (good).
        comment: Optional text comment.
        key:     Feedback dimension name shown in the LangSmith UI.
    """
    client = get_langsmith_client()
    if client is None:
        return
    client.create_feedback(
        run_id=run_id,
        key=key,
        score=score,
        comment=comment,
    )
    logger.debug("Feedback logged for run %s: %s=%.2f", run_id, key, score)
