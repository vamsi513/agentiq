"""Per agent instrumentation for the multi agent graph.

Every agent node is wrapped by ``agent_step``, which does four things on each
entry and exit:

1. Charges one step against the run budget, so the existing per run step,
   token and wall clock limits cover the multi agent path with no change to
   agent/budgets.py.
2. Times the call and records it on the existing node metrics.
3. Emits one audit record per agent through observability.audit, carrying a
   bounded, redacted summary of what went in and what came out plus the
   tokens the provider reported.
4. Appends the same summary to state under ``agent_trace`` so an evaluation
   run can read per agent cost without parsing logs.

Nothing here raises on its own. Instrumentation that can break a run is worse
than no instrumentation, so every failure path falls back to returning the
node's own result. BudgetExceeded is the deliberate exception: it is how a run
stops when it is out of budget and it must keep propagating.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# An agent's input and output summaries are capped so one long draft cannot
# flood the audit log. The full text stays on state; only the record is bounded.
_MAX_TEXT_CHARS = 200


def _redacted_head(value: Any) -> dict[str, Any] | Any:
    """Bounded, redacted description of a value for an audit record."""
    from agent.redaction import redact

    if value is None:
        return None
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        cleaned = redact(value)
        if len(cleaned) <= _MAX_TEXT_CHARS:
            return cleaned
        return {"chars": len(cleaned), "head": cleaned[:_MAX_TEXT_CHARS]}
    if isinstance(value, dict):
        return {"type": "dict", "keys": len(value)}
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "items": len(value)}
    return {"type": type(value).__name__}


def _input_summary(state: dict[str, Any]) -> dict[str, Any]:
    """What this agent could see, without copying the whole state."""
    plan = state.get("plan") or []
    return {
        "query": _redacted_head(state.get("query", "")),
        "plan_steps": len(plan),
        "doc_evidence": len(state.get("doc_evidence") or []),
        "web_evidence": len(state.get("web_evidence") or []),
        "draft_chars": len(state.get("draft") or ""),
        "retries_used": state.get("retries_used", 0),
    }


def _output_summary(update: Any) -> dict[str, Any]:
    """What this agent wrote, keyed by the state fields it touched."""
    if not isinstance(update, dict):
        return {"type": type(update).__name__}
    summary: dict[str, Any] = {"keys": sorted(k for k in update if k != "agent_trace")}
    for key in ("plan", "doc_evidence", "web_evidence", "verdicts"):
        if key in update:
            value = update[key]
            summary[key] = len(value) if isinstance(value, list) else _redacted_head(value)
    for key in ("draft", "plan_summary"):
        if key in update:
            summary[key] = _redacted_head(update[key])
    for key in ("unsupported_count", "verified", "phase", "stop_reason"):
        if key in update:
            summary[key] = update[key]
    return summary


_TOKEN_TOTALS: dict[int, int] = {}


def record_tokens(total: int) -> None:
    """Attribute provider-reported tokens to the agent currently running.

    Charges the run budget as well, so a multi agent run is bounded by the
    same BUDGET_MAX_TOKENS as a single agent one. Keyed by the running task or
    thread so two agents in the same parallel superstep do not pool their
    counts.
    """
    if not isinstance(total, int) or total <= 0:
        return
    from agent.budgets import charge_tokens

    charge_tokens(total)
    _TOKEN_TOTALS[_scope_key()] = _TOKEN_TOTALS.get(_scope_key(), 0) + total


def charge_response(response: Any) -> int:
    """Record usage from a model response and return the token count.

    Reads usage_metadata, which langchain-core fills in when the provider
    reports it. Missing or malformed usage counts as zero rather than being
    estimated, so the numbers in an evaluation only ever come from the
    provider.
    """
    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0
    total = usage.get("total_tokens")
    if not isinstance(total, int):
        return 0
    record_tokens(total)
    return total


def _scope_key() -> int:
    """Identity of the current async task, falling back to the thread.

    LangGraph runs sync nodes in a thread pool and async nodes as tasks, so
    neither alone identifies a running agent across both kinds.
    """
    try:
        task = asyncio.current_task()
        if task is not None:
            return id(task)
    except RuntimeError:
        pass
    import threading

    return threading.get_ident()


def _begin_scope() -> int:
    key = _scope_key()
    _TOKEN_TOTALS[key] = 0
    return key


def _end_scope(key: int) -> int:
    return _TOKEN_TOTALS.pop(key, 0)


def _emit(agent: str, state: dict[str, Any], update: Any, started: float,
          tokens: int, error: BaseException | None) -> dict[str, Any]:
    """Build one per agent record, write it to the audit log, return it."""
    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    record: dict[str, Any] = {
        "event": "agent_step",
        "agent": agent,
        "input": _input_summary(state),
        "output": _output_summary(update),
        "tokens": tokens,
        "duration_ms": duration_ms,
        "outcome": "error" if error is not None else "ok",
    }
    if error is not None:
        from agent.redaction import redact

        record["error_type"] = type(error).__name__
        record["error"] = redact(str(error))[:_MAX_TEXT_CHARS]

    try:
        from observability.audit import emit
        from observability.audit_sinks import current_run_id, utc_now_iso

        emit({**record, "ts": utc_now_iso(), "run_id": current_run_id()})
    except Exception:  # pragma: no cover - auditing must never break a run
        logger.warning("agent step record could not be written", exc_info=True)

    logger.info(
        "agent_step agent=%s duration_ms=%.1f tokens=%d outcome=%s",
        agent, duration_ms, tokens, record["outcome"],
    )
    try:
        from observability.metrics import record_node

        record_node(f"multi.{agent}", duration_ms / 1000.0)
    except Exception:  # pragma: no cover - metrics are best effort
        pass
    return record


def agent_step(agent: str) -> Callable:
    """Wrap one agent node with budget, timing, audit and trace.

    Handles sync and async nodes. The wrapped node's own return value is
    passed through with ``agent_trace`` appended, so a node stays a plain
    function of state and knows nothing about instrumentation.
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(state, *args, **kwargs):
                from agent.budgets import charge_step

                charge_step(f"multi.{agent}")
                started = time.perf_counter()
                key = _begin_scope()
                try:
                    update = await fn(state, *args, **kwargs)
                except BaseException as exc:
                    _emit(agent, state, None, started, _end_scope(key), exc)
                    raise
                tokens = _end_scope(key)
                record = _emit(agent, state, update, started, tokens, None)
                return _with_trace(update, record)

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            from agent.budgets import charge_step

            charge_step(f"multi.{agent}")
            started = time.perf_counter()
            key = _begin_scope()
            try:
                update = fn(state, *args, **kwargs)
            except BaseException as exc:
                _emit(agent, state, None, started, _end_scope(key), exc)
                raise
            tokens = _end_scope(key)
            record = _emit(agent, state, update, started, tokens, None)
            return _with_trace(update, record)

        return wrapper

    return decorator


def _with_trace(update: Any, record: dict[str, Any]) -> Any:
    """Append the per agent record to the update's agent_trace list."""
    if not isinstance(update, dict):
        return update
    existing = update.get("agent_trace") or []
    return {**update, "agent_trace": [*existing, record]}
