"""Audit log of tool calls.

One structured record per tool invocation: which tool, a summary of the
arguments, how long it took, and how it ended. Written as a single JSON line
per call so it can be shipped and queried without a parser.

Every string that goes into a record passes through agent.redaction first, so
a credential in an argument is masked before the record is built, not after.
Arguments are also summarised rather than copied: lengths and types for long
values, so a record stays small and bounded no matter what was passed in.

Off by default behind AUDIT_LOG_ENABLED. The decorator is generic, so a tool
added later is audited by wrapping it, with nothing else to change.
"""

from __future__ import annotations

import asyncio
import functools
import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger("agentiq.audit")

OUTCOME_OK = "ok"
OUTCOME_ERROR = "error"

# Arguments longer than this are recorded as a length, not a value.
_MAX_VALUE_CHARS = 120
# Hard cap on a rendered record, so one pathological call cannot flood a log.
_MAX_RECORD_CHARS = 2000


def _enabled() -> bool:
    try:
        from config import settings

        return bool(settings.audit_log_enabled)
    except Exception:  # pragma: no cover - config import cannot fail in practice
        return False


def _summarise(value: Any) -> Any:
    """Describe a value without copying it wholesale.

    Strings are redacted first, then truncated with their original length kept,
    because the length is often the useful part when debugging. Containers
    report their size rather than their contents.
    """
    from agent.redaction import redact

    if isinstance(value, str):
        cleaned = redact(value)
        if len(cleaned) <= _MAX_VALUE_CHARS:
            return cleaned
        return {"chars": len(cleaned), "head": cleaned[:_MAX_VALUE_CHARS]}
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, dict):
        return {"type": "dict", "keys": len(value)}
    if isinstance(value, (list, tuple, set)):
        return {"type": type(value).__name__, "items": len(value)}
    return {"type": type(value).__name__}


def summarise_args(args: tuple, kwargs: dict) -> dict[str, Any]:
    """Build a bounded, redacted summary of a call's arguments."""
    return {
        "positional": [_summarise(a) for a in args],
        "keyword": {k: _summarise(v) for k, v in kwargs.items()},
    }


def _outcome_of(result: Any) -> str:
    """Classify a successful return.

    web_search returns a fallback record rather than raising when the provider
    is unreachable, so a call that returned is not automatically a healthy one.
    """
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and first.get("is_fallback"):
            return "fallback"
    return OUTCOME_OK


def emit(record: dict[str, Any]) -> None:
    """Write one audit record. Never raises."""
    try:
        line = json.dumps(record, default=str, separators=(",", ":"))
        if len(line) > _MAX_RECORD_CHARS:
            line = json.dumps(
                {
                    "event": "tool_call",
                    "tool": record.get("tool"),
                    "outcome": record.get("outcome"),
                    "duration_ms": record.get("duration_ms"),
                    "truncated": True,
                },
                separators=(",", ":"),
            )
        logger.info(line)
    except Exception:  # pragma: no cover - auditing must never break a call
        logger.warning("audit record could not be written", exc_info=True)


def _record(tool: str, args: tuple, kwargs: dict, started: float,
            outcome: str, error: BaseException | None) -> dict[str, Any]:
    from agent.redaction import redact

    record: dict[str, Any] = {
        "event": "tool_call",
        "tool": tool,
        "args": summarise_args(args, kwargs),
        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
        "outcome": outcome,
    }
    if error is not None:
        record["error_type"] = type(error).__name__
        record["error"] = redact(str(error))[:_MAX_VALUE_CHARS]
    return record


def audited(tool: str) -> Callable:
    """Wrap a tool so each call emits one audit record.

    Handles sync and async callables. When auditing is disabled the wrapper
    calls straight through, so the cost is one boolean check.
    """

    def decorator(fn: Callable) -> Callable:
        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args, **kwargs):
                if not _enabled():
                    return await fn(*args, **kwargs)
                started = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                except BaseException as exc:
                    emit(_record(tool, args, kwargs, started, OUTCOME_ERROR, exc))
                    raise
                emit(_record(tool, args, kwargs, started, _outcome_of(result), None))
                return result

            return async_wrapper

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not _enabled():
                return fn(*args, **kwargs)
            started = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                emit(_record(tool, args, kwargs, started, OUTCOME_ERROR, exc))
                raise
            emit(_record(tool, args, kwargs, started, _outcome_of(result), None))
            return result

        return wrapper

    return decorator
