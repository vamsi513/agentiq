"""
agent/cache.py — optional Redis-backed response cache.

When REDIS_URL is set, an identical query on a *fresh* session (no
client-supplied session_id, so no prior conversation context to honour)
is served from Redis instead of re-running the whole graph. Cache entries
carry a TTL (CACHE_TTL_SECONDS, default 1h).

Not cached: multi-turn requests (the client passed a session_id), errors,
or blocked/refused queries -- caching those would either leak the wrong
context or pin a transient failure.

With REDIS_URL unset, get_cache() returns None and every call path is a
plain no-op, so the deployed services are unaffected.
"""

import hashlib
import json
import logging

from config import settings

logger = logging.getLogger(__name__)

_CACHE_PREFIX = "agentiq:answer:"
_client = None
_init_attempted = False


def get_cache():
    """Return a connected Redis client, or None if caching is not configured
    or Redis is unreachable. The connection is checked once with PING."""
    global _client, _init_attempted
    if _init_attempted:
        return _client
    _init_attempted = True

    if not settings.redis_url:
        return None

    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url, socket_timeout=2, decode_responses=True)
        client.ping()
        _client = client
        logger.info("Redis response cache connected.")
    except Exception as exc:  # pragma: no cover - depends on live Redis
        logger.warning("REDIS_URL is set but Redis is unreachable (%s: %s); caching disabled.",
                       type(exc).__name__, exc)
        _client = None
    return _client


def _key(query: str) -> str:
    normalized = " ".join(query.lower().split())
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{_CACHE_PREFIX}{digest}"


def get_cached(query: str) -> dict | None:
    """Return the cached result dict for this query, or None on miss."""
    client = get_cache()
    if client is None:
        return None
    try:
        raw = client.get(_key(query))
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis GET failed (%s); treating as cache miss.", exc)
        return None
    if raw is None:
        _record("miss")
        return None
    try:
        result = json.loads(raw)
        _record("hit")
        return result
    except json.JSONDecodeError:
        return None


def _record(result: str) -> None:
    try:
        from observability.metrics import record_cache_event
        record_cache_event(result)
    except Exception:  # pragma: no cover - metrics are best-effort
        pass


def set_cached(query: str, result: dict) -> None:
    """Store a successful result dict under this query's key with a TTL."""
    client = get_cache()
    if client is None:
        return
    try:
        client.setex(_key(query), settings.cache_ttl_seconds, json.dumps(result))
    except Exception as exc:  # pragma: no cover
        logger.warning("Redis SETEX failed (%s); result not cached.", exc)
