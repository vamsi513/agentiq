"""
agent/memory.py — checkpointer setup and conversation thread management.

LangGraph checkpoints the full graph state (messages, context, routing
decisions) keyed by (thread_id, checkpoint_id). Each user session gets a
stable thread_id so multi-turn history is preserved across invocations.

Two backends:
- MemorySaver (default): in-process, lost on restart. Fine for the demo.
- AsyncPostgresSaver: used when CHECKPOINT_DSN is set, so conversation
  state survives a process restart. The graph runs via ``ainvoke`` /
  ``astream_events``, so the *async* saver is required -- the sync
  PostgresSaver raises NotImplementedError on the async code path.

The Postgres saver needs an event loop to build (open the pool, run
``setup()``), so ``init_checkpointer()`` is awaited once at API startup
(see api/main.py's lifespan). ``get_checkpointer()`` stays sync and just
returns whatever's been initialised, defaulting to MemorySaver -- so
scripts and tests that never call ``init_checkpointer()`` still work.
"""

import logging
import uuid

from langgraph.checkpoint.memory import MemorySaver

from config import settings

logger = logging.getLogger(__name__)

# ── Process-wide singleton ───────────────────────────────────────────────────
_checkpointer = None
_pg_pool = None


async def init_checkpointer():
    """Initialise the checkpointer, awaiting the async Postgres setup when
    CHECKPOINT_DSN is configured. Idempotent. Falls back to MemorySaver if
    Postgres is configured but unreachable."""
    global _checkpointer, _pg_pool
    if _checkpointer is not None:
        return _checkpointer

    if not settings.checkpoint_dsn:
        _checkpointer = MemorySaver()
        logger.info("MemorySaver initialised (in-process checkpoints).")
        return _checkpointer

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg_pool import AsyncConnectionPool

        _pg_pool = AsyncConnectionPool(
            conninfo=settings.checkpoint_dsn,
            max_size=20,
            open=False,
            kwargs={"autocommit": True, "prepare_threshold": 0},
        )
        await _pg_pool.open()
        saver = AsyncPostgresSaver(_pg_pool)
        await saver.setup()  # idempotent CREATE TABLE IF NOT EXISTS
        _checkpointer = saver
        logger.info("AsyncPostgresSaver initialised (durable checkpoints enabled).")
    except Exception as exc:  # pragma: no cover - depends on a live DB
        logger.warning(
            "CHECKPOINT_DSN is set but the Postgres saver failed to init "
            "(%s: %s); falling back to in-process MemorySaver.",
            type(exc).__name__, exc,
        )
        _checkpointer = MemorySaver()
    return _checkpointer


def get_checkpointer():
    """Return the process-wide checkpointer. If ``init_checkpointer()`` was
    never awaited (scripts, tests), lazily create a MemorySaver."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
        logger.info("MemorySaver initialised (in-process checkpoints, lazy).")
    return _checkpointer


async def close_checkpointer() -> None:
    """Close the Postgres pool on shutdown. Safe to call unconditionally."""
    global _checkpointer, _pg_pool
    if _pg_pool is not None:
        await _pg_pool.close()
        _pg_pool = None
    _checkpointer = None


# Backwards-compatible alias -- existing callers import get_memory().
def get_memory():
    """Deprecated name for get_checkpointer(); kept so existing imports work."""
    return get_checkpointer()


def new_session_id() -> str:
    """
    Generate a new unique session / thread ID.

    Returns:
        A UUID4 string suitable for use as a LangGraph ``thread_id``.
    """
    session_id = str(uuid.uuid4())
    logger.debug("New session created: %s", session_id)
    return session_id


def get_thread_config(session_id: str) -> dict:
    """
    Build the LangGraph ``config`` dict required for checkpointed invocation.

    Pass the returned dict as the ``config`` argument to ``graph.invoke()``
    or ``graph.astream()`` to associate the run with a specific conversation
    thread.

    Args:
        session_id: Stable identifier for the conversation session.

    Returns:
        Dict with the ``configurable`` key set as LangGraph expects.
    """
    return {"configurable": {"thread_id": session_id}}
