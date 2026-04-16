"""
agent/memory.py — MemorySaver setup and conversation thread management.

LangGraph's MemorySaver stores the full graph state (messages, context,
routing decisions) as an in-memory checkpoint keyed by (thread_id, checkpoint_id).
Each user session gets a stable thread_id so that multi-turn conversation
history is preserved across graph invocations within the same process.

For production deployments that need persistence across restarts, swap
MemorySaver for SqliteSaver or PostgresSaver from langgraph.checkpoint.
"""

import logging
import uuid
from typing import Optional

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

# ── Singleton checkpointer ────────────────────────────────────────────────────
# One MemorySaver is shared across the entire process so all threads are
# held in the same in-memory store.
_memory_saver: Optional[MemorySaver] = None


def get_memory() -> MemorySaver:
    """
    Return the process-wide MemorySaver singleton.

    Lazy-initialised on first call.

    Returns:
        Shared MemorySaver instance used as the graph checkpointer.
    """
    global _memory_saver
    if _memory_saver is None:
        _memory_saver = MemorySaver()
        logger.info("MemorySaver initialised.")
    return _memory_saver


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

    Example::

        config = get_thread_config(session_id)
        async for chunk in graph.astream(inputs, config=config):
            ...
    """
    return {"configurable": {"thread_id": session_id}}
