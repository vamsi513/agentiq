"""
scripts/verify_persistence_and_cache.py — Real verification for Stage 2.

1. Durable checkpoints: run one graph turn with PostgresSaver, throw away
   the in-process objects (simulating a restart), rebuild from the same
   DSN, and confirm the conversation history for that thread_id is still
   there.

2. Response cache: run the same query twice on a fresh session with
   caching on, and show a real cache-miss (full graph run) followed by a
   real cache-hit (served from Redis), with wall-clock timings.

Requires CHECKPOINT_DSN, REDIS_URL, and OPENAI_API_KEY to be set. Makes a
small number of real OpenAI calls.

Usage:
    CHECKPOINT_DSN=postgresql://localhost/agentiq_checkpoints \\
    REDIS_URL=redis://localhost:6379/1 \\
    python -m scripts.verify_persistence_and_cache
"""

import asyncio
import time

from langchain_core.messages import HumanMessage

from config import settings


async def _one_turn(graph, thread_id: str, text: str) -> str:
    from agent.memory import get_thread_config

    state = {
        "messages": [HumanMessage(content=text)],
        "query": text,
        "context": "",
        "sources": [],
        "route_decision": "direct",
        "session_id": thread_id,
        "turn_count": 0,
        "retrieval_score": 0.0,
        "error": None,
        "metadata": {},
    }
    result = await graph.ainvoke(state, config=get_thread_config(thread_id))
    return result["messages"][-1].content


async def _reset_singletons() -> None:
    """Simulate a process restart: close the Postgres pool and drop the
    cached graph + checkpointer so the next init rebuilds everything from
    scratch, reconnecting to the same database."""
    import agent.graph
    from agent.memory import close_checkpointer, init_checkpointer

    await close_checkpointer()
    agent.graph._graph = None
    await init_checkpointer()


async def check_persistence() -> None:
    print("=" * 70)
    print("1. DURABLE CHECKPOINTS (PostgresSaver survives a restart)")
    print("=" * 70)
    assert settings.checkpoint_dsn, "CHECKPOINT_DSN must be set"

    from agent.graph import get_graph
    from agent.memory import get_thread_config, init_checkpointer

    thread_id = f"persist-check-{int(time.time())}"

    await init_checkpointer()
    graph = get_graph()
    a1 = await _one_turn(graph, thread_id, "My favourite colour is teal. Remember that.")
    print(f"  turn 1 answer: {a1[:80]!r}")

    # --- simulate a full process restart ---
    print("  ... simulating process restart (dropping all in-process state) ...")
    await _reset_singletons()

    graph2 = get_graph()
    snapshot = await graph2.aget_state(get_thread_config(thread_id))
    stored_msgs = snapshot.values.get("messages", []) if snapshot and snapshot.values else []
    print(f"  after restart, checkpoint for {thread_id} has {len(stored_msgs)} messages")
    for m in stored_msgs:
        print(f"    - {type(m).__name__}: {m.content[:70]!r}")

    a2 = await _one_turn(graph2, thread_id, "What is my favourite colour?")
    print(f"  turn 2 answer (after restart): {a2[:120]!r}")
    ok = "teal" in a2.lower()
    print(f"  --> model recalled the pre-restart fact: {ok}")
    assert stored_msgs, "checkpoint did not survive the restart"
    assert ok, "history survived but the model did not use it"


async def check_cache() -> None:
    print()
    print("=" * 70)
    print("2. RESPONSE CACHE (Redis cache-miss then cache-hit)")
    print("=" * 70)
    assert settings.redis_url, "REDIS_URL must be set"

    from agent.cache import get_cache
    from api.streaming import run_agent_sync

    client = get_cache()
    assert client is not None, "Redis cache did not connect"
    query = f"What is the capital of France? (cache probe {int(time.time())})"

    # clear any stale entry for a clean first-run
    from agent.cache import _key
    client.delete(_key(query))

    t0 = time.perf_counter()
    r1 = await run_agent_sync(query, "cache-sess-1", allow_cache=True)
    dt1 = (time.perf_counter() - t0) * 1000
    print(f"  call 1: cached={r1['cached']}  {dt1:.0f}ms  answer={r1['answer'][:60]!r}")

    t0 = time.perf_counter()
    r2 = await run_agent_sync(query, "cache-sess-2", allow_cache=True)
    dt2 = (time.perf_counter() - t0) * 1000
    print(f"  call 2: cached={r2['cached']}  {dt2:.0f}ms  answer={r2['answer'][:60]!r}")

    ttl = client.ttl(_key(query))
    print(f"  redis TTL on the cache key: {ttl}s")
    assert r1["cached"] is False, "first call should be a cache miss"
    assert r2["cached"] is True, "second call should be a cache hit"
    assert r2["answer"] == r1["answer"], "cached answer differs from the original"
    print(f"  --> cache hit was {dt1 / max(dt2, 0.001):.0f}x faster than the miss")


async def main() -> None:
    await check_persistence()
    await check_cache()
    print()
    print("ALL STAGE 2 CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
