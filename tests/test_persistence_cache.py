"""
tests/test_persistence_cache.py — Unit tests for the opt-in checkpoint
persistence and Redis response cache.

These test the config-gated fallback behaviour without a live Postgres or
Redis: with CHECKPOINT_DSN / REDIS_URL unset (the default, and what CI
has), the checkpointer is an in-process MemorySaver and every cache call
is a no-op. The real end-to-end persistence/cache proof lives in
scripts/verify_persistence_and_cache.py, which needs live infra.
"""

import importlib
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver


def _fresh_memory_module():
    import agent.memory as m
    importlib.reload(m)
    return m


class TestCheckpointerFallback:
    def test_defaults_to_memory_saver_without_dsn(self):
        with patch("config.settings.checkpoint_dsn", ""):
            m = _fresh_memory_module()
            cp = m.get_checkpointer()
            assert isinstance(cp, MemorySaver)

    def test_get_memory_alias_still_works(self):
        with patch("config.settings.checkpoint_dsn", ""):
            m = _fresh_memory_module()
            assert m.get_memory() is m.get_checkpointer()

    def test_thread_config_carries_thread_id_and_recursion_limit(self):
        m = _fresh_memory_module()
        cfg = m.get_thread_config("abc")
        assert cfg["configurable"]["thread_id"] == "abc"
        assert isinstance(cfg["recursion_limit"], int) and cfg["recursion_limit"] >= 4


class TestResponseCacheNoOp:
    def test_get_cache_returns_none_without_redis_url(self):
        with patch("config.settings.redis_url", ""):
            import agent.cache as c
            importlib.reload(c)
            assert c.get_cache() is None
            assert c.get_cached("anything") is None
            c.set_cached("anything", {"answer": "x"})  # must not raise

    def test_cache_key_normalizes_whitespace_and_case(self):
        import agent.cache as c
        importlib.reload(c)
        assert c._key("What  is  Python?") == c._key("what is python?")
        assert c._key("a") != c._key("b")
