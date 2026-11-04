"""
Checkpointer initialisation.

The failure case here is the one that matters: CHECKPOINT_DSN is set, so the
operator has asked for durable checkpoints, and Postgres cannot be reached.
That must fail loudly rather than quietly degrade to MemorySaver.

No database is involved. The Postgres saver and the connection pool are
replaced with fakes injected into sys.modules, so these run offline and in CI.
"""

import sys
import types

import pytest
from langgraph.checkpoint.memory import MemorySaver

import agent.memory as memory
from agent.memory import CheckpointerInitError, init_checkpointer
from config import settings


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Each test gets a fresh process-wide checkpointer."""
    memory._checkpointer = None
    memory._pg_pool = None
    yield
    memory._checkpointer = None
    memory._pg_pool = None


class _FakePool:
    """Stands in for AsyncConnectionPool. Records whether it was closed."""

    def __init__(self, *, fail_on_open=False, **_kwargs):
        self.fail_on_open = fail_on_open
        self.opened = False
        self.closed = False

    async def open(self):
        if self.fail_on_open:
            raise OSError("connection refused")
        self.opened = True

    async def close(self):
        self.closed = True


def _install_fakes(monkeypatch, *, pool_factory, saver_setup):
    """Put fake psycopg_pool and AsyncPostgresSaver modules in sys.modules.

    init_checkpointer imports both lazily inside the function, so replacing the
    modules is enough and nothing real is ever imported.
    """
    pool_mod = types.ModuleType("psycopg_pool")
    pool_mod.AsyncConnectionPool = pool_factory
    monkeypatch.setitem(sys.modules, "psycopg_pool", pool_mod)

    class _FakeSaver:
        def __init__(self, pool):
            self.pool = pool

        async def setup(self):
            await saver_setup()

    aio_mod = types.ModuleType("langgraph.checkpoint.postgres.aio")
    aio_mod.AsyncPostgresSaver = _FakeSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres.aio", aio_mod)
    return _FakeSaver


async def _ok():
    return None


@pytest.mark.asyncio
async def test_no_dsn_uses_memory_saver(monkeypatch):
    """The default path is unchanged: no DSN, no error, MemorySaver."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "")
    checkpointer = await init_checkpointer()
    assert isinstance(checkpointer, MemorySaver)


@pytest.mark.asyncio
async def test_dsn_set_and_setup_fails_raises(monkeypatch):
    """A failing setup() with a DSN configured must raise, not fall back."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "postgresql://nope/agentiq")

    async def _boom():
        raise RuntimeError("relation already exists, in an incompatible shape")

    _install_fakes(monkeypatch, pool_factory=_FakePool, saver_setup=_boom)

    with pytest.raises(CheckpointerInitError) as excinfo:
        await init_checkpointer()

    message = str(excinfo.value)
    assert "CHECKPOINT_DSN" in message
    assert "unset CHECKPOINT_DSN" in message
    assert isinstance(excinfo.value.__cause__, RuntimeError)


@pytest.mark.asyncio
async def test_dsn_set_and_pool_open_fails_raises(monkeypatch):
    """The same applies when the database is simply unreachable."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "postgresql://nope/agentiq")
    _install_fakes(
        monkeypatch,
        pool_factory=lambda **kw: _FakePool(fail_on_open=True, **kw),
        saver_setup=_ok,
    )

    with pytest.raises(CheckpointerInitError):
        await init_checkpointer()


@pytest.mark.asyncio
async def test_failed_init_does_not_leave_a_memory_saver_behind(monkeypatch):
    """After a failure the singleton stays unset, so nothing silently serves
    in-process checkpoints to a caller that asked for durable ones."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "postgresql://nope/agentiq")

    async def _boom():
        raise RuntimeError("no")

    _install_fakes(monkeypatch, pool_factory=_FakePool, saver_setup=_boom)

    with pytest.raises(CheckpointerInitError):
        await init_checkpointer()

    assert memory._checkpointer is None
    assert memory._pg_pool is None


@pytest.mark.asyncio
async def test_failed_init_closes_the_pool_it_opened(monkeypatch):
    """A failure after the pool opened must not leak the pool."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "postgresql://nope/agentiq")
    created = []

    def _factory(**kwargs):
        pool = _FakePool(**kwargs)
        created.append(pool)
        return pool

    async def _boom():
        raise RuntimeError("setup failed after the pool opened")

    _install_fakes(monkeypatch, pool_factory=_factory, saver_setup=_boom)

    with pytest.raises(CheckpointerInitError):
        await init_checkpointer()

    assert created, "the pool was never constructed"
    assert created[0].opened is True
    assert created[0].closed is True


@pytest.mark.asyncio
async def test_dsn_set_and_setup_succeeds_uses_postgres_saver(monkeypatch):
    """The success path still returns the Postgres saver, not MemorySaver."""
    monkeypatch.setattr(settings, "checkpoint_dsn", "postgresql://ok/agentiq")
    fake_saver_cls = _install_fakes(
        monkeypatch, pool_factory=_FakePool, saver_setup=_ok
    )

    checkpointer = await init_checkpointer()
    assert isinstance(checkpointer, fake_saver_cls)
    assert not isinstance(checkpointer, MemorySaver)
