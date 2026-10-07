"""Per-run budget tests, including the failure path where a run is stopped.

Budgets are off by default, so the first class pins that down: with the flag
off nothing is tracked and nothing can raise. The rest of the file enables
them explicitly.
"""

import time
from types import SimpleNamespace

import pytest

from agent import budgets
from agent.budgets import (
    STEPS_EXCEEDED,
    TOKENS_EXCEEDED,
    WALL_TIME_EXCEEDED,
    Budget,
    BudgetExceeded,
    BudgetTracker,
)


@pytest.fixture(autouse=True)
def clear_tracker():
    """Every test starts and ends with no tracker on the context."""
    token = budgets.set_tracker(None)
    yield
    budgets.reset_tracker(token)


@pytest.fixture
def enabled(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "budgets_enabled", True)
    return settings


class TestDisabledByDefault:
    def test_flag_is_off_in_the_default_config(self):
        from config import settings

        assert settings.budgets_enabled is False

    def test_start_run_does_nothing_when_disabled(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "budgets_enabled", False)
        tracker, token = budgets.start_run()
        assert tracker is None
        assert token is None

    def test_charging_without_a_tracker_is_a_no_op(self):
        # No tracker on the context, so these must not raise.
        budgets.charge_step("router")
        budgets.charge_tokens(10_000_000)
        assert budgets.current_tracker() is None


class TestStepBudget:
    def test_steps_under_the_limit_pass(self):
        t = BudgetTracker(budget=Budget(max_steps=3))
        t.record_step("a")
        t.record_step("b")
        t.record_step("c")
        assert t.steps == 3

    def test_step_over_the_limit_raises_with_a_named_reason(self):
        t = BudgetTracker(budget=Budget(max_steps=2))
        t.record_step("a")
        t.record_step("b")
        with pytest.raises(BudgetExceeded) as exc:
            t.record_step("c")
        assert exc.value.reason == STEPS_EXCEEDED
        assert t.stopped_reason == STEPS_EXCEEDED

    def test_zero_means_the_limit_is_off(self):
        t = BudgetTracker(budget=Budget(max_steps=0))
        for _ in range(50):
            t.record_step("a")
        assert t.steps == 50


class TestTokenBudget:
    def test_tokens_do_not_raise_on_recording(self):
        """Usage is only known after a call returns, so recording never raises."""
        t = BudgetTracker(budget=Budget(max_tokens=10))
        t.record_tokens(999)
        assert t.tokens == 999
        assert t.stopped_reason is None

    def test_overspend_is_caught_at_the_next_step(self):
        t = BudgetTracker(budget=Budget(max_tokens=10))
        t.record_tokens(999)
        with pytest.raises(BudgetExceeded) as exc:
            t.record_step("generator")
        assert exc.value.reason == TOKENS_EXCEEDED

    def test_negative_and_zero_counts_are_ignored(self):
        t = BudgetTracker(budget=Budget(max_tokens=100))
        t.record_tokens(0)
        t.record_tokens(-5)
        assert t.tokens == 0


class TestWallTimeBudget:
    def test_elapsed_time_over_the_limit_raises(self):
        t = BudgetTracker(budget=Budget(max_wall_seconds=0.05))
        time.sleep(0.08)
        with pytest.raises(BudgetExceeded) as exc:
            t.record_step("router")
        assert exc.value.reason == WALL_TIME_EXCEEDED

    def test_fast_run_is_unaffected(self):
        t = BudgetTracker(budget=Budget(max_wall_seconds=30))
        t.record_step("router")
        assert t.stopped_reason is None


class TestStatusReport:
    def test_status_reports_counters_and_limits(self):
        t = BudgetTracker(budget=Budget(max_steps=5, max_tokens=100, max_wall_seconds=10))
        t.record_step("router")
        t.record_tokens(42)
        status = t.status()
        assert status["steps"] == 1
        assert status["tokens"] == 42
        assert status["max_steps"] == 5
        assert status["stopped_reason"] is None
        assert isinstance(status["elapsed_seconds"], float)


class TestTokenAccountingFromResponses:
    def test_usage_metadata_is_charged(self, enabled):
        from agent.nodes import _charge_response_tokens

        tracker, token = budgets.start_run()
        try:
            _charge_response_tokens(SimpleNamespace(usage_metadata={"total_tokens": 123}))
            assert tracker.tokens == 123
        finally:
            budgets.reset_tracker(token)

    @pytest.mark.parametrize(
        "response",
        [
            SimpleNamespace(),
            SimpleNamespace(usage_metadata=None),
            SimpleNamespace(usage_metadata={}),
            SimpleNamespace(usage_metadata={"total_tokens": "lots"}),
            SimpleNamespace(usage_metadata="nonsense"),
        ],
    )
    def test_missing_or_malformed_usage_is_treated_as_zero(self, enabled, response):
        from agent.nodes import _charge_response_tokens

        tracker, token = budgets.start_run()
        try:
            _charge_response_tokens(response)
            assert tracker.tokens == 0
        finally:
            budgets.reset_tracker(token)


class TestNodeDecoratorChargesSteps:
    def test_each_node_entry_consumes_a_step(self, enabled, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "budget_max_steps", 100)
        from agent.nodes import _timed

        @_timed("probe")
        def node(state):
            return {"ok": True}

        tracker, token = budgets.start_run()
        try:
            node({})
            node({})
            assert tracker.steps == 2
        finally:
            budgets.reset_tracker(token)

    def test_node_is_not_run_once_the_step_budget_is_gone(self, enabled, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "budget_max_steps", 1)
        from agent.nodes import _timed

        calls = []

        @_timed("probe")
        def node(state):
            calls.append(1)
            return {"ok": True}

        tracker, token = budgets.start_run()
        try:
            node({})
            assert calls == [1]
            with pytest.raises(BudgetExceeded):
                node({})
            # The second call was rejected before the body ran.
            assert calls == [1]
        finally:
            budgets.reset_tracker(token)


class TestRunStopsCleanly:
    """The failure path: a run that exceeds its budget returns a status."""

    @pytest.mark.asyncio
    async def test_exceeded_budget_returns_a_status_not_an_exception(
        self, enabled, monkeypatch
    ):
        from config import settings

        # One step is enough to trip on the very first node.
        monkeypatch.setattr(settings, "budget_max_steps", 1)

        from api import streaming

        monkeypatch.setattr(streaming, "get_thread_config", lambda sid: {"configurable": {"thread_id": sid}})

        class ExhaustingGraph:
            async def ainvoke(self, state, config=None):
                from agent.budgets import charge_step

                charge_step("first")
                charge_step("second")
                return {}

        monkeypatch.setattr("agent.graph.get_graph", lambda: ExhaustingGraph())

        result = await streaming.run_agent_sync("a question", "session-1", allow_cache=False)

        assert result["route_decision"] == "budget_exceeded"
        assert result["status"] == STEPS_EXCEEDED
        assert result["budget"]["stopped_reason"] == STEPS_EXCEEDED
        assert result["sources"] == []

    @pytest.mark.asyncio
    async def test_tracker_is_cleared_after_a_run(self, enabled, monkeypatch):
        from api import streaming

        monkeypatch.setattr(streaming, "get_thread_config", lambda sid: {"configurable": {"thread_id": sid}})

        class TrivialGraph:
            async def ainvoke(self, state, config=None):
                return {"messages": [], "route_decision": "direct"}

        monkeypatch.setattr("agent.graph.get_graph", lambda: TrivialGraph())
        await streaming.run_agent_sync("q", "session-2", allow_cache=False)
        assert budgets.current_tracker() is None
