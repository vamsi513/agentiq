"""Per run budgets: steps, tokens and wall clock time.

A run is given a budget when it starts. Each node entry consumes one step,
the generator reports token usage, and elapsed time is checked on every node
entry. When any limit is reached the run stops with a named reason instead of
unwinding as an unexpected error.

The tracker lives in a ContextVar rather than on AgentState. That keeps the
graph's state schema untouched, works for both sync and async nodes, and means
a run that is never given a budget behaves exactly as before.

Enforcement is off unless BUDGETS_ENABLED is true. With it off, nothing here
raises and the only cost is reading a ContextVar that is None.
"""

from __future__ import annotations

import logging
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Reasons a run can stop. Surfaced to callers as a status string.
STEPS_EXCEEDED = "max_steps_exceeded"
TOKENS_EXCEEDED = "max_tokens_exceeded"
WALL_TIME_EXCEEDED = "max_wall_time_exceeded"


class BudgetExceeded(Exception):
    """Raised at a node boundary when a run has used up its budget.

    Carries a machine readable ``reason`` so the API layer can report a clear
    status rather than a generic failure.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class Budget:
    """Limits for a single run. Zero or negative means that limit is off."""

    max_steps: int = 0
    max_tokens: int = 0
    max_wall_seconds: float = 0.0

    @classmethod
    def from_settings(cls) -> Budget:
        from config import settings

        return cls(
            max_steps=settings.budget_max_steps,
            max_tokens=settings.budget_max_tokens,
            max_wall_seconds=settings.budget_max_wall_seconds,
        )


@dataclass
class BudgetTracker:
    """Mutable counters for one run, checked at each node boundary."""

    budget: Budget
    steps: int = 0
    tokens: int = 0
    started_at: float = field(default_factory=time.monotonic)
    stopped_reason: str | None = None

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def record_step(self, stage: str) -> None:
        """Consume one step and check every limit. Raises BudgetExceeded."""
        self.steps += 1
        self._check(stage)

    def record_tokens(self, count: int) -> None:
        """Add reported token usage. Does not raise on its own.

        Token usage is only known after a model call returns, so raising here
        would discard work that has already been paid for. The overspend is
        caught at the next node boundary instead.
        """
        if count > 0:
            self.tokens += count

    def _check(self, stage: str) -> None:
        b = self.budget
        if b.max_steps > 0 and self.steps > b.max_steps:
            self._stop(
                STEPS_EXCEEDED,
                f"run used {self.steps} steps, limit is {b.max_steps} (at stage {stage})",
            )
        if b.max_tokens > 0 and self.tokens > b.max_tokens:
            self._stop(
                TOKENS_EXCEEDED,
                f"run used {self.tokens} tokens, limit is {b.max_tokens} (at stage {stage})",
            )
        if b.max_wall_seconds > 0 and self.elapsed_seconds > b.max_wall_seconds:
            self._stop(
                WALL_TIME_EXCEEDED,
                f"run ran for {self.elapsed_seconds:.1f}s, limit is {b.max_wall_seconds}s"
                f" (at stage {stage})",
            )

    def _stop(self, reason: str, detail: str) -> None:
        self.stopped_reason = reason
        logger.warning("budget_exceeded reason=%s detail=%s", reason, detail)
        raise BudgetExceeded(reason, detail)

    def status(self) -> dict[str, object]:
        """Counters and limits for this run, safe to return to a caller."""
        return {
            "steps": self.steps,
            "tokens": self.tokens,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "max_steps": self.budget.max_steps,
            "max_tokens": self.budget.max_tokens,
            "max_wall_seconds": self.budget.max_wall_seconds,
            "stopped_reason": self.stopped_reason,
        }


_current: ContextVar[BudgetTracker | None] = ContextVar("agentiq_budget", default=None)


def current_tracker() -> BudgetTracker | None:
    """Tracker for the run on this context, or None when there is no budget."""
    return _current.get()


def set_tracker(tracker: BudgetTracker | None) -> Token:
    return _current.set(tracker)


def reset_tracker(token: Token) -> None:
    _current.reset(token)


def start_run() -> tuple[BudgetTracker | None, Token | None]:
    """Begin tracking a run, or do nothing when budgets are disabled."""
    from config import settings

    if not settings.budgets_enabled:
        return None, None
    tracker = BudgetTracker(budget=Budget.from_settings())
    return tracker, set_tracker(tracker)


def charge_step(stage: str) -> None:
    """Consume a step on the current run, if there is one being tracked."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record_step(stage)


def charge_tokens(count: int) -> None:
    """Record token usage on the current run, if there is one being tracked."""
    tracker = _current.get()
    if tracker is not None:
        tracker.record_tokens(count)
