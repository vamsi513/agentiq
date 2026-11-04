"""Supervisor logic.

``decide_next`` is a pure function of state. Given the state after any agent
finishes, it returns the next phase. It calls no model and consults nothing
outside the state it is handed, so a run replayed from a checkpoint takes the
same path, and the whole control flow can be tested without mocking anything.

Staleness, not history, drives the decision. Two counters make that work: the
writer stamps each draft with a version and with how much evidence existed
when it wrote, and the verifier stamps which draft version it checked. The
supervisor can then tell "this draft predates the newest evidence" from "this
draft has not been checked yet" without tracking what happened in what order.

Three caps bound every run, and all three live here:

- MAX_RETRIES (1): the verifier may send work back to a researcher once. The
  second time it asks, the request is ignored and the run finishes.
- MAX_RESEARCH_ROUNDS: a ceiling on dispatches to the researchers, which is
  the only phase that repeats and the only one that costs a tool call.
- MAX_SUPERVISOR_STEPS: a hard ceiling on supervisor turns, so a combination
  of states nobody anticipated still terminates.

The caps overlap on purpose. MAX_RETRIES is the intended limit; the other two
are backstops that do not depend on the retry accounting being correct. The
longest intended path uses seven supervisor turns, so the step ceiling sits
above it and only fires when something has genuinely gone wrong.
"""

from __future__ import annotations

import logging

from agent.multi.state import (
    PHASE_DONE,
    PHASE_RESEARCH,
    PHASE_VERIFY,
    PHASE_WRITE,
    MultiAgentState,
    all_evidence,
    planned_sources,
)

logger = logging.getLogger(__name__)

MAX_RETRIES = 1
MAX_RESEARCH_ROUNDS = 2
MAX_SUPERVISOR_STEPS = 10

STOP_RETRY_BUDGET = "retry_budget_used"
STOP_STEP_LIMIT = "supervisor_step_limit"
STOP_RESEARCH_LIMIT = "research_round_limit"
STOP_VERIFIED = "verified"
STOP_UNVERIFIED = "finished_unverified"
STOP_NO_RETRY_REQUESTED = "no_retry_requested"


def decide_next(state: MultiAgentState) -> tuple[str, str | None]:
    """Pick the next phase from state.

    Returns:
        ``(phase, stop_reason)``. ``stop_reason`` is set only when the phase
        is ``done``, and names why the run ended.
    """
    steps = state.get("supervisor_steps", 0)
    rounds = state.get("research_rounds", 0)
    retries = state.get("retries_used", 0)

    # Backstop first, so a cap breach ends the run regardless of what else the
    # state says.
    if steps >= MAX_SUPERVISOR_STEPS:
        return PHASE_DONE, STOP_STEP_LIMIT

    # Nothing researched yet and the plan asks for a real source. This is the
    # only branch that may short-circuit the draft and verify checks below,
    # and it must not fire when the plan needs no lookup: a conversational
    # question leaves research_rounds at zero forever, so a branch keyed on
    # "rounds == 0" alone would keep returning write and never terminate.
    if rounds == 0 and planned_sources(state):
        return PHASE_RESEARCH, None

    draft_version = state.get("draft_version", 0)

    # No draft yet, or the draft predates evidence that arrived after it. Both
    # mean the current evidence has not been written up.
    if draft_version == 0:
        return PHASE_WRITE, None
    if len(all_evidence(state)) > state.get("drafted_evidence_count", 0):
        return PHASE_WRITE, None

    # The draft is current but has not been checked at this version.
    if state.get("verified_version", 0) < draft_version:
        return PHASE_VERIFY, None

    if state.get("verified"):
        return PHASE_DONE, STOP_VERIFIED

    # The verifier found unsupported claims, or could not score any. Act on a
    # specific request if there is budget for one more research round.
    retry = state.get("retry_request")
    if not retry:
        return PHASE_DONE, STOP_NO_RETRY_REQUESTED
    if retries >= MAX_RETRIES:
        return PHASE_DONE, STOP_RETRY_BUDGET
    if rounds >= MAX_RESEARCH_ROUNDS:
        return PHASE_DONE, STOP_RESEARCH_LIMIT

    return PHASE_RESEARCH, None


def supervisor_node(state: MultiAgentState) -> dict:
    """Record the decision on state so the conditional edge can read it.

    The node does the bookkeeping that ``decide_next`` stays free of: counting
    its own turns, counting research rounds, and spending a retry when it
    sends work back to a researcher.
    """
    steps = state.get("supervisor_steps", 0)
    rounds = state.get("research_rounds", 0)
    retries = state.get("retries_used", 0)

    phase, stop_reason = decide_next(state)

    update: dict = {
        "supervisor_steps": steps + 1,
        "phase": phase,
        "stop_reason": stop_reason,
    }

    if phase == PHASE_RESEARCH:
        update["research_rounds"] = rounds + 1
        # A second or later round only ever happens because the verifier asked
        # for it, so that is where a retry is spent.
        if rounds > 0:
            update["retries_used"] = retries + 1

    if phase == PHASE_DONE:
        if stop_reason != STOP_VERIFIED and not state.get("verified"):
            update["stop_reason"] = stop_reason or STOP_UNVERIFIED
        logger.info(
            "Multi agent run finishing: reason=%s steps=%d rounds=%d retries=%d evidence=%d",
            update["stop_reason"], steps + 1, rounds, retries, len(all_evidence(state)),
        )

    logger.debug("Supervisor step %d chose phase=%s", steps + 1, phase)
    return update
