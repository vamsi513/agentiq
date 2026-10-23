"""State for the multi agent graph.

MultiAgentState extends the single agent AgentState with the fields the
planner, researchers, verifier and writer need. It is a separate TypedDict so
the single agent path keeps the exact state schema it has today.

Two rules shape the field layout:

1. Each researcher writes only its own evidence key. The document researcher
   writes doc_evidence and the web researcher writes web_evidence, so the two
   can run in the same superstep without a reducer and without either one
   being able to overwrite the other's work.
2. Everything the supervisor needs to pick the next step lives on the state.
   The supervisor is a plain function of state, so a run can be replayed from
   a checkpoint and take the same path.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal

from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

# Which source a sub question needs. The planner may only emit these.
SOURCE_DOCUMENTS = "documents"
SOURCE_WEB = "web"
SOURCE_NONE = "none"
ALLOWED_SOURCES: frozenset[str] = frozenset({SOURCE_DOCUMENTS, SOURCE_WEB, SOURCE_NONE})

# Phases the supervisor can dispatch to. Mirrors the node names in graph.py so
# a typo cannot route to a node that does not exist.
PHASE_RESEARCH = "research"
PHASE_WRITE = "write"
PHASE_VERIFY = "verify"
PHASE_DONE = "done"


def sanitize_source(raw: object) -> str:
    """Coerce a planner-emitted source name to an allowed one.

    Same posture as agent.state.sanitize_route: exact match wins, a single
    unambiguous substring hit is accepted, anything else falls back to the
    safe default. The default here is "none" rather than a tool, so a garbled
    or injected plan line cannot cause a tool call that was not asked for.
    """
    if isinstance(raw, str):
        text = raw.strip().lower()
        if text in ALLOWED_SOURCES:
            return text
        hits = {name for name in ALLOWED_SOURCES if name in text}
        if len(hits) == 1:
            return next(iter(hits))
    return SOURCE_NONE


class SubQuestion(TypedDict):
    """One step of the planner's plan."""

    id: str
    question: str
    source: str


class EvidenceItem(TypedDict, total=False):
    """One retrieved passage, tagged with where it came from.

    ``id`` is stable within a run so the verifier can cite a specific passage
    by reference instead of quoting it back in full.
    """

    id: str
    source: str
    title: str
    url: str
    content: str
    score: float
    sub_question: str


class ClaimVerdict(TypedDict, total=False):
    """The verifier's judgement on one claim in the draft."""

    claim: str
    supported: bool
    evidence_ids: list[str]
    reason: str


class RetryRequest(TypedDict):
    """A specific gap the verifier wants one researcher to close."""

    source: str
    question: str


class MultiAgentState(TypedDict, total=False):
    """Shared state for the multi agent graph.

    The first block mirrors AgentState so the API layer can read a finished
    multi agent run exactly the way it reads a single agent one.
    """

    # Mirrors agent.state.AgentState.
    messages: Annotated[list[BaseMessage], operator.add]
    query: str
    context: str
    sources: list[dict[str, Any]]
    route_decision: str
    session_id: str
    turn_count: int
    retrieval_score: float
    error: str | None
    metadata: dict[str, Any] | None

    # Planner output.
    plan: list[SubQuestion]
    plan_summary: str

    # Researcher output. One key per researcher, so a parallel superstep has a
    # single writer per key. Findings are the researcher's own prose summary of
    # what it found; they orient the writer but are never citable evidence.
    doc_evidence: list[EvidenceItem]
    web_evidence: list[EvidenceItem]
    doc_findings: list[str]
    web_findings: list[str]

    # Writer output. The draft is what the verifier checks; the final answer is
    # written back into messages by the writer on its last pass.
    # ``draft_version`` increments on each write and ``drafted_evidence_count``
    # records how much evidence existed at that moment, which is how the
    # supervisor tells a current draft from one that predates newer evidence.
    draft: str
    draft_version: int
    drafted_evidence_count: int

    # Verifier output. ``verified_version`` is the draft version the verdicts
    # below were produced against, so a revised draft is never reported as
    # checked on the strength of an older pass.
    verdicts: list[ClaimVerdict]
    unsupported_count: int
    retry_request: RetryRequest | None
    verified: bool
    verified_version: int

    # Supervisor bookkeeping. Written only by the supervisor node.
    phase: str
    supervisor_steps: int
    retries_used: int
    research_rounds: int
    stop_reason: str | None

    # Per agent telemetry, appended by the trace decorator.
    agent_trace: Annotated[list[dict[str, Any]], operator.add]


def all_evidence(state: MultiAgentState) -> list[EvidenceItem]:
    """Both researchers' evidence in a stable order.

    Document evidence first so a citation list reads the same way on every
    run regardless of which researcher finished first.
    """
    return list(state.get("doc_evidence") or []) + list(state.get("web_evidence") or [])


def evidence_by_id(state: MultiAgentState) -> dict[str, EvidenceItem]:
    """Index the run's evidence so a verdict can reference a passage by id."""
    return {item["id"]: item for item in all_evidence(state) if item.get("id")}


def planned_sources(state: MultiAgentState) -> set[str]:
    """The distinct real sources the plan asks for, ignoring "none"."""
    return {
        step["source"]
        for step in (state.get("plan") or [])
        if step.get("source") in (SOURCE_DOCUMENTS, SOURCE_WEB)
    }


Phase = Literal["research", "write", "verify", "done"]
