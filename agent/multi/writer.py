"""Writer agent, and the terminal publish step.

The writer produces the answer from gathered evidence, citing evidence ids
inline. It runs once to produce the draft the verifier checks, and again only
if the verifier found unsupported claims and the supervisor granted the one
allowed retry.

Writing and publishing are deliberately separate. The writer only ever sets
``draft``; ``publish_node`` is what appends the AIMessage and increments
turn_count, and it runs exactly once, at the end. Keeping them apart means a
draft that the verifier later rejects never reaches the conversation history,
and a turn is counted once no matter how many drafts it took.

``publish_node`` also republishes ``sources``, ``context`` and
``retrieval_score`` in the same shape the single agent generator produces, so
api/streaming.py and the evaluation runner read a finished multi agent run
without knowing which mode produced it.

When there is no evidence at all the writer does not call a model. It says so
directly. That path exists because the honest answer to "no evidence was
retrieved" is a fixed sentence, not a generated paragraph that might quietly
fill the gap from model knowledge.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from agent.multi.prompts import (
    DIRECT_WRITER_SYSTEM,
    WRITER_REVISION_SUFFIX,
    WRITER_SYSTEM,
)
from agent.multi.state import MultiAgentState, all_evidence, planned_sources
from agent.multi.trace import agent_step, charge_response
from agent.multi.verifier import format_evidence

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = (
    "I could not retrieve any evidence for this question, so I cannot answer "
    "it from the available documents or from a web search."
)

WRITER_ERROR_ANSWER = (
    "I encountered an error while writing the answer. Please try again."
)


def _answer_without_evidence(query: str, state: MultiAgentState) -> str:
    """Answer a question the plan decided needed no lookup.

    Carries prior turns so a conversational exchange keeps its thread, which
    is what the single agent direct route does.
    """
    history: list[dict[str, str]] = [{"role": "system", "content": DIRECT_WRITER_SYSTEM}]
    for message in (state.get("messages") or [])[:-1]:
        if isinstance(message, HumanMessage):
            history.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            history.append({"role": "assistant", "content": message.content})
    history.append({"role": "user", "content": query})

    try:
        from agent.nodes import _get_llm

        response = _get_llm().invoke(history)
        charge_response(response)
        return str(response.content)
    except Exception:
        logger.exception("Conversational writer call failed.")
        return WRITER_ERROR_ANSWER


def _unsupported_notes(state: MultiAgentState) -> str:
    """The claims the verifier rejected, for the revision pass."""
    rejected = [v["claim"] for v in (state.get("verdicts") or []) if not v.get("supported")]
    if not rejected:
        return ""
    joined = "\n".join(f"- {claim}" for claim in rejected)
    return f"\n\nClaims marked unsupported in your previous draft:\n{joined}"


def published_sources(state: MultiAgentState) -> list[dict[str, Any]]:
    """Evidence rendered in the shape the API already returns to callers."""
    published: list[dict[str, Any]] = []
    for item in all_evidence(state):
        content = item.get("content", "")
        published.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": content[:200] + ("..." if len(content) > 200 else ""),
                "score": item.get("score", 0.0),
                "type": "retrieval" if item.get("source") == "documents" else "web_search",
                "evidence_id": item.get("id", ""),
            }
        )
    return published


def _draft_update(state: MultiAgentState, answer: str, error: str | None = None) -> dict[str, Any]:
    """Record a draft and stamp it with the state it was written from."""
    update: dict[str, Any] = {
        "draft": answer,
        # The stamp is what lets the supervisor tell this draft apart from one
        # written before newer evidence arrived, and lets the verifier's
        # verdicts be tied to the exact version they were produced against.
        "draft_version": state.get("draft_version", 0) + 1,
        "drafted_evidence_count": len(all_evidence(state)),
    }
    if error is not None:
        update["error"] = error
    return update


@agent_step("writer")
def writer_node(state: MultiAgentState) -> dict[str, Any]:
    """Write the answer from evidence.

    The supervisor decides whether this is the drafting pass or the one
    allowed revision; the writer tells them apart by whether the verifier has
    already recorded unsupported claims against the current draft.

    Returns:
        Partial state carrying the draft and its version stamp.
    """
    query = state.get("query", "")
    evidence = all_evidence(state)

    if not evidence:
        # Two different situations produce no evidence, and they call for
        # opposite answers.
        #
        # The plan asked for no lookup at all: a greeting, or a general
        # knowledge question outside the corpus. Nothing was searched because
        # nothing needed to be, so answering from model knowledge is right and
        # refusing would be obtuse. This is the single agent graph's direct
        # route, which the multi agent graph otherwise has no equivalent of.
        #
        # The plan asked for a source and the lookup came back empty: refuse.
        # Falling back to model knowledge here would quietly turn a failed
        # retrieval into a confident ungrounded answer, which is the exact
        # failure the evidence discipline exists to prevent.
        if not planned_sources(state):
            logger.info("Plan asked for no lookup, answering from model knowledge.")
            return _draft_update(state, _answer_without_evidence(query, state))
        logger.info("Lookup returned nothing, returning the fixed no-evidence answer.")
        return _draft_update(state, NO_EVIDENCE_ANSWER)

    is_revision = bool(state.get("verdicts")) and state.get("unsupported_count", 0) > 0
    system = WRITER_SYSTEM + (WRITER_REVISION_SUFFIX if is_revision else "")

    findings = list(state.get("doc_findings") or []) + list(state.get("web_findings") or [])
    orientation = (
        "\n\nResearcher notes (orientation only, do not cite these):\n"
        + "\n\n".join(findings)
        if findings
        else ""
    )

    user_content = (
        f"Question: {query}\n\n"
        f"Evidence:\n{format_evidence(state)}"
        f"{orientation}"
        f"{_unsupported_notes(state) if is_revision else ''}"
    )

    try:
        from agent.nodes import _get_llm

        response = _get_llm().invoke(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ]
        )
        charge_response(response)
        answer = str(response.content)
        logger.info(
            "Writer produced %s (%d chars) from %d evidence item(s).",
            "a revision" if is_revision else "a draft", len(answer), len(evidence),
        )
        return _draft_update(state, answer)
    except Exception as exc:
        logger.exception("Writer call failed.")
        return _draft_update(state, WRITER_ERROR_ANSWER, error=str(exc))


def publish_node(state: MultiAgentState) -> dict[str, Any]:
    """Close the turn with whatever draft survived.

    Not an agent: it calls no model, so it is not wrapped by ``agent_step``
    and costs no budget step of its own. It exists so that exactly one
    AIMessage reaches the conversation history per turn.
    """
    answer = state.get("draft") or NO_EVIDENCE_ANSWER
    evidence = all_evidence(state)
    scores = [float(item.get("score", 0.0) or 0.0) for item in evidence]
    return {
        "messages": [AIMessage(content=answer)],
        "turn_count": state.get("turn_count", 0) + 1,
        "sources": published_sources(state),
        "context": format_evidence(state),
        "retrieval_score": max(scores) if scores else 0.0,
    }
