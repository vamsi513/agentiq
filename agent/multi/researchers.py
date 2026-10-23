"""Researcher agents.

Two researchers, one per source. Each owns a different tool, a different
system prompt and a different slice of state:

    document researcher  ->  retrieval.vectorstore  ->  doc_evidence
    web researcher       ->  tools.web_search       ->  web_evidence

They are written so the two can run in the same LangGraph superstep. Neither
reads the other's key and neither writes it, so there is no shared field to
race on and no reducer is needed to merge them.

Each researcher does two things per sub question: it calls its tool to gather
passages, and it calls a model with its own prompt to state what those
passages say. The passages are the evidence of record, carrying stable ids
that the writer cites and the verifier checks against. The model's summary is
a finding, recorded for the trace and shown to the writer as orientation, but
never citable. Keeping verification anchored on raw passages rather than on a
researcher's paraphrase is what stops a summarisation error from being
laundered into a supported claim.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.multi.prompts import DOC_RESEARCHER_SYSTEM, WEB_RESEARCHER_SYSTEM
from agent.multi.state import (
    SOURCE_DOCUMENTS,
    SOURCE_WEB,
    EvidenceItem,
    MultiAgentState,
)
from agent.multi.trace import agent_step, charge_response

logger = logging.getLogger(__name__)

# Passages kept per sub question. Matches the single agent retriever's own
# top 3, so a like for like comparison is not won or lost on context size.
_PASSAGES_PER_SUB_QUESTION = 3

# Evidence content is trimmed before it reaches a prompt. Long raw passages
# push cost up without improving what the writer can cite.
_MAX_EVIDENCE_CHARS = 700


def _sub_questions_for(state: MultiAgentState, source: str) -> list[str]:
    """Questions this researcher should work on this round.

    A retry request overrides the plan: when the verifier asked this
    researcher for one specific thing, that is the only question worth
    spending a call on.
    """
    retry = state.get("retry_request")
    if retry and retry.get("source") == source:
        return [retry["question"]]
    return [
        step["question"]
        for step in (state.get("plan") or [])
        if step.get("source") == source and step.get("question")
    ]


def _format_passages(items: list[EvidenceItem]) -> str:
    """Number passages for the researcher's own prompt."""
    return "\n\n".join(
        f"[{i}] {item.get('title', '')}\n{item.get('content', '')}"
        for i, item in enumerate(items, start=1)
    )


def _summarise(system_prompt: str, question: str, items: list[EvidenceItem]) -> str:
    """Ask the researcher's model what its passages say about the question."""
    if not items:
        return ""
    try:
        from agent.nodes import _get_llm

        response = _get_llm().invoke(
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Sub question: {question}\n\n"
                        f"Passages:\n{_format_passages(items)}"
                    ),
                },
            ]
        )
        charge_response(response)
        return str(response.content).strip()
    except Exception:
        logger.exception("Researcher summary call failed.")
        return ""


def _next_index(state: MultiAgentState, key: str) -> int:
    """Where this round's evidence ids should start.

    Ids stay unique across a retry round because numbering continues from the
    evidence already on state rather than restarting at one.
    """
    return len(state.get(key) or [])


# -- Document researcher ----------------------------------------------------

@agent_step("doc_researcher")
def doc_researcher_node(state: MultiAgentState) -> dict[str, Any]:
    """Search local documents for every document-backed sub question.

    Returns:
        Partial state updating only ``doc_evidence`` and ``metadata``.
    """
    questions = _sub_questions_for(state, SOURCE_DOCUMENTS)
    if not questions:
        return {
            "doc_evidence": list(state.get("doc_evidence") or []),
            "doc_findings": list(state.get("doc_findings") or []),
        }

    session_id = state.get("session_id")
    existing = list(state.get("doc_evidence") or [])
    start = _next_index(state, "doc_evidence")
    gathered: list[EvidenceItem] = []
    findings: list[str] = []

    for question in questions:
        try:
            from retrieval.vectorstore import query_vectorstore

            results = query_vectorstore(
                question, top_k=_PASSAGES_PER_SUB_QUESTION, session_id=session_id
            )
        except Exception:
            logger.exception("Document retrieval failed for a sub question.")
            results = []

        items: list[EvidenceItem] = []
        for result in results[:_PASSAGES_PER_SUB_QUESTION]:
            items.append(
                {
                    "id": f"E{start + len(gathered) + len(items) + 1}",
                    "source": SOURCE_DOCUMENTS,
                    "title": result.get("title", ""),
                    "url": result.get("source", ""),
                    "content": str(result.get("content", ""))[:_MAX_EVIDENCE_CHARS],
                    "score": float(result.get("score", 0.0)),
                    "sub_question": question,
                }
            )
        gathered.extend(items)
        finding = _summarise(DOC_RESEARCHER_SYSTEM, question, items)
        if finding:
            findings.append(f"{question}\n{finding}")

    logger.info(
        "Document researcher gathered %d passage(s) across %d sub question(s).",
        len(gathered), len(questions),
    )
    return {
        "doc_evidence": existing + gathered,
        "doc_findings": list(state.get("doc_findings") or []) + findings,
    }


# -- Web researcher ---------------------------------------------------------

@agent_step("web_researcher")
async def web_researcher_node(state: MultiAgentState) -> dict[str, Any]:
    """Search the web for every web-backed sub question.

    Async because tools.web_search.web_search is async, so a client side
    timeout genuinely cancels the request in flight. LangGraph runs sync and
    async nodes in the same graph, so this needs no special wiring.

    Returns:
        Partial state updating only ``web_evidence`` and ``metadata``.
    """
    questions = _sub_questions_for(state, SOURCE_WEB)
    if not questions:
        return {
            "web_evidence": list(state.get("web_evidence") or []),
            "web_findings": list(state.get("web_findings") or []),
        }

    existing = list(state.get("web_evidence") or [])
    start = _next_index(state, "web_evidence")
    gathered: list[EvidenceItem] = []
    findings: list[str] = []

    for question in questions:
        try:
            from tools.web_search import web_search

            results = await web_search(question)
        except Exception:
            logger.exception("Web search failed for a sub question.")
            results = []

        items: list[EvidenceItem] = []
        for result in results[:_PASSAGES_PER_SUB_QUESTION]:
            # web_search returns a fallback record instead of raising when the
            # provider is unreachable. It carries no retrievable content, so it
            # must not become citable evidence.
            if result.get("is_fallback"):
                continue
            items.append(
                {
                    "id": f"W{start + len(gathered) + len(items) + 1}",
                    "source": SOURCE_WEB,
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": str(result.get("content", ""))[:_MAX_EVIDENCE_CHARS],
                    "score": float(result.get("score", 0.0) or 0.0),
                    "sub_question": question,
                }
            )
        gathered.extend(items)
        finding = _summarise(WEB_RESEARCHER_SYSTEM, question, items)
        if finding:
            findings.append(f"{question}\n{finding}")

    logger.info(
        "Web researcher gathered %d result(s) across %d sub question(s).",
        len(gathered), len(questions),
    )
    return {
        "web_evidence": existing + gathered,
        "web_findings": list(state.get("web_findings") or []) + findings,
    }
