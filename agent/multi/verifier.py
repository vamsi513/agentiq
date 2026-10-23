"""Verifier agent.

Checks each claim in the draft against the evidence the researchers gathered,
and marks it supported or unsupported. When exactly one lookup would close a
gap, it may name that lookup once; the supervisor decides whether there is
still retry budget to act on it.

The verifier never writes the answer and never calls a tool. It reads the
draft and the evidence and emits verdicts, so a failure here degrades to
"nothing was checked" rather than to a changed answer.

Parsing is strict in the same way the planner's is. A verdict line that does
not match the expected shape is dropped, a cited evidence id that does not
exist in this run is discarded, and a MISSING line naming a source that is not
on the allowlist is ignored rather than turned into a tool call.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.multi.prompts import VERIFIER_SYSTEM
from agent.multi.state import (
    SOURCE_DOCUMENTS,
    SOURCE_WEB,
    ClaimVerdict,
    MultiAgentState,
    all_evidence,
    evidence_by_id,
)
from agent.multi.trace import agent_step, charge_response

logger = logging.getLogger(__name__)

_SUPPORTED = "SUPPORTED"
_UNSUPPORTED = "UNSUPPORTED"
_MISSING = "MISSING"

# A claim shorter than this is not a claim worth scoring.
_MIN_CLAIM_CHARS = 10

# Cap on claims parsed from one draft, so a degenerate response cannot make
# the verdict list unbounded.
_MAX_CLAIMS = 20


def format_evidence(state: MultiAgentState) -> str:
    """Render the run's evidence with ids, for the verifier and the writer."""
    lines = []
    for item in all_evidence(state):
        title = item.get("title", "")
        url = item.get("url", "")
        header = f"[{item['id']}] {title}".rstrip()
        if url:
            header = f"{header} ({url})"
        lines.append(f"{header}\n{item.get('content', '')}")
    return "\n\n".join(lines)


def parse_verdicts(
    raw: str, known_ids: set[str]
) -> tuple[list[ClaimVerdict], dict[str, str] | None]:
    """Parse verdict lines and an optional MISSING line.

    Args:
        raw: The verifier model's full response.
        known_ids: Evidence ids that actually exist in this run.

    Returns:
        ``(verdicts, retry_request_or_None)``.
    """
    verdicts: list[ClaimVerdict] = []
    retry: dict[str, str] | None = None

    if not isinstance(raw, str):
        return verdicts, retry

    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3:
            continue
        label = parts[0].upper()

        if label == _MISSING and retry is None:
            source = parts[1].strip().lower()
            question = "|".join(parts[2:]).strip()
            if source in (SOURCE_DOCUMENTS, SOURCE_WEB) and len(question) >= _MIN_CLAIM_CHARS:
                retry = {"source": source, "question": question}
            continue

        if label not in (_SUPPORTED, _UNSUPPORTED) or len(verdicts) >= _MAX_CLAIMS:
            continue

        claim = "|".join(parts[2:]).strip()
        if len(claim) < _MIN_CLAIM_CHARS:
            continue

        # Keep only ids this run actually produced. A hallucinated id is the
        # single most misleading thing a verifier can emit, because it looks
        # exactly like a real citation when the rate is computed later.
        cited = [
            token.strip()
            for token in parts[1].replace(" ", ",").split(",")
            if token.strip() and token.strip() in known_ids
        ]
        supported = label == _SUPPORTED and bool(cited)
        verdicts.append(
            {
                "claim": claim,
                "supported": supported,
                "evidence_ids": cited,
                "reason": "" if supported else "no valid supporting evidence id",
            }
        )

    return verdicts, retry


@agent_step("verifier")
def verifier_node(state: MultiAgentState) -> dict[str, Any]:
    """Check the draft against the evidence.

    Returns:
        Partial state with ``verdicts``, ``unsupported_count``, ``verified``
        and an optional ``retry_request``.
    """
    draft = state.get("draft", "")
    known = set(evidence_by_id(state))
    # Every return stamps the draft version it checked, so a later revision is
    # never reported as verified on the strength of this pass.
    version = state.get("draft_version", 0)

    if not draft.strip():
        return {
            "verdicts": [],
            "unsupported_count": 0,
            "verified": False,
            "retry_request": None,
            "verified_version": version,
        }

    if not known:
        # Nothing was retrieved, so there is nothing to check against. Saying
        # "verified" here would be a lie by omission; the run is recorded as
        # unverified with no claims scored.
        logger.info("Verifier has no evidence to check against.")
        return {
            "verdicts": [],
            "unsupported_count": 0,
            "verified": False,
            "retry_request": None,
            "verified_version": version,
        }

    try:
        from agent.nodes import _get_llm

        response = _get_llm().invoke(
            [
                {"role": "system", "content": VERIFIER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Evidence:\n{format_evidence(state)}\n\n"
                        f"Draft answer:\n{draft}"
                    ),
                },
            ]
        )
        charge_response(response)
        verdicts, retry = parse_verdicts(response.content, known)
    except Exception:
        logger.exception("Verifier call failed, recording the draft as unverified.")
        return {
            "verdicts": [],
            "unsupported_count": 0,
            "verified": False,
            "retry_request": None,
            "verified_version": version,
        }

    unsupported = sum(1 for v in verdicts if not v["supported"])
    # A retry is only worth asking for when something actually failed. A
    # MISSING line alongside a clean sheet of verdicts is dropped.
    if unsupported == 0:
        retry = None

    logger.info(
        "Verifier scored %d claim(s), %d unsupported, retry_requested=%s",
        len(verdicts), unsupported, bool(retry),
    )
    return {
        "verdicts": verdicts,
        "unsupported_count": unsupported,
        "verified": bool(verdicts) and unsupported == 0,
        "retry_request": retry,
        "verified_version": version,
    }
