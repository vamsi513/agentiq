"""
agent/security.py — Pre-routing input security boundary.

This is a heuristic, defense-in-depth layer that runs BEFORE the router
node, not a claim that prompt injection is "solved" -- it isn't, and no
purely input-side heuristic solves it against a sufficiently creative
attacker. What this module actually does:

- Bounds input size as a defensive backstop (the primary bound is
  ChatRequest.query's max_length=2000 in api/schemas.py; this re-checks
  in case the function is ever called from a path that bypasses that).
- Matches instruction-override phrasing via pattern matching. Near-
  unambiguous constructions ("ignore previous instructions", "forget
  your instructions", fake "<<<SYSTEM:" markers, "...then do what it
  says" indirect framing) score straight to BLOCK -- these essentially
  never occur in a genuine question. Softer signals ("reveal your
  system prompt", "act as if you have no restrictions") are one signal
  among several and combine into the score rather than blocking alone.
- Flags obfuscation signals: abnormal repeated-character padding, and
  long encoded-looking blobs (base64/hex) that could smuggle content
  past casual review.
- Flags explicit attempts to manipulate routing/tool selection directly
  (e.g. "call web_search(", "set route_decision") -- an attacker trying
  to talk their way past the classifier rather than ask a real question.

These signals combine into a score, which maps to one of three actions:

    ALLOW  — no signals, or below threshold. Proceeds normally.
    FLAG   — a plausible-but-not-conclusive signal. Logged with the
             matched reasons for review, but still processed -- this is
             a portfolio demo agent, not a system where blocking
             legitimate users on a weak heuristic match is an acceptable
             tradeoff.
    BLOCK  — signals strong or numerous enough to refuse outright. The
             caller (agent/graph.py's security_node) short-circuits to a
             fixed refusal message; no LLM or tool call is made with
             that input.

What this module is NOT: a claim that AgentIQ is "prompt-injection
proof." It's a documented first layer, and it can both over- and
under-trigger -- regex-based heuristics always can. The actual backstop
is architectural, not this module: the router and generator prompts
already keep system instructions and the user's query in separate
message roles (see agent/nodes.py's _ROUTER_SYSTEM_PROMPT /
_GENERATOR_SYSTEM_PROMPT), so content that slips past this heuristic
still can't literally rewrite the system message -- at most it can try
to talk the model into ignoring its own instructions, which model-level
alignment (not this module) is the real defense against.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)

# Mirrors api/schemas.py's ChatRequest.query max_length. Kept as a local
# constant (not imported from api.schemas) so this module has no dependency
# on the API layer and can be unit-tested / reused independently.
_MAX_QUERY_LENGTH = 2000

_BLOCK_THRESHOLD = 5


class SecurityAction(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    BLOCK = "block"


@dataclass
class SecurityDecision:
    action: SecurityAction
    reasons: list[str] = field(default_factory=list)
    score: int = 0


# Known instruction-override / jailbreak phrasing. Each match is one signal
# fed into the combined score below -- see evaluate_query. Not exhaustive,
# and deliberately not a giant list of arbitrary "suspicious words": every
# pattern targets a specific, well-known override construction.
# Strong, near-unambiguous override constructions. A single match here is
# on its own enough to BLOCK (weight == _BLOCK_THRESHOLD) -- these phrasings
# essentially never occur in a genuine question.
_STRONG_OVERRIDE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore (all |any )?(previous|prior|above|earlier) instructions",
        r"disregard (the |your |all )?(system|previous|prior) (prompt|instructions?)",
        r"forget (everything|all|your) (you|that|prior|previous|instructions)",
        r"you are (now |)(DAN|no longer (bound|restricted))",
        r"\bjailbreak\b",
        r"\bdeveloper mode\b",
        # Fake in-band system/instruction markers used to smuggle a new prompt.
        r"<<<\s*system\s*:",
        r"^\s*\[?system\]?\s*:",
        r"\[/?INST\]",
        # "translate/summarize X, then do what it says" -- indirect framing
        # that tries to launder an instruction through a benign-looking task.
        r"then (do|follow|execute) what (it|the text|that) says",
    ]
]

_OVERRIDE_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"reveal (your|the) (system )?(prompt|instructions)",
        r"(print|show|repeat) (your|the) (system )?(prompt|instructions)",
        r"repeat (everything|all|the text) (above|before|prior)",
        r"act as (if )?you (have no|have|had) restrictions",
        r"new instructions?\s*:",
        r"no (content|safety) (rules|guidelines|restrictions)",
    ]
]

# Attempts to directly manipulate routing/tool selection rather than ask a
# genuine question -- targets the exact boundary this module sits in front
# of (the router node and the deterministic tool dispatch in graph.py).
_ROUTING_MANIPULATION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\broute (this|the)?\s*(query)?\s*as\b",
        r"\bcall (web_search|retriever?|generator)\s*\(",
        r"\bset route_decision\b",
        r"\binvoke (the )?tool\b.*\bwith\b",
    ]
]

# Long base64-looking or hex-looking runs -- a crude but useful signal for
# an obfuscated/encoded payload riding along with an otherwise normal query.
_ENCODED_BLOB_PATTERN = re.compile(r"[A-Za-z0-9+/]{60,}={0,2}")
_HEX_BLOB_PATTERN = re.compile(r"(?:[0-9a-fA-F]{2}){40,}")


def _repeated_char_ratio(text: str) -> float:
    """Fraction of the string taken up by its single most-repeated
    character. A crude but effective signal for padding-based injection
    attempts (e.g. thousands of repeated characters meant to push real
    content past a naive truncation-based filter or fill a context
    window)."""
    if not text:
        return 0.0
    counts = Counter(text)
    return counts.most_common(1)[0][1] / len(text)


def evaluate_query(query: str) -> SecurityDecision:
    """
    Score a user query for injection/abuse signals before it reaches the
    router.

    This function only classifies -- it never raises and never blocks by
    itself. The caller (agent/graph.py's security_node) is responsible for
    acting on the returned SecurityDecision.

    Args:
        query: The raw user query string, pre-schema-validation semantics
               assumed (i.e. this may be called on already-validated input,
               but re-checks length defensively regardless).

    Returns:
        SecurityDecision with the action, the specific reasons matched,
        and a numeric score (useful for tuning thresholds against real
        traffic later; not itself exposed to end users).
    """
    reasons: list[str] = []
    score = 0

    if len(query) > _MAX_QUERY_LENGTH:
        reasons.append(f"query exceeds max length ({len(query)} > {_MAX_QUERY_LENGTH})")
        score += 3

    strong_hits = [p.pattern for p in _STRONG_OVERRIDE_PATTERNS if p.search(query)]
    if strong_hits:
        reasons.append(f"strong instruction-override construction matched ({len(strong_hits)} pattern(s))")
        # A single strong match blocks on its own.
        score += _BLOCK_THRESHOLD * len(strong_hits)

    override_hits = [p.pattern for p in _OVERRIDE_PATTERNS if p.search(query)]
    if override_hits:
        reasons.append(f"instruction-override phrasing matched ({len(override_hits)} pattern(s))")
        score += 2 * len(override_hits)

    routing_hits = [p.pattern for p in _ROUTING_MANIPULATION_PATTERNS if p.search(query)]
    if routing_hits:
        reasons.append(f"routing/tool-manipulation phrasing matched ({len(routing_hits)} pattern(s))")
        score += 3 * len(routing_hits)

    if len(query) >= 200 and _repeated_char_ratio(query) >= 0.4:
        reasons.append("abnormal repeated-character padding")
        score += 2

    if _ENCODED_BLOB_PATTERN.search(query) or _HEX_BLOB_PATTERN.search(query):
        reasons.append("long encoded-looking blob (possible obfuscated payload)")
        score += 1

    if score == 0:
        return SecurityDecision(action=SecurityAction.ALLOW, reasons=[], score=0)

    action = SecurityAction.BLOCK if score >= _BLOCK_THRESHOLD else SecurityAction.FLAG

    logger.warning(
        "security_decision action=%s score=%d reasons=%s query_excerpt=%.80r",
        action.value, score, reasons, query,
    )
    return SecurityDecision(action=action, reasons=reasons, score=score)
