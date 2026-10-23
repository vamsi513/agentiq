"""Planner agent.

Turns the user question into a short list of sub questions, each tagged with
the source that should answer it. The plan is the only thing that decides
which researchers run, so parsing it is deliberately strict: a line that does
not match the expected shape is dropped rather than guessed at, and a source
name that is not on the allowlist becomes "none" instead of a tool call.

When the model fails or returns nothing usable, the planner falls back to a
single document-backed sub question carrying the original query. That keeps a
run useful on a bad parse instead of ending it, and it is the same behaviour
the single agent router has when its own call fails.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.multi.prompts import PLANNER_SYSTEM
from agent.multi.state import (
    SOURCE_DOCUMENTS,
    SOURCE_NONE,
    MultiAgentState,
    SubQuestion,
    sanitize_source,
)
from agent.multi.trace import agent_step, charge_response

logger = logging.getLogger(__name__)

# Upper bound on plan size. The prompt asks for at most 4; this enforces it
# regardless of what comes back, so a runaway plan cannot fan out further than
# the graph is budgeted for.
MAX_SUB_QUESTIONS = 4

# A sub question shorter than this is not a question, it is a fragment.
_MIN_QUESTION_CHARS = 8


def parse_plan(raw: str, fallback_query: str) -> list[SubQuestion]:
    """Parse the planner's line format into sub questions.

    Expected shape, one per line: ``source | question text``. Lines without a
    separator, with an empty question, or with a question too short to act on
    are skipped. The result is capped at MAX_SUB_QUESTIONS.

    Returns a single document-backed step carrying ``fallback_query`` when
    nothing parses, so a caller always gets a runnable plan.
    """
    steps: list[SubQuestion] = []
    if isinstance(raw, str):
        for line in raw.splitlines():
            line = line.strip()
            if not line or "|" not in line:
                continue
            source_raw, _, question = line.partition("|")
            question = question.strip()
            if len(question) < _MIN_QUESTION_CHARS:
                continue
            steps.append(
                {
                    "id": f"s{len(steps) + 1}",
                    "question": question,
                    "source": sanitize_source(source_raw),
                }
            )
            if len(steps) >= MAX_SUB_QUESTIONS:
                break

    if not steps:
        logger.warning("Planner produced no usable plan, falling back to one document step.")
        return [{"id": "s1", "question": fallback_query, "source": SOURCE_DOCUMENTS}]

    # A plan that is entirely "none" means no lookup is needed. That is a valid
    # outcome for a conversational question and the supervisor handles it.
    return steps


def summarise_plan(steps: list[SubQuestion]) -> str:
    """One line per step, for logs and for the writer's context header."""
    return "\n".join(f"{s['id']} [{s['source']}] {s['question']}" for s in steps)


@agent_step("planner")
def planner_node(state: MultiAgentState) -> dict[str, Any]:
    """Produce the plan for this run.

    Args:
        state: Current state carrying ``query``.

    Returns:
        Partial state with ``plan`` and ``plan_summary``.
    """
    query = state.get("query", "")
    if not query:
        logger.warning("Planner received an empty query.")
        return {
            "plan": [{"id": "s1", "question": "", "source": SOURCE_NONE}],
            "plan_summary": "",
        }

    try:
        from agent.nodes import _get_llm

        response = _get_llm().invoke(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": f"Question: {query}"},
            ]
        )
        charge_response(response)
        steps = parse_plan(response.content, query)
    except Exception:
        logger.exception("Planner call failed, falling back to one document step.")
        steps = [{"id": "s1", "question": query, "source": SOURCE_DOCUMENTS}]

    summary = summarise_plan(steps)
    logger.info("Plan has %d step(s):\n%s", len(steps), summary)
    return {"plan": steps, "plan_summary": summary}
