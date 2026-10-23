"""Compare the single agent workflow against the multi agent graph.

Runs the same questions through both modes and scores the answers with the
same judges, so the only thing that differs between the two columns is the
graph that produced the answer.

What is measured, and why it is measured this way:

- answer correctness: a judge compares the answer to the ground truth and
  returns correct, partial or incorrect. Scored 1.0, 0.5 and 0.0.
- citation support rate and unsupported claim rate: a second judge splits the
  answer into claims and checks each against the context the answer was built
  from. This judge is deliberately mode blind. It never sees which graph
  produced the answer and it does not read the multi agent verifier's own
  verdicts, because scoring a system with its own verifier would make the
  comparison circular.
- latency: wall clock per query, reported as p50 and p95.
- tokens and cost: counted by wrapping the shared model accessor, so both
  modes are counted by the same code path. The run budget's own accounting is
  not used for this, because the single agent path only charges tokens in the
  generator and would undercount itself.

Confidence intervals are percentile bootstrap over queries. For the single
versus multi comparison the bootstrap is paired: the same resampled questions
are used for both modes on each iteration, which is what the design calls for
since every question is run through both.

Spend is capped. The cap is checked before every model call, and the run stops
cleanly rather than overshooting.

Usage:
    python -m evaluation.compare_modes --split dev --estimate-only
    python -m evaluation.compare_modes --split dev
    python -m evaluation.compare_modes --split holdout
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import random
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Imported for its side effects: loading .env and installing log redaction.
import config  # noqa: F401

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).parent
_REPO = _EVAL_DIR.parent
_RESULTS_DIR = _REPO / "results"
_QUERIES_FILE = _EVAL_DIR / "test_queries.json"
_MULTI_SOURCE_FILE = _EVAL_DIR / "multi_source_queries.json"
_SPLIT_FILE = _EVAL_DIR / "mode_comparison_split.json"

# gpt-4o-mini list price, dollars per million tokens. Recorded in the results
# file alongside every figure derived from it, so a later price change does
# not silently make an old number mean something different.
PRICE_INPUT_PER_M = 0.150
PRICE_OUTPUT_PER_M = 0.600

# Hard ceiling on spend for one invocation of this script.
DEFAULT_SPEND_CAP_USD = 3.0

# Split is seeded so the same questions land in the same half on every run.
SPLIT_SEED = 20261023
DEV_FRACTION = 0.5


class SpendCapExceeded(RuntimeError):
    """Raised when the next model call would cross the configured cap."""


class Meter:
    """Counts tokens and dollars across model calls.

    Wraps the model object that agent.nodes._get_llm returns, so both modes
    are measured by the same code and neither can be counted differently from
    the other. Per query totals are read by snapshotting before and after.

    Two meters are used per run: one for the agent and one for the judges.
    Judging is measurement apparatus, not a cost the deployed system would
    pay, so it is excluded from the reported cost per query. It still counts
    against the spend cap, which is why a meter can be given a ``sibling``
    whose spend is included when the cap is checked.
    """

    def __init__(self, cap_usd: float, name: str = "agent") -> None:
        self.cap_usd = cap_usd
        self.name = name
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.sibling: "Meter | None" = None

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * PRICE_INPUT_PER_M / 1_000_000
            + self.output_tokens * PRICE_OUTPUT_PER_M / 1_000_000
        )

    def snapshot(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "calls": self.calls,
            "cost_usd": self.cost_usd,
        }

    def delta(self, before: dict[str, Any]) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens - before["input_tokens"],
            "output_tokens": self.output_tokens - before["output_tokens"],
            "calls": self.calls - before["calls"],
            "cost_usd": round(self.cost_usd - before["cost_usd"], 6),
        }

    @property
    def combined_cost_usd(self) -> float:
        """This meter's spend plus its sibling's, which is what the cap governs."""
        return self.cost_usd + (self.sibling.cost_usd if self.sibling else 0.0)

    def check(self) -> None:
        total = self.combined_cost_usd
        if total >= self.cap_usd:
            raise SpendCapExceeded(
                f"spend cap reached: ${total:.4f} of ${self.cap_usd:.2f}"
            )

    def record(self, response: Any) -> None:
        """Add one response's provider-reported usage."""
        self.calls += 1
        usage = getattr(response, "usage_metadata", None)
        if not isinstance(usage, dict):
            return
        self.input_tokens += int(usage.get("input_tokens") or 0)
        self.output_tokens += int(usage.get("output_tokens") or 0)

    def wrap(self, model: Any) -> Any:
        """Return a proxy that meters every invoke call on ``model``."""
        meter = self

        class MeteredModel:
            def __getattr__(self, name):
                return getattr(model, name)

            def invoke(self, *args, **kwargs):
                meter.check()
                response = model.invoke(*args, **kwargs)
                meter.record(response)
                return response

            async def ainvoke(self, *args, **kwargs):
                meter.check()
                response = await model.ainvoke(*args, **kwargs)
                meter.record(response)
                return response

        return MeteredModel()


# -- Query set and split ----------------------------------------------------


def load_queries() -> list[dict[str, Any]]:
    """The existing evaluation set plus the multi source additions."""
    base = json.loads(_QUERIES_FILE.read_text(encoding="utf-8"))
    for item in base:
        item.setdefault("needs_sources", [item.get("context_type", "direct")])
    extra = json.loads(_MULTI_SOURCE_FILE.read_text(encoding="utf-8"))
    return base + extra


def build_split(queries: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Stratified, seeded split into a dev half and a held out half.

    Stratified by context_type so the multi source questions, which are the
    ones the upgrade is supposed to help, are represented in both halves
    rather than landing entirely in one.
    """
    buckets: dict[str, list[int]] = {}
    for item in queries:
        buckets.setdefault(item.get("context_type", "direct"), []).append(item["id"])

    rng = random.Random(SPLIT_SEED)
    dev: list[int] = []
    holdout: list[int] = []
    for kind in sorted(buckets):
        ids = sorted(buckets[kind])
        rng.shuffle(ids)
        cut = round(len(ids) * DEV_FRACTION)
        dev.extend(ids[:cut])
        holdout.extend(ids[cut:])
    return {"dev": sorted(dev), "holdout": sorted(holdout)}


def get_split(queries: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Load the split from disk, writing it the first time.

    Persisted so the held out half is fixed once and cannot drift between
    runs. A split that is recomputed each time is not a held out set.
    """
    if _SPLIT_FILE.exists():
        return json.loads(_SPLIT_FILE.read_text(encoding="utf-8"))
    split = build_split(queries)
    _SPLIT_FILE.write_text(json.dumps(split, indent=2), encoding="utf-8")
    logger.info("Wrote split to %s", _SPLIT_FILE)
    return split


def select(queries: list[dict[str, Any]], split: dict[str, list[int]], name: str):
    if name == "all":
        return queries
    wanted = set(split[name])
    return [q for q in queries if q["id"] in wanted]


# -- Judges -----------------------------------------------------------------

_CORRECTNESS_SYSTEM = """You grade whether an answer matches a reference answer.

You are given a question, a reference answer and a candidate answer.

Grade the candidate:
- CORRECT: it conveys the substance of the reference. Wording may differ. Extra accurate detail is fine.
- PARTIAL: it gets part of the substance right but omits or garbles a material part. A question with two parts where only one is addressed is PARTIAL.
- INCORRECT: it contradicts the reference, or does not answer the question.

An answer that correctly states it lacks the information to answer is not CORRECT. Grade it PARTIAL if it still covers part of the reference, otherwise INCORRECT.

Reply with exactly one word: CORRECT, PARTIAL or INCORRECT."""


_GROUNDING_SYSTEM = """You check whether an answer is supported by the context it was given.

You are given a context and an answer. Split the answer into its distinct factual claims and judge each one against the context only.

- supported: the context states the claim, or states something that directly implies it.
- unsupported: the context does not state it, or the answer overstates what the context says.

Do not use your own knowledge. A claim that is true in the world but absent from the context is unsupported.

A statement that the context is missing something, or a question restatement, is not a claim. Skip it.

Return one line per claim and nothing else:
SUPPORTED | the claim text
UNSUPPORTED | the claim text"""


def _judge_model(meter: Meter):
    """A separate, deterministic model for grading.

    Temperature zero and its own instance, so grading does not inherit the
    agent's sampling settings and a judge call cannot be confused with an
    agent call in the trace.
    """
    from langchain_openai import ChatOpenAI

    from config import settings

    return meter.wrap(
        ChatOpenAI(
            model=settings.openai_model,
            temperature=0,
            api_key=settings.openai_api_key,
            timeout=60,
            max_retries=2,
        )
    )


_GRADE_SCORES = {"CORRECT": 1.0, "PARTIAL": 0.5, "INCORRECT": 0.0}


def judge_correctness(model, question: str, reference: str, answer: str) -> tuple[float, str]:
    """Grade an answer against its reference. Returns (score, raw grade)."""
    if not answer.strip():
        return 0.0, "EMPTY"
    try:
        response = model.invoke(
            [
                {"role": "system", "content": _CORRECTNESS_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n"
                        f"Reference answer: {reference}\n\n"
                        f"Candidate answer: {answer}"
                    ),
                },
            ]
        )
    except SpendCapExceeded:
        raise
    except Exception:
        logger.exception("Correctness judge failed.")
        return float("nan"), "JUDGE_ERROR"

    text = str(response.content).strip().upper()
    # Order matters and is load bearing: "INCORRECT" contains "CORRECT", so a
    # plain substring scan that checked CORRECT first would score every
    # incorrect answer as correct and silently inflate the headline metric.
    for grade in ("INCORRECT", "PARTIAL", "CORRECT"):
        if grade in text:
            return _GRADE_SCORES[grade], grade
    return float("nan"), f"UNPARSED:{text[:40]}"


def judge_grounding(model, context: str, answer: str) -> dict[str, Any]:
    """Split an answer into claims and check each against its context."""
    if not answer.strip():
        return {"claims": 0, "supported": 0, "unsupported": 0, "status": "empty_answer"}
    if not context.strip():
        # No context means nothing could ground the answer. Recording zero
        # claims would quietly flatter the mode; this is recorded distinctly.
        return {"claims": 0, "supported": 0, "unsupported": 0, "status": "no_context"}
    try:
        response = model.invoke(
            [
                {"role": "system", "content": _GROUNDING_SYSTEM},
                {"role": "user", "content": f"Context:\n{context}\n\nAnswer:\n{answer}"},
            ]
        )
    except SpendCapExceeded:
        raise
    except Exception:
        logger.exception("Grounding judge failed.")
        return {"claims": 0, "supported": 0, "unsupported": 0, "status": "judge_error"}

    supported = unsupported = 0
    for line in str(response.content).splitlines():
        head = line.strip().upper()
        if head.startswith("SUPPORTED |"):
            supported += 1
        elif head.startswith("UNSUPPORTED |"):
            unsupported += 1
    total = supported + unsupported
    return {
        "claims": total,
        "supported": supported,
        "unsupported": unsupported,
        "status": "ok" if total else "no_claims_parsed",
    }


# -- Running one mode -------------------------------------------------------


async def run_one(question: str, session_id: str) -> dict[str, Any]:
    from api.streaming import run_agent_sync

    return await run_agent_sync(question, session_id)


def _reset_graph_for(mode: str) -> None:
    """Point the process at the graph for ``mode``.

    The compiled graph is cached per mode, so the cache is cleared between
    modes to guarantee the next call builds the one that is wanted.
    """
    import agent.graph as graph_module
    from config import settings

    settings.agent_mode = mode
    graph_module._graph = None
    graph_module._graph_mode = None


@contextlib.contextmanager
def metered_agent(meter: Meter):
    """Route every agent model call through ``meter``.

    agent.nodes._get_llm is the single accessor both modes use, so wrapping it
    counts the planner, researchers, writer and verifier in multi mode and the
    router, retriever rewrite, direct and generator calls in single mode, with
    no per mode bookkeeping that could drift between them.
    """
    import agent.nodes as nodes

    original = nodes._get_llm
    nodes._get_llm = lambda: meter.wrap(original())
    try:
        yield
    finally:
        nodes._get_llm = original


async def run_mode(
    mode: str, queries: list[dict[str, Any]], meter: Meter, judge
) -> list[dict[str, Any]]:
    """Run every query in one mode and grade each answer."""
    _reset_graph_for(mode)
    rows: list[dict[str, Any]] = []
    stamp = int(time.time())

    for i, item in enumerate(queries, start=1):
        meter.check()
        question = item["question"]
        logger.info("[%s %d/%d] id=%s %.70s", mode, i, len(queries), item["id"], question)

        before = meter.snapshot()
        started = time.perf_counter()
        try:
            with metered_agent(meter):
                result = await run_one(question, f"cmp-{mode}-{stamp}-q{item['id']}")
            failed = None
        except SpendCapExceeded:
            raise
        except Exception as exc:
            logger.exception("Run failed for id=%s", item["id"])
            result = {"answer": "", "sources": [], "context": "", "route_decision": "error"}
            failed = f"{type(exc).__name__}"
        latency_ms = round((time.perf_counter() - started) * 1000, 1)
        # Taken before judging, so the reported cost per query is the agent's
        # own spend and never includes the grading calls below.
        usage = meter.delta(before)

        answer = result.get("answer", "") or ""
        context = result.get("context", "") or ""

        score, grade = judge_correctness(judge, question, item["ground_truth"], answer)
        grounding = judge_grounding(judge, context, answer)

        rows.append(
            {
                "id": item["id"],
                "mode": mode,
                "context_type": item.get("context_type", "direct"),
                "needs_sources": item.get("needs_sources", []),
                "multi_source": len(item.get("needs_sources", [])) > 1,
                "question": question,
                "answer": answer,
                "answer_chars": len(answer),
                "correctness": score,
                "grade": grade,
                "claims": grounding["claims"],
                "supported_claims": grounding["supported"],
                "unsupported_claims": grounding["unsupported"],
                "grounding_status": grounding["status"],
                "latency_ms": latency_ms,
                "input_tokens": usage["input_tokens"],
                "output_tokens": usage["output_tokens"],
                "model_calls": usage["calls"],
                "cost_usd": usage["cost_usd"],
                "num_sources": len(result.get("sources", []) or []),
                "route_decision": result.get("route_decision", ""),
                "error": failed,
            }
        )
        await asyncio.sleep(0.3)

    return rows


# -- Statistics -------------------------------------------------------------


def _mean(values: list[float]) -> float | None:
    clean = [v for v in values if v == v]  # drop NaN
    return statistics.fmean(clean) if clean else None


def percentile(values: list[float], q: float) -> float | None:
    """Nearest rank percentile. Returns None for an empty list.

    The index is rounded with an explicit ``int(x + 0.5)`` rather than with
    ``round``, which uses banker's rounding and would send a .5 index down to
    the even neighbour. That is a silent off by one in a reported p50.
    """
    if not values:
        return None
    ordered = sorted(values)
    index = int(q * (len(ordered) - 1) + 0.5)
    return ordered[max(0, min(len(ordered) - 1, index))]


def bootstrap_ci(
    values: list[float], iterations: int = 2000, seed: int = 7, level: float = 0.95
) -> dict[str, float] | None:
    """Percentile bootstrap confidence interval for a mean."""
    clean = [v for v in values if v == v]
    if len(clean) < 2:
        return None
    rng = random.Random(seed)
    n = len(clean)
    means = [
        statistics.fmean(clean[rng.randrange(n)] for _ in range(n)) for _ in range(iterations)
    ]
    means.sort()
    lo = (1 - level) / 2
    return {
        "mean": round(statistics.fmean(clean), 4),
        "lo": round(means[int(lo * iterations)], 4),
        "hi": round(means[int((1 - lo) * iterations) - 1], 4),
        "n": n,
    }


def paired_bootstrap_ci(
    pairs: list[tuple[float, float]], iterations: int = 2000, seed: int = 7, level: float = 0.95
) -> dict[str, float] | None:
    """Bootstrap CI for the mean difference on paired observations.

    Every question is run through both modes, so the two columns are paired
    and resampling them independently would throw away that pairing and give
    a wider interval than the design supports. Each iteration resamples
    questions and takes both modes' values for the questions drawn.
    """
    clean = [(a, b) for a, b in pairs if a == a and b == b]
    if len(clean) < 2:
        return None
    rng = random.Random(seed)
    n = len(clean)
    diffs = []
    for _ in range(iterations):
        drawn = [clean[rng.randrange(n)] for _ in range(n)]
        diffs.append(statistics.fmean(b - a for a, b in drawn))
    diffs.sort()
    lo = (1 - level) / 2
    observed = statistics.fmean(b - a for a, b in clean)
    low = diffs[int(lo * iterations)]
    high = diffs[int((1 - lo) * iterations) - 1]
    return {
        "mean_diff": round(observed, 4),
        "lo": round(low, 4),
        "hi": round(high, 4),
        "n": n,
        # A CI that does not straddle zero is the only thing here that
        # supports claiming a difference at all.
        "significant": bool(low > 0 or high < 0),
    }


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate one mode's rows."""
    if not rows:
        return {}
    latencies = [r["latency_ms"] for r in rows]
    claims = sum(r["claims"] for r in rows)
    supported = sum(r["supported_claims"] for r in rows)
    unsupported = sum(r["unsupported_claims"] for r in rows)
    return {
        "n": len(rows),
        "correctness_mean": _mean([r["correctness"] for r in rows]),
        "correctness_ci": bootstrap_ci([r["correctness"] for r in rows]),
        "citation_support_rate": round(supported / claims, 4) if claims else None,
        "unsupported_claim_rate": round(unsupported / claims, 4) if claims else None,
        "total_claims": claims,
        "per_query_support_rate_ci": bootstrap_ci(
            [r["supported_claims"] / r["claims"] for r in rows if r["claims"]]
        ),
        "latency_p50_ms": percentile(latencies, 0.50),
        "latency_p95_ms": percentile(latencies, 0.95),
        "latency_mean_ms": round(statistics.fmean(latencies), 1),
        "tokens_per_query": round(
            statistics.fmean(r["input_tokens"] + r["output_tokens"] for r in rows), 1
        ),
        "input_tokens_per_query": round(statistics.fmean(r["input_tokens"] for r in rows), 1),
        "output_tokens_per_query": round(statistics.fmean(r["output_tokens"] for r in rows), 1),
        "model_calls_per_query": round(statistics.fmean(r["model_calls"] for r in rows), 2),
        "cost_per_query_usd": round(statistics.fmean(r["cost_usd"] for r in rows), 6),
        "total_cost_usd": round(sum(r["cost_usd"] for r in rows), 4),
        "errors": sum(1 for r in rows if r["error"]),
        "answers_with_no_claims": sum(1 for r in rows if not r["claims"]),
    }


def compare(single: list[dict[str, Any]], multi: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparison between the two modes on the same questions."""
    by_id_single = {r["id"]: r for r in single}
    by_id_multi = {r["id"]: r for r in multi}
    shared = sorted(set(by_id_single) & set(by_id_multi))

    def pairs(metric):
        out = []
        for qid in shared:
            out.append((by_id_single[qid][metric], by_id_multi[qid][metric]))
        return out

    def support_pairs():
        out = []
        for qid in shared:
            a, b = by_id_single[qid], by_id_multi[qid]
            if a["claims"] and b["claims"]:
                out.append(
                    (a["supported_claims"] / a["claims"], b["supported_claims"] / b["claims"])
                )
        return out

    return {
        "n_paired": len(shared),
        "correctness_diff": paired_bootstrap_ci(pairs("correctness")),
        "support_rate_diff": paired_bootstrap_ci(support_pairs()),
        "latency_diff_ms": paired_bootstrap_ci(pairs("latency_ms")),
        "cost_diff_usd": paired_bootstrap_ci(pairs("cost_usd")),
    }


def slice_rows(rows: list[dict[str, Any]], multi_source: bool) -> list[dict[str, Any]]:
    return [r for r in rows if r["multi_source"] is multi_source]


# -- Cost estimate ----------------------------------------------------------

# Per query token estimates, from the smoke runs and the prompt sizes. Used
# only to produce an up front estimate; every reported figure comes from
# provider-reported usage on a real run.
_EST = {
    "single": {"in": 1800, "out": 420, "calls": 2},
    "multi": {"in": 8500, "out": 1100, "calls": 6},
    "judge": {"in": 2600, "out": 260, "calls": 2},
}


def estimate(n_queries: int) -> dict[str, Any]:
    """Projected spend for running both modes over ``n_queries`` questions."""

    def dollars(tokens_in: int, tokens_out: int) -> float:
        return tokens_in * PRICE_INPUT_PER_M / 1e6 + tokens_out * PRICE_OUTPUT_PER_M / 1e6

    per_mode = {}
    for mode in ("single", "multi"):
        agent = _EST[mode]
        judge = _EST["judge"]
        per_query = dollars(agent["in"], agent["out"]) + dollars(judge["in"], judge["out"])
        per_mode[mode] = {
            "per_query_usd": round(per_query, 6),
            "total_usd": round(per_query * n_queries, 4),
            "model_calls": (agent["calls"] + judge["calls"]) * n_queries,
        }
    total = round(sum(m["total_usd"] for m in per_mode.values()), 4)
    return {
        "n_queries": n_queries,
        "per_mode": per_mode,
        "total_usd": total,
        # Headroom for retries firing more often than the smoke runs suggested
        # and for longer answers than estimated.
        "total_usd_with_50pct_headroom": round(total * 1.5, 4),
        "prices": {
            "input_per_million_usd": PRICE_INPUT_PER_M,
            "output_per_million_usd": PRICE_OUTPUT_PER_M,
        },
    }


# -- Entry point ------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare single and multi agent modes.")
    parser.add_argument("--split", choices=["dev", "holdout", "all"], default="dev")
    parser.add_argument("--limit", type=int, default=0, help="cap the number of queries")
    parser.add_argument("--cap", type=float, default=DEFAULT_SPEND_CAP_USD)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    queries = load_queries()
    split = get_split(queries)
    selected = select(queries, split, args.split)
    if args.limit:
        selected = selected[: args.limit]

    projection = estimate(len(selected))
    print(json.dumps({"split": args.split, "estimate": projection}, indent=2))
    if args.estimate_only:
        return

    # Separate meters so the reported cost per query is the agent's own spend.
    # Judging is measurement apparatus and would not exist in a deployment, so
    # it is excluded from that figure while still counting against the cap.
    agent_meter = Meter(cap_usd=args.cap, name="agent")
    judge_meter = Meter(cap_usd=args.cap, name="judge")
    agent_meter.sibling = judge_meter
    judge_meter.sibling = agent_meter
    judge = _judge_model(judge_meter)
    stopped_early = None

    try:
        single_rows = asyncio.run(run_mode("single", selected, agent_meter, judge))
        multi_rows = asyncio.run(run_mode("multi", selected, agent_meter, judge))
    except SpendCapExceeded as exc:
        stopped_early = str(exc)
        logger.error("Stopping: %s", exc)
        raise SystemExit(f"spend cap reached before the run finished: {exc}") from exc

    results = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "split": args.split,
            "query_ids": [q["id"] for q in selected],
            "n_queries": len(selected),
            "model": config.settings.openai_model,
            "prices": projection["prices"],
            "spend_cap_usd": args.cap,
            "actual_spend_usd": round(agent_meter.combined_cost_usd, 4),
            "agent_spend_usd": round(agent_meter.cost_usd, 4),
            "judge_spend_usd": round(judge_meter.cost_usd, 4),
            "estimate_usd": projection["total_usd"],
            "stopped_early": stopped_early,
            "bootstrap_iterations": 2000,
        },
        "overall": {
            "single": summarise(single_rows),
            "multi": summarise(multi_rows),
            "paired": compare(single_rows, multi_rows),
        },
        "single_source_only": {
            "single": summarise(slice_rows(single_rows, False)),
            "multi": summarise(slice_rows(multi_rows, False)),
            "paired": compare(slice_rows(single_rows, False), slice_rows(multi_rows, False)),
        },
        "multi_source_only": {
            "single": summarise(slice_rows(single_rows, True)),
            "multi": summarise(slice_rows(multi_rows, True)),
            "paired": compare(slice_rows(single_rows, True), slice_rows(multi_rows, True)),
        },
        "rows": single_rows + multi_rows,
    }

    _RESULTS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else _RESULTS_DIR / f"mode_comparison_{args.split}.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"\nspend: agent ${agent_meter.cost_usd:.4f} + judge ${judge_meter.cost_usd:.4f}"
        f" = ${agent_meter.combined_cost_usd:.4f} of ${args.cap:.2f} cap"
    )
    print(f"results: {out}")


if __name__ == "__main__":
    main()
