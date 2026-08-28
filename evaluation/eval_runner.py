"""
evaluation/eval_runner.py — RAGAS evaluation pipeline for AgentIQ.

Runs the agent against 50 curated test queries, scores each answer with
RAGAS metrics (answer_relevancy, faithfulness), and saves results to
evaluation_results.json.

Usage:
    python -m evaluation.eval_runner

Requirements:
    - OPENAI_API_KEY must be set (RAGAS uses OpenAI for scoring)
    - Install dependencies: pip install -r requirements.txt

RAGAS version: >=0.2.0
The 0.2.x API uses class-based metrics and renamed dataset columns:
  user_input / response / retrieved_contexts / reference
"""

import asyncio
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).parent
_QUERIES_FILE = _EVAL_DIR / "test_queries.json"
_RESULTS_FILE = _EVAL_DIR / "evaluation_results.json"


# ── Agent runner ──────────────────────────────────────────────────────────────

async def _run_single_query(
    query: str,
    session_id: str,
) -> dict[str, Any]:
    """
    Run one query through the AgentIQ graph and return the result.

    Args:
        query: Question string.
        session_id: Unique session ID for this evaluation run.

    Returns:
        Dict with ``answer``, ``sources``, ``route_decision``,
        ``retrieval_score`` keys.
    """
    from api.streaming import run_agent_sync

    try:
        result = await run_agent_sync(query, session_id)
        return result
    except Exception as exc:
        logger.exception("Query failed: %.60s", query)
        return {
            "answer": f"ERROR: {type(exc).__name__}",
            "sources": [],
            "route_decision": "error",
            "retrieval_score": 0.0,
        }


# ── RAGAS evaluation ──────────────────────────────────────────────────────────

def _run_ragas(
    questions: list[str],
    answers: list[str],
    ground_truths: list[list[str]],
    contexts: list[list[str]],
) -> dict[str, float]:
    """
    Score the answer set using RAGAS 0.2.x metrics.

    Args:
        questions: List of question strings.
        answers: List of generated answer strings.
        ground_truths: List of lists, each containing one ground truth string.
        contexts: List of lists, each containing retrieved context strings.

    Returns:
        Dict with metric names as keys and float scores as values.
    """
    try:
        from datasets import Dataset
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, Faithfulness

        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", temperature=0))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

        # RAGAS 0.2.x column names
        data = {
            "user_input": questions,
            "response": answers,
            "reference": [gt[0] if gt else "" for gt in ground_truths],
            "retrieved_contexts": contexts,
        }

        dataset = Dataset.from_dict(data)

        logger.info("Running RAGAS evaluation on %d samples…", len(questions))
        result = evaluate(
            dataset,
            metrics=[
                AnswerRelevancy(llm=llm, embeddings=embeddings),
                Faithfulness(llm=llm),
            ],
        )

        df = result.to_pandas()
        scores = {
            "answer_relevancy": round(float(df["answer_relevancy"].mean()), 4),
            "faithfulness": round(float(df["faithfulness"].mean()), 4),
        }
        logger.info("RAGAS scores: %s", scores)
        return scores

    except ImportError as exc:
        logger.error("RAGAS import failed: %s", exc)
        raise
    except Exception:
        logger.exception("RAGAS evaluation failed")
        raise


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def _collect_answers(
    queries_data: list[dict],
) -> tuple[list, list, list, list, list]:
    """
    Run all queries through the agent and return collected inputs for RAGAS.

    Separated from RAGAS scoring so the event loop is closed before
    evaluate() runs — RAGAS 0.2.x internally creates its own loop and
    fails silently when one is already active.
    """
    questions: list[str] = []
    answers: list[str] = []
    ground_truths: list[list[str]] = []
    contexts: list[list[str]] = []
    per_query_results: list[dict[str, Any]] = []

    eval_session = f"eval-{int(time.time())}"

    for i, item in enumerate(queries_data, start=1):
        q_id = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]

        logger.info("[%d/%d] Running query %d: %.60s", i, len(queries_data), q_id, question)

        start_time = time.time()
        session_id = f"{eval_session}-q{q_id}"
        result = await _run_single_query(question, session_id)
        latency_ms = round((time.time() - start_time) * 1000, 1)

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        # Build context list for RAGAS from the *full* context the generator
        # actually used (state["context"]), not the 200-char truncated
        # previews in `sources[].content` (those exist only for citation
        # display in the API response and understate what the model saw,
        # which would make faithfulness look artificially low).
        full_context = result.get("context", "")
        ctx_list = [c for c in full_context.split("\n\n---\n\n") if c.strip()]
        if not ctx_list:
            ctx_list = ["No context retrieved."]

        questions.append(question)
        answers.append(answer)
        ground_truths.append([ground_truth])
        contexts.append(ctx_list)

        per_query_results.append(
            {
                "id": q_id,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "route_decision": result.get("route_decision", "direct"),
                "retrieval_score": result.get("retrieval_score", 0.0),
                "num_sources": len(sources),
                "latency_ms": latency_ms,
                "context_type": item.get("context_type", "direct"),
            }
        )

        if i < len(queries_data):
            await asyncio.sleep(0.5)

    return questions, answers, ground_truths, contexts, per_query_results


def run_evaluation(max_queries: int = 50) -> dict[str, Any]:
    """
    Full evaluation pipeline: run agent on test queries → score with RAGAS.

    The agent collection phase runs inside asyncio.run() so the event loop
    is fully closed before _run_ragas() is called. RAGAS 0.2.x's evaluate()
    creates its own internal loop and silently fails when one already exists.

    Args:
        max_queries: Cap on number of queries to evaluate (default: all 50).

    Returns:
        Full results dict written to evaluation_results.json.
    """
    logger.info("Loading test queries from %s", _QUERIES_FILE)
    queries_data = json.loads(_QUERIES_FILE.read_text(encoding="utf-8"))
    queries_data = queries_data[:max_queries]

    logger.info("Running AgentIQ on %d queries…", len(queries_data))

    # Phase 1: collect agent answers (async) — loop is closed when done
    questions, answers, ground_truths, contexts, per_query_results = asyncio.run(
        _collect_answers(queries_data)
    )

    # Phase 2: RAGAS scoring (sync) — no active event loop at this point
    logger.info("Scoring with RAGAS…")
    ragas_scores = _run_ragas(questions, answers, ground_truths, contexts)

    # ── Aggregate statistics ──────────────────────────────────────────────────
    latencies = [r["latency_ms"] for r in per_query_results]
    route_counts: dict[str, int] = {}
    for r in per_query_results:
        route = r["route_decision"]
        route_counts[route] = route_counts.get(route, 0) + 1

    results = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            "num_queries": len(per_query_results),
            "model": "gpt-4o-mini",
            "embedding_model": "all-MiniLM-L6-v2",
            "ragas_version": "0.2.x",
        },
        "ragas_scores": ragas_scores,
        "aggregate_stats": {
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "route_distribution": route_counts,
            "avg_sources_per_query": round(
                sum(r["num_sources"] for r in per_query_results) / len(per_query_results), 2
            ),
        },
        "per_query_results": per_query_results,
    }

    # ── Save results ──────────────────────────────────────────────────────────
    _RESULTS_FILE.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results saved to %s", _RESULTS_FILE)

    return results


def main() -> None:
    """
    Entry point for running evaluation from the command line.

    Usage:
        python -m evaluation.eval_runner
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
    )

    print("\n" + "=" * 60)
    print("  AgentIQ — RAGAS Evaluation Pipeline")
    print("=" * 60)
    print("Running 50 test queries through the agent…")
    print("This will take several minutes due to API rate limits.\n")

    results = run_evaluation(max_queries=50)

    scores = results["ragas_scores"]
    stats = results["aggregate_stats"]

    print("\n" + "=" * 60)
    print("  EVALUATION COMPLETE")
    print("=" * 60)
    print(f"  Queries evaluated : {results['metadata']['num_queries']}")
    print(f"  Answer Relevancy  : {scores.get('answer_relevancy', 'N/A'):.4f}")
    print(f"  Faithfulness      : {scores.get('faithfulness', 'N/A'):.4f}")
    print(f"  Avg Latency       : {stats['avg_latency_ms']} ms")
    print(f"  Route distribution: {stats['route_distribution']}")
    print(f"\n  Results saved to: {_RESULTS_FILE}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
