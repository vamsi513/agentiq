"""
Deterministic retrieval evaluation — runs in CI without any external API calls.

Tests that the FAISS vectorstore returns semantically relevant chunks for
retrieval-type queries from the curated test set.  Hit rate is defined as:
the fraction of queries where at least one of the top-5 results contains a
key term present in both the query and its ground truth answer.

This proves the retrieval stack is working (not returning random results)
without requiring an LLM judge.  Full RAGAS evaluation lives in the
manually-triggered `eval.yml` workflow.
"""

import json
import re
from pathlib import Path

import pytest

_QUERIES_FILE = Path(__file__).parent.parent / "evaluation" / "test_queries.json"

_STOP_WORDS = {
    "a", "an", "the", "is", "it", "in", "of", "and", "to", "are", "how",
    "what", "why", "does", "do", "for", "with", "that", "this", "was",
    "its", "by", "be", "as", "at", "on", "or", "from", "their", "they",
    "have", "has", "can", "not", "more", "than", "between", "difference",
    "used", "use", "using", "which", "when", "where", "who", "into",
}

_MIN_HIT_RATE = 0.65


def _key_terms(text: str, min_len: int = 5) -> set[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return {t for t in tokens if len(t) >= min_len and t not in _STOP_WORDS}


def _retrieval_queries() -> list[dict]:
    data = json.loads(_QUERIES_FILE.read_text(encoding="utf-8"))
    return [q for q in data if q.get("context_type") == "retrieval"]


@pytest.mark.asyncio
async def test_retrieval_hit_rate():
    from retrieval.vectorstore import query_vectorstore

    queries = _retrieval_queries()
    assert queries, "No retrieval-type queries found in test_queries.json"

    hits = 0
    misses = []

    for item in queries:
        question = item["question"]
        ground_truth = item["ground_truth"]

        target_terms = _key_terms(question) | _key_terms(ground_truth)
        results = query_vectorstore(question, top_k=5)

        assert results, f"Vectorstore returned no results for: {question[:60]}"

        retrieved_text = " ".join(r["content"].lower() for r in results)
        retrieved_terms = _key_terms(retrieved_text)

        if target_terms & retrieved_terms:
            hits += 1
        else:
            misses.append(question[:80])

    hit_rate = hits / len(queries)
    assert hit_rate >= _MIN_HIT_RATE, (
        f"Retrieval hit rate {hit_rate:.2f} is below threshold {_MIN_HIT_RATE}. "
        f"Misses ({len(misses)}): {misses[:5]}"
    )


def test_retrieval_returns_scored_chunks():
    from retrieval.vectorstore import query_vectorstore

    results = query_vectorstore("What is the Transformer architecture?", top_k=5)

    assert len(results) >= 1, "Expected at least one result"
    for r in results:
        assert "content" in r, "Result missing 'content' key"
        assert "score" in r, "Result missing 'score' key"
        assert r["score"] > 0, f"Expected positive score, got {r['score']}"
        assert len(r["content"]) > 0, "Result has empty content"
