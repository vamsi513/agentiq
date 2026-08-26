"""
agent/nodes.py — All LangGraph node functions for the AgentIQ research agent.

Each function in this module is a LangGraph node: it receives the current
AgentState, performs its work, and returns a partial state dict with only
the keys it wants to update.

Node execution order (determined by graph.py):
    router → [retriever | web_search | generator(direct)] → generator → END

Nodes:
    router_node      — LLM-based query classifier (retrieval/web_search/direct)
    retriever_node   — FAISS semantic search over local documents
    web_search_node  — Live Tavily web search
    generator_node   — Final answer synthesis with citations
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agent.state import AgentState
from config import settings

logger = logging.getLogger(__name__)


# ── LLM singleton ─────────────────────────────────────────────────────────────

def _get_llm() -> ChatOpenAI:
    return _llm


_llm = ChatOpenAI(
    model=settings.openai_model,
    temperature=settings.temperature,
    max_tokens=settings.max_tokens,
    api_key=settings.openai_api_key,
    streaming=True,
)


# ── Router node ───────────────────────────────────────────────────────────────

_ROUTER_SYSTEM_PROMPT = """You are a query routing classifier for a research assistant.

Given a user's question, decide which information source to use:

- "retrieval": The question is about AI, machine learning, LLMs, transformers,
  embeddings, RAG, RLHF, LangGraph, or other ML/AI topics that may be covered
  in local research documents.
- "web_search": The question requires current/live information — news, prices,
  recent events, today's date, sports results, weather, or anything that changes
  over time.
- "direct": The question is conversational (greetings, thanks, simple facts),
  asks about you/the system, or is a general knowledge question easily answered
  from training data without retrieval.

Respond with EXACTLY one word: retrieval, web_search, or direct.
No explanation. No punctuation. Just the single routing word."""


def router_node(state: AgentState) -> dict[str, Any]:
    """
    Classify the user query and set the routing decision.

    Uses gpt-4o-mini with a strict classification prompt to decide between
    ``retrieval``, ``web_search``, and ``direct`` response paths.

    Args:
        state: Current AgentState containing the conversation messages.

    Returns:
        Partial state dict updating ``query`` and ``route_decision``.
    """
    messages = state.get("messages", [])
    query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            query = msg.content
            break

    if not query:
        logger.warning("Router received empty query — defaulting to direct.")
        return {"query": "", "route_decision": "direct"}

    logger.info("Router classifying query: %.80s", query)

    try:
        llm = _get_llm()
        response = llm.invoke(
            [
                {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Query: {query}"},
            ]
        )
        decision_raw = response.content.strip().lower()

        # Sanitise — only accept known values
        if decision_raw in {"retrieval", "web_search", "direct"}:
            decision = decision_raw
        else:
            logger.warning(
                "Router returned unexpected value '%s', defaulting to retrieval.",
                decision_raw,
            )
            decision = "retrieval"

    except Exception as exc:
        logger.exception("Router LLM call failed: %s — defaulting to direct.", exc)
        decision = "direct"

    logger.info("Route decision: %s", decision)
    return {"query": query, "route_decision": decision}


# ── Query rewriting ───────────────────────────────────────────────────────────

def _rewrite_query(query: str) -> list[str]:
    """Return the original query plus up to 2 paraphrased variants.

    Running FAISS against multiple phrasings of the same question surfaces
    chunks that use different vocabulary than the original, improving recall
    on questions that paraphrase or abbreviate technical terms.
    """
    try:
        llm = _get_llm()
        prompt = (
            "Rewrite the following question in 2 different ways that preserve "
            "its meaning but use different vocabulary. Output only the two "
            "rewritten questions, one per line, with no numbering or labels.\n\n"
            f"Question: {query}"
        )
        response = llm.invoke([{"role": "user", "content": prompt}])
        variants = [line.strip() for line in response.content.strip().splitlines() if line.strip()]
        return [query] + variants[:2]
    except Exception:
        return [query]


# ── Retriever node ────────────────────────────────────────────────────────────

def retriever_node(state: AgentState) -> dict[str, Any]:
    """
    Query the FAISS vector store and populate context + sources.

    Retrieves the top-K most semantically similar document chunks for the
    current query and formats them into a single context string.

    Args:
        state: Current AgentState containing ``query``.

    Returns:
        Partial state dict updating ``context`` and ``sources``.
    """
    query = state.get("query", "")
    if not query:
        logger.warning("Retriever called with empty query.")
        return {"context": "", "sources": []}

    logger.info("Retriever querying FAISS for: %.80s", query)

    try:
        from retrieval.vectorstore import query_vectorstore

        # Optionally expand to paraphrased variants (set ENABLE_QUERY_REWRITING=true).
        # Off by default — each rewrite adds a full LLM round-trip.
        variants = _rewrite_query(query) if settings.enable_query_rewriting else [query]
        seen: set[str] = set()
        merged: list[dict] = []
        for q in variants:
            for r in query_vectorstore(q, top_k=5):
                key = r["content"][:120]
                if key not in seen:
                    seen.add(key)
                    merged.append(r)

        # Keep top 3 by score — a tighter context window reduces hallucination.
        results = sorted(merged, key=lambda r: r["score"], reverse=True)[:3]

        if not results:
            logger.warning("FAISS returned no results for query: %.80s", query)
            return {"context": "No relevant documents found.", "sources": []}

        # Build context string from retrieved chunks
        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for i, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {i}] {result['title']}\n{result['content']}"
            )
            sources.append(
                {
                    "title": result["title"],
                    "url": result["source"],
                    "content": result["content"][:200] + "…",
                    "score": result["score"],
                    "type": "retrieval",
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        top_score = results[0]["score"]
        logger.info("Retriever found %d chunks (top score=%.3f).", len(results), top_score)
        return {"context": context, "sources": sources, "retrieval_score": top_score}

    except Exception as exc:
        logger.exception("Retriever error: %s", exc)
        return {
            "context": "Document retrieval failed. No context is available.",
            "sources": [],
        }


# ── Web search node ───────────────────────────────────────────────────────────

def web_search_node(state: AgentState) -> dict[str, Any]:
    """
    Perform a live web search via Tavily and populate context + sources.

    Args:
        state: Current AgentState containing ``query``.

    Returns:
        Partial state dict updating ``context`` and ``sources``.
    """
    query = state.get("query", "")
    if not query:
        logger.warning("Web search node called with empty query.")
        return {"context": "", "sources": []}

    logger.info("Web search node querying Tavily for: %.80s", query)

    try:
        from tools.web_search import web_search

        results = web_search(query)

        context_parts: list[str] = []
        sources: list[dict[str, Any]] = []

        for i, result in enumerate(results, start=1):
            if result.get("is_fallback"):
                context_parts.append(result["content"])
                continue

            context_parts.append(
                f"[Source {i}] {result['title']}\nURL: {result['url']}\n{result['content']}"
            )
            sources.append(
                {
                    "title": result["title"],
                    "url": result["url"],
                    "content": result["content"][:200] + "…",
                    "score": result.get("score", 0.0),
                    "type": "web_search",
                }
            )

        context = "\n\n---\n\n".join(context_parts)
        logger.info("Web search yielded %d sources.", len(sources))
        return {"context": context, "sources": sources}

    except Exception as exc:
        logger.exception("Web search node error: %s", exc)
        return {
            "context": "Web search failed. Answering from model knowledge.",
            "sources": [],
        }


# ── Generator node ────────────────────────────────────────────────────────────

_GENERATOR_SYSTEM_PROMPT = """You are AgentIQ, a precise research assistant.

Answer the user's question using ONLY the information in the provided context.
Do not add facts, explanations, or claims that are not explicitly stated in the context.
If the context does not contain enough information to fully answer the question, say exactly
what the context does cover and state clearly that you cannot answer the rest from the
available documents.

Guidelines:
- Stay strictly within what the context says.
- Cite sources inline by title, e.g. "According to [Source Title], …"
- Never fabricate citations, URLs, or facts not present in the context.
- Use markdown: bold key terms, bullet points for lists.
- Keep answers focused — 2–4 paragraphs unless the question demands more detail."""


def generator_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesise the final answer from context and conversation history.

    Combines the retrieved context (from retriever or web search) with the
    full conversation history to generate a grounded, cited response using
    gpt-4o-mini.

    Args:
        state: Current AgentState with ``query``, ``context``, ``messages``.

    Returns:
        Partial state dict appending the new AIMessage to ``messages``.
    """
    query = state.get("query", "")
    context = state.get("context", "")
    messages = state.get("messages", [])
    route = state.get("route_decision", "direct")

    logger.info("Generator synthesising answer (route=%s).", route)

    # Build the user-turn content with context injection
    if context and context.strip():
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Please answer the question based on the context above."
        )
    else:
        user_content = (
            f"Question: {query}\n\n"
            "No context was retrieved for this question. "
            "State that you do not have relevant documents to answer from."
        )

    # Rebuild conversation history for the LLM (exclude the current HumanMessage
    # since we're injecting it with context above)
    history: list[dict[str, str]] = [
        {"role": "system", "content": _GENERATOR_SYSTEM_PROMPT}
    ]

    for msg in messages[:-1]:  # All but the last (current) human message
        if isinstance(msg, HumanMessage):
            history.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            history.append({"role": "assistant", "content": msg.content})

    history.append({"role": "user", "content": user_content})

    try:
        llm = _get_llm()
        response = llm.invoke(history)
        answer = response.content

        current_turn = state.get("turn_count", 0)
        logger.info("Generator produced answer (%d chars).", len(answer))
        return {
            "messages": [AIMessage(content=answer)],
            "turn_count": current_turn + 1,
        }

    except Exception as exc:
        logger.exception("Generator LLM call failed: %s", exc)
        error_msg = (
            "I encountered an error while generating a response. "
            "Please check your OpenAI API key and try again."
        )
        current_turn = state.get("turn_count", 0)
        return {
            "messages": [AIMessage(content=error_msg)],
            "turn_count": current_turn + 1,
            "error": str(exc),
        }
