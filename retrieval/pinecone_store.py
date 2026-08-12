"""
retrieval/pinecone_store.py — Pinecone vector store as alternative to FAISS.

Provides the same interface as vectorstore.py so the agent layer can swap
between FAISS (local / offline) and Pinecone (cloud / scalable) by setting
VECTOR_BACKEND=pinecone in the .env file.

Requires:
    PINECONE_API_KEY  — Pinecone project API key
    PINECONE_INDEX    — Name of the target Pinecone index (must already exist
                        or will be created on first call)
    PINECONE_ENV      — Pinecone environment/region (e.g. "us-east-1-aws")
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from pinecone import Pinecone, ServerlessSpec
    _PINECONE_AVAILABLE = True
except ImportError:
    _PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed — Pinecone backend unavailable")

_pc_client: Any = None
_pc_index: Any = None

_INDEX_NAME = os.getenv("PINECONE_INDEX", "agentiq-docs")
_DIMENSION = 1536          # text-embedding-3-small output dimension
_METRIC = "cosine"
_NAMESPACE = "agentiq"


def _get_client() -> Any:
    global _pc_client
    if _pc_client is None:
        api_key = os.getenv("PINECONE_API_KEY", "")
        if not api_key:
            raise EnvironmentError("PINECONE_API_KEY is not set")
        _pc_client = Pinecone(api_key=api_key)
    return _pc_client


def _get_index() -> Any:
    global _pc_index
    if _pc_index is not None:
        return _pc_index

    pc = _get_client()
    existing = [idx.name for idx in pc.list_indexes()]

    if _INDEX_NAME not in existing:
        env = os.getenv("PINECONE_ENV", "us-east-1")
        pc.create_index(
            name=_INDEX_NAME,
            dimension=_DIMENSION,
            metric=_METRIC,
            spec=ServerlessSpec(cloud="aws", region=env),
        )
        logger.info("Created Pinecone index '%s' (dim=%d)", _INDEX_NAME, _DIMENSION)

    _pc_index = pc.Index(_INDEX_NAME)
    stats = _pc_index.describe_index_stats()
    logger.info(
        "Connected to Pinecone index '%s' — %s vectors",
        _INDEX_NAME,
        stats.get("total_vector_count", "?"),
    )
    return _pc_index


def upsert_to_pinecone(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> int:
    """
    Upsert document chunks and their embeddings into Pinecone.

    Args:
        chunks:     List of chunk dicts with keys title, source, content.
        embeddings: Parallel list of embedding vectors (one per chunk).

    Returns:
        Number of vectors upserted.
    """
    if not _PINECONE_AVAILABLE:
        logger.warning("Pinecone not available — skipping upsert")
        return 0

    index = _get_index()

    vectors = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vector_id = f"{_NAMESPACE}-{i}"
        vectors.append(
            {
                "id": vector_id,
                "values": emb,
                "metadata": {
                    "title": chunk.get("title", ""),
                    "source": chunk.get("source", ""),
                    "content": chunk.get("content", "")[:2000],
                },
            }
        )

    batch_size = 100
    upserted = 0
    for start in range(0, len(vectors), batch_size):
        batch = vectors[start : start + batch_size]
        index.upsert(vectors=batch, namespace=_NAMESPACE)
        upserted += len(batch)

    logger.info("Upserted %d vectors to Pinecone index '%s'", upserted, _INDEX_NAME)
    return upserted


def query_pinecone(
    query_embedding: list[float],
    top_k: int = 5,
    filter_metadata: dict | None = None,
) -> list[dict[str, Any]]:
    """
    Query Pinecone for the top-K most similar vectors.

    Args:
        query_embedding: Embedding vector for the query string.
        top_k:           Number of results to return.
        filter_metadata: Optional Pinecone metadata filter dict.

    Returns:
        List of result dicts with keys: id, score, content, title, source.
        Empty list if Pinecone is unavailable.
    """
    if not _PINECONE_AVAILABLE:
        return []

    index = _get_index()

    kwargs: dict[str, Any] = {
        "vector": query_embedding,
        "top_k": top_k,
        "namespace": _NAMESPACE,
        "include_metadata": True,
    }
    if filter_metadata:
        kwargs["filter"] = filter_metadata

    response = index.query(**kwargs)

    results = []
    for match in response.get("matches", []):
        meta = match.get("metadata", {})
        results.append(
            {
                "id": match["id"],
                "score": float(match["score"]),
                "content": meta.get("content", ""),
                "title": meta.get("title", ""),
                "source": meta.get("source", ""),
            }
        )

    logger.debug(
        "Pinecone returned %d results (top score: %.3f)",
        len(results),
        results[0]["score"] if results else 0.0,
    )
    return results


def delete_pinecone_namespace() -> None:
    """Delete all vectors in the agent's namespace (useful for re-indexing)."""
    if not _PINECONE_AVAILABLE:
        return
    index = _get_index()
    index.delete(delete_all=True, namespace=_NAMESPACE)
    logger.info("Deleted all vectors in namespace '%s'", _NAMESPACE)
