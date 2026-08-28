"""
retrieval/llamaindex_loader.py — LlamaIndex document indexing and retrieval.

Standalone reference implementation of an alternative retrieval path using
LlamaIndex VectorStoreIndex. Documents are loaded, chunked, and indexed
using LlamaIndex's built-in node parser and embedding pipeline. NOT wired
into agent/nodes.py — retriever_node always calls vectorstore.py's FAISS
implementation; nothing in the live agent calls this module.
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

try:
    from llama_index.core import (
        Settings as LlamaSettings,
    )
    from llama_index.core import (
        SimpleDirectoryReader,
        VectorStoreIndex,
    )
    from llama_index.core.node_parser import SentenceSplitter
    from llama_index.core.schema import Document as LlamaDocument
    from llama_index.embeddings.openai import OpenAIEmbedding

    _LLAMAINDEX_AVAILABLE = True
except ImportError:
    _LLAMAINDEX_AVAILABLE = False
    logger.warning("llama-index not installed — LlamaIndex retrieval unavailable")

_index: Any = None


def _configure_llama_settings() -> None:
    if not _LLAMAINDEX_AVAILABLE:
        return
    LlamaSettings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY", ""),
    )
    LlamaSettings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)
    LlamaSettings.llm = None


def build_llamaindex_from_texts(texts: list[str], metadatas: list[dict]) -> Any:
    """
    Build a LlamaIndex VectorStoreIndex from raw text strings.

    Args:
        texts: List of document text strings.
        metadatas: List of metadata dicts (title, source) aligned with texts.

    Returns:
        VectorStoreIndex instance, or None if llama-index is not installed.
    """
    global _index

    if not _LLAMAINDEX_AVAILABLE:
        return None

    _configure_llama_settings()

    docs = [
        LlamaDocument(text=t, metadata=m)
        for t, m in zip(texts, metadatas)
    ]

    _index = VectorStoreIndex.from_documents(docs)
    logger.info("LlamaIndex VectorStoreIndex built with %d documents", len(docs))
    return _index


def build_llamaindex_from_directory(directory: str) -> Any:
    """
    Build a LlamaIndex VectorStoreIndex by loading all docs from a directory.

    Args:
        directory: Path to directory containing .txt, .pdf, .md files.

    Returns:
        VectorStoreIndex instance, or None if llama-index is not installed.
    """
    global _index

    if not _LLAMAINDEX_AVAILABLE:
        return None

    _configure_llama_settings()

    reader = SimpleDirectoryReader(input_dir=directory, recursive=True)
    docs = reader.load_data()
    _index = VectorStoreIndex.from_documents(docs)
    logger.info(
        "LlamaIndex VectorStoreIndex built from directory '%s': %d docs",
        directory,
        len(docs),
    )
    return _index


def query_llamaindex(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Query the LlamaIndex VectorStoreIndex and return ranked results.

    Args:
        query: Natural language query string.
        top_k: Number of top results to return.

    Returns:
        List of result dicts with keys: content, score, metadata.
        Empty list if index not built or llama-index unavailable.
    """
    if not _LLAMAINDEX_AVAILABLE or _index is None:
        logger.warning("LlamaIndex not available or index not built")
        return []

    retriever = _index.as_retriever(similarity_top_k=top_k)
    nodes = retriever.retrieve(query)

    results = []
    for node in nodes:
        results.append(
            {
                "content": node.get_content(),
                "score": float(node.score) if node.score is not None else 0.0,
                "metadata": node.metadata,
            }
        )

    logger.debug("LlamaIndex retrieved %d results for query: %.60s", len(results), query)
    return results


def get_llamaindex_query_engine(streaming: bool = False) -> Any:
    """
    Return a LlamaIndex query engine for full LLM-powered Q&A over the index.

    Args:
        streaming: If True, returns a streaming query engine.

    Returns:
        QueryEngine instance, or None if index not built.
    """
    if not _LLAMAINDEX_AVAILABLE or _index is None:
        return None

    return _index.as_query_engine(streaming=streaming, similarity_top_k=5)
