"""
retrieval/vectorstore.py — FAISS vector store: build, persist, and query.

On first run the store is built from ``retrieval/documents/sample_docs.txt``
and saved to ``faiss_index/`` on disk.  Subsequent runs load the persisted
index, so the expensive embedding step only happens once.

Document chunking strategy:
- Each ``===DOC_START=== … ===DOC_END===`` block in sample_docs.txt is
  treated as one logical document.
- Long documents are split into 512-token chunks with 64-token overlap
  using a simple character-based splitter (≈4 chars per token heuristic).
- Each chunk stores ``title``, ``source``, and ``content`` as metadata.

Retrieval returns the top-K chunks ranked by cosine similarity, together
with their metadata and similarity scores.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import settings
from retrieval.embeddings import get_embeddings

logger = logging.getLogger(__name__)

# Character-based approximation: ~4 chars per token
_CHUNK_SIZE_CHARS = 512 * 4       # ≈ 512 tokens
_CHUNK_OVERLAP_CHARS = 64 * 4     # ≈ 64 tokens


# ── Document parsing ──────────────────────────────────────────────────────────

def _parse_documents(filepath: Path) -> list[dict[str, str]]:
    """
    Parse the sample_docs.txt file into a list of document dicts.

    Each document block is delimited by ``===DOC_START===`` / ``===DOC_END===``.
    Title and Source are extracted from the first two lines of each block.

    Args:
        filepath: Path to the documents text file.

    Returns:
        List of dicts with keys ``title``, ``source``, ``content``.
    """
    docs: list[dict[str, str]] = []
    text = filepath.read_text(encoding="utf-8")
    blocks = text.split("===DOC_START===")

    for block in blocks:
        block = block.strip()
        if not block or "===DOC_END===" not in block:
            continue
        content_raw = block.split("===DOC_END===")[0].strip()
        lines = content_raw.splitlines()

        title = ""
        source = ""
        body_lines: list[str] = []

        for line in lines:
            if line.startswith("Title:"):
                title = line.replace("Title:", "").strip()
            elif line.startswith("Source:"):
                source = line.replace("Source:", "").strip()
            else:
                body_lines.append(line)

        content = "\n".join(body_lines).strip()
        if content:
            docs.append({"title": title, "source": source, "content": content})

    logger.info("Parsed %d documents from %s", len(docs), filepath)
    return docs


def _chunk_documents(docs: list[dict[str, str]]) -> list[dict[str, Any]]:
    """
    Split documents into overlapping chunks for fine-grained retrieval.

    Args:
        docs: List of document dicts with ``title``, ``source``, ``content``.

    Returns:
        List of chunk dicts with ``title``, ``source``, ``content``,
        ``chunk_index``, and ``doc_index``.
    """
    chunks: list[dict[str, Any]] = []

    for doc_idx, doc in enumerate(docs):
        content = doc["content"]
        start = 0
        chunk_idx = 0

        while start < len(content):
            end = start + _CHUNK_SIZE_CHARS
            chunk_text = content[start:end]
            chunks.append(
                {
                    "title": doc["title"],
                    "source": doc["source"],
                    "content": chunk_text,
                    "chunk_index": chunk_idx,
                    "doc_index": doc_idx,
                }
            )
            chunk_idx += 1
            start += _CHUNK_SIZE_CHARS - _CHUNK_OVERLAP_CHARS

    logger.info("Generated %d chunks from %d documents.", len(chunks), len(docs))
    return chunks


# ── Index build & persist ─────────────────────────────────────────────────────

def _build_index(chunks: list[dict[str, Any]]) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Build a FAISS IndexFlatIP (inner-product / cosine) from chunk embeddings.

    Vectors are L2-normalised before indexing so inner-product == cosine
    similarity.

    Args:
        chunks: List of chunk dicts containing a ``content`` key.

    Returns:
        Tuple of (faiss.Index, chunks_list) where chunks_list preserves
        metadata aligned with index positions.
    """
    embeddings_model = get_embeddings()
    texts = [c["content"] for c in chunks]
    logger.info("Embedding %d chunks — this may take a moment…", len(texts))
    vectors = np.array(embeddings_model.embed_documents(texts), dtype=np.float32)

    # L2-normalise for cosine similarity via inner product
    faiss.normalize_L2(vectors)

    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)
    return index, chunks


def _save_index(
    index: faiss.Index,
    chunks: list[dict[str, Any]],
    index_path: Path,
) -> None:
    """
    Persist the FAISS index and chunk metadata to disk.

    Args:
        index: Trained FAISS index.
        chunks: Chunk metadata list aligned with index positions.
        index_path: Directory path to save files into.
    """
    index_path.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path / "index.faiss"))
    with open(index_path / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    logger.info("FAISS index saved to %s", index_path)


def _load_index(index_path: Path) -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Load a previously saved FAISS index and chunk metadata from disk.

    Args:
        index_path: Directory containing ``index.faiss`` and ``chunks.pkl``.

    Returns:
        Tuple of (faiss.Index, chunks_list).

    Raises:
        FileNotFoundError: If the index files do not exist.
    """
    faiss_file = index_path / "index.faiss"
    chunks_file = index_path / "chunks.pkl"

    if not faiss_file.exists() or not chunks_file.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")

    index = faiss.read_index(str(faiss_file))
    with open(chunks_file, "rb") as f:
        chunks = pickle.load(f)

    logger.info("FAISS index loaded from %s (%d vectors)", index_path, index.ntotal)
    return index, chunks


# ── Singleton state ───────────────────────────────────────────────────────────

_faiss_index: faiss.Index | None = None
_faiss_chunks: list[dict[str, Any]] | None = None


def get_vectorstore() -> tuple[faiss.Index, list[dict[str, Any]]]:
    """
    Return the FAISS index and chunk list, building them if necessary.

    On first call: parses documents → chunks → embeds → saves to disk.
    On subsequent calls: loads from disk.
    The result is cached in module-level globals to avoid redundant I/O.

    Returns:
        Tuple of (faiss.Index, chunks_list).
    """
    global _faiss_index, _faiss_chunks

    if _faiss_index is not None and _faiss_chunks is not None:
        return _faiss_index, _faiss_chunks

    index_path = settings.faiss_index_path

    if index_path.exists() and (index_path / "index.faiss").exists():
        logger.info("Loading existing FAISS index from disk.")
        _faiss_index, _faiss_chunks = _load_index(index_path)
    else:
        logger.info("No existing index found — building from documents.")
        docs_file = settings.documents_path / "sample_docs.txt"
        docs = _parse_documents(docs_file)
        chunks = _chunk_documents(docs)
        _faiss_index, _faiss_chunks = _build_index(chunks)
        _save_index(_faiss_index, _faiss_chunks, index_path)

    return _faiss_index, _faiss_chunks


def add_documents_to_vectorstore(texts: list[str], metadata: list[dict[str, Any]]) -> int:
    """
    Add new document chunks to the in-memory FAISS index at runtime.

    Used by the PDF upload feature to index uploaded documents without
    rebuilding the entire index from scratch.  Changes are not persisted
    to disk — they exist only for the lifetime of the current process.

    Args:
        texts:    List of text strings to embed and add.
        metadata: List of metadata dicts (same length as texts) with keys
                  ``title``, ``source``.  Other keys are optional.

    Returns:
        Number of vectors successfully added.

    Raises:
        ValueError: If texts and metadata lengths differ.
    """
    if len(texts) != len(metadata):
        raise ValueError("texts and metadata must have the same length.")

    if not texts:
        return 0

    global _faiss_index, _faiss_chunks

    # Ensure the index is loaded/built before adding
    get_vectorstore()

    try:
        embeddings_model = get_embeddings()
        vectors = np.array(
            embeddings_model.embed_documents(texts), dtype=np.float32
        )
        faiss.normalize_L2(vectors)
        _faiss_index.add(vectors)  # type: ignore[union-attr]

        # Append metadata aligned with the new index positions
        for i, (text, meta) in enumerate(zip(texts, metadata)):
            _faiss_chunks.append(  # type: ignore[union-attr]
                {
                    "title": meta.get("title", "Uploaded Document"),
                    "source": meta.get("source", ""),
                    "content": text,
                    "chunk_index": i,
                    "doc_index": -1,  # -1 signals runtime-added document
                }
            )

        logger.info("Added %d chunks to in-memory FAISS index.", len(texts))
        return len(texts)

    except Exception as exc:
        logger.exception("add_documents_to_vectorstore failed: %s", exc)
        return 0


def query_vectorstore(
    query: str,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve the top-K most relevant chunks for a given query.

    Args:
        query: Natural language query string.
        top_k: Number of results to return.  Defaults to
               ``settings.top_k_retrieval``.

    Returns:
        List of result dicts, each with keys:
        ``title``, ``source``, ``content``, ``score``.
        Sorted by descending similarity score.
    """
    k = top_k or settings.top_k_retrieval

    try:
        index, chunks = get_vectorstore()

        embeddings_model = get_embeddings()
        query_vec = np.array(
            [embeddings_model.embed_query(query)], dtype=np.float32
        )
        faiss.normalize_L2(query_vec)

        scores, indices = index.search(query_vec, k)

        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = chunks[idx]
            results.append(
                {
                    "title": chunk["title"],
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": float(score),
                }
            )

        logger.debug("Retrieved %d chunks for query: %.60s", len(results), query)
        return results

    except Exception as exc:
        logger.exception("query_vectorstore failed for query '%.60s': %s", query, exc)
        return []
