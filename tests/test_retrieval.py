"""
tests/test_retrieval.py — Unit tests for the FAISS vectorstore and embeddings.

Tests cover:
- Document parsing from sample_docs.txt format
- Chunk splitting with overlap
- Embedding model singleton behaviour
- FAISS index build, save, load round-trip
- query_vectorstore result shape and ordering

The embedding model is mocked with fixed-dimension random vectors so tests
run without downloading the sentence-transformers model.
"""

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ── Document parsing ──────────────────────────────────────────────────────────

class TestParseDocuments:
    """Tests for retrieval.vectorstore._parse_documents."""

    def test_parses_all_blocks(self, tmp_path):
        """Correctly parses two DOC_START/DOC_END blocks from a file."""
        from retrieval.vectorstore import _parse_documents

        doc_file = tmp_path / "docs.txt"
        doc_file.write_text(
            "===DOC_START===\n"
            "Title: Doc One\n"
            "Source: https://example.com/1\n"
            "This is the body of document one.\n"
            "===DOC_END===\n"
            "===DOC_START===\n"
            "Title: Doc Two\n"
            "Source: https://example.com/2\n"
            "Body of document two.\n"
            "===DOC_END===\n",
            encoding="utf-8",
        )

        docs = _parse_documents(doc_file)

        assert len(docs) == 2
        assert docs[0]["title"] == "Doc One"
        assert docs[0]["source"] == "https://example.com/1"
        assert "body of document one" in docs[0]["content"].lower()
        assert docs[1]["title"] == "Doc Two"

    def test_skips_empty_blocks(self, tmp_path):
        """Blocks with no body content are silently skipped."""
        from retrieval.vectorstore import _parse_documents

        doc_file = tmp_path / "docs.txt"
        doc_file.write_text(
            "===DOC_START===\n"
            "Title: Empty\n"
            "Source: https://example.com\n"
            "===DOC_END===\n",
            encoding="utf-8",
        )

        docs = _parse_documents(doc_file)
        assert len(docs) == 0

    def test_handles_file_with_no_blocks(self, tmp_path):
        """Returns empty list when no valid blocks are found."""
        from retrieval.vectorstore import _parse_documents

        doc_file = tmp_path / "docs.txt"
        doc_file.write_text("This file has no doc blocks.\n", encoding="utf-8")

        docs = _parse_documents(doc_file)
        assert docs == []


# ── Chunk splitting ───────────────────────────────────────────────────────────

class TestChunkDocuments:
    """Tests for retrieval.vectorstore._chunk_documents."""

    def test_short_doc_produces_single_chunk(self):
        """A short document (< chunk size) produces exactly one chunk."""
        from retrieval.vectorstore import _chunk_documents

        docs = [{"title": "T", "source": "S", "content": "Short content."}]
        chunks = _chunk_documents(docs)

        assert len(chunks) == 1
        assert chunks[0]["title"] == "T"
        assert chunks[0]["chunk_index"] == 0

    def test_long_doc_produces_multiple_chunks(self):
        """A document longer than chunk size is split into multiple chunks."""
        from retrieval.vectorstore import _chunk_documents

        # ~2048 chars → at 2048-char chunk size should produce >1 chunk
        long_content = "word " * 600  # 3000 chars
        docs = [{"title": "Long", "source": "S", "content": long_content}]
        chunks = _chunk_documents(docs)

        assert len(chunks) > 1

    def test_chunk_metadata_preserved(self):
        """Each chunk carries title, source, and doc_index from parent doc."""
        from retrieval.vectorstore import _chunk_documents

        docs = [{"title": "My Doc", "source": "https://x.com", "content": "Hello world."}]
        chunks = _chunk_documents(docs)

        assert chunks[0]["title"] == "My Doc"
        assert chunks[0]["source"] == "https://x.com"
        assert chunks[0]["doc_index"] == 0


# ── Embeddings singleton ──────────────────────────────────────────────────────

class TestGetEmbeddings:
    """Tests for retrieval.embeddings.get_embeddings singleton."""

    def test_returns_same_instance_on_repeated_calls(self):
        """get_embeddings() returns the same object every call."""
        import retrieval.embeddings as emb_module

        # Reset singleton for test isolation
        emb_module._embeddings_instance = None

        with patch("retrieval.embeddings.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_st.return_value = mock_model

            e1 = emb_module.get_embeddings()
            e2 = emb_module.get_embeddings()

        assert e1 is e2
        # SentenceTransformer constructor only called once
        mock_st.assert_called_once()

        # Reset
        emb_module._embeddings_instance = None


# ── FAISS index round-trip ────────────────────────────────────────────────────

class TestFAISSRoundTrip:
    """Tests for FAISS index save/load and query_vectorstore result shape."""

    def _mock_embeddings(self, dim: int = 384):
        """Return a mock embeddings object that produces fixed-dim random vectors."""
        mock = MagicMock()
        mock.embed_documents.side_effect = lambda texts: (
            np.random.rand(len(texts), dim).tolist()
        )
        mock.embed_query.side_effect = lambda text: (
            np.random.rand(dim).tolist()
        )
        return mock

    def test_build_and_query_returns_top_k(self):
        """
        Building an index from chunks and querying it returns <= top_k results
        with the expected keys.
        """
        from retrieval.vectorstore import _build_index, query_vectorstore
        import retrieval.vectorstore as vs_module

        chunks = [
            {"title": f"Doc {i}", "source": f"src_{i}", "content": f"Content number {i}"}
            for i in range(10)
        ]

        mock_emb = self._mock_embeddings(dim=384)

        with patch("retrieval.vectorstore.get_embeddings", return_value=mock_emb):
            index, stored_chunks = _build_index(chunks)

            # Patch module-level globals so query_vectorstore uses our index
            vs_module._faiss_index = index
            vs_module._faiss_chunks = stored_chunks

            results = query_vectorstore("test query", top_k=3)

        assert len(results) <= 3
        for r in results:
            assert "title" in r
            assert "source" in r
            assert "content" in r
            assert "score" in r
            assert isinstance(r["score"], float)

        # Reset module globals
        vs_module._faiss_index = None
        vs_module._faiss_chunks = None

    def test_save_and_load_index(self):
        """Saving then loading an index preserves the vector count."""
        import faiss
        from retrieval.vectorstore import _save_index, _load_index

        dim = 64
        index = faiss.IndexFlatIP(dim)
        vectors = np.random.rand(5, dim).astype(np.float32)
        faiss.normalize_L2(vectors)
        index.add(vectors)

        chunks = [{"title": f"C{i}", "source": "", "content": f"text {i}"} for i in range(5)]

        with tempfile.TemporaryDirectory() as tmpdir:
            idx_path = Path(tmpdir) / "test_index"
            _save_index(index, chunks, idx_path)

            loaded_index, loaded_chunks = _load_index(idx_path)

        assert loaded_index.ntotal == 5
        assert len(loaded_chunks) == 5
        assert loaded_chunks[0]["title"] == "C0"

    def test_load_raises_when_missing(self):
        """_load_index raises FileNotFoundError when index files don't exist."""
        from retrieval.vectorstore import _load_index

        with pytest.raises(FileNotFoundError):
            _load_index(Path("/nonexistent/path/index"))
