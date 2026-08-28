"""
retrieval/embeddings.py — Sentence-transformer embedding wrapper.

Provides a singleton EmbeddingModel that wraps the sentence-transformers
``all-MiniLM-L6-v2`` model and exposes two methods used by the vectorstore:
- ``embed_documents`` — bulk-encode a list of text chunks
- ``embed_query``     — encode a single query string

The wrapper is compatible with the LangChain ``Embeddings`` interface so it
can be swapped with any other LangChain embedding provider without touching
the vectorstore code.
"""

import logging

from sentence_transformers import SentenceTransformer

from config import settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings:
    """
    LangChain-compatible embedding wrapper around sentence-transformers.

    Loads the model once on construction and caches it for the lifetime of
    the process.  The model runs entirely locally — no API key required.

    Attributes:
        model_name: Hugging Face model identifier.
        _model: Underlying SentenceTransformer instance.
    """

    def __init__(self, model_name: str | None = None) -> None:
        """
        Initialise and load the sentence-transformers model.

        Args:
            model_name: Override the model from settings.  Defaults to
                        ``settings.embedding_model`` (all-MiniLM-L6-v2).
        """
        self.model_name = model_name or settings.embedding_model
        logger.info("Loading embedding model: %s", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception:
            logger.exception("Failed to load embedding model %s", self.model_name)
            raise

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Encode a list of document chunks into dense vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, one per input text.
        """
        if not texts:
            return []
        logger.debug("Embedding %d documents.", len(texts))
        try:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception:
            logger.exception("Error embedding documents")
            raise

    def embed_query(self, text: str) -> list[float]:
        """
        Encode a single query string into a dense vector.

        Args:
            text: Query string to embed.

        Returns:
            Float vector representing the query.
        """
        logger.debug("Embedding query: %.80s", text)
        try:
            embedding = self._model.encode(
                [text],
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embedding[0].tolist()
        except Exception:
            logger.exception("Error embedding query")
            raise


# ── Singleton ─────────────────────────────────────────────────────────────────
_embeddings_instance: SentenceTransformerEmbeddings | None = None


def get_embeddings() -> SentenceTransformerEmbeddings:
    """
    Return the process-wide embedding model singleton.

    Lazy-initialises on first call so imports don't trigger a model load.

    Returns:
        Shared SentenceTransformerEmbeddings instance.
    """
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = SentenceTransformerEmbeddings()
    return _embeddings_instance
