"""
config.py — Central configuration loader for AgentIQ.

Loads all settings from environment variables (via python-dotenv) and
exposes them as typed attributes on a singleton Settings object.  Every
module that needs a configuration value should import from here rather
than calling os.getenv() directly, so the entire config surface is
visible in one place.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ── Bootstrap: load .env before anything else accesses os.environ ────────────
_ENV_FILE = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_ENV_FILE, override=False)

# ── Module logger ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class Settings:
    """
    Centralised settings object populated from environment variables.

    Attributes:
        openai_api_key: OpenAI API key for gpt-4o-mini.
        tavily_api_key: Tavily Search API key.
        openai_model: OpenAI model identifier (default: gpt-4o-mini).
        embedding_model: Sentence-transformers model name.
        max_tokens: Maximum tokens for LLM responses.
        temperature: Sampling temperature (0–1).
        api_host: FastAPI bind host.
        api_port: FastAPI bind port.
        top_k_retrieval: Number of FAISS results to retrieve.
        max_web_results: Number of Tavily results to fetch.
        log_level: Logging level string.
        faiss_index_path: Filesystem path for persisted FAISS index.
        documents_path: Directory containing source documents.
    """

    def __init__(self) -> None:
        # API keys
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
        self.langsmith_api_key: str = os.getenv("LANGCHAIN_API_KEY", "")
        self.pinecone_api_key: str = os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index: str = os.getenv("PINECONE_INDEX", "agentiq-docs")

        # Model settings
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.embedding_model: str = os.getenv(
            "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
        )
        self.max_tokens: int = int(os.getenv("MAX_TOKENS", "1024"))
        self.temperature: float = float(os.getenv("TEMPERATURE", "0.2"))

        # Server settings
        self.api_host: str = os.getenv("API_HOST", "0.0.0.0")
        self.api_port: int = int(os.getenv("API_PORT", "8000"))

        # Retrieval settings
        self.top_k_retrieval: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
        self.max_web_results: int = int(os.getenv("MAX_WEB_RESULTS", "5"))
        self.enable_query_rewriting: bool = os.getenv("ENABLE_QUERY_REWRITING", "false").lower() == "true"

        # Logging
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

        # Paths (relative to this file's directory)
        _base = Path(__file__).parent
        self.faiss_index_path: Path = _base / "faiss_index"
        self.documents_path: Path = _base / "retrieval" / "documents"

        self._validate()

    def _validate(self) -> None:
        """
        Warn (but do not crash) if required API keys are missing.

        Missing keys only matter at runtime when the relevant tool is invoked,
        so we log warnings here instead of raising exceptions so that the app
        can still start and show a helpful error message in the UI.
        """
        if not self.openai_api_key:
            logger.warning(
                "OPENAI_API_KEY is not set. LLM generation will fail."
            )
        if not self.tavily_api_key:
            logger.warning(
                "TAVILY_API_KEY is not set. Web search tool will be disabled."
            )

    def is_openai_configured(self) -> bool:
        """Return True if the OpenAI API key is present."""
        return bool(self.openai_api_key)

    def is_tavily_configured(self) -> bool:
        """Return True if the Tavily API key is present."""
        return bool(self.tavily_api_key)


# ── Singleton ─────────────────────────────────────────────────────────────────
settings = Settings()
logger.debug("Configuration loaded: model=%s", settings.openai_model)
