"""
tests/test_tools.py — Unit tests for the Tavily web search tool.

Tests cover:
- Fallback behaviour when API key is missing
- Fallback behaviour when TavilyClient raises an exception
- Result normalisation for valid API responses
- The WebSearchResult dataclass serialisation

All Tavily API calls are mocked — no real network requests are made.
"""

import pytest
from unittest.mock import MagicMock, patch


# ── WebSearchResult dataclass ─────────────────────────────────────────────────

class TestWebSearchResult:
    """Tests for the WebSearchResult dataclass and its to_dict method."""

    def test_to_dict_all_fields(self):
        """to_dict returns all four expected keys."""
        from tools.web_search import WebSearchResult

        result = WebSearchResult(
            title="Test Title",
            url="https://example.com",
            content="Test content snippet.",
            score=0.92,
        )
        d = result.to_dict()

        assert d["title"] == "Test Title"
        assert d["url"] == "https://example.com"
        assert d["content"] == "Test content snippet."
        assert d["score"] == 0.92

    def test_default_score_is_zero(self):
        """score defaults to 0.0 when not provided."""
        from tools.web_search import WebSearchResult

        result = WebSearchResult(title="T", url="", content="C")
        assert result.score == 0.0


# ── Fallback result ───────────────────────────────────────────────────────────

class TestFallbackResult:
    """Tests for the _fallback_result sentinel constructor."""

    def test_fallback_has_is_fallback_flag(self):
        """Fallback result always carries is_fallback=True."""
        from tools.web_search import _fallback_result

        fb = _fallback_result("some query", reason="test")
        assert fb["is_fallback"] is True

    def test_fallback_score_is_zero(self):
        """Fallback score is 0.0."""
        from tools.web_search import _fallback_result

        fb = _fallback_result("q")
        assert fb["score"] == 0.0

    def test_fallback_url_is_empty(self):
        """Fallback URL is an empty string (no fabricated URL)."""
        from tools.web_search import _fallback_result

        fb = _fallback_result("q")
        assert fb["url"] == ""

    def test_fallback_contains_reason(self):
        """Fallback content string includes the reason text."""
        from tools.web_search import _fallback_result

        fb = _fallback_result("q", reason="rate_limit")
        assert "rate_limit" in fb["content"]


# ── web_search function ───────────────────────────────────────────────────────

class TestWebSearch:
    """Tests for the main web_search() function."""

    def test_returns_fallback_when_no_api_key(self):
        """Returns a single fallback result when TAVILY_API_KEY is not set."""
        from tools import web_search as ws_module

        # Reset cached client so our patched settings take effect
        ws_module._client = None

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = False
            mock_settings.max_web_results = 5

            results = ws_module.web_search("what is RAG?")

        assert len(results) == 1
        assert results[0]["is_fallback"] is True

    def test_returns_fallback_on_client_exception(self):
        """Returns fallback (does not raise) when TavilyClient.search throws."""
        from tools import web_search as ws_module

        mock_client = MagicMock()
        mock_client.search.side_effect = ConnectionError("network error")
        ws_module._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5

            results = ws_module.web_search("latest AI news")

        # Must not raise; must return a fallback
        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].get("is_fallback") is True

        # Reset
        ws_module._client = None

    def test_normalises_valid_api_response(self):
        """Normalises a realistic Tavily API response into expected dict shape."""
        from tools import web_search as ws_module

        fake_response = {
            "results": [
                {
                    "title": "What is RAG?",
                    "url": "https://example.com/rag",
                    "content": "RAG stands for Retrieval-Augmented Generation.",
                    "score": 0.87,
                },
                {
                    "title": "RAG overview",
                    "url": "https://example.com/rag2",
                    "content": "RAG combines retrieval and generation.",
                    "score": 0.75,
                },
            ]
        }

        mock_client = MagicMock()
        mock_client.search.return_value = fake_response
        ws_module._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5

            results = ws_module.web_search("what is RAG?")

        assert len(results) == 2
        assert results[0]["title"] == "What is RAG?"
        assert results[0]["url"] == "https://example.com/rag"
        assert results[0]["score"] == 0.87
        assert "is_fallback" not in results[0]

        # Reset
        ws_module._client = None

    def test_returns_fallback_when_empty_results(self):
        """Returns fallback when Tavily returns an empty results list."""
        from tools import web_search as ws_module

        mock_client = MagicMock()
        mock_client.search.return_value = {"results": []}
        ws_module._client = mock_client

        with patch("tools.web_search.settings") as mock_settings:
            mock_settings.is_tavily_configured.return_value = True
            mock_settings.max_web_results = 5

            results = ws_module.web_search("obscure query")

        assert len(results) == 1
        assert results[0]["is_fallback"] is True

        # Reset
        ws_module._client = None
