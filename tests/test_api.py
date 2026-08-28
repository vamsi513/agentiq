"""
tests/test_api.py — Tests for API middleware: rate limiting and request-ID propagation.
"""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from api.main import _RATE_LIMIT, _rate_counters, app

client = TestClient(app)


def _clear_rate_counters():
    _rate_counters.clear()


# ── Health endpoint ───────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_includes_version(self):
        response = client.get("/health")
        assert "version" in response.json()

    def test_health_is_minimal_and_does_not_leak_provider_config(self):
        """/health is public and unauthenticated — it must not disclose which
        providers are configured, only that the service is up."""
        data = client.get("/health").json()
        assert data["status"] == "ok"
        assert "openai_configured" not in data
        assert "tavily_configured" not in data


# ── Request-ID middleware ─────────────────────────────────────────────────────

class TestRequestIdMiddleware:
    def test_response_contains_request_id_header(self):
        response = client.get("/health")
        assert "x-request-id" in response.headers

    def test_custom_request_id_is_echoed_back(self):
        response = client.get("/health", headers={"X-Request-ID": "my-custom-id-123"})
        assert response.headers["x-request-id"] == "my-custom-id-123"

    def test_auto_generated_request_id_is_uuid_format(self):
        import re
        response = client.get("/health")
        rid = response.headers["x-request-id"]
        assert re.match(
            r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
            rid,
        ), f"Expected UUID, got: {rid}"


# ── Rate limiting ─────────────────────────────────────────────────────────────

class TestRateLimiting:
    def setup_method(self):
        _clear_rate_counters()

    def test_requests_within_limit_succeed(self):
        """First N requests from the same IP should not be rate-limited."""
        from unittest.mock import MagicMock

        from api.main import _check_rate_limit

        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.1"
        mock_request.headers = {}

        for _ in range(_RATE_LIMIT - 1):
            _check_rate_limit(mock_request)

    def test_request_over_limit_raises_429(self):
        """The (_RATE_LIMIT + 1)th request from the same IP raises HTTP 429."""
        from fastapi import HTTPException

        from api.main import _check_rate_limit

        mock_request = MagicMock()
        mock_request.client.host = "10.0.0.2"
        mock_request.headers = {}

        for _ in range(_RATE_LIMIT):
            _check_rate_limit(mock_request)

        with pytest.raises(HTTPException) as exc_info:
            _check_rate_limit(mock_request)

        assert exc_info.value.status_code == 429

    def test_different_ips_tracked_separately(self):
        """Each IP has its own independent counter."""
        from fastapi import HTTPException

        from api.main import _check_rate_limit

        req_a = MagicMock()
        req_a.client.host = "10.0.0.3"
        req_a.headers = {}

        req_b = MagicMock()
        req_b.client.host = "10.0.0.4"
        req_b.headers = {}

        for _ in range(_RATE_LIMIT):
            _check_rate_limit(req_a)

        with pytest.raises(HTTPException):
            _check_rate_limit(req_a)

        # IP B should still pass — has its own empty counter
        _check_rate_limit(req_b)

    def test_x_forwarded_for_used_when_present(self):
        """Rate limiter uses X-Forwarded-For header over request.client.host."""
        from unittest.mock import MagicMock

        from api.main import _real_client_ip

        mock_request = MagicMock()
        mock_request.client.host = "127.0.0.1"
        mock_request.headers = {"X-Forwarded-For": "203.0.113.42, 10.0.0.1"}

        ip = _real_client_ip(mock_request)
        assert ip == "203.0.113.42"

    def test_fallback_to_client_host_when_no_forwarded_header(self):
        """Falls back to request.client.host when X-Forwarded-For is absent."""
        from unittest.mock import MagicMock

        from api.main import _real_client_ip

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.5"
        mock_request.headers = {}

        ip = _real_client_ip(mock_request)
        assert ip == "192.168.1.5"

    def test_ip_table_bounded_by_max_tracked_ips(self):
        """Rate counter table never grows beyond _MAX_TRACKED_IPS entries."""
        from api.main import _MAX_TRACKED_IPS, _check_rate_limit

        for i in range(_MAX_TRACKED_IPS + 50):
            req = MagicMock()
            req.client.host = f"192.168.{i // 256}.{i % 256}"
            req.headers = {}
            _check_rate_limit(req)

        assert len(_rate_counters) <= _MAX_TRACKED_IPS
