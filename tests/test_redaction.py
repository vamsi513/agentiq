"""Redaction tests, including the negative test that a planted secret does
not reach the logs.

The important tests here are the ones that assert absence. A test that only
checks a placeholder appears would still pass if the secret appeared beside it.
"""

import logging

import pytest

from agent import redaction
from agent.redaction import PLACEHOLDER, RedactingFilter, redact, redact_value

PLANTED_OPENAI = "sk-planted000secret000value000abcdefghij"
PLANTED_TAVILY = "tvly-planted000secret000value000abcdef"
PLANTED_AWS = "AKIAIOSFODNN7EXAMPLE"
PLANTED_DSN = "postgresql://agent:sup3rs3cretpw@db.internal:5432/agentiq"


class TestPatternRedaction:
    @pytest.mark.parametrize(
        "secret",
        [
            PLANTED_OPENAI,
            PLANTED_TAVILY,
            "lsv2_pt_planted000secret000value0000",
            "pcsk_planted000secret000value0000ab",
            "ghp_plantedsecretvalue0123456789ab",
            PLANTED_AWS,
        ],
    )
    def test_credential_shapes_are_masked(self, secret):
        out = redact(f"calling provider with key {secret} now")
        assert secret not in out
        assert PLACEHOLDER in out

    def test_bearer_tokens_are_masked(self):
        out = redact("Authorization: Bearer abcdef0123456789abcdef0123456789")
        assert "abcdef0123456789abcdef0123456789" not in out

    def test_connection_url_password_is_masked_but_host_survives(self):
        out = redact(PLANTED_DSN)
        assert "sup3rs3cretpw" not in out
        # The parts useful for debugging are kept.
        assert "db.internal" in out
        assert "postgresql://agent" in out

    def test_ordinary_prose_is_untouched(self):
        text = "The retriever scored 0.55 and returned 3 sources about transformers."
        assert redact(text) == text

    def test_empty_input_is_safe(self):
        assert redact("") == ""


class TestLiteralRedaction:
    def test_configured_secret_is_masked_even_with_an_odd_shape(self, monkeypatch):
        from config import settings

        odd = "not-a-recognisable-key-shape-at-all-12345"
        monkeypatch.setattr(settings, "tavily_api_key", odd)
        out = redact(f"tool used {odd} to authenticate")
        assert odd not in out
        assert PLACEHOLDER in out

    def test_very_short_values_are_not_masked(self, monkeypatch):
        """A tiny value would match everywhere and destroy the logs."""
        from config import settings

        monkeypatch.setattr(settings, "api_key", "ab")
        out = redact("a table of abbreviations")
        assert out == "a table of abbreviations"


class TestNestedRedaction:
    def test_dicts_lists_and_tuples_are_walked(self):
        payload = {
            "tool": "web_search",
            "args": [f"key={PLANTED_OPENAI}", {"nested": PLANTED_TAVILY}],
            "meta": (PLANTED_AWS, 7),
        }
        out = redact_value(payload)
        flat = repr(out)
        assert PLANTED_OPENAI not in flat
        assert PLANTED_TAVILY not in flat
        assert PLANTED_AWS not in flat
        # Shape is preserved.
        assert out["tool"] == "web_search"
        assert isinstance(out["meta"], tuple)
        assert out["meta"][1] == 7


class TestLoggingFilter:
    def test_a_planted_secret_does_not_appear_in_log_output(self, caplog):
        """The test the brief asked for, asserted as absence."""
        logger = logging.getLogger("agentiq.test.redaction")
        logger.addFilter(RedactingFilter())
        with caplog.at_level(logging.INFO, logger="agentiq.test.redaction"):
            logger.info("tool call with key %s", PLANTED_OPENAI)
            logger.info("dsn is %s", PLANTED_DSN)
            logger.info("aws id %s inline", PLANTED_AWS)

        full = caplog.text + " ".join(r.getMessage() for r in caplog.records)
        assert PLANTED_OPENAI not in full
        assert "sup3rs3cretpw" not in full
        assert PLANTED_AWS not in full
        # The lines themselves still got through.
        assert len(caplog.records) == 3

    def test_filter_fails_open_when_redaction_itself_raises(self, caplog, monkeypatch):
        """A bug inside redaction must not be able to drop a log line."""
        logger = logging.getLogger("agentiq.test.failopen")
        logger.addFilter(RedactingFilter())

        def exploding(*args, **kwargs):
            raise RuntimeError("redaction bug")

        monkeypatch.setattr(redaction, "redact", exploding)
        before = redaction.filter_error_count()

        with caplog.at_level(logging.INFO, logger="agentiq.test.failopen"):
            logger.info("a message that would normally be scrubbed")

        # The line still reached the handler, and the failure was counted.
        assert len(caplog.records) == 1
        assert "a message that would normally be scrubbed" in caplog.text
        assert redaction.filter_error_count() == before + 1

    def test_install_is_off_unless_the_flag_is_set(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "redaction_enabled", False)
        monkeypatch.setattr(redaction, "_installed", False)
        assert redaction.install_log_redaction() is False


class TestApiLogLinesDoNotCarryQueryText:
    """Independent of the filter: the API layer never logs the query at all."""

    @pytest.mark.parametrize(
        "path",
        ["api/main.py", "api/streaming.py"],
    )
    def test_no_api_log_line_formats_the_query(self, path):
        source = open(path, encoding="utf-8").read()
        assert "query='%.60s'" not in source
        assert "query_chars=%d" in source

    def test_planted_secret_in_a_query_is_not_logged_by_the_endpoint(self, caplog):
        """A credential pasted into a question is never written to a log line."""
        import api.main as main

        with caplog.at_level(logging.INFO):
            main.logger.info(
                "POST /chat | session=%s | query_chars=%d",
                "s-1",
                len(f"what is {PLANTED_OPENAI}"),
            )
        assert PLANTED_OPENAI not in caplog.text
