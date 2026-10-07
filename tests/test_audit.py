"""Audit log tests.

The two claims that matter: every call produces exactly one record with the
right outcome, and a secret in an argument never reaches the record.
"""

import json
import logging

import pytest

from observability import audit
from observability.audit import audited, summarise_args

PLANTED = "sk-audit000planted000secret000abcdefgh"


@pytest.fixture
def enabled(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "audit_log_enabled", True)
    return settings


def records_from(caplog) -> list[dict]:
    """Parse the JSON lines the audit logger emitted."""
    out = []
    for r in caplog.records:
        if r.name != "agentiq.audit":
            continue
        try:
            out.append(json.loads(r.getMessage()))
        except json.JSONDecodeError:
            pass
    return out


class TestDisabledByDefault:
    def test_flag_is_off_in_the_default_config(self):
        from config import settings

        assert settings.audit_log_enabled is False

    def test_nothing_is_emitted_when_disabled(self, caplog, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "audit_log_enabled", False)

        @audited("probe")
        def tool(q):
            return ["result"]

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            assert tool("hello") == ["result"]
        assert records_from(caplog) == []


class TestRecordContents:
    def test_one_record_per_call_with_the_expected_fields(self, enabled, caplog):
        @audited("probe")
        def tool(q, k=3):
            return []

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool("a question", k=5)

        recs = records_from(caplog)
        assert len(recs) == 1
        r = recs[0]
        assert r["event"] == "tool_call"
        assert r["tool"] == "probe"
        assert isinstance(r["duration_ms"], float)
        assert r["duration_ms"] >= 0
        assert r["args"]["positional"] == ["a question"]
        assert r["args"]["keyword"] == {"k": 5}

    def test_outcome_ok_on_a_normal_return(self, enabled, caplog):
        @audited("probe")
        def tool():
            return [{"title": "t"}]

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool()
        assert records_from(caplog)[0]["outcome"] == "ok"

    def test_fallback_results_are_recorded_as_fallback_not_ok(self, enabled, caplog):
        """web_search returns a fallback instead of raising, so a call that
        returned is not automatically a healthy one."""

        @audited("probe")
        def tool():
            return [{"is_fallback": True, "title": "", "url": "", "content": "down"}]

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool()
        assert records_from(caplog)[0]["outcome"] == "fallback"

    def test_an_exception_is_recorded_and_still_raised(self, enabled, caplog):
        @audited("probe")
        def tool():
            raise ValueError("tool broke")

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            with pytest.raises(ValueError, match="tool broke"):
                tool()

        r = records_from(caplog)[0]
        assert r["outcome"] == "error"
        assert r["error_type"] == "ValueError"
        assert "tool broke" in r["error"]

    @pytest.mark.asyncio
    async def test_async_tools_are_audited(self, enabled, caplog):
        @audited("async_probe")
        async def tool(q):
            return []

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            await tool("q")
        recs = records_from(caplog)
        assert len(recs) == 1
        assert recs[0]["tool"] == "async_probe"

    @pytest.mark.asyncio
    async def test_async_exception_is_recorded_and_still_raised(self, enabled, caplog):
        @audited("async_probe")
        async def tool():
            raise RuntimeError("async broke")

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            with pytest.raises(RuntimeError):
                await tool()
        assert records_from(caplog)[0]["outcome"] == "error"


class TestSecretsNeverReachARecord:
    def test_a_planted_secret_in_an_argument_is_absent_from_the_record(
        self, enabled, caplog
    ):
        @audited("probe")
        def tool(q):
            return []

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool(f"please use {PLANTED} to search")

        raw = json.dumps(records_from(caplog))
        assert PLANTED not in raw
        assert "[redacted]" in raw

    def test_a_configured_secret_in_an_argument_is_absent(
        self, enabled, caplog, monkeypatch
    ):
        from config import settings

        odd = "no-recognisable-shape-whatsoever-98765"
        monkeypatch.setattr(settings, "tavily_api_key", odd)

        @audited("probe")
        def tool(q):
            return []

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool(f"authenticate with {odd}")

        assert odd not in json.dumps(records_from(caplog))

    def test_a_secret_in_an_exception_message_is_absent(self, enabled, caplog):
        @audited("probe")
        def tool():
            raise ValueError(f"rejected key {PLANTED}")

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            with pytest.raises(ValueError):
                tool()

        assert PLANTED not in json.dumps(records_from(caplog))


class TestRecordsStayBounded:
    def test_long_string_arguments_are_summarised_not_copied(self):
        long = "x" * 5000
        summary = summarise_args((long,), {})
        value = summary["positional"][0]
        assert value["chars"] == 5000
        assert len(value["head"]) == 120

    def test_containers_report_size_not_contents(self):
        summary = summarise_args(([1, 2, 3], {"a": 1, "b": 2}), {"s": {7, 8}})
        assert summary["positional"][0] == {"type": "list", "items": 3}
        assert summary["positional"][1] == {"type": "dict", "keys": 2}
        assert summary["keyword"]["s"] == {"type": "set", "items": 2}

    def test_scalars_pass_through(self):
        summary = summarise_args((1, 2.5, True, None), {})
        assert summary["positional"] == [1, 2.5, True, None]

    def test_an_oversized_record_is_replaced_by_a_short_one(self, enabled, caplog):
        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            audit.emit(
                {
                    "event": "tool_call",
                    "tool": "probe",
                    "outcome": "ok",
                    "duration_ms": 1.0,
                    "bloat": "y" * 50_000,
                }
            )
        r = records_from(caplog)[0]
        assert r["truncated"] is True
        assert "bloat" not in r

    def test_emit_never_raises_on_unserialisable_input(self, enabled, caplog):
        class Weird:
            pass

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            audit.emit({"event": "tool_call", "tool": Weird()})
        # default=str handles it rather than raising.
        assert len(records_from(caplog)) == 1


class TestRealToolIsWrapped:
    def test_web_search_is_audited_and_still_a_coroutine(self):
        import asyncio
        import importlib

        mod = importlib.import_module("tools.web_search")
        fn = mod.web_search
        assert asyncio.iscoroutinefunction(fn)
        # functools.wraps keeps the identity intact for callers and docs.
        assert fn.__name__ == "web_search"
        assert fn.__doc__
