"""Tests for the audit log storage backends.

Everything here runs with no external service. The Mongo backend is driven
through mongomock, which implements the same client interface, so the sink is
exercised rather than stubbed out. One integration test against a real server
lives in tests/test_audit_mongo_integration.py and is skipped without a
connection variable.
"""

from __future__ import annotations

import json
import logging

import mongomock
import pytest

from observability import audit
from observability.audit_sinks import (
    BACKEND_LOG,
    BACKEND_MONGO,
    LoggingSink,
    MongoSink,
    build_sink,
    current_run_id,
    utc_now_iso,
)

PLANTED = "sk-mongo000planted000secret000abcdefgh"


def record(run_id="run-1", ts="2026-10-09T12:00:00.000Z", tool="web_search", **extra):
    base = {
        "event": "tool_call",
        "tool": tool,
        "args": {"positional": ["a question"], "keyword": {}},
        "duration_ms": 12.5,
        "outcome": "ok",
        "ts": ts,
        "run_id": run_id,
    }
    base.update(extra)
    return base


@pytest.fixture
def sink():
    """A MongoSink backed by mongomock, drained synchronously."""
    s = MongoSink(
        uri="mongodb://localhost:27017",
        database="agentiq_audit_test",
        client=mongomock.MongoClient(),
        queue_size=100,
        batch_size=10,
    )
    yield s
    s.close(timeout=1.0)


class TestDefaultsUnchanged:
    def test_backend_defaults_to_log(self):
        from config import settings

        assert settings.audit_backend == BACKEND_LOG

    def test_build_sink_returns_the_logging_sink_when_unset(self):
        class S:
            audit_backend = "log"

        assert isinstance(build_sink(S()), LoggingSink)

    def test_mongo_without_a_uri_falls_back_to_logging(self, caplog):
        class S:
            audit_backend = "mongo"
            audit_mongo_uri = ""

        with caplog.at_level(logging.WARNING, logger="agentiq.audit"):
            built = build_sink(S())
        assert isinstance(built, LoggingSink)
        assert "AUDIT_MONGO_URI" in caplog.text

    def test_logging_sink_still_writes_one_json_line(self, caplog):
        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            LoggingSink().write(record())
        lines = [r.getMessage() for r in caplog.records if r.name == "agentiq.audit"]
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["event"] == "tool_call"
        assert parsed["tool"] == "web_search"
        assert parsed["ts"] == "2026-10-09T12:00:00.000Z"
        assert parsed["run_id"] == "run-1"

    def test_logging_sink_query_returns_empty_rather_than_raising(self):
        assert LoggingSink().query(run_id="anything") == []


class TestWriteAndReadByRunId:
    def test_a_written_record_can_be_read_back_by_run_id(self, sink):
        sink.write(record(run_id="run-A"))
        assert sink.flush(timeout=1.0) == 1
        found = sink.query(run_id="run-A")
        assert len(found) == 1
        assert found[0]["tool"] == "web_search"
        assert found[0]["run_id"] == "run-A"

    def test_query_by_run_id_excludes_other_runs(self, sink):
        sink.write(record(run_id="run-A"))
        sink.write(record(run_id="run-B"))
        sink.flush(timeout=1.0)
        assert [r["run_id"] for r in sink.query(run_id="run-A")] == ["run-A"]

    def test_mongo_internal_id_is_not_returned(self, sink):
        sink.write(record())
        sink.flush(timeout=1.0)
        assert "_id" not in sink.query()[0]

    def test_stats_count_what_was_written(self, sink):
        for i in range(3):
            sink.write(record(run_id=f"r{i}"))
        sink.flush(timeout=1.0)
        stats = sink.stats()
        assert stats["written"] == 3
        assert stats["dropped"] == 0
        assert stats["failed"] == 0


class TestTimeRangeAndLimit:
    def test_time_range_selects_only_records_inside_the_window(self, sink):
        sink.write(record(run_id="a", ts="2026-10-01T00:00:00.000Z"))
        sink.write(record(run_id="b", ts="2026-10-05T00:00:00.000Z"))
        sink.write(record(run_id="c", ts="2026-10-09T00:00:00.000Z"))
        sink.flush(timeout=1.0)
        found = sink.query(start="2026-10-02T00:00:00.000Z", end="2026-10-06T00:00:00.000Z")
        assert [r["run_id"] for r in found] == ["b"]

    def test_limit_caps_the_number_of_records(self, sink):
        for i in range(10):
            sink.write(record(run_id=f"r{i}", ts=f"2026-10-09T12:00:{i:02d}.000Z"))
        sink.flush(timeout=1.0)
        assert len(sink.query(limit=4)) == 4

    def test_results_come_back_newest_first(self, sink):
        sink.write(record(run_id="old", ts="2026-10-01T00:00:00.000Z"))
        sink.write(record(run_id="new", ts="2026-10-09T00:00:00.000Z"))
        sink.flush(timeout=1.0)
        assert [r["run_id"] for r in sink.query()] == ["new", "old"]

    def test_run_id_and_time_range_combine(self, sink):
        sink.write(record(run_id="a", ts="2026-10-01T00:00:00.000Z"))
        sink.write(record(run_id="a", ts="2026-10-09T00:00:00.000Z"))
        sink.flush(timeout=1.0)
        found = sink.query(run_id="a", start="2026-10-05T00:00:00.000Z")
        assert len(found) == 1
        assert found[0]["ts"] == "2026-10-09T00:00:00.000Z"


class TestIndexes:
    def test_run_id_and_ts_indexes_are_created(self, sink):
        sink.write(record())
        sink.flush(timeout=1.0)
        names = set(sink._get_collection().index_information())
        assert "run_id_idx" in names
        assert "ts_idx" in names

    def test_ttl_index_is_absent_by_default(self, sink):
        sink.write(record())
        sink.flush(timeout=1.0)
        assert "created_at_ttl_idx" not in set(sink._get_collection().index_information())

    def test_ttl_index_is_created_when_configured(self):
        s = MongoSink(
            uri="mongodb://localhost:27017",
            database="agentiq_audit_ttl",
            client=mongomock.MongoClient(),
            ttl_days=7,
        )
        try:
            s.write(record())
            s.flush(timeout=1.0)
            info = s._get_collection().index_information()
            assert "created_at_ttl_idx" in info
            assert info["created_at_ttl_idx"].get("expireAfterSeconds") == 7 * 86400
        finally:
            s.close(timeout=1.0)


class TestUnreachableDatabaseDoesNotBreakARequest:
    def test_a_failing_write_is_absorbed_and_counted(self, caplog):
        class Exploding:
            def __getitem__(self, _):
                return self

            def insert_many(self, *a, **k):
                raise RuntimeError("server selection timed out")

            def create_index(self, *a, **k):
                return None

            def close(self):
                return None

        s = MongoSink(
            uri="mongodb://unreachable:27017",
            database="d",
            client=Exploding(),
        )
        try:
            with caplog.at_level(logging.WARNING, logger="agentiq.audit"):
                s.write(record())
                s.flush(timeout=1.0)
            assert s.stats()["failed"] == 1
            # A warning, and it carries no record contents and no URI.
            assert "failed" in caplog.text
            assert "unreachable:27017" not in caplog.text
            assert "a question" not in caplog.text
        finally:
            s.close(timeout=1.0)

    def test_write_never_raises_even_when_the_queue_is_full(self):
        s = MongoSink(
            uri="mongodb://localhost:27017",
            database="d",
            client=mongomock.MongoClient(),
            queue_size=1,
            drain_interval=60,  # keep the worker idle so the queue fills
        )
        try:
            for _ in range(20):
                s.write(record())  # must not raise
            assert s.stats()["dropped"] >= 1
        finally:
            s.close(timeout=1.0)

    def test_a_dead_database_warns_once_not_per_record(self, caplog):
        class Exploding:
            def __getitem__(self, _):
                return self

            def insert_many(self, *a, **k):
                raise RuntimeError("down")

            def create_index(self, *a, **k):
                return None

            def close(self):
                return None

        s = MongoSink(uri="mongodb://x:27017", database="d", client=Exploding())
        try:
            with caplog.at_level(logging.WARNING, logger="agentiq.audit"):
                for _ in range(5):
                    s.write(record())
                    s.flush(timeout=0.5)
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert len(warnings) == 1, "a dead database must not flood the log"
        finally:
            s.close(timeout=1.0)

    def test_emit_through_a_broken_sink_does_not_raise(self, monkeypatch):
        class Broken:
            def write(self, record):
                raise RuntimeError("sink is broken")

        monkeypatch.setattr(audit, "_sink", Broken())
        try:
            audit.emit(record())  # must not raise
        finally:
            monkeypatch.setattr(audit, "_sink", None)


class TestRedactionReachesTheStoredDocument:
    def test_a_planted_secret_never_reaches_mongo(self, sink, monkeypatch):
        """The decorator builds the record, so redaction must already be applied."""
        from config import settings

        monkeypatch.setattr(settings, "audit_log_enabled", True)
        monkeypatch.setattr(audit, "_sink", sink)

        @audit.audited("probe")
        def tool(q):
            return []

        tool(f"please use {PLANTED} to search")
        sink.flush(timeout=1.0)

        stored = json.dumps(sink.query())
        assert PLANTED not in stored
        assert "[redacted]" in stored

    def test_a_configured_secret_never_reaches_mongo(self, sink, monkeypatch):
        from config import settings

        odd = "no-recognisable-shape-at-all-5551212"
        monkeypatch.setattr(settings, "audit_log_enabled", True)
        monkeypatch.setattr(settings, "tavily_api_key", odd)
        monkeypatch.setattr(audit, "_sink", sink)

        @audit.audited("probe")
        def tool(q):
            return []

        tool(f"authenticate with {odd}")
        sink.flush(timeout=1.0)
        assert odd not in json.dumps(sink.query())

    def test_a_secret_in_an_exception_never_reaches_mongo(self, sink, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "audit_log_enabled", True)
        monkeypatch.setattr(audit, "_sink", sink)

        @audit.audited("probe")
        def tool():
            raise ValueError(f"rejected key {PLANTED}")

        with pytest.raises(ValueError):
            tool()
        sink.flush(timeout=1.0)
        assert PLANTED not in json.dumps(sink.query())


    def test_a_raw_unredacted_record_written_straight_to_the_sink_is_still_masked(self, sink):
        """Defence in depth: the sink is a public interface.

        Records built by the decorator arrive already redacted, but anything
        can call write(), and a stored document is durable.
        """
        sink.write(record(run_id="raw", args={"positional": [f"key {PLANTED}"], "keyword": {}}))
        sink.flush(timeout=1.0)
        stored = json.dumps(sink.query(run_id="raw"))
        assert PLANTED not in stored
        assert "[redacted]" in stored


class TestRecordFields:
    def test_ts_is_utc_iso_8601_with_a_z(self):
        ts = utc_now_iso()
        assert ts.endswith("Z")
        from datetime import datetime

        datetime.fromisoformat(ts.replace("Z", "+00:00"))  # parses

    def test_run_id_is_none_outside_a_request(self):
        import structlog

        structlog.contextvars.clear_contextvars()
        assert current_run_id() is None

    def test_run_id_comes_from_the_bound_request_id(self):
        import structlog

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id="req-xyz")
        try:
            assert current_run_id() == "req-xyz"
        finally:
            structlog.contextvars.clear_contextvars()

    def test_a_record_built_by_the_decorator_carries_ts_and_run_id(self, monkeypatch, caplog):
        from config import settings

        monkeypatch.setattr(settings, "audit_log_enabled", True)
        monkeypatch.setattr(audit, "_sink", LoggingSink())

        @audit.audited("probe")
        def tool():
            return []

        with caplog.at_level(logging.INFO, logger="agentiq.audit"):
            tool()
        parsed = json.loads(
            [r.getMessage() for r in caplog.records if r.name == "agentiq.audit"][0]
        )
        assert parsed["ts"].endswith("Z")
        assert "run_id" in parsed


class TestSinkSelection:
    def test_mongo_backend_is_selected_when_configured(self, monkeypatch):
        built = {}

        class FakeSink:
            def __init__(self, **kwargs):
                built.update(kwargs)

        monkeypatch.setattr("observability.audit_sinks.MongoSink", FakeSink)

        class S:
            audit_backend = "mongo"
            audit_mongo_uri = "mongodb://localhost:27017"
            audit_mongo_db = "adb"
            audit_mongo_collection = "calls"
            audit_queue_size = 7
            audit_queue_batch_size = 3
            audit_drain_interval_seconds = 0.1
            audit_mongo_ttl_days = 5

        build_sink(S())
        assert built["database"] == "adb"
        assert built["collection"] == "calls"
        assert built["queue_size"] == 7
        assert built["ttl_days"] == 5

    def test_an_unknown_backend_name_falls_back_to_logging(self):
        class S:
            audit_backend = "cassandra"

        assert isinstance(build_sink(S()), LoggingSink)

    def test_backend_names_are_the_two_supported(self):
        from observability.audit_sinks import sink_names

        assert set(sink_names()) == {BACKEND_LOG, BACKEND_MONGO}
