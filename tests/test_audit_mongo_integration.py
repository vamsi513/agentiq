"""Integration test for the Mongo audit sink against a real server.

Skipped unless AUDIT_MONGO_TEST_URI is set, so the default suite needs no
external service. CI provides it through a pinned MongoDB service container,
and locally it can point at a Docker container:

    docker run -d --name agentiq-audit-mongo -p 27017:27017 mongo:7.0.14
    AUDIT_MONGO_TEST_URI=mongodb://localhost:27017 pytest tests/test_audit_mongo_integration.py -v
"""

from __future__ import annotations

import json
import os
import uuid

import pytest

from observability.audit_sinks import MongoSink, utc_now_iso

URI = os.getenv("AUDIT_MONGO_TEST_URI", "")

pytestmark = pytest.mark.skipif(
    not URI, reason="AUDIT_MONGO_TEST_URI is not set, no real MongoDB to test against"
)

PLANTED = "sk-integration000planted000secret000ab"


@pytest.fixture
def live_sink():
    """A sink against the real server, in a database unique to this run."""
    database = f"agentiq_audit_it_{uuid.uuid4().hex[:10]}"
    sink = MongoSink(uri=URI, database=database, queue_size=200, batch_size=20)
    yield sink
    try:
        sink._get_collection().database.client.drop_database(database)
    finally:
        sink.close(timeout=2.0)


def _record(run_id: str, ts: str | None = None, text: str = "a question"):
    return {
        "event": "tool_call",
        "tool": "web_search",
        "args": {"positional": [text], "keyword": {}},
        "duration_ms": 10.0,
        "outcome": "ok",
        "ts": ts or utc_now_iso(),
        "run_id": run_id,
    }


def test_write_read_indexes_and_redaction_against_a_real_server(live_sink):
    """One test covering the whole round trip, so the suite stays quick."""
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    live_sink.write(_record(run_id, ts="2026-10-01T00:00:00.000Z"))
    live_sink.write(_record(run_id, ts="2026-10-09T00:00:00.000Z"))
    live_sink.write(_record("other-run", ts="2026-10-09T00:00:00.000Z"))
    assert live_sink.flush(timeout=5.0) == 3
    assert live_sink.stats()["failed"] == 0

    # read by run id
    found = live_sink.query(run_id=run_id)
    assert len(found) == 2
    assert {r["run_id"] for r in found} == {run_id}
    assert "_id" not in found[0]

    # newest first
    assert found[0]["ts"] > found[1]["ts"]

    # time range with a limit
    windowed = live_sink.query(run_id=run_id, start="2026-10-05T00:00:00.000Z", limit=1)
    assert len(windowed) == 1
    assert windowed[0]["ts"] == "2026-10-09T00:00:00.000Z"

    # indexes really exist on the server
    names = set(live_sink._get_collection().index_information())
    assert "run_id_idx" in names
    assert "ts_idx" in names
    assert "created_at_ttl_idx" not in names  # off by default

    # a secret in a record never lands in a stored document
    live_sink.write(_record(run_id, text=f"use {PLANTED}"))
    live_sink.flush(timeout=5.0)
    assert PLANTED not in json.dumps(live_sink.query(run_id=run_id))
