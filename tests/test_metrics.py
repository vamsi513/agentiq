"""
tests/test_metrics.py — the OpenTelemetry metrics add-on.

The recording helpers must be safe no-ops before setup_metrics() has run
(so importing agent code in a test or a non-API context never blows up),
and setup_metrics() must mount a working /metrics endpoint that returns
Prometheus text with the custom instruments after traffic.
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient


def test_recording_helpers_are_noops_before_setup():
    # Fresh import state: no meter provider configured yet in this process
    # is not guaranteed (other tests may have run), so this just asserts
    # the helpers never raise regardless.
    from observability import metrics

    metrics.record_turn("direct", cached=False, duration_s=0.1)
    metrics.record_node("router", 0.05)
    metrics.record_tool_failure("web_search", "timeout")
    metrics.record_cache_event("miss")


def test_setup_mounts_metrics_endpoint_with_custom_instruments():
    from observability.metrics import (
        record_cache_event,
        record_tool_failure,
        record_turn,
        setup_metrics,
    )

    app = FastAPI()
    setup_metrics(app)

    record_turn("retrieval", cached=False, duration_s=0.4)
    record_cache_event("hit")
    record_tool_failure("web_search", "rate limited")

    client = TestClient(app)
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.text
    assert "agentiq_turns_total" in body
    assert "agentiq_cache_events_total" in body
    assert "agentiq_tool_failures_total" in body
