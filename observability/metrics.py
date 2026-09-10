"""
observability/metrics.py — OpenTelemetry metrics wired to a Prometheus
scrape endpoint.

setup_metrics(app):
  - builds an OTel MeterProvider whose reader is a PrometheusMetricReader
    (so every OTel instrument below is exported in Prometheus text format)
  - auto-instruments the FastAPI app for HTTP request count / duration /
    in-flight
  - mounts GET /metrics returning the Prometheus exposition

All of this is a no-op-safe add-on: if OTel isn't installed the import
fails loudly at startup, but nothing here changes agent behaviour, and
the deployed services simply don't scrape the endpoint.

Custom agent instruments (recorded from the code paths that matter):
  agentiq_turns_total{route,cached}        - completed agent turns
  agentiq_turn_duration_seconds            - end-to-end turn latency
  agentiq_node_duration_seconds{node}      - per-graph-node latency
  agentiq_tool_failures_total{tool,reason} - tool fallbacks that fired
  agentiq_cache_events_total{result}       - response-cache hit / miss
"""

import logging

logger = logging.getLogger(__name__)

_meter = None
turns_total = None
turn_duration = None
node_duration = None
tool_failures_total = None
cache_events_total = None


def setup_metrics(app) -> None:
    """Configure OTel + Prometheus and mount /metrics on the given app."""
    global _meter, turns_total, turn_duration, node_duration
    global tool_failures_total, cache_events_total

    from fastapi import Response
    from opentelemetry import metrics as otel_metrics
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import SERVICE_NAME, Resource
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    reader = PrometheusMetricReader()
    provider = MeterProvider(
        resource=Resource.create({SERVICE_NAME: "agentiq"}),
        metric_readers=[reader],
    )
    otel_metrics.set_meter_provider(provider)
    _meter = otel_metrics.get_meter("agentiq")

    turns_total = _meter.create_counter(
        "agentiq_turns_total", description="Completed agent turns", unit="1",
    )
    turn_duration = _meter.create_histogram(
        "agentiq_turn_duration_seconds", description="End-to-end agent turn latency", unit="s",
    )
    node_duration = _meter.create_histogram(
        "agentiq_node_duration_seconds", description="Per graph-node latency", unit="s",
    )
    tool_failures_total = _meter.create_counter(
        "agentiq_tool_failures_total", description="Tool fallbacks that fired", unit="1",
    )
    cache_events_total = _meter.create_counter(
        "agentiq_cache_events_total", description="Response cache hit/miss", unit="1",
    )

    FastAPIInstrumentor.instrument_app(app)

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    logger.info("OpenTelemetry metrics configured; /metrics endpoint mounted.")


# ── Recording helpers (safe no-ops until setup_metrics has run) ──────────────

def record_turn(route: str, cached: bool, duration_s: float) -> None:
    if turns_total is not None:
        turns_total.add(1, {"route": route, "cached": str(cached).lower()})
        turn_duration.record(duration_s, {"route": route})


def record_node(node: str, duration_s: float) -> None:
    if node_duration is not None:
        node_duration.record(duration_s, {"node": node})


def record_tool_failure(tool: str, reason: str) -> None:
    if tool_failures_total is not None:
        tool_failures_total.add(1, {"tool": tool, "reason": reason})


def record_cache_event(result: str) -> None:
    if cache_events_total is not None:
        cache_events_total.add(1, {"result": result})
