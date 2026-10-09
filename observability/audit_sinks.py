"""Storage backends for the tool call audit log.

Two sinks behind one small interface:

LoggingSink is the default and is what this module did before any backend
existed: one JSON line per record on the `agentiq.audit` logger. Selected when
AUDIT_BACKEND is unset or set to "log".

MongoSink stores one document per record in a collection. Selected with
AUDIT_BACKEND=mongo. Writes go through a bounded queue drained by a single
daemon thread, so an unreachable database can never block or fail an agent
request: a full queue drops the record and counts it, and a failing write is
logged once and counted rather than retried forever.

Records are already redacted before they reach a sink, because
observability.audit builds them through agent.redaction. MongoSink redacts
again in its worker thread before storing, since a stored document is durable
and a sink is a public interface that anything could write to. redact is
idempotent, so the second pass changes nothing for records built the normal
way. A sink adds nothing of its own from the request.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from datetime import UTC, datetime
from typing import Any, Iterable, Protocol

logger = logging.getLogger("agentiq.audit")

BACKEND_LOG = "log"
BACKEND_MONGO = "mongo"

# Hard cap on a rendered log line, matching the previous behaviour.
_MAX_RECORD_CHARS = 2000


def utc_now_iso() -> str:
    """Timestamp for a record, UTC ISO 8601 with a trailing Z."""
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def current_run_id() -> str | None:
    """Request id bound by the API middleware, or None outside a request.

    api/main.py binds request_id into structlog's contextvars for every HTTP
    request, so this reads an identifier that already exists rather than
    threading a new argument through the graph.
    """
    try:
        import structlog

        value = structlog.contextvars.get_contextvars().get("request_id")
    except Exception:  # pragma: no cover - structlog is a hard dependency
        return None
    return str(value) if value else None


def _redacted(record: dict[str, Any]) -> dict[str, Any]:
    """Final redaction pass before a record is stored.

    Records built by observability.audit are already redacted, so this is
    normally a no-op: redact is idempotent. It exists because a sink is a
    public interface, and anything written through it ends up durable. Running
    it here, in the worker thread, keeps the cost off the request path.
    """
    from agent.redaction import redact_value

    try:
        cleaned = redact_value(record)
        return cleaned if isinstance(cleaned, dict) else record
    except Exception:  # pragma: no cover - redaction must not block a write
        logger.warning("could not redact an audit record before storing it")
        return record


class AuditSink(Protocol):
    """Where audit records go."""

    def write(self, record: dict[str, Any]) -> None:
        """Store one record. Must not raise and must not block a request."""

    def query(
        self,
        run_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Read records back, newest first."""

    def close(self, timeout: float = 2.0) -> None:
        """Best effort flush and release resources."""


class LoggingSink:
    """One JSON line per record, byte for byte as before plus ts and run_id."""

    name = BACKEND_LOG

    def write(self, record: dict[str, Any]) -> None:
        import json

        try:
            line = json.dumps(record, default=str, separators=(",", ":"))
            if len(line) > _MAX_RECORD_CHARS:
                line = json.dumps(
                    {
                        "event": "tool_call",
                        "tool": record.get("tool"),
                        "outcome": record.get("outcome"),
                        "duration_ms": record.get("duration_ms"),
                        "truncated": True,
                    },
                    separators=(",", ":"),
                )
            logger.info(line)
        except Exception:  # pragma: no cover - auditing must never break a call
            logger.warning("audit record could not be written", exc_info=True)

    def query(
        self,
        run_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Not supported. Log lines are not a queryable store.

        Returns an empty list rather than raising, so a caller can treat the
        read path uniformly across backends.
        """
        logger.debug("audit query is not supported by the logging backend")
        return []

    def close(self, timeout: float = 2.0) -> None:
        return None


class MongoSink:
    """One document per record, written off the request path.

    The queue is the whole point. emit() runs inside a tool call, so it must
    never wait on a socket. Writes are handed to a bounded queue and drained by
    one daemon thread; everything that can go wrong with the database is
    absorbed there.
    """

    name = BACKEND_MONGO

    def __init__(
        self,
        uri: str,
        database: str,
        collection: str = "tool_calls",
        queue_size: int = 1000,
        batch_size: int = 50,
        drain_interval: float = 0.5,
        ttl_days: int = 0,
        client: Any = None,
        connect_timeout_ms: int = 2000,
    ) -> None:
        self._uri = uri
        self._database = database
        self._collection_name = collection
        self._batch_size = max(1, batch_size)
        self._drain_interval = max(0.01, drain_interval)
        self._ttl_days = max(0, ttl_days)
        self._connect_timeout_ms = connect_timeout_ms

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(1, queue_size))
        self._dropped = 0
        self._failed = 0
        self._written = 0
        self._warned = False
        self._lock = threading.Lock()
        self._stopping = threading.Event()

        self._client = client
        self._indexes_ready = False

        self._worker = threading.Thread(
            target=self._drain_loop, name="agentiq-audit-mongo", daemon=True
        )
        self._worker.start()

    # ── connection ──────────────────────────────────────────────────────────

    def _get_collection(self) -> Any:
        if self._client is None:
            from pymongo import MongoClient

            self._client = MongoClient(
                self._uri,
                serverSelectionTimeoutMS=self._connect_timeout_ms,
                connectTimeoutMS=self._connect_timeout_ms,
                # Without this a socket that stops responding mid-operation
                # blocks the worker indefinitely.
                socketTimeoutMS=self._connect_timeout_ms,
            )
        collection = self._client[self._database][self._collection_name]
        if not self._indexes_ready:
            self._ensure_indexes(collection)
        return collection

    def _ensure_indexes(self, collection: Any) -> None:
        """Index run_id and ts. The TTL index is optional and off by default."""
        collection.create_index("run_id", name="run_id_idx")
        collection.create_index("ts", name="ts_idx")
        if self._ttl_days > 0:
            collection.create_index(
                "created_at",
                name="created_at_ttl_idx",
                expireAfterSeconds=self._ttl_days * 86400,
            )
        self._indexes_ready = True

    # ── write path ──────────────────────────────────────────────────────────

    def write(self, record: dict[str, Any]) -> None:
        """Hand the record to the queue. Never blocks, never raises."""
        try:
            self._queue.put_nowait(dict(record))
        except queue.Full:
            with self._lock:
                self._dropped += 1
            self._count_failure("queue_full")
            self._warn_once("audit queue is full, dropping records")

    def _drain_loop(self) -> None:  # pragma: no cover - exercised via flush()
        while not self._stopping.is_set():
            self._drain_once(block=True)
        self._drain_once(block=False)

    def _drain_once(self, block: bool) -> int:
        batch: list[dict[str, Any] | None] = []
        try:
            if block:
                batch.append(self._queue.get(timeout=self._drain_interval))
            while len(batch) < self._batch_size:
                batch.append(self._queue.get_nowait())
        except queue.Empty:
            pass
        if not batch:
            return 0
        # None is the wake-up sentinel close() uses to unblock the get above.
        records = [r for r in batch if r is not None]
        if records:
            self._insert(records)
        for _ in batch:
            self._queue.task_done()
        return len(records)

    def _insert(self, batch: list[dict[str, Any]]) -> None:
        """Insert a batch. Absorbs every database error."""
        try:
            collection = self._get_collection()
            collection.insert_many([_redacted(r) for r in batch], ordered=False)
            with self._lock:
                self._written += len(batch)
        except Exception as exc:
            with self._lock:
                self._failed += len(batch)
            self._indexes_ready = False
            self._count_failure(type(exc).__name__)
            # No record contents and no connection string in the message.
            self._warn_once(
                f"audit write to mongo failed and {len(batch)} record(s) were dropped "
                f"({type(exc).__name__})"
            )

    def _count_failure(self, reason: str) -> None:
        try:
            from observability.metrics import record_audit_failure

            record_audit_failure(self.name, reason)
        except Exception:  # pragma: no cover - metrics are best-effort
            pass

    def _warn_once(self, message: str) -> None:
        """Warn the first time only, so a dead database cannot flood the log."""
        with self._lock:
            if self._warned:
                return
            self._warned = True
        logger.warning(message)

    # ── read path ───────────────────────────────────────────────────────────

    def query(
        self,
        run_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Records matching a run id and/or a ts range, newest first."""
        criteria: dict[str, Any] = {}
        if run_id is not None:
            criteria["run_id"] = run_id
        window: dict[str, Any] = {}
        if start is not None:
            window["$gte"] = start
        if end is not None:
            window["$lte"] = end
        if window:
            criteria["ts"] = window

        collection = self._get_collection()
        cursor = collection.find(criteria, {"_id": False}).sort("ts", -1).limit(max(1, limit))
        return list(cursor)

    # ── lifecycle and introspection ─────────────────────────────────────────

    def flush(self, timeout: float = 2.0) -> int:
        """Drain the queue synchronously. Used by tests and by close()."""
        deadline = time.monotonic() + timeout
        drained = 0
        while time.monotonic() < deadline:
            moved = self._drain_once(block=False)
            drained += moved
            if moved == 0:
                break
        return drained

    def close(self, timeout: float = 2.0) -> None:
        """Stop the worker, giving queued records one bounded chance to land.

        The final drain happens on the worker thread, not here. Doing it on the
        caller's thread would make shutdown as slow as one hanging insert,
        however short the timeout. The worker is a daemon, so if it is still
        stuck in a write when the timeout expires the process can still exit.
        """
        self._stopping.set()
        try:
            # Unblock the worker if it is waiting on an empty queue, so it
            # reaches its final drain immediately rather than after a full
            # drain_interval.
            self._queue.put_nowait(None)
        except queue.Full:  # pragma: no cover - a full queue wakes it anyway
            pass
        self._worker.join(timeout=timeout)
        if self._client is not None:
            try:
                self._client.close()
            except Exception:  # pragma: no cover - close is best effort
                pass

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "written": self._written,
                "dropped": self._dropped,
                "failed": self._failed,
                "queued": self._queue.qsize(),
            }


def build_sink(settings_obj: Any = None) -> AuditSink:
    """Build the sink the configuration selects. Falls back to logging."""
    if settings_obj is None:
        from config import settings as settings_obj  # type: ignore[no-redef]

    backend = str(getattr(settings_obj, "audit_backend", BACKEND_LOG) or BACKEND_LOG).lower()
    if backend != BACKEND_MONGO:
        return LoggingSink()

    uri = getattr(settings_obj, "audit_mongo_uri", "")
    if not uri:
        logger.warning(
            "AUDIT_BACKEND is mongo but AUDIT_MONGO_URI is not set, using the logging backend"
        )
        return LoggingSink()

    try:
        return MongoSink(
            uri=uri,
            database=getattr(settings_obj, "audit_mongo_db", "agentiq_audit"),
            collection=getattr(settings_obj, "audit_mongo_collection", "tool_calls"),
            queue_size=getattr(settings_obj, "audit_queue_size", 1000),
            batch_size=getattr(settings_obj, "audit_queue_batch_size", 50),
            drain_interval=getattr(settings_obj, "audit_drain_interval_seconds", 0.5),
            ttl_days=getattr(settings_obj, "audit_mongo_ttl_days", 0),
        )
    except Exception as exc:  # pragma: no cover - construction is cheap
        logger.warning("could not start the mongo audit sink (%s), using the logging backend", type(exc).__name__)
        return LoggingSink()


def sink_names() -> Iterable[str]:
    return (BACKEND_LOG, BACKEND_MONGO)
