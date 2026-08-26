"""
api/main.py — FastAPI application for AgentIQ.

Endpoints:
    GET  /health  — Liveness check, returns API version and key status.
    POST /chat    — Non-streaming agent invocation (JSON response).
    POST /chat/stream — Streaming agent invocation (SSE response).

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import hmac
import logging
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.schemas import ChatRequest, ChatResponse, HealthResponse, SourceItem
from api.streaming import run_agent_sync, stream_agent_response
from config import settings

logger = logging.getLogger(__name__)


async def _require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Reject requests that don't carry the configured API key.
    When AGENTIQ_API_KEY is unset the check is skipped (dev/demo mode)."""
    if not settings.api_key:
        return
    if x_api_key and hmac.compare_digest(x_api_key, settings.api_key):
        return
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API key.")


# ── Lifespan (replaces deprecated @app.on_event) ─────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-warm the agent graph and FAISS index on server startup.

    Using the lifespan context manager (FastAPI >= 0.93) instead of the
    deprecated @app.on_event("startup") decorator.
    """
    logger.info("AgentIQ API starting up…")
    try:
        from agent.graph import get_graph
        get_graph()
        logger.info("Agent graph pre-warmed successfully.")
    except Exception as exc:
        logger.warning("Graph pre-warm failed (will retry on first request): %s", exc)

    try:
        from retrieval.vectorstore import get_vectorstore
        get_vectorstore()
        logger.info("FAISS vectorstore pre-loaded successfully.")
    except Exception as exc:
        logger.warning("FAISS pre-load failed (will build on first request): %s", exc)

    yield  # Application runs here

    logger.info("AgentIQ API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgentIQ API",
    description=(
        "Multi-step agentic research assistant powered by LangGraph, "
        "FAISS retrieval, and Tavily web search."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
# allow_credentials=True is incompatible with allow_origins=["*"] per the CORS
# spec — browsers reject credentialed requests to wildcard origins.
# We set allow_credentials=False (the safe default for a public API).
# Tighten allow_origins to specific domains in a production deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


# ── Health endpoint ───────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["System"],
)
async def health() -> HealthResponse:
    """
    Return API health status and configuration flags.

    Returns:
        HealthResponse with status, version, and API key presence flags.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        openai_configured=settings.is_openai_configured(),
        tavily_configured=settings.is_tavily_configured(),
    )


# ── Non-streaming chat endpoint ───────────────────────────────────────────────

@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Run agent and return full response",
    tags=["Agent"],
)
async def chat(request: ChatRequest, _: None = Depends(_require_api_key)) -> ChatResponse:
    """
    Invoke the AgentIQ graph and return the complete answer as JSON.

    Use this endpoint when you don't need streaming (e.g., API integrations,
    testing).  For real-time token streaming use POST /chat/stream.

    Args:
        request: ChatRequest with ``query`` and optional ``session_id``.

    Returns:
        ChatResponse with answer, sources, routing info, and session metadata.

    Raises:
        HTTPException 500: If the agent graph raises an unexpected error.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info("POST /chat | session=%s | query='%.60s'", session_id, request.query)

    try:
        result = await run_agent_sync(request.query, session_id)

        sources = [
            SourceItem(
                title=s.get("title", ""),
                url=s.get("url", ""),
                content=s.get("content", ""),
                score=s.get("score", 0.0),
                type=s.get("type", "direct"),
            )
            for s in result.get("sources", [])
        ]

        return ChatResponse(
            answer=result["answer"],
            sources=sources,
            route_decision=result.get("route_decision", "direct"),
            session_id=session_id,
            turn_count=result.get("turn_count", 0),
            retrieval_score=result.get("retrieval_score", 0.0),
        )

    except Exception as exc:
        logger.exception("Unhandled error in /chat: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Internal agent error. Check server logs for details.",
        )


# ── Streaming chat endpoint ───────────────────────────────────────────────────

@app.post(
    "/chat/stream",
    summary="Stream agent response via SSE",
    tags=["Agent"],
)
async def chat_stream(request: ChatRequest, _: None = Depends(_require_api_key)) -> StreamingResponse:
    """
    Invoke the AgentIQ graph and stream the response as Server-Sent Events.

    The client receives a stream of JSON lines, each with a ``type`` field:
    - ``token``   — one chunk of the answer text.
    - ``sources`` — list of cited sources (sent after all tokens).
    - ``done``    — stream termination signal.
    - ``error``   — error description (followed by ``done``).

    Args:
        request: ChatRequest with ``query`` and optional ``session_id``.

    Returns:
        StreamingResponse with ``text/event-stream`` content type.
    """
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(
        "POST /chat/stream | session=%s | query='%.60s'",
        session_id,
        request.query,
    )

    return StreamingResponse(
        stream_agent_response(request.query, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "X-Session-Id": session_id,
        },
    )
