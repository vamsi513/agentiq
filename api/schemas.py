"""
api/schemas.py — Pydantic v2 request and response models for AgentIQ API.

All API boundaries are validated through these models so that invalid
input is rejected before it reaches the agent graph.
"""

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Request body for the POST /chat endpoint.

    Attributes:
        query: The user's natural language question.
        session_id: Stable session identifier for conversation memory.
                    If omitted, a new session ID is generated server-side.
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question or message.",
    )
    session_id: str | None = Field(
        default=None,
        description="Session ID for multi-turn memory. Auto-generated if omitted.",
    )


class SourceItem(BaseModel):
    """
    A single cited source returned with an agent response.

    Attributes:
        title: Document or page title.
        url: Source URL (empty string for local documents without a URL).
        content: Short excerpt from the source.
        score: Relevance score (0–1).
        type: One of ``retrieval``, ``web_search``, or ``direct``.
    """

    title: str
    url: str
    content: str
    score: float = 0.0
    type: str = "direct"


class ChatResponse(BaseModel):
    """
    Response body for the POST /chat endpoint (non-streaming).

    Attributes:
        answer: The agent's generated answer.
        sources: List of cited sources used to produce the answer.
        route_decision: Which tool the router selected.
        session_id: The session ID used (echoed back for client tracking).
        turn_count: How many turns have occurred in this session.
        retrieval_score: Top retrieval similarity score (0 if not retrieval).
    """

    answer: str
    sources: list[SourceItem] = []
    route_decision: str = "direct"
    session_id: str
    turn_count: int = 0
    retrieval_score: float = 0.0


class HealthResponse(BaseModel):
    """
    Response body for the GET /health endpoint.

    Deliberately minimal — this is a public, unauthenticated liveness check
    (hit by anyone, including the EC2 deploy script's health-check loop), so
    it reports only that the service is up, not which providers are
    configured. Nothing in either frontend actually reads provider-config
    flags over HTTP; app.py checks settings directly since it runs
    in-process.

    Attributes:
        status: Always ``"ok"`` when the service is running.
        version: API version string.
    """

    status: str = "ok"
    version: str = "1.0.0"


class StreamChunk(BaseModel):
    """
    A single Server-Sent Event chunk for streaming responses.

    Attributes:
        type: Event type — ``"token"``, ``"sources"``, ``"done"``, or ``"error"``.
        data: Payload — a token string, sources list, or error message.
    """

    type: str
    data: Any
