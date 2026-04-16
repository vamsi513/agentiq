"""
api — FastAPI backend package for AgentIQ.

Exports the FastAPI application instance so it can be imported by uvicorn:
    uvicorn api.main:app --reload
"""

from api.main import app

__all__ = ["app"]
