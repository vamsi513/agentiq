"""A small broker that hands tools scoped, short lived credential handles.

A tool is given a `CredentialHandle`, which is an opaque id plus a scope and
an expiry. It is not the secret. The tool exchanges the handle for the real
value at the moment it builds its outbound request, by calling `use()`, and
the broker checks the scope and the expiry before returning anything.

What this buys, concretely:

- The secret is never an argument to a tool, so it cannot be captured in an
  audit entry, a traceback frame that logs its arguments, or graph state.
- A handle is useless outside its scope, so a leaked handle for `web_search`
  cannot be replayed against the model provider.
- A handle expires, so one captured in an old log is dead.

What this does not buy: the secret is still in this process's memory, because
something has to make the HTTPS call. This is scoping and blast radius
reduction, not a hardware boundary.

The repr and str of a handle never include the secret, and `__slots__` keeps a
handle from carrying extra attributes someone might stuff a secret into.
"""

from __future__ import annotations

import logging
import secrets
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Scope names. A scope maps to exactly one configured secret.
SCOPE_WEB_SEARCH = "web_search"
SCOPE_LLM = "llm"
SCOPE_EMBEDDINGS = "embeddings"

_SCOPE_TO_SETTING = {
    SCOPE_WEB_SEARCH: "tavily_api_key",
    SCOPE_LLM: "openai_api_key",
    SCOPE_EMBEDDINGS: "openai_api_key",
}


class CredentialError(Exception):
    """Raised when a handle is unknown, expired, or used outside its scope."""


@dataclass(frozen=True)
class CredentialHandle:
    """An opaque reference to a credential. Carries no secret material."""

    __slots__ = ("handle_id", "scope", "expires_at")

    handle_id: str
    scope: str
    expires_at: float

    @property
    def expired(self) -> bool:
        return time.monotonic() >= self.expires_at

    def __repr__(self) -> str:
        state = "expired" if self.expired else "live"
        return f"CredentialHandle(scope={self.scope!r}, id={self.handle_id[:8]}..., {state})"

    __str__ = __repr__


class CredentialBroker:
    """Mints and redeems credential handles."""

    def __init__(self, default_ttl_seconds: float | None = None) -> None:
        self._issued: dict[str, CredentialHandle] = {}
        self._default_ttl = default_ttl_seconds

    @property
    def default_ttl(self) -> float:
        if self._default_ttl is not None:
            return self._default_ttl
        from config import settings

        return settings.credential_ttl_seconds

    def issue(self, scope: str, ttl_seconds: float | None = None) -> CredentialHandle:
        """Mint a handle for `scope`. Raises for an unknown scope."""
        if scope not in _SCOPE_TO_SETTING:
            raise CredentialError(f"unknown credential scope: {scope!r}")
        ttl = self.default_ttl if ttl_seconds is None else ttl_seconds
        handle = CredentialHandle(
            handle_id=secrets.token_urlsafe(16),
            scope=scope,
            expires_at=time.monotonic() + ttl,
        )
        self._issued[handle.handle_id] = handle
        logger.debug("credential issued scope=%s ttl=%.0fs", scope, ttl)
        return handle

    def use(self, handle: CredentialHandle, scope: str) -> str:
        """Exchange a handle for its secret, checking scope and expiry.

        `scope` is what the caller claims to be doing. It must match the scope
        the handle was issued for, so a handle cannot be redirected.
        """
        known = self._issued.get(handle.handle_id)
        if known is None:
            raise CredentialError("unknown credential handle")
        if known.scope != handle.scope:
            raise CredentialError("credential handle does not match its record")
        if handle.scope != scope:
            raise CredentialError(
                f"handle is scoped to {handle.scope!r}, used for {scope!r}"
            )
        if handle.expired:
            self._issued.pop(handle.handle_id, None)
            raise CredentialError("credential handle has expired")
        return self._secret_for(handle.scope)

    def revoke(self, handle: CredentialHandle) -> None:
        self._issued.pop(handle.handle_id, None)

    def purge_expired(self) -> int:
        """Drop expired handles. Returns how many were removed."""
        dead = [hid for hid, h in self._issued.items() if h.expired]
        for hid in dead:
            self._issued.pop(hid, None)
        return len(dead)

    @property
    def live_handle_count(self) -> int:
        return sum(1 for h in self._issued.values() if not h.expired)

    @staticmethod
    def _secret_for(scope: str) -> str:
        from config import settings

        value = getattr(settings, _SCOPE_TO_SETTING[scope], "")
        if not value:
            raise CredentialError(f"no credential configured for scope {scope!r}")
        return value


_broker: CredentialBroker | None = None


def get_broker() -> CredentialBroker:
    """Process wide broker."""
    global _broker
    if _broker is None:
        _broker = CredentialBroker()
    return _broker


def reset_broker() -> None:
    """Drop the process wide broker. Used by tests."""
    global _broker
    _broker = None
