"""Redaction of secrets from log records and free text.

Two layers, because neither alone is enough:

1. Pattern matching catches credential shapes the process has never seen,
   such as a key a user pastes into a question.
2. Literal matching catches this process's own configured secrets, read from
   `config.settings`, including ones whose shape no pattern would recognise,
   such as a Postgres DSN password.

The logging filter fails open on purpose. A bug in redaction must not be able
to drop a log line, so any error while rewriting a record leaves the record as
it was and the failure is counted rather than raised.

Installing the filter is gated on REDACTION_ENABLED and off by default. The
query log lines in the API layer do not depend on it: they never include the
query text at all, so a secret pasted into a question is not collected in the
first place.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterable

PLACEHOLDER = "[redacted]"

# Credential shapes. Each pattern is anchored on a distinctive prefix so it
# does not match ordinary prose.
_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),            # OpenAI style
    re.compile(r"tvly-[A-Za-z0-9_\-]{16,}"),          # Tavily
    re.compile(r"lsv2_[A-Za-z0-9_\-]{16,}"),          # LangSmith
    re.compile(r"pcsk_[A-Za-z0-9_\-]{16,}"),          # Pinecone
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),              # GitHub personal token
    re.compile(r"gho_[A-Za-z0-9]{20,}"),              # GitHub OAuth token
    re.compile(r"AKIA[0-9A-Z]{16}"),                  # AWS access key id
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-]{16,}"),
    # Credentials embedded in a connection URL, for example
    # postgresql://user:password@host/db
    re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://[^\s:/@]+):([^\s/@]+)@"),
)

# Settings attributes that hold a secret. Checked by name so a new secret
# added to Settings is a one line change here.
_SECRET_SETTING_NAMES: tuple[str, ...] = (
    "openai_api_key",
    "tavily_api_key",
    "langsmith_api_key",
    "pinecone_api_key",
    "checkpoint_dsn",
    "api_key",
)

# Values shorter than this are not masked as literals. A one or two character
# secret would match everywhere and make logs unreadable.
_MIN_LITERAL_LENGTH = 8

_filter_errors = 0


def known_secret_values() -> tuple[str, ...]:
    """Non-empty secret values this process is configured with."""
    try:
        from config import settings
    except Exception:  # pragma: no cover - config import cannot fail in practice
        return ()
    values = []
    for name in _SECRET_SETTING_NAMES:
        value = getattr(settings, name, "")
        if isinstance(value, str) and len(value) >= _MIN_LITERAL_LENGTH:
            values.append(value)
    # Longest first, so a secret that contains another is masked whole.
    return tuple(sorted(set(values), key=len, reverse=True))


def redact(text: str, extra_values: Iterable[str] = ()) -> str:
    """Return `text` with credential shapes and known secret values masked."""
    if not text:
        return text
    out = text
    for value in known_secret_values():
        if value in out:
            out = out.replace(value, PLACEHOLDER)
    for value in extra_values:
        if isinstance(value, str) and len(value) >= _MIN_LITERAL_LENGTH and value in out:
            out = out.replace(value, PLACEHOLDER)
    for pattern in _PATTERNS:
        if pattern.groups == 2:
            # Connection URL: keep the scheme and user, mask only the password.
            out = pattern.sub(rf"\1:{PLACEHOLDER}@", out)
        else:
            out = pattern.sub(PLACEHOLDER, out)
    return out


def redact_value(value: Any) -> Any:
    """Redact strings anywhere inside a nested structure, leaving shape intact."""
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        rebuilt = [redact_value(v) for v in value]
        return type(value)(rebuilt) if isinstance(value, tuple) else rebuilt
    return value


class RedactingFilter(logging.Filter):
    """Masks secrets in a record's message and arguments.

    Fails open: if rewriting raises for any reason the record passes through
    unchanged, because losing a log line is worse than an unmasked one slipping
    past in an edge case nobody anticipated.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        global _filter_errors
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                if isinstance(record.args, dict):
                    record.args = {k: redact_value(v) for k, v in record.args.items()}
                else:
                    record.args = tuple(redact_value(a) for a in record.args)
        except Exception:  # pragma: no cover - defensive, see docstring
            _filter_errors += 1
        return True


def filter_error_count() -> int:
    """How many times the filter failed open. Exposed for tests and debugging."""
    return _filter_errors


_installed = False


def install_log_redaction(force: bool = False) -> bool:
    """Attach the filter to the root logger's handlers.

    Returns True when the filter is installed. Does nothing unless
    REDACTION_ENABLED is true, or `force` is passed, which tests use.
    """
    global _installed
    if _installed:
        return True
    if not force:
        from config import settings

        if not settings.redaction_enabled:
            return False
    root = logging.getLogger()
    f = RedactingFilter()
    for handler in root.handlers:
        handler.addFilter(f)
    root.addFilter(f)
    _installed = True
    return True
