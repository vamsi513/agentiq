"""Credential broker tests.

The claims worth proving are negative: a handle carries no secret, cannot be
used outside its scope, and is dead once it expires.
"""

import time

import pytest

from agent.credentials import (
    SCOPE_EMBEDDINGS,
    SCOPE_LLM,
    SCOPE_WEB_SEARCH,
    CredentialBroker,
    CredentialError,
    get_broker,
    reset_broker,
)

FAKE_TAVILY = "tvly-broker000test000value000abcdefgh"
FAKE_OPENAI = "sk-broker000test000value000abcdefghij"


@pytest.fixture(autouse=True)
def fresh_broker():
    reset_broker()
    yield
    reset_broker()


@pytest.fixture
def keys(monkeypatch):
    from config import settings

    monkeypatch.setattr(settings, "tavily_api_key", FAKE_TAVILY)
    monkeypatch.setattr(settings, "openai_api_key", FAKE_OPENAI)
    return settings


class TestHandleCarriesNoSecret:
    def test_handle_repr_does_not_contain_the_secret(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        assert FAKE_TAVILY not in repr(handle)
        assert FAKE_TAVILY not in str(handle)

    def test_handle_fields_do_not_contain_the_secret(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        assert FAKE_TAVILY not in handle.handle_id
        assert handle.scope == SCOPE_WEB_SEARCH

    def test_a_handle_cannot_be_given_extra_attributes(self, keys):
        """__slots__ stops anyone stuffing a secret onto a handle."""
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        with pytest.raises((AttributeError, TypeError)):
            handle.secret = FAKE_TAVILY  # type: ignore[attr-defined]


class TestScoping:
    def test_a_handle_redeems_its_own_scope(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        assert broker.use(handle, SCOPE_WEB_SEARCH) == FAKE_TAVILY

    def test_a_handle_cannot_be_redirected_to_another_scope(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        with pytest.raises(CredentialError, match="scoped to"):
            broker.use(handle, SCOPE_LLM)

    def test_unknown_scope_cannot_be_issued(self):
        broker = CredentialBroker(default_ttl_seconds=60)
        with pytest.raises(CredentialError, match="unknown credential scope"):
            broker.issue("root")

    def test_scopes_map_to_their_own_secret(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        assert broker.use(broker.issue(SCOPE_LLM), SCOPE_LLM) == FAKE_OPENAI
        assert broker.use(broker.issue(SCOPE_EMBEDDINGS), SCOPE_EMBEDDINGS) == FAKE_OPENAI

    def test_missing_configuration_is_an_error_not_an_empty_string(self, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "tavily_api_key", "")
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        with pytest.raises(CredentialError, match="no credential configured"):
            broker.use(handle, SCOPE_WEB_SEARCH)


class TestExpiry:
    def test_an_expired_handle_is_rejected(self, keys):
        broker = CredentialBroker(default_ttl_seconds=0.01)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        time.sleep(0.03)
        assert handle.expired is True
        with pytest.raises(CredentialError, match="expired"):
            broker.use(handle, SCOPE_WEB_SEARCH)

    def test_a_revoked_handle_is_rejected(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        handle = broker.issue(SCOPE_WEB_SEARCH)
        broker.revoke(handle)
        with pytest.raises(CredentialError, match="unknown credential handle"):
            broker.use(handle, SCOPE_WEB_SEARCH)

    def test_a_forged_handle_is_rejected(self, keys):
        from agent.credentials import CredentialHandle

        broker = CredentialBroker(default_ttl_seconds=60)
        forged = CredentialHandle(
            handle_id="forged", scope=SCOPE_WEB_SEARCH, expires_at=time.monotonic() + 999
        )
        with pytest.raises(CredentialError, match="unknown credential handle"):
            broker.use(forged, SCOPE_WEB_SEARCH)

    def test_purge_removes_only_expired_handles(self, keys):
        broker = CredentialBroker(default_ttl_seconds=60)
        live = broker.issue(SCOPE_WEB_SEARCH)
        short = CredentialBroker(default_ttl_seconds=0.01)
        short.issue(SCOPE_WEB_SEARCH)
        time.sleep(0.03)
        assert short.purge_expired() == 1
        assert broker.purge_expired() == 0
        assert broker.live_handle_count == 1
        assert broker.use(live, SCOPE_WEB_SEARCH) == FAKE_TAVILY


class TestWebSearchIntegration:
    def test_default_path_reads_settings_directly(self, keys, monkeypatch):
        """With the broker flag off, behaviour matches what it was before."""
        from config import settings

        monkeypatch.setattr(settings, "credential_broker_enabled", False)
        from tools.web_search import _resolve_api_key

        assert _resolve_api_key() == FAKE_TAVILY

    def test_broker_path_returns_the_same_key(self, keys, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "credential_broker_enabled", True)
        from tools.web_search import _resolve_api_key

        assert _resolve_api_key() == FAKE_TAVILY

    def test_broker_path_leaves_no_live_handles_behind(self, keys, monkeypatch):
        from config import settings

        monkeypatch.setattr(settings, "credential_broker_enabled", True)
        from tools.web_search import _resolve_api_key

        _resolve_api_key()
        assert get_broker().live_handle_count == 0

    def test_broker_failure_does_not_raise_into_the_tool(self, monkeypatch):
        """A broker problem degrades to an empty key, which the tool handles."""
        from config import settings

        monkeypatch.setattr(settings, "credential_broker_enabled", True)
        monkeypatch.setattr(settings, "tavily_api_key", "")
        from tools.web_search import _resolve_api_key

        assert _resolve_api_key() == ""


class TestFlagDefaults:
    def test_broker_is_off_by_default(self):
        from config import settings

        assert settings.credential_broker_enabled is False

    def test_redaction_is_on_by_default(self):
        """Redaction is the one feature here that defaults to on, because node
        and tool logs retain query text and a user can paste a key into one."""
        from config import settings

        assert settings.redaction_enabled is True
