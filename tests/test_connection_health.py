"""Connection-health + typed auth-error tests.

Covers the must-hold guarantees:
  (a) Jira empty search + dead auth  -> ExternalAuthError (never silently [])
  (b) Jira empty search + healthy auth -> [] exactly as before
  (c) agent.execute_tool renders the stable CONNECTION_AUTH_ERROR string
  (d) MCP 401 during tool discovery does NOT cache an empty tool list
  (e) run_connection_health_check aggregates probes and alerts only on
      transitions (first failure alerts, repeat within cooldown doesn't,
      reminder after cooldown does, recovery alerts)
  (f) Chat send non-2xx no longer reports success (and 401 never crashes)

Sibling test files stub modules during collection. Load the real production
modules lazily in a private, restoring module graph instead of globally
purging siblings' stubs. All network/Firestore access is patched at seams.
"""

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from test_reauth_tools import real_auth_modules  # shared, restoring isolation fixture

if TYPE_CHECKING:  # runtime bindings come from _real_modules
    import agent
    import chat_service
    import config
    import connection_health
    import jira_service
    import mcp_client
    from connection_errors import ExternalAuthError, format_for_agent


@pytest.fixture(autouse=True)
def _real_modules(monkeypatch, real_auth_modules):
    for name in ("config", "jira_service", "mcp_client", "chat_service", "connection_health", "agent"):
        monkeypatch.setitem(globals(), name, real_auth_modules[name])
    for name in ("ExternalAuthError", "format_for_agent"):
        monkeypatch.setitem(globals(), name, getattr(real_auth_modules["connection_errors"], name))


# ── Helpers ─────────────────────────────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeSnapshot:
    def __init__(self, data):
        self._data = data
        self.exists = data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeDocRef:
    def __init__(self, store, key):
        self._store = store
        self._key = key

    def get(self):
        return _FakeSnapshot(self._store.get(self._key))

    def set(self, doc):
        self._store[self._key] = dict(doc)


class _FakeDB:
    def __init__(self):
        self.store = {}

    def collection(self, _name):
        return self

    def document(self, doc_id):
        return _FakeDocRef(self.store, doc_id)


# ── (a)/(b) Jira empty-search vs dead auth ──────────────────────────────────

def _fake_empty_search(monkeypatch):
    monkeypatch.setattr(
        jira_service, "_request",
        lambda method, path, json_body=None, timeout=None: _FakeResponse(200, {"issues": []}),
    )


def test_jira_empty_search_with_dead_auth_raises(monkeypatch):
    _fake_empty_search(monkeypatch)
    monkeypatch.setattr(jira_service, "jira_auth_ok", lambda force=False: False)
    with pytest.raises(ExternalAuthError) as excinfo:
        jira_service._search("project = OSD")
    assert excinfo.value.connector == "jira"


def test_jira_empty_search_with_healthy_auth_returns_empty(monkeypatch):
    _fake_empty_search(monkeypatch)
    monkeypatch.setattr(jira_service, "jira_auth_ok", lambda force=False: True)
    assert jira_service._search("project = OSD") == []


def test_jira_request_helper_raises_typed_on_401(monkeypatch):
    monkeypatch.setattr(
        jira_service.httpx, "request",
        lambda *a, **k: _FakeResponse(401, text="Unauthorized"),
    )
    with pytest.raises(ExternalAuthError) as excinfo:
        jira_service._request("GET", "/myself")
    assert excinfo.value.status == 401


def test_jira_auth_ok_fails_open_on_indeterminate(monkeypatch):
    monkeypatch.setattr(jira_service, "_auth_cache", {"ok": None, "checked_at": 0.0})
    monkeypatch.setattr(jira_service, "check_jira_auth",
                        lambda: (None, "network unreachable"))
    assert jira_service.jira_auth_ok(force=True) is True


# ── (c) execute_tool renders CONNECTION_AUTH_ERROR ──────────────────────────

def test_execute_tool_renders_connection_auth_error(monkeypatch):
    err = ExternalAuthError("jira", "Jira credentials are dead",
                            status=401, reconnect_hint="rotate the token")

    def _boom(name, args, **kwargs):
        raise err

    monkeypatch.setattr(agent, "_dispatch", _boom)
    result = agent.execute_tool("get_jira_tickets", {})
    assert result == format_for_agent(err)
    assert result.startswith("CONNECTION_AUTH_ERROR connector=jira action=reconnect")
    assert "no data" not in result.lower()


# ── (d) MCP 401 during discovery does not cache empty ───────────────────────

class _FakeAuthErr(Exception):
    status_code = 401


def test_mcp_auth_error_discovery_not_cached(monkeypatch):
    srv = {"name": "testsrv", "url": "https://example/mcp",
           "auth": "bearer", "bearer_token": "t", "enabled": True}
    monkeypatch.setattr(mcp_client, "_get_server_config", lambda name: srv)
    mcp_client._tool_discovery_cache.pop("testsrv", None)

    def _run_raises(coro, timeout=15):
        coro.close()
        raise _FakeAuthErr("401 Unauthorized")

    monkeypatch.setattr(mcp_client, "_run", _run_raises)
    with pytest.raises(ExternalAuthError):
        mcp_client.list_server_tools("testsrv")
    assert "testsrv" not in mcp_client._tool_discovery_cache


def test_mcp_non_auth_discovery_failure_still_caches_empty(monkeypatch):
    srv = {"name": "testsrv2", "url": "https://example/mcp",
           "auth": "bearer", "bearer_token": "t", "enabled": True}
    monkeypatch.setattr(mcp_client, "_get_server_config", lambda name: srv)
    mcp_client._tool_discovery_cache.pop("testsrv2", None)

    def _run_raises(coro, timeout=15):
        coro.close()
        raise RuntimeError("connection reset")  # non-auth

    monkeypatch.setattr(mcp_client, "_run", _run_raises)
    assert mcp_client.list_server_tools("testsrv2") == []
    assert "testsrv2" in mcp_client._tool_discovery_cache  # 60s backoff kept
    mcp_client._tool_discovery_cache.pop("testsrv2", None)


def test_mcp_is_auth_error_handles_403_and_wrapped():
    class _Err403(Exception):
        status_code = 403

    class _Group(Exception):
        def __init__(self, subs):
            self.exceptions = subs

    assert mcp_client._is_auth_error(_Err403("no")) is True
    assert mcp_client._is_auth_error(_Group([_FakeAuthErr("401")])) is True
    assert mcp_client._is_auth_error(RuntimeError("boom")) is False


# ── (e) Health check: aggregation + transition-only alerting ────────────────

def _probe_result(connector, status, **overrides):
    base = {
        "connector": connector, "enabled": True, "status": status,
        "checked_at": "2026-08-05T00:00:00+00:00",
        "error_kind": "auth" if status == "auth_failed" else None,
        "error_message": "401" if status == "auth_failed" else "",
        "user_message": "reconnect jira" if status == "auth_failed" else "",
    }
    base.update(overrides)
    return base


def test_health_check_alerts_only_on_transition(monkeypatch):
    fake_db = _FakeDB()
    alerts = []
    monkeypatch.setattr(connection_health, "get_db", lambda: fake_db)
    monkeypatch.setattr(connection_health, "_send_alert", lambda text: alerts.append(text) or True)

    failing = [("jira", lambda: _probe_result("jira", "auth_failed"))]
    recovered = [("jira", lambda: _probe_result("jira", "ok"))]

    # First failure: alert + degraded aggregate.
    out1 = connection_health.run_connection_health_check(probes=failing)
    assert out1["status"] == "degraded"
    assert out1["connectors"][0]["status"] == "auth_failed"
    assert len(alerts) == 1 and "jira" in alerts[0]

    # Still failing within the 12h cooldown: NO new alert.
    out2 = connection_health.run_connection_health_check(probes=failing)
    assert out2["status"] == "degraded"
    assert len(alerts) == 1

    # Still failing but cooldown elapsed: reminder alert.
    fake_db.store["jira"]["last_alerted_at"] -= 13 * 3600
    connection_health.run_connection_health_check(probes=failing)
    assert len(alerts) == 2 and "reminder" in alerts[1]

    # Recovery: one recovery note, aggregate back to ok.
    out3 = connection_health.run_connection_health_check(probes=recovered)
    assert out3["status"] == "ok"
    assert len(alerts) == 3 and "restored" in alerts[2]

    # Steady-state ok: silent.
    connection_health.run_connection_health_check(probes=recovered)
    assert len(alerts) == 3


def test_health_check_skipped_and_mixed_aggregate(monkeypatch):
    monkeypatch.setattr(connection_health, "get_db", lambda: _FakeDB())
    alerts = []
    monkeypatch.setattr(connection_health, "_send_alert", alerts.append)
    probes = [
        ("jira", lambda: _probe_result("jira", "skipped", enabled=False)),
        ("granola", lambda: _probe_result("granola", "ok")),
    ]
    out = connection_health.run_connection_health_check(probes=probes)
    assert out["status"] == "ok"
    statuses = {r["connector"]: r["status"] for r in out["connectors"]}
    assert statuses == {"jira": "skipped", "granola": "ok"}
    assert alerts == []  # skipped/ok never alert


def test_health_check_firestore_down_best_effort_alert(monkeypatch):
    def _db_down():
        raise RuntimeError("firestore unreachable")

    alerts = []
    monkeypatch.setattr(connection_health, "get_db", _db_down)
    monkeypatch.setattr(connection_health, "_send_alert", alerts.append)
    out = connection_health.run_connection_health_check(
        probes=[("jira", lambda: _probe_result("jira", "auth_failed"))],
    )
    # Persistence skipped, but the failure still alerts best-effort.
    assert out["status"] == "degraded"
    assert len(alerts) == 1


def test_failed_health_alert_does_not_record_delivery_and_retries(monkeypatch):
    fake_db = _FakeDB()
    send = MagicMock(return_value=False)
    monkeypatch.setattr(connection_health, "get_db", lambda: fake_db)
    monkeypatch.setattr(connection_health, "_send_alert", send)
    failed = _probe_result("jira", "auth_failed")
    connection_health._persist_and_alert(failed)
    assert fake_db.store["jira"]["status"] == "auth_failed"
    assert fake_db.store["jira"]["last_alerted_at"] == 0
    send.return_value = True
    connection_health._persist_and_alert(failed)
    assert send.call_count == 2
    last_sent = fake_db.store["jira"]["last_alerted_at"]
    assert last_sent > 0
    send.return_value = False
    connection_health._persist_and_alert(_probe_result("jira", "ok"))
    assert fake_db.store["jira"]["status"] == "ok"
    assert fake_db.store["jira"]["last_alerted_at"] == last_sent


@pytest.mark.parametrize("connector", ["google_workspace", "granola"])
def test_health_provider_cooldown_is_authoritative_even_with_recent_health_alert(monkeypatch, connector):
    import google_auth
    import granola_service

    fake_db = _FakeDB()
    fake_db.store[connector] = {
        "status": "auth_failed", "last_alerted_at": connection_health.time.time(),
    }
    sender = MagicMock(return_value=False)
    module, name = (google_auth, "_send_throttled_reauth_alert") if connector == "google_workspace" else (
        granola_service, "send_reauth_alert"
    )
    monkeypatch.setattr(module, name, sender)
    monkeypatch.setattr(connection_health, "get_db", lambda: fake_db)
    direct_send = MagicMock()
    monkeypatch.setattr(connection_health, "_send_alert", direct_send)
    before = fake_db.store[connector]["last_alerted_at"]
    connection_health._persist_and_alert(_probe_result(connector, "auth_failed"))
    sender.assert_called_once_with()
    direct_send.assert_not_called()
    assert fake_db.store[connector]["last_alerted_at"] == before


@pytest.mark.parametrize("connector", ["google_workspace", "granola"])
def test_health_firestore_down_still_routes_through_provider(monkeypatch, connector):
    import google_auth
    import granola_service

    monkeypatch.setattr(connection_health, "get_db", MagicMock(side_effect=RuntimeError("offline")))
    module, name = (google_auth, "_send_throttled_reauth_alert") if connector == "google_workspace" else (
        granola_service, "send_reauth_alert"
    )
    sender = MagicMock(return_value=False)
    monkeypatch.setattr(module, name, sender)
    direct_send = MagicMock()
    monkeypatch.setattr(connection_health, "_send_alert", direct_send)
    result = _probe_result(
        connector, "auth_failed",
        user_message=f"{connector} needs re-auth — ask momo for a fresh reconnect link",
    )
    connection_health._persist_and_alert(result)
    sender.assert_called_once_with()
    # Only the unavailable health-state read permits this safe, link-free
    # fallback when the provider could not deliver its own notification.
    message = connection_health._alert_text(result, "failure")
    direct_send.assert_called_once_with(message)
    assert "fresh reconnect link" in message
    for broken_link in ("http://", "https://", "/google-auth/start", "/granola-auth/start", "?t="):
        assert broken_link not in message


# ── (f) Chat send non-2xx is a failure, never a success ─────────────────────

class _FakeSession:
    def __init__(self, status_code):
        self.status_code = status_code
        self.posts = []

    def post(self, url, json=None):
        self.posts.append((url, json))
        return _FakeResponse(self.status_code, text="err")


def test_chat_send_non_200_reports_failure(monkeypatch):
    session = _FakeSession(500)
    monkeypatch.setattr(chat_service, "_get_chat_session", lambda: session)
    ok = chat_service.send_chat_message("spaces/test", text="hello")
    assert ok is False
    assert len(session.posts) == 1  # non-2xx is not retried, just failed


def test_chat_send_200_reports_success(monkeypatch):
    session = _FakeSession(200)
    monkeypatch.setattr(chat_service, "_get_chat_session", lambda: session)
    assert chat_service.send_chat_message("spaces/test", text="hello") is True


def test_chat_send_401_logs_auth_and_never_raises(monkeypatch):
    session = _FakeSession(401)
    monkeypatch.setattr(chat_service, "_get_chat_session", lambda: session)
    # Scheduled jobs must survive: auth failure -> False, no exception.
    assert chat_service.send_chat_message("spaces/test", text="hello") is False


def test_chat_send_with_retry_raises_typed_on_403(monkeypatch):
    session = _FakeSession(403)
    with pytest.raises(ExternalAuthError) as excinfo:
        chat_service._send_with_retry(session, "url", "spaces/test", text="x")
    assert excinfo.value.connector == "google_chat"
