"""Recovery/cooldown regressions with real provider logic and no external I/O."""

import asyncio
import importlib.util
import itertools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest


PROVIDERS = ("google_workspace", "granola")


class MemoryDB:
    def __init__(self):
        self.docs = {}
        self.events = []
        self.fail_reads = set()
        self.fail_deletes = set()

    def collection(self, collection):
        def document(name):
            key = (collection, name)

            def get():
                if key in self.fail_reads:
                    raise RuntimeError("storage unavailable")
                data = self.docs.get(key)
                return SimpleNamespace(exists=data is not None, to_dict=lambda: dict(data or {}))

            def set_doc(data):
                self.events.append(("set", key))
                self.docs[key] = dict(data)

            def delete():
                if key in self.fail_deletes:
                    raise RuntimeError("delete unavailable")
                self.events.append(("delete", key))
                self.docs.pop(key, None)

            return SimpleNamespace(get=get, set=set_doc, delete=delete)

        return SimpleNamespace(document=document)


@pytest.fixture
def app(monkeypatch):
    """Fresh, restoring module graph independent of sibling test-module mocks."""
    def stub(name, **attributes):
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
        return module

    def load(name):
        path = Path(__file__).resolve().parents[1] / f"{name}.py"
        spec = importlib.util.spec_from_file_location(name, path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    def blocked():
        return Mock(side_effect=AssertionError("unstubbed external boundary"))

    cfg = stub(
        "config", MOMO_SERVICE_URL="https://momo.example", CHAT_SPACE_ID="spaces/recovery-test",
        GOOGLE_CLIENT_SECRET_FILE="unused-client.json", GOOGLE_SCOPES=["test-scope"],
        GRANOLA_ENABLED=True, GRANOLA_TOKEN="", GRANOLA_MCP_URL="https://granola.example/mcp",
    )
    for name in ("google", "google.auth", "google.auth.transport", "google.oauth2", "google_auth_oauthlib"):
        stub(name)
    stub("google.auth.transport.requests", Request=object)
    stub("google.oauth2.credentials", Credentials=blocked())
    stub("google_auth_oauthlib.flow", Flow=blocked())
    stub("googleapiclient")
    stub("googleapiclient.discovery", build=blocked())
    stub("mcp", ClientSession=blocked())
    stub("mcp.client")
    stub("mcp.client.streamable_http", streamablehttp_client=blocked())
    http = stub("httpx", Auth=object, AsyncClient=blocked(), post=blocked(), get=blocked())
    send = Mock(return_value=True)
    history = Mock()
    stub("chat_service", send_chat_message=send)
    db = MemoryDB()
    stub(
        "conversation_store", get_db=lambda: db, add_turn=history,
        conversation_scope=lambda *, space: f"space:{space}",
    )
    load("connection_errors")
    load("reauth_service")
    google = load("google_auth")
    granola = load("granola_service")
    health = load("connection_health")
    clock = [100000.0]
    for module in (google, granola, health):
        monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]))
    tickets = itertools.count(1)
    monkeypatch.setattr(google, "secrets", SimpleNamespace(token_urlsafe=lambda _: f"ticket-{next(tickets)}"))
    monkeypatch.setattr(google, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda _: True)))
    monkeypatch.setattr(granola, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda _: False)))
    monkeypatch.setattr(granola, "_GRANOLA_TOKEN_JSON_ENV", "")
    # Keep real Firestore persistence against MemoryDB, but never write token files.
    monkeypatch.setattr(google, "_write_credentials_to_file", Mock())
    monkeypatch.setattr(granola, "_write_token_to_file", Mock())
    return SimpleNamespace(
        modules={"google_workspace": google, "granola": granola}, health=health,
        config=cfg, http=http, db=db, clock=clock, send=send, history=history,
    )


def cooldown_key(app, provider):
    module = app.modules[provider]
    return (module._FIRESTORE_REAUTH_COLLECTION, module._FIRESTORE_REAUTH_ALERT_DOC)


def sender(app, provider):
    module = app.modules[provider]
    return module._send_throttled_reauth_alert if provider == "google_workspace" else module.send_reauth_alert


def failure_result(app, provider):
    return app.health._result(
        provider, True, "auth_failed", error_kind="auth",
        user_message=f"{provider} needs re-auth — ask momo for a fresh reconnect link",
    )


def prepare_callback(app, monkeypatch, provider):
    module = app.modules[provider]
    state_key = (module._FIRESTORE_REAUTH_COLLECTION, "pending-state")
    app.db.docs[state_key] = {
        "created_at": app.clock[0], "expires_at": app.clock[0] + 600,
        "redirect_uri": f"https://momo.example/{'google' if provider == 'google_workspace' else provider}-auth/callback",
        "token_endpoint": "https://granola.example/token", "client_id": "test-client",
        "code_verifier": "test-pkce-verifier",
    }
    if provider == "google_workspace":
        exchange = Mock()
        flow = SimpleNamespace(
            fetch_token=exchange,
            oauth2session=SimpleNamespace(token={"access_token": "test-token", "scope": ["test-scope"]}),
            credentials=SimpleNamespace(
                to_json=lambda: '{"token": "test-token"}', granted_scopes=["test-scope"],
            ),
        )
        monkeypatch.setattr(module, "OAuthFlow", SimpleNamespace(from_client_secrets_file=Mock(return_value=flow)))
        token_key = (module._FIRESTORE_GOOGLE_AUTH_COLLECTION, module._FIRESTORE_GOOGLE_AUTH_DOC)
    else:
        response = SimpleNamespace(
            raise_for_status=Mock(), json=lambda: {"access_token": "test-token", "expires_in": 3600},
        )
        exchange = AsyncMock(return_value=response)
        client = SimpleNamespace(post=exchange)
        context = AsyncMock()
        context.__aenter__.return_value = client
        monkeypatch.setattr(app.http, "AsyncClient", Mock(return_value=context))
        token_key = (module._FIRESTORE_GRANOLA_COLLECTION, module._FIRESTORE_GRANOLA_DOC)
    return SimpleNamespace(state_key=state_key, token_key=token_key, exchange=exchange)


def failing_probe(app, monkeypatch, provider):
    """Run the actual probe, including the provider alert inside token refresh."""
    module = app.modules[provider]
    if provider == "google_workspace":
        creds = SimpleNamespace(
            valid=False, expired=True, refresh_token="test-refresh",
            refresh=Mock(side_effect=RuntimeError("invalid_grant")),
        )
        monkeypatch.setattr(module, "_load_credentials_from_sources", lambda: creds)
        return app.health.probe_google_workspace
    monkeypatch.setattr(module, "_read_token_from_firestore", lambda: {
        "access_token": "expired-token", "refresh_token": "test-refresh",
        "_client_id": "test-client", "_token_endpoint": "https://granola.example/token",
        "_expires_at": app.clock[0] - 1,
    })
    monkeypatch.setattr(app.http, "post", Mock(side_effect=RuntimeError("401 token expired")))
    return app.health.probe_granola


@pytest.mark.parametrize("provider", PROVIDERS)
def test_successful_reauth_allows_renewed_failure_during_old_cooldown(app, monkeypatch, provider):
    health = app.health
    failed = failure_result(app, provider)
    health._persist_and_alert(failed)
    assert app.send.call_count == 1
    old_sent_at = app.db.docs[cooldown_key(app, provider)]["sent_at"]
    other_provider = next(other for other in PROVIDERS if other != provider)
    other_key = cooldown_key(app, other_provider)
    app.db.docs[other_key] = {"sent_at": old_sent_at}
    app.clock[0] += 30
    callback = prepare_callback(app, monkeypatch, provider)

    assert asyncio.run(app.modules[provider].complete_web_reauth("test-code", "pending-state")) is True

    assert callback.token_key in app.db.docs
    assert callback.state_key not in app.db.docs
    assert cooldown_key(app, provider) not in app.db.docs
    assert app.db.docs[other_key] == {"sent_at": old_sent_at}
    assert app.db.events.index(("set", callback.token_key)) < app.db.events.index(
        ("delete", cooldown_key(app, provider))
    )
    if provider == "granola":
        callback.exchange.assert_awaited_once_with("https://granola.example/token", data={
            "grant_type": "authorization_code", "code": "test-code",
            "redirect_uri": "https://momo.example/granola-auth/callback",
            "client_id": "test-client", "code_verifier": "test-pkce-verifier",
        })
    else:
        callback.exchange.assert_called_once_with(code="test-code")

    app.clock[0] += 30
    assert app.clock[0] - old_sent_at < 12 * 3600
    health._persist_and_alert(failed)  # The health document still says auth_failed.
    assert app.send.call_count == app.history.call_count == 2
    assert app.db.docs[cooldown_key(app, provider)]["sent_at"] == app.clock[0]
    if provider == "google_workspace":
        assert "?t=ticket-1" in app.send.call_args_list[0].args[1]
        assert "?t=ticket-2" in app.send.call_args_list[1].args[1]
    # Replaying the consumed callback cannot clear the new episode's cooldown.
    assert asyncio.run(app.modules[provider].complete_web_reauth("test-code", "pending-state")) is False
    assert sender(app, provider)() is False
    assert app.send.call_count == 2


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("failure", ["missing_state", "expired_state", "exchange", "persistence", "state_delete"])
def test_failed_callback_keeps_cooldown(app, monkeypatch, provider, failure):
    assert sender(app, provider)() is True
    key = cooldown_key(app, provider)
    previous = dict(app.db.docs[key])
    callback = prepare_callback(app, monkeypatch, provider)
    if failure == "missing_state":
        app.db.docs.pop(callback.state_key)
    elif failure == "expired_state":
        app.db.docs[callback.state_key].update(created_at=app.clock[0] - 601, expires_at=app.clock[0] - 1)
    elif failure == "exchange":
        callback.exchange.side_effect = RuntimeError("exchange failed")
    elif failure == "persistence":
        persist = "_persist_credentials" if provider == "google_workspace" else "_persist_token"
        monkeypatch.setattr(app.modules[provider], persist, Mock(side_effect=RuntimeError("persistence failed")))
    else:
        app.db.fail_deletes.add(callback.state_key)

    completion = app.modules[provider].complete_web_reauth("test-code", "pending-state")
    if failure in ("exchange", "persistence", "state_delete"):
        expected_error = app.modules[provider].GoogleReauthError if provider == "google_workspace" else RuntimeError
        with pytest.raises(expected_error) as exc:
            asyncio.run(completion)
        if provider == "google_workspace":
            assert getattr(exc.value, "reason") == ("exchange_failed" if failure == "exchange" else "persistence_failed")
    else:
        assert asyncio.run(completion) is False
    assert app.db.docs[key] == previous
    assert ("delete", key) not in app.db.events
    assert sender(app, provider)() is False
    assert app.send.call_count == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_callback_cooldown_reset_is_best_effort(app, monkeypatch, provider):
    assert sender(app, provider)() is True
    app.db.fail_deletes.add(cooldown_key(app, provider))
    callback = prepare_callback(app, monkeypatch, provider)

    assert asyncio.run(app.modules[provider].complete_web_reauth("test-code", "pending-state")) is True
    assert callback.token_key in app.db.docs
    assert callback.state_key not in app.db.docs


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("recovery_notice_delivered", [True, False])
def test_natural_recovery_from_known_ok_resets_before_next_failure(
    app, monkeypatch, provider, recovery_notice_delivered,
):
    health = app.health
    probe = failing_probe(app, monkeypatch, provider)
    healthy = lambda: health._result(provider, True, "ok")
    assert health.run_connection_health_check(probes=[(provider, healthy)])["status"] == "ok"
    assert health.run_connection_health_check(probes=[(provider, probe)])["status"] == "degraded"
    assert app.send.call_count == 1
    old_sent_at = app.db.docs[cooldown_key(app, provider)]["sent_at"]

    app.clock[0] += 60
    app.send.return_value = recovery_notice_delivered
    assert health.run_connection_health_check(probes=[(provider, healthy)])["status"] == "ok"
    assert cooldown_key(app, provider) not in app.db.docs
    assert app.db.docs[(health._COLLECTION, provider)]["status"] == "ok"
    assert "connection restored" in app.send.call_args.args[1]

    app.clock[0] += 60
    app.send.return_value = True
    assert app.clock[0] - old_sent_at < 12 * 3600
    assert health.run_connection_health_check(probes=[(provider, probe)])["status"] == "degraded"
    assert app.send.call_count == 3  # Initial failure, recovery, renewed failure; no duplicate.
    assert app.history.call_count == (3 if recovery_notice_delivered else 2)
    assert app.db.docs[cooldown_key(app, provider)]["sent_at"] == app.clock[0]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("previous,status", [
    ("auth_failed", "auth_failed"), ("auth_failed", "unavailable"), ("auth_failed", "skipped"),
    ("ok", "ok"), ("ok", "auth_failed"),
])
def test_non_recovery_health_transitions_do_not_clear_cooldown(app, provider, previous, status):
    assert sender(app, provider)() is True
    key = cooldown_key(app, provider)
    previous_cooldown = dict(app.db.docs[key])
    app.db.docs[(app.health._COLLECTION, provider)] = {"status": previous}

    app.health._persist_and_alert(app.health._result(provider, True, status))

    assert app.db.docs[key] == previous_cooldown
    assert ("delete", key) not in app.db.events
    assert app.send.call_count == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_health_cooldown_reset_failure_does_not_block_recovery(app, provider):
    assert sender(app, provider)() is True
    app.db.docs[(app.health._COLLECTION, provider)] = {"status": "auth_failed"}
    app.db.fail_deletes.add(cooldown_key(app, provider))

    app.health._persist_and_alert(app.health._result(provider, True, "ok"))

    assert app.db.docs[(app.health._COLLECTION, provider)]["status"] == "ok"
    assert "connection restored" in app.send.call_args.args[1]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_storage_unavailable_still_alerts_without_inventing_reconnect_url(app, monkeypatch, provider):
    probe = failing_probe(app, monkeypatch, provider)
    app.db.fail_reads.update({(app.health._COLLECTION, provider), cooldown_key(app, provider)})
    result = probe()  # Its own provider notification cannot read the cooldown.
    assert result["status"] == "auth_failed"
    assert "fresh reconnect link" in result["user_message"]
    app.send.assert_not_called()

    app.health._persist_and_alert(result)

    message = app.health._alert_text(result, "failure")
    app.send.assert_called_once_with(app.config.CHAT_SPACE_ID, message)
    app.history.assert_called_once_with(f"space:{app.config.CHAT_SPACE_ID}", "assistant", message)
    for broken_link in ("http://", "https://", "/google-auth/start", "/granola-auth/start", "?t="):
        assert broken_link not in message
    assert not any(name.startswith("ticket:") for _, name in app.db.docs)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_storage_unavailable_provider_exception_uses_safe_fallback(app, monkeypatch, provider):
    app.db.fail_reads.add((app.health._COLLECTION, provider))
    method = "_send_throttled_reauth_alert" if provider == "google_workspace" else "send_reauth_alert"
    monkeypatch.setattr(app.modules[provider], method, Mock(side_effect=RuntimeError("sender unavailable")))

    result = failure_result(app, provider)
    app.health._persist_and_alert(result)

    app.send.assert_called_once_with(app.config.CHAT_SPACE_ID, app.health._alert_text(result, "failure"))


@pytest.mark.parametrize("provider", PROVIDERS)
def test_health_read_unavailable_does_not_duplicate_successful_provider_alert(app, monkeypatch, provider):
    app.db.fail_reads.add((app.health._COLLECTION, provider))
    fallback = Mock(wraps=app.health._send_alert)
    monkeypatch.setattr(app.health, "_send_alert", fallback)

    app.health._persist_and_alert(failure_result(app, provider))

    app.send.assert_called_once()
    fallback.assert_not_called()
    assert cooldown_key(app, provider) in app.db.docs


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("previous", [None, "ok", "auth_failed"])
def test_probe_alert_followed_by_normal_health_path_remains_single(app, monkeypatch, provider, previous):
    if previous is not None:
        app.db.docs[(app.health._COLLECTION, provider)] = {"status": previous}
    fallback = Mock(wraps=app.health._send_alert)
    monkeypatch.setattr(app.health, "_send_alert", fallback)
    probe = failing_probe(app, monkeypatch, provider)

    result = app.health.run_connection_health_check(probes=[(provider, probe)])

    assert result["status"] == "degraded"
    app.send.assert_called_once()
    app.history.assert_called_once()
    fallback.assert_not_called()
    assert sender(app, provider)() is False


def test_unavailable_health_storage_does_not_retry_non_provider_alert(app):
    app.db.fail_reads.add((app.health._COLLECTION, "jira"))
    app.send.return_value = False

    app.health._persist_and_alert(failure_result(app, "jira"))

    app.send.assert_called_once()
