"""Confirmed-send history and shared provider cooldown regression tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from test_reauth_tools import real_auth_modules  # shared, restoring isolation fixture


class FakeDB:
    def __init__(self):
        self.store = {}

    def collection(self, collection):
        def document(name):
            key = (collection, name)

            def get():
                return SimpleNamespace(
                    exists=key in self.store, to_dict=lambda: dict(self.store.get(key, {})),
                )

            def set_doc(data):
                self.store[key] = dict(data)

            def delete():
                self.store.pop(key, None)

            return SimpleNamespace(get=get, set=set_doc, delete=delete)
        return SimpleNamespace(document=document)


@pytest.fixture
def app(monkeypatch, real_auth_modules):
    modules = real_auth_modules
    cfg = modules["config"]
    monkeypatch.setattr(cfg, "MOMO_SERVICE_URL", "https://momo.example")
    monkeypatch.setattr(cfg, "CHAT_SPACE_ID", "spaces/auth-notification-test")
    monkeypatch.setattr(cfg, "GRANOLA_ENABLED", True)
    db = FakeDB()
    for provider in ("google_auth", "granola_service"):
        monkeypatch.setattr(modules[provider], "_get_db", lambda: db)
    monkeypatch.setattr(modules["connection_health"], "get_db", lambda: db)
    events = []
    send = MagicMock(side_effect=lambda space, text: events.append(("send", space, text)) or True)
    save = MagicMock(side_effect=lambda *args: events.append(("history", *args)))
    monkeypatch.setattr(modules["chat_service"], "send_chat_message", send)
    monkeypatch.setattr(modules["conversation_store"], "add_turn", save)
    modules.update(db=db, send=send, save=save, events=events)
    return modules


def sender(app, provider):
    if provider == "google_workspace":
        return app["google_auth"]._send_throttled_reauth_alert()
    if provider == "granola":
        return app["granola_service"].send_reauth_alert()
    kind = "recovery" if provider == "recovery" else "failure"
    health = app["connection_health"]
    return health._send_alert(health._alert_text(health._result("jira", True, "ok"), kind))


def cooldowns(app):
    return {key: data for key, data in app["db"].store.items() if key[1] == "last_reauth_alert"}


@pytest.mark.parametrize("provider", ["google_workspace", "granola", "recovery", "other"])
def test_successful_auth_alert_saved_after_delivery_in_space_history(app, provider):
    assert sender(app, provider) is True
    space, message = app["send"].call_args.args
    assert space == "spaces/auth-notification-test"
    app["save"].assert_called_once_with(f"space:{space}", "assistant", message)
    assert [event[0] for event in app["events"]] == ["send", "history"]
    if provider == "google_workspace":
        assert "<https://momo.example/google-auth/start?t=" in message
        assert "single-use" in message and "10 minutes" in message
        assert "fresh google reconnect link" in message
    if provider == "granola":
        assert "<https://momo.example/granola-auth/start|" in message
    for overpromise in ("takes 10 seconds", "resume pulling", "resume syncing", "back to normal"):
        assert overpromise not in message


@pytest.mark.parametrize("provider", ["google_workspace", "granola", "recovery", "other"])
@pytest.mark.parametrize("delivery", [False, None, "raises"])
def test_failed_send_does_not_persist_history_or_cooldown(app, provider, delivery, capsys):
    app["send"].side_effect = RuntimeError("ticket=private") if delivery == "raises" else None
    app["send"].return_value = delivery
    assert sender(app, provider) is False
    app["save"].assert_not_called()
    assert cooldowns(app) == {}
    assert "ticket=private" not in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["google_workspace", "granola", "recovery", "other"])
def test_history_failure_does_not_change_delivery_result(app, provider, capsys):
    app["save"].side_effect = RuntimeError("ticket=private")
    assert sender(app, provider) is True
    if provider in ("google_workspace", "granola"):
        assert len(cooldowns(app)) == 1
        assert sender(app, provider) is False
    app["send"].assert_called_once()
    assert "ticket=private" not in capsys.readouterr().out


@pytest.mark.parametrize("provider", ["google_workspace", "granola", "recovery", "other"])
def test_missing_space_never_sends_or_saves(app, monkeypatch, provider):
    monkeypatch.setattr(app["config"], "CHAT_SPACE_ID", "")
    assert sender(app, provider) is False
    app["send"].assert_not_called()
    app["save"].assert_not_called()


@pytest.mark.parametrize("provider", ["google_workspace", "granola"])
@pytest.mark.parametrize("url", ["", "/relative", "http://momo.example"])
def test_automatic_alert_without_safe_config_never_sends_a_relative_link(app, monkeypatch, provider, url):
    monkeypatch.setattr(app["config"], "MOMO_SERVICE_URL", url)
    assert sender(app, provider) is False
    app["send"].assert_not_called()
    assert app["db"].store == {}


@pytest.mark.parametrize("provider", ["google_workspace", "granola"])
@pytest.mark.parametrize("health_first", [True, False])
def test_health_and_provider_failures_share_delivery_cooldown_and_recovery(app, monkeypatch,
                                                                         provider, health_first):
    health = app["connection_health"]
    clock = [100000.0]
    monkeypatch.setattr(health.time, "time", lambda: clock[0])
    if provider == "google_workspace":
        google = app["google_auth"]
        creds = MagicMock(valid=False, expired=True, refresh_token="fake-refresh")
        creds.refresh.side_effect = RuntimeError("invalid_grant")
        monkeypatch.setattr(google, "_load_credentials_from_sources", lambda: creds)
        monkeypatch.setattr(google, "_reauth_required", False)
        monkeypatch.setattr(google, "_cached_creds", None)
        probe = health.probe_google_workspace  # refresh path alerts inside the probe

        def service_failure():
            assert google.classify_google_auth_error(
                SimpleNamespace(status_code=401), source="gmail_service"
            ) is not None
    else:
        granola = app["granola_service"]
        monkeypatch.setattr(granola, "_load_token", lambda: None)
        probe = health.probe_granola

        def service_failure():
            with pytest.raises(app["connection_errors"].ExternalAuthError):
                granola.query_granola("notes")

    def check():
        return health.run_connection_health_check(probes=[(provider, probe)])

    if not health_first:
        service_failure()
    assert check()["status"] == "degraded"
    service_failure()
    assert check()["status"] == "degraded"
    assert app["send"].call_count == app["save"].call_count == 1
    assert app["db"].store[("connection_health", provider)]["status"] == "auth_failed"

    # A health reminder also goes through the provider and mints a fresh ticket.
    clock[0] += 13 * 3600
    check()
    assert app["send"].call_count == app["save"].call_count == 2
    service_failure()
    assert app["send"].call_count == 2
    if provider == "google_workspace":
        messages = [call.args[1] for call in app["send"].call_args_list]
        assert messages[0] != messages[1]
        assert len([key for key in app["db"].store if key[1].startswith("ticket:")]) == 2

    # Recovery remains independent of the provider failure-notification cooldown.
    recovered = [(provider, lambda: health._result(provider, True, "ok"))]
    assert health.run_connection_health_check(probes=recovered)["status"] == "ok"
    assert app["send"].call_count == app["save"].call_count == 3
    assert "restored" in app["send"].call_args.args[1]
    assert app["db"].store[("connection_health", provider)]["status"] == "ok"
    assert cooldowns(app) == {}  # Recovery ends the previous failure episode.
    health.run_connection_health_check(probes=recovered)
    assert app["send"].call_count == 3


def test_granola_missing_auth_is_typed_not_empty_and_agent_surfaces_it(app, monkeypatch):
    granola = app["granola_service"]
    monkeypatch.setattr(granola, "_load_token", lambda: None)
    result = app["agent"].execute_tool("get_meeting_notes", {"query": "notes"})
    assert result.startswith("CONNECTION_AUTH_ERROR connector=granola action=reconnect")
    assert "No meeting notes found" not in result
    assert "ask momo for a fresh Granola reconnect link" in result


def test_granola_healthy_empty_result_stays_empty(app, monkeypatch):
    async def empty_result(*args, **kwargs):
        return SimpleNamespace(content=[])

    monkeypatch.setattr(app["granola_service"], "_call_tool", empty_result)
    assert app["granola_service"].query_granola("notes") == ""
    assert app["agent"].execute_tool("get_meeting_notes", {"query": "notes"}) == "No meeting notes found."
    app["send"].assert_not_called()


def test_granola_final_401_is_still_typed_after_retry(app, monkeypatch):
    class Unauthorized(Exception):
        status_code = 401

    granola = app["granola_service"]
    monkeypatch.setattr(granola, "_load_token", lambda: "fake-access")
    monkeypatch.setattr(granola, "_cached_token", None)
    transport = MagicMock(side_effect=Unauthorized("401 Unauthorized"))
    monkeypatch.setattr(granola, "streamablehttp_client", transport)
    result = app["agent"].execute_tool("get_meeting_notes", {"query": "notes"})
    assert result.startswith("CONNECTION_AUTH_ERROR connector=granola action=reconnect")
    assert "No meeting notes found" not in result
    assert transport.call_count == 2
    app["send"].assert_called_once()
    app["save"].assert_called_once()
