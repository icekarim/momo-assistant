"""Real OAuthlib scope parsing with fake HTTP/Firestore and isolated SDK imports."""

import asyncio
import importlib
import importlib.util
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest


REQUIRED = [
    "https://www.googleapis.com/auth/calendar.readonly",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/tasks",
]
PREVIOUS_GRANTS = [
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/calendar.events",
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/drive.file",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/tasks.readonly",
    "https://www.googleapis.com/auth/chat.messages",
    "https://www.googleapis.com/auth/contacts.readonly",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "openid",
]
CLIENT_CONFIG = {"web": {
    "client_id": "test-client", "client_secret": "private-client-secret",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
}}
CALLBACK = "https://momo.example/google-auth/callback"
STATE_KEY = ("google_auth_pending", "private-state")
TOKEN_KEY = ("google_auth", "token")
STATUS_KEY = ("google_auth_pending", "reauth_required")
COOLDOWN_KEY = ("google_auth_pending", "last_reauth_alert")
PRIVATE_VALUES = (
    "private-access-token", "private-refresh-token", "private-client-secret",
    "private-code", "private-state", "private-provider-description",
)


class MemoryDB:
    def __init__(self):
        self.docs = {}
        self.fail_reads = set()
        self.fail_writes = set()
        self.fail_deletes = set()

    def collection(self, collection):
        def document(name):
            key = (collection, name)

            def get():
                if key in self.fail_reads:
                    raise RuntimeError("private-provider-description")
                data = self.docs.get(key)
                return SimpleNamespace(exists=data is not None, to_dict=lambda: dict(data or {}))

            def set_doc(data):
                if key in self.fail_writes:
                    raise RuntimeError("private-provider-description")
                self.docs[key] = dict(data)

            def delete():
                if key in self.fail_deletes:
                    raise RuntimeError("private-provider-description")
                self.docs.pop(key, None)

            return SimpleNamespace(get=get, set=set_doc, delete=delete)

        return SimpleNamespace(document=document)


@contextmanager
def isolated_auth_imports():
    # Sibling tests may install SDK mocks during collection. Replace and restore
    # only these namespaces, leaving unrelated lazy imports intact.
    roots = {
        "google", "google_auth_oauthlib", "oauthlib", "requests_oauthlib", "cachetools",
        "google_auth", "config", "connection_errors", "reauth_service",
    }
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] in roots}
    for name in saved:
        sys.modules.pop(name)
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                sys.modules.pop(name)
        sys.modules.update(saved)


@pytest.fixture
def app(monkeypatch):
    with isolated_auth_imports(), monkeypatch.context() as patcher:
        cfg = ModuleType("config")
        cfg.__dict__.update(
            GOOGLE_SCOPES=list(REQUIRED), GOOGLE_CLIENT_SECRET_FILE="unused-client.json",
            MOMO_SERVICE_URL="https://momo.example", LANGFUSE_TRACING_ENABLED=False,
        )
        sys.modules["config"] = cfg
        for name in ("connection_errors", "reauth_service", "google_auth"):
            path = Path(__file__).resolve().parents[1] / f"{name}.py"
            spec = importlib.util.spec_from_file_location(name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            spec.loader.exec_module(module)
        google = sys.modules["google_auth"]
        requests = importlib.import_module("requests")
        parameters = importlib.import_module("oauthlib.oauth2.rfc6749.parameters")
        assert not os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE"), "Run with OAuthlib's scope checks enabled"
        blocked_network = Mock(side_effect=AssertionError("live network is forbidden"))
        patcher.setattr(requests.sessions.Session, "send", blocked_network)

        db = MemoryDB()
        db.docs.update({
            STATE_KEY: {"expires_at": 100550.0, "redirect_uri": CALLBACK},
            TOKEN_KEY: {"credentials_json": "previous-credentials"},
            STATUS_KEY: {"reauth_required": True},
            COOLDOWN_KEY: {"sent_at": 99990.0},
        })
        previous_creds = object()
        patcher.setattr(google, "_get_db", lambda: db)
        patcher.setattr(google, "time", SimpleNamespace(time=lambda: 100000.0))
        patcher.setattr(google, "os", SimpleNamespace(path=SimpleNamespace(exists=lambda _: False)))
        patcher.setattr(google, "_load_web_client_config_from_sources", Mock(return_value=CLIENT_CONFIG))
        patcher.setattr(google, "_cached_creds", previous_creds)
        patcher.setattr(google, "_reauth_required", True)
        file_write = Mock()
        patcher.setattr(google, "_write_credentials_to_file", file_write)
        state = SimpleNamespace(
            google=google, cfg=cfg, db=db, previous_creds=previous_creds,
            file_write=file_write, flows=[], parsers=[], warnings=[], parameters=parameters,
            normal_scope_return=False, blocked_network=blocked_network,
            payload={
                "access_token": "private-access-token", "refresh_token": "private-refresh-token",
                "token_type": "Bearer", "expires_in": 3600, "scope": " ".join(REQUIRED),
            },
        )

        def respond(**kwargs):
            response = requests.Response()
            response.status_code = 400 if "error" in state.payload else 200
            response._content = json.dumps(state.payload).encode()
            response.headers["Content-Type"] = "application/json"
            response.request = SimpleNamespace(url=kwargs["url"], headers={}, body="fake-request")
            return response

        state.http = Mock(side_effect=respond)
        real_factory = google.OAuthFlow.from_client_config

        def build_flow(*args, **kwargs):
            flow = real_factory(*args, **kwargs)
            state.flows.append(flow)
            flow.oauth2session.request = state.http
            real_parser = flow.oauth2session._client.parse_request_body_response

            def parse(*parser_args, **parser_kwargs):
                try:
                    return real_parser(*parser_args, **parser_kwargs)
                except Warning as warning:
                    state.warnings.append(warning)
                    if state.normal_scope_return:
                        # Exercise the application's non-Warning path with the
                        # real parser's validated token, without changing env.
                        flow.oauth2session.token = getattr(warning, "token")
                        return getattr(warning, "token")
                    raise

            parser = Mock(side_effect=parse)
            state.parsers.append(parser)
            flow.oauth2session._client.parse_request_body_response = parser
            return flow

        state.factory = Mock(side_effect=build_flow)
        patcher.setattr(google.OAuthFlow, "from_client_config", state.factory)
        yield state
        blocked_network.assert_not_called()


def complete(app, code="private-code", state="private-state"):
    before = os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE")
    try:
        return asyncio.run(app.google.complete_web_reauth(code, state))
    finally:
        assert os.environ.get("OAUTHLIB_RELAX_TOKEN_SCOPE") == before
        assert app.cfg.GOOGLE_SCOPES == REQUIRED


def assert_no_recovery(app):
    assert app.google._cached_creds is app.previous_creds
    assert app.google._reauth_required is True
    assert app.db.docs[STATUS_KEY] == {"reauth_required": True}
    assert app.db.docs[COOLDOWN_KEY] == {"sent_at": 99990.0}


def assert_safe_failure(app, reason, capsys, caplog):
    with pytest.raises(app.google.GoogleReauthError) as exc:
        complete(app)
    assert exc.value.reason == reason
    output = capsys.readouterr().out + caplog.text + str(exc.value)
    for private in (*PRIVATE_VALUES, "Scope has changed from", "https://oauth2.googleapis.com/token"):
        assert private not in output
    assert_no_recovery(app)
    return exc.value


@pytest.mark.parametrize("granted", [REQUIRED, REQUIRED + PREVIOUS_GRANTS], ids=["exact", "superset-15"])
def test_real_parser_accepts_required_scopes_or_superset_once(app, granted, capsys, caplog):
    app.payload["scope"] = " ".join(granted)
    assert app.db.docs[STATE_KEY]["expires_at"] - app.google.time.time() > 8 * 60

    assert complete(app) is True

    app.http.assert_called_once()
    app.parsers[0].assert_called_once()
    assert app.http.call_args.kwargs["data"]["code"] == "private-code"
    assert app.http.call_args.kwargs["data"]["redirect_uri"] == CALLBACK
    session = app.flows[0].oauth2session
    assert session.access_token == "private-access-token"  # Public setter populated the client.
    assert set(session.token["scope"]) == set(granted)
    assert set(app.google._cached_creds.granted_scopes) == set(granted)
    assert app.google._cached_creds.scopes == REQUIRED
    saved = json.loads(app.db.docs[TOKEN_KEY]["credentials_json"])
    assert set(saved["granted_scopes"]) == set(granted)
    assert saved["scopes"] == REQUIRED
    assert saved["refresh_token"] == "private-refresh-token"
    if len(granted) > len(REQUIRED):
        assert len(app.warnings) == 1
        warning = app.warnings[0]
        assert type(warning) is Warning
        token = getattr(warning, "token")
        assert isinstance(token, app.parameters.OAuth2Token)
        assert set(getattr(warning, "old_scope")) == set(REQUIRED)
        assert set(getattr(warning, "new_scope")) == set(granted)
        assert session.token is token
        assert session._client.refresh_token == "private-refresh-token"
        assert session._client._expires_at == token["expires_at"]
    else:
        assert app.warnings == []
    assert STATE_KEY not in app.db.docs
    assert COOLDOWN_KEY not in app.db.docs
    assert app.google._reauth_required is False
    output = capsys.readouterr().out + caplog.text
    for private in (*PRIVATE_VALUES, "Scope has changed from"):
        assert private not in output


def test_real_parser_omitted_scope_means_requested_permissions_unchanged(app):
    app.payload.pop("scope")

    assert complete(app) is True

    app.http.assert_called_once()
    assert app.warnings == []
    assert "scope" not in app.flows[0].oauth2session.token
    assert app.google._cached_creds.granted_scopes is None
    saved = json.loads(app.db.docs[TOKEN_KEY]["credentials_json"])
    assert set(saved["granted_scopes"]) == set(REQUIRED)
    assert saved["scopes"] == REQUIRED


@pytest.mark.parametrize("normal_return", [False, True], ids=["scope-warning", "normal-token-return"])
def test_missing_required_scope_is_rejected_before_persistence(app, normal_return, capsys, caplog):
    app.payload["scope"] = " ".join(REQUIRED[:-1] + PREVIOUS_GRANTS)
    app.normal_scope_return = normal_return

    assert_safe_failure(app, "insufficient_scope", capsys, caplog)

    app.http.assert_called_once()
    assert len(app.warnings) == 1
    assert isinstance(app.warnings[0].token, app.parameters.OAuth2Token)
    assert app.flows[0].oauth2session.scope == REQUIRED  # Requested scopes alone would wrongly pass.
    assert app.db.docs[TOKEN_KEY] == {"credentials_json": "previous-credentials"}
    app.file_write.assert_not_called()


@pytest.mark.parametrize("scope", ["", [], None])
def test_explicit_empty_scope_is_not_treated_as_omitted_scope(app, scope, capsys, caplog):
    app.payload["scope"] = scope

    assert_safe_failure(app, "insufficient_scope", capsys, caplog)

    app.http.assert_called_once()
    app.file_write.assert_not_called()


@pytest.mark.parametrize("failure", ["invalid_grant", "missing_token", "empty_token"])
def test_real_parser_provider_and_token_errors_are_not_false_expiry(app, failure, capsys, caplog):
    app.payload["scope"] = " ".join(REQUIRED + PREVIOUS_GRANTS)
    if failure == "invalid_grant":
        app.payload.update(error="invalid_grant", error_description="private-provider-description")
    elif failure == "missing_token":
        app.payload.pop("access_token")
    else:
        app.payload["access_token"] = ""

    assert_safe_failure(app, "exchange_failed", capsys, caplog)

    app.http.assert_called_once()
    app.file_write.assert_not_called()
    if failure != "empty_token":
        assert app.warnings == []  # OAuthlib rejects these before scope-change handling.


@pytest.mark.parametrize("warning_kind", ["unrelated", "unrelated_with_token", "missing_metadata"])
def test_unrelated_warnings_are_not_recovered(app, warning_kind, capsys, caplog):
    warning = Warning("private-provider-description")
    if warning_kind != "unrelated":
        token_body = dict(app.payload, scope=" ".join(REQUIRED + PREVIOUS_GRANTS))
        with pytest.raises(Warning) as parsed:
            app.parameters.parse_token_response(json.dumps(token_body), scope=REQUIRED)
        warning = parsed.value
        if warning_kind == "unrelated_with_token":
            warning.args = ("private-provider-description",)
        else:
            delattr(warning, "new_scope")
    app.http.side_effect = warning

    assert_safe_failure(app, "exchange_failed", capsys, caplog)

    app.http.assert_called_once()
    app.file_write.assert_not_called()


@pytest.mark.parametrize("state_kind", ["missing", "unknown", "expired", "malformed"])
def test_only_genuinely_invalid_state_returns_false_without_exchange(app, state_kind):
    state = "private-state"
    if state_kind == "missing":
        state = ""
    elif state_kind == "unknown":
        app.db.docs.pop(STATE_KEY)
    elif state_kind == "expired":
        app.db.docs[STATE_KEY]["expires_at"] = app.google.time.time() - 1
    else:
        app.db.docs[STATE_KEY]["expires_at"] = "not-a-timestamp"

    assert complete(app, state=state) is False

    app.http.assert_not_called()
    app.factory.assert_not_called()
    assert_no_recovery(app)


def test_valid_state_with_missing_code_is_exchange_failure(app):
    with pytest.raises(app.google.GoogleReauthError) as exc:
        complete(app, code="")
    assert exc.value.reason == "exchange_failed"
    app.http.assert_not_called()
    assert_no_recovery(app)


def test_state_storage_read_failure_is_not_reported_as_expired(app, capsys, caplog):
    app.db.fail_reads.add(STATE_KEY)

    assert_safe_failure(app, "persistence_failed", capsys, caplog)

    app.http.assert_not_called()
    app.file_write.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_config", "client_construction"])
def test_configuration_failures_have_safe_distinct_error(app, failure, capsys, caplog):
    if failure == "missing_config":
        app.google._load_web_client_config_from_sources.return_value = None
    else:
        app.factory.side_effect = ValueError("private-provider-description")

    assert_safe_failure(app, "config_unavailable", capsys, caplog)

    app.http.assert_not_called()
    app.file_write.assert_not_called()


def test_strict_callback_save_failure_keeps_previous_credentials_flags_and_cooldown(app, capsys, caplog):
    app.payload["scope"] = " ".join(REQUIRED + PREVIOUS_GRANTS)
    app.db.fail_writes.add(TOKEN_KEY)

    assert_safe_failure(app, "persistence_failed", capsys, caplog)

    app.http.assert_called_once()
    assert app.db.docs[TOKEN_KEY] == {"credentials_json": "previous-credentials"}
    assert STATE_KEY in app.db.docs
    app.file_write.assert_not_called()


def test_state_cleanup_storage_failure_does_not_announce_recovery(app, capsys, caplog):
    app.db.fail_deletes.add(STATE_KEY)

    assert_safe_failure(app, "persistence_failed", capsys, caplog)

    app.http.assert_called_once()
    assert json.loads(app.db.docs[TOKEN_KEY]["credentials_json"])["token"] == "private-access-token"


def test_non_strict_refresh_persistence_retains_best_effort_contract(app):
    app.db.fail_writes.add(TOKEN_KEY)
    credentials = SimpleNamespace(to_json=lambda: '{"token": "legacy-token"}')

    assert app.google._persist_credentials(credentials) is None

    app.file_write.assert_called_once_with('{"token": "legacy-token"}')
    assert app.db.docs[TOKEN_KEY] == {"credentials_json": "previous-credentials"}


def test_firestore_writer_reports_confirmed_success_and_failure(app):
    assert app.google._write_credentials_to_firestore("first-save") is True
    app.db.fail_writes.add(TOKEN_KEY)
    assert app.google._write_credentials_to_firestore("failed-save") is False
    assert app.db.docs[TOKEN_KEY]["credentials_json"] == "first-save"


@pytest.mark.parametrize("reason", ["insufficient_scope", "exchange_failed", "config_unavailable", "persistence_failed"])
def test_typed_error_exposes_reason_and_static_message(app, reason):
    error = app.google.GoogleReauthError(reason)
    assert error.reason == reason
    assert str(error) == app.google.GoogleReauthError._MESSAGES[reason]


def test_start_keeps_fixed_required_scopes_state_ttl_and_single_use_ticket(app):
    app.db.docs[("google_auth_pending", "ticket:start-ticket")] = {"used": False, "expires_at": 100550.0}

    url = asyncio.run(app.google.start_web_reauth(CALLBACK, "start-ticket"))

    query = parse_qs(urlsplit(url).query)
    assert set(query["scope"][0].split()) == set(REQUIRED)
    assert "include_granted_scopes" not in query
    assert query["redirect_uri"] == [CALLBACK]
    assert query["prompt"] == ["consent"]
    state = app.db.docs[("google_auth_pending", query["state"][0])]
    assert state["expires_at"] - state["created_at"] == 600
    assert asyncio.run(app.google.start_web_reauth(CALLBACK, "start-ticket")) is None
    app.factory.assert_called_once()
    app.http.assert_not_called()
