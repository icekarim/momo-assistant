"""Core reconnect tool contracts. All credentials, persistence and Chat IO are fake."""

import importlib
import importlib.machinery
import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import pytest


def _load_fresh_module(name):
    """Load real code without accepting a sibling's cached MagicMock module."""
    spec = importlib.machinery.PathFinder.find_spec(name)
    assert spec is not None and spec.loader is not None, name
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    assert isinstance(module, ModuleType) and not isinstance(module, MagicMock)
    return module


@contextmanager
def isolated_module_registry(*roots):
    """Restore only replaced namespaces, not unrelated lazy SDK imports.

    Clearing all of sys.modules on exit breaks packages such as Pydantic that
    cache lazy imports as package attributes. Their new, real submodules must
    remain registered. Our replaced modules and their children are restored
    exactly, including any sibling mocks.
    """
    roots = set(roots)
    saved = {name: module for name, module in sys.modules.items() if name.split(".")[0] in roots}
    try:
        yield
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                sys.modules.pop(name)
        sys.modules.update(saved)


@contextmanager
def isolated_auth_modules():
    """Private first-party module graph; restore replaced namespaces afterward.

    Legacy siblings replace google/config/etc during collection. Never evict
    those globally or reuse first-party modules already bound to their mocks.
    Google SDK IO is deliberately stubbed, while classifiers, tool dispatch,
    providers and history logic are loaded from their real production files.
    Fresh stub packages also avoid corrupting an existing google namespace's
    cached subpackages. No external clients may connect in this test context.
    """
    names = (
        "config", "connection_errors", "reauth_service", "google_auth", "chat_service",
        "conversation_store", "observability", "claude_client", "agent", "granola_service",
        "connection_health", "calendar_service", "jira_service", "mcp_client",
    )
    with isolated_module_registry(
        *names, "google", "googleapiclient", "google_auth_oauthlib", "anthropic", "cachetools",
    ), patch.dict("os.environ", {"LANGFUSE_TRACING_ENABLED": "false"}):
        for name in (
            "google", "google.auth", "google.auth.transport", "google.auth.transport.requests",
            "google.oauth2", "google.oauth2.credentials", "google.cloud", "google.cloud.firestore",
            "google_auth_oauthlib", "google_auth_oauthlib.flow",
        ):
            module = ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, module)
        blocked = lambda: MagicMock(side_effect=AssertionError("unstubbed external SDK boundary"))
        sys.modules["google.auth"].default = blocked()
        sys.modules["google.auth.transport.requests"].Request = MagicMock()
        sys.modules["google.auth.transport.requests"].AuthorizedSession = blocked()
        sys.modules["google.oauth2.credentials"].Credentials = blocked()
        sys.modules["google.cloud.firestore"].Client = blocked()
        sys.modules["google_auth_oauthlib.flow"].Flow = blocked()
        # Keep real HttpError semantics but never perform discovery/network IO.
        for name in list(sys.modules):
            if name == "googleapiclient" or name.startswith("googleapiclient."):
                sys.modules.pop(name)
        _load_fresh_module("googleapiclient")
        discovery = ModuleType("googleapiclient.discovery")
        discovery.build = blocked()
        sys.modules["googleapiclient.discovery"] = discovery
        sys.modules["googleapiclient"].discovery = discovery
        # Agent tests exercise the real loop/enum, not the Anthropic transport.
        anthropic = ModuleType("anthropic")
        anthropic.Anthropic = MagicMock(return_value=MagicMock())
        anthropic.Anthropic.return_value.messages.create = blocked()
        sys.modules["anthropic"] = anthropic
        _load_fresh_module("cachetools")
        modules = {}
        for name in names:
            modules[name] = _load_fresh_module(name)
            assert Path(modules[name].__file__).resolve() == Path(__file__).resolve().parents[1] / f"{name}.py"
        assert modules["config"].LANGFUSE_TRACING_ENABLED is False
        yield modules


@pytest.fixture
def real_auth_modules():
    with isolated_auth_modules() as modules:
        yield modules


@pytest.fixture
def app(monkeypatch, real_auth_modules):
    modules = real_auth_modules
    cfg = modules["config"]
    monkeypatch.setattr(cfg, "MOMO_SERVICE_URL", "https://momo.example/")
    monkeypatch.setattr(cfg, "CHAT_SPACE_ID", "spaces/reauth-test")
    for flag in ("GRANOLA_ENABLED", "JIRA_ENABLED", "MCP_ENABLED", "USER_MEMORY_ENABLED"):
        monkeypatch.setattr(cfg, flag, False)
    db = MagicMock()
    # A just-sent automatic alert must not inhibit explicit requests.
    snapshot = db.collection.return_value.document.return_value.get.return_value
    snapshot.exists = True
    snapshot.to_dict.return_value = {"sent_at": modules["google_auth"].time.time()}
    monkeypatch.setattr(modules["google_auth"], "_get_db", lambda: db)
    for provider, method in (("google_auth", "get_credentials"), ("granola_service", "_load_token")):
        monkeypatch.setattr(modules[provider], method, MagicMock(side_effect=AssertionError("must not check auth")))
    send = MagicMock(side_effect=AssertionError("explicit links must not send Chat alerts"))
    monkeypatch.setattr(modules["chat_service"], "send_chat_message", send)
    modules.update(db=db, send=send)
    return modules


def test_auth_module_isolation_restores_sibling_mocks(monkeypatch):
    sibling_modules = {name: MagicMock() for name in (
        "google", "google_auth", "config", "connection_errors", "agent", "granola_service",
    )}
    for name, module in sibling_modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    with isolated_auth_modules() as modules:
        assert modules["calendar_service"].raise_if_google_auth_error is modules["google_auth"].raise_if_google_auth_error
        assert modules["google_auth"].ExternalAuthError is modules["connection_errors"].ExternalAuthError
        assert modules["agent"].ExternalConnectionError is modules["connection_errors"].ExternalConnectionError
    for name, module in sibling_modules.items():
        assert sys.modules[name] is module


def dispatch(app, service="all"):
    return json.loads(app["agent"].execute_tool("get_reauth_links", {"service": service}))


def test_reauth_tool_registered_in_core_with_optional_integrations_disabled(app):
    core = [tool for tool in app["agent"]._CORE_TOOLS if tool["name"] == "get_reauth_links"]
    assert len(core) == 1
    assert core[0] in app["agent"]._get_all_tools()
    assert core[0]["input_schema"]["properties"]["service"]["enum"] == [
        "google_workspace", "granola", "all",
    ]


@pytest.mark.parametrize("service,expected", [
    ("all", {"google_workspace", "granola"}),
    ("google_workspace", {"google_workspace"}),
    ("granola", {"granola"}),
])
def test_core_tool_dispatch_all_and_single_provider(app, service, expected):
    result = dispatch(app, service)
    assert result["status"] == "ok"
    assert set(result["providers"]) == expected
    for provider, outcome in result["providers"].items():
        parsed = urlsplit(outcome["url"])
        assert parsed.scheme == "https"
        assert parsed.netloc == "momo.example"
        if provider == "google_workspace":
            assert parsed.path == "/google-auth/start"
            assert parse_qs(parsed.query)["t"][0]
            assert outcome["single_use"] is True
            assert outcome["expires_in_seconds"] == 600
        else:
            assert outcome["url"] == "https://momo.example/granola-auth/start"
    if service == "granola":
        app["db"].collection.assert_not_called()
    app["send"].assert_not_called()


@pytest.mark.parametrize("service", ["invalid", "google", "", None, [], {}])
def test_invalid_service_is_structured_and_does_not_issue_tickets(app, service):
    result = dispatch(app, service)
    assert result["status"] == "error"
    assert result["error"] == "invalid_service"
    assert result["providers"] == {}
    app["db"].collection.assert_not_called()


def test_missing_config_returns_provider_failures_without_relative_links(app, monkeypatch):
    monkeypatch.setattr(app["config"], "MOMO_SERVICE_URL", "")
    result = dispatch(app)
    assert result["status"] == "error"
    for outcome in result["providers"].values():
        assert outcome["error"] == "missing_config"
        assert "url" not in outcome
    app["db"].collection.assert_not_called()


@pytest.mark.parametrize("url", [
    "/relative", "//momo.example", "http://momo.example", "https://",
    "https://user:secret@momo.example", "https://momo.example/path",
    "https://momo.example?x=1", "https://momo.example#fragment",
    "https://momo.example?", "https://momo.example#", "https://momo.example:",
    "https://momo.example:99999", "https://momo.example:0",
    "https://momo.example\n", "https://momo.example\\@evil.example",
    "https://momo.example|bad", "https://momo.example/<bad>",
    "https://momo%2eexample", "https://momo.example\x00", "https://-bad.example",
])
def test_unsafe_service_url_is_rejected_before_ticket_creation(app, monkeypatch, url):
    monkeypatch.setattr(app["config"], "MOMO_SERVICE_URL", url)
    result = dispatch(app)
    assert result["status"] == "error"
    assert all(item["error"] == "invalid_config" for item in result["providers"].values())
    app["db"].collection.assert_not_called()


@pytest.mark.parametrize("provider", ["google_auth", "granola_service"])
def test_provider_helpers_reject_untrusted_override(app, provider):
    with pytest.raises(app["reauth_service"].ReauthLinkError):
        app[provider].create_reauth_link("https://untrusted.example")
    app["db"].collection.assert_not_called()


@pytest.mark.parametrize("failed_provider,module", [
    ("google_workspace", "google_auth"), ("granola", "granola_service"),
])
def test_one_provider_failure_preserves_other_link_without_leaking_error(app, monkeypatch, capsys,
                                                                      failed_provider, module):
    monkeypatch.setattr(app[module], "create_reauth_link", MagicMock(
        side_effect=RuntimeError("refresh_token=private client_secret=secret ticket=secret")
    ))
    result = dispatch(app)
    assert result["status"] == "partial"
    assert result["providers"][failed_provider]["error"] == "link_unavailable"
    other = "granola" if failed_provider == "google_workspace" else "google_workspace"
    assert result["providers"][other]["url"].startswith("https://momo.example/")
    assert "secret" not in json.dumps(result) + capsys.readouterr().out


def test_ticket_store_failure_preserves_granola_link(app):
    app["db"].collection.return_value.document.return_value.set.side_effect = RuntimeError("offline")
    result = dispatch(app)
    assert result["status"] == "partial"
    assert result["providers"]["google_workspace"]["error"] == "ticket_unavailable"
    assert result["providers"]["granola"]["url"] == "https://momo.example/granola-auth/start"


def test_repeated_explicit_requests_create_new_tickets_during_cooldown(app, capsys):
    google = app["google_auth"]
    assert google._should_send_throttled_reauth_alert() is False
    app["db"].reset_mock()
    first = dispatch(app, "google_workspace")["providers"]["google_workspace"]["url"]
    second = dispatch(app, "google_workspace")["providers"]["google_workspace"]["url"]
    assert first != second
    docs = app["db"].collection.return_value.document
    assert docs.call_count == 2
    assert all(call.args[0].startswith("ticket:") for call in docs.call_args_list)
    docs.return_value.get.assert_not_called()  # no cooldown reads on explicit path
    app["send"].assert_not_called()
    logs = capsys.readouterr().out
    for url in (first, second):
        assert parse_qs(urlsplit(url).query)["t"][0] not in logs


def test_explicit_request_does_not_need_configured_chat_space(app, monkeypatch):
    monkeypatch.setattr(app["config"], "CHAT_SPACE_ID", "")
    assert dispatch(app)["status"] == "ok"
    app["send"].assert_not_called()


def test_prompt_requires_tool_and_respectful_support(app):
    prompt = app["agent"].AGENT_SYSTEM_PROMPT
    for instruction in (
        "call get_reauth_links FIRST", "Never say you cannot provide a link without checking",
        "Missing conversation history is not proof the user is wrong", "Acknowledge quoted notifications",
        "single-use and expire in 10 minutes", "still share the other provider's valid link",
    ):
        assert instruction in prompt


def test_reauth_tool_span_does_not_capture_ticket_urls(app, monkeypatch):
    agent = app["agent"]
    observation = MagicMock()
    span = observation.return_value.__enter__.return_value
    monkeypatch.setattr(agent.observability, "start_as_current_observation", observation)
    results = []

    def fake_loop(**kwargs):
        results.append(kwargs["dispatch"]("get_reauth_links", {"service": "all"}))
        return "here are your reconnect links", "end_turn"

    monkeypatch.setattr(agent, "run_tool_loop", fake_loop)
    agent._run_agent_loop_inner("reconnect both", [])
    link = json.loads(results[0])["providers"]["google_workspace"]["url"]
    ticket = parse_qs(urlsplit(link).query)["t"][0]
    assert ticket not in str(span.update.call_args_list)
    assert "withheld" in str(span.update.call_args_list)
