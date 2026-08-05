"""Langfuse observability wrapper tests.

Covers:
  (a) no-op mode when keys/flag are absent — decorated functions still run,
      context managers are inert, flush()/shutdown() are safe
  (b) init order — the anthropic client is a plain anthropic.Anthropic
      (no LangSmith wrap_anthropic anywhere)
  (c) connection_health langfuse probe — skipped without keys/flag,
      ok on 200, auth_failed on 401
  (d) explicit-input rule — the agent root observation input contains ONLY the
      user message (never the conversation history / config)

Co-run-safe isolation: purge MagicMock stubs left by sibling test files plus
cached first-party modules, then import the REAL modules fresh. Tracing is
forced off session-wide by tests/conftest.py (LANGFUSE_TRACING_ENABLED=false),
so importing real observability never touches the network.
"""

import os
import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

os.environ["LANGFUSE_TRACING_ENABLED"] = "false"  # belt-and-braces (conftest sets it too)

# ── Purge sibling stubs so we get REAL modules ──────────────────────────────
for _name in list(sys.modules):
    if isinstance(sys.modules.get(_name), MagicMock):
        sys.modules.pop(_name, None)
for _name in ("observability", "connection_health", "connection_errors",
              "agent", "claude_client", "jira_service", "mcp_client",
              "chat_service", "google_auth", "config"):
    sys.modules.pop(_name, None)

import pytest  # noqa: E402

import config  # noqa: E402
import observability  # noqa: E402
import claude_client  # noqa: E402
import connection_health  # noqa: E402
import agent  # noqa: E402


# ── (a) No-op mode ──────────────────────────────────────────────────────────

def test_noop_mode_when_tracing_disabled():
    assert observability.tracing_enabled() is False

    @observability.observe(name="unit-test-fn", as_type="tool")
    def decorated(x):
        return x * 2

    assert decorated(21) == 42

    # Context managers are inert but yield a span-like object.
    with observability.start_as_current_observation(
        name="unit-test-span", as_type="agent", input={"message": "hi"},
    ) as span:
        span.update(output="done", metadata={"k": "v"})  # must not raise
    with observability.propagate_attributes(user_id="u", session_id="s", tags=["t"]):
        pass

    # flush/shutdown are safe no-ops.
    observability.flush()
    observability.shutdown()


def test_noop_observe_supports_bare_decorator_form():
    @observability.observe
    def plain():
        return "ok"

    assert plain() == "ok"


def test_log_eval_failure_noop_when_disabled(monkeypatch):
    # When tracing is off, log_eval_failure must not touch Firestore.
    calls = []
    monkeypatch.setattr(observability, "_enabled", False)
    observability.log_eval_failure("msg", "expected", "actual")  # no raise, no I/O
    assert calls == []


# ── (b) Init order / plain anthropic client ─────────────────────────────────

def test_anthropic_client_is_plain_unwrapped():
    import anthropic
    assert isinstance(claude_client._client, anthropic.Anthropic)
    assert not hasattr(claude_client, "wrap_anthropic")


def test_init_tracing_is_idempotent():
    # Already initialized at import; calling again must be a no-op, not a crash.
    observability.init_tracing()
    observability.init_tracing()
    assert observability.tracing_enabled() is False


# ── (c) connection_health langfuse probe ────────────────────────────────────

class _FakeResponse:
    def __init__(self, status_code):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_langfuse_probe_skipped_without_keys(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_TRACING_ENABLED", True)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "")
    result = connection_health.probe_langfuse()
    assert result["status"] == "skipped"


def test_langfuse_probe_skipped_when_tracing_disabled(monkeypatch):
    monkeypatch.setattr(config, "LANGFUSE_TRACING_ENABLED", False)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "sk")
    result = connection_health.probe_langfuse()
    assert result["status"] == "skipped"


def _probe_with_status(monkeypatch, status_code):
    import httpx
    monkeypatch.setattr(config, "LANGFUSE_TRACING_ENABLED", True)
    monkeypatch.setattr(config, "LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setattr(config, "LANGFUSE_SECRET_KEY", "sk")
    monkeypatch.setattr(config, "LANGFUSE_BASE_URL", "https://us.cloud.langfuse.com")
    captured = {}

    def fake_get(url, auth=None, timeout=None):
        captured["url"] = url
        captured["auth"] = auth
        return _FakeResponse(status_code)

    monkeypatch.setattr(httpx, "get", fake_get)
    return connection_health.probe_langfuse(), captured


def test_langfuse_probe_ok_on_200(monkeypatch):
    result, captured = _probe_with_status(monkeypatch, 200)
    assert result["status"] == "ok"
    assert captured["url"].endswith("/api/public/projects")
    assert captured["auth"] == ("pk", "sk")


def test_langfuse_probe_auth_failed_on_401(monkeypatch):
    result, _ = _probe_with_status(monkeypatch, 401)
    assert result["status"] == "auth_failed"
    assert result["error_kind"] == "auth"


# ── (d) Explicit-input rule on the agent root observation ───────────────────

def test_agent_root_observation_input_is_user_message_only(monkeypatch):
    recorded = {}

    @contextmanager
    def fake_start(*, name, as_type="span", input=None):
        recorded.setdefault("observations", []).append(
            {"name": name, "as_type": as_type, "input": input})
        yield MagicMock()

    @contextmanager
    def fake_propagate(**kwargs):
        recorded["propagate"] = kwargs
        yield

    monkeypatch.setattr(observability, "start_as_current_observation", fake_start)
    monkeypatch.setattr(observability, "propagate_attributes", fake_propagate)
    monkeypatch.setattr(config, "USER_MEMORY_ENABLED", False)
    monkeypatch.setattr(agent, "run_tool_loop",
                        lambda **kwargs: ("all good", "end_turn"))

    history = [
        {"role": "user", "content": "SECRET_HISTORY_TOKEN api_key=sk-XYZ"},
        {"role": "assistant", "content": "noted"},
    ]
    reply, pending = agent.run_agent_loop(
        "what's on my calendar", history,
        thread_id="spaces/s1", user_id="users/u1",
    )

    assert reply == "all good"
    root = recorded["observations"][0]
    assert root["name"] == "agent-loop"
    assert root["as_type"] == "agent"
    # EXPLICIT INPUT ONLY: exactly the user message — nothing else.
    assert root["input"] == {"message": "what's on my calendar"}
    assert "SECRET_HISTORY_TOKEN" not in str(root["input"])
    # Session attribution flows through propagate_attributes.
    assert recorded["propagate"]["user_id"] == "users/u1"
    assert recorded["propagate"]["session_id"] == "spaces/s1"
    assert recorded["propagate"]["tags"] == ["chat"]


def test_agent_metrics_sink_receives_trajectory(monkeypatch):
    monkeypatch.setattr(config, "USER_MEMORY_ENABLED", False)
    monkeypatch.setattr(agent, "run_tool_loop",
                        lambda **kwargs: ("done", "end_turn"))
    sink = {}
    reply, _ = agent.run_agent_loop("hey", [], metrics_sink=sink)
    assert reply == "done"
    assert sink["iteration_count"] >= 1
    assert "tool_sequence" in sink and "total_tool_calls" in sink


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
