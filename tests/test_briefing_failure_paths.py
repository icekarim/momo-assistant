"""Morning-briefing silent-failure regression tests.

Regression for the incident where the briefing model (a reasoning model)
spent its entire max_tokens budget on a thinking block (stop_reason "length",
ZERO text blocks): extract_text() returned "", the empty string was sent to
Google Chat (API 400 "Message cannot be empty"), yet the pipeline logged
"delivered" and /briefing returned 200 — a silent failure. Covers the three
new seams:

  (a) gemini_service.generate_morning_briefing — passes the dedicated
      briefing budget, retries ONCE with double budget on empty text, and
      raises RuntimeError (never returns "") if still empty
  (b) chat_service.send_chat_message — refuses empty/whitespace text without
      calling the API
  (c) briefing.run_morning_briefing — generation errors send a best-effort
      fallback notice and propagate; a failed Chat send yields status
      "failed" (no "delivered" log path, no proactive-message store)

Every test is hermetic: Claude/Chat/Gmail/Firestore boundaries are
monkeypatched (conventions follow tests/test_p1_consumers.py and
tests/test_meeting_prep_relevance.py — no network, no LLM calls).
"""

import sys
from unittest.mock import MagicMock

import pytest

# Earlier-collected sibling tests (e.g. test_addon_envelope_risk.py) install
# sys.modules mocks at import time and only restore them at teardown, so a
# full-suite run would hand this module MagicMock modules during collection.
# Evict leaked mocks for the modules that MUST be real here so a fresh real
# import happens (same trick as tests/test_chat_service_cards.py:25). Leaked
# mocks for heavy transitive deps (google*, googleapiclient, gmail/calendar/
# tasks/granola/conversation_store/knowledge_graph) are deliberately KEPT —
# every test below patches those seams explicitly, and re-importing the real
# google namespace mid-session corrupts its partially-cached subpackages.
_REAL_NEEDED = {
    "briefing", "chat_service", "config", "gemini_service", "claude_client",
    "observability", "connection_errors", "anthropic",
}
for _name in list(sys.modules):
    if _name.split(".")[0] in _REAL_NEEDED and isinstance(sys.modules[_name], MagicMock):
        del sys.modules[_name]

# When a leaked mock google namespace is present, real imports THROUGH it are
# impossible (a MagicMock is not a package) — seed the missing links. In a
# clean (solo) run nothing here fires and every module imports for real.
if isinstance(sys.modules.get("google"), MagicMock) or isinstance(sys.modules.get("google.auth"), MagicMock):
    sys.modules.setdefault("google.auth.transport", MagicMock())
    sys.modules.setdefault("google.auth.transport.requests", MagicMock())
    # First-party deps of briefing whose real imports are blocked by the
    # mocked google namespace. Every briefing-level binding from these is
    # explicitly patched in the tests below.
    for _dep in ("gmail_service", "calendar_service", "tasks_service",
                 "granola_service", "conversation_store", "knowledge_graph"):
        sys.modules.setdefault(_dep, MagicMock())

import briefing  # noqa: E402
import chat_service  # noqa: E402
import config  # noqa: E402
import gemini_service  # noqa: E402
from connection_errors import ExternalAuthError  # noqa: E402


def _msg(text=None, stop_reason="max_tokens"):
    """Fake Anthropic message. text=None → zero text blocks (all-thinking)."""
    content = []
    if text is not None:
        content.append(type("B", (), {"type": "text", "text": text})())
    return type("M", (), {"stop_reason": stop_reason, "content": content})()


# ════════════════════════════════════════════════════════════
# (a) generate_morning_briefing — budget, retry-once, raise
# ════════════════════════════════════════════════════════════


def test_briefing_passes_dedicated_budget_and_returns_text(monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_MAX_TOKENS_BRIEFING", 8192)
    calls = []

    def fake_generate(prompt=None, **kw):
        calls.append(kw)
        return _msg("the briefing", stop_reason="end_turn")

    monkeypatch.setattr(gemini_service, "generate", fake_generate)

    result = gemini_service.generate_morning_briefing("e", "m", "t")

    assert result == "the briefing"
    assert len(calls) == 1
    assert calls[0]["max_tokens"] == 8192


def test_briefing_retries_once_with_double_budget_then_recovers(monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_MAX_TOKENS_BRIEFING", 8192)
    calls = []

    def fake_generate(prompt=None, **kw):
        calls.append(kw)
        if len(calls) == 1:
            return _msg()  # all-reasoning: zero text blocks
        return _msg("recovered briefing", stop_reason="end_turn")

    monkeypatch.setattr(gemini_service, "generate", fake_generate)

    result = gemini_service.generate_morning_briefing("e", "m", "t")

    assert result == "recovered briefing"
    assert [c["max_tokens"] for c in calls] == [8192, 16384]


def test_briefing_raises_after_persistently_empty_text(monkeypatch):
    monkeypatch.setattr(config, "CLAUDE_MAX_TOKENS_BRIEFING", 8192)
    calls = []

    def fake_generate(prompt=None, **kw):
        calls.append(kw)
        return _msg()  # empty every time

    monkeypatch.setattr(gemini_service, "generate", fake_generate)

    with pytest.raises(RuntimeError, match="no text.*max_tokens"):
        gemini_service.generate_morning_briefing("e", "m", "t")

    assert len(calls) == 2, "must retry exactly once, then raise — never return ''"


# ════════════════════════════════════════════════════════════
# (b) send_chat_message — empty-text guard
# ════════════════════════════════════════════════════════════


@pytest.mark.parametrize("empty_text", ["", "   \n\t ", None])
def test_send_chat_message_refuses_empty_text(monkeypatch, empty_text):
    session_factory = MagicMock()
    monkeypatch.setattr(chat_service, "_get_chat_session", session_factory)

    assert chat_service.send_chat_message("spaces/s", text=empty_text) is False
    session_factory.assert_not_called()  # API never touched


def test_send_chat_message_cards_path_unaffected_by_guard(monkeypatch):
    session = MagicMock()
    session.post.return_value = MagicMock(status_code=200)
    monkeypatch.setattr(chat_service, "_get_chat_session", lambda: session)

    cards = [{"cardId": "c"}]
    assert chat_service.send_chat_message("spaces/s", cards=cards) is True
    session.post.assert_called_once()


# ════════════════════════════════════════════════════════════
# (c) run_morning_briefing — failure propagation
# ════════════════════════════════════════════════════════════


@pytest.fixture
def briefing_env(monkeypatch):
    """Hermetic run_morning_briefing: one open task so the pipeline doesn't
    short-circuit to 'nothing to report'; all external boundaries stubbed."""
    monkeypatch.setattr(config, "CHAT_SPACE_ID", "spaces/test")
    monkeypatch.setattr(config, "GRANOLA_ENABLED", False)
    monkeypatch.setattr(config, "JIRA_ENABLED", False)
    monkeypatch.setattr(config, "PROACTIVE_INTELLIGENCE_ENABLED", False)
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    monkeypatch.setattr(briefing, "fetch_unread_client_emails", lambda: [])
    monkeypatch.setattr(briefing, "fetch_todays_meetings", lambda: [])
    monkeypatch.setattr(briefing, "fetch_open_tasks", lambda: [{"title": "t"}])
    monkeypatch.setattr(briefing, "format_emails_for_context", lambda e: "emails")
    monkeypatch.setattr(briefing, "format_meetings_for_context", lambda m: "meetings")
    monkeypatch.setattr(briefing, "format_tasks_for_context", lambda t: "tasks")
    monkeypatch.setattr(briefing, "format_for_google_chat", lambda s: s)
    monkeypatch.setattr(briefing, "_process_debrief_tasks", lambda s, **k: s)
    monkeypatch.setattr(briefing, "_store_proactive_message", MagicMock())
    monkeypatch.setattr(briefing, "_extract_briefing_sources_to_kg", MagicMock())


def test_generation_failure_sends_fallback_notice_and_propagates(monkeypatch, briefing_env):
    def _boom(*a, **k):
        raise RuntimeError("briefing generation returned no text (stop_reason=max_tokens)")

    monkeypatch.setattr(briefing, "generate_morning_briefing", _boom)
    sent = []
    monkeypatch.setattr(briefing, "send_chat_message",
                        lambda space, text=None, **k: sent.append(text) or True)

    with pytest.raises(RuntimeError, match="no text"):
        briefing.run_morning_briefing()

    assert len(sent) == 1, "must attempt exactly one fallback notice"
    assert "failed" in sent[0].lower()


def test_generation_failure_propagates_even_if_fallback_send_raises(monkeypatch, briefing_env):
    monkeypatch.setattr(briefing, "generate_morning_briefing",
                        MagicMock(side_effect=RuntimeError("generation dead")))
    monkeypatch.setattr(briefing, "send_chat_message",
                        MagicMock(side_effect=ConnectionError("chat also dead")))

    with pytest.raises(RuntimeError, match="generation dead"):
        briefing.run_morning_briefing()


def test_chat_send_failure_returns_failed_status(monkeypatch, briefing_env):
    monkeypatch.setattr(briefing, "generate_morning_briefing",
                        lambda *a, **k: "the briefing")
    monkeypatch.setattr(briefing, "send_chat_message", lambda *a, **k: False)
    store_mock = MagicMock()
    monkeypatch.setattr(briefing, "_store_proactive_message", store_mock)

    result = briefing.run_morning_briefing()

    assert result["status"] == "failed"
    assert result["reason"] == "chat delivery failed"
    store_mock.assert_not_called()  # undelivered message must not enter history


def test_successful_send_returns_sent_status(monkeypatch, briefing_env):
    monkeypatch.setattr(briefing, "generate_morning_briefing",
                        lambda *a, **k: "the briefing")
    monkeypatch.setattr(briefing, "send_chat_message", lambda *a, **k: True)
    kg_mock = MagicMock()
    monkeypatch.setattr(briefing, "_extract_briefing_sources_to_kg", kg_mock)

    result = briefing.run_morning_briefing()

    assert result["status"] == "sent"
    kg_mock.assert_called_once()  # post-send KG extraction still runs


@pytest.mark.parametrize("has_other_data", [False, True], ids=["calendar_only", "with_tasks"])
def test_calendar_auth_failure_is_visible_not_silently_empty(monkeypatch, briefing_env, has_other_data):
    monkeypatch.setattr(briefing, "fetch_todays_meetings", MagicMock(
        side_effect=ExternalAuthError("google_workspace", "Calendar credentials expired"),
    ))
    if not has_other_data:
        monkeypatch.setattr(briefing, "fetch_open_tasks", lambda: [])
    generate = MagicMock(return_value="the briefing")
    send = MagicMock(return_value=True)
    monkeypatch.setattr(briefing, "generate_morning_briefing", generate)
    monkeypatch.setattr(briefing, "send_chat_message", send)

    result = briefing.run_morning_briefing()

    assert result["status"] == "sent", "auth failure must not become 'nothing to report'"
    generate.assert_called_once()
    send.assert_called_once()
    assert "Google Calendar connection needs re-auth" in send.call_args.args[1]
    assert "the briefing" in send.call_args.args[1]


@pytest.mark.parametrize("has_other_data", [False, True], ids=["calendar_only", "with_tasks"])
def test_legitimately_empty_calendar_has_no_auth_notice(monkeypatch, briefing_env, has_other_data):
    if not has_other_data:
        monkeypatch.setattr(briefing, "fetch_open_tasks", lambda: [])
    generate = MagicMock(return_value="the briefing")
    send = MagicMock(return_value=True)
    monkeypatch.setattr(briefing, "generate_morning_briefing", generate)
    monkeypatch.setattr(briefing, "send_chat_message", send)

    result = briefing.run_morning_briefing()

    if has_other_data:
        assert result["status"] == "sent"
        generate.assert_called_once()
        send.assert_called_once_with("spaces/test", "the briefing")
    else:
        assert result == {"status": "skipped", "reason": "nothing to report"}
        generate.assert_not_called()
        send.assert_not_called()
