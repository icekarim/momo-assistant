"""Safety tests for the dedicated Jira write-approval flow in main.py.

Asserts the two core guarantees:
  (b) a bare "yes" NEVER confirms a Jira write — explicit Jira-scoped confirm only
      the applier refuses any non-Jira action (allow-list).
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# Mock external dependencies before importing the app modules (mirrors the
# established pattern in test_task_approval_safety.py).
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["google.auth"] = MagicMock()
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


sys.modules["cachetools"] = MagicMock(TTLCache=DummyCache)
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()

langsmith_mock = MagicMock()


def _traceable_mock(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


langsmith_mock.traceable = _traceable_mock
sys.modules["langsmith_config"] = langsmith_mock

config_mock = MagicMock()
config_mock.CHAT_SPACE_ID = "spaces/test_space"
config_mock.AGENTIC_MODE_ENABLED = True
config_mock.KNOWLEDGE_GRAPH_ENABLED = False
config_mock.MAX_CHAT_EMAILS = 5
sys.modules["config"] = config_mock

sys.modules["briefing"] = MagicMock()
sys.modules["gmail_service"] = MagicMock()
sys.modules["calendar_service"] = MagicMock()
sys.modules["tasks_service"] = MagicMock()
sys.modules["gemini_service"] = MagicMock()
sys.modules["chat_service"] = MagicMock(
    format_for_google_chat=lambda text: text,
    send_chat_message=MagicMock(),
    download_attachment=MagicMock(),
    _SUPPORTED_AUDIO_TYPES=frozenset(["audio/mp3"]),
)
sys.modules["conversation_store"] = MagicMock(
    get_conversation=MagicMock(),
    add_turn=MagicMock(),
    conversation_scope=MagicMock(),
)
sys.modules["agent"] = MagicMock()

import main  # noqa: E402


_CREATE = {"action": "create_jira", "project": "OSD", "summary": "Test"}
_COMMENT = {"action": "comment_jira", "key": "OSD-123", "comment": "hi"}


class TestJiraReplyParsing(unittest.TestCase):
    def test_bare_yes_never_confirms(self):
        for word in ("yes", "approve", "approved", "confirm", "ok", "go ahead"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertEqual(parsed["intent"], "needs_explicit",
                             f"bare '{word}' must NOT confirm a Jira write")

    def test_explicit_jira_confirm(self):
        for phrase in ("confirm jira", "yes jira", "approve jira"):
            parsed = main._parse_pending_jira_reply(phrase, [_CREATE])
            self.assertEqual(parsed["intent"], "confirm")
            self.assertEqual(parsed["selected_indices"], {0})

    def test_ticket_key_selective_confirm(self):
        actions = [_CREATE, _COMMENT]
        parsed = main._parse_pending_jira_reply("yes osd-123", actions)
        self.assertEqual(parsed["intent"], "confirm")
        self.assertEqual(parsed["selected_indices"], {1})  # only the OSD-123 action

    def test_decline_words(self):
        for word in ("no", "cancel", "nope", "stop"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertEqual(parsed["intent"], "decline")

    def test_unrelated_message_is_none(self):
        parsed = main._parse_pending_jira_reply("what's the status of the deal", [_CREATE])
        self.assertIsNone(parsed["intent"])


class TestJiraIntentGuard(unittest.TestCase):
    def test_intent_shapes(self):
        self.assertTrue(main._check_pending_jira_intent("confirm jira"))
        self.assertTrue(main._check_pending_jira_intent("no"))
        self.assertTrue(main._check_pending_jira_intent("yes osd-1"))
        # bare yes is NOT jira-shaped (so it can't phantom-fire a jira write)
        self.assertFalse(main._check_pending_jira_intent("yes"))
        self.assertFalse(main._check_pending_jira_intent("tell me about the project"))


class TestApplierAllowList(unittest.TestCase):
    @patch("main.record_jira_write_audit", MagicMock())
    @patch("main.add_turn", MagicMock())
    @patch("main.send_chat_message")
    def test_non_jira_action_is_refused(self, mock_send):
        import jira_service
        called = {"hit": False}

        def _track(*a, **k):
            called["hit"] = True
            return {"success": True}

        with patch.object(jira_service, "create_jira_ticket", _track), \
             patch.object(jira_service, "add_jira_comment", _track), \
             patch.object(jira_service, "transition_jira_ticket", _track):
            # A disallowed/foreign action must never reach a REST write.
            main._apply_pending_jira_actions_background(
                [{"action": "delete", "find": "some task"}],
                [], "spaces/test", "user:scope", None, "confirm jira",
            )
        self.assertFalse(called["hit"], "a non-Jira action must not call any REST write")
        sent = mock_send.call_args.args[1]
        self.assertIn("failed", sent.lower())


if __name__ == "__main__":
    unittest.main()
