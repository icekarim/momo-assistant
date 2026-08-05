"""Safety tests for the dedicated Jira write-approval flow in main.py.

Asserts the core guarantees:
  (b) a bare "yes" NEVER confirms a Jira write — explicit Jira-scoped confirm only;
      the applier refuses any non-Jira action (allow-list).
  (c) dark launch: with JIRA_WRITE_ENABLED off, the phantom-approval guard's Jira
      half is skipped entirely — normal messages ("go check osd-123 for me",
      "stop", ...) are never swallowed and no pending-Jira Firestore read happens.

Isolation mirrors test_handle_card_click.py:1-97 (HANDOFF_addon_cards.md §8):
identity-decorator fastapi stub so the real async handle_message survives
import, plus sys.modules.pop("main") immediately before `import main`, so this
file is co-run-safe and order-independent with sibling test files.
"""

import sys
import asyncio
import unittest
from unittest.mock import MagicMock, patch

# ── Stub all heavy dependencies BEFORE importing main ──────────────────────
sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["google.auth"] = MagicMock()
sys.modules["googleapiclient"] = MagicMock()
sys.modules["googleapiclient.discovery"] = MagicMock()
sys.modules["googleapiclient.errors"] = MagicMock()


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


sys.modules["cachetools"] = MagicMock(TTLCache=DummyCache)

# Route/middleware decorators must be identity so the real async handle_message
# survives import (the guard regression tests drive it). A bare MagicMock would
# replace it with a non-awaitable mock.
_fastapi_mock = MagicMock()
_app_mock = MagicMock()


def _identity_decorator(*args, **kwargs):
    def deco(func):
        return func
    return deco


_app_mock.get.side_effect = _identity_decorator
_app_mock.post.side_effect = _identity_decorator
_app_mock.api_route.side_effect = _identity_decorator
_app_mock.on_event.side_effect = _identity_decorator
_app_mock.middleware.side_effect = _identity_decorator
_fastapi_mock.FastAPI.return_value = _app_mock
sys.modules["fastapi"] = _fastapi_mock
sys.modules["fastapi.responses"] = MagicMock()

langsmith_mock = MagicMock()


def _traceable_mock(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


langsmith_mock.traceable = _traceable_mock
langsmith_mock.traced_chat_send = MagicMock()
langsmith_mock.traced_generate_content = MagicMock()
sys.modules["langsmith_config"] = langsmith_mock

config_mock = MagicMock()
config_mock.CHAT_SPACE_ID = "spaces/test_space"
config_mock.AGENTIC_MODE_ENABLED = True
config_mock.KNOWLEDGE_GRAPH_ENABLED = False
config_mock.MAX_CHAT_EMAILS = 5
config_mock.MOMO_SERVICE_URL = "https://momo.example"
config_mock.JIRA_WRITE_ENABLED = False  # dark-launch default; tests toggle
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
    get_conversation=MagicMock(return_value=[]),
    add_turn=MagicMock(),
    clear_conversation=MagicMock(),
    conversation_scope=MagicMock(),
    get_pending_task_actions=MagicMock(return_value=None),
    get_pending_jira_actions=MagicMock(return_value=None),
    clear_pending_task_actions=MagicMock(),
    clear_pending_jira_actions=MagicMock(),
    store_pending_task_actions=MagicMock(),
    store_pending_task_actions_if_empty=MagicMock(),
    store_pending_jira_actions_if_empty=MagicMock(),
    record_jira_write_audit=MagicMock(),
    get_task_batch=MagicMock(),
    update_task_batch=MagicMock(),
)
sys.modules["agent"] = MagicMock()

sys.modules.pop("main", None)  # drop any sibling-cached main so the identity-decorator fastapi stub above yields a real coroutine handle_message
import main  # noqa: E402


_CREATE = {"action": "create_jira", "project": "OSD", "summary": "Test"}
_COMMENT = {"action": "comment_jira", "key": "OSD-123", "comment": "hi"}


class TestJiraReplyParsing(unittest.TestCase):
    def test_bare_yes_never_confirms(self):
        for word in ("yes", "approve", "approved", "confirm", "confirmed",
                     "go ahead", "do it"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertEqual(parsed["intent"], "needs_explicit",
                             f"bare '{word}' must NOT confirm a Jira write")

    def test_loose_tokens_are_not_approvals(self):
        # Removed from _JIRA_APPROVE_WORDS: too common in normal conversation.
        for word in ("ok", "okay", "go", "send", "post", "ship", "apply"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertIsNone(parsed["intent"],
                              f"loose '{word}' must not be treated as an approval reply")

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
        for word in ("no", "cancel", "nope", "dont do it"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertEqual(parsed["intent"], "decline")

    def test_bare_stop_dont_no_longer_decline(self):
        # Removed from _JIRA_DECLINE_WORDS: bare "stop"/"don't" are normal
        # conversational messages and must fall through to the agent.
        for word in ("stop", "don't", "dont"):
            parsed = main._parse_pending_jira_reply(word, [_CREATE])
            self.assertIsNone(parsed["intent"],
                              f"bare '{word}' must not be treated as a decline reply")

    def test_unrelated_message_is_none(self):
        parsed = main._parse_pending_jira_reply("what's the status of the deal", [_CREATE])
        self.assertIsNone(parsed["intent"])


class TestJiraIntentGuard(unittest.TestCase):
    def test_intent_shapes(self):
        self.assertTrue(main._check_pending_jira_intent("confirm jira"))
        self.assertTrue(main._check_pending_jira_intent("approve jira"))
        self.assertTrue(main._check_pending_jira_intent("no"))
        # bare yes is NOT jira-shaped (so it can't phantom-fire a jira write)
        self.assertFalse(main._check_pending_jira_intent("yes"))
        # a ticket key alone no longer makes a message jira-approval-shaped —
        # the literal word "jira" is required.
        self.assertFalse(main._check_pending_jira_intent("yes osd-1"))
        self.assertFalse(main._check_pending_jira_intent("tell me about the project"))

    def test_normal_messages_never_match_guard(self):
        # Regression (expert review): these dead-ended with "nothing pending to
        # approve" when approve tokens were loose and any ticket-key-shaped
        # word counted as Jira context.
        for msg in (
            "go check osd-123 for me",
            "send me the summary of rokt-45",
            "ok can you look at osd-123",
            "post it in osd-99 please",
            "go with option-2",
            "stop",
            "dont",
        ):
            self.assertFalse(main._check_pending_jira_intent(msg),
                             f"normal message '{msg}' must not look like a Jira approval reply")


def _ev(text, is_addon=False):
    return {
        "text": text,
        "user_id": "users/456",
        "space": "spaces/123",
        "is_addon": is_addon,
        "attachments": [],
    }


class TestGuardFlagGating(unittest.TestCase):
    """End-to-end handle_message: the Jira half of the phantom-approval guard is
    gated on config.JIRA_WRITE_ENABLED — flag off must be behaviorally identical
    to pre-feature (messages route to the agent, zero pending-Jira reads)."""

    def setUp(self):
        self._orig_get_pending_task = main._get_pending_task_request
        self._orig_get_conversation = main.get_conversation
        self._orig_conversation_scope = main.conversation_scope
        self._orig_get_pending_jira_actions = main.get_pending_jira_actions
        self._orig_flag = config_mock.JIRA_WRITE_ENABLED

        main._get_pending_task_request = MagicMock(return_value=(None, None))
        main.conversation_scope = MagicMock(return_value="space:spaces/123:users/456")
        main.get_conversation = MagicMock(return_value=[])
        main.add_turn = MagicMock()
        main.claim_message_once = MagicMock(return_value=True)
        main.release_message_claim = MagicMock()
        # Firestore read for pending Jira approvals — must NOT fire when the
        # flag is off (_get_pending_jira_request is left REAL to prove it).
        main.get_pending_jira_actions = MagicMock(return_value=None)
        self.bg = MagicMock()
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("ok, here's that.", []))

    def tearDown(self):
        main._get_pending_task_request = self._orig_get_pending_task
        main.get_conversation = self._orig_get_conversation
        main.conversation_scope = self._orig_conversation_scope
        main.get_pending_jira_actions = self._orig_get_pending_jira_actions
        config_mock.JIRA_WRITE_ENABLED = self._orig_flag

    def _agent_processed(self):
        return sys.modules["agent"].run_agent_loop.called

    def test_flag_off_normal_messages_route_to_agent(self):
        config_mock.JIRA_WRITE_ENABLED = False
        for msg in (
            "go check osd-123 for me",
            "send me the summary of rokt-45",
            "ok can you look at osd-123",
            "post it in osd-99 please",
            "go with option-2",
            "stop",
            "dont",
        ):
            sys.modules["agent"].run_agent_loop = MagicMock(return_value=("ok, here's that.", []))
            response = asyncio.run(main.handle_message(_ev(msg), self.bg))
            self.assertNotIn("nothing pending", response.get("text", ""),
                             f"'{msg}' was swallowed by the guard with the flag OFF")
            self.assertTrue(self._agent_processed(),
                            f"'{msg}' did not reach the agent with the flag OFF")

    def test_flag_off_even_jira_shaped_message_is_not_swallowed(self):
        # Flag off => the Jira guard branch must be skipped entirely, so even a
        # genuinely jira-shaped reply falls through to the agent (pre-WIP parity).
        config_mock.JIRA_WRITE_ENABLED = False
        response = asyncio.run(main.handle_message(_ev("confirm jira"), self.bg))
        self.assertNotIn("nothing pending", response.get("text", ""))
        self.assertTrue(self._agent_processed())

    def test_flag_off_costs_zero_pending_jira_reads(self):
        config_mock.JIRA_WRITE_ENABLED = False
        asyncio.run(main.handle_message(_ev("go check osd-123 for me"), self.bg))
        main.get_pending_jira_actions.assert_not_called()

    def test_flag_on_confirm_jira_still_guarded(self):
        config_mock.JIRA_WRITE_ENABLED = True
        response = asyncio.run(main.handle_message(_ev("confirm jira"), self.bg))
        self.assertIn("nothing pending", response.get("text", ""))
        self.assertFalse(self._agent_processed())


class TestApplierAllowList(unittest.TestCase):
    def test_non_jira_action_is_refused(self):
        import jira_service
        called = {"hit": False}

        def _track(*a, **k):
            called["hit"] = True
            return {"success": True}

        # patch.object on the BOUND module references (not string targets like
        # @patch("main....")): sibling test files pop+reimport "main" during
        # pytest collection, so a string target would resolve to a different
        # main module object than the one this file imported (co-run safety).
        with patch.object(main, "record_jira_write_audit", MagicMock()), \
             patch.object(main, "add_turn", MagicMock()), \
             patch.object(main, "send_chat_message") as mock_send, \
             patch.object(jira_service, "create_jira_ticket", _track), \
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
