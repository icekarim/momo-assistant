import sys
import asyncio
import unittest
from unittest.mock import MagicMock


# ── Stub all heavy dependencies BEFORE importing main ──────────────────────
# (mirrors test_handle_card_click.py isolation so this file is co-run-safe and
# order-independent: identity-decorator fastapi stub + sys.modules.pop("main")
# before import main).
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


def traceable_mock(*args, **kwargs):
    def decorator(func):
        return func
    return decorator


langsmith_mock.traceable = traceable_mock
langsmith_mock.traced_chat_send = MagicMock()
langsmith_mock.traced_generate_content = MagicMock()
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
    clear_conversation=MagicMock(),
    conversation_scope=MagicMock(),
    get_pending_task_actions=MagicMock(),
    clear_pending_task_actions=MagicMock(),
    store_pending_task_actions=MagicMock(),
    store_pending_task_actions_if_empty=MagicMock(),
    get_task_batch=MagicMock(),
    update_task_batch=MagicMock(),
)
sys.modules["agent"] = MagicMock()

sys.modules.pop("main", None)  # drop any sibling-cached bare-mock main so the identity-decorator fastapi stub above yields a real coroutine handle_message
import main  # noqa: E402


def _ev(text, is_addon=False):
    return {
        "text": text,
        "user_id": "users/456",
        "space": "spaces/123",
        "is_addon": is_addon,
        "attachments": [],
    }


class _BareYesNoBase(unittest.TestCase):
    """Each test mocks the no-pending path: _get_pending_task_request returns
    no pending, conversation_scope returns a stub id, and get_conversation is
    set per-test to control the last-assistant-turn heuristic."""

    def setUp(self):
        self._orig_get_pending = main._get_pending_task_request
        self._orig_get_conversation = main.get_conversation
        self._orig_conversation_scope = main.conversation_scope

        main._get_pending_task_request = MagicMock(return_value=(None, None))
        main.conversation_scope = MagicMock(return_value="space:spaces/123:users/456")
        main.get_conversation = MagicMock(return_value=[])
        main.add_turn = MagicMock()
        self.bg = MagicMock()
        main.claim_message_once = MagicMock(return_value=True)
        main.release_message_claim = MagicMock()
        main.store_task_batch = MagicMock()
        main.store_pending_task_actions_if_empty = MagicMock(return_value=True)
        # Fall-through text now runs the agent synchronously in-request; the
        # yes-dead-end behavior is unchanged, only the mechanism is synchronous.
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("ok, here's that.", []))

    def tearDown(self):
        main._get_pending_task_request = self._orig_get_pending
        main.get_conversation = self._orig_get_conversation
        main.conversation_scope = self._orig_conversation_scope

    def _agent_processed(self):
        """Fall-through routes to the synchronous agent path: the signal is that
        run_agent_loop ran in-request (vs the phantom-approval guard, which
        short-circuits before any agent call)."""
        return sys.modules["agent"].run_agent_loop.called


class TestBareYesDeadEnd(_BareYesNoBase):

    def test_bare_yes_after_clarifying_question_routes_to_agent(self):
        # Last assistant turn is a clarifying question; a bare "yes" answers it.
        main.get_conversation = MagicMock(return_value=[
            {"role": "user", "content": "add a new task to respond to fl"},
            {"role": "assistant",
             "content": "is this something different from the completed one?"},
        ])

        response = asyncio.run(main.handle_message(_ev("yes"), self.bg))

        # Did NOT swallow with the canned guard message ...
        self.assertNotIn("nothing pending", response.get("text", ""))
        # ... and fell through to the normal agent path.
        self.assertTrue(self._agent_processed())

    def test_bare_yes_no_question_context_still_guarded(self):
        # No trailing-"?" assistant turn -> genuinely context-free bare "yes".
        main.get_conversation = MagicMock(return_value=[
            {"role": "assistant", "content": "here are your open tasks."},
        ])

        response = asyncio.run(main.handle_message(_ev("yes"), self.bg))

        text = response.get("text", "")
        self.assertIn("nothing pending", text)
        # Reverted: must NOT mention the card surface.
        self.assertNotIn("task card", text)
        self.assertNotIn("Add", text)
        self.assertNotIn("Dismiss", text)
        # Phantom-loop protection: agent path must NOT run.
        self.assertFalse(self._agent_processed())

    def test_bare_yes_empty_history_still_guarded(self):
        # Empty history must not crash and must keep the guard.
        main.get_conversation = MagicMock(return_value=[])

        response = asyncio.run(main.handle_message(_ev("yes"), self.bg))

        self.assertIn("nothing pending", response.get("text", ""))
        self.assertFalse(self._agent_processed())

    def test_bare_no_after_question_routes_to_agent(self):
        # Symmetry: "no" answering a question also falls through to the agent.
        main.get_conversation = MagicMock(return_value=[
            {"role": "assistant",
             "content": "do you want me to keep the existing one?"},
        ])

        response = asyncio.run(main.handle_message(_ev("no"), self.bg))

        self.assertNotIn("nothing pending", response.get("text", ""))
        self.assertTrue(self._agent_processed())


if __name__ == "__main__":
    unittest.main()
