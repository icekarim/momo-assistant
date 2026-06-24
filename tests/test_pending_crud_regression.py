import sys
import asyncio
import unittest
from unittest.mock import MagicMock


# ── Stub heavy deps BEFORE importing main (co-run-safe) ──────────────────────
# Mirrors test_handle_card_click.py:1-97. The identity-decorator fastapi stub
# keeps the real async handle_message coroutine intact across co-runs, and the
# sys.modules.pop("main", None) below drops any sibling-cached bare-mock main.
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

sys.modules.pop("main", None)  # drop any sibling-cached bare-mock main
import main  # noqa: E402


class _FakeBackgroundTasks:
    """Captures FastAPI BackgroundTasks.add_task calls for assertion."""

    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


_USER_SCOPE = "user:spaces/123:users/456"
_EV_BASE = {"user_id": "users/456", "space": "spaces/123", "is_addon": False, "attachments": []}


def _ev(text):
    return {"text": text, **_EV_BASE}


class TestConversationalCrudRegression(unittest.TestCase):
    """End-to-end lock for conversational task-CRUD approval, which SHARES the
    pending-task DSL machinery with the now-removed debrief routing.

    Proves the full chain survives the cleanup:
      agent queues actions -> stored on USER scope -> user 'yes' ->
      _apply_pending_task_actions_background executes the Google Tasks mutation.
    """

    def setUp(self):
        main.get_conversation = MagicMock(return_value=[])
        main.add_turn = MagicMock()
        main.conversation_scope = MagicMock(return_value="conv:spaces/123:users/456")
        main.send_chat_message = MagicMock()
        main.format_for_google_chat = lambda t: t
        main.clear_pending_task_actions = MagicMock()
        main.store_pending_task_actions = MagicMock()
        main.create_task = MagicMock(
            return_value={"status": "created", "title": "Follow up with Sarah"}
        )
        main.update_task = MagicMock(
            return_value={"status": "updated", "title": "Follow up with Sarah"}
        )

    def test_crud_end_to_end_queue_approve_execute(self):
        # Creates now emit a tray card; the text-approval chain that this test
        # locks is exercised by update/complete/delete, so the end-to-end lock
        # uses an update action.
        actions = [{"action": "update", "find": "Follow up with Sarah", "due": "2026-06-19"}]

        # ── Stage 1: agent queues actions -> persisted on the USER scope ──────
        sys.modules["agent"].run_agent_loop = MagicMock(
            return_value=("queued it.", actions)
        )
        stored = {}

        def _store_if_empty(acts, scope_id="latest", meeting_title="", approval_message=""):
            stored["actions"] = acts
            stored["scope_id"] = scope_id
            return True

        main.store_pending_task_actions_if_empty = MagicMock(side_effect=_store_if_empty)

        main._process_message_background(
            "push the follow up task to friday", "users/456", "spaces/123"
        )

        self.assertEqual(stored["scope_id"], _USER_SCOPE,
                         "pending actions must be stored on the user scope")
        self.assertEqual(stored["actions"], actions)

        # ── Stage 2: user 'yes' -> routed to background apply on USER scope ───
        def _get(scope_id="latest"):
            if scope_id == _USER_SCOPE:
                return {"actions": actions, "meeting_title": "", "approval_message": "queued"}
            return None

        main.get_pending_task_actions = MagicMock(side_effect=_get)

        bg = _FakeBackgroundTasks()
        response = asyncio.run(main.handle_message(_ev("yes"), bg))

        self.assertEqual(response, {})
        self.assertEqual(len(bg.tasks), 1, "exactly one background apply task queued")
        func, args, _ = bg.tasks[0]
        self.assertIs(func, main._apply_pending_task_actions_background)
        self.assertEqual(args[0], actions, "selected actions == queued actions")
        self.assertEqual(args[4], _USER_SCOPE, "apply runs against the user scope")

        # ── Stage 3: run the queued fn -> Google Tasks mutation + reply ───────
        main.send_chat_message = MagicMock()  # reset (stage 1 also sent a message)
        func(*args)

        main.update_task.assert_called_once()
        self.assertEqual(main.update_task.call_args.kwargs["task_title"], "Follow up with Sarah")
        main.send_chat_message.assert_called_once()
        self.assertIn("Follow up with Sarah", main.send_chat_message.call_args.args[1])

    def test_multi_select_decline_still_supported(self):
        """Bulk conversational ops ('drop 1') keep working against user scope."""
        actions = [
            {"action": "update", "find": "weekly report", "due": "2026-06-19"},
            {"action": "create", "title": "Email the client"},
        ]
        main.get_pending_task_actions = MagicMock(
            side_effect=lambda scope_id="latest": (
                {"actions": actions, "meeting_title": "", "approval_message": "queued"}
                if scope_id == _USER_SCOPE else None
            )
        )
        bg = _FakeBackgroundTasks()
        response = asyncio.run(main.handle_message(_ev("drop 1"), bg))

        # declining item #1 leaves item #2 pending; no apply task queued
        self.assertEqual(len(bg.tasks), 0)
        self.assertIn("removed that task", response.get("text", ""))


class TestThinFallback(unittest.TestCase):
    """Bare approval/decline word with NO pending request and no preceding
    clarifying question returns a NEUTRAL message that does NOT reference the
    card surface (cards are a different path). It lives INSIDE the
    _check_pending_task_intent guard so it never shadows conversational CRUD
    or normal messages.
    """

    def setUp(self):
        main._get_pending_task_request = MagicMock(return_value=(None, None))
        main.add_turn = MagicMock()
        main.conversation_scope = MagicMock(return_value="conv:spaces/123:users/456")

    def test_bare_yes_no_pending_returns_neutral_message(self):
        bg = _FakeBackgroundTasks()
        response = asyncio.run(main.handle_message(_ev("yes"), bg))
        text = response.get("text", "")

        self.assertIn("nothing pending", text)
        # reverted: the conversational-CRUD path must not point at a task card
        self.assertNotIn("task card", text)
        self.assertNotIn("Add", text)
        self.assertNotIn("Dismiss", text)
        # neutral: must not falsely claim a card/approval already exists
        self.assertNotIn("to approve right now", text)
        # fallback short-circuits — no agent processing queued
        self.assertEqual(len(bg.tasks), 0)

    def test_non_approval_message_passes_through_to_agent(self):
        """A normal message still routes to the agent (fallback doesn't shadow).
        Text now runs SYNCHRONOUSLY in-request (_process_message_sync via
        asyncio.to_thread): a plain reply rides back as the HTTP response and the
        agent loop runs in-request rather than being queued to the background."""
        main.conversation_scope = MagicMock(return_value="conv:spaces/123:users/456")
        main.get_conversation = MagicMock(return_value=[])
        main.add_turn = MagicMock()
        main.claim_message_once = MagicMock(return_value=True)
        main.store_task_batch = MagicMock()
        main.store_pending_task_actions_if_empty = MagicMock(return_value=True)
        sys.modules["agent"].run_agent_loop = MagicMock(
            return_value=("here's your calendar.", [])
        )
        bg = _FakeBackgroundTasks()
        response = asyncio.run(main.handle_message(_ev("what's on my calendar today?"), bg))

        self.assertEqual(response, {"text": "here's your calendar."})
        sys.modules["agent"].run_agent_loop.assert_called_once()
        self.assertNotIn(
            main._process_message_background, [t[0] for t in bg.tasks]
        )


if __name__ == "__main__":
    unittest.main()
