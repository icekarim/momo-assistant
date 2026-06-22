"""TDD tests for conversational task CREATION via the SYNCHRONOUS add-on card.

momo is a Google Workspace ADD-ON Chat app. For a card button to call back, the
card MUST be returned SYNCHRONOUSLY as the HTTP response to the MESSAGE event and
each button's onClick.action.function MUST be the full /chat URL (action in
parameters.actionName). This file locks in that behavior:

  user "add a task to respond to FL"
    -> handle_message claims the message (idempotency), runs the agent loop
       in-request (asyncio.to_thread), sees a CREATE action, builds a task_batch +
       interactive tray card, and RETURNS it synchronously (createMessageAction for
       add-on / cardsV2 for standard Chat). No send_chat_message(cards=).
  user clicks Add  -> handled by handle_card_click (separate file).

Non-create actions (update/complete/delete) and mixed turns stay on the TEXT
"reply yes" approval flow. Plain replies return text. Voice stays on the async
background path.

Isolation mirrors test_handle_card_click.py:1-98 (identity-decorator fastapi stub
+ sys.modules.pop("main") before import main). The cards module is real (pure).
"""

import sys
import asyncio
import unittest
from unittest.mock import MagicMock


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
config_mock.MOMO_SERVICE_URL = "https://momo.example"
sys.modules["config"] = config_mock

CHAT_URL = "https://momo.example/chat"

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
    store_task_batch=MagicMock(),
    get_task_batch=MagicMock(),
    update_task_batch=MagicMock(),
    claim_message_once=MagicMock(),
    release_message_claim=MagicMock(),
)

_agent_mock = MagicMock()
sys.modules["agent"] = _agent_mock

sys.modules.pop("main", None)
import main  # noqa: E402
import cards  # noqa: E402  (real, pure module)


# ── helpers ────────────────────────────────────────────────────────────────

class _FakeBackgroundTasks:
    """Captures FastAPI BackgroundTasks.add_task calls for assertion."""

    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


def _ev(text, attachments=None, is_addon=False):
    return {
        "text": text,
        "user_id": "users/456",
        "space": "spaces/123",
        "is_addon": is_addon,
        "attachments": attachments or [],
        "message_name": "spaces/123/messages/m1",
    }


def _cards_sent(send_mock):
    """Return the cards= payload of the first send_chat_message(cards=...) call."""
    for call in send_mock.call_args_list:
        if "cards" in call.kwargs:
            return call.kwargs["cards"]
    return None


def _text_sent(send_mock):
    for call in send_mock.call_args_list:
        if len(call.args) >= 2:
            return call.args[1]
    return None


def _stored_batch_args():
    """Return (batch_id, source, space, rows) from the store_task_batch call."""
    return main.store_task_batch.call_args.args


class _SyncBase(unittest.TestCase):
    """Reset patched main symbols + the agent loop on every test so calls never
    leak between cases."""

    def setUp(self):
        main.conversation_scope = MagicMock(return_value="conv:spaces/123:users/456")
        main.get_conversation = MagicMock(return_value=[])
        main.add_turn = MagicMock()
        main.send_chat_message = MagicMock()
        main.format_for_google_chat = lambda t: t
        main.store_task_batch = MagicMock()
        main.update_task_batch = MagicMock()
        main.store_pending_task_actions_if_empty = MagicMock(return_value=True)
        main.get_pending_task_actions = MagicMock(return_value=None)
        main._get_pending_task_request = MagicMock(return_value=(None, None))
        main.claim_message_once = MagicMock(return_value=True)
        main.release_message_claim = MagicMock()
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("ok", []))

    def _sync(self, text, is_addon=False, bg=None, history=None):
        return main._process_message_sync(
            text, "users/456", "spaces/123", history or [],
            bg or _FakeBackgroundTasks(), is_addon,
        )

    def _send(self, ev, bg=None):
        return asyncio.run(main.handle_message(ev, bg or _FakeBackgroundTasks()))


# ── create -> SYNCHRONOUS tray card (no text approval) ───────────────────────

class TestConversationalCreateSyncCard(_SyncBase):

    def test_create_returns_sync_tray_card(self):
        create = [{"action": "create", "title": "Respond to FL", "due": "2026-06-18"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        result = self._sync("add a task to respond to FL")

        main.store_task_batch.assert_called_once()
        batch_id, source, space, rows = _stored_batch_args()
        self.assertTrue(batch_id)
        self.assertEqual(source, "New tasks")
        self.assertEqual(space, "spaces/123")
        self.assertEqual(rows[0]["title"], "Respond to FL")
        self.assertEqual(rows[0]["due"], "2026-06-18")
        self.assertEqual(rows[0]["state"], "pending")
        self.assertEqual(rows[0]["taskId"], "t1")

        # a card rides back synchronously (standard Chat shape for is_addon=False)
        self.assertIn("cardsV2", result)
        # NOT a text approval, NOT posted via send_chat_message(cards=)
        main.store_pending_task_actions_if_empty.assert_not_called()
        self.assertIsNone(_cards_sent(main.send_chat_message))

    def test_create_card_uses_full_chat_url_and_actionName(self):
        create = [{"action": "create", "title": "Respond to FL"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        result = self._sync("add a task")

        import json
        blob = json.dumps(result["cardsV2"])
        self.assertIn(CHAT_URL, blob)
        self.assertIn("task_add", blob)        # actionName value
        self.assertNotIn('"function": "task_add"', blob)  # never a bare function

    def test_create_standard_envelope_has_no_actionResponse(self):
        create = [{"action": "create", "title": "A"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        result = self._sync("add a task", is_addon=False)

        self.assertIn("cardsV2", result)
        self.assertNotIn("actionResponse", result)  # NEW message, not UPDATE
        self.assertEqual(result.get("text"), "on it.")

    def test_create_addon_envelope_is_createMessageAction(self):
        create = [{"action": "create", "title": "A"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        result = self._sync("add a task", is_addon=True)

        cda = result["hostAppDataAction"]["chatDataAction"]
        self.assertIn("createMessageAction", cda)
        self.assertIn("cardsV2", cda["createMessageAction"]["message"])

    def test_multi_create_single_batch(self):
        creates = [
            {"action": "create", "title": "Respond to FL"},
            {"action": "create", "title": "Send SDK+ links"},
        ]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("queued both.", creates))

        self._sync("add two tasks")

        main.store_task_batch.assert_called_once()
        rows = _stored_batch_args()[3]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["taskId"], "t1")
        self.assertEqual(rows[1]["taskId"], "t2")
        self.assertEqual(rows[1]["title"], "Send SDK+ links")

    def test_create_carries_notes_into_row(self):
        create = [{"action": "create", "title": "Respond to FL", "notes": "context about FL"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        self._sync("add a task")

        rows = _stored_batch_args()[3]
        self.assertEqual(rows[0]["notes"], "context about FL")


# ── mixed (create + non-create) -> TEXT approval (no card) ───────────────────

class TestMixedActionsTextApproval(_SyncBase):

    def test_mixed_create_and_update_routes_to_text(self):
        mixed = [
            {"action": "create", "title": "Respond to FL"},
            {"action": "update", "find": "weekly report", "due": "2026-03-27"},
        ]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("done.", mixed))

        result = self._sync("add one and push the report")

        main.store_task_batch.assert_not_called()
        main.store_pending_task_actions_if_empty.assert_called_once()
        self.assertNotIn("cardsV2", result)
        self.assertIn("text", result)


# ── non-create -> TEXT approval (unchanged flow) ─────────────────────────────

class TestNonCreateTextApproval(_SyncBase):

    def test_update_routes_through_text_approval(self):
        update = [{"action": "update", "find": "weekly report", "due": "2026-03-27"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("done.", update))

        result = self._sync("push weekly report to friday")

        main.store_task_batch.assert_not_called()
        main.store_pending_task_actions_if_empty.assert_called_once()
        self.assertNotIn("cardsV2", result)
        self.assertIn("weekly report", result.get("text", ""))


# ── plain reply -> TEXT (no batch, no pending) ───────────────────────────────

class TestPlainMessageSync(_SyncBase):

    def test_plain_reply_returns_text(self):
        sys.modules["agent"].run_agent_loop = MagicMock(
            return_value=("here's your calendar.", [])
        )

        result = self._sync("what's on my calendar?")

        main.store_task_batch.assert_not_called()
        main.store_pending_task_actions_if_empty.assert_not_called()
        self.assertEqual(result, {"text": "here's your calendar."})
        self.assertIsNone(_cards_sent(main.send_chat_message))


# ── handle_message routes TEXT to the synchronous path ───────────────────────

class TestHandleMessageSyncDispatch(_SyncBase):

    def test_text_create_via_handle_message_returns_card(self):
        create = [{"action": "create", "title": "Respond to FL"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        resp = self._send(_ev("add a task to respond to FL"))

        self.assertIn("cardsV2", resp)
        main.claim_message_once.assert_called_once_with("spaces/123/messages/m1")
        main.store_task_batch.assert_called_once()

    def test_text_addon_create_returns_createMessageAction(self):
        create = [{"action": "create", "title": "A"}]
        sys.modules["agent"].run_agent_loop = MagicMock(return_value=("on it.", create))

        resp = self._send(_ev("add a task", is_addon=True))

        cda = resp["hostAppDataAction"]["chatDataAction"]
        self.assertIn("createMessageAction", cda)


# ── voice message -> stays on the async background path ───────────────────────

class TestVoiceMessageStillAsync(_SyncBase):

    def test_voice_message_still_async(self):
        bg = _FakeBackgroundTasks()
        ev = _ev("", attachments=[{"contentType": "audio/mp3"}])
        resp = asyncio.run(main.handle_message(ev, bg))

        self.assertEqual(resp, {})
        self.assertEqual(len(bg.tasks), 1)
        self.assertIs(bg.tasks[0][0], main._process_message_background)
        sys.modules["agent"].run_agent_loop.assert_not_called()
        main.claim_message_once.assert_not_called()


if __name__ == "__main__":
    unittest.main()
