"""TDD tests for the inbound-message idempotency guard on main.handle_message's
SYNCHRONOUS text path.

momo is a Google Workspace ADD-ON Chat app. To make create-card buttons call
back, a conversational create returns an interactive tray card SYNCHRONOUSLY as
the HTTP response to the MESSAGE event — which means the agent loop runs
in-request. Google Chat retries the webhook when a sync handler exceeds its 30s
deadline, so handle_message claims ev["message_name"] (claim_message_once) before
processing: a retry that finds a live claim no-ops, so create_task /
store_task_batch never run twice. The claim is kept on success (a late >30s
response may be lost but a task is never double-created) and released ONLY on
exception so a transient failure can be retried.

Contract asserted here:
  - text message -> claim_message_once(message_name) called, agent runs in-request,
    a create returns a card synchronously
  - duplicate (claim returns False) -> empty {} no-op, no agent, no batch
  - empty message_name -> claim skipped, still processed synchronously
  - exception in the sync path -> release_message_claim called, error text returned
  - voice/audio -> stays on the async background path, NO claim

Isolation mirrors test_handle_card_click.py:1-98 (identity-decorator fastapi stub
+ sys.modules.pop("main", None) before import main). cards stays real (pure).
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
sys.modules["agent"] = MagicMock()

sys.modules.pop("main", None)
import main  # noqa: E402
import cards  # noqa: E402  (real, pure module)


class _FakeBackgroundTasks:
    def __init__(self):
        self.tasks = []

    def add_task(self, func, *args, **kwargs):
        self.tasks.append((func, args, kwargs))


def _ev(text, message_name="", attachments=None, is_addon=False):
    return {
        "text": text,
        "user_id": "users/456",
        "space": "spaces/123",
        "is_addon": is_addon,
        "attachments": attachments or [],
        "message_name": message_name,
    }


class _SyncClaimBase(unittest.TestCase):

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
        sys.modules["agent"].run_agent_loop = MagicMock(
            return_value=("on it.", [{"action": "create", "title": "Respond to FL"}])
        )

    def _send(self, ev, bg=None):
        return asyncio.run(main.handle_message(ev, bg or _FakeBackgroundTasks()))


class TestSyncClaimGuard(_SyncClaimBase):

    def test_text_message_claims_and_runs_sync(self):
        resp = self._send(_ev("add a task to respond to FL",
                              message_name="spaces/s/messages/m1"))

        main.claim_message_once.assert_called_once_with("spaces/s/messages/m1")
        sys.modules["agent"].run_agent_loop.assert_called_once()
        # a create returns a card synchronously (NOT an empty ack)
        self.assertIn("cardsV2", resp)
        main.store_task_batch.assert_called_once()

    def test_duplicate_claim_returns_empty_noop(self):
        main.claim_message_once = MagicMock(return_value=False)

        resp = self._send(_ev("add a task to respond to FL",
                              message_name="spaces/s/messages/m1"))

        self.assertEqual(resp, {})
        sys.modules["agent"].run_agent_loop.assert_not_called()
        main.store_task_batch.assert_not_called()

    def test_empty_message_name_skips_claim_and_runs(self):
        resp = self._send(_ev("add a task to respond to FL", message_name=""))

        main.claim_message_once.assert_not_called()
        sys.modules["agent"].run_agent_loop.assert_called_once()
        self.assertIn("cardsV2", resp)

    def test_exception_releases_claim(self):
        sys.modules["agent"].run_agent_loop = MagicMock(
            side_effect=RuntimeError("agent boom")
        )

        resp = self._send(_ev("add a task", message_name="spaces/s/messages/m1"))

        main.release_message_claim.assert_called_once_with("spaces/s/messages/m1")
        self.assertIn("text", resp)
        self.assertIn("went wrong", resp["text"])

    def test_exception_without_message_name_does_not_release(self):
        sys.modules["agent"].run_agent_loop = MagicMock(
            side_effect=RuntimeError("agent boom")
        )

        resp = self._send(_ev("add a task", message_name=""))

        main.release_message_claim.assert_not_called()
        self.assertIn("text", resp)

    def test_voice_still_async_no_claim(self):
        bg = _FakeBackgroundTasks()
        ev = _ev("", message_name="spaces/s/messages/v1",
                 attachments=[{"contentType": "audio/mp3"}])
        resp = asyncio.run(main.handle_message(ev, bg))

        self.assertEqual(resp, {})
        self.assertEqual(len(bg.tasks), 1)
        self.assertIs(bg.tasks[0][0], main._process_message_background)
        main.claim_message_once.assert_not_called()
        sys.modules["agent"].run_agent_loop.assert_not_called()


if __name__ == "__main__":
    unittest.main()
