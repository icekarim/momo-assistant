import asyncio
import sys
import unittest
from unittest.mock import MagicMock, patch


# Mock external dependencies before importing the app modules.
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
)
sys.modules["agent"] = MagicMock()

import main
import agent


class TestTaskApprovalSafety(unittest.TestCase):
    @patch("main.send_chat_message")
    @patch("main.add_turn")
    @patch("main.store_pending_task_actions_if_empty")
    @patch("main.get_pending_task_actions")
    @patch("main.get_conversation")
    @patch("main.conversation_scope")
    def test_new_task_request_does_not_overwrite_existing_pending(
        self,
        mock_scope,
        mock_get_conversation,
        mock_get_pending,
        mock_store_pending_if_empty,
        mock_add_turn,
        mock_send_message,
    ):
        mock_scope.return_value = "space:spaces/123"
        mock_get_conversation.return_value = []
        mock_store_pending_if_empty.return_value = False
        mock_get_pending.return_value = {
            "actions": [
                {"action": "update", "find": "Update weekly report", "due": "2026-03-27"},
            ],
            "meeting_title": "",
            "approval_message": "",
        }
        agent.run_agent_loop.return_value = (
            "done. i've queued that update.",
            [{"action": "create", "title": "Follow up on project sync, credential rotation, and partner stats"}],
        )

        main._process_message_background("move the report task to friday", "users/456", "spaces/123")

        mock_store_pending_if_empty.assert_called_once()
        sent_text = mock_send_message.call_args.args[1]
        self.assertIn("didn't queue this new task change", sent_text)
        self.assertIn("update *Update weekly report* (set due 2026-03-27)", sent_text)
        self.assertNotIn("Follow up on QVC migration", sent_text)
        mock_add_turn.assert_any_call("space:spaces/123", "assistant", sent_text)

    @patch("main.store_pending_task_actions")
    @patch("main._get_pending_task_request")
    def test_confirm_reply_clears_pending_before_async_apply(
        self,
        mock_get_pending_request,
        mock_store_pending,
    ):
        state = {"cleared": False}
        thread_instance = MagicMock()
        thread_instance.start.side_effect = lambda: self.assertTrue(state["cleared"])

        mock_get_pending_request.return_value = (
            {
                "actions": [
                    {"action": "update", "find": "Update weekly report", "due": "2026-03-27"},
                ],
                "meeting_title": "",
                "approval_message": "queued",
            },
            "user:spaces/123:users/456",
        )

        def mark_cleared(*args, **kwargs):
            state["cleared"] = True

        with patch("main.clear_pending_task_actions", side_effect=mark_cleared) as mock_clear, patch(
            "main.threading.Thread", return_value=thread_instance
        ):
            response = asyncio.run(
                main.handle_message(
                    {
                        "text": "yes",
                        "user_id": "users/456",
                        "space": "spaces/123",
                        "is_addon": False,
                        "attachments": [],
                    }
                )
            )

        self.assertEqual(response, {})
        mock_store_pending.assert_not_called()
        mock_clear.assert_called_once_with(scope_id="user:spaces/123:users/456")
        thread_instance.start.assert_called_once()


if __name__ == "__main__":
    unittest.main()
