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
        self.assertNotIn("Follow up on project sync", sent_text)
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


class TestParsePendingTaskReply(unittest.TestCase):
    """Unit tests for the pending-task reply parser, covering semantic
    inversion (''no just X''), decline action verbs, keep-only patterns,
    and compound-word fuzzy matching."""

    ACTIONS = [
        {"action": "create", "title": "Send Foot Locker SDK secret and key"},
        {"action": "create", "title": "Ask Maggie about BJ's support"},
    ]

    # --- Exact approve / decline ---

    def test_yes_approves_all(self):
        result = main._parse_pending_task_reply("yes", self.ACTIONS)
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {0, 1})
        self.assertFalse(result["dismiss_rest"])

    def test_no_declines_all(self):
        result = main._parse_pending_task_reply("no", self.ACTIONS)
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0, 1})

    # --- Semantic inversion: "no just X" keeps only X ---

    def test_no_just_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "no just the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    def test_no_only_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "no only the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    def test_no_keep_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "no, keep the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    # --- Decline action verbs ---

    def test_dont_need_the_footlocker_task(self):
        result = main._parse_pending_task_reply(
            "dont need the footlocker task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    def test_drop_the_footlocker_task(self):
        result = main._parse_pending_task_reply(
            "drop the foot locker task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    def test_remove_foot_locker(self):
        result = main._parse_pending_task_reply(
            "remove the foot locker one", self.ACTIONS
        )
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    def test_get_rid_of_footlocker(self):
        result = main._parse_pending_task_reply(
            "get rid of the footlocker task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    # --- Keep-only standalone patterns ---

    def test_just_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "just the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    def test_keep_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "keep the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    def test_only_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "only the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    def test_i_just_want_the_maggie_task(self):
        result = main._parse_pending_task_reply(
            "i just want the maggie task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertTrue(result["dismiss_rest"])

    # --- Compound word matching ---

    def test_footlocker_matches_foot_locker(self):
        result = main._parse_pending_task_reply(
            "cancel the footlocker task", self.ACTIONS
        )
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    # --- Numbered partial approve/decline ---

    def test_approve_2(self):
        result = main._parse_pending_task_reply("approve 2", self.ACTIONS)
        self.assertEqual(result["intent"], "confirm")
        self.assertEqual(result["selected_indices"], {1})
        self.assertFalse(result["dismiss_rest"])

    def test_cancel_1(self):
        result = main._parse_pending_task_reply("cancel 1", self.ACTIONS)
        self.assertEqual(result["intent"], "decline")
        self.assertEqual(result["selected_indices"], {0})

    # --- Unrelated messages pass through ---

    def test_unrelated_message_returns_no_intent(self):
        result = main._parse_pending_task_reply(
            "whats the weather today", self.ACTIONS
        )
        self.assertIsNone(result["intent"])


class TestStripLlmApprovalBlock(unittest.TestCase):
    """Ensure _strip_llm_approval_block removes LLM-generated duplicates."""

    def test_strips_single_item_block(self):
        text = (
            "queued the task.\n\n"
            "📝 Approve this Google Tasks change\n"
            "  1. create Ask BJ's about paymenttype fix\n"
            "\n"
            "Reply yes to approve, or no to cancel"
        )
        result = main._strip_llm_approval_block(text)
        self.assertNotIn("Approve", result)
        self.assertNotIn("Reply yes", result)
        self.assertIn("queued the task", result)

    def test_strips_multi_item_block(self):
        text = (
            "got it, here are the tasks.\n\n"
            "📝 Approve these Google Tasks changes\n"
            "  1. create Task A\n"
            "  2. create Task B\n"
            "\n"
            "Reply yes to apply all 2 changes in Google Tasks, or no to cancel"
        )
        result = main._strip_llm_approval_block(text)
        self.assertNotIn("Approve", result)
        self.assertIn("got it", result)

    def test_leaves_clean_text_alone(self):
        text = "queued the task to ask about paymenttype fix."
        result = main._strip_llm_approval_block(text)
        self.assertEqual(result, text)


class TestPhantomApprovalGuard(unittest.TestCase):
    """Ensure bare 'yes'/'no' with no pending tasks doesn't fall through to agent."""

    @patch("main._get_pending_task_request")
    def test_bare_yes_with_no_pending_returns_nothing_pending(
        self, mock_get_pending
    ):
        mock_get_pending.return_value = (None, None)
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
        self.assertIn("nothing pending", response.get("text", ""))

    @patch("main._get_pending_task_request")
    def test_bare_no_with_no_pending_returns_nothing_pending(
        self, mock_get_pending
    ):
        mock_get_pending.return_value = (None, None)
        response = asyncio.run(
            main.handle_message(
                {
                    "text": "no",
                    "user_id": "users/456",
                    "space": "spaces/123",
                    "is_addon": False,
                    "attachments": [],
                }
            )
        )
        self.assertIn("nothing pending", response.get("text", ""))


if __name__ == "__main__":
    unittest.main()
