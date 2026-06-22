import sys
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

import main  # noqa: E402


# ── Fixture ────────────────────────────────────────────────────────────────

CARDS = [{"cardId": "tray-b1", "card": {"header": {"title": "Test Tray"}}}]


# ── Standard Chat (is_addon=False) ─────────────────────────────────────────

class TestMakeCardResponseStandard(unittest.TestCase):
    """_make_card_response for standard Chat (is_addon=False)."""

    def test_update_true_action_response_type(self):
        resp = main._make_card_response(CARDS, is_addon=False, update=True)
        self.assertEqual(resp["actionResponse"]["type"], "UPDATE_MESSAGE")

    def test_update_true_cardsV2(self):
        resp = main._make_card_response(CARDS, is_addon=False, update=True)
        self.assertEqual(resp["cardsV2"], CARDS)

    def test_update_false_no_action_response(self):
        resp = main._make_card_response(CARDS, is_addon=False, update=False)
        self.assertNotIn("actionResponse", resp)

    def test_update_false_cardsV2(self):
        resp = main._make_card_response(CARDS, is_addon=False, update=False)
        self.assertEqual(resp["cardsV2"], CARDS)

    def test_update_defaults_to_true(self):
        """update parameter defaults to True."""
        resp = main._make_card_response(CARDS, is_addon=False)
        self.assertIn("actionResponse", resp)
        self.assertEqual(resp["actionResponse"]["type"], "UPDATE_MESSAGE")


# ── Workspace Add-on (is_addon=True) ──────────────────────────────────────

class TestMakeCardResponseAddon(unittest.TestCase):
    """_make_card_response for Workspace Add-on (is_addon=True)."""

    def test_addon_update_true_uses_updateMessageAction(self):
        resp = main._make_card_response(CARDS, is_addon=True, update=True)
        action = resp["hostAppDataAction"]["chatDataAction"]
        self.assertIn("updateMessageAction", action)

    def test_addon_update_true_cardsV2_nested(self):
        resp = main._make_card_response(CARDS, is_addon=True, update=True)
        action = resp["hostAppDataAction"]["chatDataAction"]
        self.assertEqual(action["updateMessageAction"]["message"]["cardsV2"], CARDS)

    def test_addon_update_false_uses_createMessageAction(self):
        resp = main._make_card_response(CARDS, is_addon=True, update=False)
        action = resp["hostAppDataAction"]["chatDataAction"]
        self.assertIn("createMessageAction", action)

    def test_addon_update_false_cardsV2_nested(self):
        resp = main._make_card_response(CARDS, is_addon=True, update=False)
        action = resp["hostAppDataAction"]["chatDataAction"]
        self.assertEqual(action["createMessageAction"]["message"]["cardsV2"], CARDS)

    def test_addon_update_true_no_top_level_cardsV2(self):
        """Add-on envelopes nest cardsV2; it must not appear at top level."""
        resp = main._make_card_response(CARDS, is_addon=True, update=True)
        self.assertNotIn("cardsV2", resp)

    def test_addon_update_false_no_top_level_cardsV2(self):
        resp = main._make_card_response(CARDS, is_addon=True, update=False)
        self.assertNotIn("cardsV2", resp)

    def test_addon_update_true_no_action_response_key(self):
        """Add-on format must not include 'actionResponse' key."""
        resp = main._make_card_response(CARDS, is_addon=True, update=True)
        self.assertNotIn("actionResponse", resp)


if __name__ == "__main__":
    unittest.main()
