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


# ── Helpers ────────────────────────────────────────────────────────────────

def _standard_card_clicked_body():
    return {
        "type": "CARD_CLICKED",
        "common": {
            "invokedFunction": "task_add",
            "parameters": [
                {"key": "batchId", "value": "b1"},
                {"key": "taskId", "value": "t1"},
            ],
            "formInputs": {},
        },
        "message": {"name": "spaces/s/messages/m"},
        "user": {"name": "users/123", "displayName": "Karim"},
        "space": {"name": "spaces/s"},
    }


# ── Standard Chat format ───────────────────────────────────────────────────

class TestParseEventStandardCardClicked(unittest.TestCase):
    """_parse_event surfaces new keys for standard Chat CARD_CLICKED events."""

    def _ev(self):
        return main._parse_event(_standard_card_clicked_body())

    def test_invoked_function(self):
        self.assertEqual(self._ev()["invoked_function"], "task_add")

    def test_parameters_list_converted_to_dict(self):
        self.assertEqual(self._ev()["parameters"], {"batchId": "b1", "taskId": "t1"})

    def test_message_name(self):
        self.assertEqual(self._ev()["message_name"], "spaces/s/messages/m")

    def test_form_inputs(self):
        self.assertEqual(self._ev()["form_inputs"], {})

    def test_dialog_event_type_absent(self):
        self.assertEqual(self._ev()["dialog_event_type"], "")

    def test_dialog_event_type_present(self):
        body = _standard_card_clicked_body()
        body["dialogEventType"] = "SUBMIT_DIALOG"
        ev = main._parse_event(body)
        self.assertEqual(ev["dialog_event_type"], "SUBMIT_DIALOG")

    def test_existing_keys_intact(self):
        ev = self._ev()
        self.assertEqual(ev["event_type"], "CARD_CLICKED")
        self.assertEqual(ev["user_id"], "users/123")
        self.assertEqual(ev["space"], "spaces/s")
        self.assertFalse(ev["is_addon"])


# ── Workspace Add-on format ────────────────────────────────────────────────

class TestParseEventAddonCardClicked(unittest.TestCase):
    """_parse_event surfaces new keys for Workspace Add-on CARD_CLICKED events."""

    def _addon_body(self, parameters=None):
        # Workspace add-on event shape (HANDOFF §4b): commonEventObject is
        # TOP-LEVEL (sibling of "chat") and carries parameters/invokedFunction;
        # chat.buttonClickedPayload carries the message.
        if parameters is None:
            parameters = {"batchId": "b2", "taskId": "t2"}
        return {
            "commonEventObject": {
                "invokedFunction": "task_dismiss",
                "parameters": parameters,
                "formInputs": {},
            },
            "chat": {
                "buttonClickedPayload": {
                    "message": {"name": "spaces/s/messages/m2"},
                },
                "user": {"name": "users/456", "displayName": "Karim"},
            },
        }

    def _ev(self):
        return main._parse_event(self._addon_body())

    def test_is_addon(self):
        self.assertTrue(self._ev()["is_addon"])

    def test_invoked_function(self):
        self.assertEqual(self._ev()["invoked_function"], "task_dismiss")

    def test_parameters_dict_preserved(self):
        self.assertEqual(self._ev()["parameters"], {"batchId": "b2", "taskId": "t2"})

    def test_message_name(self):
        self.assertEqual(self._ev()["message_name"], "spaces/s/messages/m2")

    def test_parameters_already_dict_no_conversion(self):
        """Add-on params arrive as dict — must be preserved as-is."""
        ev = main._parse_event(self._addon_body(parameters={"batchId": "b3"}))
        self.assertEqual(ev["parameters"], {"batchId": "b3"})

    def test_nested_commonEventObject_still_parsed(self):
        """Robustness: a nested chat.buttonClickedPayload.commonEventObject (older
        inferred shape) must ALSO parse, so we tolerate either location."""
        body = {
            "chat": {
                "buttonClickedPayload": {
                    "commonEventObject": {
                        "invokedFunction": "task_add",
                        "parameters": {"batchId": "b9", "taskId": "t1"},
                    },
                    "message": {"name": "spaces/s/messages/m9"},
                },
            }
        }
        ev = main._parse_event(body)
        self.assertEqual(ev["parameters"], {"batchId": "b9", "taskId": "t1"})
        self.assertEqual(ev["invoked_function"], "task_add")


# ── Regression: MESSAGE events unchanged ──────────────────────────────────

class TestParseEventMessageRegression(unittest.TestCase):
    """Normal MESSAGE events keep all existing keys; new keys default to empty."""

    def _ev(self):
        return main._parse_event({
            "type": "MESSAGE",
            "message": {"text": "hello momo", "argumentText": "hello momo"},
            "user": {"name": "users/789", "displayName": "Karim"},
            "space": {"name": "spaces/test"},
        })

    def test_event_type(self):
        self.assertEqual(self._ev()["event_type"], "MESSAGE")

    def test_text(self):
        self.assertEqual(self._ev()["text"], "hello momo")

    def test_user_id(self):
        self.assertEqual(self._ev()["user_id"], "users/789")

    def test_space(self):
        self.assertEqual(self._ev()["space"], "spaces/test")

    def test_is_addon_false(self):
        self.assertFalse(self._ev()["is_addon"])

    def test_invoked_function_default(self):
        self.assertEqual(self._ev()["invoked_function"], "")

    def test_parameters_default(self):
        self.assertEqual(self._ev()["parameters"], {})

    def test_form_inputs_default(self):
        self.assertEqual(self._ev()["form_inputs"], {})

    def test_message_name_default(self):
        self.assertEqual(self._ev()["message_name"], "")

    def test_dialog_event_type_default(self):
        self.assertEqual(self._ev()["dialog_event_type"], "")


if __name__ == "__main__":
    unittest.main()
