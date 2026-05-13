import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock


_SENTINEL = object()
_MODULES_TO_ISOLATE = [
    "briefing",
    "config",
    "gmail_service",
    "calendar_service",
    "tasks_service",
    "gemini_service",
    "chat_service",
    "conversation_store",
    "knowledge_graph",
    "granola_service",
    "google",
    "google.generativeai",
]
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _SENTINEL)
    for name in _MODULES_TO_ISOLATE
}
for name in _MODULES_TO_ISOLATE:
    sys.modules.pop(name, None)

sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

config_mock = MagicMock()
config_mock.GRANOLA_ENABLED = True
config_mock.CHAT_SPACE_ID = "spaces/test"
config_mock.MEETING_DEBRIEF_LOOKBACK_MINUTES = 120
config_mock.MEETING_DEBRIEF_GRACE_MINUTES = 45
config_mock.MEETING_DEBRIEF_MIN_WAIT_MINUTES = 15
config_mock.MEETING_DEBRIEF_MIN_NOTE_WORDS = 50
sys.modules["config"] = config_mock

_GMAIL_MOCK = MagicMock()
_CALENDAR_MOCK = MagicMock()
_TASKS_MOCK = MagicMock()
_GEMINI_MOCK = MagicMock(
    generate_morning_briefing=MagicMock(),
    generate_post_meeting_debrief=MagicMock(return_value="Debrief Content"),
)
_CHAT_MOCK = MagicMock(
    format_for_google_chat=lambda text: text,
    send_chat_message=MagicMock(),
)
_STORE_MOCK = MagicMock(
    add_turn=MagicMock(),
    conversation_scope=lambda space="": f"space:{space}" if space else "latest",
    has_email_alert_been_sent=MagicMock(),
    mark_email_alert_sent=MagicMock(),
    has_debrief_been_sent=MagicMock(return_value=False),
    mark_debrief_sent=MagicMock(),
)
_KG_MOCK = MagicMock(
    extract_and_store_background=MagicMock(),
    extract_and_store_via_bg_tasks=MagicMock(),
)
_GRANOLA_MOCK = MagicMock()

sys.modules["gmail_service"] = _GMAIL_MOCK
sys.modules["calendar_service"] = _CALENDAR_MOCK
sys.modules["tasks_service"] = _TASKS_MOCK
sys.modules["gemini_service"] = _GEMINI_MOCK
sys.modules["chat_service"] = _CHAT_MOCK
sys.modules["conversation_store"] = _STORE_MOCK
sys.modules["knowledge_graph"] = _KG_MOCK
sys.modules["granola_service"] = _GRANOLA_MOCK

import briefing


def tearDownModule():
    for name in _MODULES_TO_ISOLATE:
        original = _ORIGINAL_MODULES[name]
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


class TestPostMeetingDebriefGraceWindow(unittest.TestCase):
    def setUp(self):
        sys.modules["granola_service"] = _GRANOLA_MOCK
        self.granola = _GRANOLA_MOCK
        self.gemini = _GEMINI_MOCK
        self.chat = _CHAT_MOCK
        self.store = _STORE_MOCK

        self.granola.reset_mock()
        self.gemini.generate_post_meeting_debrief.reset_mock(return_value=True)
        self.gemini.generate_post_meeting_debrief.return_value = "Debrief Content"
        self.chat.send_chat_message.reset_mock()
        self.store.has_debrief_been_sent.reset_mock(return_value=True)
        self.store.has_debrief_been_sent.return_value = False
        self.store.mark_debrief_sent.reset_mock()

    def _meeting(self, minutes_since_end):
        ended_at = datetime.now().astimezone() - timedelta(minutes=minutes_since_end)
        return {
            "id": "calendar-1",
            "title": "Partner sync",
            "end_iso": ended_at.isoformat(),
            "end_time": ended_at.strftime("%I:%M %p").lstrip("0"),
            "attendees": [{"name": "Ari"}],
        }

    def test_no_granola_match_sends_without_notes_after_grace_window(self):
        briefing.fetch_recently_ended_meetings.return_value = [self._meeting(60)]
        self.granola.build_meeting_id_map.return_value = {}

        result = briefing.run_post_meeting_debrief()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["debriefs_sent"], 1)
        self.gemini.generate_post_meeting_debrief.assert_called_once_with(
            "Partner sync",
            ["Ari"],
            "",
            unittest.mock.ANY,
        )
        self.chat.send_chat_message.assert_called_once_with("spaces/test", "Debrief Content")
        self.store.mark_debrief_sent.assert_called_once_with("calendar-1", "Partner sync")

    def test_matched_meeting_with_no_notes_sends_without_notes_after_grace_window(self):
        briefing.fetch_recently_ended_meetings.return_value = [self._meeting(60)]
        self.granola.build_meeting_id_map.return_value = {"partner sync": "granola-1"}
        self.granola.match_meeting_id.return_value = "granola-1"
        self.granola.fetch_meeting_notes_batch.return_value = {}

        result = briefing.run_post_meeting_debrief()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["debriefs_sent"], 1)
        self.gemini.generate_post_meeting_debrief.assert_called_once_with(
            "Partner sync",
            ["Ari"],
            "",
            unittest.mock.ANY,
        )
        self.chat.send_chat_message.assert_called_once_with("spaces/test", "Debrief Content")
        self.store.mark_debrief_sent.assert_called_once_with("calendar-1", "Partner sync")

    def test_thin_notes_send_without_notes_after_grace_window(self):
        briefing.fetch_recently_ended_meetings.return_value = [self._meeting(60)]
        self.granola.build_meeting_id_map.return_value = {"partner sync": "granola-1"}
        self.granola.match_meeting_id.return_value = "granola-1"
        self.granola.fetch_meeting_notes_batch.return_value = {
            "granola-1": "one two three four five six"
        }

        result = briefing.run_post_meeting_debrief()

        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["debriefs_sent"], 1)
        self.gemini.generate_post_meeting_debrief.assert_called_once_with(
            "Partner sync",
            ["Ari"],
            "",
            unittest.mock.ANY,
        )
        self.chat.send_chat_message.assert_called_once_with("spaces/test", "Debrief Content")
        self.store.mark_debrief_sent.assert_called_once_with("calendar-1", "Partner sync")


if __name__ == "__main__":
    unittest.main()
