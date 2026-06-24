"""TDD tests for the debrief task-suggestion pipeline (reverted to TEXT).

Async REST cards have dead buttons in Google Chat (a service-account card
posted asynchronously never delivers CARD_CLICKED), so debrief [CREATE_TASK]
suggestions are rendered as a PLAIN-TEXT "Suggested follow-ups" list appended
to the cleaned debrief — NOT an interactive tray card.

_process_debrief_tasks must:
  (a) parse [CREATE_TASK] tags and dedup-before-show against open tasks,
  (b) append a plain-text "Suggested follow-ups" list (no card, no DSL),
  (c) NOT store a task batch and NOT send a card,
  (d) return cleaned text with NO [CREATE_TASK] tags and NO approval DSL.

Test isolation follows the pattern in test_post_meeting_debrief.py:8-39.
"""

import sys
import unittest
from unittest.mock import MagicMock

# ── module isolation ──────────────────────────────────────────────────────────
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
    "claude_client",
    "anthropic",
    "langsmith",
    "langsmith.wrappers",
    "langsmith_config",
]
_ORIGINAL_MODULES = {
    name: sys.modules.get(name, _SENTINEL)
    for name in _MODULES_TO_ISOLATE
}
for name in _MODULES_TO_ISOLATE:
    sys.modules.pop(name, None)

sys.modules["google"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()

from enum import Enum as _Enum
class _FakeTaskComplexity(_Enum):
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"

_claude_client_mock = MagicMock()
_claude_client_mock.TaskComplexity = _FakeTaskComplexity
_claude_client_mock.generate = MagicMock(return_value=MagicMock())
_claude_client_mock.extract_text = MagicMock(return_value="")
_claude_client_mock.extract_json = MagicMock(return_value={})
sys.modules["anthropic"] = MagicMock()
sys.modules["langsmith"] = MagicMock()
sys.modules["langsmith.wrappers"] = MagicMock()
sys.modules["langsmith_config"] = MagicMock()
sys.modules["claude_client"] = _claude_client_mock

# ── config mock ───────────────────────────────────────────────────────────────
_config_mock = MagicMock()
_config_mock.CHAT_SPACE_ID = "spaces/test"
_config_mock.GRANOLA_ENABLED = True
_config_mock.MEETING_DEBRIEF_LOOKBACK_MINUTES = 120
_config_mock.MEETING_DEBRIEF_GRACE_MINUTES = 45
_config_mock.MEETING_DEBRIEF_MIN_WAIT_MINUTES = 15
_config_mock.MEETING_DEBRIEF_MIN_NOTE_WORDS = 50
sys.modules["config"] = _config_mock


# ── REAL identity matcher ─────────────────────────────────────────────────────
# The prior simplified local matcher masked DEFECT A: it compared raw due
# prefixes, so a display-formatted open-task due ("Jun 17, 2026") was treated
# as not-equal to an ISO suggestion ("2026-06-17"). We now load a fresh, real
# copy of tasks_service (heavy deps stubbed) and wire its production
# _task_identity_match / _normalize_due so dedup tests actually cover the format.
import os
import importlib.util


def _load_real_tasks_service():
    for _name in ("googleapiclient", "googleapiclient.discovery",
                  "googleapiclient.errors", "google_auth"):
        sys.modules.setdefault(_name, MagicMock())
    spec = importlib.util.spec_from_file_location(
        "_real_tasks_service_dedup",
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tasks_service.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_REAL_TASKS = _load_real_tasks_service()
_real_identity_match = _REAL_TASKS._task_identity_match


# ── service mocks ─────────────────────────────────────────────────────────────
_GMAIL_MOCK = MagicMock()
_CALENDAR_MOCK = MagicMock()

_TASKS_MOCK = MagicMock()
_TASKS_MOCK._task_identity_match.side_effect = _real_identity_match
_TASKS_MOCK.fetch_open_tasks.return_value = []

_GEMINI_MOCK = MagicMock(
    generate_post_meeting_debrief=MagicMock(return_value="Debrief Content"),
)
_CHAT_MOCK = MagicMock(
    send_chat_message=MagicMock(),
    format_for_google_chat=lambda text: text,
)
_STORE_MOCK = MagicMock(
    store_task_batch=MagicMock(),
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

# Pop briefing so it picks up the mocks above (avoids leaked-mock shadowing
# when run alongside test_post_meeting_debrief.py).
sys.modules.pop("briefing", None)
import briefing  # noqa: E402


def tearDownModule():
    for name in _MODULES_TO_ISOLATE:
        original = _ORIGINAL_MODULES[name]
        if original is _SENTINEL:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = original


# ── helpers ───────────────────────────────────────────────────────────────────
_TASK_A_TITLE = "Send meeting notes"
_TASK_A_DUE = "2026-06-17"
_TASK_B_TITLE = "Follow up with client"
_TASK_B_DUE = "2026-06-24"

_TWO_TAG_DEBRIEF = (
    "Meeting summary.\n"
    f'[CREATE_TASK] title="{_TASK_A_TITLE}" due="{_TASK_A_DUE}"\n'
    f'[CREATE_TASK] title="{_TASK_B_TITLE}" due="{_TASK_B_DUE}"\n'
)


# ── tests ─────────────────────────────────────────────────────────────────────
class TestDebriefReturnsTextSuggestions(unittest.TestCase):

    def setUp(self):
        sys.modules["chat_service"] = _CHAT_MOCK
        sys.modules["conversation_store"] = _STORE_MOCK
        sys.modules["tasks_service"] = _TASKS_MOCK
        _TASKS_MOCK.reset_mock()
        _TASKS_MOCK._task_identity_match.side_effect = _real_identity_match
        _TASKS_MOCK.fetch_open_tasks.return_value = []
        _CHAT_MOCK.reset_mock()
        _STORE_MOCK.reset_mock()

    # ── (a) text suggestions appended, no card, no batch ──────────────────────
    def test_debrief_returns_text_suggestions(self):
        """[CREATE_TASK] tags render as a plain-text 'Suggested follow-ups' list."""
        _TASKS_MOCK.fetch_open_tasks.return_value = []

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Team Sync",
            space="spaces/test",
        )

        self.assertIn("Suggested follow-ups", result)
        self.assertIn(_TASK_A_TITLE, result)
        self.assertIn(_TASK_B_TITLE, result)
        # cleaned debrief body still present, raw tags gone
        self.assertIn("Meeting summary.", result)
        self.assertNotIn("[CREATE_TASK]", result)

        # NO card sent, NO batch stored — this is a text-only path now.
        _STORE_MOCK.store_task_batch.assert_not_called()
        _CHAT_MOCK.send_chat_message.assert_not_called()
        _STORE_MOCK.store_pending_task_actions.assert_not_called()

    # ── (b) dedup-before-show still drops already-open tasks ──────────────────
    def test_dedup_excludes_open_task(self):
        """One suggestion already open → only the other appears in the list."""
        _TASKS_MOCK.fetch_open_tasks.return_value = [
            {"title": _TASK_A_TITLE, "due": _TASK_A_DUE},
        ]

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Team Sync",
            space="spaces/test",
        )

        self.assertIn("Suggested follow-ups", result)
        self.assertNotIn(_TASK_A_TITLE, result)  # deduped out
        self.assertIn(_TASK_B_TITLE, result)
        _STORE_MOCK.store_task_batch.assert_not_called()

    # ── (b2) DEFECT A: display-formatted open-task due dedups ISO suggestion ───
    def test_dedup_matches_display_formatted_due(self):
        _TASKS_MOCK.fetch_open_tasks.return_value = [
            {"title": _TASK_A_TITLE, "due": "Jun 17, 2026"},
        ]

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Team Sync",
            space="spaces/test",
        )

        self.assertNotIn(_TASK_A_TITLE, result)
        self.assertIn(_TASK_B_TITLE, result)

    # ── (c) no DSL approval text in the result ────────────────────────────────
    def test_no_dsl_text(self):
        _TASKS_MOCK.fetch_open_tasks.return_value = []

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Team Sync",
            space="spaces/test",
        )

        for forbidden in ("Reply", "approve 2", "remove 1", "yes to create", "[CREATE_TASK]"):
            self.assertNotIn(
                forbidden, result,
                f"reverted debrief text must not contain {forbidden!r}",
            )

    # ── (d) all dupes → no suggestions section, just cleaned text ─────────────
    def test_all_dupes_no_suggestions_section(self):
        _TASKS_MOCK.fetch_open_tasks.return_value = [
            {"title": _TASK_A_TITLE, "due": _TASK_A_DUE},
            {"title": _TASK_B_TITLE, "due": _TASK_B_DUE},
        ]

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Team Sync",
            space="spaces/test",
        )

        self.assertNotIn("Suggested follow-ups", result)
        self.assertEqual(result, "Meeting summary.")
        _STORE_MOCK.store_task_batch.assert_not_called()
        _CHAT_MOCK.send_chat_message.assert_not_called()

    # ── morning-briefing call-site compat: accepts scope_id= kwarg ────────────
    def test_scope_id_kwarg_accepted(self):
        """run_morning_briefing calls _process_debrief_tasks with scope_id=,
        not space= — the signature must accept it without raising."""
        _TASKS_MOCK.fetch_open_tasks.return_value = []

        result = briefing._process_debrief_tasks(
            _TWO_TAG_DEBRIEF,
            meeting_title="Morning Briefing",
            scope_id="conv:spaces/test",
        )

        self.assertIn("Suggested follow-ups", result)
        self.assertIn(_TASK_A_TITLE, result)

    # ── no tags → text returned unchanged ─────────────────────────────────────
    def test_no_tags_returns_text_unchanged(self):
        result = briefing._process_debrief_tasks(
            "Just a plain debrief with no task tags.",
            meeting_title="Team Sync",
            space="spaces/test",
        )
        self.assertEqual(result, "Just a plain debrief with no task tags.")
        _CHAT_MOCK.send_chat_message.assert_not_called()
        _STORE_MOCK.store_task_batch.assert_not_called()


if __name__ == "__main__":
    unittest.main()
