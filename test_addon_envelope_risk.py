"""
test_addon_envelope_risk.py — Dual-shape envelope validation for the INFERRED
Workspace Add-on envelopes (UPDATE_MESSAGE and DIALOG).

STATUS: UNVERIFIED ADDON ENVELOPE — see PLAN_card_task_ux.md §10.
The add-on shapes (hostAppDataAction wrapper for UPDATE_MESSAGE and the dialog
response for is_addon=True) are inferred by mirroring _make_response's
createMessageAction pattern and have NOT been validated against a live Add-on
event.  These tests assert structural integrity only; a live-event smoke test
is required before relying on these paths in production.

References: PLAN_card_task_ux.md §6.6, §5.2, §10.
"""

import sys
import unittest
from unittest.mock import MagicMock


# ── Stub all heavy dependencies BEFORE importing main ──────────────────────
# Exact isolation pattern from test_handle_card_click.py lines 1-97.
# The cards module is intentionally NOT stubbed — it is pure and we exercise
# the real tray renderer.
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

# Route/middleware decorators must be identity so the real async chat_webhook
# survives import.  A bare MagicMock replaces it with a non-awaitable mock.
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
    get_task_batch=MagicMock(),
    update_task_batch=MagicMock(),
)
sys.modules["agent"] = MagicMock()

# Drop any cached main so the identity-decorator fastapi stub above takes effect
sys.modules.pop("main", None)
import main  # noqa: E402
import cards  # noqa: E402  (real, pure module)


# ── Helpers ────────────────────────────────────────────────────────────────

def _sample_cards():
    return [{"cardId": "test-card", "card": {"header": {"title": "Test"}}}]


def _row(task_id, title="Confirm OneTrust status", due="2026-06-17",
         owner="Karim", priority="high", state="pending"):
    return {
        "taskId": task_id,
        "title": title,
        "due": due,
        "owner": owner,
        "priority": priority,
        "state": state,
    }


def _batch(rows, source="Petco SDK+ Sync", space="spaces/s"):
    return {"source": source, "space": space, "rows": rows}


def _ev(invoked_function, batch_id="b1", task_id=None, form_inputs=None,
        is_addon=False):
    params = {"batchId": batch_id}
    if task_id is not None:
        params["taskId"] = task_id
    return {
        "event_type": "CARD_CLICKED",
        "invoked_function": invoked_function,
        "parameters": params,
        "form_inputs": form_inputs or {},
        "message_name": "spaces/s/messages/m",
        "is_addon": is_addon,
        "space": "spaces/s",
    }


class _Base(unittest.TestCase):
    """Resets the mocked store + create_task on every test so calls and
    in-place row mutations never leak between cases."""

    def setUp(self):
        self._batch_doc = None

        def _get(batch_id):
            return self._batch_doc

        main.get_task_batch = MagicMock(side_effect=_get)
        main.update_task_batch = MagicMock()
        main.create_task = MagicMock(return_value={"status": "created", "id": "x"})

    def _install_batch(self, rows, **kw):
        self._batch_doc = _batch(rows, **kw)
        return self._batch_doc


# ── Test 1 ─────────────────────────────────────────────────────────────────

class TestStandardUpdateMessageShape(unittest.TestCase):
    """_make_card_response(cards, is_addon=False) == standard UPDATE_MESSAGE envelope."""

    def test_standard_update_message_shape(self):
        cards_v2 = _sample_cards()
        result = main._make_card_response(cards_v2, is_addon=False)
        self.assertEqual(result, {
            "actionResponse": {"type": "UPDATE_MESSAGE"},
            "cardsV2": cards_v2,
        })


# ── Test 2 ─────────────────────────────────────────────────────────────────

class TestAddonUpdateMessageShape(unittest.TestCase):
    """_make_card_response(cards, is_addon=True) -> hostAppDataAction wrapper with
    cardsV2 nested at the correct path (UNVERIFIED ADDON ENVELOPE, PLAN §10)."""

    def test_addon_update_message_shape(self):
        cards_v2 = _sample_cards()
        result = main._make_card_response(cards_v2, is_addon=True)

        # top-level key must be hostAppDataAction (not actionResponse)
        self.assertIn("hostAppDataAction", result)
        self.assertNotIn("actionResponse", result)

        # cardsV2 must be at the full nested path
        actual_cards = (
            result["hostAppDataAction"]["chatDataAction"]
                  ["updateMessageAction"]["message"]["cardsV2"]
        )
        self.assertEqual(actual_cards, cards_v2)


# ── Test 3 ─────────────────────────────────────────────────────────────────

class TestRouterEmitsCorrectEnvelopePerIsAddon(_Base):
    """handle_card_click with task_add emits the correct envelope shape for each
    is_addon value (PLAN §6.6)."""

    def test_router_emits_correct_envelope_per_is_addon(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links")]
        self._install_batch(rows)

        # ── standard Chat (is_addon=False) ─────────────────────────────────
        resp_std = main.handle_card_click(_ev("task_add", task_id="t1", is_addon=False))
        self.assertIn("actionResponse", resp_std)
        self.assertEqual(resp_std["actionResponse"], {"type": "UPDATE_MESSAGE"})
        self.assertNotIn("hostAppDataAction", resp_std)

        # reset rows for the add-on pass
        for r in rows:
            r["state"] = "pending"

        # ── Workspace Add-on (is_addon=True) ──────────────────────────────
        # UNVERIFIED ADDON ENVELOPE (PLAN §10): validate against a live Add-on event
        resp_addon = main.handle_card_click(_ev("task_add", task_id="t1", is_addon=True))
        self.assertIn("hostAppDataAction", resp_addon)
        self.assertNotIn("actionResponse", resp_addon)

        # cardsV2 at the nested path must equal the re-rendered tray
        inner_cards = (
            resp_addon["hostAppDataAction"]["chatDataAction"]
                      ["updateMessageAction"]["message"]["cardsV2"]
        )
        expected_tray = cards.build_task_tray_card(
            "b1", "Petco SDK+ Sync", rows, chat_url=CHAT_URL
        )
        self.assertEqual(inner_cards, expected_tray)


# ── Test 4 ─────────────────────────────────────────────────────────────────

class TestDialogEnvelopeShapeBothFormats(_Base):
    """task_edit dialog: standard shape confirmed; add-on path is guard-safe
    (UNVERIFIED ADDON ENVELOPE, PLAN §10)."""

    def test_dialog_envelope_shape_both_formats(self):
        # ── standard Chat: full structural assertion ───────────────────────
        rows = [_row("t1")]
        self._install_batch(rows)

        resp_std = main.handle_card_click(_ev("task_edit", task_id="t1", is_addon=False))

        self.assertEqual(resp_std["actionResponse"]["type"], "DIALOG")
        widgets = (
            resp_std["actionResponse"]["dialogAction"]["dialog"]["body"]
                    ["sections"][0]["widgets"]
        )
        pickers = [w["dateTimePicker"] for w in widgets if "dateTimePicker" in w]
        self.assertEqual(len(pickers), 1)
        self.assertEqual(pickers[0]["name"], "due")
        # a create must never happen on edit
        main.create_task.assert_not_called()

        # ── add-on: guard + is_addon kwarg acceptance (PLAN §10) ──────────
        # _make_edit_dialog must accept is_addon=True without raising
        # (this is the guard requirement: TypeError before the fix → passes after).
        row = _row("t1")
        resp_addon_direct = main._make_edit_dialog("b1", row, is_addon=True)
        self.assertIsInstance(resp_addon_direct, dict)
        self.assertTrue(len(resp_addon_direct) > 0)

        # handle_card_click must also not raise for is_addon=True on task_edit
        rows2 = [_row("t1")]
        self._install_batch(rows2)
        resp_addon = main.handle_card_click(_ev("task_edit", task_id="t1", is_addon=True))
        self.assertIsInstance(resp_addon, dict)
        self.assertTrue(len(resp_addon) > 0)


if __name__ == "__main__":
    unittest.main()
