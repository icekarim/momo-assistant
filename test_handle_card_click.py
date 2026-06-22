import sys
import json
import asyncio
import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock


# ── Stub all heavy dependencies BEFORE importing main ──────────────────────
# (mirrors test_card_click_parsing.py). The cards module is intentionally NOT
# stubbed — it is pure and we exercise the real tray renderer.
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
# survives import (round-trip drives it). A bare MagicMock replaces it with a
# non-awaitable mock.
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

CHAT_URL = "https://momo.example/chat"  # config.MOMO_SERVICE_URL.rstrip("/") + "/chat"

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

sys.modules.pop("main", None)  # drop any sibling-cached bare-mock main so the identity-decorator fastapi stub above yields a real coroutine chat_webhook
import main  # noqa: E402
import cards  # noqa: E402  (real, pure module)


# ── Helpers ────────────────────────────────────────────────────────────────

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
        "space": space if (space := "spaces/s") else "",
    }


def _ms_for(date_str):
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _row_by_id(rows, task_id):
    return next(r for r in rows if r["taskId"] == task_id)


class _CardClickBase(unittest.TestCase):
    """Resets the mocked store + create_task on every test so calls and
    in-place row mutations never leak between cases."""

    def setUp(self):
        self.rows = None  # set per-test

        def _get(batch_id):
            return self._batch_doc

        self._batch_doc = None
        main.get_task_batch = MagicMock(side_effect=_get)
        main.update_task_batch = MagicMock()
        main.create_task = MagicMock(return_value={"status": "created", "id": "x"})

    def _install_batch(self, rows, **kw):
        self._batch_doc = _batch(rows, **kw)
        return self._batch_doc


# ── task_add ────────────────────────────────────────────────────────────────

class TestTaskAdd(_CardClickBase):

    def test_task_add_creates_and_updates(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links"),
                _row("t3", title="Follow up w/ Travis")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add", task_id="t1"))

        # exactly one create for the clicked row
        self.assertEqual(main.create_task.call_count, 1)
        _, kwargs = main.create_task.call_args
        self.assertEqual(kwargs.get("title"), "Confirm OneTrust status")
        self.assertEqual(kwargs.get("due_date"), "2026-06-17")

        # row flipped to "added", batch persisted
        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")
        main.update_task_batch.assert_called_once()

        # standard UPDATE_MESSAGE envelope, row collapsed in the rendered tray
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})
        self.assertIn("cardsV2", resp)
        self.assertIn("Added to Google Tasks", json.dumps(resp["cardsV2"]))

    def test_task_add_already_exists_path(self):
        main.create_task = MagicMock(return_value={"status": "already_exists",
                                                   "id": "y"})
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add", task_id="t1"))

        self.assertEqual(_row_by_id(rows, "t1")["state"], "already_exists")
        self.assertIn("Already in your tasks", json.dumps(resp["cardsV2"]))

    def test_task_add_passes_notes_to_create(self):
        """A row carrying notes (conversational creates can) must pass them
        through to create_task — without this, the notes silently drop."""
        rows = [_row("t1")]
        rows[0]["notes"] = "context about FL"
        self._install_batch(rows)

        main.handle_card_click(_ev("task_add", task_id="t1"))

        _, kwargs = main.create_task.call_args
        self.assertEqual(kwargs.get("title"), "Confirm OneTrust status")
        self.assertEqual(kwargs.get("notes"), "context about FL")
        self.assertEqual(kwargs.get("due_date"), "2026-06-17")


# ── task_dismiss ──────────────────────────────────────────────────────────────

class TestTaskDismiss(_CardClickBase):

    def test_task_dismiss_marks_dismissed(self):
        rows = [_row("t1"), _row("t2")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_dismiss", task_id="t1"))

        self.assertEqual(_row_by_id(rows, "t1")["state"], "dismissed")
        self.assertEqual(_row_by_id(rows, "t2")["state"], "pending")
        main.create_task.assert_not_called()
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})
        self.assertIn("Dismissed", json.dumps(resp["cardsV2"]))


# ── task_add_all / task_dismiss_all ───────────────────────────────────────────

class TestBulkActions(_CardClickBase):

    def test_add_all_creates_all_pending(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links"),
                _row("t3", title="Follow up w/ Travis")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add_all"))

        self.assertEqual(main.create_task.call_count, 3)
        self.assertTrue(all(r["state"] == "added" for r in rows))
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})

    def test_add_all_skips_non_pending(self):
        rows = [_row("t1", state="dismissed"), _row("t2"),
                _row("t3", state="added")]
        self._install_batch(rows)

        main.handle_card_click(_ev("task_add_all"))

        # only the single pending row triggers a create
        self.assertEqual(main.create_task.call_count, 1)
        self.assertEqual(_row_by_id(rows, "t2")["state"], "added")
        self.assertEqual(_row_by_id(rows, "t1")["state"], "dismissed")

    def test_dismiss_all(self):
        rows = [_row("t1"), _row("t2"), _row("t3")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_dismiss_all"))

        self.assertTrue(all(r["state"] == "dismissed" for r in rows))
        main.create_task.assert_not_called()
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})


# ── create_task failure handling (DEFECT B) ──────────────────────────────────

class TestCreateFailureHandling(_CardClickBase):
    """A failing create_task must never 500 the click, must label the row
    'failed' (not 'already_exists'), and must still persist the batch."""

    def test_task_add_create_error_sets_failed_state(self):
        main.create_task = MagicMock(return_value={"error": "No task lists found"})
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add", task_id="t1"))

        self.assertEqual(_row_by_id(rows, "t1")["state"], "failed")
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})
        self.assertIn("cardsV2", resp)
        main.update_task_batch.assert_called_once()

    def test_task_add_create_raises_no_500(self):
        main.create_task = MagicMock(side_effect=RuntimeError("Tasks API down"))
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add", task_id="t1"))

        self.assertIsInstance(resp, dict)
        self.assertEqual(_row_by_id(rows, "t1")["state"], "failed")
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})

    def test_add_all_partial_failure_persists(self):
        def _create(title=None, due_date=None, **kw):
            if title == "Send SDK+ links":
                raise RuntimeError("boom")
            return {"status": "created", "id": "x"}

        main.create_task = MagicMock(side_effect=_create)
        rows = [_row("t1"),
                _row("t2", title="Send SDK+ links"),
                _row("t3", title="Follow up w/ Travis")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_add_all"))

        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")
        self.assertEqual(_row_by_id(rows, "t2")["state"], "failed")
        self.assertEqual(_row_by_id(rows, "t3")["state"], "added")
        main.update_task_batch.assert_called_once()
        persisted_rows = main.update_task_batch.call_args[0][1]
        states = {r["taskId"]: r["state"] for r in persisted_rows}
        self.assertEqual(states, {"t1": "added", "t2": "failed", "t3": "added"})
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})


# ── task_edit (dialog) ────────────────────────────────────────────────────────

class TestTaskEdit(_CardClickBase):

    def test_task_edit_returns_dialog(self):
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(_ev("task_edit", task_id="t1"))

        self.assertEqual(resp["actionResponse"]["type"], "DIALOG")
        widgets = (
            resp["actionResponse"]["dialogAction"]["dialog"]["body"]["sections"][0]["widgets"]
        )
        pickers = [
            w["dateTimePicker"] for w in widgets if "dateTimePicker" in w
        ]
        self.assertEqual(len(pickers), 1)
        self.assertEqual(pickers[0]["name"], "due")
        # a create must never happen on edit
        main.create_task.assert_not_called()


# ── task_edit_submit ──────────────────────────────────────────────────────────

class TestTaskEditSubmit(_CardClickBase):

    def test_edit_submit_updates_due_keeps_pending(self):
        rows = [_row("t1", title="Old title", due="2026-06-17")]
        self._install_batch(rows)

        form_inputs = {
            "title": {"stringInputs": {"value": ["Refined title"]}},
            "due": {"dateInput": {"msSinceEpoch": _ms_for("2026-06-20")}},
        }
        resp = main.handle_card_click(
            _ev("task_edit_submit", task_id="t1", form_inputs=form_inputs)
        )

        row = _row_by_id(rows, "t1")
        self.assertEqual(row["due"], "2026-06-20")
        self.assertEqual(row["title"], "Refined title")
        self.assertEqual(row["state"], "pending")
        main.create_task.assert_not_called()
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})


# ── safety: unknown function / missing batch ──────────────────────────────────

class TestSafeNoOp(_CardClickBase):

    def test_unknown_function_no_crash(self):
        rows = [_row("t1")]
        self._install_batch(rows)
        resp = main.handle_card_click(_ev("task_explode", task_id="t1"))
        self.assertNotIn("actionResponse", resp)

    def test_missing_batch_no_crash(self):
        self._batch_doc = None  # get_task_batch returns None
        resp = main.handle_card_click(_ev("task_add", task_id="t1"))
        self.assertNotIn("actionResponse", resp)
        main.create_task.assert_not_called()


# ── ROUND-TRIP: real /chat surface ────────────────────────────────────────────

class TestChatWebhookRoundTrip(_CardClickBase):
    """Mandatory real-surface artifact: a real CARD_CLICKED body flows through
    the async chat_webhook and returns the exact standard UPDATE_MESSAGE dict."""

    class _FakeRequest:
        def __init__(self, body):
            self._body = body
            self.method = "POST"

        async def json(self):
            return self._body

    def test_round_trip_card_clicked_returns_update_message(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links")]
        self._install_batch(rows)
        main.create_task = MagicMock(return_value={"status": "created", "id": "z"})

        body = {
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

        resp = asyncio.run(
            main.chat_webhook(self._FakeRequest(body), MagicMock())
        )

        # exact standard-format envelope
        self.assertEqual(set(resp.keys()), {"actionResponse", "cardsV2"})
        self.assertEqual(resp["actionResponse"], {"type": "UPDATE_MESSAGE"})

        # tray re-rendered with the full /chat URL on every button
        expected = cards.build_task_tray_card(
            "b1", "Petco SDK+ Sync", rows, chat_url=CHAT_URL
        )
        self.assertEqual(resp["cardsV2"], expected)
        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")


# ── ADD-ON dispatch: parameters.actionName (function is now the /chat URL) ────

class TestAddonActionNameDispatch(_CardClickBase):
    """On the add-on framework the button function is the full /chat URL, so the
    real action arrives in parameters.actionName. Dispatch must read actionName
    (falling back to invoked_function for the standard format)."""

    def _addon_ev(self, action_name, batch_id="b1", task_id=None, form_inputs=None):
        params = {"actionName": action_name, "batchId": batch_id}
        if task_id is not None:
            params["taskId"] = task_id
        return {
            "event_type": "CARD_CLICKED",
            "invoked_function": "",
            "parameters": params,
            "form_inputs": form_inputs or {},
            "message_name": "spaces/s/messages/m",
            "is_addon": True,
            "space": "spaces/s",
        }

    def test_addon_card_click_dispatches_on_actionName(self):
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(self._addon_ev("task_add", task_id="t1"))

        self.assertEqual(main.create_task.call_count, 1)
        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")
        self.assertIn("hostAppDataAction", resp)
        self.assertIn(
            "updateMessageAction", resp["hostAppDataAction"]["chatDataAction"]
        )

    def test_addon_dismiss_dispatches_on_actionName(self):
        rows = [_row("t1")]
        self._install_batch(rows)

        resp = main.handle_card_click(self._addon_ev("task_dismiss", task_id="t1"))

        self.assertEqual(_row_by_id(rows, "t1")["state"], "dismissed")
        main.create_task.assert_not_called()
        self.assertIn("hostAppDataAction", resp)

    def test_actionName_wins_over_invoked_function(self):
        rows = [_row("t1")]
        self._install_batch(rows)
        ev = self._addon_ev("task_add", task_id="t1")
        ev["invoked_function"] = "task_dismiss"

        main.handle_card_click(ev)

        self.assertEqual(main.create_task.call_count, 1)
        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")

    def test_rerendered_tray_carries_full_chat_url(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links")]
        self._install_batch(rows)

        resp = main.handle_card_click(self._addon_ev("task_dismiss", task_id="t1"))

        cards_v2 = resp["hostAppDataAction"]["chatDataAction"][
            "updateMessageAction"
        ]["message"]["cardsV2"]
        self.assertIn(CHAT_URL, json.dumps(cards_v2))


class TestAddonChatWebhookRoundTrip(_CardClickBase):
    """Drives the REAL Workspace add-on CARD_CLICKED event (commonEventObject at
    the TOP LEVEL, parameters as a map) through the async chat_webhook end-to-end.
    This is the path that shipped broken: the click reached the endpoint (200 OK)
    but _parse_event read commonEventObject from under chat, so parameters parsed
    empty, the batch lookup failed, and the card never updated."""

    class _FakeRequest:
        def __init__(self, body):
            self._body = body
            self.method = "POST"

        async def json(self):
            return self._body

    def _addon_click_body(self, action_name="task_add", batch_id="b1", task_id="t1"):
        return {
            "commonEventObject": {
                "invokedFunction": CHAT_URL,
                "parameters": {
                    "actionName": action_name,
                    "batchId": batch_id,
                    "taskId": task_id,
                },
                "formInputs": {},
            },
            "chat": {
                "buttonClickedPayload": {
                    "message": {"name": "spaces/s/messages/m"},
                },
                "user": {"name": "users/123", "displayName": "Karim"},
            },
        }

    def test_addon_click_roundtrip_updates_card(self):
        rows = [_row("t1"), _row("t2", title="Send SDK+ links")]
        self._install_batch(rows)
        main.create_task = MagicMock(return_value={"status": "created", "id": "z"})

        resp = asyncio.run(
            main.chat_webhook(self._FakeRequest(self._addon_click_body()), MagicMock())
        )

        self.assertEqual(main.create_task.call_count, 1)
        self.assertEqual(_row_by_id(rows, "t1")["state"], "added")

        cda = resp["hostAppDataAction"]["chatDataAction"]
        self.assertIn("updateMessageAction", cda)
        cards_v2 = cda["updateMessageAction"]["message"]["cardsV2"]
        self.assertIn("Added to Google Tasks", json.dumps(cards_v2))
        self.assertIn(CHAT_URL, json.dumps(cards_v2))

    def test_addon_click_roundtrip_dismiss(self):
        rows = [_row("t1"), _row("t2")]
        self._install_batch(rows)

        resp = asyncio.run(
            main.chat_webhook(
                self._FakeRequest(self._addon_click_body(action_name="task_dismiss")),
                MagicMock(),
            )
        )

        self.assertEqual(_row_by_id(rows, "t1")["state"], "dismissed")
        main.create_task.assert_not_called()
        self.assertIn(
            "updateMessageAction", resp["hostAppDataAction"]["chatDataAction"]
        )

    def test_addon_click_roundtrip_missing_batch_no_crash(self):
        self._batch_doc = None
        resp = asyncio.run(
            main.chat_webhook(self._FakeRequest(self._addon_click_body()), MagicMock())
        )
        self.assertIsInstance(resp, dict)
        self.assertNotIn("hostAppDataAction", resp)
        main.create_task.assert_not_called()


if __name__ == "__main__":
    unittest.main()
