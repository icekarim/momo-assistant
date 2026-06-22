"""Tests for task-creation dedup identity and atomicity in tasks_service.

These cover two production bugs:
  1. Contradictory "created + already exists" (non-atomic / double-insert).
  2. False "already exists" when the title matches but the due date differs.

Dedup identity is title + due date (normalized to 'YYYY-MM-DD'), not a
title-only fuzzy match. Two tasks with the same title but different due dates
are distinct tasks.

Heavy/external imports are stubbed via sys.modules BEFORE importing
tasks_service so the suite runs without Google credentials or network.

Run from the repo root:
    python3 -m pytest test_tasks_identity.py -q
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

# --- Stub external/heavy imports before importing tasks_service ------------
_googleapiclient = MagicMock()
_discovery = MagicMock()
_googleapiclient.discovery = _discovery
sys.modules["google"] = MagicMock()
sys.modules["googleapiclient"] = _googleapiclient
sys.modules["googleapiclient.discovery"] = _discovery
sys.modules["googleapiclient.errors"] = MagicMock()
sys.modules["config"] = MagicMock()
sys.modules["google_auth"] = MagicMock()

sys.modules.pop("tasks_service", None)  # evict any leaked sibling mock; import REAL module
import tasks_service  # noqa: E402


def _make_service(open_tasks=None, completed_tasks=None):
    """Build a mock Google Tasks service.

    tasklists().list() -> one list "My Tasks".
    tasks().list(showCompleted=False) -> open_tasks.
    tasks().list(showCompleted=True)  -> completed_tasks (used by
        find_completed_task).
    tasks().insert(body=...) -> echoes the body's title with a new id.
    """
    open_tasks = open_tasks or []
    completed_tasks = completed_tasks or []

    svc = MagicMock()
    svc.tasklists.return_value.list.return_value.execute.return_value = {
        "items": [{"id": "list1", "title": "My Tasks"}]
    }

    tasks_resource = svc.tasks.return_value

    def _list_side_effect(**kwargs):
        resp = MagicMock()
        if kwargs.get("showCompleted"):
            resp.execute.return_value = {"items": completed_tasks}
        else:
            resp.execute.return_value = {"items": open_tasks}
        return resp

    tasks_resource.list.side_effect = _list_side_effect

    def _insert_side_effect(**kwargs):
        body = kwargs.get("body", {})
        resp = MagicMock()
        resp.execute.return_value = {"id": "new-task-id", "title": body.get("title", "")}
        return resp

    tasks_resource.insert.side_effect = _insert_side_effect
    return svc


class NormalizeDueHelperTests(unittest.TestCase):
    """_normalize_due must coerce every due shape that flows through dedup to
    'YYYY-MM-DD': already-ISO, RFC3339 (Tasks API), and the human display
    format ('%b %d, %Y') produced by fetch_open_tasks. None -> None."""

    def test_iso_date_passthrough(self):
        self.assertEqual(tasks_service._normalize_due("2026-06-17"), "2026-06-17")

    def test_rfc3339_timestamp_truncated(self):
        self.assertEqual(
            tasks_service._normalize_due("2026-06-17T00:00:00.000Z"), "2026-06-17"
        )

    def test_display_format_parsed(self):
        self.assertEqual(tasks_service._normalize_due("Jun 17, 2026"), "2026-06-17")

    def test_none_returns_none(self):
        self.assertIsNone(tasks_service._normalize_due(None))

    def test_empty_returns_none(self):
        self.assertIsNone(tasks_service._normalize_due("  "))


class TaskIdentityMatchHelperTests(unittest.TestCase):
    """Unit tests for the _task_identity_match(title, due, ...) helper."""

    def test_same_title_same_due_matches(self):
        self.assertTrue(
            tasks_service._task_identity_match(
                "Send guide", "2026-06-17", "Send guide", "2026-06-17T00:00:00.000Z"
            )
        )

    def test_same_title_different_due_does_not_match(self):
        self.assertFalse(
            tasks_service._task_identity_match(
                "Send guide", "2026-06-30", "Send guide", "2026-06-17T00:00:00.000Z"
            )
        )

    def test_both_missing_due_matches_on_title(self):
        self.assertTrue(
            tasks_service._task_identity_match("Send guide", None, "Send guide", None)
        )

    def test_one_due_missing_does_not_match(self):
        self.assertFalse(
            tasks_service._task_identity_match(
                "Send guide", None, "Send guide", "2026-06-17T00:00:00.000Z"
            )
        )


class CreateTaskCharacterizationTests(unittest.TestCase):
    """Pin the status contract consumed by callers: must keep returning
    'created', 'already_exists', and 'already_completed'."""

    def test_no_match_returns_created(self):
        svc = _make_service(open_tasks=[], completed_tasks=[])
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Brand new task")
        self.assertEqual(result["status"], "created")

    def test_open_match_returns_already_exists(self):
        svc = _make_service(
            open_tasks=[{"id": "e1", "title": "Existing task", "status": "needsAction"}]
        )
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Existing task")
        self.assertEqual(result["status"], "already_exists")

    def test_completed_match_returns_already_completed(self):
        svc = _make_service(
            open_tasks=[],
            completed_tasks=[{"id": "c1", "title": "Finished task", "status": "completed"}],
        )
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Finished task")
        self.assertEqual(result["status"], "already_completed")


class CreateTaskIdentityTests(unittest.TestCase):
    """The two production bugs and the atomicity guard."""

    def test_same_title_different_due_not_duplicate(self):
        # Existing "Send guide" due 2026-06-17; create same title due 2026-06-30.
        svc = _make_service(
            open_tasks=[
                {
                    "id": "e1",
                    "title": "Send guide",
                    "due": "2026-06-17T00:00:00.000Z",
                    "status": "needsAction",
                }
            ]
        )
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Send guide", due_date="2026-06-30")
        self.assertEqual(result["status"], "created")
        self.assertEqual(svc.tasks.return_value.insert.call_count, 1)

    def test_same_title_same_due_is_duplicate(self):
        svc = _make_service(
            open_tasks=[
                {
                    "id": "e1",
                    "title": "Send guide",
                    "due": "2026-06-17T00:00:00.000Z",
                    "status": "needsAction",
                }
            ]
        )
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Send guide", due_date="2026-06-17")
        self.assertEqual(result["status"], "already_exists")
        self.assertEqual(svc.tasks.return_value.insert.call_count, 0)

    def test_exactly_one_insert_per_create(self):
        svc = _make_service(open_tasks=[], completed_tasks=[])
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Unique task", due_date="2026-06-20")
        self.assertEqual(result["status"], "created")
        self.assertEqual(svc.tasks.return_value.insert.call_count, 1)

    def test_no_due_dates_behaves_sanely(self):
        # Both new and existing lack a due date -> title match still dedups.
        svc = _make_service(
            open_tasks=[{"id": "e1", "title": "Email the team", "status": "needsAction"}]
        )
        with patch.object(tasks_service, "get_tasks_service", return_value=svc):
            result = tasks_service.create_task("Email the team")
        self.assertEqual(result["status"], "already_exists")
        self.assertEqual(svc.tasks.return_value.insert.call_count, 0)


if __name__ == "__main__":
    unittest.main()
