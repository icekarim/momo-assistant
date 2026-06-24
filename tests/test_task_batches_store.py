"""TDD tests for task-batch state store (store_task_batch / get_task_batch / update_task_batch).

Run from repo root:
    python3 -m pytest test_task_batches_store.py -q
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub all external dependencies BEFORE importing conversation_store.
# Mirror the pattern from test_task_approval_safety.py:7-44.
# ---------------------------------------------------------------------------

sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["google.auth"] = MagicMock()


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


sys.modules["cachetools"] = MagicMock(TTLCache=DummyCache)

# Config stub — only the constants conversation_store actually references.
config_mock = MagicMock()
config_mock.GCP_PROJECT_ID = "test-project"
config_mock.FIRESTORE_DATABASE = "testing"
config_mock.FIRESTORE_COLLECTION = "conversations"
config_mock.FIRESTORE_PENDING_TASKS_COLLECTION = "pending_task_proposals"
config_mock.FIRESTORE_TASK_BATCHES_COLLECTION = "task_batches"
config_mock.MAX_CONVERSATION_TURNS = 50
sys.modules["config"] = config_mock

# ---------------------------------------------------------------------------
# In-memory Firestore stand-in
# ---------------------------------------------------------------------------

_STORE: dict[str, dict] = {}  # keyed by collection/doc_id


class FakeDocRef:
    def __init__(self, collection: str, doc_id: str):
        self._col = collection
        self._doc_id = doc_id
        self._key = f"{collection}/{doc_id}"

    def set(self, data: dict):
        _STORE[self._key] = dict(data)

    def get(self):
        return FakeDocSnap(self._key)

    def update(self, data: dict):
        if self._key in _STORE:
            _STORE[self._key].update(data)

    def delete(self):
        _STORE.pop(self._key, None)


class FakeDocSnap:
    def __init__(self, key: str):
        self._key = key

    @property
    def exists(self):
        return self._key in _STORE

    def to_dict(self):
        return dict(_STORE[self._key])


class FakeCollection:
    def __init__(self, name: str):
        self._name = name

    def document(self, doc_id: str):
        return FakeDocRef(self._name, doc_id)


class FakeDB:
    def collection(self, name: str):
        return FakeCollection(name)


# ---------------------------------------------------------------------------
# Patch get_db() so conversation_store uses FakeDB.
# ---------------------------------------------------------------------------

import importlib

# We need to reset _db each test run; easiest is to patch before import.
sys.modules.pop("conversation_store", None)  # evict any leaked sibling mock; import REAL module
import conversation_store  # noqa: E402 — must come after sys.modules stubs

_fake_db = FakeDB()
conversation_store._db = _fake_db  # bypass lazy init


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SAMPLE_ROWS = [
    {"taskId": "t1", "title": "Write tests", "due": "2026-06-20", "owner": "karim",
     "priority": "High", "state": "pending"},
    {"taskId": "t2", "title": "Ship feature", "due": None, "owner": None,
     "priority": "Medium", "state": "pending"},
    {"taskId": "t3", "title": "Update docs", "due": None, "owner": "alice",
     "priority": "Low", "state": "pending"},
]


class TestTaskBatchStore(unittest.TestCase):

    def setUp(self):
        _STORE.clear()

    # (a) ------------------------------------------------------------------
    def test_store_and_get_roundtrip(self):
        """store_task_batch then get_task_batch returns all 3 rows + source + space."""
        conversation_store.store_task_batch(
            batch_id="batch-001",
            source="meeting",
            space="spaces/ABC",
            rows=SAMPLE_ROWS,
        )
        result = conversation_store.get_task_batch("batch-001")
        self.assertIsNotNone(result)
        self.assertEqual(result["source"], "meeting")
        self.assertEqual(result["space"], "spaces/ABC")
        self.assertEqual(len(result["rows"]), 3)
        self.assertEqual(result["rows"][0]["taskId"], "t1")
        self.assertEqual(result["rows"][2]["state"], "pending")

    # (b) ------------------------------------------------------------------
    def test_update_marks_row_added(self):
        """update_task_batch flips t2 to 'added'; re-get reflects the change."""
        conversation_store.store_task_batch(
            batch_id="batch-002",
            source="chat",
            space="spaces/XYZ",
            rows=SAMPLE_ROWS,
        )
        updated_rows = [dict(r) for r in SAMPLE_ROWS]
        updated_rows[1]["state"] = "added"
        conversation_store.update_task_batch("batch-002", updated_rows)

        result = conversation_store.get_task_batch("batch-002")
        self.assertIsNotNone(result)
        self.assertEqual(result["rows"][1]["state"], "added")
        # Other rows unchanged
        self.assertEqual(result["rows"][0]["state"], "pending")
        self.assertEqual(result["rows"][2]["state"], "pending")

    # (c) ------------------------------------------------------------------
    def test_expiry_after_24h(self):
        """get_task_batch returns None and deletes the doc when created_at is 25h old."""
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
        key = "task_batches/batch-003"
        _STORE[key] = {
            "source": "meeting",
            "space": "spaces/OLD",
            "rows": SAMPLE_ROWS,
            "created_at": old_ts,
        }
        result = conversation_store.get_task_batch("batch-003")
        self.assertIsNone(result)
        # Lazy delete should have removed the doc.
        self.assertNotIn(key, _STORE)

    # (d) ------------------------------------------------------------------
    def test_two_batches_no_collision(self):
        """Two independent batches are stored and retrieved without overwriting each other.

        This is the canonical regression test for the vanishing-task bug:
        the old single-slot model let batch-2 silently overwrite batch-1.
        """
        rows_b1 = [{"taskId": "b1-t1", "title": "Alpha", "due": None,
                     "owner": None, "priority": None, "state": "pending"}]
        rows_b2 = [{"taskId": "b2-t1", "title": "Beta", "due": None,
                     "owner": None, "priority": None, "state": "pending"},
                   {"taskId": "b2-t2", "title": "Gamma", "due": None,
                    "owner": None, "priority": None, "state": "added"}]

        conversation_store.store_task_batch("batch-A", "chat", "spaces/S1", rows_b1)
        conversation_store.store_task_batch("batch-B", "meeting", "spaces/S2", rows_b2)

        ra = conversation_store.get_task_batch("batch-A")
        rb = conversation_store.get_task_batch("batch-B")

        self.assertIsNotNone(ra)
        self.assertIsNotNone(rb)
        self.assertEqual(len(ra["rows"]), 1)
        self.assertEqual(ra["rows"][0]["taskId"], "b1-t1")
        self.assertEqual(len(rb["rows"]), 2)
        self.assertEqual(rb["rows"][1]["state"], "added")
        # Sources are independent
        self.assertEqual(ra["source"], "chat")
        self.assertEqual(rb["source"], "meeting")


if __name__ == "__main__":
    unittest.main()
