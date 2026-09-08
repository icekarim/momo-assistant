"""TDD tests for the inbound-message idempotency primitives
(claim_message_once / release_message_claim) in conversation_store.

Run from repo root:
    python3 -m pytest test_message_idempotency_store.py -q

Mirrors the in-memory Firestore stand-in from test_task_batches_store.py, but
adds create() semantics (raises if the doc already exists) since the claim
primitive is built on the same create-if-absent pattern as
store_pending_task_actions_if_empty.
"""

import sys
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Stub external dependencies BEFORE importing conversation_store.
# ---------------------------------------------------------------------------

sys.modules["google"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.firestore"] = MagicMock()
sys.modules["google.auth"] = MagicMock()


class DummyCache(dict):
    def __init__(self, maxsize, ttl):
        super().__init__()


sys.modules["cachetools"] = MagicMock(TTLCache=DummyCache)

config_mock = MagicMock()
config_mock.GCP_PROJECT_ID = "test-project"
config_mock.FIRESTORE_DATABASE = "testing"
config_mock.FIRESTORE_COLLECTION = "conversations"
config_mock.FIRESTORE_PENDING_TASKS_COLLECTION = "pending_task_proposals"
config_mock.FIRESTORE_TASK_BATCHES_COLLECTION = "task_batches"
config_mock.FIRESTORE_PROCESSED_MESSAGES_COLLECTION = "processed_messages"
config_mock.MAX_CONVERSATION_TURNS = 50
sys.modules["config"] = config_mock


# ---------------------------------------------------------------------------
# In-memory Firestore stand-in with create() that raises on conflict.
# ---------------------------------------------------------------------------

_STORE: dict[str, dict] = {}


class _AlreadyExists(Exception):
    pass


class FakeDocRef:
    def __init__(self, collection: str, doc_id: str):
        self._key = f"{collection}/{doc_id}"

    def create(self, data: dict):
        if self._key in _STORE:
            raise _AlreadyExists(self._key)
        _STORE[self._key] = dict(data)

    def set(self, data: dict):
        _STORE[self._key] = dict(data)

    def get(self):
        return FakeDocSnap(self._key)

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


sys.modules.pop("conversation_store", None)  # evict any leaked sibling mock; import REAL module
import conversation_store  # noqa: E402

conversation_store._db = FakeDB()


class TestClaimMessageOnce(unittest.TestCase):

    def setUp(self):
        _STORE.clear()

    def test_first_claim_returns_true(self):
        self.assertTrue(conversation_store.claim_message_once("spaces/s/messages/m1"))

    def test_duplicate_claim_returns_false(self):
        self.assertTrue(conversation_store.claim_message_once("spaces/s/messages/m1"))
        self.assertFalse(conversation_store.claim_message_once("spaces/s/messages/m1"))

    def test_release_allows_reclaim(self):
        self.assertTrue(conversation_store.claim_message_once("spaces/s/messages/m1"))
        conversation_store.release_message_claim("spaces/s/messages/m1")
        self.assertTrue(conversation_store.claim_message_once("spaces/s/messages/m1"))

    def test_doc_id_is_sanitized(self):
        conversation_store.claim_message_once("spaces/AAA/messages/BBB.CCC")
        self.assertIn("processed_messages/spaces_AAA_messages_BBB.CCC", _STORE)

    def test_records_created_at_and_ttl(self):
        conversation_store.claim_message_once("spaces/s/messages/m1", ttl_hours=1)
        doc = _STORE["processed_messages/spaces_s_messages_m1"]
        self.assertIn("created_at", doc)
        self.assertEqual(doc["ttl_hours"], 1)

    def test_stale_claim_is_lazily_reclaimed(self):
        key = "processed_messages/spaces_s_messages_m1"
        _STORE[key] = {
            "message_name": "spaces/s/messages/m1",
            "created_at": (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat(),
            "ttl_hours": 1,
        }
        self.assertTrue(conversation_store.claim_message_once("spaces/s/messages/m1", ttl_hours=1))

    def test_fresh_claim_within_ttl_is_not_reclaimed(self):
        key = "processed_messages/spaces_s_messages_m1"
        _STORE[key] = {
            "message_name": "spaces/s/messages/m1",
            "created_at": (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(),
            "ttl_hours": 1,
        }
        self.assertFalse(conversation_store.claim_message_once("spaces/s/messages/m1", ttl_hours=1))


if __name__ == "__main__":
    unittest.main()
