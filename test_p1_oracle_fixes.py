"""KG v2 Phase 1 — oracle-review defect fixes (4 reviewer-mandated defects).

DEFECT 1: knowledge_resolution.apply_resolution must not resurrect decided
          ("approved"/"rejected") merge-queue docs back to "pending" on rerun.
DEFECT 2: agent review_merge_suggestion must fail CLOSED on unrecognized
          decisions ("deny"/"dismiss"/"no"/"") — error, no writes.
DEFECT 3: agent._match_merge_pair must resolve overlapping substring matches to
          the most specific pair, and report genuine ambiguity with no writes.
DEFECT 4: knowledge_graph.semantic_search must canonicalize owner /
          related_people in returned dicts when KG_RESOLUTION_ENABLED, and be
          byte-identical when the flag is off. Raw Firestore docs untouched.

Every test is hermetic: Firestore/Gemini are faked or monkeypatched, no
network, no LLM calls.
"""

import json
from datetime import datetime, timezone

import pytest

import agent
import config
import knowledge_graph as kg
import knowledge_resolution as kr


_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=timezone.utc)


# ── In-memory Firestore fake (overlay collections only) ──────


class _FakeDocSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    @property
    def exists(self):
        return self._data is not None

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeDocRef:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self._id = doc_id

    def get(self):
        return _FakeDocSnapshot(self._id, self._collection.docs.get(self._id))

    def set(self, data):
        self._collection.docs[self._id] = dict(data)


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.docs = {}

    def document(self, doc_id):
        return _FakeDocRef(self, doc_id)

    def stream(self):
        return [_FakeDocSnapshot(i, d) for i, d in self.docs.items()]


class FakeDB:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name)
        return self.collections[name]


def _seed_queue(db, a, b, status, kind="person", confidence=0.80):
    coll = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION)
    doc_id = kr._merge_queue_id(a, b, kind)
    coll.docs[doc_id] = {
        "pair": [a, b],
        "kind": kind,
        "confidence": confidence,
        "status": status,
        "proposed_at": _NOW.isoformat(),
        "decided_at": _NOW.isoformat() if status != "pending" else None,
    }
    return doc_id


# ════════════════════════════════════════════════════════════
# DEFECT 1 — reruns must not resurrect decided queue docs
# ════════════════════════════════════════════════════════════


class TestRerunPreservesDecisions:
    # Entities that re-mine the ("Karim", "Alex Rivera") queue-band pair.
    _ENTITIES = [{"related_people": ["Karim", "Alex Rivera"],
                  "related_projects": []}]

    @pytest.mark.parametrize("decided_status", ["rejected", "approved"])
    def test_rerun_preserves_decided_status(self, decided_status):
        db = FakeDB()
        doc_id = _seed_queue(db, "Karim", "Alex Rivera", decided_status)

        summary = kr.run_resolution(self._ENTITIES, db, _NOW)

        doc = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs[doc_id]
        assert doc["status"] == decided_status, (
            f"rerun must not flip a '{decided_status}' decision back to pending"
        )
        assert summary["preserved"] == 1
        assert summary["queued"] == 0

    def test_rerun_still_queues_genuinely_pending(self):
        """A pre-existing 'pending' doc is still (re)queued, not preserved."""
        db = FakeDB()
        doc_id = _seed_queue(db, "Karim", "Alex Rivera", "pending")

        summary = kr.run_resolution(self._ENTITIES, db, _NOW)

        doc = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs[doc_id]
        assert doc["status"] == "pending"
        assert summary["queued"] == 1
        assert summary["preserved"] == 0


# ════════════════════════════════════════════════════════════
# DEFECT 2 — unrecognized decisions must fail closed
# ════════════════════════════════════════════════════════════


class TestDecisionParsing:
    def _setup(self, monkeypatch):
        monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
        db = FakeDB()
        _seed_queue(db, "Sarah", "Sarah Chen", "pending", confidence=0.82)
        monkeypatch.setattr("conversation_store.get_db", lambda: db)
        return db

    @pytest.mark.parametrize("decision", ["deny", "dismiss", "no", ""])
    def test_unrecognized_decision_errors_with_no_write(self, monkeypatch, decision):
        db = self._setup(monkeypatch)

        called = []
        monkeypatch.setattr("knowledge_resolution.apply_merge",
                            lambda *a, **k: called.append("apply"))
        queue_before = dict(
            db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs
        )

        result = json.loads(agent._dispatch(
            "review_merge_suggestion",
            {"pair": "Sarah and Sarah Chen", "decision": decision},
        ))

        assert result["status"] == "error"
        assert "approve" in result["message"] and "reject" in result["message"]
        assert called == [], "apply_merge must never run on an unknown decision"
        assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
        assert db.collection(
            config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION
        ).docs == queue_before, "queue docs must be untouched"

    @pytest.mark.parametrize("decision", ["approve", "approved"])
    def test_approve_variants_take_approve_path(self, monkeypatch, decision):
        db = self._setup(monkeypatch)
        result = json.loads(agent._dispatch(
            "review_merge_suggestion",
            {"pair": "Sarah and Sarah Chen", "decision": decision},
        ))
        assert result["status"] == "approved"
        assert len(db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs) == 1

    @pytest.mark.parametrize("decision", ["reject", "rejected"])
    def test_reject_variants_take_reject_path(self, monkeypatch, decision):
        db = self._setup(monkeypatch)
        result = json.loads(agent._dispatch(
            "review_merge_suggestion",
            {"pair": "Sarah and Sarah Chen", "decision": decision},
        ))
        assert result["status"] == "rejected"
        assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
        qdoc = next(iter(
            db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs.values()
        ))
        assert qdoc["status"] == "rejected"


# ════════════════════════════════════════════════════════════
# DEFECT 3 — overlapping substring matches must pick the most
#            specific pair; genuine ambiguity → no writes
# ════════════════════════════════════════════════════════════


class TestMatchMergePairSpecificity:
    def _pending(self):
        return [
            {"pair": ["Karim", "Alex Rivera"], "kind": "person",
             "confidence": 0.80, "status": "pending"},
            {"pair": ["Alex Rivera", "user@example.com"], "kind": "person",
             "confidence": 0.95, "status": "pending"},
        ]

    def test_most_specific_pair_wins_over_substring(self):
        match = agent._match_merge_pair(
            "approve Alex Rivera and user@example.com", self._pending()
        )
        assert match is not None
        assert match.get("pair") == ["Alex Rivera", "user@example.com"]

    def test_single_full_match_unaffected(self):
        match = agent._match_merge_pair("Karim and Alex Rivera", self._pending())
        assert match is not None
        assert match.get("pair") == ["Karim", "Alex Rivera"]

    def test_genuinely_ambiguous_returns_ambiguous_no_write(self, monkeypatch):
        monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
        db = FakeDB()
        # Two pairs with no substring relation between their names — a text
        # naming all four is genuinely ambiguous.
        _seed_queue(db, "Sarah", "Bob", "pending")
        _seed_queue(db, "Alice", "Zoe", "pending")
        monkeypatch.setattr("conversation_store.get_db", lambda: db)
        queue_before = {
            doc_id: dict(doc) for doc_id, doc in
            db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs.items()
        }

        result = json.loads(agent._dispatch(
            "review_merge_suggestion",
            {"pair": "approve Sarah Bob Alice Zoe", "decision": "approve"},
        ))

        assert result["status"] == "ambiguous"
        assert sorted(map(tuple, result["candidates"])) == sorted(
            [("Alice", "Zoe"), ("Sarah", "Bob")]
        )
        assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
        assert db.collection(
            config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION
        ).docs == queue_before, "ambiguity must produce zero writes"

    def test_no_match_still_none(self):
        assert agent._match_merge_pair("nothing relevant", self._pending()) is None


# ════════════════════════════════════════════════════════════
# DEFECT 4 — semantic_search display canonicalization
# ════════════════════════════════════════════════════════════


class _FakeVectorQuery:
    def __init__(self, docs):
        self._docs = docs

    def stream(self):
        return list(self._docs)


class _FakeKGCollection:
    def __init__(self, docs):
        self._docs = docs

    def find_nearest(self, **kwargs):
        return _FakeVectorQuery(self._docs)


class _FakeKGDB:
    def __init__(self, kg_docs):
        self._coll = _FakeKGCollection(kg_docs)

    def collection(self, name):
        assert name == config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION
        return self._coll


class TestSemanticSearchCanonicalization:
    def _raw_doc_data(self):
        return {
            "entity_type": "commitment",
            "name": "Q3 report",
            "content": "Sarah owns the Q3 report",
            "owner": "Sarah",
            "related_people": ["Sarah", "Bob"],
            "source_date": "2026-06-01",
        }

    def _run_search(self, monkeypatch, flag_on):
        raw_data = self._raw_doc_data()
        snapshot = _FakeDocSnapshot("ent1", raw_data)
        monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", flag_on)
        monkeypatch.setattr(kg, "_get_embedding", lambda *a, **k: [0.1, 0.2])
        monkeypatch.setattr(kg, "get_db", lambda: _FakeKGDB([snapshot]))
        monkeypatch.setattr(
            kg, "resolve_canonical",
            lambda names: {n: ("Sarah Chen" if n == "Sarah" else n) for n in names},
        )
        results = kg.semantic_search("report", limit=5, threshold=0.0)
        return results, raw_data

    def test_flag_on_maps_owner_and_related_people(self, monkeypatch):
        results, raw_data = self._run_search(monkeypatch, flag_on=True)
        assert len(results) == 1
        assert results[0]["owner"] == "Sarah Chen"
        assert results[0]["related_people"] == ["Sarah Chen", "Bob"]
        # raw Firestore doc data must never be mutated
        assert raw_data["owner"] == "Sarah"
        assert raw_data["related_people"] == ["Sarah", "Bob"]

    def test_flag_off_results_identical(self, monkeypatch):
        results, raw_data = self._run_search(monkeypatch, flag_on=False)
        assert len(results) == 1
        assert results[0]["owner"] == "Sarah"
        assert results[0]["related_people"] == ["Sarah", "Bob"]
        assert raw_data["owner"] == "Sarah"
