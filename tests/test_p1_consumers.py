"""KG v2 Phase 1 — consumer integration + merge-approval flow tests.

Proves the four Phase-1 consumer surfaces behind KG_RESOLUTION_ENABLED:
  1. Meeting prep alias expansion (proactive_intelligence._build_meeting_prep)
  2. Pattern-detection canonicalization (proactive_intelligence._run_pattern_engine)
  3. Merge-queue briefing section (briefing._build_merge_suggestions_block)
  4. review_merge_suggestion agent tool (agent._dispatch)

Every test is hermetic: Firestore/Gmail/Gemini/Chat are faked or monkeypatched,
no network, no LLM calls. The central guarantee is the flag-off NOOP: the same
fixture inputs are run flag-on vs flag-off and the outputs diffed, proving the
flag default (false) is a behavioural no-op.
"""

from datetime import datetime, timezone

import pytest

import config
import knowledge_graph as kg
import knowledge_resolution as kr
import proactive_intelligence as pi
import briefing
import agent


_NOW = datetime(2026, 6, 9, 12, 0, 0, tzinfo=timezone.utc)


# ── In-memory Firestore fake (overlay collections only) ──────


class _FakeDocSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return dict(self._data) if self._data is not None else None


class _FakeDocRef:
    def __init__(self, collection, doc_id):
        self._collection = collection
        self._id = doc_id

    def set(self, data):
        self._collection.docs[self._id] = dict(data)


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.docs = {}
        self.stream_calls = 0

    def document(self, doc_id):
        return _FakeDocRef(self, doc_id)

    def stream(self):
        self.stream_calls += 1
        return [_FakeDocSnapshot(i, d) for i, d in self.docs.items()]


class FakeDB:
    def __init__(self):
        self.collections = {}

    def collection(self, name):
        if name not in self.collections:
            self.collections[name] = _FakeCollection(name)
        return self.collections[name]


def _seed_canonical(db, display_name, aliases, kind="person"):
    coll = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION)
    doc = {
        "canonical_id": kr._canonical_id(aliases),
        "kind": kind,
        "display_name": display_name,
        "aliases": aliases,
        "alias_tokens": kr._alias_tokens(aliases, kind),
        "confidence": 0.95,
        "created_at": _NOW.isoformat(),
        "source": "auto",
    }
    coll.docs[doc["canonical_id"]] = doc
    return doc


def _seed_pending(db, a, b, kind="person", confidence=0.80):
    coll = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION)
    doc_id = kr._merge_queue_id(a, b, kind)
    coll.docs[doc_id] = {
        "pair": [a, b],
        "kind": kind,
        "confidence": confidence,
        "status": "pending",
        "proposed_at": _NOW.isoformat(),
    }
    return doc_id


@pytest.fixture(autouse=True)
def _clear_canonical_cache():
    kg._canonical_cache.clear()
    yield
    kg._canonical_cache.clear()


# ════════════════════════════════════════════════════════════
# Deliverable 1 — meeting-prep alias expansion
# ════════════════════════════════════════════════════════════


def _meeting():
    return {"title": "Sync", "start_time": "2pm",
            "attendees": [{"name": "Sarah"}, {"name": "Bob"}]}


def _capture_prep_queries(monkeypatch):
    """Run _build_meeting_prep with all KG queries stubbed; return the set of
    names passed to query_by_person."""
    queried: list[str] = []

    def _fake_query_by_person(name, since=None, limit=50):
        queried.append(name)
        return []

    monkeypatch.setattr(pi, "query_by_person", _fake_query_by_person)
    monkeypatch.setattr(pi, "query_by_project", lambda *a, **k: [])
    monkeypatch.setattr(kg, "semantic_search", lambda *a, **k: [])
    # generate() is the only LLM boundary — stub to a fixed brief.
    monkeypatch.setattr(pi, "generate", lambda **k: "brief")
    monkeypatch.setattr(pi, "extract_text", lambda msg: "brief")

    pi._build_meeting_prep(_meeting())
    return set(queried)


def test_meeting_prep_query_set_identical_flag_off(monkeypatch):
    """NOOP: flag off → query set is exactly the attendee names."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)

    def _boom():
        raise AssertionError("flag off must not query kg_canonical")

    monkeypatch.setattr(kg, "get_db", _boom)
    assert _capture_prep_queries(monkeypatch) == {"Sarah", "Bob"}


def test_meeting_prep_flag_on_off_diff(monkeypatch):
    """Same fixture, flag-on adds canonical alias queries that flag-off omits."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    monkeypatch.setattr(kg, "get_db", lambda: FakeDB())
    off = _capture_prep_queries(monkeypatch)

    kg._canonical_cache.clear()
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = FakeDB()
    _seed_canonical(db, "Sarah Chen", ["Sarah", "Sarah Chen", "sarah.chen@x.com"])
    monkeypatch.setattr(kg, "get_db", lambda: db)
    on = _capture_prep_queries(monkeypatch)

    assert off == {"Sarah", "Bob"}
    # flag-on expands Sarah's lookup with her canonical aliases
    assert "Sarah Chen" in on
    assert "sarah.chen@x.com" in on
    assert on > off


def test_meeting_prep_dedupes_by_entity_id(monkeypatch):
    """Alias expansion that surfaces the same KG entity twice is deduped by id."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = FakeDB()
    _seed_canonical(db, "Sarah Chen", ["Sarah", "Sarah Chen"])
    monkeypatch.setattr(kg, "get_db", lambda: db)

    shared = {"id": "ent1", "related_people": ["Sarah Chen"], "related_projects": []}

    def _fake_query_by_person(name, since=None, limit=50):
        # Both "Sarah" and "Sarah Chen" lookups return the same entity.
        return [shared] if name in ("Sarah", "Sarah Chen") else []

    captured = {}

    def _fake_format(entries):
        captured["entries"] = entries
        return "ctx"

    monkeypatch.setattr(pi, "query_by_person", _fake_query_by_person)
    monkeypatch.setattr(pi, "query_by_project", lambda *a, **k: [])
    monkeypatch.setattr(kg, "semantic_search", lambda *a, **k: [])
    monkeypatch.setattr(pi, "format_knowledge_for_context", _fake_format)
    monkeypatch.setattr(pi, "generate", lambda **k: "brief")
    monkeypatch.setattr(pi, "extract_text", lambda msg: "brief")

    pi._build_meeting_prep({"title": "Sync", "start_time": "2pm",
                            "attendees": [{"name": "Sarah"}]})

    ids = [e["id"] for e in captured["entries"]]
    assert ids.count("ent1") == 1, "the same entity must not be added twice"


# ════════════════════════════════════════════════════════════
# Deliverable 2 — pattern-detection canonicalization
# ════════════════════════════════════════════════════════════


def _pattern_entries():
    # 5 entries (engine needs >=5). Sarah=2, Sarah Chen=1 → split max 2 (below
    # the >=3 threshold), but they merge to a canonical count of 3. Same for
    # Ads (2) / ads (1). The two filler entries only pad the corpus to 5.
    return [
        {"related_people": ["Sarah"], "related_projects": ["Ads"], "tags": [],
         "entity_type": "topic"},
        {"related_people": ["Sarah"], "related_projects": ["Ads"], "tags": [],
         "entity_type": "decision"},
        {"related_people": ["Sarah Chen"], "related_projects": ["ads"], "tags": [],
         "entity_type": "topic"},
        {"related_people": ["Zoe"], "related_projects": ["Other"], "tags": [],
         "entity_type": "topic"},
        {"related_people": ["Yan"], "related_projects": ["Misc"], "tags": [],
         "entity_type": "topic"},
    ]


def _run_pattern_capture(monkeypatch):
    """Run _run_pattern_engine up to the LLM boundary; capture the patterns text
    handed to generate()."""
    captured = {}

    def _fake_generate(prompt=None, **k):
        captured["prompt"] = prompt
        return "msg"

    monkeypatch.setattr(pi, "query_recent", lambda **k: _pattern_entries())
    monkeypatch.setattr(pi, "has_nudge_been_sent", lambda key: False)
    monkeypatch.setattr(pi, "generate", _fake_generate)
    monkeypatch.setattr(pi, "extract_text", lambda msg: "- insight")

    pi._run_pattern_engine()
    return captured.get("prompt", "")


def test_pattern_counters_identical_flag_off(monkeypatch):
    """NOOP: flag off → counters use raw names, split counts stay split."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    monkeypatch.setattr(pi, "resolve_canonical",
                        lambda names: {n: n for n in names})
    prompt = _run_pattern_capture(monkeypatch)
    # Sarah=2, Sarah Chen=1 — neither reaches the >=3 threshold, so no people line.
    assert "Sarah: mentioned in 3" not in prompt
    assert "Ads: 3 mentions" not in prompt


def test_pattern_counters_merge_when_flag_on(monkeypatch):
    """Flag on → split counts merge onto the canonical display name."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)

    def _resolve(names):
        out = {}
        for n in names:
            low = n.strip().lower()
            if low in ("sarah", "sarah chen"):
                out[n] = "Sarah Chen"
            elif low in ("ads",):
                out[n] = "Ads"
            else:
                out[n] = n
        return out

    monkeypatch.setattr(pi, "resolve_canonical", _resolve)
    prompt = _run_pattern_capture(monkeypatch)
    # Sarah (2) + Sarah Chen (1) collapse to canonical "Sarah Chen" = 3 → surfaces.
    assert "Sarah Chen: mentioned in 3 entries" in prompt
    assert "Ads: 3 mentions" in prompt


# ════════════════════════════════════════════════════════════
# Deliverable 3 — merge-queue briefing section
# ════════════════════════════════════════════════════════════


def test_briefing_merge_block_empty_flag_off(monkeypatch):
    """NOOP: flag off → no merge block, no Firestore read."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)

    def _boom(*a, **k):
        raise AssertionError("flag off must not read the merge queue")

    monkeypatch.setattr(briefing, "get_pending_merge_suggestions", _boom, raising=False)
    assert briefing._build_merge_suggestions_block() == ""


def test_briefing_merge_block_lists_pending_when_flag_on(monkeypatch):
    """Flag on with pending docs → compact block, max 3, 'A' ↔ 'B' (conf)."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)

    pending = [
        {"pair": ["Sarah", "Sarah Chen"], "kind": "person", "confidence": 0.82,
         "status": "pending"},
        {"pair": ["Ads", "Ads Team"], "kind": "project", "confidence": 0.78,
         "status": "pending"},
    ]
    monkeypatch.setattr(briefing, "get_pending_merge_suggestions",
                        lambda limit=3: pending[:limit], raising=False)

    block = briefing._build_merge_suggestions_block()
    assert block
    assert "'Sarah' ↔ 'Sarah Chen'" in block
    assert "0.82" in block
    assert "'Ads' ↔ 'Ads Team'" in block


def test_briefing_merge_block_caps_at_three(monkeypatch):
    """Never more than 3 suggestions in the block."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    many = [
        {"pair": [f"A{i}", f"B{i}"], "kind": "person", "confidence": 0.8,
         "status": "pending"}
        for i in range(6)
    ]
    monkeypatch.setattr(briefing, "get_pending_merge_suggestions",
                        lambda limit=3: many[:limit], raising=False)

    block = briefing._build_merge_suggestions_block()
    assert block.count("↔") <= 3


def test_get_pending_merge_suggestions_flag_off():
    """The resolution helper itself is a no-op when the flag is off."""
    orig = config.KG_RESOLUTION_ENABLED
    config.KG_RESOLUTION_ENABLED = False
    try:
        assert kr.get_pending_merge_suggestions(limit=3, db=FakeDB()) == []
    finally:
        config.KG_RESOLUTION_ENABLED = orig


def test_get_pending_merge_suggestions_flag_on():
    """Flag on → only pending docs, highest confidence first, capped at limit."""
    orig = config.KG_RESOLUTION_ENABLED
    config.KG_RESOLUTION_ENABLED = True
    try:
        db = FakeDB()
        _seed_pending(db, "Sarah", "Sarah Chen", confidence=0.78)
        _seed_pending(db, "Karim", "Karim T", confidence=0.88)
        # a non-pending doc must be ignored
        db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs["done"] = {
            "pair": ["X", "Y"], "kind": "person", "confidence": 0.99,
            "status": "approved",
        }
        out = kr.get_pending_merge_suggestions(limit=3, db=db)
        assert [d["confidence"] for d in out] == [0.88, 0.78]
        assert all(d["status"] == "pending" for d in out)
    finally:
        config.KG_RESOLUTION_ENABLED = orig


# ════════════════════════════════════════════════════════════
# Deliverable 4 — review_merge_suggestion agent tool
# ════════════════════════════════════════════════════════════


def test_tool_not_declared_when_flag_off(monkeypatch):
    """NOOP: flag off → tool set is byte-identical (tool absent)."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    names = [t["name"] for t in agent._get_all_tools()]
    assert "review_merge_suggestion" not in names


def test_tool_declared_when_flag_on(monkeypatch):
    """Flag on → the tool is declared following the optional-tool pattern."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    names = [t["name"] for t in agent._get_all_tools()]
    assert "review_merge_suggestion" in names


def test_tool_returns_disabled_when_flag_off(monkeypatch):
    """NOOP: executor branch returns 'resolution disabled' when flag off."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    result = agent._dispatch("review_merge_suggestion",
                             {"pair": "Sarah and Sarah Chen", "decision": "approve"})
    assert result == "resolution disabled"


def test_tool_approve_writes_canonical_and_marks_queue(monkeypatch):
    """Approve → kg_canonical doc (source 'approved') + queue doc 'approved'."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = FakeDB()
    _seed_pending(db, "Sarah", "Sarah Chen", confidence=0.82)
    monkeypatch.setattr("conversation_store.get_db", lambda: db)

    result = agent._dispatch(
        "review_merge_suggestion",
        {"pair": "Sarah / Sarah Chen", "decision": "approve"},
    )

    canonical = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs
    assert len(canonical) == 1
    cdoc = next(iter(canonical.values()))
    assert cdoc["source"] == "approved"
    assert set(cdoc["aliases"]) == {"Sarah", "Sarah Chen"}

    queue = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs
    qdoc = next(iter(queue.values()))
    assert qdoc["status"] == "approved"
    assert "decided_at" in qdoc
    assert "approved" in result


def test_tool_reject_only_flips_status(monkeypatch):
    """Reject → queue doc 'rejected', NO canonical doc written."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = FakeDB()
    _seed_pending(db, "Sarah", "Sarah Chen", confidence=0.82)
    monkeypatch.setattr("conversation_store.get_db", lambda: db)

    result = agent._dispatch(
        "review_merge_suggestion",
        {"pair": "Sarah and Sarah Chen", "decision": "reject"},
    )

    assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
    queue = db.collection(config.FIRESTORE_KG_MERGE_QUEUE_COLLECTION).docs
    qdoc = next(iter(queue.values()))
    assert qdoc["status"] == "rejected"
    assert "rejected" in result


def test_tool_no_match_returns_not_found(monkeypatch):
    """A pair text that matches no pending merge → not_found, no writes."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = FakeDB()
    _seed_pending(db, "Sarah", "Sarah Chen", confidence=0.82)
    monkeypatch.setattr("conversation_store.get_db", lambda: db)

    result = agent._dispatch(
        "review_merge_suggestion",
        {"pair": "totally unrelated names", "decision": "approve"},
    )
    assert "not_found" in result
    assert db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION).docs == {}
