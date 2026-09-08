"""Tests for knowledge_graph.resolve_canonical — KG v2 Phase 1 read-time overlay.

resolve_canonical maps raw name/email strings to canonical display names using
an in-memory TTL cache of the kg_canonical overlay (mirrors the existing
_kg_cache pattern). It returns an identity mapping when KG_RESOLUTION_ENABLED is
false OR when the overlay is empty, and NEVER reads or mutates the raw
knowledge_graph collection.

Firestore is faked in-memory and get_db is monkeypatched, so nothing here hits a
live backend.
"""

import pytest

import config
import knowledge_graph as kg


# ── In-memory Firestore fake (counts stream() calls for TTL assertions) ──


class _FakeDocSnapshot:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _FakeCollection:
    def __init__(self, name):
        self.name = name
        self.docs = {}
        self.stream_calls = 0

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


def _canonical_doc(display_name, aliases, kind="person"):
    return {
        "canonical_id": "x" * 16,
        "kind": kind,
        "display_name": display_name,
        "aliases": aliases,
        "alias_tokens": [],
        "confidence": 0.95,
        "created_at": "2026-06-09T00:00:00+00:00",
        "source": "auto",
    }


@pytest.fixture(autouse=True)
def _clear_canonical_cache():
    """The TTL cache is module-level; clear it around every test so cases don't
    leak cached mappings into one another."""
    kg._canonical_cache.clear()
    yield
    kg._canonical_cache.clear()


def _seed_db(monkeypatch, docs):
    db = FakeDB()
    coll = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION)
    for i, doc in enumerate(docs):
        coll.docs[f"doc{i}"] = doc
    monkeypatch.setattr(kg, "get_db", lambda: db)
    return db


# ── Flag-off behaviour ───────────────────────────────────────


def test_identity_when_flag_off(monkeypatch):
    """KG_RESOLUTION_ENABLED false → identity mapping, and no Firestore query."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)

    def _boom():
        raise AssertionError("resolve_canonical must not query when flag is off")

    monkeypatch.setattr(kg, "get_db", _boom)

    result = kg.resolve_canonical(["Sarah Chen", "sarah.chen@example.com"])
    assert result == {
        "Sarah Chen": "Sarah Chen",
        "sarah.chen@example.com": "sarah.chen@example.com",
    }


# ── Flag-on behaviour ────────────────────────────────────────


def test_identity_when_overlay_empty(monkeypatch):
    """Flag on but kg_canonical empty → identity mapping."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    _seed_db(monkeypatch, [])

    result = kg.resolve_canonical(["Sarah Chen"])
    assert result == {"Sarah Chen": "Sarah Chen"}


def test_maps_alias_to_display_name(monkeypatch):
    """Flag on with overlay data → raw alias maps to canonical display name."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    _seed_db(
        monkeypatch,
        [_canonical_doc("Sarah Chen", ["Sarah Chen", "sarah.chen@example.com"])],
    )

    result = kg.resolve_canonical(["sarah.chen@example.com", "Bob"])
    assert result["sarah.chen@example.com"] == "Sarah Chen"
    # unknown names pass through unchanged
    assert result["Bob"] == "Bob"


def test_maps_case_insensitively(monkeypatch):
    """Lookup is normalised → a case/spacing variant still resolves."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    _seed_db(
        monkeypatch,
        [_canonical_doc("Aidan McWeeney", ["Aidan McWeeney", "Aidan Mcweeney"])],
    )

    result = kg.resolve_canonical(["aidan  mcweeney"])
    assert result["aidan  mcweeney"] == "Aidan McWeeney"


# ── TTL cache behaviour ──────────────────────────────────────


def test_ttl_cache_avoids_requery(monkeypatch):
    """A second call within the TTL window must hit the cache, not Firestore."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = _seed_db(
        monkeypatch,
        [_canonical_doc("Sarah Chen", ["Sarah Chen", "sarah.chen@example.com"])],
    )
    coll = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION)

    kg.resolve_canonical(["sarah.chen@example.com"])
    assert coll.stream_calls == 1

    # second call within TTL → cached, no extra stream()
    kg.resolve_canonical(["Sarah Chen"])
    assert coll.stream_calls == 1


def test_cache_repopulates_after_clear(monkeypatch):
    """Clearing the cache forces a fresh query (sanity check for the fixture)."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)
    db = _seed_db(
        monkeypatch,
        [_canonical_doc("Sarah Chen", ["Sarah Chen", "sarah.chen@example.com"])],
    )
    coll = db.collection(config.FIRESTORE_KG_CANONICAL_COLLECTION)

    kg.resolve_canonical(["Sarah Chen"])
    assert coll.stream_calls == 1

    kg._canonical_cache.clear()
    kg.resolve_canonical(["Sarah Chen"])
    assert coll.stream_calls == 2
