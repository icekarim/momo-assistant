"""KG full-collection scan — field projection + explicit stream retry.

Regression for the drift-engine failure: `query_all_entries(limit=5000)` →
`_load_all_entries` streamed the ENTIRE knowledge_graph collection with every
doc's 2048-dim `embedding` array (~100MB across ~5.7k docs), intermittently
hitting Firestore 503 "Query timed out". The client's DEFAULT mid-stream retry
is broken in all google-cloud-firestore 2.x releases (upstream #596:
`_retry` lookup on a raw `_UnaryStreamMultiCallable` → AttributeError), so the
transient 503 became the hard failure "'_UnaryStreamMultiCallable' object has
no attribute '_retry'". The fix: project the scan down to `_ENTRY_FIELDS`
(write schema minus `embedding`) and pass an explicit `_KG_STREAM_RETRY`.

Covers:
  (a) _load_all_entries projects with _ENTRY_FIELDS (no `embedding`) and
      streams with retry=_KG_STREAM_RETRY + a finite timeout
  (b) query_all_entries still sorts by source_date desc and applies limit
  (c) _KG_STREAM_RETRY's predicate retries transient stream errors
      (ServiceUnavailable / DeadlineExceeded / InternalServerError) but not
      terminal ones (PermissionDenied)

Hermetic: Firestore is faked at the get_db() seam, no network. Isolation: a
fresh REAL copy of knowledge_graph.py is exec'd under a private module name
(pattern: tests/test_debrief_emits_tray_card.py::_load_real_tasks_service)
with its heavy google-namespace deps stubbed during the exec, and sys.modules
is snapshot-restored byte-identically once this module finishes importing —
the (leaked-mock) topology other test files depend on sees zero drift.
"""

import importlib.util
import os
import sys
from unittest.mock import MagicMock

import pytest

# ── zero-drift session hygiene ───────────────────────────────
# Sibling test files leak sys.modules mocks at import time (restored only at
# teardown), and later-collected real-module tests depend on the exact leaked
# topology. Snapshot sys.modules now and restore it byte-identically once this
# module's imports are done: our own references stay bound, the session sees
# no delta.
_SYS_MODULES_BEFORE = dict(sys.modules)

# The retry-predicate tests need the REAL google.api_core exception classes.
# A leaked mock `google` parent blocks importing them — evict it for the
# duration of this module's import (the snapshot-restore below puts it back).
if isinstance(sys.modules.get("google"), MagicMock):
    del sys.modules["google"]

from google.api_core import exceptions as gapi_exceptions  # noqa: E402

_SENTINEL = object()
# knowledge_graph deps that cannot (or must not) really-import mid-session:
# the mocked google.auth.transport/google.cloud entries leaked by siblings
# break the real firestore_v1 chain, and a real google.generativeai import
# risks namespace-cache corruption once siblings pop it again. Nothing in
# this file touches FieldFilter/Vector/genai/get_db paths un-stubbed.
_KG_STUB_DEPS = (
    "google.generativeai",
    "google.cloud",
    "google.cloud.firestore_v1",
    "google.cloud.firestore_v1.base_query",
    "google.cloud.firestore_v1.base_vector_query",
    "google.cloud.firestore_v1.vector",
    "conversation_store",
)


def _load_real_knowledge_graph():
    """Exec a fresh REAL knowledge_graph.py under a private name with heavy
    deps stubbed, restoring sys.modules exactly afterwards (no session drift)."""
    saved = {name: sys.modules.get(name, _SENTINEL) for name in _KG_STUB_DEPS}
    for name in _KG_STUB_DEPS:
        sys.modules[name] = MagicMock()
    try:
        spec = importlib.util.spec_from_file_location(
            "_kg_scan_bounded_real",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "knowledge_graph.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for name, original in saved.items():
            if original is _SENTINEL:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


kg = _load_real_knowledge_graph()

# Restore the session exactly as found (see snapshot above). Skip this module
# itself — deleting a module from sys.modules mid-import breaks the import
# machinery. Everything this module imported stays alive via bound references.
for _name in list(sys.modules):
    if _name == __name__:
        continue
    if _name not in _SYS_MODULES_BEFORE:
        del sys.modules[_name]
    elif sys.modules[_name] is not _SYS_MODULES_BEFORE[_name]:
        sys.modules[_name] = _SYS_MODULES_BEFORE[_name]


# ── Firestore fakes (get_db seam) ────────────────────────────


class _FakeDoc:
    def __init__(self, doc_id, data):
        self.id = doc_id
        self._data = data

    def to_dict(self):
        return dict(self._data)


class _FakeCollection:
    """Collection that records .select() args and .stream() kwargs."""

    def __init__(self, docs):
        self._docs = docs
        self.select_fields = None
        self.stream_kwargs = None
        self.bare_stream_calls = 0

    def select(self, field_paths):
        self.select_fields = list(field_paths)
        return self

    def stream(self, **kwargs):
        if self.select_fields is None:
            self.bare_stream_calls += 1  # unprojected scan — the regression
        self.stream_kwargs = kwargs
        return iter(self._docs)


class _FakeDB:
    def __init__(self, collection):
        self._collection = collection
        self.requested = None

    def collection(self, name):
        self.requested = name
        return self._collection


@pytest.fixture(autouse=True)
def _clear_kg_cache():
    with kg._kg_cache_lock:
        kg._kg_cache.clear()
    yield
    with kg._kg_cache_lock:
        kg._kg_cache.clear()


def _wire(monkeypatch, docs):
    coll = _FakeCollection(docs)
    db = _FakeDB(coll)
    monkeypatch.setattr(kg, "get_db", lambda: db)
    return coll, db


# ── (a) projection + explicit retry on the scan ──────────────


def test_load_all_entries_projects_fields_without_embedding(monkeypatch):
    coll, db = _wire(monkeypatch, [_FakeDoc("d1", {"name": "x", "source_date": "2026-08-01"})])

    entries = kg._load_all_entries()

    assert db.requested == kg.config.FIRESTORE_KNOWLEDGE_GRAPH_COLLECTION
    assert coll.bare_stream_calls == 0, "scan must be projected, never bare"
    assert coll.select_fields is not None
    assert coll.select_fields == kg._ENTRY_FIELDS
    assert "embedding" not in coll.select_fields, "heavy vector field must be excluded"
    # _doc_to_dict semantics preserved
    assert entries[0]["id"] == "d1"
    assert entries[0]["name"] == "x"


def test_entry_fields_match_write_schema_minus_embedding():
    """_ENTRY_FIELDS is exactly the _store_entries write schema minus the
    embedding vector (embedding_model stays — it's a tiny string)."""
    expected = {
        "entity_type", "name", "content", "status", "owner",
        "related_people", "related_projects", "tags",
        "_search_people", "_search_projects",
        "source_type", "source_id", "source_title", "source_date",
        "extracted_at", "embedding_model",
    }
    assert set(kg._ENTRY_FIELDS) == expected
    assert len(kg._ENTRY_FIELDS) == len(expected), "no duplicate field paths"


def test_load_all_entries_streams_with_explicit_retry_and_timeout(monkeypatch):
    coll, _ = _wire(monkeypatch, [])

    kg._load_all_entries()

    assert coll.stream_kwargs is not None
    assert coll.stream_kwargs["retry"] is kg._KG_STREAM_RETRY
    timeout = coll.stream_kwargs["timeout"]
    assert timeout is not None and timeout > 0, "stream must be time-bounded"


def test_load_all_entries_result_is_cached(monkeypatch):
    """Cache semantics unchanged: second call served from the 300s TTL cache."""
    coll, _ = _wire(monkeypatch, [_FakeDoc("d1", {"name": "x"})])

    first = kg._load_all_entries()
    coll.select_fields = None  # would count a bare stream if re-queried
    second = kg._load_all_entries()

    assert second is first
    assert coll.stream_kwargs is not None and coll.bare_stream_calls == 0


# ── (b) query_all_entries ordering + limit ───────────────────


def test_query_all_entries_sorts_desc_and_applies_limit(monkeypatch):
    _wire(monkeypatch, [
        _FakeDoc("mid", {"name": "b", "source_date": "2026-06-15"}),
        _FakeDoc("newest", {"name": "c", "source_date": "2026-08-01"}),
        _FakeDoc("oldest", {"name": "a", "source_date": "2025-01-01"}),
        _FakeDoc("undated", {"name": "d"}),  # sorts last (datetime.min)
    ])

    limited = kg.query_all_entries(limit=2)
    assert [e["id"] for e in limited] == ["newest", "mid"]

    everything = kg.query_all_entries()
    assert [e["id"] for e in everything] == ["newest", "mid", "oldest", "undated"]


# ── (c) retry predicate — transient yes, terminal no ─────────


@pytest.mark.parametrize("exc", [
    gapi_exceptions.ServiceUnavailable("503 Query timed out"),
    gapi_exceptions.DeadlineExceeded("deadline"),
    gapi_exceptions.InternalServerError("500"),
])
def test_stream_retry_predicate_matches_transient_errors(exc):
    assert kg._KG_STREAM_RETRY._predicate(exc) is True


@pytest.mark.parametrize("exc", [
    gapi_exceptions.PermissionDenied("403"),
    gapi_exceptions.NotFound("404"),
    ValueError("not an api error"),
])
def test_stream_retry_predicate_rejects_terminal_errors(exc):
    assert kg._KG_STREAM_RETRY._predicate(exc) is False
