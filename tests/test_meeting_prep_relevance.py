"""Meeting-prep context relevance — owner exclusion, recency window, reranking.

Regression tests for the bug where meeting prep attached stale (~17-month-old)
and unrelated KG context to meetings (e.g. an intro meeting picked up the
owner's old project history). Covers:

  1. Owner attendee/alias exclusion from query_by_person retrieval
  2. since=cutoff propagation to person + project KG queries
  3. Date post-filtering of semantic results (undated entries kept)
  4. Single rerank pass over the merged set (reranker-dropped entries excluded)
  5. Per-attendee "no prior context" notes
  6. Graceful fallback to the unreranked list when reranking fails

Every test is hermetic: Firestore/KG/Claude are monkeypatched, no network,
no LLM calls (conventions follow tests/test_p1_consumers.py).
"""

from datetime import datetime, timedelta

import pytest

import claude_client
import config
import knowledge_graph as kg
import proactive_intelligence as pi


def _meeting(attendees=("Cameron",), title="Karim / Cameron intro"):
    return {"title": title, "start_time": "2pm",
            "attendees": [{"name": n} for n in attendees]}


def _entry(eid, date: str | None = "RECENT", people=(), projects=(),
           name="fact", content="details"):
    if date == "RECENT":
        date = datetime.now().strftime("%Y-%m-%d")
    return {"id": eid, "source_date": date, "name": name, "content": content,
            "related_people": list(people), "related_projects": list(projects)}


def _expected_cutoff(days=90):
    return (
        datetime.now() - timedelta(days=days)
    ).replace(hour=0, minute=0, second=0, microsecond=0).strftime("%Y-%m-%d")


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch):
    """No Firestore, no LLM, deterministic config."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", False)
    monkeypatch.setattr(config, "OWNER_NAME", "Karim Tanber")
    monkeypatch.setattr(config, "MEETING_PREP_CONTEXT_MAX_AGE_DAYS", 90)
    monkeypatch.setattr(config, "RERANK_ENABLED", True)
    monkeypatch.setattr(pi, "generate", lambda **k: "msg")
    monkeypatch.setattr(pi, "extract_text", lambda msg: "- brief")
    # Identity reranker by default; individual tests override it.
    monkeypatch.setattr(claude_client, "rerank",
                        lambda query, texts, top_k=None: list(range(len(texts))))
    # KG boundaries default to empty; individual tests override as needed.
    monkeypatch.setattr(pi, "query_by_person", lambda *a, **k: [])
    monkeypatch.setattr(pi, "query_by_project", lambda *a, **k: [])
    monkeypatch.setattr(kg, "semantic_search", lambda *a, **k: [])


def _capture_formatted_ids(monkeypatch):
    captured = {}

    def _fake_format(entries):
        captured["ids"] = [e["id"] for e in entries]
        return "ctx"

    monkeypatch.setattr(pi, "format_knowledge_for_context", _fake_format)
    return captured


# ── (a) owner attendee/alias exclusion ───────────────────────


def test_owner_attendee_excluded_from_person_queries(monkeypatch):
    queried = []
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, since=None, limit=50: queried.append(name) or [])

    pi._build_meeting_prep(_meeting(attendees=("Karim", "Cameron")))

    assert queried == ["Cameron"], "owner must never be queried by person"


def test_owner_alias_excluded_via_identity_resolution(monkeypatch):
    """Owner aliases from canonical resolution are dropped too."""
    monkeypatch.setattr(config, "KG_RESOLUTION_ENABLED", True)

    def _aliases(name):
        if "karim" in name.lower():
            return ["KT", "karim.t@example.com"]
        return []

    monkeypatch.setattr(pi, "get_canonical_aliases", _aliases)
    queried = []
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, since=None, limit=50: queried.append(name) or [])

    # "KT" is an owner alias — must be dropped even though it doesn't
    # literally contain "Karim".
    pi._build_meeting_prep(_meeting(attendees=("KT", "Cameron")))

    assert queried == ["Cameron"]


# ── (b) since=cutoff on person + project queries ─────────────


def test_since_cutoff_passed_to_person_and_project_queries(monkeypatch):
    calls = {"person": [], "project": []}

    def _fake_person(name, since=None, limit=50):
        calls["person"].append(since)
        return [_entry("p1", people=["Cameron"], projects=["Atlas"])]

    def _fake_project(name, since=None, limit=50):
        calls["project"].append((name, since))
        return []

    monkeypatch.setattr(pi, "query_by_person", _fake_person)
    monkeypatch.setattr(pi, "query_by_project", _fake_project)

    pi._build_meeting_prep(_meeting())

    expected = _expected_cutoff(90)
    assert calls["person"] == [expected]
    assert calls["project"] == [("Atlas", expected)]


# ── (c) semantic results date-filtered, undated kept ─────────


def test_stale_semantic_results_filtered_undated_kept(monkeypatch):
    stale = _entry("stale", date="2025-01-15")   # ~17 months old
    fresh = _entry("fresh")                       # today
    undated = _entry("undated", date=None)        # unparseable → KEPT
    garbled = _entry("garbled", date="not a date")

    monkeypatch.setattr(kg, "semantic_search",
                        lambda *a, **k: [stale, fresh, undated, garbled])
    captured = _capture_formatted_ids(monkeypatch)

    pi._build_meeting_prep(_meeting())

    assert "stale" not in captured["ids"]
    assert {"fresh", "undated", "garbled"} == set(captured["ids"])


# ── (d) merged set reranked once, dropped entries excluded ───


def test_merged_results_reranked_and_dropped_entries_excluded(monkeypatch):
    person_entry = _entry("p1", people=["Cameron"], content="Cameron intro notes")
    semantic_keep = _entry("s1", content="agenda for quarterly sync")
    semantic_drop = _entry("s2", name="RAF DSW", content="old owner project")

    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, since=None, limit=50: [person_entry])

    seen_semantic = {}

    def _fake_semantic(query, limit=10, rerank=True, **k):
        seen_semantic["rerank"] = rerank
        return [semantic_keep, semantic_drop]

    monkeypatch.setattr(kg, "semantic_search", _fake_semantic)

    seen_rerank = {}

    def _fake_rerank(query, texts, top_k=None):
        seen_rerank["query"] = query
        seen_rerank["n"] = len(texts)
        # Reranker drops the RAF entry as irrelevant to this meeting.
        return [i for i, t in enumerate(texts) if "RAF" not in t]

    monkeypatch.setattr(claude_client, "rerank", _fake_rerank)
    captured = _capture_formatted_ids(monkeypatch)

    pi._build_meeting_prep(_meeting(attendees=("Cameron",), title="Quarterly sync"))

    # semantic_search must not rerank on its own (single rerank pass)
    assert seen_semantic["rerank"] is False
    # the ONE rerank pass saw the full merged set, keyed on title + attendees
    assert seen_rerank["n"] == 3
    assert "Quarterly sync" in seen_rerank["query"]
    assert "Cameron" in seen_rerank["query"]
    # reranker-dropped entry is excluded from the LLM context
    assert set(captured["ids"]) == {"p1", "s1"}


# ── (e) per-attendee "no prior context" note ─────────────────


def test_attendee_with_zero_results_gets_no_context_note(monkeypatch):
    def _fake_person(name, since=None, limit=50):
        if name == "Cameron":
            return [_entry("p1", people=["Cameron"])]
        return []

    monkeypatch.setattr(pi, "query_by_person", _fake_person)

    prompts = {}

    def _fake_generate(prompt=None, **k):
        prompts["prompt"] = prompt
        return "msg"

    monkeypatch.setattr(pi, "generate", _fake_generate)

    pi._build_meeting_prep(_meeting(attendees=("Cameron", "Dana")))

    assert "(No prior context found for Dana.)" in prompts["prompt"]
    assert "(No prior context found for Cameron.)" not in prompts["prompt"]


# ── (f) rerank failure → unreranked fallback ─────────────────


def test_rerank_failure_falls_back_to_unreranked_list(monkeypatch):
    entries = [_entry("p1", people=["Cameron"]), _entry("p2", people=["Cameron"])]
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, since=None, limit=50: entries)

    def _boom(query, texts, top_k=None):
        raise RuntimeError("reranker down")

    monkeypatch.setattr(claude_client, "rerank", _boom)
    captured = _capture_formatted_ids(monkeypatch)

    result = pi._build_meeting_prep(_meeting())

    assert captured["ids"] == ["p1", "p2"], "must fall back to the unreranked list"
    assert result is not None
