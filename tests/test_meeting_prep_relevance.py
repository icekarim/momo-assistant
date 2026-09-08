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
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest
from test_reauth_tools import _load_fresh_module, isolated_module_registry

import claude_client
import config
import knowledge_graph as kg
import meeting_prep_accuracy as accuracy
import proactive_intelligence as pi
if TYPE_CHECKING:  # The new run-path regressions bind real classes at execution time.
    from connection_errors import ExternalAuthError, ExternalUnavailableError


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


def _capture_evidence_ids(monkeypatch):
    captured = {}
    select = accuracy.select_prep_evidence
    format_context = accuracy.format_prep_evidence_context

    def _capture_select(meeting, entries):
        captured["candidate_ids"] = [e["id"] for e in entries]
        return select(meeting, entries)

    def _capture_format(evidence):
        captured["ids"] = [item.entry["id"] for item in evidence]
        return format_context(evidence)

    monkeypatch.setattr(accuracy, "select_prep_evidence", _capture_select)
    monkeypatch.setattr(accuracy, "format_prep_evidence_context", _capture_format)
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
    captured = _capture_evidence_ids(monkeypatch)

    pi._build_meeting_prep(_meeting())

    assert "stale" not in captured["candidate_ids"]
    assert {"fresh", "undated", "garbled"} == set(captured["candidate_ids"])
    # Passing recency alone is not strong evidence of relevance.
    assert captured["ids"] == []


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
    captured = _capture_evidence_ids(monkeypatch)

    # Use a specific title: generic short sync titles intentionally skip this leg.
    result = pi._build_meeting_prep(_meeting(attendees=("Cameron",), title="Quarterly planning review"))

    # semantic_search must not rerank on its own (single rerank pass)
    assert seen_semantic["rerank"] is False
    # the ONE rerank pass saw the full merged set, keyed on title + attendees
    assert seen_rerank["n"] == 3
    assert "Quarterly planning review" in seen_rerank["query"]
    assert "Cameron" in seen_rerank["query"]
    # Reranker-dropped entries never reach deterministic scoring; the remaining
    # person-only (15 points) and semantic-only (0 points) hits are still weak.
    assert set(captured["candidate_ids"]) == {"p1", "s1"}
    assert captured["ids"] == []
    assert "I don't have strong prep context for this one yet." in result


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

    result = pi._build_meeting_prep(_meeting(attendees=("Cameron", "Dana")))

    assert "(No relevant prior context found for Dana.)" in result
    assert "(No relevant prior context found for Cameron.)" not in result
    # Absence notices are deterministic, not model evidence with fake IDs.
    assert "(No relevant prior context found for Dana.)" not in prompts["prompt"]
    assert "source:" not in result


# ── (f) rerank failure → unreranked fallback ─────────────────


def test_rerank_failure_falls_back_to_unreranked_list(monkeypatch):
    entries = [_entry("p1", people=["Cameron"]), _entry("p2", people=["Cameron"])]
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, since=None, limit=50: entries)

    def _boom(query, texts, top_k=None):
        raise RuntimeError("reranker down")

    monkeypatch.setattr(claude_client, "rerank", _boom)
    captured = _capture_evidence_ids(monkeypatch)

    result = pi._build_meeting_prep(_meeting())

    assert captured["ids"] == ["p1", "p2"], "must fall back to the unreranked list"
    assert result is not None


def test_planned_people_and_aliases_share_bounded_fanout_excluding_owner(monkeypatch):
    aliases = {
        "Karim Tanber": ["KT"],
        "Cameron": ["KT", "cameron", "Cam", "Cameron Smith", "Cameron S"],
        "Dana": ["Dana Smith"],
    }
    monkeypatch.setattr(pi, "get_canonical_aliases", lambda name: aliases.get(name, []))
    person = MagicMock(return_value=[_entry("p1", projects=["Z", "C", "B", "A"])])
    project = MagicMock(return_value=[])
    semantic = MagicMock(return_value=[])
    monkeypatch.setattr(pi, "query_by_person", person)
    monkeypatch.setattr(pi, "query_by_project", project)
    monkeypatch.setattr(kg, "semantic_search", semantic)
    meeting = _meeting(
        attendees=("Karim", "Cameron", "Dana", *[f"Person {i}" for i in range(7)]),
        title="weekly sync",
    )
    meeting.update(organizer="KT", description="Cameron and Dana handovers")

    result = pi._build_meeting_prep(meeting)

    assert {call.args[0] for call in person.call_args_list} == {
        "Cameron", "Dana", "Cam", "Cameron Smith",
    }
    assert person.call_count == accuracy.MEETING_PREP_MAX_PERSON_QUERIES
    assert all(call.args[1:] == (_expected_cutoff(), 8) for call in person.call_args_list)
    assert {call.args for call in project.call_args_list} == {
        (name, _expected_cutoff(), 5) for name in ("A", "B", "C")
    }
    semantic.assert_not_called()
    # Skipped people have not been searched, so cannot receive absence notices.
    for i in range(7):
        assert f"(No relevant prior context found for Person {i}.)" not in result
    assert "(No relevant prior context found for Karim.)" not in result
    for name in ("Cameron", "Dana"):
        assert f"(No relevant prior context found for {name}.)" in result
    assert sum(line.startswith("- ") for line in result.splitlines()) == 1


def test_unqueried_attendee_beyond_person_budget_gets_no_absence_notice(monkeypatch):
    names = ("Cameron Smith", "Dana", "Alex", "Morgan", "Cameron Jones")
    person = MagicMock(return_value=[])
    monkeypatch.setattr(pi, "query_by_person", person)

    result = pi._build_meeting_prep(_meeting(attendees=names, title="weekly sync"))

    assert {call.args[0] for call in person.call_args_list} == set(names[:4])
    for name in names[:4]:
        assert f"(No relevant prior context found for {name}.)" in result
    # Sharing a name token with a queried person is not an identity match.
    assert "(No relevant prior context found for Cameron Jones.)" not in result
    assert sum(line.startswith("- ") for line in result.splitlines()) == 1


def test_failed_person_query_gets_no_absence_notice(monkeypatch):
    def _person(name, *args):
        if name == "Dana":
            raise RuntimeError("KG unavailable")
        return []

    person = MagicMock(side_effect=_person)
    monkeypatch.setattr(pi, "query_by_person", person)

    result = pi._build_meeting_prep(_meeting(attendees=("Cameron", "Dana")))

    assert {call.args[0] for call in person.call_args_list} == {"Cameron", "Dana"}
    assert "(No relevant prior context found for Cameron.)" in result
    assert "(No relevant prior context found for Dana.)" not in result


def test_large_generic_meeting_only_reports_successfully_queried_organizer(monkeypatch):
    names = ("Cameron", *[f"Person {i}" for i in range(8)])
    meeting = _meeting(attendees=names, title="weekly sync")
    meeting["organizer"] = "Cameron"
    person = MagicMock(return_value=[])
    semantic = MagicMock(return_value=[])
    monkeypatch.setattr(pi, "query_by_person", person)
    monkeypatch.setattr(kg, "semantic_search", semantic)

    result = pi._build_meeting_prep(meeting)

    person.assert_called_once_with("Cameron", _expected_cutoff(), 8)
    semantic.assert_not_called()
    assert "(No relevant prior context found for Cameron.)" in result
    for name in names[1:]:
        assert f"(No relevant prior context found for {name}.)" not in result
    assert sum(line.startswith("- ") for line in result.splitlines()) == 1


def test_alias_evidence_counts_for_base_attendee_after_selection(monkeypatch):
    monkeypatch.setattr(pi, "get_canonical_aliases",
                        lambda name: ["Cam Smith"] if name == "Cameron" else [])
    entry = _entry("alias-hit", people=["Cameron"])
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, *a: [entry] if name == "Cam Smith" else [])
    captured = _capture_evidence_ids(monkeypatch)

    result = pi._build_meeting_prep(_meeting())

    assert captured["ids"] == ["alias-hit"]
    assert "(No relevant prior context found for Cameron.)" not in result


def test_only_owner_generic_meeting_has_no_retrieval(monkeypatch):
    person = MagicMock(return_value=[])
    semantic = MagicMock(return_value=[])
    monkeypatch.setattr(pi, "query_by_person", person)
    monkeypatch.setattr(kg, "semantic_search", semantic)

    result = pi._build_meeting_prep(_meeting(attendees=("Karim",), title="weekly sync"))

    person.assert_not_called()
    semantic.assert_not_called()
    assert "I don't have strong prep context for this one yet." in result
    assert "No relevant prior context found for Karim" not in result


@pytest.mark.parametrize("drop_by", ["rerank", "score"])
def test_filtered_person_hits_still_get_missing_context_notes(monkeypatch, drop_by):
    entry = _entry("weak", people=["Cameron"] if drop_by == "rerank" else [])
    monkeypatch.setattr(pi, "query_by_person", lambda *a: [entry, _entry("noise")])
    if drop_by == "rerank":
        monkeypatch.setattr(claude_client, "rerank", lambda *a: [])

    result = pi._build_meeting_prep(_meeting())

    assert "(No relevant prior context found for Cameron.)" in result
    assert "I don't have strong prep context for this one yet." in result


def test_strong_merged_evidence_uses_claude_and_finalizer_with_reserved_note(monkeypatch):
    title = "Atlas launch review"
    shared = _entry("shared", people=["Cameron"], projects=["Atlas"], name="Launch checklist")
    shared["source_title"] = title
    dropped = _entry("dropped", name="RAF DSW", projects=["Atlas"])
    # This would pass deterministic scoring if the rerank filter were bypassed.
    dropped["source_title"] = title
    project_entry = _entry("project", projects=["Atlas"], name="Atlas decision")
    project_entry["source_title"] = "Earlier launch review"
    monkeypatch.setattr(pi, "query_by_person",
                        lambda name, *a: [shared] if name == "Cameron" else [])
    monkeypatch.setattr(pi, "query_by_project", lambda *a: [shared, project_entry])
    semantic = MagicMock(return_value=[shared, dropped])
    monkeypatch.setattr(kg, "semantic_search", semantic)
    rerank = MagicMock(side_effect=lambda query, texts: [
        i for i, text in enumerate(texts) if "RAF" not in text
    ])
    monkeypatch.setattr(claude_client, "rerank", rerank)
    captured = _capture_evidence_ids(monkeypatch)
    raw = "\n".join([
        "📋 *meeting prep — garbled title*",
        *[f"- Checklist detail {i} [E1]" for i in range(1, 7)],
        "- Unsupported claim.",
    ])
    generate = MagicMock(return_value=SimpleNamespace(
        content=[SimpleNamespace(type="text", text=raw)],
    ))
    monkeypatch.setattr(pi, "generate", generate)
    monkeypatch.setattr(pi, "extract_text", claude_client.extract_text)

    result = pi._build_meeting_prep(_meeting(attendees=("Cameron", "Dana"), title=title))

    semantic.assert_called_once_with(title, limit=10, rerank=False)
    rerank.assert_called_once()
    assert set(captured["candidate_ids"]) == {"shared", "project"}
    assert captured["ids"] == ["shared", "project"]
    generate.assert_called_once()
    assert generate.call_args.kwargs["tier"] == claude_client.TaskComplexity.LIGHT
    prompt = generate.call_args.kwargs["prompt"]
    assert "[E1]" in prompt and "[E2]" in prompt
    assert "Launch checklist" in prompt and "Atlas decision" in prompt
    assert "RAF" not in prompt
    assert result.splitlines()[0] == f"📋 *meeting prep — {title}*"
    assert result.count("📋") == 1
    assert "garbled title" not in result
    assert "Unsupported claim" not in result
    assert "[E1]" not in result
    assert f"_(source: {title}, {shared['source_date']})_" in result
    assert "Checklist detail 5" in result and "Checklist detail 6" not in result
    assert result.splitlines()[-1] == "- (No relevant prior context found for Dana.)"
    assert sum(line.startswith("- ") for line in result.splitlines()) == 6


# ── (g) calendar failure must stop prep before KG or delivery ─


@pytest.fixture
def prep_error_types(monkeypatch):
    """Match run_meeting_prep's lazy import without retaining a sibling's class.

    Keep this scoped to the new run-path tests; legacy relevance tests and
    sibling module bindings are restored unchanged afterward.
    """
    with isolated_module_registry("connection_errors"):
        errors = _load_fresh_module("connection_errors")
        for name in ("ExternalAuthError", "ExternalUnavailableError"):
            monkeypatch.setitem(globals(), name, getattr(errors, name))
        yield errors


@pytest.fixture
def prep_run_env(monkeypatch, prep_error_types):
    monkeypatch.setattr(config, "PROACTIVE_INTELLIGENCE_ENABLED", True)
    monkeypatch.setattr(config, "MEETING_PREP_ENABLED", True)
    monkeypatch.setattr(config, "KNOWLEDGE_GRAPH_ENABLED", True)
    monkeypatch.setattr(config, "CHAT_SPACE_ID", "spaces/test")
    mocks = {}
    for name in (
        "has_prep_been_sent", "_run_meeting_prep_traced", "_build_meeting_prep",
        "query_by_person", "query_by_project", "generate", "send_chat_message",
        "_store_proactive_message", "mark_prep_sent",
    ):
        mocks[name] = MagicMock()
        monkeypatch.setattr(pi, name, mocks[name])
    mocks["semantic_search"] = MagicMock()
    monkeypatch.setattr(kg, "semantic_search", mocks["semantic_search"])
    return mocks


def test_run_meeting_prep_reports_calendar_auth_failure(monkeypatch, prep_run_env, capsys):
    fetch = MagicMock(side_effect=ExternalAuthError("google_workspace", "Reconnect Calendar"))
    monkeypatch.setattr(pi, "fetch_upcoming_meetings", fetch)

    result = pi.run_meeting_prep()

    assert result["status"] == "auth_failed"
    assert result["source"] == "calendar_events"
    assert result["preps_sent"] == 0
    assert "re-auth" in result["reason"]
    fetch.assert_called_once_with(hours=config.MEETING_PREP_LOOKAHEAD_HOURS)
    for mock in prep_run_env.values():
        mock.assert_not_called()
    assert "no meetings" not in capsys.readouterr().out.lower()


def test_run_meeting_prep_preserves_legitimately_empty_calendar(monkeypatch, prep_run_env):
    monkeypatch.setattr(pi, "fetch_upcoming_meetings", MagicMock(return_value=[]))

    assert pi.run_meeting_prep() == {"status": "no_meetings", "preps_sent": 0}

    for mock in prep_run_env.values():
        mock.assert_not_called()


@pytest.mark.parametrize("failure", [
    lambda: ConnectionError("Calendar unreachable"),
    lambda: ExternalUnavailableError("google_workspace", "Calendar unavailable"),
])
def test_run_meeting_prep_does_not_misclassify_outage(monkeypatch, prep_run_env, failure):
    failure = failure()
    monkeypatch.setattr(pi, "fetch_upcoming_meetings", MagicMock(side_effect=failure))

    with pytest.raises(type(failure)) as caught:
        pi.run_meeting_prep()

    assert caught.value is failure
    for mock in prep_run_env.values():
        mock.assert_not_called()
