"""Tests for the commitment-nudge dual-read dedup migration (KG v2 Phase 0.5).

The commitment follow-up engine is migrating its dedup key from the Firestore
entity ``id`` (auto-generated, changes when entities merge) to ``stable_key``
(source_id + normalized name, survives merges). During the 7-day cooldown
transition window the engine must *read* both keys (so existing
``proactive_nudges_sent`` docs keep suppressing) but only *write* the new key.

These tests monkeypatch every Firestore/Gmail/Gemini boundary, so nothing here
touches a live backend.
"""

import config
import proactive_intelligence as pi
from knowledge_graph import stable_key


def _fixture_commitment() -> dict:
    """A single overdue commitment as returned by query_open_by_age.

    ``id`` is the legacy Firestore doc id; ``source_id`` + ``name`` feed the new
    stable key. The two identifiers are intentionally different so a key built
    from one can never accidentally equal a key built from the other.
    """
    return {
        "id": "fsdoc_auto_abc123",
        "source_id": "granola-mtg-42",
        "name": "Send Q3 deck to Sarah",
        "owner": "karim",
        "source_date": "2026-06-01",
        "content": "send the Q3 deck over",
        "source_title": "Product sync",
        "entity_type": "commitment",
    }


def _stub_commitment_engine(monkeypatch, entry, has_sent):
    """Wire up the shared monkeypatches for the commitment engine.

    ``has_sent`` is a predicate (key) -> bool standing in for has_nudge_been_sent.
    Returns the captured-keys list passed to mark_nudge_sent.
    """
    sent_keys: list[str] = []
    monkeypatch.setattr(pi, "query_open_by_age", lambda **kwargs: [entry])
    monkeypatch.setattr(pi, "_check_commitment_evidence", lambda commitment: None)
    monkeypatch.setattr(pi, "has_nudge_been_sent", has_sent)
    monkeypatch.setattr(
        pi, "mark_nudge_sent", lambda key, ntype, title: sent_keys.append(key)
    )
    return sent_keys


def test_suppressed_by_legacy_key(monkeypatch):
    """A doc written under the OLD id-based key must still suppress the nudge
    (legacy read must survive the migration)."""
    entry = _fixture_commitment()
    old_key = pi._nudge_key("commitment", entry["id"])

    _stub_commitment_engine(monkeypatch, entry, lambda key: key == old_key)

    nudges = pi._run_commitment_engine()
    assert nudges == [], "legacy-key dedup doc must suppress the commitment nudge"


def test_suppressed_by_new_key(monkeypatch):
    """A doc written under the NEW stable key must suppress the nudge.

    This fails before the dual-read change, because the current code only checks
    the legacy id-based key."""
    entry = _fixture_commitment()
    new_key = pi._nudge_key("commitment", stable_key(entry))

    _stub_commitment_engine(monkeypatch, entry, lambda key: key == new_key)

    nudges = pi._run_commitment_engine()
    assert nudges == [], "new stable-key dedup doc must suppress the commitment nudge"


def test_writes_only_new_key(monkeypatch):
    """When the nudge actually fires, the dedup write must use ONLY the new
    stable key — never the legacy id-based key.

    Drives through generate_daily_nudges() (the coordinator that owns the
    mark_nudge_sent write), with the pattern and drift engines stubbed out so we
    isolate the commitment write path."""
    entry = _fixture_commitment()
    new_key = pi._nudge_key("commitment", stable_key(entry))
    old_key = pi._nudge_key("commitment", entry["id"])

    sent_keys = _stub_commitment_engine(monkeypatch, entry, lambda key: False)

    # Isolate the commitment engine inside the coordinator.
    monkeypatch.setattr(pi, "_run_pattern_engine", lambda: [])
    monkeypatch.setattr(pi, "_run_drift_engine", lambda: [])
    monkeypatch.setattr(config, "PROACTIVE_INTELLIGENCE_ENABLED", True)
    monkeypatch.setattr(config, "KNOWLEDGE_GRAPH_ENABLED", True)
    monkeypatch.setattr(config, "CHAT_SPACE_ID", "")  # skip standalone Chat send

    pi.generate_daily_nudges()

    assert sent_keys == [new_key], (
        f"send path must write exactly the new stable key; wrote {sent_keys}"
    )
    assert old_key not in sent_keys, "legacy id-based key must never be written on send"


def test_drift_key_unchanged(monkeypatch):
    """Adjacent-surface regression guard: the drift engine must keep using its
    CURRENT identifiers — entry id for drift_commitment (~:492) and the project
    name for drift_project (~:517) — not the new stable key."""
    drift_entry = {
        "id": "fsdoc_drift_xyz789",
        "source_id": "granola-mtg-99",
        "name": "Migrate billing service",
        "source_date": "2026-01-01",
        "source_title": "Eng planning",
        "entity_type": "commitment",
        "related_projects": ["Apollo"],
    }
    project_activity = {
        "id": "fsdoc_act_001",
        "name": "Apollo kickoff",
        "source_type": "meeting",
        "source_date": "2026-01-01",
        "related_projects": ["Apollo"],
    }

    monkeypatch.setattr(pi, "query_open_by_age", lambda **kwargs: [drift_entry])
    monkeypatch.setattr(pi, "query_recent", lambda **kwargs: [])  # no recent activity
    monkeypatch.setattr(pi, "query_all_entries", lambda **kwargs: [project_activity])
    monkeypatch.setattr(pi, "has_nudge_been_sent", lambda key: False)

    nudges = pi._run_drift_engine()

    by_title = {n["title"]: n for n in nudges}
    assert "Migrate billing service" in by_title, "stale commitment should produce a drift nudge"
    assert "Apollo — gone quiet" in by_title, "stale project should produce a drift nudge"

    expected_commitment_key = pi._nudge_key("drift_commitment", drift_entry["id"])
    expected_project_key = pi._nudge_key("drift_project", "Apollo")

    assert by_title["Migrate billing service"]["_nudge_key"] == expected_commitment_key, (
        "drift_commitment key must stay built from the entry id"
    )
    assert by_title["Apollo — gone quiet"]["_nudge_key"] == expected_project_key, (
        "drift_project key must stay built from the project name"
    )

    # Guard against an accidental copy-paste of the stable-key migration into drift.
    assert by_title["Migrate billing service"]["_nudge_key"] != pi._nudge_key(
        "drift_commitment", stable_key(drift_entry)
    ), "drift_commitment must NOT have migrated to stable_key"
