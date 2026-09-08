"""Phase 0.5 acceptance gate — end-to-end replay regression for the dual-read
stable nudge identity (KG v2 Phase 0.5).

Where the unit-level coverage in ``test_nudge_dual_read.py`` proves a single
read/write of the dual-read keys, this module proves the *behaviour over time*:
two full nudge runs against an in-memory sent-store, including a simulated
entity-identity (Firestore doc-id) change between runs.

The gate is ``test_no_dup_after_reid``: run 1 records the commitment nudge under
the new stable key (source_id + normalized name); run 2 then changes ONLY the
Firestore ``id`` (as happens on an entity merge/re-id) and must produce ZERO
commitment nudges. Against the pre-dual-read code — which keyed dedup solely on
the Firestore id — run 2's legacy key would differ and a duplicate nudge would
fire. That is the regression this file locks down.

Everything that would touch Firestore/Gmail/Gemini/Chat is monkeypatched, so
nothing here hits a live backend. The approach mirrors
``test_nudge_dual_read.py``: stub ``query_open_by_age`` /
``_check_commitment_evidence`` / ``has_nudge_been_sent`` / ``mark_nudge_sent``
and the non-commitment engines, plus ``config.CHAT_SPACE_ID`` empty so no
standalone Chat send is attempted.

When every scenario passes, the final test writes an evidence summary to
``qa/phase0_5_dedup.txt`` (containing the line ``run2 nudges=0``) for the
acceptance record.
"""

import os
from datetime import datetime, timedelta

import config
import proactive_intelligence as pi
from knowledge_graph import stable_key

# Two distinct Firestore doc ids for the same logical commitment. The entity is
# "re-id'd" from DOC_A to DOC_B between runs to simulate a merge; source_id and
# name stay constant so the stable key never changes.
DOC_A = "fsdoc_auto_DOC_A"
DOC_B = "fsdoc_auto_DOC_B"

# Evidence collected by the scenario tests and flushed to disk by the final
# test. Keyed so the writer can assert every scenario actually ran.
_EVIDENCE: dict = {"scenarios": {}}

_EVIDENCE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "qa", "phase0_5_dedup.txt")


def _commitment_fixture(doc_id: str = DOC_A) -> dict:
    """One overdue commitment as returned by query_open_by_age.

    ``id`` is the (mutable) Firestore doc id; ``source_id`` + ``name`` feed the
    stable key and are held constant across runs.
    """
    return {
        "id": doc_id,
        "source_id": "cal-evt-123",
        "name": "Send proposal to ACME",
        "owner": "Karim",
        "source_date": (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
        "content": "send the ACME proposal over",
        "source_title": "ACME sync",
        "entity_type": "commitment",
    }


class _FakeSentStore:
    """In-memory stand-in for the ``proactive_nudges_sent`` Firestore collection.

    ``sent`` is the persistent cooldown store (survives across runs within a
    test). ``marked`` captures every mark_nudge_sent call for the current run so
    a scenario can count exactly how many nudges the write path recorded.
    """

    def __init__(self) -> None:
        self.sent: dict[str, bool] = {}
        self.marked: list[dict] = []

    def has(self, key: str) -> bool:
        return key in self.sent

    def mark(self, key: str, ntype: str, title: str) -> None:
        self.sent[key] = True
        self.marked.append({"key": key, "type": ntype, "title": title})

    def reset_run(self) -> None:
        """Clear per-run capture but keep the persistent cooldown store."""
        self.marked = []

    def commitments_this_run(self) -> list[dict]:
        return [m for m in self.marked if m["type"] == "commitment"]


def _wire_commitment_only(monkeypatch, store: _FakeSentStore, entry_holder: dict,
                          evidence=lambda commitment: None) -> None:
    """Drive generate_daily_nudges() with ONLY the commitment engine live.

    ``entry_holder`` is a 1-key dict {"entry": <commitment>} so a scenario can
    swap the live entity (e.g. re-id it) between runs and have the stubbed
    query_open_by_age return the current one.
    """
    monkeypatch.setattr(pi, "query_open_by_age", lambda **kwargs: [entry_holder["entry"]])
    monkeypatch.setattr(pi, "_check_commitment_evidence", evidence)
    monkeypatch.setattr(pi, "has_nudge_been_sent", store.has)
    monkeypatch.setattr(pi, "mark_nudge_sent", store.mark)
    # Isolate the commitment engine inside the coordinator.
    monkeypatch.setattr(pi, "_run_pattern_engine", lambda: [])
    monkeypatch.setattr(pi, "_run_drift_engine", lambda: [])
    monkeypatch.setattr(config, "PROACTIVE_INTELLIGENCE_ENABLED", True)
    monkeypatch.setattr(config, "KNOWLEDGE_GRAPH_ENABLED", True)
    monkeypatch.setattr(config, "CHAT_SPACE_ID", "")  # skip standalone Chat send


def test_no_dup_same_id(monkeypatch):
    """Happy path: two identical runs of the same commitment produce exactly one
    nudge total — run 1 fires and records the key, run 2 is suppressed."""
    entry_holder = {"entry": _commitment_fixture(DOC_A)}
    store = _FakeSentStore()
    _wire_commitment_only(monkeypatch, store, entry_holder)

    stable = pi._nudge_key("commitment", stable_key(entry_holder["entry"]))
    legacy = pi._nudge_key("commitment", DOC_A)

    # Run 1 — fresh store, nudge must fire and be recorded.
    store.reset_run()
    pi.generate_daily_nudges()
    run1 = store.commitments_this_run()
    assert len(run1) == 1, f"run1 must emit exactly 1 commitment nudge, got {len(run1)}"
    assert run1[0]["key"] == stable, "run1 must record the NEW stable key"
    assert legacy not in store.sent, "legacy id-based key must never be written"

    # Run 2 — identical entity, cooldown store now holds the stable key.
    store.reset_run()
    pi.generate_daily_nudges()
    run2 = store.commitments_this_run()
    assert len(run2) == 0, f"run2 must emit 0 commitment nudges, got {len(run2)}"

    _EVIDENCE["run1_nudges"] = len(run1)
    _EVIDENCE["run2_nudges"] = len(run2)
    _EVIDENCE["legacy_key"] = legacy
    _EVIDENCE["stable_key"] = stable
    _EVIDENCE["scenarios"]["test_no_dup_same_id"] = (
        f"run1=1 nudge (key={stable}), run2=0 — identical id suppressed"
    )


def test_no_dup_after_reid(monkeypatch):
    """THE gate: re-id the entity (DOC_A -> DOC_B) between runs, keeping
    source_id + name constant. The stable key is unchanged, so run 2 must emit
    ZERO commitment nudges. This FAILS against the pre-dual-read code, whose
    legacy id key changes with the doc id."""
    entry_holder = {"entry": _commitment_fixture(DOC_A)}
    store = _FakeSentStore()
    _wire_commitment_only(monkeypatch, store, entry_holder)

    stable = pi._nudge_key("commitment", stable_key(entry_holder["entry"]))
    legacy_a = pi._nudge_key("commitment", DOC_A)
    legacy_b = pi._nudge_key("commitment", DOC_B)
    assert legacy_a != legacy_b, "fixture invalid: re-id must change the legacy key"

    # Run 1 under DOC_A — nudge fires, stable key recorded.
    store.reset_run()
    pi.generate_daily_nudges()
    run1 = store.commitments_this_run()
    assert len(run1) == 1, f"run1 must emit exactly 1 commitment nudge, got {len(run1)}"
    assert run1[0]["key"] == stable, "run1 must record the stable key"

    # Simulate a merge/re-id: ONLY the Firestore id changes.
    entry_holder["entry"] = _commitment_fixture(DOC_B)
    assert stable_key(entry_holder["entry"]) == stable_key(_commitment_fixture(DOC_A)), (
        "stable key must survive the re-id (source_id + name unchanged)"
    )

    # Run 2 under DOC_B — dual-read on the stable key must suppress.
    store.reset_run()
    pi.generate_daily_nudges()
    run2 = store.commitments_this_run()
    assert len(run2) == 0, (
        f"run2 must emit 0 commitment nudges after re-id, got {len(run2)} "
        "(regression: dedup leaked because the legacy id key changed)"
    )
    assert legacy_b not in store.sent, "re-id legacy key must never be written"

    _EVIDENCE["reid_legacy_key_a"] = legacy_a
    _EVIDENCE["reid_legacy_key_b"] = legacy_b
    _EVIDENCE["reid_stable_key"] = stable
    _EVIDENCE["scenarios"]["test_no_dup_after_reid"] = (
        f"run1=1 (id={DOC_A}), re-id to {DOC_B}, run2=0 — stable key "
        f"{stable} suppressed despite legacy key {legacy_a} -> {legacy_b}"
    )


def test_drift_unaffected(monkeypatch):
    """Adjacent-surface guard: with the drift engine live (query functions
    stubbed to fixed fixtures), drift keys must still be built from the entry id
    (drift_commitment) and the project name (drift_project) — NOT the stable
    key. The Phase 0.5 migration must not bleed into drift."""
    stale_days = config.DRIFT_THRESHOLD_DAYS + 100
    old_date = (datetime.now() - timedelta(days=stale_days)).strftime("%Y-%m-%d")

    drift_entry = {
        "id": "fsdoc_drift_xyz789",
        "source_id": "granola-mtg-99",
        "name": "Migrate billing service",
        "source_date": old_date,
        "source_title": "Eng planning",
        "entity_type": "commitment",
        "related_projects": ["Apollo"],
    }
    project_activity = {
        "id": "fsdoc_act_001",
        "name": "Apollo kickoff",
        "source_type": "meeting",
        "source_date": old_date,
        "related_projects": ["Apollo"],
    }

    monkeypatch.setattr(pi, "query_open_by_age", lambda **kwargs: [drift_entry])
    monkeypatch.setattr(pi, "query_recent", lambda **kwargs: [])  # no recent activity
    monkeypatch.setattr(pi, "query_all_entries", lambda **kwargs: [project_activity])
    monkeypatch.setattr(pi, "has_nudge_been_sent", lambda key: False)

    nudges = pi._run_drift_engine()

    by_title = {n["title"]: n for n in nudges}
    assert "Migrate billing service" in by_title, "stale commitment should drift"
    assert "Apollo — gone quiet" in by_title, "stale project should drift"

    expected_commitment_key = pi._nudge_key("drift_commitment", drift_entry["id"])
    expected_project_key = pi._nudge_key("drift_project", "Apollo")

    assert by_title["Migrate billing service"]["_nudge_key"] == expected_commitment_key, (
        "drift_commitment key must stay built from the entry id"
    )
    assert by_title["Apollo — gone quiet"]["_nudge_key"] == expected_project_key, (
        "drift_project key must stay built from the project name"
    )
    # Guard against an accidental copy-paste of the stable-key migration.
    assert by_title["Migrate billing service"]["_nudge_key"] != pi._nudge_key(
        "drift_commitment", stable_key(drift_entry)
    ), "drift_commitment must NOT have migrated to stable_key"

    _EVIDENCE["scenarios"]["test_drift_unaffected"] = (
        f"drift_commitment={expected_commitment_key} (entry id), "
        f"drift_project={expected_project_key} (project name) — unchanged by migration"
    )


def test_status_write_targets_current_doc(monkeypatch):
    """When evidence resolves a commitment, update_entity_status must target the
    CURRENT Firestore doc id (DOC_B after a re-id) — never the stale id (DOC_A).
    Uses a fresh store so the dedup gate doesn't short-circuit before the
    evidence/resolve path."""
    # The entity now lives under DOC_B (re-id'd from DOC_A historically).
    entry_holder = {"entry": _commitment_fixture(DOC_B)}
    store = _FakeSentStore()

    status_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(pi, "query_open_by_age", lambda **kwargs: [entry_holder["entry"]])
    monkeypatch.setattr(
        pi, "_check_commitment_evidence",
        lambda commitment: "Found matching email: ACME proposal sent",
    )
    monkeypatch.setattr(pi, "has_nudge_been_sent", store.has)
    monkeypatch.setattr(pi, "mark_nudge_sent", store.mark)
    monkeypatch.setattr(
        pi, "update_entity_status",
        lambda doc_id, status: status_calls.append((doc_id, status)),
    )

    nudges = pi._run_commitment_engine()

    assert nudges == [], "evidence found — the commitment must resolve, not nudge"
    assert status_calls == [(DOC_B, "resolved")], (
        f"status write must target the current doc {DOC_B}, got {status_calls}"
    )
    captured_ids = [doc_id for doc_id, _ in status_calls]
    assert DOC_A not in captured_ids, "status write must never target the stale id"
    assert not store.marked, "resolved commitment must not record a nudge"

    _EVIDENCE["scenarios"]["test_status_write_targets_current_doc"] = (
        f"update_entity_status({DOC_B}, 'resolved') — targets current doc, "
        f"never stale {DOC_A}"
    )


def test_zzz_write_evidence_file():
    """Final step: flush the acceptance evidence to qa/phase0_5_dedup.txt.

    Asserts every scenario recorded a result first, so the file is only written
    once the full replay has passed. Named with a zzz prefix to run last within
    the module (pytest preserves definition order)."""
    required = {
        "test_no_dup_same_id",
        "test_no_dup_after_reid",
        "test_drift_unaffected",
        "test_status_write_targets_current_doc",
    }
    missing = required - set(_EVIDENCE["scenarios"])
    assert not missing, f"scenarios did not all run/pass: missing {missing}"

    lines = [
        "Phase 0.5 acceptance gate — dual-read stable nudge identity replay",
        f"generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "REPLAY (test_no_dup_same_id / test_no_dup_after_reid):",
        f"run1 nudges={_EVIDENCE['run1_nudges']}",
        f"run2 nudges={_EVIDENCE['run2_nudges']}",
        "",
        "KEYS:",
        f"legacy key (DOC_A)      = {_EVIDENCE['legacy_key']}",
        f"stable key              = {_EVIDENCE['stable_key']}",
        f"re-id legacy key (DOC_A)= {_EVIDENCE['reid_legacy_key_a']}",
        f"re-id legacy key (DOC_B)= {_EVIDENCE['reid_legacy_key_b']}",
        f"re-id stable key        = {_EVIDENCE['reid_stable_key']}",
        "",
        "SCENARIOS:",
    ]
    for name in (
        "test_no_dup_same_id",
        "test_no_dup_after_reid",
        "test_drift_unaffected",
        "test_status_write_targets_current_doc",
    ):
        lines.append(f"PASS {name}: {_EVIDENCE['scenarios'][name]}")
    lines.append("")
    content = "\n".join(lines) + "\n"

    os.makedirs(os.path.dirname(_EVIDENCE_PATH), exist_ok=True)
    with open(_EVIDENCE_PATH, "w", encoding="utf-8") as fh:
        fh.write(content)

    # Verify the file landed with the required acceptance line.
    with open(_EVIDENCE_PATH, encoding="utf-8") as fh:
        written = fh.read()
    assert "run2 nudges=0" in written, "evidence file must record run2 nudges=0"
    assert written.count("PASS ") == 4, "evidence file must record 4 PASS lines"
